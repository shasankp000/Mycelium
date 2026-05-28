"""
Phase 2 Validation Pipeline — Main Orchestrator.

Ties all six phases (2.1–2.6) together into a single callable
pipeline that accepts raw user input and produces a structured
FinalDecisionResult with confidence scores and explanations.

Example:
    >>> from mycelium.pipeline.phase2.pipeline import Phase2Pipeline
    >>> pipeline = Phase2Pipeline()
    >>> result = pipeline.run("What treatment is best for hypertension?")
    >>> print(result.final_decision)
"""

import logging
import time
from typing import Any, Dict, List, Optional

from mycelium.pipeline.phase2.config.phase2_config import Phase2Config
from mycelium.pipeline.phase2.phases.phase_2_1_input_normalization import (
    InputNormalizationPipeline,
)
from mycelium.pipeline.phase2.phases.phase_2_2_semantic_understanding import (
    SemanticUnderstandingPipeline,
)
from mycelium.pipeline.phase2.phases.phase_2_3_expert_selection import (
    ExpertSelectionPipeline,
)
from mycelium.pipeline.phase2.phases.phase_2_4_inference import (
    MultiExpertInferencePipeline,
)
from mycelium.pipeline.phase2.phases.phase_2_5_calibration import (
    CalibrationPipeline,
)
from mycelium.pipeline.phase2.phases.phase_2_6_synthesis import (
    DecisionSynthesisPipeline,
    FinalDecisionResult,
)

logger = logging.getLogger(__name__)


class Phase2PipelineError(Exception):
    """Raised when the Phase 2 pipeline encounters a critical error."""


class Phase2Pipeline:
    """Orchestrate all six Phase 2 reasoning stages.

    Runs Phases 2.1 → 2.2 → 2.3 → 2.4 → 2.5 → 2.6 in sequence
    and returns a :class:`FinalDecisionResult`.

    Attributes:
        config: Shared configuration for all sub-pipelines.
    """

    def __init__(
        self,
        config: Optional[Phase2Config] = None,
    ) -> None:
        """Initialise all sub-phase pipelines.

        Args:
            config: Optional shared configuration.  Uses defaults
                when not provided.
        """
        self.config = config or Phase2Config()
        self._norm_pipeline = InputNormalizationPipeline(self.config)
        self._semantic_pipeline = SemanticUnderstandingPipeline(
            self.config
        )
        self._selection_pipeline = ExpertSelectionPipeline(self.config)
        self._inference_pipeline = MultiExpertInferencePipeline(
            self.config
        )
        self._calibration_pipeline = CalibrationPipeline(self.config)
        self._synthesis_pipeline = DecisionSynthesisPipeline(self.config)
        logger.info("Phase2Pipeline initialised")

    # ------------------------------------------------------------------
    # Internal helper: extract evidence_hints from depth_config
    # ------------------------------------------------------------------

    @staticmethod
    def _build_evidence_hints(
        depth_config: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Extract Phase-D evidence keys from *depth_config*.

        Returns a dict suitable for passing as *evidence_hints* to
        :meth:`CalibrationPipeline.calibrate_and_quantify`, or
        ``None`` when neither key is present so that the callee can
        distinguish "no hints" from "hints present but zero-valued".

        Keys read from *depth_config* (both optional):
          * ``evidence_confidence`` — float in [0, 1] from
            EvidenceScorer.weighted_confidence
          * ``evidence_dst`` — summary dict from
            EvidenceDSTAdapter.fuse().summary()

        Args:
            depth_config: The per-sentence depth config dict built
                in run_workflow.py, or ``None`` for callers that
                don't supply one.

        Returns:
            Dict with at least one key, or ``None``.
        """
        if not depth_config:
            return None

        hints: Dict[str, Any] = {}

        ev_conf = depth_config.get("evidence_confidence")
        if ev_conf is not None:
            try:
                hints["evidence_confidence"] = float(ev_conf)
            except (TypeError, ValueError):
                pass

        ev_dst = depth_config.get("evidence_dst")
        if isinstance(ev_dst, dict) and ev_dst:
            hints["evidence_dst"] = ev_dst

        return hints if hints else None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        text: str,
        skip_cache: bool = False,
        routing_context: Optional[Dict[str, Any]] = None,
        available_experts: Optional[List[Dict[str, object]]] = None,
        filtered_experts: Optional[Dict[str, Any]] = None,
        depth_config: Optional[Dict[str, Any]] = None,
    ) -> FinalDecisionResult:
        """Execute the full Phase 2 pipeline for *text*.

        Args:
            text: Raw input text to process.
            skip_cache: When True, notes the request to skip
                caches (used after a complete failure).
            routing_context: Optional routing context (RoutingResult
                or plain dict) to influence Phase 2.3 expert selection.
            available_experts: Optional list of expert configs in the
                [{name, domain}] schema consumed by Phase 2.3.
                Takes precedence over *filtered_experts* if both are
                supplied.
            filtered_experts: Optional dict mapping domain name to
                model object, as produced by run_workflow.py.  When
                provided (and *available_experts* is not), this dict
                is converted into the [{name, domain}] schema so that
                Phase 2.3 receives the pre-resolved expert pool
                instead of re-deriving an empty candidate list from
                scratch.  This is the primary integration point
                between run_workflow and Phase 2.
            depth_config: Optional dict from get_depth_config().
                When present, ``expert_top_k`` caps the number of
                selected experts passed to Phase 2.4 inference.
                Fast mode passes 1 expert, balanced 2, deep 3.
                Phase-D keys ``evidence_confidence`` and
                ``evidence_dst`` are forwarded to Phase 2.5
                :meth:`CalibrationPipeline.calibrate_and_quantify`
                as *evidence_hints* so that the blended temperature
                and DST ceiling are applied during calibration.

        Returns:
            :class:`FinalDecisionResult` with all phase outputs.

        Raises:
            Phase2PipelineError: If a critical failure occurs.
        """
        start = time.perf_counter()
        logger.info(
            "Phase2Pipeline.run: len(text)=%d skip_cache=%s",
            len(text),
            skip_cache,
        )

        try:
            # Phase 2.1 — Input Normalisation
            norm_result = self._norm_pipeline.normalize(text)

            # Phase 2.2 — Semantic Understanding
            semantic_result = self._semantic_pipeline.understand(
                norm_result,
                list(norm_result.domain_mapping.keys()),
            )

            # Phase 2.3 — Expert Selection
            # Priority: explicit available_experts > filtered_experts
            # conversion > internal build_default_experts.
            experts = available_experts
            if experts is None and filtered_experts:
                # Convert {domain: model} dict → [{name, domain}] list
                # so Phase 2.3 ranking/filtering can operate on it.
                experts = [
                    {"name": f"expert_{domain}", "domain": domain}
                    for domain in filtered_experts.keys()
                    if domain  # skip empty keys defensively
                ]
                logger.debug(
                    "Phase2Pipeline: converted %d filtered_experts "
                    "into available_experts for Phase 2.3",
                    len(experts),
                )
            if experts is None:
                experts = self._selection_pipeline.build_default_experts(
                    semantic_result
                )

            selection_result = self._selection_pipeline.select_experts(
                semantic_result,
                experts,
                routing_context=routing_context,
            )

            # Cap selected experts to expert_top_k from depth_config
            if depth_config:
                top_k = depth_config.get("expert_top_k")
                if top_k is not None and isinstance(top_k, int) and top_k > 0:
                    selected = selection_result.selected_experts
                    if len(selected) > top_k:
                        logger.debug(
                            "Phase2Pipeline: capping selected_experts %d → %d "
                            "(top_k_from_depth_config=%d)",
                            len(selected), top_k, top_k,
                        )
                        selection_result.selected_experts = selected[:top_k]

            # Phase 2.4 — Multi-Expert Inference
            inference_result = self._inference_pipeline.infer(
                text=norm_result.cleaned_text,
                selected_experts=selection_result.selected_experts,
                expert_configs=None,
            )

            # Phase 2.5 — Calibration
            # Build evidence_hints from Phase-D keys in depth_config so
            # that CalibrationPipeline can apply:
            #   • blended temperature  (evidence_confidence path)
            #   • DST confidence ceiling (evidence_dst.net_confidence path)
            # _build_evidence_hints returns None (not {}) when neither key
            # is present, preserving the CalibrationPipeline's existing
            # "no hints" branch unchanged.
            _evidence_hints = self._build_evidence_hints(depth_config)
            logger.debug(
                "Phase2Pipeline: evidence_hints=%s",
                (
                    {k: v for k, v in _evidence_hints.items() if k != "evidence_dst"}
                    if _evidence_hints
                    else None
                ),
            )
            calibration_result = (
                self._calibration_pipeline.calibrate_and_quantify(
                    inference_result,
                    evidence_hints=_evidence_hints,
                )
            )

            # Phase 2.6 — Decision Synthesis
            final_result = self._synthesis_pipeline.synthesize(
                norm_result,
                semantic_result,
                selection_result,
                inference_result,
                calibration_result,
            )

            elapsed = (time.perf_counter() - start) * 1000.0
            logger.info(
                "Phase2Pipeline complete: decision=%s "
                "confidence=%.3f time=%.1fms",
                final_result.final_decision,
                final_result.decision_confidence,
                elapsed,
            )
            return final_result

        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000.0
            logger.exception(
                "Phase2Pipeline failed after %.1fms: %s", elapsed, exc
            )
            raise Phase2PipelineError(
                f"Phase 2 pipeline failed: {exc}"
            ) from exc
