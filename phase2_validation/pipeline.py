"""
Phase 2 Validation Pipeline — Main Orchestrator.

Ties all six phases (2.1–2.6) together into a single callable
pipeline that accepts raw user input and produces a structured
FinalDecisionResult with confidence scores and explanations.

Example:
    >>> from phase2_validation.pipeline import Phase2Pipeline
    >>> pipeline = Phase2Pipeline()
    >>> result = pipeline.run("What treatment is best for hypertension?")
    >>> print(result.final_decision)
"""

import logging
import time
from typing import Any, Dict, List, Optional

from phase2_validation.config.phase2_config import Phase2Config
from phase2_validation.phases.phase_2_1_input_normalization import (
    InputNormalizationPipeline,
)
from phase2_validation.phases.phase_2_2_semantic_understanding import (
    SemanticUnderstandingPipeline,
)
from phase2_validation.phases.phase_2_3_expert_selection import (
    ExpertSelectionPipeline,
)
from phase2_validation.phases.phase_2_4_inference import (
    MultiExpertInferencePipeline,
)
from phase2_validation.phases.phase_2_5_calibration import (
    CalibrationPipeline,
)
from phase2_validation.phases.phase_2_6_synthesis import (
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

    def run(
        self,
        text: str,
        skip_cache: bool = False,
        routing_context: Optional[Dict[str, Any]] = None,
        available_experts: Optional[List[Dict[str, object]]] = None,
    ) -> FinalDecisionResult:
        """Execute the full Phase 2 pipeline for *text*.

        Args:
            text: Raw input text to process.
            skip_cache: When True, notes the request to skip
                caches (used after a complete failure).
            routing_context: Optional Layer 0 routing context to
                influence expert selection.
            available_experts: Optional list of expert configs
                passed to Phase 2.3 selection.

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
            experts = available_experts
            if experts is None:
                experts = self._selection_pipeline.build_default_experts(
                    semantic_result
                )

            selection_result = self._selection_pipeline.select_experts(
                semantic_result,
                experts,
                routing_context=routing_context,
            )

            # Phase 2.4 — Multi-Expert Inference
            inference_result = self._inference_pipeline.infer(
                text=norm_result.cleaned_text,
                selected_experts=selection_result.selected_experts,
                expert_configs=None,
            )

            # Phase 2.5 — Calibration
            calibration_result = (
                self._calibration_pipeline.calibrate_and_quantify(
                    inference_result
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
