"""
expert_post_check.p6_adapter
=============================
6-Phase Pipeline adapter.

Wraps the existing **Phase 2** reasoning pipeline as the second
independent reasoner in the post-check system.  Builds a minimal
input from (query, domain), runs Phase2Pipeline.run(), then extracts
the answer text from the synthesised decision.

Using Phase2Pipeline (not Phase3To5Pipeline) here is intentional and
critical for correctness.  Phase3To5Pipeline calls
ActionExecutionPipeline.execute() → _execute_use_existing() →
PostCheckRunner.run() → P6Adapter.reason(), which would create a
fully re-entrant loop:

    Phase3 → PostCheck → P6 → Phase3 → PostCheck → P6 → …

The loop only broke when the inner P6 call hit the 150 s timeout,
producing the double "PostCheck START" entries and the
"ValidationOrchestrator started" log visible after TRM completes.
Phase 2 runs only the six reasoning phases (routing → synthesis →
decision) and never touches Phase 3 or the post-check system, making
re-entry structurally impossible.

Returns the same ``ReasoningResult`` shape as ``TRMAdapter`` so the
fuzzy verifier can treat both sources uniformly.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from mycelium.pipeline.expert_post_check.trm_adapter import ReasoningResult  # shared dataclass

logger = logging.getLogger(__name__)

_ANSWER_KEYS = ("final_answer", "answer", "response", "output", "text")


class P6Adapter:
    """Wrap Phase2Pipeline as a second-opinion reasoner.

    Phase2Pipeline runs the six core reasoning phases
    (routing → decomposition → evidence → critique → scoring →
    synthesis) and returns a FinalDecisionResult.  The answer text is
    extracted from that result using the same key-priority logic
    already present in run_workflow.py.

    Phase3To5Pipeline is deliberately NOT used here because it calls
    back into PostCheckRunner, creating a re-entrant loop.
    """

    def __init__(self) -> None:
        self._pipeline = self._load_pipeline()

    def reason(self, query: str, domain: str) -> ReasoningResult:
        """Run the Phase 2 reasoning pipeline on *query*.

        Args:
            query:  Raw user query string.
            domain: Domain name (e.g. ``"physics"``).

        Returns:
            A ``ReasoningResult`` with the pipeline's answer and metadata.
        """
        start = time.perf_counter()
        if self._pipeline is None:
            latency_ms = (time.perf_counter() - start) * 1000.0
            return ReasoningResult(
                answer="",
                confidence=0.0,
                latency_ms=latency_ms,
                source="p6_pipeline",
                raw={"error": "Phase2Pipeline unavailable"},
            )

        try:
            p6_result = self._pipeline.run(query)
            answer = self._extract_answer(p6_result)
            confidence = self._extract_confidence(p6_result)
            latency_ms = (time.perf_counter() - start) * 1000.0
            logger.info(
                "P6 adapter answered in %.1f ms (conf=%.3f)",
                latency_ms, confidence,
            )
            return ReasoningResult(
                answer=answer,
                confidence=confidence,
                latency_ms=latency_ms,
                source="p6_pipeline",
                raw=self._to_dict(p6_result),
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000.0
            logger.warning("P6 adapter failed: %s", exc)
            return ReasoningResult(
                answer="",
                confidence=0.0,
                latency_ms=latency_ms,
                source="p6_pipeline",
                raw={"error": str(exc)},
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_pipeline(self):
        """Load Phase2Pipeline.

        Phase2Pipeline is the correct backend for a second-opinion
        reasoner: it runs the six reasoning phases without invoking
        Phase 3 execution or the post-check system, so re-entry into
        PostCheckRunner.run() is structurally impossible.
        """
        try:
            from phase2_validation.pipeline import Phase2Pipeline  # noqa: PLC0415
            return Phase2Pipeline()
        except Exception as exc:
            logger.warning("Could not load Phase2Pipeline: %s", exc)
            return None

    def _extract_answer(self, result: Any) -> str:
        """Extract the answer string from a Phase2Pipeline result.

        Tries the standard answer-key priority order, then falls back
        to the ``final_decision`` field that Phase 2 populates.
        """
        d = self._to_dict(result)
        for key in _ANSWER_KEYS:
            val = d.get(key)
            if val and isinstance(val, str):
                return val
        # Phase 2 stores its synthesised answer in final_decision
        val = d.get("final_decision")
        if val and isinstance(val, str):
            return val
        # Last resort: reasoning text
        val = d.get("reasoning_text") or d.get("reasoning")
        if val and isinstance(val, str):
            return val
        return ""

    def _extract_confidence(self, result: Any) -> float:
        d = self._to_dict(result)
        for key in ("confidence", "expert_confidence", "score"):
            val = d.get(key)
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    pass
        return 0.0

    @staticmethod
    def _to_dict(obj: Any) -> Dict:
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "__dict__"):
            return vars(obj)
        try:
            from dataclasses import asdict, is_dataclass
            if is_dataclass(obj):
                return asdict(obj)
        except Exception:
            pass
        return {}
