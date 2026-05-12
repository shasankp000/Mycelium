"""
expert_post_check.p6_adapter
=============================
6-Phase Pipeline adapter.

Wraps the existing Phase3To5Pipeline as the second independent reasoner
in the post-check system.  Builds a minimal FinalDecisionResult from the
query + domain and runs the complete pipeline, then extracts the answer
text from whatever key the pipeline populated.

Returns the same ``ReasoningResult`` shape as ``TRMAdapter`` so the
fuzzy verifier can treat both sources uniformly.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict

from expert_post_check.trm_adapter import ReasoningResult  # shared dataclass

logger = logging.getLogger(__name__)

_ANSWER_KEYS = ("final_answer", "answer", "response", "output", "text")


class P6Adapter:
    """Wrap Phase3To5Pipeline as a second-opinion reasoner.

    A lightweight FinalDecisionResult is synthesised from (query, domain)
    and fed into the pipeline.  The resulting answer text is extracted
    from the pipeline's output using the same key-priority logic already
    present in run_workflow.py.
    """

    def __init__(self) -> None:
        self._pipeline = self._load_pipeline()

    def reason(self, query: str, domain: str) -> ReasoningResult:
        """Run the 6-phase pipeline on *query* and return a ReasoningResult.

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
                raw={"error": "Phase3To5Pipeline unavailable"},
            )

        try:
            fdr = self._build_fdr(query, domain)
            p6_result = self._pipeline.run_complete_pipeline(fdr)
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
        try:
            from phase3_validation.pipeline import Phase3To5Pipeline
            return Phase3To5Pipeline()
        except Exception as exc:
            logger.warning("Could not load Phase3To5Pipeline: %s", exc)
            return None

    def _build_fdr(self, query: str, domain: str):
        """Synthesise a minimal FinalDecisionResult for the pipeline."""
        from phase3_validation.utils.types import FinalDecisionResult
        return FinalDecisionResult(
            decision="USE_EXISTING_EXPERT",
            confidence=0.5,
            reasoning="post-check second-opinion call",
            action="use_existing",
            expert_name=domain,
            domain=domain,
            metadata={
                "original_query": query,
                "unified_decision_type": "USE_EXISTING_EXPERT",
                "unified_selected_experts": [domain],
            },
        )

    def _extract_answer(self, result: Any) -> str:
        d = self._to_dict(result)
        for key in _ANSWER_KEYS:
            val = d.get(key)
            if val and isinstance(val, str):
                return val
        action = d.get("action_result") or {}
        if isinstance(action, dict):
            for key in _ANSWER_KEYS:
                val = action.get(key)
                if val and isinstance(val, str):
                    return val
        return ""

    def _extract_confidence(self, result: Any) -> float:
        d = self._to_dict(result)
        dr = d.get("decision_result") or {}
        if isinstance(dr, dict):
            return float(dr.get("confidence", 0.0))
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
