"""
expert_post_check.runner
=========================
Top-level orchestrator for the Expert Post-Check System.

Flow
----
1. Hardware probe  →  decide PARALLEL vs SERIAL
2. Run TRM-slot (qwen3:9b stub) + 6-phase pipeline
   — parallel via ThreadPoolExecutor on capable hardware
   — serial (TRM first, then P6) on low-resource machines
3. Fuzzy verify both answers with cosine-sim
4. Apply RL weight update for the domain
5. Return PostCheckResult with verified_answer + full metadata

PostCheckResult is attached to action_result.metadata["post_check"]
by the phase_3_1_action_executor and flows through to the API trace
and the conversation layer LLM.

PostCheckResult fields
----------------------
    verified_answer     : str    — best answer to hand to conversation LLM
    verification_status : str    — "verified" | "conflict" | "trm_only" | "p6_only"
    trm_answer          : str
    p6_answer           : str
    similarity          : float
    preferred_source    : str    — "trm" | "p6" | "conflict"
    trm_confidence      : float
    p6_confidence       : float
    rl_update           : dict   — WeightUpdateResult as dict
    mode                : str    — "parallel" | "serial"
    conflict_hint       : str    — populated on mismatch for escalation
    total_latency_ms    : float
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from expert_post_check.hardware_check import probe_hardware, PostCheckMode
from expert_post_check.trm_adapter import TRMAdapter, ReasoningResult
from expert_post_check.p6_adapter import P6Adapter
from expert_post_check.fuzzy_verifier import FuzzyVerifier
from expert_post_check.rl_weight import RLWeightStore

logger = logging.getLogger(__name__)


@dataclass
class PostCheckResult:
    verified_answer: str
    verification_status: str
    trm_answer: str
    p6_answer: str
    similarity: float
    preferred_source: str
    trm_confidence: float
    p6_confidence: float
    rl_update: Dict[str, Any]
    mode: str
    conflict_hint: str
    total_latency_ms: float


class PostCheckRunner:
    """Orchestrate the expert post-check pipeline.

    Instantiating this class runs a one-time hardware probe.  The result
    is cached so subsequent calls within the same server process reuse it.
    """

    def __init__(
        self,
        fuzzy_threshold: float = 0.72,
    ) -> None:
        self._hw = probe_hardware()
        self._trm = TRMAdapter()
        self._p6 = P6Adapter()
        self._verifier = FuzzyVerifier(threshold=fuzzy_threshold)
        self._rl = RLWeightStore()
        logger.info(
            "PostCheckRunner initialised — mode=%s", self._hw.mode.value
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        domain: str,
        expert_name: str = "",
    ) -> PostCheckResult:
        """Run the full post-check pipeline for *query* in *domain*.

        Args:
            query:        Raw user query string.
            domain:       Domain selected by the unified expert decision.
            expert_name:  Expert name (used in logging; optional).

        Returns:
            A ``PostCheckResult`` with the verified answer and all metadata.
        """
        start = time.perf_counter()
        logger.info(
            "PostCheck START — query='%s' domain=%s expert=%s mode=%s",
            query[:80], domain, expert_name, self._hw.mode.value,
        )

        if self._hw.mode == PostCheckMode.PARALLEL:
            trm_res, p6_res = self._run_parallel(query, domain)
        else:
            trm_res, p6_res = self._run_serial(query, domain)

        verification = self._verifier.verify(
            trm_answer=trm_res.answer,
            p6_answer=p6_res.answer,
            trm_confidence=trm_res.confidence,
            p6_confidence=p6_res.confidence,
            domain=domain,
        )

        rl_update = self._rl.update(domain, verification.confidence_vote)

        verified_answer, status = self._select_answer(
            trm_res, p6_res, verification
        )

        total_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "PostCheck END — status=%s sim=%.3f latency=%.1f ms",
            status, verification.similarity, total_ms,
        )

        return PostCheckResult(
            verified_answer=verified_answer,
            verification_status=status,
            trm_answer=trm_res.answer,
            p6_answer=p6_res.answer,
            similarity=verification.similarity,
            preferred_source=verification.preferred,
            trm_confidence=trm_res.confidence,
            p6_confidence=p6_res.confidence,
            rl_update=asdict(rl_update),
            mode=self._hw.mode.value,
            conflict_hint=verification.conflict_hint,
            total_latency_ms=round(total_ms, 3),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_parallel(
        self, query: str, domain: str
    ):
        """Run TRM-slot and P6 concurrently."""
        trm_res: Optional[ReasoningResult] = None
        p6_res: Optional[ReasoningResult] = None

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_trm = pool.submit(self._trm.reason, query, domain)
            fut_p6 = pool.submit(self._p6.reason, query, domain)
            for fut in as_completed([fut_trm, fut_p6]):
                if fut is fut_trm:
                    try:
                        trm_res = fut.result()
                    except Exception as exc:
                        logger.warning("TRM parallel future failed: %s", exc)
                        trm_res = ReasoningResult("", 0.0, 0.0, "ollama/error", {"error": str(exc)})
                else:
                    try:
                        p6_res = fut.result()
                    except Exception as exc:
                        logger.warning("P6 parallel future failed: %s", exc)
                        p6_res = ReasoningResult("", 0.0, 0.0, "p6_pipeline/error", {"error": str(exc)})

        return trm_res, p6_res

    def _run_serial(
        self, query: str, domain: str
    ):
        """Run TRM-slot then P6 sequentially."""
        try:
            trm_res = self._trm.reason(query, domain)
        except Exception as exc:
            logger.warning("TRM serial call failed: %s", exc)
            trm_res = ReasoningResult("", 0.0, 0.0, "ollama/error", {"error": str(exc)})
        try:
            p6_res = self._p6.reason(query, domain)
        except Exception as exc:
            logger.warning("P6 serial call failed: %s", exc)
            p6_res = ReasoningResult("", 0.0, 0.0, "p6_pipeline/error", {"error": str(exc)})
        return trm_res, p6_res

    @staticmethod
    def _select_answer(
        trm_res: ReasoningResult,
        p6_res: ReasoningResult,
        verification,
    ):
        """Pick the verified answer and status string."""
        if verification.match:
            # Agreement — prefer higher-confidence source
            if verification.preferred == "trm":
                return trm_res.answer, "verified"
            return p6_res.answer, "verified"
        # Conflict — return both answers concatenated so the conversation
        # LLM can reconcile; flag status as "conflict"
        parts = []
        if trm_res.answer:
            parts.append(f"[Reasoner A] {trm_res.answer}")
        if p6_res.answer:
            parts.append(f"[Reasoner B] {p6_res.answer}")
        combined = "\n\n".join(parts) if parts else ""
        return combined, "conflict"
