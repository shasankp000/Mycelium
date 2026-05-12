"""
expert_post_check.runner
=========================
Top-level orchestrator for the Expert Post-Check System.

Flow
----
1. Hardware probe  →  decide PARALLEL vs SERIAL
2. Run TRM-slot (qwen3.5:9b stub) + 6-phase pipeline
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
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
    as_completed,
    wait,
    FIRST_COMPLETED,
)
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple

from expert_post_check.hardware_check import probe_hardware, PostCheckMode
from expert_post_check.trm_adapter import TRMAdapter, ReasoningResult
from expert_post_check.p6_adapter import P6Adapter
from expert_post_check.fuzzy_verifier import FuzzyVerifier
from expert_post_check.rl_weight import RLWeightStore

logger = logging.getLogger(__name__)

# Wall-clock budget for each reasoner call (seconds).
# Keeps the post-check from hanging the entire request when a model stalls.
REASONER_TIMEOUT_S = 60

_EMPTY = ReasoningResult(answer="", confidence=0.0, latency_ms=0.0,
                         source="timeout", raw={"error": "timeout"})


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
        reasoner_timeout: int = REASONER_TIMEOUT_S,
    ) -> None:
        self._hw = probe_hardware()
        self._trm = TRMAdapter()
        self._p6 = P6Adapter()
        self._verifier = FuzzyVerifier(threshold=fuzzy_threshold)
        self._rl = RLWeightStore()
        self._timeout = reasoner_timeout
        logger.info(
            "PostCheckRunner initialised — mode=%s timeout=%ds",
            self._hw.mode.value, self._timeout,
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
    ) -> Tuple[ReasoningResult, ReasoningResult]:
        """Run TRM-slot and P6 concurrently with a hard wall-clock timeout."""
        trm_res: ReasoningResult = _EMPTY
        p6_res: ReasoningResult = _EMPTY

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_trm: Future = pool.submit(self._trm.reason, query, domain)
            fut_p6: Future = pool.submit(self._p6.reason, query, domain)

            try:
                for fut in as_completed(
                    [fut_trm, fut_p6], timeout=self._timeout
                ):
                    if fut is fut_trm:
                        try:
                            trm_res = fut.result()
                        except Exception as exc:
                            logger.warning("TRM parallel future failed: %s", exc)
                            trm_res = ReasoningResult(
                                "", 0.0, 0.0, "ollama/error",
                                {"error": str(exc)},
                            )
                    else:
                        try:
                            p6_res = fut.result()
                        except Exception as exc:
                            logger.warning("P6 parallel future failed: %s", exc)
                            p6_res = ReasoningResult(
                                "", 0.0, 0.0, "p6_pipeline/error",
                                {"error": str(exc)},
                            )
            except FuturesTimeoutError:
                logger.warning(
                    "PostCheck parallel timeout after %ds — "
                    "cancelling outstanding futures",
                    self._timeout,
                )
                # Collect whatever finished before the timeout
                if fut_trm.done():
                    try:
                        trm_res = fut_trm.result()
                    except Exception:
                        pass
                if fut_p6.done():
                    try:
                        p6_res = fut_p6.result()
                    except Exception:
                        pass
                # Cancel pending futures (best-effort)
                fut_trm.cancel()
                fut_p6.cancel()

        return trm_res, p6_res

    def _run_serial(
        self, query: str, domain: str
    ) -> Tuple[ReasoningResult, ReasoningResult]:
        """Run TRM-slot then P6 sequentially, each with a timeout."""
        trm_res = self._call_with_timeout(
            self._trm.reason, query, domain, label="TRM"
        )
        p6_res = self._call_with_timeout(
            self._p6.reason, query, domain, label="P6"
        )
        return trm_res, p6_res

    def _call_with_timeout(
        self,
        fn,
        query: str,
        domain: str,
        label: str = "",
    ) -> ReasoningResult:
        """Call *fn(query, domain)* with a hard wall-clock timeout.

        Uses a single-thread executor so the timeout works on all platforms
        (signal.alarm is UNIX-only and can't be used in threads).
        """
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut: Future = pool.submit(fn, query, domain)
            try:
                return fut.result(timeout=self._timeout)
            except FuturesTimeoutError:
                fut.cancel()
                logger.warning(
                    "%s call timed out after %ds", label, self._timeout
                )
                return ReasoningResult(
                    answer="",
                    confidence=0.0,
                    latency_ms=float(self._timeout * 1000),
                    source=f"{label.lower()}/timeout",
                    raw={"error": f"timeout after {self._timeout}s"},
                )
            except Exception as exc:
                logger.warning("%s call failed: %s", label, exc)
                return ReasoningResult(
                    answer="",
                    confidence=0.0,
                    latency_ms=0.0,
                    source=f"{label.lower()}/error",
                    raw={"error": str(exc)},
                )

    @staticmethod
    def _select_answer(
        trm_res: ReasoningResult,
        p6_res: ReasoningResult,
        verification,
    ) -> Tuple[str, str]:
        """Pick the verified answer and status string."""
        if verification.match:
            if verification.preferred == "trm":
                return trm_res.answer, "verified"
            return p6_res.answer, "verified"
        # Conflict — return both answers so the conversation LLM can reconcile
        parts = []
        if trm_res.answer:
            parts.append(f"[Reasoner A] {trm_res.answer}")
        if p6_res.answer:
            parts.append(f"[Reasoner B] {p6_res.answer}")
        combined = "\n\n".join(parts) if parts else ""
        return combined, "conflict"
