"""
expert_post_check.runner
=========================
Top-level orchestrator for the Expert Post-Check System.

Flow
----
1. Evict any stale Ollama model from VRAM (ollama_guard)
2. Run TRM-slot (qwen3.5:9b stub) with per-call timeout
3. Unload TRM model from GPU VRAM immediately after TRM finishes
4. Run 6-phase pipeline with per-call timeout
5. Fuzzy verify both answers with cosine-sim
6. Apply RL weight update for the domain
7. Return PostCheckResult with verified_answer + full metadata

NOTE: Parallel mode is intentionally disabled until the TRM architecture
replaces the qwen3.5 stub.  Running two LLM inferences concurrently on
a single GPU saturates VRAM and causes runaway timeout loops.  Re-enable
by swapping the _run_serial() call in run() back to a hardware-probed
dispatch once the stub is replaced.

NOTE on VRAM management:
At the top of _run_serial(), ollama_guard.evict_all() clears any model
that was left resident from a previous crashed request.  After TRM
inference completes, _run_serial() calls self._trm.unload() to evict
qwen3.5 from GPU VRAM before P6 (Phase2Pipeline) starts.  This frees
~4.8 GiB, which is enough headroom for all-mpnet-base-v2 (420 MB,
always CPU) and Phase2Pipeline's own memory needs.

PostCheckResult fields
----------------------
    verified_answer     : str    -- best answer to hand to conversation LLM
    verification_status : str    -- "verified" | "conflict" | "trm_only" | "p6_only" | "degraded"
    trm_answer          : str
    p6_answer           : str
    similarity          : float
    preferred_source    : str    -- "trm" | "p6" | "conflict"
    trm_confidence      : float
    p6_confidence       : float
    rl_update           : dict   -- WeightUpdateResult as dict
    mode                : str    -- always "serial" for now
    conflict_hint       : str    -- populated on mismatch for escalation
    total_latency_ms    : float
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
)
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Tuple

from expert_post_check.trm_adapter import TRMAdapter, ReasoningResult
from expert_post_check.p6_adapter import P6Adapter
from expert_post_check.fuzzy_verifier import FuzzyVerifier
from expert_post_check.rl_weight import RLWeightStore
from expert_post_check.ollama_guard import evict_all as _evict_ollama

logger = logging.getLogger(__name__)

# Wall-clock budget per reasoner call (seconds).
#
# Raised from 60 s → 150 s to accommodate Qwen3.5 cold-start inference
# time (~113 s observed in logs).  The 60 s budget was firing before the
# model finished loading, returning an empty result; the background thread
# then completed 53 s later and the orphaned result was discarded, which
# caused run_workflow.py to treat the empty verified_answer as a signal
# to re-invoke the entire phase — producing the infinite loop visible in
# logs at 21:03:40 → 21:05:33 → 21:06:27.
#
# 150 s provides enough headroom for cold-start on consumer-grade GPUs
# while still bounding truly stalled calls.  Raise to 240 s if inference
# on larger Qwen3 variants still exceeds this budget.
REASONER_TIMEOUT_S = 150


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
    """Orchestrate the expert post-check pipeline (serial mode).

    Both reasoners run sequentially: TRM first, then P6.  Each call has
    an independent wall-clock timeout so a stalled LLM cannot block the
    entire request indefinitely.

    VRAM discipline:
      - _run_serial() calls ollama_guard.evict_all() at the very start to
        flush any model left resident from a previous crashed request.
      - TRM (qwen3.5:9b, ~4.8 GiB) is unloaded from the GPU immediately
        after its future resolves, before P6 starts.
      - P6 uses Phase2Pipeline which itself uses all-mpnet-base-v2 pinned
        to CPU, so no VRAM is consumed after the unload.
    """

    def __init__(
        self,
        fuzzy_threshold: float = 0.72,
        reasoner_timeout: int = REASONER_TIMEOUT_S,
    ) -> None:
        self._trm = TRMAdapter()
        self._p6 = P6Adapter()
        self._verifier = FuzzyVerifier(threshold=fuzzy_threshold)
        self._rl = RLWeightStore()
        self._timeout = reasoner_timeout
        logger.info(
            "PostCheckRunner initialised -- mode=serial timeout=%ds",
            self._timeout,
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
            "PostCheck START -- query='%s' domain=%s expert=%s",
            query[:80], domain, expert_name,
        )

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
            "PostCheck END -- status=%s sim=%.3f latency=%.1f ms",
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
            mode="serial",
            conflict_hint=verification.conflict_hint,
            total_latency_ms=round(total_ms, 3),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_serial(
        self, query: str, domain: str
    ) -> Tuple[ReasoningResult, ReasoningResult]:
        """Run TRM-slot then P6 sequentially, each guarded by a timeout.

        Starts with a best-effort VRAM flush via ollama_guard.evict_all()
        so that stale models from a previous crashed request do not eat
        into the GPU budget before Qwen3.5 loads.

        TRM is unloaded from GPU VRAM immediately after its future
        resolves so that P6 and any downstream embedders start with a
        clean VRAM budget.  The unload is best-effort: a failure is
        logged as a warning and does not abort the pipeline.
        """
        # --- VRAM pre-flight: evict any stale Ollama resident ---------------
        try:
            evicted = _evict_ollama()
            if evicted:
                logger.info(
                    "_run_serial: pre-flight evicted stale Ollama models: %s",
                    evicted,
                )
        except Exception as exc:
            logger.warning(
                "_run_serial: ollama pre-flight eviction failed (non-fatal): %s", exc
            )
        # --------------------------------------------------------------------

        trm_res = self._call_with_timeout(
            self._trm.reason, query, domain, label="TRM"
        )
        # Evict qwen3.5 from VRAM before starting P6 / embedding stages.
        self._trm.unload()

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

        Uses a single-thread executor so the timeout works on all
        platforms (signal.alarm is UNIX-only and cannot be used inside
        threads).

        Note on fut.cancel():
        ---------------------
        For a ThreadPoolExecutor future that is already *running*,
        cancel() is a no-op -- the underlying thread cannot be
        interrupted mid-call.  This is intentional: we return the empty
        result immediately so the pipeline can proceed, while the
        background thread finishes harmlessly and its result is discarded
        when the executor context exits.  The timeout value must therefore
        be set high enough that legitimate slow calls (e.g. Qwen3.5
        cold-start) are not prematurely abandoned.
        """
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut: Future = pool.submit(fn, query, domain)
            try:
                return fut.result(timeout=self._timeout)
            except FuturesTimeoutError:
                fut.cancel()
                logger.warning(
                    "%s call timed out after %ds -- returning empty result",
                    label, self._timeout,
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
        # Conflict -- return both answers so the conversation LLM can reconcile
        parts = []
        if trm_res.answer:
            parts.append(f"[Reasoner A] {trm_res.answer}")
        if p6_res.answer:
            parts.append(f"[Reasoner B] {p6_res.answer}")
        combined = "\n\n".join(parts) if parts else ""
        return combined, "conflict"
