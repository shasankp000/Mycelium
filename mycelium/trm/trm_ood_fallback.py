"""
Part D — TRM OOD Fallback
==========================
When TRMReasoner's halt_confidence is low AND its domain prediction
diverges from the spectral prior, this module provides a structured
fallback that routes the query through the 6-phase reasoning pipeline
before handing back to MultiLensRouter.

Design
------
Two components are defined here:

    TRMOODHead
    ----------
    A lightweight nn.Module that attaches to TRMReasoner and produces
    an OOD confidence score from final_z.mean(dim=-1).  Trained jointly
    with TRMReasoner during the sim run (trm_training_sim.py) using
    binary cross-entropy against a synthetic OOD label (1 = OOD,
    0 = in-distribution).

    TRMOODFallback
    --------------
    Pure-Python orchestrator.  Called by the pipeline (run_workflow.py)
    when TRMOODHead fires.  Executes the following chain:

        1. Layer 1-2 contradiction analysis
           (mycelium/contradiction/classifier.py + dag_decomposer.py)
        2. [STUB] Layers 3-6  — predicate generation → evidence grounding
           → hypothesis evaluation → synthesis.  Each layer is called as
           an optional component; if unavailable the chain degrades
           gracefully to MultiLensRouter.
        3. MultiLensRouter final routing pass, optionally enriched with
           the DAG context produced by Layer 2.

    The fallback never raises — every failure is caught, logged, and
    the call falls through to the router so the pipeline keeps running.

OOD trigger heuristic (two conditions, both must hold)
------------------------------------------------------
    halt_confidence < cfg.ood_halt_threshold   (default 0.55)
        TRM never reached a stable answer — not confident in its halt.

    domain_divergence > cfg.ood_divergence_threshold  (default 0.35)
        argmax(domain_probs) disagrees strongly with spectral_vec peak.
        Measured as  1 - spectral_vec[primary_domain_idx].

Only queries that satisfy BOTH conditions are considered OOD by the
heuristic.  The learned TRMOODHead adds a third signal once trained.

Fallback escalation levels
--------------------------
    LEVEL_0  — heuristic only: straight to router (no reasoning chain)
    LEVEL_1  — Layer 1-2: contradiction classifier + DAG decomposer
    LEVEL_2  — Layers 1-6: full reasoning pipeline (stubs for L3-L6)

The level is selected at runtime based on ood_head_confidence:
    ood_head_confidence >= 0.8  → LEVEL_2
    0.5 <= ood_head_confidence < 0.8  → LEVEL_1
    < 0.5  → LEVEL_0  (heuristic was enough, head agrees it's borderline)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .config import TRMConfig

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# OOD Head
# ---------------------------------------------------------------------------

class TRMOODHead(nn.Module):
    """
    Lightweight OOD detector that operates on TRMReasoner.final_z.

    Architecture
    ------------
    final_z : [B, n_domains, D]
        → mean-pool over sequence dim  → [B, D]
        → LayerNorm
        → Linear(D, hidden)  + GELU
        → Linear(hidden, 1)
        → sigmoid  → ood_confidence in [0, 1]

    A score near 1.0 means TRM believes the query is OOD.
    A score near 0.0 means TRM is in-distribution.

    Parameters
    ----------
    hidden_size : int
        Must match TRMConfig.hidden_size.
    bottleneck : int
        Inner projection size (default hidden_size // 4).
    """

    def __init__(self, hidden_size: int, bottleneck: Optional[int] = None) -> None:
        super().__init__()
        bottleneck = bottleneck or max(hidden_size // 4, 16)
        self.norm   = nn.LayerNorm(hidden_size)
        self.proj1  = nn.Linear(hidden_size, bottleneck)
        self.act    = nn.GELU()
        self.proj2  = nn.Linear(bottleneck, 1)

    def forward(self, final_z: Tensor) -> Tensor:
        """
        Parameters
        ----------
        final_z : Tensor [B, seq, D]
            final_z from TRMOutput (already detached by TRMReasoner).

        Returns
        -------
        Tensor [B]  — OOD confidence in [0, 1].
        """
        # Pool over sequence dimension
        z = final_z.mean(dim=1)          # [B, D]
        z = self.norm(z)
        z = self.act(self.proj1(z))      # [B, bottleneck]
        logit = self.proj2(z).squeeze(-1)  # [B]
        return logit.sigmoid()


# ---------------------------------------------------------------------------
# Escalation level
# ---------------------------------------------------------------------------

class FallbackLevel(IntEnum):
    LEVEL_0 = 0   # heuristic only — straight to router
    LEVEL_1 = 1   # Layer 1-2 reasoning (contradiction + DAG)
    LEVEL_2 = 2   # Full 6-phase pipeline (L3-L6 stubs in)


# ---------------------------------------------------------------------------
# Fallback result dataclass
# ---------------------------------------------------------------------------

@dataclass
class OODFallbackResult:
    """
    Return value from TRMOODFallback.route().

    Attributes
    ----------
    triggered : bool
        True if the OOD fallback path was taken.
    fallback_level : FallbackLevel
        Which escalation level was used.
    ood_head_confidence : float
        Raw TRMOODHead score (0-1).  0.0 if head not available.
    selected_domain : int
        Final domain index selected by the fallback chain.
    router_result : Any
        Raw return value from MultiLensRouter.route().  The pipeline
        should use this for the actual routing decision.
    dag_context : Optional[Dict]
        Decomposed sub-claims from DAGDecomposer (Level 1+), or None.
    reasoning_trace : Optional[Dict]
        Structured trace from the reasoning chain (Level 2), or None.
    reason : str
        Human-readable explanation of why the fallback fired.
    """
    triggered: bool
    fallback_level: FallbackLevel
    ood_head_confidence: float
    selected_domain: int
    router_result: Any
    dag_context: Optional[Dict] = None
    reasoning_trace: Optional[Dict] = None
    reason: str = ""


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

class TRMOODFallback:
    """
    Orchestrates the 6-phase reasoning pipeline as a fallback when
    TRMReasoner reports an OOD query.

    Parameters
    ----------
    cfg : TRMConfig
        Used for ood_halt_threshold and ood_divergence_threshold.
    ood_head : TRMOODHead, optional
        If provided, used for the learned OOD confidence signal.
        If None, only the heuristic trigger is used.
    multi_lens_router : Any, optional
        The live MultiLensRouter instance.  Its .route() method is
        called as the final step (and as LEVEL_0 fallback).
    dag_decomposer : Any, optional
        mycelium.reasoning.dag_decomposer.DAGDecomposer instance.
        Required for LEVEL_1+.  If None, Level 1 is skipped.
    contradiction_classifier : Any, optional
        mycelium.contradiction.classifier.ContradictionClassifier.
        Required for LEVEL_1+.  Provides Layer 1 signals.
    layer3_predicate_gen : Any, optional
        [STUB] Future: mycelium.reasoning.predicate_generator.PredicateGenerator.
    layer4_evidence_grounder : Any, optional
        [STUB] Future: mycelium.reasoning.evidence_grounder.EvidenceGrounder.
    layer5_hypothesis_eval : Any, optional
        [STUB] Future: mycelium.reasoning.hypothesis_evaluator.HypothesisEvaluator.
    layer6_synthesizer : Any, optional
        [STUB] Future: mycelium.reasoning.reasoning_synthesizer.ReasoningSynthesizer.
    """

    def __init__(
        self,
        cfg: TRMConfig,
        *,
        ood_head: Optional[TRMOODHead] = None,
        multi_lens_router: Optional[Any] = None,
        dag_decomposer: Optional[Any] = None,
        contradiction_classifier: Optional[Any] = None,
        layer3_predicate_gen: Optional[Any] = None,
        layer4_evidence_grounder: Optional[Any] = None,
        layer5_hypothesis_eval: Optional[Any] = None,
        layer6_synthesizer: Optional[Any] = None,
    ) -> None:
        self.cfg = cfg
        self.ood_head = ood_head
        self.router = multi_lens_router
        self.dag_decomposer = dag_decomposer
        self.contradiction_classifier = contradiction_classifier
        # Layers 3-6 — all optional / stubbed
        self._l3 = layer3_predicate_gen
        self._l4 = layer4_evidence_grounder
        self._l5 = layer5_hypothesis_eval
        self._l6 = layer6_synthesizer

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def should_trigger(
        self,
        trm_output,          # TRMOutput from TRMReasoner.forward()
        spectral_vec: Tensor,  # [B, n_domains] — MultiLensRouter soft labels
    ) -> tuple[bool, float, str]:
        """
        Decide whether the OOD fallback should fire for this query.

        Returns
        -------
        (triggered, ood_head_confidence, reason)
        """
        halt_conf   = trm_output.halt_confidence.mean().item()
        domain_probs = trm_output.domain_probs          # [B, n_domains]
        primary_idx  = trm_output.primary_domain_idx    # [B]

        # Heuristic: how much does TRM's top domain agree with spectral prior?
        # spectral_vec[b, primary_idx[b]] — gather the spectral score for
        # whatever domain TRM picked.
        batch_size = primary_idx.shape[0]
        spectral_agreement = spectral_vec[
            torch.arange(batch_size, device=primary_idx.device),
            primary_idx,
        ].mean().item()
        domain_divergence = 1.0 - spectral_agreement

        heuristic_ood = (
            halt_conf < self.cfg.ood_halt_threshold
            and domain_divergence > self.cfg.ood_divergence_threshold
        )

        # Learned head signal
        ood_head_conf = 0.0
        if self.ood_head is not None:
            with torch.no_grad():
                ood_head_conf = self.ood_head(trm_output.final_z).mean().item()

        triggered = heuristic_ood or (ood_head_conf >= 0.5)

        if triggered:
            reason = (
                f"halt_conf={halt_conf:.3f} "
                f"domain_divergence={domain_divergence:.3f} "
                f"ood_head={ood_head_conf:.3f}"
            )
        else:
            reason = "in-distribution"

        return triggered, ood_head_conf, reason

    def route(
        self,
        query: str,
        trm_output,            # TRMOutput
        spectral_vec: Tensor,  # [B, n_domains]
        domain: str = "unknown",
        expert_metadata: Optional[Dict] = None,
    ) -> OODFallbackResult:
        """
        Full fallback routing chain.

        Always returns an OODFallbackResult.  Never raises.

        Parameters
        ----------
        query : str
            Raw query text.
        trm_output : TRMOutput
            Output from TRMReasoner.forward().
        spectral_vec : Tensor [B, n_domains]
            Soft domain labels from MultiLensRouter.
        domain : str
            Best-guess domain string (for DAGDecomposer template lookup).
        expert_metadata : dict, optional
            Pre-computed expert statistics for Layer 1 analysis.
        """
        triggered, ood_conf, reason = self.should_trigger(trm_output, spectral_vec)

        if not triggered:
            # Fast path — TRM is in-distribution, route directly
            return OODFallbackResult(
                triggered=False,
                fallback_level=FallbackLevel.LEVEL_0,
                ood_head_confidence=ood_conf,
                selected_domain=trm_output.primary_domain_idx[0].item(),
                router_result=None,
                reason=reason,
            )

        logger.info("TRMOODFallback: triggered — %s", reason)

        # Choose escalation level from head confidence
        if ood_conf >= 0.8:
            level = FallbackLevel.LEVEL_2
        elif ood_conf >= 0.5:
            level = FallbackLevel.LEVEL_1
        else:
            # Heuristic fired but head is uncertain — LEVEL_1 still better
            # than LEVEL_0 as long as we have the classifier.
            level = FallbackLevel.LEVEL_1 if self.contradiction_classifier else FallbackLevel.LEVEL_0

        dag_ctx: Optional[Dict] = None
        reasoning_trace: Optional[Dict] = None

        # ---------------------------------------------------------------- #
        # LEVEL_1: Layer 1-2 — contradiction detection + DAG decomposition #
        # ---------------------------------------------------------------- #
        if level >= FallbackLevel.LEVEL_1:
            dag_ctx, level = self._run_layers_1_2(
                query, domain, expert_metadata, ood_conf, level
            )

        # ---------------------------------------------------------------- #
        # LEVEL_2: Layers 3-6 (predicate → evidence → eval → synthesize)  #
        # ---------------------------------------------------------------- #
        if level >= FallbackLevel.LEVEL_2 and dag_ctx is not None:
            reasoning_trace = self._run_layers_3_6(query, domain, dag_ctx)

        # ---------------------------------------------------------------- #
        # Final step: MultiLensRouter with optional DAG context            #
        # ---------------------------------------------------------------- #
        router_result = self._call_router(query, dag_ctx)
        selected_domain = self._extract_domain(router_result, trm_output)

        return OODFallbackResult(
            triggered=True,
            fallback_level=level,
            ood_head_confidence=ood_conf,
            selected_domain=selected_domain,
            router_result=router_result,
            dag_context=dag_ctx,
            reasoning_trace=reasoning_trace,
            reason=reason,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _run_layers_1_2(
        self,
        query: str,
        domain: str,
        expert_metadata: Optional[Dict],
        ood_conf: float,
        current_level: FallbackLevel,
    ) -> tuple[Optional[Dict], FallbackLevel]:
        """Run Layer 1 (ContradictionClassifier) + Layer 2 (DAGDecomposer).

        Returns (dag_context, effective_level).  On any exception the
        level is downgraded and dag_context is None.
        """
        if self.contradiction_classifier is None or self.dag_decomposer is None:
            logger.debug("TRMOODFallback: L1/L2 components not available, downgrading to L0")
            return None, FallbackLevel.LEVEL_0

        try:
            # Layer 1: classify the input against learned patterns
            # ContradictionClassifier.classify_raw() accepts raw text + metadata
            l1_result = self.contradiction_classifier.classify_raw(
                text=query,
                expert_metadata=expert_metadata or {},
            )
            logger.debug(
                "TRMOODFallback L1: type=%s contradiction_score=%.3f",
                l1_result.get("type", "unknown"),
                l1_result.get("contradiction_score", 0.0),
            )

            # If Layer 1 says NOISE — stop, not worth decomposing
            if l1_result.get("type") == "NOISE":
                logger.debug("TRMOODFallback: Layer 1 classified as NOISE, downgrading to L0")
                return None, FallbackLevel.LEVEL_0

            # Layer 2: DAG decomposition of the core claim
            dag_result = self.dag_decomposer.decompose(
                claim_text=query,
                domain=domain,
                expert_metadata=expert_metadata or {},
            )
            logger.debug(
                "TRMOODFallback L2: %d sub-claims decomposed",
                len(dag_result.get("sub_claims", [])),
            )

            dag_ctx = {
                "l1": l1_result,
                "l2": dag_result,
                "domain": domain,
            }
            return dag_ctx, current_level

        except Exception as exc:
            logger.warning("TRMOODFallback: Layer 1-2 failed (%s), falling back to L0", exc)
            return None, FallbackLevel.LEVEL_0

    def _run_layers_3_6(
        self,
        query: str,
        domain: str,
        dag_ctx: Dict,
    ) -> Optional[Dict]:
        """
        Run Layers 3-6 (predicate → evidence → eval → synthesize).

        All four layers are optional.  Results are accumulated into a
        single reasoning_trace dict.  Any missing layer is skipped with
        a warning; the chain continues from whatever was last produced.
        """
        trace: Dict = {"query": query, "domain": domain, "layers": {}}

        try:
            # Layer 3 — Predicate generator
            l3_result = None
            if self._l3 is not None:
                l3_result = self._l3.generate(
                    decomposed_claim=dag_ctx["l2"],
                    claim_type=dag_ctx["l2"].get("claim_type", {}),
                )
                trace["layers"]["l3"] = l3_result
                logger.debug(
                    "TRMOODFallback L3: %d predicates generated",
                    len(l3_result.get("positive_predicates", [])),
                )
            else:
                logger.debug("TRMOODFallback: Layer 3 not available (stub)")

            # Layer 4 — Evidence grounder
            l4_result = None
            if self._l4 is not None and l3_result is not None:
                l4_result = self._l4.ground(
                    predicates=l3_result,
                    domain=domain,
                )
                trace["layers"]["l4"] = l4_result
                logger.debug(
                    "TRMOODFallback L4: evidence_score=%.3f",
                    l4_result.get("evidence_score", 0.0),
                )
            else:
                logger.debug("TRMOODFallback: Layer 4 not available (stub)")

            # Layer 5 — Hypothesis evaluator
            l5_result = None
            if self._l5 is not None and l4_result is not None:
                l5_result = self._l5.evaluate(
                    predicates=l3_result,
                    evidence=l4_result,
                    dag_context=dag_ctx["l2"],
                )
                trace["layers"]["l5"] = l5_result
                logger.debug(
                    "TRMOODFallback L5: hypothesis=%s confidence=%.3f",
                    l5_result.get("verdict", "unknown"),
                    l5_result.get("confidence", 0.0),
                )
            else:
                logger.debug("TRMOODFallback: Layer 5 not available (stub)")

            # Layer 6 — Reasoning synthesizer + cache
            if self._l6 is not None:
                l6_result = self._l6.synthesize(
                    query=query,
                    l3=l3_result,
                    l4=l4_result,
                    l5=l5_result,
                    dag_context=dag_ctx,
                )
                trace["layers"]["l6"] = l6_result
                trace["conclusion"] = l6_result.get("conclusion", "UNCERTAIN")
                trace["confidence"] = l6_result.get("confidence", 0.0)
                logger.debug(
                    "TRMOODFallback L6: conclusion=%s confidence=%.3f",
                    trace["conclusion"],
                    trace["confidence"],
                )
            else:
                logger.debug("TRMOODFallback: Layer 6 not available (stub)")
                trace["conclusion"] = "UNCERTAIN"
                trace["confidence"] = 0.0

        except Exception as exc:
            logger.warning("TRMOODFallback: Layers 3-6 chain failed (%s)", exc)
            trace["error"] = str(exc)

        return trace

    def _call_router(self, query: str, dag_ctx: Optional[Dict]) -> Any:
        """Call MultiLensRouter, optionally passing dag_context."""
        if self.router is None:
            logger.debug("TRMOODFallback: no router configured — returning raw dag_ctx")
            return dag_ctx
        try:
            # MultiLensRouter.route() accepts **kwargs; pass dag_context if
            # the router supports it (duck-typed — ignore TypeError if not).
            try:
                return self.router.route(query, dag_context=dag_ctx)
            except TypeError:
                return self.router.route(query)
        except Exception as exc:
            logger.error("TRMOODFallback: router call failed (%s)", exc)
            return None

    def _extract_domain(
        self,
        router_result: Any,
        trm_output,
    ) -> int:
        """Pull a domain index out of whatever the router returned."""
        # Most routers return an object with .selected_domain or .domain_idx
        for attr in ("selected_domain", "domain_idx", "domain_index"):
            val = getattr(router_result, attr, None)
            if val is not None:
                return int(val)
        # Dict-style
        if isinstance(router_result, dict):
            for key in ("selected_domain", "domain_idx", "domain_index"):
                if key in router_result:
                    return int(router_result[key])
        # Ultimate fallback: TRM's own (low-confidence) prediction
        return int(trm_output.primary_domain_idx[0].item())
