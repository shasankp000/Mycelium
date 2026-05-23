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
    Pure-Python orchestrator.  Called by the pipeline when TRMOODHead
    fires.  Executes the following chain:

        Layer 1-2
        ~~~~~~~~~
        Raw query text
            → CanonicalizeAndHash.process()     ← §46 pipeline: SRL →
              canonical_form → semantic_hash → IRNode
            → DAGDecomposer.decompose(root_node) ← DFS sub-claim expansion
            → IRGraph  (the structured representation of the query)

        Contradiction check (optional, on sub-claim pairs)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        If the DAG contains ≥2 sibling nodes, ContradictionClassifier.classify()
        is called on the first pair to detect whether the query itself
        contains an internal contradiction (e.g. "X helps but also hurts Y").
        This is informational only — the fallback never halts on it.

        Layers 3-6 (stubs — filled by subsequent files)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        PredicateGenerator, EvidenceGrounder, HypothesisEvaluator,
        ReasoningSynthesizer — each is an optional injectable component.
        Missing layers degrade gracefully to MultiLensRouter.

        MultiLensRouter
        ~~~~~~~~~~~~~~~
        Always the terminal step.  Called with optional dag_context so
        the spectral soft-labels can be enriched with structural signal.

    The fallback never raises — every failure is caught, logged, and
    the call falls through to the router so the pipeline keeps running.

OOD trigger heuristic (two conditions, both must hold)
------------------------------------------------------
    halt_confidence < cfg.ood_halt_threshold   (default 0.55)
        TRM never reached a stable answer.

    domain_divergence > cfg.ood_divergence_threshold  (default 0.35)
        argmax(domain_probs) disagrees strongly with spectral_vec peak.
        Measured as  1 - spectral_vec[primary_domain_idx].

    The learned TRMOODHead adds a third signal once trained.

Fallback escalation levels
--------------------------
    LEVEL_0  — heuristic only: straight to router (no reasoning chain)
    LEVEL_1  — Layer 1-2: CanonicalizeAndHash + DAGDecomposer
    LEVEL_2  — Layers 1-6: full reasoning pipeline

The level is selected from ood_head_confidence:
    >= 0.8   → LEVEL_2
    >= 0.5   → LEVEL_1
    <  0.5   → LEVEL_1 if canonicalizer available, else LEVEL_0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, Optional

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

    Score near 1.0 = TRM believes query is OOD.
    Score near 0.0 = in-distribution.

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
        self.norm  = nn.LayerNorm(hidden_size)
        self.proj1 = nn.Linear(hidden_size, bottleneck)
        self.act   = nn.GELU()
        self.proj2 = nn.Linear(bottleneck, 1)

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
        z = final_z.mean(dim=1)             # [B, D]
        z = self.norm(z)
        z = self.act(self.proj1(z))          # [B, bottleneck]
        return self.proj2(z).squeeze(-1).sigmoid()  # [B]


# ---------------------------------------------------------------------------
# Escalation level
# ---------------------------------------------------------------------------

class FallbackLevel(IntEnum):
    LEVEL_0 = 0   # straight to router
    LEVEL_1 = 1   # CanonicalizeAndHash + DAGDecomposer
    LEVEL_2 = 2   # full 6-phase pipeline


# ---------------------------------------------------------------------------
# Result dataclass
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
        Raw return value from MultiLensRouter.route().
    ir_graph : Any
        IRGraph produced by DAGDecomposer (Level 1+), or None.
    dag_context : Optional[Dict]
        Flat summary dict derived from the IRGraph for downstream layers.
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
    ir_graph: Any = None
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
    ood_head : TRMOODHead, optional
    multi_lens_router : Any, optional
        Live MultiLensRouter instance.  Its .route() is the terminal step.
    canonicalizer : CanonicalizeAndHash, optional
        Phase B pipeline.  Required for LEVEL_1+.  If None, Level 1
        is skipped and the fallback degrades to LEVEL_0.
    dag_decomposer : DAGDecomposer, optional
        Phase D DAG/DFS decomposer.  Takes an IRNode, returns an IRGraph.
    contradiction_classifier : ContradictionClassifier, optional
        Phase E classifier.  Used on sibling node pairs inside the
        produced IRGraph (informational — never blocks routing).
    layer3_predicate_gen : PredicateGenerator, optional
    layer4_evidence_grounder : EvidenceGrounder, optional
    layer5_hypothesis_eval : HypothesisEvaluator, optional
    layer6_synthesizer : ReasoningSynthesizer, optional
    """

    def __init__(
        self,
        cfg: TRMConfig,
        *,
        ood_head: Optional[TRMOODHead] = None,
        multi_lens_router: Optional[Any] = None,
        canonicalizer: Optional[Any] = None,
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
        self.canonicalizer = canonicalizer
        self.dag_decomposer = dag_decomposer
        self.contradiction_classifier = contradiction_classifier
        self._l3 = layer3_predicate_gen
        self._l4 = layer4_evidence_grounder
        self._l5 = layer5_hypothesis_eval
        self._l6 = layer6_synthesizer

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def should_trigger(
        self,
        trm_output,
        spectral_vec: Tensor,
    ) -> tuple[bool, float, str]:
        """
        Decide whether the OOD fallback should fire.

        Returns (triggered, ood_head_confidence, reason).
        """
        halt_conf    = trm_output.halt_confidence.mean().item()
        primary_idx  = trm_output.primary_domain_idx          # [B]
        batch_size   = primary_idx.shape[0]

        spectral_agreement = spectral_vec[
            torch.arange(batch_size, device=primary_idx.device),
            primary_idx,
        ].mean().item()
        domain_divergence = 1.0 - spectral_agreement

        heuristic_ood = (
            halt_conf < self.cfg.ood_halt_threshold
            and domain_divergence > self.cfg.ood_divergence_threshold
        )

        ood_head_conf = 0.0
        if self.ood_head is not None:
            with torch.no_grad():
                ood_head_conf = self.ood_head(trm_output.final_z).mean().item()

        triggered = heuristic_ood or (ood_head_conf >= 0.5)
        reason = (
            f"halt_conf={halt_conf:.3f} "
            f"domain_divergence={domain_divergence:.3f} "
            f"ood_head={ood_head_conf:.3f}"
        ) if triggered else "in-distribution"

        return triggered, ood_head_conf, reason

    def route(
        self,
        query: str,
        trm_output,
        spectral_vec: Tensor,
        domain: str = "unknown",
        expert_metadata: Optional[Dict] = None,
    ) -> OODFallbackResult:
        """
        Full fallback routing chain.  Always returns an OODFallbackResult.
        Never raises.
        """
        triggered, ood_conf, reason = self.should_trigger(trm_output, spectral_vec)

        if not triggered:
            return OODFallbackResult(
                triggered=False,
                fallback_level=FallbackLevel.LEVEL_0,
                ood_head_confidence=ood_conf,
                selected_domain=int(trm_output.primary_domain_idx[0].item()),
                router_result=None,
                reason=reason,
            )

        logger.info("TRMOODFallback: triggered — %s", reason)

        # Escalation level
        if ood_conf >= 0.8:
            level = FallbackLevel.LEVEL_2
        elif ood_conf >= 0.5:
            level = FallbackLevel.LEVEL_1
        else:
            level = FallbackLevel.LEVEL_1 if self.canonicalizer else FallbackLevel.LEVEL_0

        ir_graph    = None
        dag_ctx: Optional[Dict] = None
        reasoning_trace: Optional[Dict] = None

        # ---------------------------------------------------------------- #
        # LEVEL_1  —  CanonicalizeAndHash → IRNode → DAGDecomposer         #
        # ---------------------------------------------------------------- #
        if level >= FallbackLevel.LEVEL_1:
            ir_graph, dag_ctx, level = self._run_layers_1_2(
                query, domain, expert_metadata, level
            )

        # ---------------------------------------------------------------- #
        # LEVEL_2  —  Layers 3-6 (predicate → evidence → eval → synthesize) #
        # ---------------------------------------------------------------- #
        if level >= FallbackLevel.LEVEL_2 and dag_ctx is not None:
            reasoning_trace = self._run_layers_3_6(query, domain, dag_ctx)

        # ---------------------------------------------------------------- #
        # Terminal: MultiLensRouter                                        #
        # ---------------------------------------------------------------- #
        router_result  = self._call_router(query, dag_ctx)
        selected_domain = self._extract_domain(router_result, trm_output)

        return OODFallbackResult(
            triggered=True,
            fallback_level=level,
            ood_head_confidence=ood_conf,
            selected_domain=selected_domain,
            router_result=router_result,
            ir_graph=ir_graph,
            dag_context=dag_ctx,
            reasoning_trace=reasoning_trace,
            reason=reason,
        )

    # ------------------------------------------------------------------ #
    # Internal — Layer 1-2                                                #
    # ------------------------------------------------------------------ #

    def _run_layers_1_2(
        self,
        query: str,
        domain: str,
        expert_metadata: Optional[Dict],
        current_level: FallbackLevel,
    ) -> tuple[Any, Optional[Dict], FallbackLevel]:
        """
        Layer 1: CanonicalizeAndHash.process(query) → list[IRNode]
        Layer 2: DAGDecomposer.decompose(root_node)  → IRGraph

        Returns (ir_graph, dag_context_dict, effective_level).
        On any failure degrades to LEVEL_0 and returns (None, None, LEVEL_0).
        """
        if self.canonicalizer is None or self.dag_decomposer is None:
            logger.debug("TRMOODFallback: canonicalizer/decomposer not available, downgrading to L0")
            return None, None, FallbackLevel.LEVEL_0

        try:
            # ---- Layer 1: §46 pipeline — raw text → IRNode(s) ----------
            ir_nodes = self.canonicalizer.process(
                query,
                abstraction_level=2,
                source="trm_ood_fallback",
            )
            if not ir_nodes:
                logger.debug("TRMOODFallback L1: no IRNodes produced, downgrading to L0")
                return None, None, FallbackLevel.LEVEL_0

            root_node = ir_nodes[0]
            logger.debug(
                "TRMOODFallback L1: IRNode id=%s family=%s",
                root_node.id,
                root_node.semantic_signature.predicate_family,
            )

            # ---- Optional: check sibling pairs for internal contradiction ---
            contradiction_signal = None
            if self.contradiction_classifier is not None and len(ir_nodes) >= 2:
                try:
                    contradiction_signal = self.contradiction_classifier.classify(
                        ir_nodes[0], ir_nodes[1]
                    )
                    logger.debug(
                        "TRMOODFallback L1 contradiction: type=%s severity=%.2f",
                        contradiction_signal.contradiction_type,
                        contradiction_signal.severity,
                    )
                except Exception as exc:
                    logger.debug("TRMOODFallback: contradiction check skipped (%s)", exc)

            # ---- Layer 2: DFS DAG decomposition → IRGraph ----------------
            ir_graph = self.dag_decomposer.decompose(
                root_node,
                request_id="ood-fallback",
            )
            logger.debug(
                "TRMOODFallback L2: IRGraph id=%s nodes=%d edges=%d",
                ir_graph.graph_id,
                len(ir_graph.nodes),
                len(ir_graph.edges),
            )

            # ---- Build flat dag_context dict for Layers 3-6 --------------
            dag_ctx: Dict = {
                "root_node_id": root_node.id,
                "root_label": root_node.label,
                "predicate_family": root_node.semantic_signature.predicate_family,
                "canonical_form": root_node.semantic_signature.canonical_form,
                "equivalence_family": root_node.semantic_signature.equivalence_family,
                "abstraction_level": root_node.semantic_signature.abstraction_level,
                "sub_claims": [
                    {
                        "id": n.id,
                        "label": n.label,
                        "predicate_family": n.semantic_signature.predicate_family,
                        "confidence": n.confidence_state.overall_confidence,
                    }
                    for n in ir_graph.nodes
                ],
                "n_edges": len(ir_graph.edges),
                "domain": domain,
                "contradiction_signal": (
                    {
                        "type": contradiction_signal.contradiction_type,
                        "severity": contradiction_signal.severity,
                    }
                    if contradiction_signal is not None else None
                ),
            }
            return ir_graph, dag_ctx, current_level

        except Exception as exc:
            logger.warning("TRMOODFallback: Layer 1-2 failed (%s), downgrading to L0", exc)
            return None, None, FallbackLevel.LEVEL_0

    # ------------------------------------------------------------------ #
    # Internal — Layers 3-6                                               #
    # ------------------------------------------------------------------ #

    def _run_layers_3_6(
        self,
        query: str,
        domain: str,
        dag_ctx: Dict,
    ) -> Optional[Dict]:
        """
        Run Layers 3-6 (predicate → evidence → eval → synthesize).
        All four layers are optional.  Results are accumulated into a
        single reasoning_trace dict.  Any missing layer is skipped;
        the chain continues from whatever was last produced.
        """
        trace: Dict = {"query": query, "domain": domain, "layers": {}}

        try:
            # Layer 3 — Predicate generator
            l3_result = None
            if self._l3 is not None:
                l3_result = self._l3.generate(
                    dag_context=dag_ctx,
                )
                trace["layers"]["l3"] = l3_result
                logger.debug(
                    "TRMOODFallback L3: %d predicates generated",
                    len(l3_result.get("positive_predicates", [])),
                )
            else:
                logger.debug("TRMOODFallback: Layer 3 not available")

            # Layer 4 — Evidence grounder
            l4_result = None
            if self._l4 is not None and l3_result is not None:
                l4_result = self._l4.ground(
                    predicates=l3_result,
                    domain=domain,
                    dag_context=dag_ctx,
                )
                trace["layers"]["l4"] = l4_result
                logger.debug(
                    "TRMOODFallback L4: evidence_score=%.3f",
                    l4_result.get("evidence_score", 0.0),
                )
            else:
                logger.debug("TRMOODFallback: Layer 4 not available")

            # Layer 5 — Hypothesis evaluator
            l5_result = None
            if self._l5 is not None and l4_result is not None:
                l5_result = self._l5.evaluate(
                    predicates=l3_result,
                    evidence=l4_result,
                    dag_context=dag_ctx,
                )
                trace["layers"]["l5"] = l5_result
                logger.debug(
                    "TRMOODFallback L5: verdict=%s confidence=%.3f",
                    l5_result.get("verdict", "unknown"),
                    l5_result.get("confidence", 0.0),
                )
            else:
                logger.debug("TRMOODFallback: Layer 5 not available")

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
            else:
                logger.debug("TRMOODFallback: Layer 6 not available")
                trace["conclusion"] = "UNCERTAIN"
                trace["confidence"] = 0.0

        except Exception as exc:
            logger.warning("TRMOODFallback: Layers 3-6 chain failed (%s)", exc)
            trace["error"] = str(exc)

        return trace

    # ------------------------------------------------------------------ #
    # Internal — router + domain extraction                               #
    # ------------------------------------------------------------------ #

    def _call_router(self, query: str, dag_ctx: Optional[Dict]) -> Any:
        """Call MultiLensRouter, optionally passing dag_context."""
        if self.router is None:
            return dag_ctx
        try:
            try:
                return self.router.route(query, dag_context=dag_ctx)
            except TypeError:
                return self.router.route(query)
        except Exception as exc:
            logger.error("TRMOODFallback: router call failed (%s)", exc)
            return None

    def _extract_domain(self, router_result: Any, trm_output) -> int:
        """Pull a domain index out of whatever the router returned."""
        for attr in ("selected_domain", "domain_idx", "domain_index"):
            val = getattr(router_result, attr, None)
            if val is not None:
                return int(val)
        if isinstance(router_result, dict):
            for key in ("selected_domain", "domain_idx", "domain_index"):
                if key in router_result:
                    return int(router_result[key])
        return int(trm_output.primary_domain_idx[0].item())
