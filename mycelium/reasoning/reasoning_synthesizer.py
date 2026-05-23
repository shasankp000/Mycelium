"""
Layer 6 — Reasoning Synthesizer
=================================
Final stage of the 6-phase OOD fallback reasoning chain.

Takes the cumulative outputs of Layers 3-5 and produces:
    1. A structured conclusion dict for MultiLensRouter to consume.
    2. A human-readable reasoning_summary string (for logging/UI).
    3. A cache write to GraphStore so future queries on the same topic
       benefit from this fallback result without re-running the chain.

Synthesis strategy
------------------
The synthesizer follows Mycelium's core epistemic rule:
    Uncertainty is not falsity.  INCONCLUSIVE ≠ REFUTED.

Conclusion labels:
    CONFIDENT_SUPPORT     — verdict=SUPPORTED with confidence >= 0.70
    WEAK_SUPPORT          — verdict=SUPPORTED with confidence < 0.70
    CONFIDENT_REFUTATION  — verdict=REFUTED with confidence >= 0.70
    WEAK_REFUTATION       — verdict=REFUTED with confidence < 0.70
    INCONCLUSIVE          — verdict=INCONCLUSIVE or UNCERTAIN
    CONTRADICTORY         — verdict=CONTRADICTORY
    ESCALATE              — emitted when confidence is below a floor
                            threshold (meaning even Layer 6 can't decide;
                            the pipeline should escalate further or
                            return MultiLensRouter's raw softmax).

Confidence floor: if synthesis_confidence < 0.25, emit ESCALATE and let
MultiLensRouter handle domain selection without any reasoning bias.

Cache write
-----------
If a GraphStore is available, the synthesizer serialises the reasoning
trace into a minimal IRGraph (single CLAIM node labelled with the
conclusion) and calls GraphStore.put().  This means the next time an
identical or near-identical OOD query triggers the fallback, Layer 2's
DAGDecomposer can find pre-computed evidence in the node pool, and
EvidenceGrounder will return higher Tier 1 scores.

Output dict
-----------
    conclusion       : str    — one of the labels above
    confidence       : float  — [0,1]
    verdict          : str    — passthrough from Layer 5
    support_score    : float  — passthrough from Layer 5
    refutation_score : float  — passthrough from Layer 5
    reasoning_summary: str    — multi-line human-readable explanation
    cached           : bool   — True if a cache write succeeded
    domain           : str
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# IR layer — only needed for cache write, fully optional
try:
    from mycelium.ir.graph import IRGraph, IRNode
    from mycelium.ir.primitives import (
        ConfidenceState, TemporalState, ProvenanceChain,
        SemanticSignature, GraphFingerprint,
    )
    from mycelium.ir.serialization import compute_semantic_hash
    _IR_AVAILABLE = True
except ImportError:
    _IR_AVAILABLE = False


# ---------------------------------------------------------------------------
# Conclusion labels
# ---------------------------------------------------------------------------
CONCL_CONFIDENT_SUPPORT    = "CONFIDENT_SUPPORT"
CONCL_WEAK_SUPPORT         = "WEAK_SUPPORT"
CONCL_CONFIDENT_REFUTATION = "CONFIDENT_REFUTATION"
CONCL_WEAK_REFUTATION      = "WEAK_REFUTATION"
CONCL_INCONCLUSIVE         = "INCONCLUSIVE"
CONCL_CONTRADICTORY        = "CONTRADICTORY"
CONCL_ESCALATE             = "ESCALATE"

_CONFIDENCE_FLOOR  = 0.25
_CONFIDENCE_STRONG = 0.70


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ReasoningSynthesizer:
    """
    Layer 6: synthesize a conclusion from the full reasoning chain.

    Parameters
    ----------
    graph_store : GraphStore, optional
        If provided, successful conclusions are cached as single-node
        IRGraphs for future Tier 1 grounding.
    confidence_floor : float
        Below this synthesis confidence, emit ESCALATE (default 0.25).
    confidence_strong : float
        Above this confidence, emit CONFIDENT_* labels (default 0.70).
    """

    def __init__(
        self,
        *,
        graph_store: Optional[Any] = None,
        confidence_floor: float = _CONFIDENCE_FLOOR,
        confidence_strong: float = _CONFIDENCE_STRONG,
    ) -> None:
        self._store = graph_store
        self._floor  = confidence_floor
        self._strong = confidence_strong

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(
        self,
        query: str,
        l3: Optional[Dict],
        l4: Optional[Dict],
        l5: Optional[Dict],
        dag_context: Optional[Dict] = None,
    ) -> Dict:
        """
        Produce a final conclusion from the reasoning chain outputs.

        Parameters
        ----------
        query : str
            The original raw query string.
        l3 : dict or None
            PredicateGenerator output.
        l4 : dict or None
            EvidenceGrounder output.
        l5 : dict or None
            HypothesisEvaluator output.
        dag_context : dict or None
            Original dag_context from Layer 2.

        Returns
        -------
        dict  — see module docstring for schema.
        """
        # If any upstream layer is missing, escalate immediately.
        if l5 is None:
            return self._escalate_result(query, reason="Layer 5 output missing")

        verdict    = l5.get("verdict", "UNCERTAIN")
        confidence = l5.get("confidence", 0.0)
        support    = l5.get("support_score", 0.0)
        refutation = l5.get("refutation_score", 0.0)
        dst_frame  = l5.get("dst_frame")
        domain     = (l4 or {}).get("domain", "unknown")
        n_grounded = (l4 or {}).get("n_grounded", 0)
        n_preds    = (l3 or {}).get("all_predicates", []) or []

        # Map verdict → conclusion label
        conclusion = self._map_conclusion(verdict, confidence)

        # Build human-readable summary
        reasoning_summary = self._build_summary(
            query, verdict, conclusion, confidence,
            support, refutation, n_grounded, len(n_preds),
            dst_frame, dag_context,
        )

        logger.info(
            "ReasoningSynthesizer: query=%r verdict=%s conclusion=%s conf=%.3f",
            query[:60], verdict, conclusion, confidence,
        )

        # Cache write
        cached = False
        if conclusion != CONCL_ESCALATE and self._store is not None:
            cached = self._write_cache(
                query, conclusion, confidence, domain, dag_context
            )

        return {
            "conclusion":        conclusion,
            "confidence":        confidence,
            "verdict":           verdict,
            "support_score":     support,
            "refutation_score":  refutation,
            "reasoning_summary": reasoning_summary,
            "cached":            cached,
            "domain":            domain,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _map_conclusion(self, verdict: str, confidence: float) -> str:
        if confidence < self._floor:
            return CONCL_ESCALATE
        if verdict == "SUPPORTED":
            return (
                CONCL_CONFIDENT_SUPPORT
                if confidence >= self._strong
                else CONCL_WEAK_SUPPORT
            )
        if verdict == "REFUTED":
            return (
                CONCL_CONFIDENT_REFUTATION
                if confidence >= self._strong
                else CONCL_WEAK_REFUTATION
            )
        if verdict == "CONTRADICTORY":
            return CONCL_CONTRADICTORY
        # INCONCLUSIVE | UNCERTAIN
        return CONCL_INCONCLUSIVE

    def _build_summary(
        self,
        query: str,
        verdict: str,
        conclusion: str,
        confidence: float,
        support: float,
        refutation: float,
        n_grounded: int,
        n_total_preds: int,
        dst_frame: Optional[Dict],
        dag_context: Optional[Dict],
    ) -> str:
        lines = [
            f"Query: {query}",
            f"Verdict: {verdict}  |  Conclusion: {conclusion}  |  Confidence: {confidence:.3f}",
            f"Evidence: {n_grounded}/{n_total_preds} predicates grounded.",
            f"Support score: {support:.3f}  |  Refutation score: {refutation:.3f}",
        ]
        if dst_frame:
            lines.append(
                f"DST frame: m_true={dst_frame['m_true']:.3f}  "
                f"m_false={dst_frame['m_false']:.3f}  "
                f"m_unknown={dst_frame['m_unknown']:.3f}  "
                f"m_conflict={dst_frame['m_conflict']:.3f}"
            )
        if dag_context:
            cs = dag_context.get("contradiction_signal")
            if cs:
                lines.append(
                    f"Internal contradiction detected: type={cs['type']}  "
                    f"severity={cs['severity']:.3f}"
                )
            lines.append(
                f"Predicate family: {dag_context.get('predicate_family', 'UNKNOWN')}  "
                f"Sub-claims: {len(dag_context.get('sub_claims', []))}"
            )
        lines.append(
            "Epistemic note: INCONCLUSIVE means insufficient evidence, "
            "NOT falsity."
        )
        return "\n".join(lines)

    def _write_cache(
        self,
        query: str,
        conclusion: str,
        confidence: float,
        domain: str,
        dag_context: Optional[Dict],
    ) -> bool:
        """
        Write the conclusion as a minimal IRGraph to GraphStore.
        The graph contains a single CLAIM node labelled with the conclusion.
        Future EvidenceGrounder calls will find this node via Tier 1
        matching and receive a small evidence_score boost.
        """
        if not _IR_AVAILABLE:
            return False
        try:
            import datetime, hashlib, uuid

            predicate_family = (
                dag_context.get("predicate_family", "UNKNOWN")
                if dag_context else "UNKNOWN"
            )
            label = (
                f"{conclusion}::{query[:80]}::"
                f"{predicate_family}::conf{confidence:.2f}"
            )
            canonical = label.lower()
            sem_hash = hashlib.sha256(canonical.encode()).hexdigest()

            sig = SemanticSignature(
                semantic_hash=sem_hash,
                embedding_signature=[],
                spectral_signature=[],
                predicate_family=predicate_family,
                abstraction_level=0,
                canonical_form=canonical,
                equivalence_family=[query],
            )
            node = IRNode(
                id=f"synth-{sem_hash[:12]}",
                type="CLAIM",
                label=label,
                semantic_signature=sig,
                confidence_state=ConfidenceState(overall_confidence=confidence),
                temporal_state=TemporalState(),
                provenance=ProvenanceChain(
                    sources=["reasoning_synthesizer"],
                    reasoning_paths=["layer6_synthesis"],
                    decomposition_origin="synthesis_cache",
                    evidence_nodes=[],
                    ontology_resolution_path=[],
                    worker_threads=["main"],
                    timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
                ),
            )

            fp = GraphFingerprint(
                semantic_hash=sem_hash,
                structural_hash=sem_hash,
                predicate_family_hash=hashlib.sha256(
                    predicate_family.encode()
                ).hexdigest(),
                temporal_signature="",
                ontology_signature="",
                canonicalization_version="v1.0",
            )

            now = datetime.datetime.now(datetime.UTC).isoformat()
            graph = IRGraph(
                graph_id=f"synth-graph-{uuid.uuid4().hex[:8]}",
                nodes=[node],
                edges=[],
                fingerprint=fp,
                state="ACTIVE",
                version="v1",
                confidence_state=ConfidenceState(overall_confidence=confidence),
                ontology_version="1.0",
                created_at=now,
                updated_at=now,
                parent_graph_id=None,
            )

            self._store.put(graph)
            logger.debug(
                "ReasoningSynthesizer: cached graph %s (conclusion=%s)",
                graph.graph_id, conclusion,
            )
            return True

        except Exception as exc:
            logger.warning(
                "ReasoningSynthesizer: cache write failed (%s)", exc
            )
            return False

    @staticmethod
    def _escalate_result(query: str, reason: str) -> Dict:
        return {
            "conclusion":        CONCL_ESCALATE,
            "confidence":        0.0,
            "verdict":           "UNCERTAIN",
            "support_score":     0.0,
            "refutation_score":  0.0,
            "reasoning_summary": (
                f"Query: {query}\n"
                f"ESCALATE: {reason}\n"
                "Routing handed back to MultiLensRouter without reasoning bias."
            ),
            "cached": False,
            "domain": "unknown",
        }
