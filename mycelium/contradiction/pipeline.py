"""
Phase E — Contradiction Pipeline
=================================
End-to-end orchestration of Steps 8-9 from §46:

    Step 8: Classify contradiction type between two IRNodes.
    Step 9: If severity >= TRM threshold, contest the affected graphs
            in the TRM and run leverage propagation on both.

PhaseEPipeline is the single entry-point callers use.  It wires together:
    - ContradictionClassifier (Phase E)
    - LeveragePropagator (Phase E)
    - TRMEngine (Phase D) — optional, for full end-to-end integration

ContradictionReport
-------------------
The pipeline returns a ContradictionReport containing the full audit trail
required by £29 (ProvenanceChain) and £44 (propagation halting evidence):
    - source_graph_id / target_graph_id
    - classification: ClassificationResult
    - trm_contested: bool  (whether Phase D was triggered)
    - propagation_results: list[PropagationResult]
    - timestamp: str

Design notes
------------
    Phase D integration is optional (trm_engine=None = dry-run mode).
    This allows Phase E to be unit-tested independently of Phase D.

    Severity threshold for TRM contestation reuses
    PromotionPolicy.THRESHOLDS["contradiction_severity"] so there is one
    source of truth for that constant.
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

from .classifier import ContradictionClassifier, ClassificationResult
from .propagation import LeveragePropagator, PropagationResult

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

try:
    from mycelium.trm.promotion import PromotionPolicy
    _TRM_THRESHOLD = PromotionPolicy.THRESHOLDS["contradiction_severity"]
except Exception:
    _TRM_THRESHOLD = 0.4  # fallback


@dataclass
class ContradictionReport:
    """Full audit record for a Phase E contradiction analysis run.

    Attributes
    ----------
    source_graph_id : str
        Graph that contains node_a.
    target_graph_id : str
        Graph that contains node_b.
    classification : ClassificationResult
        The output of ContradictionClassifier.classify().
    trm_contested : bool
        True if TRMEngine.record_contradiction() was called.
    propagation_results : list[PropagationResult]
        One entry per graph (source + target) after leverage propagation.
    timestamp : str
        ISO 8601 UTC timestamp of the run.
    """

    source_graph_id: str
    target_graph_id: str
    classification: ClassificationResult
    trm_contested: bool = False
    propagation_results: List[PropagationResult] = field(default_factory=list)
    timestamp: str = ""


class PhaseEPipeline:
    """End-to-end Phase E: classify → contest → propagate.

    Parameters
    ----------
    trm_engine : TRMEngine, optional
        Phase D engine.  If None, operates in dry-run mode (classify +
        propagate but no TRM state changes).
    embedding_fn : callable, optional
        Forwarded to ContradictionClassifier for Stage 4 embedding gate.
    graph_store : GraphStore, optional
        Forwarded to LeveragePropagator for persistence of decayed
        confidence values.  Typically the same store used by trm_engine.

    Usage
    -----
    pipeline = PhaseEPipeline(trm_engine=trm)
    report = pipeline.run(
        node_a, source_graph,
        node_b, target_graph,
    )
    """

    def __init__(
        self,
        *,
        trm_engine: Optional[Any] = None,
        embedding_fn: Optional[Any] = None,
        graph_store: Optional[Any] = None,
    ) -> None:
        self._trm = trm_engine
        self._clf = ContradictionClassifier(embedding_fn=embedding_fn)
        self._prop = LeveragePropagator(
            graph_store=graph_store or (
                trm_engine.store if trm_engine is not None else None
            ),
        )

    def run(
        self,
        node_a: Any,
        source_graph: Any,
        node_b: Any,
        target_graph: Any,
        *,
        leverage_edges_source: Optional[List[Any]] = None,
        leverage_edges_target: Optional[List[Any]] = None,
    ) -> ContradictionReport:
        """Run the full Phase E pipeline for a pair of claim nodes.

        Steps:
            8. Classify the contradiction between node_a and node_b.
            9a. If severity >= TRM threshold AND trm_engine is set:
                - Call TRMEngine.record_contradiction() on both graphs.
                - trm_contested = True.
            9b. Run LeveragePropagator on both graphs regardless of
                TRM contestation (dry-run propagation is always useful
                for the audit trail).

        Parameters
        ----------
        node_a : IRNode
            First claim node.
        source_graph : IRGraph
            Graph containing node_a.
        node_b : IRNode
            Second claim node.
        target_graph : IRGraph
            Graph containing node_b.
        leverage_edges_source, leverage_edges_target : list[LeverageEdge], optional
            Explicit leverage edges for each graph.  Passed through to
            LeveragePropagator (falls back to IRGraph.edges if None).

        Returns
        -------
        ContradictionReport
        """
        now = datetime.datetime.utcnow().isoformat() + "Z"

        # Step 8: classify
        classification = self._clf.classify(node_a, node_b)
        logger.debug(
            "PhaseEPipeline: classified %s vs %s → %s (severity=%.2f)",
            node_a.id, node_b.id,
            classification.contradiction_type,
            classification.severity,
        )

        trm_contested = False

        # Step 9a: TRM contestation if threshold met
        if (
            self._trm is not None
            and classification.severity >= _TRM_THRESHOLD
        ):
            try:
                # Build a ContradictionEdge-compatible object for TRMEngine
                ce = _ContradictionEdgeProxy(
                    type=classification.contradiction_type,
                    severity=classification.severity,
                    confidence=classification.confidence,
                    scope=classification.scope,
                    source_graph_id=source_graph.graph_id,
                    target_graph_id=target_graph.graph_id,
                )
                self._trm.record_contradiction(source_graph.graph_id, ce)
                self._trm.record_contradiction(target_graph.graph_id, ce)
                trm_contested = True
                logger.debug(
                    "PhaseEPipeline: TRM contested %s and %s",
                    source_graph.graph_id, target_graph.graph_id,
                )
            except Exception as exc:
                logger.warning("PhaseEPipeline: TRM contestation failed: %s", exc)

        # Step 9b: leverage propagation (both graphs, always)
        prop_results: List[PropagationResult] = []

        prop_a = self._prop.propagate(
            source_graph,
            node_a.id,
            initial_severity=classification.severity,
            leverage_edges=leverage_edges_source,
        )
        prop_results.append(prop_a)

        if target_graph.graph_id != source_graph.graph_id:
            prop_b = self._prop.propagate(
                target_graph,
                node_b.id,
                initial_severity=classification.severity,
                leverage_edges=leverage_edges_target,
            )
            prop_results.append(prop_b)

        return ContradictionReport(
            source_graph_id=source_graph.graph_id,
            target_graph_id=target_graph.graph_id,
            classification=classification,
            trm_contested=trm_contested,
            propagation_results=prop_results,
            timestamp=now,
        )


class _ContradictionEdgeProxy:
    """Lightweight stand-in for ContradictionEdge to avoid circular imports.

    TRMEngine.record_contradiction() only reads .severity, so this proxy
    satisfies the interface without importing mycelium.ir.contradiction
    (which would create a Phase A → Phase E → Phase D import cycle).
    """

    __slots__ = (
        "type", "severity", "confidence", "scope",
        "source_graph_id", "target_graph_id",
    )

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
