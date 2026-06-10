"""
mycelium/pipeline/phase2/contradiction_integrator.py
=====================================================
Phase D Step 11 — Contradiction integration for PredicateFrame pairs.

Spec ref: implementation spec v0.2.1 — Phase D Step 11.

Purpose
-------
The ContradictionClassifier (Phase E) operates on graph IRNodes.
PredicateFrames are semantic IR objects, not graph nodes.

This module bridges the two by:
  1. Wrapping PredicateFrame pairs into lightweight stub IRNodes that
     satisfy ContradictionClassifier's duck-type interface.
  2. Running classify() on each (original, negated_form) pair.
  3. Populating contradiction_trace and stabilization_notes on each
     ScoredBundle in-place, so EvidenceDSTAdapter and DSTFusion have
     structured provenance to carry forward.

Design rules
------------
- NO new NLP stack, NO new embedding calls (Stage 4 in the classifier
  is skipped gracefully when embedding_fn=None).
- NO graph store access, NO SQLite, NO LLM calls.
- DETERMINISTIC: same PredicateFrame pair always produces the same
  contradiction classification.
- IN-PLACE mutation of ScoredBundle attributes only. No new dataclasses
  added to existing modules.
- FORWARD-COMPAT: ScoredBundle fields contradiction_trace and
  stabilization_notes are patched in if absent (the scorer predates
  these spec v0.2.1 additions).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from mycelium.contradiction.classifier import ContradictionClassifier
from mycelium.pipeline.predicates.predicate_types import PredicateFrame

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# Stub types — duck-type IRNode interface for ContradictionClassifier
# ---------------------------------------------------------------------------

class _SemanticSignatureStub:
    """Minimal stub satisfying ContradictionClassifier's sig interface.

    ContradictionClassifier reads:
        sig.predicate_family    (str)   — Stage 2, 3, 5, 6
        sig.canonical_form      (str)   — Stage 6  (FAMILY::subj::pred::obj::depthN)
        sig.embedding_signature (list)  — Stage 4  (can be [] to skip)
    """

    def __init__(self, frame: PredicateFrame) -> None:
        self.predicate_family: str = (
            frame.predicate_type.value
            if hasattr(frame.predicate_type, "value")
            else str(frame.predicate_type)
        )
        subject_text: str = (
            frame.subject.text
            if hasattr(frame.subject, "text")
            else str(frame.subject)
        )
        relation_lemma: str = (
            frame.relation.lemma
            if hasattr(frame.relation, "lemma")
            else str(frame.relation)
        )
        object_text: str = (
            frame.object.text
            if hasattr(frame.object, "text")
            else str(frame.object)
        )
        # canonical_form: FAMILY::subject::relation::object::depth0
        self.canonical_form: str = (
            f"{self.predicate_family}::{subject_text}::"
            f"{relation_lemma}::{object_text}::depth0"
        )
        # Empty embedding — Stage 4 skipped gracefully when embedding_fn=None
        self.embedding_signature: List[float] = []


class _TemporalStateStub:
    """Minimal stub satisfying ContradictionClassifier's temporal interface.

    ContradictionClassifier Stage 1 reads:
        ts.type               (str)  — "UNKNOWN" skips temporal gate
        ts.historical_validity (bool | None)
        ts.end, ts.start      (optional)
    """

    def __init__(self, frame: PredicateFrame) -> None:
        self.type: str = "UNKNOWN"      # disables temporal gate (Stage 1)
        # A negated frame is treated as a historical claim (no longer active)
        self.historical_validity: bool = bool(frame.negated)
        self.end: Optional[Any] = None
        self.start: Optional[Any] = None


class _ConfidenceStateStub:
    """Minimal stub satisfying ContradictionClassifier's confidence interface.

    ContradictionClassifier Stage 7 reads:
        cs.overall_confidence (float)
    """

    def __init__(self, frame: PredicateFrame) -> None:
        self.overall_confidence: float = float(
            getattr(frame, "provenance_confidence", 1.0) or 1.0
        )


class PredicateFrameNode:
    """Lightweight IRNode adapter for a PredicateFrame.

    Wraps a single PredicateFrame and exposes the attributes that
    ContradictionClassifier.classify() requires via duck typing:
        .label             (str)
        .semantic_signature (_SemanticSignatureStub)
        .temporal_state    (_TemporalStateStub)
        .confidence_state  (_ConfidenceStateStub)

    This object is ephemeral — created per classify() call, never stored.
    """

    def __init__(self, frame: PredicateFrame) -> None:
        self.label: str = frame.surface_form or ""
        self.semantic_signature = _SemanticSignatureStub(frame)
        self.temporal_state = _TemporalStateStub(frame)
        self.confidence_state = _ConfidenceStateStub(frame)
        self._frame = frame  # kept for predicate_id access


# ---------------------------------------------------------------------------
# Contradiction Integrator
# ---------------------------------------------------------------------------

class ContradictionIntegrator:
    """Populate contradiction_trace and stabilization_notes on ScoredBundles.

    For every ScoredBundle whose PredicateFrame has a negated_form, this
    class runs ContradictionClassifier.classify() on the (original, negated)
    pair and records the result in-place on the bundle.

    Usage
    -----
    integrator = ContradictionIntegrator()
    integrator.integrate(scored_evidence)         # mutates bundles in-place
    integrator.integrate(scored_evidence, dst_result)  # also annotates DST
    """

    # contradiction_types that indicate meaningful semantic conflict
    _SIGNIFICANT_TYPES = frozenset({
        "DIRECT_CONTRADICTION",
        "PARTIAL_CONTRADICTION",
        "TEMPORAL_CONTRADICTION",
        "CONTEXTUAL_CONTRADICTION",
        "PROBABILISTIC_DISAGREEMENT",
        "TRADEOFF_RELATION",
        # NON_CONTRADICTORY_DIVERGENCE is deliberately excluded
    })

    def __init__(self) -> None:
        # embedding_fn=None: Stage 4 skipped gracefully (no extra model)
        self._classifier = ContradictionClassifier(embedding_fn=None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def integrate(
        self,
        scored_evidence: Any,
        dst_result: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Classify predicate pairs and populate contradiction provenance.

        Parameters
        ----------
        scored_evidence:
            ScoredEvidenceResult from EvidenceScorer.score().
        dst_result:
            Optional EvidenceDSTResult from EvidenceDSTAdapter.fuse().
            When provided, contradiction_trace entries are also appended
            to dst_result.contradiction_trace (aggregated view).

        Returns
        -------
        dict
            Summary for SSE event emission:
            {
              "classified": int,      # bundles where classify() ran
              "contradictions": int,  # bundles with significant type
              "skipped": int,         # bundles with no negated_form
              "type_counts": dict,    # contradiction_type → count
              "severity_max": float,  # max severity seen
              "severity_mean": float, # mean severity across contradictions
            }
        """
        scored_bundles: List[Any] = list(
            getattr(scored_evidence, "scored_bundles", []) or []
        )

        classified = 0
        contradictions = 0
        skipped = 0
        type_counts: Dict[str, int] = {}
        severities: List[float] = []

        for bundle in scored_bundles:
            # Ensure forward-compat fields exist on ScoredBundle.
            # Use setattr() so this works on both dataclass instances and
            # plain SimpleNamespace objects (object.__setattr__ raises
            # AttributeError on SimpleNamespace which has no __slots__).
            if not hasattr(bundle, "contradiction_trace"):
                setattr(bundle, "contradiction_trace", [])
            if not hasattr(bundle, "stabilization_notes"):
                setattr(bundle, "stabilization_notes", [])
            # Also ensure lists are mutable (not frozen)
            if bundle.contradiction_trace is None:
                bundle.contradiction_trace = []
            if bundle.stabilization_notes is None:
                bundle.stabilization_notes = []

            # Retrieve PredicateFrame from the nested EvidenceBundle
            raw_bundle = getattr(bundle, "bundle", None)
            frame: Optional[PredicateFrame] = (
                getattr(raw_bundle, "frame", None) if raw_bundle else None
            )
            if frame is None:
                skipped += 1
                continue

            negated_form: Optional[PredicateFrame] = getattr(
                frame, "negated_form", None
            )
            if negated_form is None:
                skipped += 1
                continue

            predicate_id: str = str(
                getattr(frame, "predicate_id", "unknown")
            )

            # Build stub IRNodes
            try:
                original_node = PredicateFrameNode(frame)
                negated_node = PredicateFrameNode(negated_form)
            except Exception as _node_err:
                logger.debug(
                    "ContradictionIntegrator: failed to build node for %s: %s",
                    predicate_id, _node_err,
                )
                skipped += 1
                continue

            # Classify
            try:
                result = self._classifier.classify(original_node, negated_node)
                classified += 1
            except Exception as _clf_err:
                logger.debug(
                    "ContradictionIntegrator: classify() failed for %s: %s",
                    predicate_id, _clf_err,
                )
                skipped += 1
                continue

            ctype: str = result.contradiction_type
            type_counts[ctype] = type_counts.get(ctype, 0) + 1

            if ctype not in self._SIGNIFICANT_TYPES:
                # NON_CONTRADICTORY_DIVERGENCE: nothing to record
                continue

            contradictions += 1
            severities.append(result.severity)

            # --- Populate contradiction_trace ---
            trace_entry = (
                f"[{predicate_id}] {ctype} "
                f"sev={result.severity:.3f} conf={result.confidence:.3f} "
                f"scope={result.scope} stage={result.stage_reached}: "
                f"{result.explanation}"
            )
            bundle.contradiction_trace.append(trace_entry)

            # --- Populate stabilization_notes ---
            stab_note = (
                f"Contradiction ({ctype}, sev={result.severity:.3f}) detected "
                f"for predicate {predicate_id}. "
                f"Downstream stabilization should treat this bundle's "
                f"net_confidence as an upper bound only."
            )
            bundle.stabilization_notes.append(stab_note)

            logger.debug(
                "ContradictionIntegrator [%s]: %s sev=%.3f stage=%d",
                predicate_id, ctype, result.severity, result.stage_reached,
            )

            # --- Propagate to DST result if provided ---
            if dst_result is not None:
                _dst_trace = getattr(dst_result, "contradiction_trace", None)
                if _dst_trace is not None:
                    _dst_trace.append(trace_entry)
                _dst_notes = getattr(dst_result, "stabilization_notes", None)
                if _dst_notes is not None:
                    _dst_notes.append(stab_note)

        severity_max = max(severities) if severities else 0.0
        severity_mean = (
            sum(severities) / len(severities) if severities else 0.0
        )

        return {
            "classified": classified,
            "contradictions": contradictions,
            "skipped": skipped,
            "type_counts": type_counts,
            "severity_max": round(severity_max, 4),
            "severity_mean": round(severity_mean, 4),
        }


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_integrator_instance: Optional[ContradictionIntegrator] = None


def get_contradiction_integrator() -> ContradictionIntegrator:
    """Return the process-level singleton ContradictionIntegrator.

    Stateless; singleton avoids re-constructing ContradictionClassifier
    on every sentence.
    """
    global _integrator_instance
    if _integrator_instance is None:
        _integrator_instance = ContradictionIntegrator()
    return _integrator_instance
