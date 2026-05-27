# mycelium/pipeline/predicates/predicate_relations.py
# Future-compatible graph linkage primitives for the predicate engine.
#
# Defines typed directed edges between PredicateFrames.
# The relation graph is populated during extraction and negation but
# cross-predicate inference (e.g. transitivity, theorem proving) remains
# OUT OF SCOPE for v0.2. This module exists so later systems can integrate
# without redesigning PredicateFrame.
#
# Spec ref: implementation spec v0.2.1 — Section 4.2

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PredicateRelationType(str, Enum):
    """Typed directed relation between two PredicateFrames.

    Edges are stored in PredicateStore and accessible via
    PredicateStore.related() and PredicateStore.add_relation().

    Notes
    -----
    - SUPPORTS / CONTRADICTS: for evidence and contradiction propagation.
    - PRESUPPOSES: source frame takes target frame as a hidden assumption.
    - SPECIALIZES / GENERALIZES: scoping relations (source is a narrower/
      broader claim than target).
    - DEPENDS_ON: source frame's validity depends on target being true.
    - TEMPORALLY_PRECEDES: source event precedes target event in time.
    """

    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    PRESUPPOSES = "PRESUPPOSES"
    SPECIALIZES = "SPECIALIZES"
    GENERALIZES = "GENERALIZES"
    DEPENDS_ON = "DEPENDS_ON"
    TEMPORALLY_PRECEDES = "TEMPORALLY_PRECEDES"


@dataclass(frozen=True)
class PredicateGraphRelation:
    """An immutable directed edge in the predicate relation graph.

    Frozen so it can be stored in sets and used as a dict key.
    Create via PredicateStore.add_relation() rather than directly.

    Parameters
    ----------
    source_predicate_id:
        predicate_id of the originating frame.
    target_predicate_id:
        predicate_id of the destination frame.
    relation_type:
        Semantic type of the directed edge.
    confidence:
        Float in [0, 1] representing how certain the system is about
        this relation. Set by the component that adds the edge.
    """

    source_predicate_id: str
    target_predicate_id: str
    relation_type: PredicateRelationType
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"PredicateGraphRelation confidence must be in [0, 1], "
                f"got {self.confidence}"
            )
        if self.source_predicate_id == self.target_predicate_id:
            raise ValueError(
                "PredicateGraphRelation source and target must differ; "
                f"both are '{self.source_predicate_id}'"
            )
