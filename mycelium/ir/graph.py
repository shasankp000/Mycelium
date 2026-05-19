"""
Phase A — IRNode, IREdge, IRGraph.

Design alignment notes
----------------------
- NODE_TYPES includes REASONING_MOTIF (§17: decomposition attractors
  that TRM promotes to canonical reusable motifs).
- GRAPH_STATES maps exactly to §33 definitions.
- IRGraph.parent_graph_id enables the Git-style semantic branching
  model described in §39-41 (branch_graph in Phase D uses this).
- Dynamic argument structure (§3): IREdge does not use fixed-arity
  predicates; instead, relation_type is a string from RELATION_FAMILIES
  and additional argument roles live in the metadata dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .primitives import (
    SemanticSignature,
    ConfidenceState,
    TemporalState,
    ProvenanceChain,
    GraphFingerprint,
)


NODE_TYPES = [
    "ENTITY",
    "CLAIM",
    "PREDICATE",
    "RELATION",
    "EVIDENCE",
    "HYPOTHESIS",
    "TEMPORAL_RELATION",
    "CONTRADICTION",
    "ABSTRACTION",
    "REASONING_MOTIF",   # §17: TRM-promoted canonical decomposition attractor
]

RELATION_FAMILIES = [
    "SUPPORTS",
    "CONTRADICTS",
    "CAUSES",
    "CORRELATES_WITH",
    "IMPLIES",
    "PRECEDES",
    "PART_OF",
    "SUBTYPE_OF",
    "INSTANCE_OF",
    "DEPENDS_ON",
    "DERIVED_FROM",
]

# §33 — Graph lifecycle states.  Definitions:
#   DRAFT       — fresh decomposition, not yet verified
#   CANDIDATE   — seen independently ≥3 times (TRM threshold)
#   STABILIZED  — reasoning convergence reached (≥10 observations)
#   CANONICAL   — heavily reinforced + evidenced (≥25 observations)
#   CONTESTED   — strong contradiction exists; NOT immediately deprecated
#   DEPRECATED  — superseded by fresher evidence after TRM confirmation
#   ARCHIVED    — preserved historically but inactive
GRAPH_STATES = [
    "DRAFT",
    "CANDIDATE",
    "STABILIZED",
    "CANONICAL",
    "CONTESTED",
    "DEPRECATED",
    "ARCHIVED",
]


@dataclass
class IRNode:
    """Semantic node in the IR graph.

    Can represent: entities, predicates, evidence, hypotheses, temporal
    relations, contradictions, abstractions, or reasoning motifs (§25).

    The dynamic argument structure (§3) is realised through metadata:
    for multi-role predicates (giver/receiver/object), roles are stored
    as metadata["roles"] = {"giver": …, "receiver": …, "object": …}
    rather than forcing fixed-arity.
    """

    id: str
    type: str                         # from NODE_TYPES
    label: str
    semantic_signature: SemanticSignature
    confidence_state: ConfidenceState
    temporal_state: TemporalState
    provenance: ProvenanceChain
    metadata: dict                    = field(default_factory=dict)
    ontology_version: str             = "v1.0"
    state: str                        = "DRAFT"


@dataclass
class IREdge:
    """Directed semantic relation between two IRNodes.

    relation_type must be a value from RELATION_FAMILIES (or a predicate
    family from PREDICATE_FAMILY_MAP in Phase B).  Additional role
    information lives in metadata to support dynamic arity (§3).
    """

    id: str
    source: str                       # IRNode.id
    target: str                       # IRNode.id
    relation_type: str                # from RELATION_FAMILIES
    confidence_state: ConfidenceState
    temporal_state: TemporalState
    weight: float                     = 1.0
    metadata: dict                    = field(default_factory=dict)


@dataclass
class IRGraph:
    """Stabilised reasoning structure — the unit of TRM operations.

    Immutability contract (§38 / D.2):
        Graphs are NEVER mutated after creation.  Use GraphStore.add_revision()
        to create a new versioned copy.  Version naming: G123-v1 → G123-v2.

    parent_graph_id (§39): enables Git-style semantic branching where
    competing hypotheses diverge from a common parent rather than
    overwriting each other.  branch_graph() in Phase D populates this.
    """

    graph_id: str
    nodes: list                       # list[IRNode]
    edges: list                       # list[IREdge]
    fingerprint: GraphFingerprint
    state: str                        = "DRAFT"
    version: str                      = "v1"
    confidence_state: ConfidenceState = field(
        default_factory=lambda: ConfidenceState(overall_confidence=0.0)
    )
    ontology_version: str             = "v1.0"
    created_at: str                   = ""
    updated_at: str                   = ""
    parent_graph_id: Optional[str]    = None   # §39: semantic branching
