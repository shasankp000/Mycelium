"""Mycelium Formal Intermediate Representation (IR) Layer — Phase A."""

from .primitives import (
    SemanticSignature,
    ConfidenceState,
    TemporalState,
    ProvenanceChain,
    GraphFingerprint,
)
from .graph import (
    NODE_TYPES,
    RELATION_FAMILIES,
    GRAPH_STATES,
    IRNode,
    IREdge,
    IRGraph,
)
from .leverage import LeverageEdge
from .contradiction import CONTRADICTION_CLASSES, ContradictionEdge
from .serialization import (
    ir_to_json,
    ir_from_json,
    compute_semantic_hash,
)

__all__ = [
    # Primitives
    "SemanticSignature",
    "ConfidenceState",
    "TemporalState",
    "ProvenanceChain",
    "GraphFingerprint",
    # Graph
    "NODE_TYPES",
    "RELATION_FAMILIES",
    "GRAPH_STATES",
    "IRNode",
    "IREdge",
    "IRGraph",
    # Leverage
    "LeverageEdge",
    # Contradiction
    "CONTRADICTION_CLASSES",
    "ContradictionEdge",
    # Serialization
    "ir_to_json",
    "ir_from_json",
    "compute_semantic_hash",
]
