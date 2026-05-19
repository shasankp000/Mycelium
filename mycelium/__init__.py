"""Mycelium — Semantic Reasoning Substrate with Epistemic Stabilization.

Phase A: mycelium.ir                — IR primitives, graph, serialization
Phase B: mycelium.canonicalization  — SRL, canonical form, semantic hash pipeline
Phase C: mycelium.router            — SemanticRouter (MultiLensRouter + IR bridge)
Phase D: mycelium.trm               — TRMEngine (Graph Store, DFS Lookup, Promotion)
Phase E: mycelium.contradiction     — ContradictionClassifier, LeveragePropagator, PhaseEPipeline
Phase F: mycelium.fusion            — DST Fusion, OntologyGovernor, EdgeBuilder, ProvenanceBuilder
"""

from mycelium.router import IRBridge, SemanticRouter
from mycelium.trm import TRMEngine, TRMLookupResult
from mycelium.contradiction import (
    PhaseEPipeline,
    ClassificationResult,
    ContradictionReport,
)
from mycelium.fusion import DSTFusion, DSTFrame, ConfidenceStateFusion
from mycelium.ir.ontology_governor import OntologyGovernor
from mycelium.ir.edge_builder import EdgeBuilder
from mycelium.ir.provenance import ProvenanceBuilder

__all__ = [
    # Phase C
    "SemanticRouter",
    "IRBridge",
    # Phase D
    "TRMEngine",
    "TRMLookupResult",
    # Phase E
    "PhaseEPipeline",
    "ClassificationResult",
    "ContradictionReport",
    # Phase F
    "DSTFusion",
    "DSTFrame",
    "ConfidenceStateFusion",
    "OntologyGovernor",
    "EdgeBuilder",
    "ProvenanceBuilder",
]
