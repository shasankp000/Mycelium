"""Mycelium — Semantic Reasoning Substrate with Epistemic Stabilization.

Phase A: mycelium.ir                — IR primitives, graph, serialization
Phase B: mycelium.canonicalization  — SRL, canonical form, semantic hash pipeline
Phase C: mycelium.router            — SemanticRouter (MultiLensRouter + IR bridge)
Phase D: mycelium.trm               — TRMEngine (Graph Store, DFS Lookup, Promotion)
"""

from mycelium.router import IRBridge, SemanticRouter
from mycelium.trm import TRMEngine, TRMLookupResult

__all__ = ["SemanticRouter", "IRBridge", "TRMEngine", "TRMLookupResult"]
