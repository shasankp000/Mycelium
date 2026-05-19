"""Mycelium — Semantic Reasoning Substrate with Epistemic Stabilization.

Phase A: mycelium.ir          — IR primitives, graph, serialization
Phase B: mycelium.canonicalization — SRL, canonical form, semantic hash pipeline
Phase C: mycelium.router      — SemanticRouter (MultiLensRouter + IR bridge)
"""

from mycelium.router import IRBridge, SemanticRouter

__all__ = ["SemanticRouter", "IRBridge"]
