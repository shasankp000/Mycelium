"""
mycelium.reasoning
==================
Phase D reasoning components.

Exports
-------
DAGDecomposer   — DFS claim decomposition with cycle detection and DAG reuse.
"""

from mycelium.reasoning.dag_decomposer import DAGDecomposer

__all__ = ["DAGDecomposer"]
