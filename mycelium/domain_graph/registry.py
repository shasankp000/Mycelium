from __future__ import annotations
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional

from mycelium.domain_graph.models import DomainNode, DomainEdge
from mycelium.domain_graph.state import DomainState, GateState, DomainMode
from mycelium.domain_graph.persistence import load_graph, save_graph, snapshot_graph

logger = logging.getLogger(__name__)


class DomainGraphRegistry:
    """
    Thread-safe in-memory registry of all DomainNodes and DomainEdges.
    Backed by a JSON file at `graph_path`. The registry is the only
    allowed source of domain identity — no module may hardcode domain names.
    """

    def __init__(self, graph_path: str, *, auto_save: bool = True) -> None:
        self._path = Path(graph_path)
        self._auto_save = auto_save
        self._lock = threading.RLock()
        self._nodes: Dict[str, DomainNode] = {}
        self._edges: List[DomainEdge] = []
        self._schema_version: str = "1.0"
        self._load()

    # ------------------------------------------------------------------
    # Load / Save
    # ------------------------------------------------------------------

    def _load(self) -> None:
        nodes, edges, schema = load_graph(self._path)
        with self._lock:
            self._nodes = nodes
            self._edges = edges
            self._schema_version = schema

    def save(self) -> None:
        with self._lock:
            save_graph(self._path, self._nodes, self._edges)

    def snapshot(self, tag: str = "") -> Path:
        with self._lock:
            return snapshot_graph(self._path.parent / "snapshots", self._nodes, self._edges, tag=tag)

    def _maybe_save(self) -> None:
        if self._auto_save:
            self.save()

    # ------------------------------------------------------------------
    # Node CRUD
    # ------------------------------------------------------------------

    def register(self, node: DomainNode) -> None:
        with self._lock:
            if node.domain_id in self._nodes:
                raise ValueError(f"Domain '{node.domain_id}' already registered.")
            self._nodes[node.domain_id] = node
            logger.info("Registered domain '%s' (state=%s)", node.domain_id, node.state.value)
            self._maybe_save()

    def get(self, domain_id: str) -> Optional[DomainNode]:
        with self._lock:
            return self._nodes.get(domain_id)

    def require(self, domain_id: str) -> DomainNode:
        node = self.get(domain_id)
        if node is None:
            raise KeyError(f"Domain '{domain_id}' not found in registry.")
        return node

    def update(self, node: DomainNode) -> None:
        with self._lock:
            if node.domain_id not in self._nodes:
                raise KeyError(f"Domain '{node.domain_id}' not found — use register() first.")
            self._nodes[node.domain_id] = node
            self._maybe_save()

    def all_domains(self) -> List[DomainNode]:
        with self._lock:
            return list(self._nodes.values())

    def all_hot_domains(self) -> List[DomainNode]:
        """Domains that are HOT and gate=OPEN — the live routing set."""
        with self._lock:
            return [
                n for n in self._nodes.values()
                if n.state == DomainState.HOT and n.gate == GateState.OPEN
            ]

    def domain_ids(self) -> List[str]:
        with self._lock:
            return list(self._nodes.keys())

    # ------------------------------------------------------------------
    # State transitions (safe wrappers)
    # ------------------------------------------------------------------

    def set_state(self, domain_id: str, state: DomainState) -> None:
        with self._lock:
            node = self.require(domain_id)
            old = node.state
            node.state = state
            logger.info("Domain '%s': %s → %s", domain_id, old.value, state.value)
            self._maybe_save()

    def set_gate(self, domain_id: str, gate: GateState) -> None:
        with self._lock:
            node = self.require(domain_id)
            node.gate = gate
            logger.debug("Domain '%s' gate → %s", domain_id, gate.value)
            self._maybe_save()

    def set_mode(self, domain_id: str, mode: DomainMode) -> None:
        with self._lock:
            node = self.require(domain_id)
            node.mode = mode
            self._maybe_save()

    def mark_active(self, domain_id: str) -> None:
        with self._lock:
            node = self.require(domain_id)
            node.mark_active()
            self._maybe_save()

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_edge(self, edge: DomainEdge) -> None:
        with self._lock:
            self._edges.append(edge)
            self._maybe_save()

    def edges_for(self, domain_id: str) -> List[DomainEdge]:
        with self._lock:
            return [
                e for e in self._edges
                if e.source_id == domain_id or e.target_id == domain_id
            ]

    def all_edges(self) -> List[DomainEdge]:
        with self._lock:
            return list(self._edges)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, int]:
        with self._lock:
            counts: Dict[str, int] = {}
            for n in self._nodes.values():
                key = n.state.value
                counts[key] = counts.get(key, 0) + 1
            counts["total"] = len(self._nodes)
            counts["edges"] = len(self._edges)
            return counts
