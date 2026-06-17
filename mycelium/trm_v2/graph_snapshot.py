"""
mycelium/trm_v2/graph_snapshot.py

Versioned JSON snapshots for DomainGraph.

Goals:
    - serialize domain topology for the web UI
    - preserve graph_version / schema_version
    - support full snapshots and small incremental diffs
    - remain stdlib-only

Snapshot format:
    {
      "graph_version": "...",
      "schema_version": "...",
      "timestamp": ...,
      "nodes": {...},
      "edges": {...}
    }
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

from mycelium.trm_v2.types import DomainGraph, DomainNode, DriftProfile, DomainState, DomainMode


def _encode(obj: Any) -> Any:
    if is_dataclass(obj):
        out = asdict(obj)
        out["__class__"] = obj.__class__.__name__
        return out
    if isinstance(obj, (DomainState, DomainMode)):
        return obj.value
    if isinstance(obj, set):
        return sorted(obj)
    return obj


class GraphSnapshotWriter:
    """Serialize DomainGraph into JSON snapshots."""

    def __init__(self, graph: DomainGraph) -> None:
        self.graph = graph
        self._last_snapshot: Optional[Dict[str, Any]] = None

    def snapshot(self) -> Dict[str, Any]:
        nodes = {}
        for did, node in self.graph.nodes.items():
            nodes[did] = {
                "domain_id": node.domain_id,
                "name": node.name,
                "description": node.description,
                "domain_version": node.domain_version,
                "schema_version": node.schema_version,
                "head_version": node.head_version,
                "retrieval_version": node.retrieval_version,
                "state": node.state.value,
                "mode": node.mode.value,
                "semantic_centroid": node.semantic_centroid,
                "spectral_signature": node.spectral_signature,
                "head_ref": node.head_ref,
                "cold_storage_ref": node.cold_storage_ref,
                "sqlite_shard_ref": node.sqlite_shard_ref,
                "neighbors": node.neighbors,
                "parent_domains": node.parent_domains,
                "derived_from": node.derived_from,
                "creation_reason": node.creation_reason,
                "created_by": node.created_by,
                "creation_timestamp": node.creation_timestamp,
                "activation_stats": node.activation_stats,
                "drift_profile": asdict(node.drift_profile),
                "lineage": node.lineage,
            }

        data = {
            "graph_version": self.graph.graph_version,
            "schema_version": self.graph.schema_version,
            "timestamp": time.time(),
            "creation_timestamp": self.graph.creation_timestamp,
            "last_stabilized": self.graph.last_stabilized,
            "nodes": nodes,
            "edges": self.graph.edges,
        }
        self._last_snapshot = data
        return data

    def snapshot_json(self, indent: int = 2) -> str:
        return json.dumps(self.snapshot(), default=_encode, indent=indent)

    def diff(self) -> Dict[str, Any]:
        """Return a lightweight diff against the previous snapshot."""
        current = self.snapshot()
        if self._last_snapshot is None:
            return {"full": current}

        prev = self._last_snapshot
        added = [k for k in current["nodes"] if k not in prev["nodes"]]
        removed = [k for k in prev["nodes"] if k not in current["nodes"]]
        changed = {}
        for k in current["nodes"]:
            if k in prev["nodes"] and current["nodes"][k] != prev["nodes"][k]:
                changed[k] = current["nodes"][k]

        return {
            "graph_version": current["graph_version"],
            "schema_version": current["schema_version"],
            "timestamp": current["timestamp"],
            "added": added,
            "removed": removed,
            "changed": changed,
        }
