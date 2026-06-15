from __future__ import annotations
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

from mycelium.domain_graph.models import DomainNode, DomainEdge

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"


def _graph_payload(
    nodes: Dict[str, DomainNode],
    edges: List[DomainEdge],
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "nodes": {k: v.to_dict() for k, v in nodes.items()},
        "edges": [e.to_dict() for e in edges],
    }


def save_graph(
    path: str | os.PathLike,
    nodes: Dict[str, DomainNode],
    edges: List[DomainEdge],
    *,
    indent: int = 2,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    payload = _graph_payload(nodes, edges)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent)
    tmp.replace(p)
    logger.debug("Graph saved → %s  (%d nodes, %d edges)", p, len(nodes), len(edges))


def load_graph(
    path: str | os.PathLike,
) -> Tuple[Dict[str, DomainNode], List[DomainEdge], str]:
    """Returns (nodes, edges, schema_version)."""
    p = Path(path)
    if not p.exists():
        logger.info("No graph file at %s — starting empty.", p)
        return {}, [], SCHEMA_VERSION
    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)
    schema = raw.get("schema_version", "unknown")
    nodes = {k: DomainNode.from_dict(v) for k, v in raw.get("nodes", {}).items()}
    edges = [DomainEdge.from_dict(e) for e in raw.get("edges", [])]
    logger.debug("Graph loaded ← %s  (%d nodes, %d edges, schema=%s)", p, len(nodes), len(edges), schema)
    return nodes, edges, schema


def snapshot_graph(
    snapshot_dir: str | os.PathLike,
    nodes: Dict[str, DomainNode],
    edges: List[DomainEdge],
    tag: str = "",
) -> Path:
    """Write a timestamped snapshot alongside the live graph."""
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"graph_{ts}{'_' + tag if tag else ''}.json"
    dest = Path(snapshot_dir) / name
    save_graph(dest, nodes, edges)
    return dest
