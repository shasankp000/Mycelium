"""
patch_dag.py
────────────
Parts 2.1 – 2.4 of the Mycelium × Lexis integration plan.

Defines:
  PatchNode       – dataclass representing a single structured knowledge node
  PatchDAG        – in-memory directed acyclic graph of PatchNode objects with
                    JSON persistence, cosine k-NN nearest-node lookup, and
                    overlap-edge computation
  PatchCreation…  – thin async wrapper; full pipeline wired in a separate step

Storage layout (relative to project root):
  patches/
    <patch_id>.lexi          – Lexis compressed binary
    <patch_id>.centroid.npy  – 384-dim MiniLM centroid
    <patch_id>.pos.npy       – POS histogram fingerprint
    <patch_id>.entity.json   – entity graph adjacency dict
    <patch_id>.json          – patch node sidecar (all metadata)
  patch_dag.json             – serialised DAG adjacency list
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

_PATCHES_DIR  = Path("patches")
_DAG_FILE     = Path("patch_dag.json")


# ---------------------------------------------------------------------------
# PatchNode dataclass
# ---------------------------------------------------------------------------

@dataclass
class PatchNode:
    """
    A single structured knowledge node in the patch DAG.

    All array paths are *relative* to the project root so the patches/
    directory can be moved without invalidating the sidecar JSON.
    """
    patch_id:             str
    created_at:           str                           # ISO 8601
    domain_label:         str
    lexi_path:            str                           # relative path to .lexi binary
    centroid_path:        str                           # relative path to .npy
    pos_fingerprint_path: str                           # relative path to .npy
    entity_graph_path:    str                           # relative path to .json
    parent_domains:       List[str]           = field(default_factory=list)
    overlap_edges:        Dict[str, float]    = field(default_factory=dict)
    confidence_threshold: float               = 0.72
    novel_chunk_count:    int                 = 0
    source_document:      Optional[str]       = None

    # ------------------------------------------------------------------
    # Lazy-loaded numpy arrays (not serialised; reloaded from disk)
    # ------------------------------------------------------------------
    _centroid:        Optional[np.ndarray]  = field(default=None, repr=False, compare=False)
    _pos_fingerprint: Optional[np.ndarray]  = field(default=None, repr=False, compare=False)
    _entity_graph:    Optional[Dict]        = field(default=None, repr=False, compare=False)

    @property
    def centroid(self) -> np.ndarray:
        if self._centroid is None:
            self._centroid = np.load(self.centroid_path)
        return self._centroid

    @property
    def pos_fingerprint(self) -> np.ndarray:
        if self._pos_fingerprint is None:
            self._pos_fingerprint = np.load(self.pos_fingerprint_path)
        return self._pos_fingerprint

    @property
    def entity_graph(self) -> Dict:
        if self._entity_graph is None:
            with open(self.entity_graph_path, encoding="utf-8") as f:
                self._entity_graph = json.load(f)
        return self._entity_graph

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to plain dict (no numpy arrays)."""
        d = asdict(self)
        # Remove private lazy-load fields from serialisation
        for key in list(d.keys()):
            if key.startswith("_"):
                d.pop(key)
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PatchNode":
        """Deserialise from plain dict (as stored in sidecar JSON)."""
        # Strip any private keys that may have slipped in
        clean = {k: v for k, v in d.items() if not k.startswith("_")}
        return PatchNode(**clean)

    def save_sidecar(self) -> None:
        """Write the sidecar JSON for this node."""
        sidecar = Path(self.lexi_path).parent / f"{self.patch_id}.json"
        sidecar.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        log.debug("PatchNode sidecar saved: %s", sidecar)

    @staticmethod
    def load_sidecar(sidecar_path: str | Path) -> "PatchNode":
        """Load a PatchNode from its sidecar JSON file."""
        with open(sidecar_path, encoding="utf-8") as f:
            return PatchNode.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# PatchDAG
# ---------------------------------------------------------------------------

class PatchDAG:
    """
    In-memory directed acyclic graph of PatchNode objects.

    Provides:
      - nearest(query_embedding, threshold) -> PatchNode | None
      - insert(node)                         -> None
      - compute_overlap_edges(node)          -> Dict[str, float]
      - save() / load()                      for JSON persistence
    """

    def __init__(self, dag_path: str | Path = _DAG_FILE) -> None:
        self._dag_path: Path             = Path(dag_path)
        self._nodes:    Dict[str, PatchNode] = {}   # patch_id -> PatchNode
        # Adjacency list: patch_id -> list of (neighbour_id, edge_type, weight)
        self._edges:    Dict[str, List]  = {}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """
        Serialise DAG to JSON.  Skips nodes whose sidecar files are missing
        (e.g., partially-failed patch creation) to avoid corrupt state.
        """
        payload = {
            "nodes": {
                pid: node.to_dict()
                for pid, node in self._nodes.items()
                if Path(node.lexi_path).exists()
            },
            "edges": self._edges,
        }
        self._dag_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log.info("PatchDAG saved to %s (%d nodes)", self._dag_path, len(payload["nodes"]))

    def load(self) -> None:
        """
        Load DAG from JSON.  Missing sidecar files are skipped with a warning.
        Safe to call on an empty / non-existent file.
        """
        if not self._dag_path.exists():
            log.debug("No patch DAG file found at %s -- starting fresh.", self._dag_path)
            return
        with open(self._dag_path, encoding="utf-8") as f:
            payload = json.load(f)
        for pid, d in payload.get("nodes", {}).items():
            try:
                self._nodes[pid] = PatchNode.from_dict(d)
            except (TypeError, KeyError) as exc:
                log.warning("Skipping malformed node %s: %s", pid, exc)
        self._edges = payload.get("edges", {})
        log.info("PatchDAG loaded from %s (%d nodes)", self._dag_path, len(self._nodes))

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def insert(self, node: PatchNode) -> None:
        """
        Insert a PatchNode into the DAG and save the updated graph.

        Overlap edges to existing nodes are computed automatically when
        at least two nodes share entity graph vocabulary above 0.25 Jaccard.
        """
        self._nodes[node.patch_id] = node
        self._edges.setdefault(node.patch_id, [])
        # Compute and store overlap edges to all existing nodes
        overlap = self.compute_overlap_edges(node)
        for other_id, weight in overlap.items():
            # Bidirectional
            self._edges[node.patch_id].append((other_id, "overlap", weight))
            self._edges.setdefault(other_id, []).append((node.patch_id, "overlap", weight))
            # Also persist into node metadata
            node.overlap_edges[other_id] = weight
            self._nodes[other_id].overlap_edges[node.patch_id] = weight
        node.save_sidecar()
        self.save()
        log.info("Inserted patch node %s into DAG (overlap edges: %d)", node.patch_id, len(overlap))

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def nearest(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.72,
    ) -> Optional[PatchNode]:
        """
        Return the PatchNode whose centroid is most similar to *query_embedding*,
        provided the cosine similarity exceeds *threshold*.

        Returns None if no node meets the threshold (caller should fall back to
        CREATE_NEW_PATCH or domain-expert routing).
        """
        if not self._nodes:
            return None

        best_node:  Optional[PatchNode] = None
        best_score: float               = -1.0
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)

        for node in self._nodes.values():
            try:
                c_norm = node.centroid / (np.linalg.norm(node.centroid) + 1e-9)
                score  = float(np.dot(q_norm, c_norm))
            except Exception as exc:  # noqa: BLE001
                log.warning("Error computing similarity for node %s: %s", node.patch_id, exc)
                continue
            if score > best_score:
                best_score = score
                best_node  = node

        if best_score >= threshold:
            log.debug("Nearest patch node: %s (score=%.4f)", best_node.patch_id, best_score)
            return best_node

        log.debug("No patch node above threshold %.2f (best=%.4f)", threshold, best_score)
        return None

    # ------------------------------------------------------------------
    # Overlap edge computation
    # ------------------------------------------------------------------

    def compute_overlap_edges(
        self,
        node: PatchNode,
        similarity_threshold: float = 0.25,
    ) -> Dict[str, float]:
        """
        Compute Jaccard similarity between *node*'s entity graph vocabulary
        and all other nodes.  Returns {other_patch_id: jaccard_score} for
        pairs above *similarity_threshold*.

        Falls back gracefully if entity graphs cannot be loaded.
        """
        try:
            node_terms = set(node.entity_graph.keys())
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not load entity graph for %s: %s", node.patch_id, exc)
            return {}

        overlap: Dict[str, float] = {}
        for other_id, other_node in self._nodes.items():
            if other_id == node.patch_id:
                continue
            try:
                other_terms = set(other_node.entity_graph.keys())
            except Exception:  # noqa: BLE001
                continue
            if not node_terms or not other_terms:
                continue
            intersection = len(node_terms & other_terms)
            union        = len(node_terms | other_terms)
            jaccard      = intersection / union if union else 0.0
            if jaccard >= similarity_threshold:
                overlap[other_id] = round(jaccard, 4)

        return overlap

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, patch_id: str) -> bool:
        return patch_id in self._nodes

    def get_node(self, patch_id: str) -> Optional[PatchNode]:
        return self._nodes.get(patch_id)

    def all_nodes(self) -> List[PatchNode]:
        return list(self._nodes.values())


# ---------------------------------------------------------------------------
# Patch creation pipeline skeleton
# (wired to lexis_bridge + UnifiedExpertSystem in a subsequent commit)
# ---------------------------------------------------------------------------

async def create_patch_async(
    document: str,
    domain_label: str,
    dag: PatchDAG,
    patches_dir: Path = _PATCHES_DIR,
    compact: bool = True,
) -> Optional[PatchNode]:
    """
    Background async task: compress *document* with Lexis and insert a new
    PatchNode into *dag*.

    This is the entry point for Step 2 of the patch creation pipeline (§2.2).
    Embedding, POS fingerprinting, and DAG placement are wired here; the
    MiniLM encode call and POS histogram computation are implemented by
    callers that have access to the model registry.

    Parameters
    ----------
    document     : full text of the novel document (already OOD-filtered).
    domain_label : human-readable domain string (e.g., 'quantum_mechanics').
    dag          : the live PatchDAG instance to insert the new node into.
    patches_dir  : directory to store .lexi and sidecar files.
    compact      : passed through to lexi_compress().

    Returns
    -------
    The newly created PatchNode, or None if creation failed.

    Notes
    -----
    This function blocks the asyncio event loop only during file I/O.  The
    Lexis subprocess call is run in a thread-pool via asyncio.to_thread() to
    avoid blocking the main event loop.
    """
    from lexis_bridge import lexi_compress  # local import avoids circular at module load

    patches_dir.mkdir(parents=True, exist_ok=True)
    patch_id = f"{domain_label.replace(' ', '_')}-{uuid.uuid4().hex[:8]}"
    lexi_path = str(patches_dir / f"{patch_id}.lexi")

    # Step 2: Compress document via Lexis (CPU-bound subprocess → thread pool)
    try:
        await asyncio.to_thread(lexi_compress, document, lexi_path, compact)
    except Exception as exc:  # noqa: BLE001
        log.error("Patch creation failed during Lexis compress for %s: %s", patch_id, exc)
        return None

    # Placeholder arrays written now; callers with model-registry access
    # should replace these with real MiniLM / POS histogram outputs.
    centroid_path = str(patches_dir / f"{patch_id}.centroid.npy")
    pos_path      = str(patches_dir / f"{patch_id}.pos.npy")
    entity_path   = str(patches_dir / f"{patch_id}.entity.json")

    # Default empty placeholders (real values filled in by the caller or a
    # subsequent embedding pass triggered by PatchCreationPipeline).
    np.save(centroid_path, np.zeros(384, dtype=np.float32))
    np.save(pos_path,      np.zeros(18,  dtype=np.float32))  # 18 universal POS tags
    Path(entity_path).write_text(json.dumps({}), encoding="utf-8")

    node = PatchNode(
        patch_id=patch_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        domain_label=domain_label,
        lexi_path=lexi_path,
        centroid_path=centroid_path,
        pos_fingerprint_path=pos_path,
        entity_graph_path=entity_path,
        source_document=None,
    )

    dag.insert(node)
    log.info("Patch node created: %s", patch_id)
    return node
