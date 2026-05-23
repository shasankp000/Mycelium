"""
Phase D — Graph Store
======================
Versioned, immutable in-memory store for IRGraph objects.

Design (Consolidation Notes §38-41):
    Immutability contract (§38):
        Stored graphs are NEVER mutated after their first write.
        Every structural change creates a new version via add_revision().
        Version naming: G{id}-v1 → G{id}-v2 → …

    Git-style branching (§39-41):
        branch_graph() forks an existing graph by creating a new IRGraph
        with the original set as parent_graph_id.  Competing hypotheses
        diverge from a common parent rather than overwriting it.

    Observation tracking:
        GraphStore maintains a per-graph_id integer observation_count.
        Each call to observe() increments it.  PromotionPolicy reads
        this counter to decide state transitions.

    Storage layout (in-memory, single process):
        _store: dict[graph_id, list[IRGraph]]   — version history
        _obs:   dict[graph_id, int]             — observation counts
        _hash_index: dict[semantic_hash, set[graph_id]]  — fast hash lookup

Persistence (Phase D / Consolidation Notes §20)
----------------------------------------------
    When persistence_path is set, GraphStore writes every graph to a
    newline-delimited JSON file (one line per graph version) and reloads
    all graphs on the next boot.  This gives the OOD fallback chain a
    warm Tier 1 evidence pool across restarts.

    Format:  <persistence_path>/graphstore.jsonl
        Each line is the JSON output of dataclasses.asdict(IRGraph).
        Append-only during a session; compacted on next boot (one pass
        that keeps only the latest version per graph_id).

    Decay pass:
        After loading, GraphStore runs GraphDecayManager.decay_pass()
        so stale / historically-invalid graphs are transitioned to
        DEPRECATED before EvidenceGrounder can query them.
        DEPRECATED graphs are excluded from list_all() by default.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import pathlib
from typing import Dict, List, Optional, Set

from mycelium.trm.graph_decay import GraphDecayManager

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

try:
    from mycelium.ir.graph import IRGraph
    from mycelium.ir.primitives import ConfidenceState, GraphFingerprint
    from mycelium.ir.serialization import ir_from_json, ir_to_json
    _IR_AVAILABLE = True
except ImportError:
    IRGraph          = None   # type: ignore
    ConfidenceState  = None   # type: ignore
    GraphFingerprint = None   # type: ignore
    ir_from_json     = None   # type: ignore
    ir_to_json       = None   # type: ignore
    _IR_AVAILABLE    = False

_STORE_FILENAME = "graphstore.jsonl"


# ---------------------------------------------------------------------------
# GraphStore
# ---------------------------------------------------------------------------

class GraphStore:
    """Versioned immutable in-memory store for IRGraph objects.

    Parameters
    ----------
    persistence_path : str or Path, optional
        Directory where graphstore.jsonl is written and loaded.
        When None (default) the store is in-memory only (old behaviour).
    run_decay_on_load : bool
        Whether to run GraphDecayManager.decay_pass() after loading
        persisted graphs (default True).  Set False in tests.
    decay_manager : GraphDecayManager, optional
        Provide a custom decay manager.  If None a default one is created.

    Thread-safety: not guaranteed (single-worker Phase D).
    Phase F will add a lock wrapper for multi-worker use.
    """

    def __init__(
        self,
        persistence_path: Optional[str] = None,
        run_decay_on_load: bool = True,
        decay_manager: Optional[GraphDecayManager] = None,
    ) -> None:
        # graph_id → ordered list of versions (index 0 = v1)
        self._store: Dict[str, List] = {}
        # graph_id → observation count
        self._obs: Dict[str, int] = {}
        # semantic_hash → set of graph_ids that carry that hash
        self._hash_index: Dict[str, Set[str]] = {}

        self._decay_mgr = decay_manager or GraphDecayManager()

        # Persistence setup
        self._persist_path: Optional[pathlib.Path] = None
        self._jsonl_path:   Optional[pathlib.Path] = None
        if persistence_path is not None:
            p = pathlib.Path(persistence_path)
            p.mkdir(parents=True, exist_ok=True)
            self._persist_path = p
            self._jsonl_path   = p / _STORE_FILENAME
            self._load_from_disk()
            if run_decay_on_load and len(self._store) > 0:
                summary = self._decay_mgr.decay_pass(self)
                logger.info(
                    "GraphStore: boot decay pass complete — %s", summary
                )
                # Compact the file after decay so deprecated graphs
                # are not needlessly reloaded next time.
                self._compact_to_disk()

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def put(self, graph) -> str:
        """Store a new graph.  Returns the graph_id.

        Writes to disk when persistence_path is set.
        """
        if not _IR_AVAILABLE:
            raise RuntimeError("GraphStore.put: mycelium IR stack unavailable")

        gid = graph.graph_id
        if gid not in self._store:
            self._store[gid] = []
            self._obs[gid] = 0

        self._store[gid].append(graph)
        self._index_graph(graph)

        # Persist to disk (append-only)
        self._append_to_disk(graph)

        logger.debug("GraphStore.put: stored %s (v%d)", gid, len(self._store[gid]))
        return gid

    def observe(self, graph_id: str) -> int:
        """Increment and return the observation count for graph_id."""
        if graph_id not in self._obs:
            self._obs[graph_id] = 0
        self._obs[graph_id] += 1
        return self._obs[graph_id]

    def add_revision(
        self,
        graph_id: str,
        *,
        new_state: Optional[str] = None,
        new_confidence: Optional[float] = None,
        extra_metadata: Optional[dict] = None,
    ):
        """Create a new version of an existing graph (§38 immutability)."""
        current = self.get_latest(graph_id)
        if current is None:
            raise KeyError(
                f"GraphStore.add_revision: graph_id '{graph_id}' not found"
            )

        version_num     = len(self._store[graph_id]) + 1
        new_version_str = f"v{version_num}"
        now             = datetime.datetime.now(datetime.UTC).isoformat()

        conf = current.confidence_state
        if new_confidence is not None and ConfidenceState is not None:
            conf = ConfidenceState(
                overall_confidence=new_confidence,
                semantic_confidence=conf.semantic_confidence,
                structural_confidence=conf.structural_confidence,
                epistemic_confidence=conf.epistemic_confidence,
                evidence_confidence=conf.evidence_confidence,
                temporal_confidence=conf.temporal_confidence,
                contradiction_penalty=conf.contradiction_penalty,
                aggregation_method=conf.aggregation_method,
                confidence_sources=list(conf.confidence_sources),
            )

        revised = IRGraph(
            graph_id=graph_id,
            nodes=list(current.nodes),
            edges=list(current.edges),
            fingerprint=current.fingerprint,
            state=new_state if new_state is not None else current.state,
            version=new_version_str,
            confidence_state=conf,
            ontology_version=current.ontology_version,
            created_at=current.created_at,
            updated_at=now,
            parent_graph_id=current.parent_graph_id,
        )

        self._store[graph_id].append(revised)
        self._index_graph(revised)
        self._append_to_disk(revised)

        logger.debug(
            "GraphStore.add_revision: %s → %s (state=%s)",
            graph_id, new_version_str, revised.state,
        )
        return revised

    def branch_graph(
        self,
        parent_graph_id: str,
        new_graph_id: str,
        *,
        initial_state: str = "DRAFT",
    ):
        """Fork a graph (§39-41 branching model)."""
        parent = self.get_latest(parent_graph_id)
        if parent is None:
            raise KeyError(
                f"GraphStore.branch_graph: parent '{parent_graph_id}' not found"
            )

        now = datetime.datetime.now(datetime.UTC).isoformat()
        branch = IRGraph(
            graph_id=new_graph_id,
            nodes=list(parent.nodes),
            edges=list(parent.edges),
            fingerprint=parent.fingerprint,
            state=initial_state,
            version="v1",
            confidence_state=ConfidenceState(
                overall_confidence=parent.confidence_state.overall_confidence
            ),
            ontology_version=parent.ontology_version,
            created_at=now,
            updated_at=now,
            parent_graph_id=parent_graph_id,
        )
        self.put(branch)
        logger.debug(
            "GraphStore.branch_graph: %s ← parent=%s",
            new_graph_id, parent_graph_id,
        )
        return branch

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def get_latest(self, graph_id: str) -> Optional[object]:
        """Return the most recent version of a graph, or None."""
        versions = self._store.get(graph_id)
        return versions[-1] if versions else None

    def get_version(self, graph_id: str, version: str) -> Optional[object]:
        """Return a specific version of a graph, or None."""
        for g in self._store.get(graph_id, []):
            if g.version == version:
                return g
        return None

    def get_all_versions(self, graph_id: str) -> List:
        """Return all versions of a graph, oldest first."""
        return list(self._store.get(graph_id, []))

    def list_all(self, *, state: Optional[str] = None) -> List:
        """
        Return the latest version of every stored graph.

        Parameters
        ----------
        state : str, optional
            Filter to graphs whose latest version has this state.
            When None (default) all non-DEPRECATED graphs are returned.
            Pass state="DEPRECATED" explicitly to include deprecated ones.

        Returns
        -------
        list[IRGraph]
            Latest versions, sorted by graph_id for deterministic order.
        """
        results = []
        for gid in sorted(self._store.keys()):
            g = self.get_latest(gid)
            if g is None:
                continue
            if state is None:
                # Default: exclude DEPRECATED from regular queries
                if g.state == "DEPRECATED":
                    continue
            elif g.state != state:
                continue
            results.append(g)
        return results

    def find_by_semantic_hash(self, semantic_hash: str) -> List:
        """Return latest versions of all graphs whose nodes include this hash."""
        results = []
        for gid in self._hash_index.get(semantic_hash, set()):
            g = self.get_latest(gid)
            if g is not None:
                results.append(g)
        results.sort(key=lambda g: g.graph_id)
        return results

    def find_by_state(self, state: str) -> List:
        """Return latest versions of all graphs in the given state.

        Delegates to list_all(state=state).  Kept for API compatibility.
        """
        return self.list_all(state=state)

    def find_by_predicate_family(self, predicate_family: str) -> List:
        """Return latest graphs where any node's predicate_family matches."""
        results = []
        for gid in self._store:
            g = self.get_latest(gid)
            if g is None:
                continue
            for node in g.nodes:
                try:
                    if node.semantic_signature.predicate_family == predicate_family:
                        results.append(g)
                        break
                except AttributeError:
                    pass
        results.sort(key=lambda g: g.graph_id)
        return results

    def observation_count(self, graph_id: str) -> int:
        """Return how many times graph_id has been observed."""
        return self._obs.get(graph_id, 0)

    def all_graph_ids(self) -> List[str]:
        """Return all known graph IDs, sorted."""
        return sorted(self._store.keys())

    def __len__(self) -> int:
        return len(self._store)

    # ------------------------------------------------------------------
    # Persistence — internal
    # ------------------------------------------------------------------

    def _append_to_disk(self, graph) -> None:
        """Append a single graph version to the JSONL file."""
        if self._jsonl_path is None or not _IR_AVAILABLE or ir_to_json is None:
            return
        try:
            line = ir_to_json(graph)          # serialization.py: IRGraph → str
            with self._jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception as exc:
            logger.warning("GraphStore: disk write failed (%s)", exc)

    def _load_from_disk(self) -> None:
        """Load graphs from graphstore.jsonl on boot.

        Strategy: read all lines, keep only the last version seen
        per graph_id (the file is append-only so the last line for a
        given graph_id is the current head).  Then replay them through
        the normal put() flow (minus the disk write to avoid re-appending).
        """
        if self._jsonl_path is None or not self._jsonl_path.exists():
            return
        if not _IR_AVAILABLE or ir_from_json is None:
            logger.warning(
                "GraphStore: IR stack unavailable, skipping persistence load"
            )
            return

        latest_per_graph: Dict[str, str] = {}  # graph_id → last JSON line
        line_count = 0
        try:
            with self._jsonl_path.open("r", encoding="utf-8") as fh:
                for raw_line in fh:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    try:
                        # Peek at graph_id without full deserialisation
                        probe = json.loads(raw_line)
                        gid   = probe.get("graph_id", "")
                        if gid:
                            latest_per_graph[gid] = raw_line
                            line_count += 1
                    except json.JSONDecodeError:
                        continue
        except Exception as exc:
            logger.error("GraphStore: could not read %s (%s)", self._jsonl_path, exc)
            return

        loaded = 0
        for gid, line in latest_per_graph.items():
            try:
                graph = ir_from_json(line)    # serialization.py: str → IRGraph
                # Replay into memory only (no disk append)
                g_id = graph.graph_id
                if g_id not in self._store:
                    self._store[g_id] = []
                    self._obs[g_id]   = 0
                self._store[g_id].append(graph)
                self._index_graph(graph)
                loaded += 1
            except Exception as exc:
                logger.debug(
                    "GraphStore: skipping unparseable line for %s (%s)", gid, exc
                )

        logger.info(
            "GraphStore: loaded %d/%d graphs from %s",
            loaded, line_count, self._jsonl_path,
        )

    def _compact_to_disk(self) -> None:
        """Rewrite the JSONL file keeping only the latest version per graph.

        Called after a decay pass so the file stays lean.
        All DEPRECATED graphs are still written (immutability — they exist)
        but future list_all() calls exclude them by default.
        """
        if self._jsonl_path is None or not _IR_AVAILABLE or ir_to_json is None:
            return
        try:
            tmp_path = self._jsonl_path.with_suffix(".jsonl.tmp")
            with tmp_path.open("w", encoding="utf-8") as fh:
                for gid in sorted(self._store.keys()):
                    graph = self.get_latest(gid)
                    if graph is None:
                        continue
                    fh.write(ir_to_json(graph) + "\n")
            tmp_path.replace(self._jsonl_path)
            logger.debug(
                "GraphStore: compacted to %d graphs at %s",
                len(self._store), self._jsonl_path,
            )
        except Exception as exc:
            logger.warning("GraphStore: compact failed (%s)", exc)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _index_graph(self, graph) -> None:
        """Update the semantic_hash index for fast lookup."""
        try:
            for node in graph.nodes:
                h = node.semantic_signature.semantic_hash
                if h:
                    self._hash_index.setdefault(h, set()).add(graph.graph_id)
        except Exception as exc:
            logger.debug("GraphStore._index_graph: %s", exc)
