# mycelium/pipeline/predicates/predicate_store.py
# Request-scoped semantic working memory for the predicate engine.
#
# PredicateStore holds all PredicateFrames for a single pipeline request.
# It is attached to the workflow context after L1 routing and is accessible
# from Phase 2 and Phase 3 without re-parsing the raw query.
#
# Design constraints (from spec):
# - Iteration order MUST be deterministic via sequence_number, not dict
#   insertion order assumptions.
# - Provenance traversal (lineage, descendants) must be supported.
# - Replay-safe: same insertion sequence -> same iteration order always.
#
# Spec ref: implementation spec v0.2.1 — Section 4.5

from __future__ import annotations

from typing import Dict, Iterator, List, Optional

from mycelium.pipeline.predicates.predicate_relations import (
    PredicateGraphRelation,
    PredicateRelationType,
)
from mycelium.pipeline.predicates.predicate_types import (
    PredicateFrame,
    PredicateType,
)


class PredicateStore:
    """Request-scoped semantic working memory.

    Holds all PredicateFrames extracted for one pipeline request.
    Provides indexed retrieval, relation graph storage, and
    provenance traversal.

    Usage
    -----
    Constructed by the predicate extraction step in run_workflow.py
    after L1 routing::

        store = PredicateStore()
        for frame in extracted_frames:
            store.add(frame)
        context.predicate_store = store

    Thread safety
    -------------
    Not thread-safe. Each request owns its own PredicateStore instance.
    Do not share across concurrent requests.
    """

    def __init__(self) -> None:
        # Primary index: predicate_id -> PredicateFrame
        self._frames: Dict[str, PredicateFrame] = {}
        # Insertion-ordered sequence numbers for deterministic iteration
        self._sequence: Dict[str, int] = {}
        self._counter: int = 0
        # Relation graph: predicate_id -> list of outgoing edges
        self._relations: Dict[str, List[PredicateGraphRelation]] = {}

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------

    def add(self, frame: PredicateFrame) -> None:
        """Insert a PredicateFrame into the store.

        If a frame with the same predicate_id already exists, it is
        replaced and its sequence number is preserved (update-in-place
        semantics for re-extraction / scope narrowing).
        """
        if frame.predicate_id not in self._sequence:
            self._sequence[frame.predicate_id] = self._counter
            self._counter += 1
        self._frames[frame.predicate_id] = frame
        if frame.predicate_id not in self._relations:
            self._relations[frame.predicate_id] = []

    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: PredicateRelationType,
        confidence: float = 1.0,
    ) -> PredicateGraphRelation:
        """Add a directed typed edge between two frames already in the store.

        Returns the created PredicateGraphRelation.
        Raises KeyError if either id is not present in the store.
        """
        if source_id not in self._frames:
            raise KeyError(f"Source predicate '{source_id}' not in store")
        if target_id not in self._frames:
            raise KeyError(f"Target predicate '{target_id}' not in store")
        edge = PredicateGraphRelation(
            source_predicate_id=source_id,
            target_predicate_id=target_id,
            relation_type=relation_type,
            confidence=confidence,
        )
        self._relations[source_id].append(edge)
        return edge

    # ------------------------------------------------------------------
    # Core retrieval
    # ------------------------------------------------------------------

    def get(self, predicate_id: str) -> Optional[PredicateFrame]:
        """Return the frame for the given id, or None if not present."""
        return self._frames.get(predicate_id)

    def all(self) -> List[PredicateFrame]:
        """Return all frames in deterministic insertion order."""
        return sorted(
            self._frames.values(),
            key=lambda f: self._sequence[f.predicate_id],
        )

    def primary(self) -> Optional[PredicateFrame]:
        """Return the first inserted frame (sequence_number == 0), or None.

        Conventionally the primary predicate is the main claim of the
        query. PredicateExtractor inserts it first.
        """
        if not self._frames:
            return None
        return min(
            self._frames.values(),
            key=lambda f: self._sequence[f.predicate_id],
        )

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterator[PredicateFrame]:
        return iter(self.all())

    def __contains__(self, predicate_id: str) -> bool:
        return predicate_id in self._frames

    # ------------------------------------------------------------------
    # Filtered retrieval
    # ------------------------------------------------------------------

    def by_type(self, ptype: PredicateType) -> List[PredicateFrame]:
        """Return all frames of the given PredicateType, in insertion order."""
        return [
            f for f in self.all()
            if f.predicate_type == ptype
        ]

    def by_domain(self, domain: str) -> List[PredicateFrame]:
        """Return all frames whose domain_hint matches the given string."""
        return [
            f for f in self.all()
            if f.domain_hint == domain
        ]

    def falsifiable(self) -> List[PredicateFrame]:
        """Return all frames where falsifiable=True.

        This is the set EvidenceFinder operates over.
        NORMATIVE frames (falsifiable=False) are excluded.
        """
        return [f for f in self.all() if f.falsifiable]

    # ------------------------------------------------------------------
    # Relation / provenance traversal
    # ------------------------------------------------------------------

    def related(
        self,
        predicate_id: str,
        relation_type: Optional[PredicateRelationType] = None,
    ) -> List[PredicateFrame]:
        """Return frames directly reachable from predicate_id via outgoing edges.

        Parameters
        ----------
        predicate_id:
            Source frame to traverse from.
        relation_type:
            If given, only edges of this type are followed.
            If None, all outgoing edges are followed.

        Returns an empty list if predicate_id is not in the store.
        """
        edges = self._relations.get(predicate_id, [])
        if relation_type is not None:
            edges = [e for e in edges if e.relation_type == relation_type]
        result = []
        for edge in edges:
            frame = self._frames.get(edge.target_predicate_id)
            if frame is not None:
                result.append(frame)
        return result

    def lineage(self, predicate_id: str) -> List[PredicateFrame]:
        """Return all ancestor frames by walking derived_from recursively.

        Returns frames in breadth-first order from the immediate parents
        up to the root extraction frame(s). Returns an empty list if
        predicate_id is not in the store or has no ancestors.
        """
        visited: set[str] = set()
        queue: List[str] = list(
            self._frames[predicate_id].derived_from
            if predicate_id in self._frames
            else []
        )
        result: List[PredicateFrame] = []
        while queue:
            pid = queue.pop(0)
            if pid in visited:
                continue
            visited.add(pid)
            frame = self._frames.get(pid)
            if frame is not None:
                result.append(frame)
                queue.extend(frame.derived_from)
        return result

    def descendants(self, predicate_id: str) -> List[PredicateFrame]:
        """Return all frames that list predicate_id in their derived_from chain.

        Performs a full scan of the store. For large stores with many
        derivations, this is O(n * depth). Acceptable for v0.2 request
        scope sizes.
        """
        visited: set[str] = set()
        result: List[PredicateFrame] = []

        def _walk(pid: str) -> None:
            for frame in self.all():
                if pid in frame.derived_from and frame.predicate_id not in visited:
                    visited.add(frame.predicate_id)
                    result.append(frame)
                    _walk(frame.predicate_id)

        _walk(predicate_id)
        return result

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a lightweight dict summary suitable for SSE events and logging."""
        return {
            "total": len(self._frames),
            "falsifiable": len(self.falsifiable()),
            "by_type": {
                ptype.value: len(self.by_type(ptype))
                for ptype in PredicateType
                if self.by_type(ptype)
            },
            "relation_edges": sum(len(v) for v in self._relations.values()),
            "primary_id": self.primary().predicate_id if self.primary() else None,
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"PredicateStore(total={s['total']}, "
            f"falsifiable={s['falsifiable']}, "
            f"edges={s['relation_edges']})"
        )
