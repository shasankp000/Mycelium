from __future__ import annotations
import heapq
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from mycelium.domain_graph.registry import DomainGraphRegistry
from mycelium.domain_graph.state import DomainState, GateState

logger = logging.getLogger(__name__)


@dataclass(order=True)
class ReactivationTicket:
    priority: float                         # lower = higher priority (min-heap)
    domain_id: str = field(compare=False)
    reason: str = field(compare=False, default="")
    enqueued_at: float = field(compare=False, default_factory=time.time)
    novelty_score: float = field(compare=False, default=0.0)
    query_count: int = field(compare=False, default=0)


class ReactivationQueue:
    """
    Min-heap priority queue of cold domains pending reactivation.

    Priority = -(novelty_score * log1p(query_count))
    Higher novelty + more demand → floats to top.

    Contract:
      - Only COLD domains are ever enqueued.
      - Calling pop() does NOT change registry state — the caller must
        call registry.set_state(domain_id, DomainState.WARM) explicitly.
      - Duplicate enqueues for the same domain update priority in-place.
    """

    def __init__(self, registry: DomainGraphRegistry) -> None:
        self._registry = registry
        self._heap: List[ReactivationTicket] = []
        self._domain_index: Dict[str, ReactivationTicket] = {}

    # ------------------------------------------------------------------
    # Enqueue / update
    # ------------------------------------------------------------------

    def enqueue(
        self,
        domain_id: str,
        novelty_score: float,
        query_count: int,
        reason: str = "",
    ) -> None:
        import math
        priority = -(novelty_score * math.log1p(query_count))

        if domain_id in self._domain_index:
            # Update priority of existing ticket — mark old as stale by setting
            # a sentinel domain_id, then push a new one
            old = self._domain_index[domain_id]
            old.domain_id = "__stale__"

        ticket = ReactivationTicket(
            priority=priority,
            domain_id=domain_id,
            reason=reason,
            novelty_score=novelty_score,
            query_count=query_count,
        )
        heapq.heappush(self._heap, ticket)
        self._domain_index[domain_id] = ticket
        logger.debug("ReactivationQueue: enqueued '%s' priority=%.4f", domain_id, priority)

    def pop(self) -> Optional[ReactivationTicket]:
        """Returns the highest-priority non-stale ticket, or None if empty."""
        while self._heap:
            ticket = heapq.heappop(self._heap)
            if ticket.domain_id == "__stale__":
                continue
            self._domain_index.pop(ticket.domain_id, None)
            return ticket
        return None

    def peek(self) -> Optional[ReactivationTicket]:
        """Non-destructive peek at the highest-priority ticket."""
        for ticket in self._heap:
            if ticket.domain_id != "__stale__":
                return ticket
        return None

    def __len__(self) -> int:
        return sum(1 for t in self._heap if t.domain_id != "__stale__")

    def domain_ids(self) -> List[str]:
        return [t.domain_id for t in self._heap if t.domain_id != "__stale__"]
