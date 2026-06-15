from __future__ import annotations
import collections
import logging
import time
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ReplayEntry:
    embedding: List[float]
    query_text: str
    domain_id: str
    label: int = 1              # 1 = in-domain confirmed, 0 = negative
    timestamp: float = field(default_factory=time.time)
    meta: Dict = field(default_factory=dict)


class ReplayBuffer:
    """
    Fixed-capacity per-domain circular buffer of ReplayEntry objects.

    Contract:
      - max_size is enforced; oldest entries are evicted when full.
      - Only embeddings + labels are stored — raw text kept for debugging only.
      - drain() empties the buffer and returns all entries for training.
    """

    def __init__(self, domain_id: str, max_size: int = 2000) -> None:
        self.domain_id = domain_id
        self._max_size = max_size
        self._buffer: Deque[ReplayEntry] = collections.deque(maxlen=max_size)

    def push(self, entry: ReplayEntry) -> None:
        if entry.domain_id != self.domain_id:
            logger.warning("ReplayBuffer[%s]: rejecting entry for domain '%s'.",
                           self.domain_id, entry.domain_id)
            return
        self._buffer.append(entry)

    def drain(self) -> List[ReplayEntry]:
        entries = list(self._buffer)
        self._buffer.clear()
        return entries

    def sample(self, n: int) -> List[ReplayEntry]:
        import random
        buf = list(self._buffer)
        return random.sample(buf, min(n, len(buf)))

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        return len(self._buffer) >= self._max_size

    def label_counts(self) -> Tuple[int, int]:
        """Returns (positive_count, negative_count)."""
        pos = sum(1 for e in self._buffer if e.label == 1)
        return pos, len(self._buffer) - pos


class ReplayBufferManager:
    """
    Manages one ReplayBuffer per domain_id.
    """

    def __init__(self, default_max_size: int = 2000) -> None:
        self._default_max = default_max_size
        self._buffers: Dict[str, ReplayBuffer] = {}

    def get_or_create(self, domain_id: str, max_size: Optional[int] = None) -> ReplayBuffer:
        if domain_id not in self._buffers:
            self._buffers[domain_id] = ReplayBuffer(
                domain_id, max_size or self._default_max
            )
        return self._buffers[domain_id]

    def push(self, entry: ReplayEntry) -> None:
        buf = self.get_or_create(entry.domain_id)
        buf.push(entry)

    def drain(self, domain_id: str) -> List[ReplayEntry]:
        return self.get_or_create(domain_id).drain()

    def sizes(self) -> Dict[str, int]:
        return {did: len(buf) for did, buf in self._buffers.items()}

    def all_domain_ids(self) -> List[str]:
        return list(self._buffers.keys())
