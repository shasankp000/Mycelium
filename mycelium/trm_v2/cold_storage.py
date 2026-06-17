"""
mycelium/trm_v2/cold_storage.py

ColdStorageManager — Hot / Warm / Cold lifecycle manager for domain heads
and their associated DomainShards.

Lifecycle transitions managed here:
    HOT  -> WARM   : head stays in memory, shard stays open, usage counter
                     drops below warm_threshold
    WARM -> COLD   : head checkpoint serialised to disk, head evicted from
                     memory, shard closed and left on disk
    COLD -> REMEMBERING -> WARM/HOT : shard reopened, head checkpoint
                     reloaded into HeadRegistry, state advanced

State machine rules (from types.py — enforced via DomainGraph.transition_state):
    No domain is ever hard-deleted.
    COLD domains keep their SQLite shard on disk indefinitely.
    Head checkpoints are written atomically (tmp file + rename).

Eviction policy:
    LRU (Least Recently Used) over HOT+WARM domains.
    When total_hot_warm > max_hot_warm the coldest LRU domain is offloaded.

Spec refs:
    mycelium_trm_v2_theoretical_spec_v0.2.md  §Hot/Warm/Cold Lifecycle
    mycelium_trm_v2_theoretical_spec_hardening_additions.md  Addition E
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional

import torch

from mycelium.trm_v2.shared_encoder import DomainHead, HeadRegistry
from mycelium.trm_v2.domain_shard import DomainShard
from mycelium.trm_v2.types import (
    DomainGraph,
    DomainNode,
    DomainState,
    DomainMode,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Filename constants — single source of truth for shard and checkpoint names
# ---------------------------------------------------------------------------
SHARD_FILENAME = "shard.db"
HEAD_CKPT_FILENAME = "head.pt"


# ---------------------------------------------------------------------------
# Internal tracking record
# ---------------------------------------------------------------------------

class _DomainRecord:
    """Runtime tracking for one domain within ColdStorageManager."""

    __slots__ = (
        "domain_id", "shard_path", "head_ckpt_path",
        "last_access", "access_count", "shard",
    )

    def __init__(self, domain_id: str, shard_path: str, head_ckpt_path: str) -> None:
        self.domain_id     = domain_id
        self.shard_path    = shard_path
        self.head_ckpt_path = head_ckpt_path
        self.last_access   = time.monotonic()
        self.access_count  = 0
        self.shard: Optional[DomainShard] = None

    def touch(self) -> None:
        self.last_access  = time.monotonic()
        self.access_count += 1


# ---------------------------------------------------------------------------
# ColdStorageManager
# ---------------------------------------------------------------------------

class ColdStorageManager:
    """Manages memory pressure and persistence for all domain heads + shards.

    Parameters
    ----------
    storage_root  : str | Path   base directory for all shard + ckpt files
    head_registry : HeadRegistry   shared registry of in-memory DomainHeads
    domain_graph  : DomainGraph    shared graph (state transitions go here)
    max_hot_warm  : int   max domains to keep in HOT+WARM state simultaneously
    warm_threshold: int   access count below which a HOT domain becomes WARM
    """

    def __init__(
        self,
        storage_root:   str | Path,
        head_registry:  HeadRegistry,
        domain_graph:   DomainGraph,
        max_hot_warm:   int = 16,
        warm_threshold: int = 5,
    ) -> None:
        self.root          = Path(storage_root)
        self.registry      = head_registry
        self.graph         = domain_graph
        self.max_hot_warm  = max_hot_warm
        self.warm_threshold = warm_threshold

        self.root.mkdir(parents=True, exist_ok=True)

        # LRU-ordered dict: domain_id -> _DomainRecord
        # Most recently used at the right (end), LRU at the left (front).
        self._lru: OrderedDict[str, _DomainRecord] = OrderedDict()

    # ---- Path helpers ------------------------------------------------------

    def _shard_path(self, domain_id: str) -> Path:
        return self.root / domain_id / SHARD_FILENAME

    def _head_ckpt_path(self, domain_id: str) -> Path:
        return self.root / domain_id / HEAD_CKPT_FILENAME

    # ---- Domain registration -----------------------------------------------

    def register_domain(
        self,
        node: DomainNode,
        head: Optional[DomainHead] = None,
    ) -> DomainShard:
        """Register a new domain.

        - Creates the storage directory.
        - Opens (or creates) the DomainShard.
        - Registers the head in HeadRegistry (if provided).
        - Advances node state CREATING -> HOT.
        - Enforces LRU eviction if needed.

        Returns the open DomainShard for the caller to populate.
        """
        did = node.domain_id
        if did in self._lru:
            raise ValueError(f"ColdStorageManager: domain '{did}' already registered.")

        (self.root / did).mkdir(parents=True, exist_ok=True)

        rec = _DomainRecord(
            domain_id=did,
            shard_path=str(self._shard_path(did)),
            head_ckpt_path=str(self._head_ckpt_path(did)),
        )
        shard = DomainShard(rec.shard_path)
        shard.open()
        rec.shard = shard
        rec.touch()

        # Update node storage refs
        node.sqlite_shard_ref = rec.shard_path
        node.cold_storage_ref = str(self.root / did)
        if head is not None:
            node.head_ref = rec.head_ckpt_path

        # Register head
        if head is not None:
            self.registry.register(did, head)

        # State: CREATING -> HOT
        self.graph.transition_state(did, DomainState.HOT)

        self._lru[did] = rec
        self._lru.move_to_end(did)

        logger.info("ColdStorageManager: domain '%s' registered -> HOT.", did)

        # Evict if over budget
        self._maybe_evict()

        return shard

    # ---- Access (touch LRU) ------------------------------------------------

    def access(self, domain_id: str) -> Optional[DomainShard]:
        """Record an access, advance LRU position, return open shard.

        If domain is COLD this triggers REMEMBERING -> reactivation.
        Returns None if domain_id is unknown.
        """
        node = self.graph.get_node(domain_id)
        if node is None:
            logger.warning("ColdStorageManager.access: unknown domain '%s'.", domain_id)
            return None

        if node.state == DomainState.COLD:
            return self._reactivate(domain_id)

        rec = self._lru.get(domain_id)
        if rec is None:
            return None

        rec.touch()
        self._lru.move_to_end(domain_id)

        # Promote WARM -> HOT if access count passes threshold
        if node.state == DomainState.WARM and rec.access_count >= self.warm_threshold:
            self.graph.transition_state(domain_id, DomainState.HOT)
            logger.debug("ColdStorageManager: '%s' WARM -> HOT.", domain_id)

        return rec.shard

    # ---- Offload (HOT/WARM -> COLD) ----------------------------------------

    def offload(self, domain_id: str) -> None:
        """Persist head + close shard, evict from memory -> COLD.

        Safe to call on a WARM domain. Raises if domain is already COLD
        or not in an offloadable state.
        """
        node = self.graph.get_node(domain_id)
        if node is None:
            raise KeyError(f"ColdStorageManager.offload: unknown domain '{domain_id}'.")
        if node.state not in (DomainState.HOT, DomainState.WARM):
            raise ValueError(
                f"ColdStorageManager.offload: domain '{domain_id}' is in state "
                f"{node.state.value}, expected HOT or WARM."
            )

        rec = self._lru.get(domain_id)
        if rec is None:
            raise KeyError(f"ColdStorageManager.offload: no record for '{domain_id}'.")

        # Persist head checkpoint atomically
        head = self.registry.get(domain_id)
        if head is not None:
            self._write_head_ckpt(head, rec.head_ckpt_path)
            self.registry.remove(domain_id)
            logger.debug("ColdStorageManager: head checkpoint written for '%s'.", domain_id)

        # Close shard
        if rec.shard is not None:
            rec.shard.close()
            rec.shard = None

        # State transition: HOT/WARM -> COLD
        # Must go through WARM first if currently HOT
        if node.state == DomainState.HOT:
            self.graph.transition_state(domain_id, DomainState.WARM)
        self.graph.transition_state(domain_id, DomainState.COLD)

        logger.info("ColdStorageManager: domain '%s' offloaded -> COLD.", domain_id)

    # ---- Reactivation (COLD -> REMEMBERING -> WARM) ------------------------

    def _reactivate(self, domain_id: str) -> Optional[DomainShard]:
        """Reload head + reopen shard. Returns the open shard."""
        node = self.graph.get_node(domain_id)
        if node is None:
            return None

        self.graph.transition_state(domain_id, DomainState.REMEMBERING)
        logger.info("ColdStorageManager: domain '%s' -> REMEMBERING.", domain_id)

        rec = self._lru.get(domain_id)
        if rec is None:
            # Reconstruct record from node storage refs
            rec = _DomainRecord(
                domain_id=domain_id,
                shard_path=node.sqlite_shard_ref or str(self._shard_path(domain_id)),
                head_ckpt_path=node.head_ref or str(self._head_ckpt_path(domain_id)),
            )
            self._lru[domain_id] = rec

        # Reopen shard
        shard = DomainShard(rec.shard_path)
        shard.open()
        rec.shard = shard

        # Reload head if checkpoint exists
        ckpt_path = Path(rec.head_ckpt_path)
        if ckpt_path.exists():
            head = self._load_head_ckpt(rec.head_ckpt_path)
            if head is not None:
                try:
                    self.registry.register(domain_id, head)
                except KeyError:
                    pass  # already registered (concurrent call)

        rec.touch()
        self._lru.move_to_end(domain_id)

        self.graph.transition_state(domain_id, DomainState.WARM)
        logger.info("ColdStorageManager: domain '%s' -> WARM (reactivated).", domain_id)

        self._maybe_evict()
        return shard

    # ---- Deprecation -------------------------------------------------------

    def deprecate(self, domain_id: str) -> None:
        """Transition a domain through COLD -> DEPRECATED -> (eventually ARCHIVED).

        The shard and head checkpoint are retained on disk.
        The domain is removed from the LRU pool.
        """
        node = self.graph.get_node(domain_id)
        if node is None:
            raise KeyError(f"ColdStorageManager.deprecate: unknown '{domain_id}'.")

        # Ensure it's offloaded first
        if node.state in (DomainState.HOT, DomainState.WARM):
            self.offload(domain_id)

        if node.state == DomainState.COLD:
            self.graph.transition_state(domain_id, DomainState.DEPRECATED)

        self._lru.pop(domain_id, None)
        logger.info("ColdStorageManager: domain '%s' -> DEPRECATED.", domain_id)

    # ---- LRU eviction ------------------------------------------------------

    def _maybe_evict(self) -> None:
        """Offload the LRU domain if the hot+warm count exceeds budget."""
        hot_warm = [
            did for did in self._lru
            if self.graph.get_node(did) is not None
            and self.graph.get_node(did).state in (DomainState.HOT, DomainState.WARM)
        ]
        while len(hot_warm) > self.max_hot_warm:
            lru_id = hot_warm.pop(0)
            logger.info(
                "ColdStorageManager: LRU eviction of '%s' (budget=%d).",
                lru_id, self.max_hot_warm,
            )
            self.offload(lru_id)

    # ---- Checkpoint helpers ------------------------------------------------

    @staticmethod
    def _write_head_ckpt(head: DomainHead, path: str) -> None:
        """Atomic write: tmp file + os.replace."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
        os.close(fd)
        try:
            torch.save(
                {
                    "domain_id":   head.domain_id,
                    "input_dim":   head.input_dim,
                    "head_dim":    head.head_dim,
                    "num_classes": head.num_classes,
                    "state_dict":  head.state_dict(),
                },
                tmp,
            )
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @staticmethod
    def _load_head_ckpt(path: str) -> Optional[DomainHead]:
        """Load a DomainHead from a checkpoint file."""
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            head = DomainHead(
                domain_id=ckpt["domain_id"],
                input_dim=ckpt["input_dim"],
                head_dim=ckpt["head_dim"],
                num_classes=ckpt["num_classes"],
            )
            head.load_state_dict(ckpt["state_dict"])
            return head
        except Exception as e:
            logger.error("ColdStorageManager: failed to load head ckpt '%s': %s", path, e)
            return None

    # ---- Diagnostics -------------------------------------------------------

    def status(self) -> Dict[str, str]:
        """Return {domain_id: state_value} for all tracked domains."""
        out = {}
        for did in self._lru:
            node = self.graph.get_node(did)
            out[did] = node.state.value if node else "UNKNOWN"
        return out
