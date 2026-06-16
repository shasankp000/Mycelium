"""
mycelium/trm_v2/pipeline.py

TRMV2Pipeline — the top-level wiring that ties together:
    SharedEncoder  -> GatingNetwork -> DomainHead(s) -> QueryStore(s)

Query flow:
    1. text/embedding -> SharedEncoder -> latent [B, D]
    2. latent -> GatingNetwork -> GateDecision (top-k domain ids + scores)
    3. For each selected domain:
         a. ColdStorageManager.access() -> open DomainShard
         b. QueryStore.retrieve()       -> top-k ChunkResults
         c. DomainHead.forward()        -> domain-specific projection/logits
    4. Results fused + returned as PipelineResult

Domain management API (exposed here so callers don't need to touch
internals directly):
    pipeline.add_domain(name, description, head_config)
    pipeline.deprecate_domain(domain_id)
    pipeline.ingest(domain_id, text, embedding)

Observability: every routing decision emits a DomainEvent which is
appended to pipeline.event_log (in-memory ring buffer, max 10 000 events).
Consumers (e.g. the web-ui) can drain this log via pipeline.drain_events().

Spec refs:
    mycelium_trm_v2_theoretical_spec_v0.2.md  §TRMV2Pipeline
    mycelium_trm_v2_theoretical_spec_hardening_additions.md  Additions F, H
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import Tensor

from mycelium.trm_v2.types import (
    DomainGraph,
    DomainNode,
    DomainState,
    DomainMode,
    DomainEvent,
    EVT_DOMAIN_CREATE,
    EVT_DOMAIN_ACTIVATE,
    EVT_DOMAIN_DEPRECATE,
)
from mycelium.trm_v2.shared_encoder import SharedEncoder, DomainHead, HeadRegistry
from mycelium.trm_v2.gate import GatingNetwork, GateDecision
from mycelium.trm_v2.cold_storage import ColdStorageManager
from mycelium.trm_v2.domain_shard import DomainShard, QueryStore, ChunkResult

logger = logging.getLogger(__name__)

_MAX_EVENT_LOG = 10_000


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class DomainResult:
    """Per-domain output within a PipelineResult."""
    domain_id:    str
    domain_name:  str
    gate_score:   float
    chunks:       List[ChunkResult]          # top-k retrieved chunks
    head_output:  Optional[Tensor]           # projection or logits from DomainHead
    retrieval_ms: float = 0.0


@dataclass
class PipelineResult:
    """Full output of a single TRMV2Pipeline.query() call."""
    query_text:    str
    latent:        Tensor                    # [1, D] shared encoder output
    gate_decision: GateDecision
    domain_results: List[DomainResult]
    event_id:      str = ""
    elapsed_ms:    float = 0.0


# ---------------------------------------------------------------------------
# Head configuration helper
# ---------------------------------------------------------------------------

@dataclass
class HeadConfig:
    head_dim:    int   = 128
    num_classes: int   = 0
    dropout:     float = 0.1


# ---------------------------------------------------------------------------
# TRMV2Pipeline
# ---------------------------------------------------------------------------

class TRMV2Pipeline:
    """Top-level TRM v2 orchestrator.

    Parameters
    ----------
    storage_root  : str   base path for all shard + checkpoint files
    encoder       : SharedEncoder   (created externally or via make_default())
    hidden_size   : int   must match encoder.hidden_size
    gate_dim      : int   GatingNetwork projection width
    top_k         : int   domains to activate per query
    max_hot_warm  : int   ColdStorageManager budget
    retrieve_k    : int   chunks per domain returned by QueryStore
    """

    def __init__(
        self,
        storage_root: str,
        encoder:      Optional[SharedEncoder] = None,
        hidden_size:  int   = 256,
        gate_dim:     int   = 64,
        top_k:        int   = 2,
        max_hot_warm: int   = 16,
        retrieve_k:   int   = 5,
    ) -> None:
        self.storage_root = storage_root
        self.hidden_size  = hidden_size
        self.retrieve_k   = retrieve_k

        # Sub-systems
        self.encoder = encoder or SharedEncoder(
            hidden_size=hidden_size, frozen=True
        )
        self.graph    = DomainGraph()
        self.registry = HeadRegistry()
        self.gate     = GatingNetwork(
            input_dim=hidden_size,
            gate_dim=gate_dim,
            top_k=top_k,
        )
        self.storage  = ColdStorageManager(
            storage_root=storage_root,
            head_registry=self.registry,
            domain_graph=self.graph,
            max_hot_warm=max_hot_warm,
        )

        # Observability ring buffer
        self._event_log: deque[DomainEvent] = deque(maxlen=_MAX_EVENT_LOG)
        self._seq: int = 0

    # ---- Event helpers -----------------------------------------------------

    def _emit(self, event_type: str, domain_id: Optional[str] = None,
              payload: Optional[Dict[str, Any]] = None) -> DomainEvent:
        self._seq += 1
        ev = DomainEvent(
            event_type=event_type,
            sequence_number=self._seq,
            domain_id=domain_id,
            payload=payload or {},
        )
        self._event_log.append(ev)
        return ev

    def drain_events(self) -> List[DomainEvent]:
        """Return and clear all buffered events."""
        events = list(self._event_log)
        self._event_log.clear()
        return events

    # ---- Domain management -------------------------------------------------

    def add_domain(
        self,
        name:        str,
        description: str = "",
        head_cfg:    Optional[HeadConfig] = None,
        created_by:  str = "pipeline",
    ) -> DomainNode:
        """Register a new domain end-to-end.

        Creates DomainNode, DomainHead, opens DomainShard, registers with
        GatingNetwork, emits EVT_DOMAIN_CREATE.
        Returns the new DomainNode.
        """
        cfg = head_cfg or HeadConfig()
        node = DomainNode(
            name=name,
            description=description,
            creation_reason=f"add_domain called by {created_by}",
            created_by=created_by,
        )
        self.graph.add_node(node)

        head = DomainHead(
            domain_id=node.domain_id,
            input_dim=self.hidden_size,
            head_dim=cfg.head_dim,
            num_classes=cfg.num_classes,
            dropout=cfg.dropout,
        )

        self.storage.register_domain(node, head=head)
        self.gate.register_domain(node.domain_id)

        node.mode = DomainMode.FULL_DOMAIN
        node.head_ref = str(self.storage._head_ckpt_path(node.domain_id))

        self._emit(EVT_DOMAIN_CREATE, domain_id=node.domain_id,
                   payload={"name": name, "description": description})

        logger.info("TRMV2Pipeline: domain '%s' (%s) added.", node.domain_id, name)

        # Advance gate state if we now have enough domains
        self.gate.mark_stable()

        return node

    def deprecate_domain(self, domain_id: str) -> None:
        """Deprecate a domain: offload -> DEPRECATED, unregister from gate."""
        self.storage.deprecate(domain_id)
        self.gate.unregister_domain(domain_id)
        self._emit(EVT_DOMAIN_DEPRECATE, domain_id=domain_id)
        logger.info("TRMV2Pipeline: domain '%s' deprecated.", domain_id)

    # ---- Ingestion ---------------------------------------------------------

    def ingest(
        self,
        domain_id:  str,
        text:       str,
        embedding:  Optional[np.ndarray] = None,
        source:     Optional[str] = None,
        metadata:   Optional[Dict[str, Any]] = None,
    ) -> str:
        """Insert a text chunk (+ optional embedding) into a domain shard.

        Touches the domain in the LRU (may trigger reactivation if COLD).
        Returns the assigned chunk_id.
        """
        shard = self.storage.access(domain_id)
        if shard is None:
            raise KeyError(f"TRMV2Pipeline.ingest: unknown domain '{domain_id}'.")
        return shard.insert_chunk(text, embedding=embedding,
                                  source=source, metadata=metadata)

    # ---- Encoding ----------------------------------------------------------

    def encode(self, text_or_ids: Any) -> Tensor:
        """Convenience: encode raw token ids or a pre-built Tensor -> latent.

        For full text encoding, callers should tokenise externally and pass
        token id tensors. This method handles both:
            - Tensor input  [B, L] int  -> SharedEncoder forward
            - np.ndarray    [D]         -> unsqueeze + encode_embeddings
        """
        if isinstance(text_or_ids, np.ndarray):
            emb = torch.tensor(text_or_ids, dtype=torch.float32)
            if emb.dim() == 1:
                emb = emb.unsqueeze(0).unsqueeze(0)  # [1, 1, D]
            return self.encoder.encode_embeddings(emb)
        if isinstance(text_or_ids, Tensor):
            if text_or_ids.dtype in (torch.float32, torch.float16, torch.bfloat16):
                if text_or_ids.dim() == 1:
                    text_or_ids = text_or_ids.unsqueeze(0).unsqueeze(0)
                return self.encoder.encode_embeddings(text_or_ids)
            return self.encoder(text_or_ids)
        raise TypeError(f"encode: unsupported input type {type(text_or_ids)}")

    # ---- Query -------------------------------------------------------------

    def query(
        self,
        latent:     Tensor,
        query_text: str = "",
        top_k:      Optional[int] = None,
        retrieve_k: Optional[int] = None,
    ) -> PipelineResult:
        """Run a full query through the pipeline.

        Parameters
        ----------
        latent      : [1, D] or [B, D]  from encode() or external encoder
        query_text  : raw text for FTS prefiltering in QueryStore
        top_k       : override gate top_k for this call
        retrieve_k  : override chunks-per-domain for this call

        Returns
        -------
        PipelineResult with per-domain chunks + head outputs
        """
        t0 = time.perf_counter()
        rk = retrieve_k or self.retrieve_k

        # Step 1 — gate routing
        gate_dec = self.gate.forward(latent, top_k=top_k)

        domain_results: List[DomainResult] = []

        # Step 2+3 — per-domain retrieval + head forward
        for did, gscore in zip(gate_dec.domain_ids, gate_dec.scores):
            node = self.graph.get_node(did)
            t_ret = time.perf_counter()

            # Touch shard (reactivates if COLD)
            shard = self.storage.access(did)
            chunks: List[ChunkResult] = []
            if shard is not None:
                qs = QueryStore(shard)
                q_emb = latent.mean(dim=0).detach().cpu().numpy().astype("float32")
                chunks = qs.retrieve(q_emb, query_text=query_text, top_k=rk)

            ret_ms = (time.perf_counter() - t_ret) * 1000

            # Head forward (embedding projection)
            head = self.registry.get(did)
            head_out: Optional[Tensor] = None
            if head is not None:
                with torch.no_grad():
                    head_out = head(latent)

            domain_results.append(DomainResult(
                domain_id=did,
                domain_name=node.name if node else did,
                gate_score=gscore,
                chunks=chunks,
                head_output=head_out,
                retrieval_ms=ret_ms,
            ))

            self._emit(EVT_DOMAIN_ACTIVATE, domain_id=did,
                       payload={"gate_score": gscore, "chunks_retrieved": len(chunks)})

        elapsed = (time.perf_counter() - t0) * 1000
        ev = self._emit("pipeline_query_complete",
                        payload={"elapsed_ms": elapsed, "domains_activated": len(domain_results)})

        return PipelineResult(
            query_text=query_text,
            latent=latent,
            gate_decision=gate_dec,
            domain_results=domain_results,
            event_id=ev.event_id,
            elapsed_ms=elapsed,
        )

    # ---- Diagnostics -------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        return {
            "gate_state":      self.gate.state.value,
            "active_domains":  self.gate.active_domain_count(),
            "domain_states":   self.storage.status(),
            "event_log_size":  len(self._event_log),
            "encoder_frozen":  self.encoder.is_frozen,
        }
