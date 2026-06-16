"""
mycelium/trm_v2/shared_encoder.py

SharedEncoder — frozen backbone that produces a fixed-dimensional latent
representation shared across ALL domains.

DomainHead — per-domain trainable projection + optional classifier head
that sits on top of the SharedEncoder output. Only this head is ever
trained when a new domain is added, keeping the encoder weights frozen
and preventing cross-domain interference.

Architecture:
    SharedEncoder
        input tokens  [B, L]
        -> token embedding  [B, L, D]
        -> positional encoding (sinusoidal, no learned params)
        -> N x TRMCell (2-layer, RMSNorm, RoPE, SwiGLU, no bias)
        -> mean-pool over sequence  -> [B, D]   (the shared latent)

    DomainHead (one per domain)
        shared latent  [B, D]
        -> LayerNorm
        -> Linear(D -> head_dim)   <- trainable
        -> GELU
        -> Linear(head_dim -> num_classes) if num_classes > 0  <- trainable
        (or just the projection if num_classes == 0 for embedding-only mode)

Freezing contract:
    - SharedEncoder parameters are frozen at construction by default.
      Call encoder.unfreeze() ONLY for initial pre-training; re-freeze
      with encoder.freeze() before any domain-head training.
    - DomainHead parameters are always trainable (they start from random init).

Spec refs:
    mycelium_trm_v2_theoretical_spec_v0.2.md  §Shared Frozen Encoder
    mycelium_trm_v2_theoretical_spec_hardening_additions.md  Addition C
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Re-use the battle-tested primitives from the v1 TRMCell
from mycelium.trm.network import RMSNorm, TRMCell


# ---------------------------------------------------------------------------
# Sinusoidal positional encoding (no learned params — safe to freeze)
# ---------------------------------------------------------------------------

class _SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding.

    Using a fixed (non-learned) PE means the SharedEncoder has zero
    extra parameters for positional information, keeping the frozen
    backbone truly parameter-stable across domain additions.
    """

    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # [1, max_len, D]

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, L, D] — add positional encoding in-place."""
        return x + self.pe[:, : x.size(1), :]


# ---------------------------------------------------------------------------
# SharedEncoder
# ---------------------------------------------------------------------------

class SharedEncoder(nn.Module):
    """Frozen backbone shared by all domain heads.

    Parameters
    ----------
    vocab_size  : int   vocabulary size (token embedding table)
    hidden_size : int   model width (D)
    n_layers    : int   number of TRMCell blocks
    n_heads     : int   attention heads per block
    max_seq_len : int   maximum input sequence length
    pad_token_id: int   token id used for padding (masked out in mean-pool)

    Typical small-footprint config (matching TRM paper ~7 M params):
        vocab_size=32_000, hidden_size=256, n_layers=2, n_heads=4
    """

    def __init__(
        self,
        vocab_size:   int = 32_000,
        hidden_size:  int = 256,
        n_layers:     int = 2,
        n_heads:      int = 4,
        max_seq_len:  int = 512,
        pad_token_id: int = 0,
        ffn_expansion: int = 4,
        frozen:       bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size  = hidden_size
        self.pad_token_id = pad_token_id

        self.token_emb = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.pos_enc   = _SinusoidalPE(max_seq_len, hidden_size)
        self.cell      = TRMCell(
            hidden_size=hidden_size,
            n_layers=n_layers,
            n_heads=n_heads,
            ffn_expansion=ffn_expansion,
            use_rope=True,
        )
        self.out_norm = RMSNorm(hidden_size)

        if frozen:
            self.freeze()

    # ---- Freezing API ------------------------------------------------------

    def freeze(self) -> None:
        """Freeze all encoder parameters. Call after pre-training."""
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze(self) -> None:
        """Temporarily unfreeze for pre-training. Re-freeze afterwards."""
        for p in self.parameters():
            p.requires_grad_(True)

    @property
    def is_frozen(self) -> bool:
        return not any(p.requires_grad for p in self.parameters())

    # ---- Forward -----------------------------------------------------------

    def forward(
        self,
        input_ids:      Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        input_ids      : [B, L]  integer token ids
        attention_mask : [B, L]  1 = real token, 0 = padding (optional)

        Returns
        -------
        latent : [B, D]  mean-pooled sequence representation
        """
        x = self.token_emb(input_ids)           # [B, L, D]
        x = self.pos_enc(x)                     # [B, L, D]
        x = self.cell(q=x, context=None)        # [B, L, D]  self-attn only
        x = self.out_norm(x)                    # [B, L, D]

        # Mean-pool, masking out padding tokens
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()   # [B, L, 1]
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            x = x.mean(dim=1)

        return x  # [B, D]

    def encode_embeddings(self, embeddings: Tensor) -> Tensor:
        """
        Alternative entry point: accepts pre-computed dense embeddings
        [B, L, D_in] and projects them through the TRMCell backbone.

        Useful when plugging in sentence-transformer embeddings directly
        (e.g. from the existing mycelium.trm.embeddings pipeline) without
        needing a tokenizer.

        embeddings : [B, L, D]  where D == self.hidden_size
        Returns    : [B, D]     mean-pooled latent
        """
        if embeddings.dim() == 2:
            # [B, D] -> treat as single-token sequence [B, 1, D]
            embeddings = embeddings.unsqueeze(1)
        x = self.pos_enc(embeddings)
        x = self.cell(q=x, context=None)
        x = self.out_norm(x)
        return x.mean(dim=1)


# ---------------------------------------------------------------------------
# DomainHead
# ---------------------------------------------------------------------------

class DomainHead(nn.Module):
    """Per-domain trainable head.

    Sits on top of SharedEncoder output. Only these weights are trained
    when a new domain is registered or fine-tuned.

    Parameters
    ----------
    domain_id   : str   matches DomainNode.domain_id
    input_dim   : int   must equal SharedEncoder.hidden_size
    head_dim    : int   internal projection width
    num_classes : int   >0 for classification, 0 for embedding-only
    dropout     : float
    """

    def __init__(
        self,
        domain_id:   str,
        input_dim:   int = 256,
        head_dim:    int = 128,
        num_classes: int = 0,
        dropout:     float = 0.1,
    ) -> None:
        super().__init__()
        self.domain_id   = domain_id
        self.input_dim   = input_dim
        self.head_dim    = head_dim
        self.num_classes = num_classes

        self.norm     = nn.LayerNorm(input_dim)
        self.proj     = nn.Linear(input_dim, head_dim, bias=True)
        self.act      = nn.GELU()
        self.dropout  = nn.Dropout(dropout)

        self.classifier: Optional[nn.Linear] = None
        if num_classes > 0:
            self.classifier = nn.Linear(head_dim, num_classes, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        if self.classifier is not None:
            nn.init.xavier_uniform_(self.classifier.weight)
            nn.init.zeros_(self.classifier.bias)

    def forward(self, latent: Tensor) -> Tensor:
        """
        latent : [B, D]  from SharedEncoder

        Returns
        -------
        If num_classes > 0 : logits [B, num_classes]
        Else               : projection embedding [B, head_dim]
        """
        x = self.norm(latent)
        x = self.proj(x)
        x = self.act(x)
        x = self.dropout(x)
        if self.classifier is not None:
            x = self.classifier(x)
        return x

    def embedding(self, latent: Tensor) -> Tensor:
        """Always return the head_dim projection (before classifier).
        Useful for retrieval/similarity even when num_classes > 0.
        """
        x = self.norm(latent)
        x = self.proj(x)
        x = self.act(x)
        return x

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# HeadRegistry — in-memory map of domain_id -> DomainHead
# ---------------------------------------------------------------------------

class HeadRegistry(nn.Module):
    """Manages all active DomainHead instances.

    Kept as an nn.Module so heads can be saved/loaded as part of a
    single checkpoint via state_dict().

    Usage:
        registry = HeadRegistry()
        registry.register("dom-uuid-123", DomainHead("dom-uuid-123", ...))
        head = registry.get("dom-uuid-123")
        registry.remove("dom-uuid-123")  # archive path only — head is returned
    """

    def __init__(self) -> None:
        super().__init__()
        self._heads: nn.ModuleDict = nn.ModuleDict()

    def register(self, domain_id: str, head: DomainHead) -> None:
        if domain_id in self._heads:
            raise KeyError(f"HeadRegistry: '{domain_id}' already registered.")
        self._heads[domain_id] = head

    def get(self, domain_id: str) -> Optional[DomainHead]:
        if domain_id not in self._heads:
            return None
        return self._heads[domain_id]  # type: ignore[return-value]

    def remove(self, domain_id: str) -> DomainHead:
        """Remove and return the head (for archival). Raises if not found."""
        if domain_id not in self._heads:
            raise KeyError(f"HeadRegistry: '{domain_id}' not found.")
        head = self._heads[domain_id]
        del self._heads._modules[domain_id]
        return head  # type: ignore[return-value]

    def all_domain_ids(self) -> list:
        return list(self._heads.keys())

    def __len__(self) -> int:
        return len(self._heads)

    def total_head_params(self) -> Dict[str, int]:
        return {did: h.param_count() for did, h in self._heads.items()}  # type: ignore
