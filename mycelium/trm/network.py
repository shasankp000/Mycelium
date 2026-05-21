"""
Phase D — TRMCell
==================
The single reusable 2-layer transformer block used for BOTH the latent
update and the answer update steps in the Samsung TRM algorithm
(arXiv:2510.04871, §4.3).

Architecture per layer:
    RMSNorm → Multi-head self-attention (RoPE) → residual add
    RMSNorm → SwiGLU FFN                        → residual add

The presence or absence of an external conditioning tensor `context` in the
forward call is the only structural difference between the two TRM operations:
    latent update:  TRMCell(q = z,   context = x + y)   # x IS present
    answer update:  TRMCell(q = y+z, context = None)     # x absent

Design notes:
    - No bias parameters anywhere (PaLM convention, §2.1 of the paper).
    - Weight tying between the latent-update and answer-update calls is NOT
      applied here; TRMReasoner uses one shared TRMCell instance for both,
      which achieves weight-sharing automatically (§4.3 "single network").
    - RoPE is applied inside each attention layer independently.
    - SwiGLU uses a single weight matrix gated with a sigmoid (simpler than
      the original two-matrix form but equivalent in expressivity for small D).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# --------------------------------------------------------------------------- #
# Utility: RMSNorm                                                             #
# --------------------------------------------------------------------------- #

class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation (no bias, no mean subtraction)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


# --------------------------------------------------------------------------- #
# Utility: Rotary Position Embeddings (RoPE)                                  #
# --------------------------------------------------------------------------- #

def _build_rope_cache(seq_len: int, head_dim: int, device: torch.device) -> Tensor:
    """Returns cos/sin cache of shape [seq_len, head_dim/2]."""
    theta = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, theta)          # [seq_len, head_dim/2]
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)  # [seq_len, head_dim]


def _apply_rope(x: Tensor, rope: Tensor) -> Tensor:
    """x: [B, heads, seq, head_dim]; rope: [seq, head_dim]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = rope[..., :half].unsqueeze(0).unsqueeze(0)
    sin = rope[..., half:].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# --------------------------------------------------------------------------- #
# Attention layer                                                              #
# --------------------------------------------------------------------------- #

class _Attention(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, use_rope: bool = True) -> None:
        super().__init__()
        assert hidden_size % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.use_rope = use_rope
        # No bias (PaLM convention)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        """
        Self-attention when context is None; cross-attention otherwise.
        x       : [B, L_q, D]
        context : [B, L_kv, D] or None
        """
        B, L_q, _ = x.shape
        kv_src = context if context is not None else x
        L_kv = kv_src.shape[1]

        q = self.q_proj(x).view(B, L_q,  self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_src).view(B, L_kv, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_src).view(B, L_kv, self.n_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            rope_q = _build_rope_cache(L_q,  self.head_dim, x.device)
            rope_k = _build_rope_cache(L_kv, self.head_dim, x.device)
            q = _apply_rope(q, rope_q)
            k = _apply_rope(k, rope_k)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)                # [B, heads, L_q, head_dim]
        out = out.transpose(1, 2).reshape(B, L_q, -1)
        return self.o_proj(out)


# --------------------------------------------------------------------------- #
# SwiGLU FFN                                                                  #
# --------------------------------------------------------------------------- #

class _SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: int = 4) -> None:
        super().__init__()
        inner = hidden_size * expansion
        self.gate = nn.Linear(hidden_size, inner, bias=False)
        self.up   = nn.Linear(hidden_size, inner, bias=False)
        self.down = nn.Linear(inner, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


# --------------------------------------------------------------------------- #
# TRMCell — one transformer layer stack (n_layers deep)                       #
# --------------------------------------------------------------------------- #

class TRMCell(nn.Module):
    """
    Single reusable cell used for both latent update and answer update.

    Call signatures:
        latent update : cell(q=z,   context=x_plus_y)  # cross-attends to x+y
        answer update : cell(q=y_z, context=None)       # self-attention only

    where x_plus_y = x + y (both projected to hidden_size beforehand).

    Parameters
    ----------
    hidden_size : int
    n_layers    : int   number of (Attn + FFN) blocks stacked
    n_heads     : int
    ffn_expansion : int  SwiGLU inner dim multiplier
    use_rope    : bool
    """

    def __init__(
        self,
        hidden_size: int,
        n_layers: int = 2,
        n_heads: int = 8,
        ffn_expansion: int = 4,
        use_rope: bool = True,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm1": RMSNorm(hidden_size),
                "attn":  _Attention(hidden_size, n_heads, use_rope=use_rope),
                "norm2": RMSNorm(hidden_size),
                "ffn":   _SwiGLU(hidden_size, ffn_expansion),
            })
            for _ in range(n_layers)
        ])

    def forward(self, q: Tensor, context: Optional[Tensor] = None) -> Tensor:
        """
        q       : [B, L, D]  — primary sequence (z or y+z)
        context : [B, L, D] or None  — cross-attention source (x+y) or None
        """
        x = q
        for layer in self.layers:
            # Attention (self or cross)
            residual = x
            x = layer["norm1"](x)
            x = layer["attn"](x, context=context)
            x = x + residual
            # FFN
            residual = x
            x = layer["norm2"](x)
            x = layer["ffn"](x)
            x = x + residual
        return x
