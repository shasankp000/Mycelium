"""
Phase D — TRMReasoner
======================
Full Samsung TRM neural reasoner adapted for Mycelium's domain-routing
task (arXiv:2510.04871).

Algorithm (matches §4 of the paper exactly)
--------------------------------------------
Inputs:
    x  : [B, L_total, D]  — QueryEmbedder output  (query context)
    y0 : [B, n_domains, D] — AnswerEmbedder output  (initial domain probs)

Procedure:
    z  ← zeros_like(y0)
    for step = 1 … N_sup:
        # T-1 no-grad supervision passes ("look-ahead" stabilisation)
        with no_grad:
            for _ in range(T - 1):
                z ← cell(q=z,   context=x+y)   # latent update
                y ← cell(q=y+z, context=None)   # answer update
        # one gradient-enabled pass
        z ← cell(q=z,   context=x+y)
        y ← cell(q=y+z, context=None)

        logits ← domain_head(y)            # [B, n_domains]
        halt   ← halt_head(y)              # [B]

        if sigmoid(halt).all() > threshold:
            break

    return TRMOutput(logits, probs, halt, step, y, z)

Key design decisions (all paper-justified):
    - Single TRMCell instance used for BOTH latent and answer updates
      (§4.3: one network beats two, 87.4% vs 85.2%).
    - Separate y and z tensors (§4.2: merged single-z → 71.9%).
    - No IFT / fixed-point assumption — backprop through full recursion
      (§4.1: IFT degrades 87.4% → 56.5%).
    - Simplified ACT BCE halt (§4.6: removes second forward pass).
    - EMA applied by TRMTrainer, not here (§4.7).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .config import TRMConfig
from .network import TRMCell
from .embeddings import QueryEmbedder, AnswerEmbedder
from .output_head import DomainHead, HaltHead

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class TRMOutput:
    """Structured return value from TRMReasoner.forward()."""

    domain_logits: Tensor
    """[B, n_domains]  raw logits (pre-softmax)."""

    domain_probs: Tensor
    """[B, n_domains]  softmax probabilities."""

    halt_confidence: Tensor
    """[B]  sigmoid(halt_logit) — proxy for answer stability."""

    n_steps_taken: int
    """Number of outer supervision steps actually executed."""

    primary_domain_idx: Tensor
    """[B]  argmax over domain_probs."""

    final_y: Tensor
    """[B, n_domains, D]  final answer state (detached)."""

    final_z: Tensor
    """[B, L_total, D]  final latent state (detached)."""


class TRMReasoner(nn.Module):
    """
    Drop-in replacement for the local LLM in the TRM layer.

    Parameters
    ----------
    cfg : TRMConfig
    vocab_size : int   canonical query vocabulary size for QueryEmbedder
    n_predicate_families : int  number of SRL predicate families
    """

    def __init__(
        self,
        cfg: TRMConfig,
        vocab_size: int = 8192,
        n_predicate_families: int = 32,
    ) -> None:
        super().__init__()
        self.cfg = cfg

        # Embedders
        self.query_embedder = QueryEmbedder(
            vocab_size=vocab_size,
            hidden_size=cfg.hidden_size,
            n_domains=cfg.n_domains,
            context_len=cfg.context_len,
            n_predicate_families=n_predicate_families,
        )
        self.answer_embedder = AnswerEmbedder(
            hidden_size=cfg.hidden_size,
            n_domains=cfg.n_domains,
        )

        # Single shared cell (paper §4.3: one network beats two)
        self.cell = TRMCell(
            hidden_size=cfg.hidden_size,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            ffn_expansion=cfg.ffn_expansion,
            use_rope=cfg.use_rope,
        )

        # Output heads
        self.domain_head = DomainHead(cfg.hidden_size, cfg.n_domains)
        self.halt_head   = HaltHead(cfg.hidden_size)

    # ------------------------------------------------------------------ #
    # Inner recursion                                                      #
    # ------------------------------------------------------------------ #

    def _latent_recursion(
        self,
        x: Tensor,
        y: Tensor,
        z: Tensor,
        n: int,
    ) -> tuple[Tensor, Tensor]:
        """
        Run n inner latent steps then one answer update.

        Latent update  (§4.3): z ← cell(q=z,   context=x+y)
            The cell cross-attends z against x+y so the latent state
            integrates both query information and current answer belief.

        Answer update  (§4.3): y ← cell(q=y+z, context=None)
            Self-attention only — x is excluded so the answer refines
            itself using only what the latent state has distilled.

        Returns updated (y, z).
        """
        # x has shape [B, L_total, D]; y and z have shape [B, n_domains, D].
        # For cross-attention, x+y need compatible sequence lengths.
        # We broadcast x (L_total) and y (n_domains) separately as context.
        # The paper concatenates them: context = cat([x, y], dim=1).
        context = torch.cat([x, y], dim=1)  # [B, L_total + n_domains, D]

        for _ in range(n):
            z = self.cell(q=z, context=context)        # latent update
            context = torch.cat([x, y], dim=1)         # recompute after y update below

        y = self.cell(q=y + z, context=None)           # answer update (self-attn only)
        return y, z

    # ------------------------------------------------------------------ #
    # Forward pass                                                         #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        token_ids: Tensor,           # [B, context_len]
        spectral_vec: Tensor,        # [B, n_domains]
        predicate_family_id: Tensor, # [B]
        initial_domain_probs: Tensor,# [B, n_domains]
    ) -> TRMOutput:
        """
        Full TRM forward pass with deep supervision loop.

        At training time this method is called once per batch; the outer
        supervision loop runs T times with gradients on the last pass only
        (§4 deep supervision).  At inference time, it runs until halt or
        max_supervision_steps, whichever comes first.

        Returns TRMOutput.
        """
        cfg = self.cfg
        device = token_ids.device

        # Build x and y0
        x  = self.query_embedder(token_ids, spectral_vec, predicate_family_id)
        y  = self.answer_embedder(initial_domain_probs)
        z  = torch.zeros_like(y)  # [B, n_domains, D]

        # --- Deep supervision outer loop ---
        training = self.training
        n_steps = cfg.n_supervision if training else cfg.max_supervision_steps
        T = cfg.n_supervision
        last_logits: Optional[Tensor] = None
        last_halt:   Optional[Tensor] = None

        for step in range(n_steps):
            if training:
                # T-1 no-grad stabilisation passes
                if T > 1:
                    with torch.no_grad():
                        for _ in range(T - 1):
                            y, z = self._latent_recursion(x, y, z, cfg.n_recursions)
                # One gradient-enabled pass
                y, z = self._latent_recursion(x, y, z, cfg.n_recursions)
            else:
                # Inference: T passes, all no-grad
                with torch.no_grad():
                    for _ in range(T):
                        y, z = self._latent_recursion(x, y, z, cfg.n_recursions)

            last_logits = self.domain_head(y)   # [B, n_domains]
            last_halt   = self.halt_head(y)     # [B]

            # Halt check (inference only — never halt during training)
            if not training:
                halt_prob = last_halt.sigmoid()
                if halt_prob.min().item() > cfg.halt_threshold:
                    logger.debug(
                        "TRMReasoner: halted at step %d (min_halt=%.3f)",
                        step + 1, halt_prob.min().item(),
                    )
                    break

        assert last_logits is not None and last_halt is not None

        probs = last_logits.softmax(dim=-1)
        return TRMOutput(
            domain_logits=last_logits,
            domain_probs=probs,
            halt_confidence=last_halt.sigmoid(),
            n_steps_taken=step + 1,
            primary_domain_idx=probs.argmax(dim=-1),
            final_y=y.detach(),
            final_z=z.detach(),
        )
