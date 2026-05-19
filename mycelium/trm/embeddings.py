"""
Phase D — QueryEmbedder & AnswerEmbedder
=========================================
Converts Phase B/C output (canonical query tokens + spectral signature
+ predicate family) into the TRM input tensors x and y.

Input pipeline
--------------
  QueryEmbedder produces x : [B, L + n_domains + 1, D]
      - token_ids  : canonical query token IDs from CanonicalizeAndHash
                     (padded / truncated to context_len)
      - spectral_vec: float[n_domains] from RuntimeSpectralAnalyzer
                     appended as n_domains "virtual tokens" after text tokens
      - predicate_family_id: int from predicate_families.py
                     appended as a single "type token" at the end

  AnswerEmbedder produces y : [B, n_domains, D]
      - initial_domain_probs: float[n_domains] fused scores from
                              MultiLensRouter, projected per-domain
                              into hidden_size embedding space.

Design notes
------------
    Spectral scores and predicate-family ID are "virtual tokens" placed
    after the text sequence.  This lets TRMCell's self-attention
    cross-attend between query semantics and routing-relevant features
    without any architectural changes (same attention over all tokens).

    A learned SpectralTypeEmbedding (one vector per domain index) is
    added to each spectral virtual token so the model can distinguish
    domain-0 score from domain-1 score etc.

    The AnswerEmbedder does NOT use positional encodings — the n_domains
    slots in y are addressed by their intrinsic type embedding only,
    matching the paper's treatment of y as a set not a sequence.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class QueryEmbedder(nn.Module):
    """
    Maps (token_ids, spectral_vec, predicate_family_id) → x [B, L_total, D].

    L_total = context_len + n_domains + 1
                ↑ text        ↑ spectral  ↑ predicate-family

    Parameters
    ----------
    vocab_size           : int   canonical query vocabulary size
    hidden_size          : int   D
    n_domains            : int   number of routing expert domains
    context_len          : int   max text token length (L)
    n_predicate_families : int   number of SRL predicate families
    pad_token_id         : int   used for padding; default 0
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        n_domains: int,
        context_len: int,
        n_predicate_families: int = 32,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.context_len = context_len
        self.n_domains = n_domains
        self.pad_token_id = pad_token_id

        # Text token embedding + positional
        self.token_embed = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.pos_embed   = nn.Embedding(context_len, hidden_size)

        # Spectral virtual tokens: one learned type-vector per domain
        self.spectral_type_embed = nn.Embedding(n_domains, hidden_size)
        # Project scalar spectral score → D
        self.spectral_proj = nn.Linear(1, hidden_size, bias=False)

        # Predicate-family virtual token
        self.predicate_embed = nn.Embedding(n_predicate_families, hidden_size)

        # Segment IDs: 0=text, 1=spectral, 2=predicate
        self.segment_embed = nn.Embedding(3, hidden_size)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight,   std=0.02)
        nn.init.normal_(self.spectral_type_embed.weight, std=0.02)
        nn.init.normal_(self.predicate_embed.weight,     std=0.02)
        nn.init.normal_(self.segment_embed.weight,       std=0.02)
        nn.init.xavier_uniform_(self.spectral_proj.weight)

    def forward(
        self,
        token_ids: Tensor,          # [B, context_len]  int64
        spectral_vec: Tensor,       # [B, n_domains]     float32
        predicate_family_id: Tensor,# [B]                int64
    ) -> Tensor:
        """
        Returns x : [B, context_len + n_domains + 1, hidden_size]
        """
        B = token_ids.shape[0]
        device = token_ids.device

        # --- text tokens ---
        positions = torch.arange(self.context_len, device=device).unsqueeze(0)  # [1, L]
        text = (
            self.token_embed(token_ids)
            + self.pos_embed(positions)
            + self.segment_embed(torch.zeros(B, self.context_len, dtype=torch.long, device=device))
        )  # [B, L, D]

        # --- spectral virtual tokens ---
        domain_ids = torch.arange(self.n_domains, device=device).unsqueeze(0).expand(B, -1)  # [B, n_domains]
        spectral_type = self.spectral_type_embed(domain_ids)  # [B, n_domains, D]
        spectral_val  = self.spectral_proj(spectral_vec.unsqueeze(-1))  # [B, n_domains, D]
        spectral_seg  = self.segment_embed(
            torch.ones(B, self.n_domains, dtype=torch.long, device=device)
        )  # [B, n_domains, D]
        spectral = spectral_type + spectral_val + spectral_seg  # [B, n_domains, D]

        # --- predicate-family virtual token ---
        pred = (
            self.predicate_embed(predicate_family_id).unsqueeze(1)
            + self.segment_embed(torch.full((B, 1), 2, dtype=torch.long, device=device))
        )  # [B, 1, D]

        # Concatenate: [text | spectral | predicate]
        return torch.cat([text, spectral, pred], dim=1)  # [B, L+n_domains+1, D]


class AnswerEmbedder(nn.Module):
    """
    Maps initial_domain_probs → y [B, n_domains, D].

    Each of the n_domains slots gets:
        domain_type_embed[d]  +  linear_proj(prob[d])

    This gives the model a per-domain identity anchor that survives
    the iterative latent recursion without collapsing to a uniform vector.

    Parameters
    ----------
    hidden_size : int
    n_domains   : int
    """

    def __init__(self, hidden_size: int, n_domains: int) -> None:
        super().__init__()
        self.n_domains = n_domains
        self.domain_type_embed = nn.Embedding(n_domains, hidden_size)
        self.prob_proj = nn.Linear(1, hidden_size, bias=False)

        nn.init.normal_(self.domain_type_embed.weight, std=0.02)
        nn.init.xavier_uniform_(self.prob_proj.weight)

    def forward(self, initial_domain_probs: Tensor) -> Tensor:
        """
        initial_domain_probs : [B, n_domains]  float32, sums to ~1
        Returns y            : [B, n_domains, hidden_size]
        """
        B = initial_domain_probs.shape[0]
        device = initial_domain_probs.device
        domain_ids = torch.arange(self.n_domains, device=device).unsqueeze(0).expand(B, -1)
        type_emb = self.domain_type_embed(domain_ids)           # [B, n_domains, D]
        prob_emb = self.prob_proj(initial_domain_probs.unsqueeze(-1))  # [B, n_domains, D]
        return type_emb + prob_emb
