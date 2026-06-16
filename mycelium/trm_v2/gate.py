"""
mycelium/trm_v2/gate.py

GatingNetwork — routes a query latent [B, D] to the top-k domain heads.

Architecture:
    query latent [B, D]
    -> GateProjection: Linear(D -> gate_dim) + GELU + LayerNorm
    -> domain score heads: one Linear(gate_dim -> 1) per registered domain
    -> softmax over active domains -> top-k selection

GateState machine (from types.py):
    BOOTSTRAP     — fewer than min_domains registered; random/uniform routing
    EXPANSION     — new domains being added; gate retrains on expanded set
    STABLE        — normal operation; routing is deterministic top-k softmax
    THAWING       — drift detected; gate projection is temporarily unfrozen
    RECALIBRATING — local recalibration running; new routing weights settling

Hard rules enforced here:
    - Global retraining is FORBIDDEN during THAWING / RECALIBRATING.
      Only the affected domain score head(s) may be updated.
    - Routing decisions are logged as DomainEvent(EVT_DOMAIN_ACTIVATE)
      by the caller (pipeline.py), not here. Gate is stateless per call.
    - Temperature scaling is supported for calibration (Addition D).

Spec refs:
    mycelium_trm_v2_theoretical_spec_v0.2.md  §GatingNetwork
    mycelium_trm_v2_theoretical_spec_hardening_additions.md  Additions B, D
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mycelium.trm_v2.types import GateState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class GateDecision:
    """Output of a single GatingNetwork forward call.

    domain_ids  : ordered list of selected domain ids (highest score first)
    scores      : softmax-normalised confidence for each selected domain
    all_scores  : full score dict over ALL active domains (for diagnostics)
    gate_state  : the GateState at the time of the decision
    temperature : temperature used for this call
    """
    domain_ids:  List[str]
    scores:      List[float]
    all_scores:  Dict[str, float]
    gate_state:  GateState
    temperature: float = 1.0


# ---------------------------------------------------------------------------
# GateProjection — shared query encoder for the gate
# ---------------------------------------------------------------------------

class _GateProjection(nn.Module):
    """Projects the shared encoder latent into gate space."""

    def __init__(self, input_dim: int, gate_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj    = nn.Linear(input_dim, gate_dim, bias=True)
        self.act     = nn.GELU()
        self.norm    = nn.LayerNorm(gate_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.norm(self.act(self.proj(x))))


# ---------------------------------------------------------------------------
# GatingNetwork
# ---------------------------------------------------------------------------

class GatingNetwork(nn.Module):
    """Routes query latents to the top-k most relevant domain heads.

    Parameters
    ----------
    input_dim       : int   must equal SharedEncoder.hidden_size
    gate_dim        : int   internal gate projection width
    top_k           : int   number of domains to activate per query
    min_domains     : int   minimum registered domains before STABLE routing
    temperature     : float initial softmax temperature (calibration param)
    dropout         : float
    """

    def __init__(
        self,
        input_dim:    int   = 256,
        gate_dim:     int   = 64,
        top_k:        int   = 2,
        min_domains:  int   = 2,
        temperature:  float = 1.0,
        dropout:      float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim   = input_dim
        self.gate_dim    = gate_dim
        self.top_k       = top_k
        self.min_domains = min_domains
        self.temperature = temperature

        self._state: GateState = GateState.BOOTSTRAP

        self.projection = _GateProjection(input_dim, gate_dim, dropout)

        # One score head per domain — stored in a ModuleDict so they are
        # tracked by PyTorch and included in state_dict().
        self._score_heads: nn.ModuleDict = nn.ModuleDict()

        # Ordered list of active domain ids (determines softmax slice order)
        self._active_domains: List[str] = []

    # ---- State management --------------------------------------------------

    @property
    def state(self) -> GateState:
        return self._state

    def _recompute_state(self) -> None:
        n = len(self._active_domains)
        if n < self.min_domains:
            self._state = GateState.BOOTSTRAP
        elif self._state == GateState.BOOTSTRAP and n >= self.min_domains:
            self._state = GateState.EXPANSION

    # ---- Domain registration -----------------------------------------------

    def register_domain(self, domain_id: str) -> None:
        """Add a new score head for domain_id. Idempotent."""
        if domain_id in self._score_heads:
            logger.debug("GatingNetwork: domain '%s' already registered.", domain_id)
            return
        head = nn.Linear(self.gate_dim, 1, bias=True)
        nn.init.xavier_uniform_(head.weight)
        nn.init.zeros_(head.bias)
        self._score_heads[domain_id] = head
        self._active_domains.append(domain_id)
        self._recompute_state()
        logger.debug(
            "GatingNetwork: registered domain '%s'. Active=%d state=%s",
            domain_id, len(self._active_domains), self._state.value,
        )

    def unregister_domain(self, domain_id: str) -> None:
        """Remove a domain score head (called on DEPRECATED/ARCHIVED)."""
        if domain_id not in self._score_heads:
            return
        del self._score_heads[domain_id]
        self._active_domains = [d for d in self._active_domains if d != domain_id]
        self._recompute_state()
        logger.debug(
            "GatingNetwork: unregistered domain '%s'. Active=%d state=%s",
            domain_id, len(self._active_domains), self._state.value,
        )

    def mark_stable(self) -> None:
        """Manually advance state to STABLE (call after initial training)."""
        if self._state in (GateState.EXPANSION, GateState.BOOTSTRAP):
            if len(self._active_domains) >= self.min_domains:
                self._state = GateState.STABLE

    def begin_recalibration(self, domain_ids: Optional[List[str]] = None) -> None:
        """Enter THAWING for local recalibration of specified domain heads.

        Hard rule: if domain_ids is None, ALL heads would be touched —
        that counts as global retraining and is rejected.
        """
        if domain_ids is None:
            raise ValueError(
                "GatingNetwork.begin_recalibration: domain_ids=None would trigger "
                "global retraining, which is forbidden. Pass a specific list."
            )
        self._state = GateState.THAWING
        # Freeze everything, then unfreeze only the affected score heads
        for p in self.parameters():
            p.requires_grad_(False)
        for did in domain_ids:
            if did in self._score_heads:
                for p in self._score_heads[did].parameters():
                    p.requires_grad_(True)
        logger.info(
            "GatingNetwork: THAWING started. Local recalibration targets: %s", domain_ids
        )

    def finish_recalibration(self) -> None:
        """Exit RECALIBRATING -> STABLE. Re-freeze all heads."""
        for p in self.parameters():
            p.requires_grad_(False)
        self._state = GateState.STABLE
        logger.info("GatingNetwork: recalibration complete. State -> STABLE.")

    # ---- Temperature calibration (Addition D) ------------------------------

    def set_temperature(self, temperature: float) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = temperature

    # ---- Forward -----------------------------------------------------------

    def forward(
        self,
        latent: Tensor,
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> GateDecision:
        """
        Parameters
        ----------
        latent      : [B, D]  from SharedEncoder (B=1 for inference)
        top_k       : override instance top_k for this call
        temperature : override instance temperature for this call

        Returns
        -------
        GateDecision  with domain_ids and scores for the top-k domains.

        BOOTSTRAP behaviour: returns uniform scores over all registered domains
        (or an empty decision if no domains registered yet).
        """
        k    = top_k if top_k is not None else self.top_k
        temp = temperature if temperature is not None else self.temperature
        n    = len(self._active_domains)

        if n == 0:
            return GateDecision(
                domain_ids=[],
                scores=[],
                all_scores={},
                gate_state=self._state,
                temperature=temp,
            )

        # BOOTSTRAP: uniform routing (gate projection not yet reliable)
        if self._state == GateState.BOOTSTRAP:
            uniform = 1.0 / n
            selected = self._active_domains[:k]
            return GateDecision(
                domain_ids=selected,
                scores=[uniform] * len(selected),
                all_scores={d: uniform for d in self._active_domains},
                gate_state=self._state,
                temperature=temp,
            )

        # Project query into gate space — use first item in batch (B=1 at inference)
        # For training B>1 we take the mean to produce a single routing decision
        # (domain routing is per-query, not per-batch-item in the current spec).
        q = latent.mean(dim=0, keepdim=True)  # [1, D]
        gate_q = self.projection(q)            # [1, gate_dim]

        # Score each active domain
        raw_scores: Dict[str, float] = {}
        score_tensor = []
        for did in self._active_domains:
            s = self._score_heads[did](gate_q).squeeze()  # scalar
            score_tensor.append(s)
            raw_scores[did] = s.item()

        logits = torch.stack(score_tensor)                    # [n]
        probs  = F.softmax(logits / temp, dim=0)              # [n]

        all_scores = {
            did: probs[i].item()
            for i, did in enumerate(self._active_domains)
        }

        # Top-k selection
        k_actual = min(k, n)
        top_indices = probs.topk(k_actual).indices.tolist()
        selected_ids    = [self._active_domains[i] for i in top_indices]
        selected_scores = [all_scores[did] for did in selected_ids]

        return GateDecision(
            domain_ids=selected_ids,
            scores=selected_scores,
            all_scores=all_scores,
            gate_state=self._state,
            temperature=temp,
        )

    # ---- Diagnostics -------------------------------------------------------

    def active_domain_count(self) -> int:
        return len(self._active_domains)

    def score_head_param_count(self) -> Dict[str, int]:
        return {did: sum(p.numel() for p in h.parameters())
                for did, h in self._score_heads.items()}  # type: ignore
