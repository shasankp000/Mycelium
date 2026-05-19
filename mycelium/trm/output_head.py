"""
Phase D — DomainHead & HaltHead
================================
Output projection heads for TRMReasoner.

DomainHead
----------
Projects the answer state y [B, n_domains, D] → domain logits [B, n_domains]
by applying a per-domain linear probe then squeezing the D dimension.
Softmax is NOT applied here; callers use log_softmax for stable CE loss.

HaltHead
--------
Simplified ACT halting signal (§4.6 of arXiv:2510.04871).
Projects the mean-pooled answer state → scalar halt logit.
Trained with BCE against the binary signal (argmax(y_pred) == y_true).

This removes the expensive second forward pass required by the full
HRM Q-learning ACT formulation while achieving equivalent performance
(87.4% with simplified vs 86.1% with continue-loss, §4.6).
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class DomainHead(nn.Module):
    """
    y [B, n_domains, D] → logits [B, n_domains]

    Uses a weight matrix of shape [n_domains, D] applied as a batched
    dot-product: logits[b, d] = dot(W[d], y[b, d]).
    """

    def __init__(self, hidden_size: int, n_domains: int) -> None:
        super().__init__()
        # One probe vector per domain — no shared projection matrix
        self.probe = nn.Parameter(
            nn.init.xavier_uniform_(
                __import__('torch').empty(n_domains, hidden_size)
            )
        )
        self.bias = nn.Parameter(__import__('torch').zeros(n_domains))

    def forward(self, y: Tensor) -> Tensor:
        """
        y      : [B, n_domains, D]
        returns: [B, n_domains]  raw logits (pre-softmax)
        """
        # logits[b, d] = (y[b, d] * probe[d]).sum(-1) + bias[d]
        return (y * self.probe.unsqueeze(0)).sum(-1) + self.bias


class HaltHead(nn.Module):
    """
    Simplified ACT halt head.

    Projects mean-pooled y → scalar halt logit q.
    Sigmoid(q) > halt_threshold → stop recursion at inference time.
    Trained with BCE against (argmax_pred == argmax_true).

    Parameters
    ----------
    hidden_size : int
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        # Mean-pool over domain dim then project to scalar
        self.proj = nn.Linear(hidden_size, 1, bias=True)

    def forward(self, y: Tensor) -> Tensor:
        """
        y      : [B, n_domains, D]
        returns: [B]  raw halt logit (pre-sigmoid)
        """
        pooled = y.mean(dim=1)          # [B, D]
        return self.proj(pooled).squeeze(-1)  # [B]
