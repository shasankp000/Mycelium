from __future__ import annotations
import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DomainHead(nn.Module):
    """
    Lightweight per-domain classification/confidence head.
    ~3 layers, hidden_dim=128 by default.
    Training one head never touches any other head or the shared encoder.
    """

    def __init__(
        self,
        domain_id: str,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 1,       # 1 = confidence score; set > 1 for sub-class heads
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.domain_id = domain_id
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def confidence(self, x: torch.Tensor) -> torch.Tensor:
        """Returns sigmoid confidence score in [0, 1], shape (batch,)."""
        logits = self.forward(x)
        if self.output_dim == 1:
            return torch.sigmoid(logits.squeeze(-1))
        return torch.softmax(logits, dim=-1).max(dim=-1).values

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class HeadRegistry:
    """
    In-memory registry of all loaded DomainHeads.
    One head per domain_id. Thread safety is the caller's responsibility
    at training time; inference is read-only after warmup.
    """

    def __init__(self) -> None:
        self._heads: Dict[str, DomainHead] = {}

    def register(self, head: DomainHead) -> None:
        if head.domain_id in self._heads:
            logger.warning("HeadRegistry: overwriting head for domain '%s'.", head.domain_id)
        self._heads[head.domain_id] = head
        logger.debug("HeadRegistry: registered head for '%s' (%d params)",
                     head.domain_id, head.parameter_count())

    def get(self, domain_id: str) -> Optional[DomainHead]:
        return self._heads.get(domain_id)

    def require(self, domain_id: str) -> DomainHead:
        head = self.get(domain_id)
        if head is None:
            raise KeyError(f"No head registered for domain '{domain_id}'.")
        return head

    def remove(self, domain_id: str) -> None:
        self._heads.pop(domain_id, None)

    def all_domain_ids(self) -> List[str]:
        return list(self._heads.keys())

    def score_all(self, embedding: torch.Tensor) -> Dict[str, float]:
        """
        Run all registered heads against `embedding` (shape: [1, dim] or [dim]).
        Returns {domain_id: confidence_score}.
        """
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        scores: Dict[str, float] = {}
        for domain_id, head in self._heads.items():
            head.eval()
            with torch.no_grad():
                scores[domain_id] = head.confidence(embedding).item()
        return scores

    def top_k(self, embedding: torch.Tensor, k: int = 3) -> List[tuple[str, float]]:
        """Returns [(domain_id, score), ...] sorted descending, up to k."""
        scores = self.score_all(embedding)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]
