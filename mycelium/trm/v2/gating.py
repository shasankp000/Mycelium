from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from mycelium.domain_graph.registry import DomainGraphRegistry
from mycelium.domain_graph.state import GateState as RegistryGateState
from mycelium.trm.v2.heads import HeadRegistry

logger = logging.getLogger(__name__)


@dataclass
class GateOutput:
    """Result of one gate pass for a single query."""
    selected_domain_id: Optional[str]
    confidence: float
    top_k: List[Tuple[str, float]]          # [(domain_id, score), ...]
    gate_open: bool                          # False → route to OOD fallback
    reason: str = ""
    meta: Dict = field(default_factory=dict)


class DomainGate:
    """
    Combines HeadRegistry scores with DomainGraphRegistry state to produce
    a routing decision for a single query embedding.

    Rules (in order):
      1. Only domains with gate=OPEN in the registry are candidates.
      2. Score all candidate heads.
      3. If max score >= confidence_threshold → select that domain.
      4. If max score < confidence_threshold → gate_open=False → OOD fallback.
    """

    def __init__(
        self,
        head_registry: HeadRegistry,
        domain_registry: DomainGraphRegistry,
        confidence_threshold: float = 0.55,
        top_k: int = 3,
    ) -> None:
        self._heads = head_registry
        self._domains = domain_registry
        self._threshold = confidence_threshold
        self._top_k = top_k

    def route(self, embedding: torch.Tensor) -> GateOutput:
        """
        embedding: shape (dim,) or (1, dim) — the SharedEncoder output for one query.
        """
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)

        # Only score domains that are OPEN in the registry
        open_ids = {
            n.domain_id
            for n in self._domains.all_hot_domains()
            if n.gate == RegistryGateState.OPEN
        }

        if not open_ids:
            return GateOutput(
                selected_domain_id=None,
                confidence=0.0,
                top_k=[],
                gate_open=False,
                reason="No domains are currently OPEN in registry.",
            )

        scores: Dict[str, float] = {}
        for domain_id in open_ids:
            head = self._heads.get(domain_id)
            if head is None:
                continue
            head.eval()
            with torch.no_grad():
                scores[domain_id] = head.confidence(embedding).item()

        if not scores:
            return GateOutput(
                selected_domain_id=None,
                confidence=0.0,
                top_k=[],
                gate_open=False,
                reason="No loaded heads for any OPEN domain.",
            )

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top = ranked[: self._top_k]
        best_id, best_score = ranked[0]

        if best_score >= self._threshold:
            return GateOutput(
                selected_domain_id=best_id,
                confidence=best_score,
                top_k=top,
                gate_open=True,
                reason=f"confidence={best_score:.4f} >= threshold={self._threshold}",
            )

        return GateOutput(
            selected_domain_id=None,
            confidence=best_score,
            top_k=top,
            gate_open=False,
            reason=f"confidence={best_score:.4f} < threshold={self._threshold} — deferring to OOD.",
        )

    def update_threshold(self, new_threshold: float) -> None:
        self._threshold = new_threshold
        logger.info("DomainGate: confidence threshold updated to %.3f", new_threshold)
