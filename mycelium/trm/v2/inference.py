from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from mycelium.domain_graph.novelty import DomainNoveltyPolicy, NoveltyResult
from mycelium.domain_graph.registry import DomainGraphRegistry
from mycelium.domain_graph.events import emit_novelty_decision
from mycelium.domain_graph.state import NoveltyDecision
from mycelium.trm.v2.encoder import SharedEncoder
from mycelium.trm.v2.gating import DomainGate, GateOutput
from mycelium.trm.v2.halt import HaltControllerV2, HaltState

logger = logging.getLogger(__name__)


@dataclass
class RouteResult:
    """
    The single structured output of TRMV2InferenceEngine.route().
    Downstream modules consume this — no other routing signal exists.
    """
    selected_domain_id: Optional[str]      # None → OOD fallback
    novelty_decision: NoveltyDecision
    gate_confidence: float
    novelty_similarity: float
    halt_state: HaltState
    top_k_domains: List[Tuple[str, float]] # [(domain_id, score), ...]
    embedding: Optional[List[float]] = None
    ood_fallback: bool = False
    reason: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


class TRMV2InferenceEngine:
    """
    The single routing authority for TRM v2.

    Route flow:
      1. Encode query → embedding via SharedEncoder
      2. DomainNoveltyPolicy.evaluate() → NoveltyResult
      3. If ROUTE_EXISTING: DomainGate.route() → GateOutput
         - If gate open: return selected domain
         - If gate closed: emit OOD
      4. If REACTIVATE_COLD: emit reactivation event, return pending domain
      5. If EXPAND_EXISTING / CREATE_NEW / DEFER_OOD: return OOD with decision
      6. HaltController applied at each step
    """

    def __init__(
        self,
        encoder: SharedEncoder,
        gate: DomainGate,
        novelty_policy: DomainNoveltyPolicy,
        domain_registry: DomainGraphRegistry,
        halt_controller: Optional[HaltControllerV2] = None,
        tokenizer=None,         # optional: any HF-compatible tokenizer
        max_length: int = 128,
        device: str = "cpu",
    ) -> None:
        self._encoder = encoder
        self._gate = gate
        self._novelty = novelty_policy
        self._registry = domain_registry
        self._halt = halt_controller or HaltControllerV2()
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._device = torch.device(device)
        self._encoder.to(self._device)

    # ------------------------------------------------------------------
    # Primary entry point — the only allowed routing authority
    # ------------------------------------------------------------------

    def route(
        self,
        query_text: str,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_embedding: bool = False,
    ) -> RouteResult:
        """
        Route a single query to a domain or OOD fallback.

        Either supply pre-tokenised `input_ids` + `attention_mask`,
        or supply only `query_text` and set a tokenizer at construction.
        """
        self._halt.reset()

        # --- Step 1: encode ---
        ids, mask = self._resolve_inputs(query_text, input_ids, attention_mask)
        embedding = self._encoder.encode(ids.to(self._device), mask.to(self._device) if mask is not None else None)
        emb_list: List[float] = embedding[0].cpu().tolist()

        halt_state = self._halt.step(confidence=0.0, step=0)

        # --- Step 2: novelty evaluation ---
        novelty: NoveltyResult = self._novelty.evaluate(emb_list, query_text)

        emit_novelty_decision(
            domain_id=novelty.matched_domain_id,
            decision=novelty.decision,
            similarity=novelty.similarity,
            novelty_score=novelty.novelty_score,
            query_text=query_text,
            spectral_signal=novelty.spectral_signal,
            reason=novelty.reason,
        )

        # --- Step 3: gate routing for ROUTE_EXISTING ---
        if novelty.decision == NoveltyDecision.ROUTE_EXISTING:
            gate_out: GateOutput = self._gate.route(embedding[0])
            halt_state = self._halt.step(confidence=gate_out.confidence, step=1)

            if gate_out.gate_open and gate_out.selected_domain_id:
                self._registry.mark_active(gate_out.selected_domain_id)
                return RouteResult(
                    selected_domain_id=gate_out.selected_domain_id,
                    novelty_decision=novelty.decision,
                    gate_confidence=gate_out.confidence,
                    novelty_similarity=novelty.similarity,
                    halt_state=halt_state,
                    top_k_domains=gate_out.top_k,
                    embedding=emb_list if return_embedding else None,
                    ood_fallback=False,
                    reason=gate_out.reason,
                )

            # Gate closed despite novelty match → OOD fallback
            return RouteResult(
                selected_domain_id=None,
                novelty_decision=novelty.decision,
                gate_confidence=gate_out.confidence,
                novelty_similarity=novelty.similarity,
                halt_state=halt_state,
                top_k_domains=gate_out.top_k,
                embedding=emb_list if return_embedding else None,
                ood_fallback=True,
                reason=f"Novelty matched but gate closed: {gate_out.reason}",
            )

        # --- Step 4: reactivation ---
        if novelty.decision == NoveltyDecision.REACTIVATE_COLD:
            return RouteResult(
                selected_domain_id=novelty.matched_domain_id,
                novelty_decision=novelty.decision,
                gate_confidence=novelty.similarity,
                novelty_similarity=novelty.similarity,
                halt_state=halt_state,
                top_k_domains=[],
                embedding=emb_list if return_embedding else None,
                ood_fallback=False,
                reason=f"Cold domain queued for reactivation: {novelty.reason}",
                meta={"reactivation_pending": True},
            )

        # --- Step 5: all other decisions → OOD ---
        return RouteResult(
            selected_domain_id=None,
            novelty_decision=novelty.decision,
            gate_confidence=0.0,
            novelty_similarity=novelty.similarity,
            halt_state=halt_state,
            top_k_domains=[],
            embedding=emb_list if return_embedding else None,
            ood_fallback=True,
            reason=novelty.reason,
        )

    # ------------------------------------------------------------------
    # Input resolution
    # ------------------------------------------------------------------

    def _resolve_inputs(
        self,
        query_text: str,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if input_ids is not None:
            return input_ids, attention_mask

        if self._tokenizer is not None:
            enc = self._tokenizer(
                query_text,
                return_tensors="pt",
                truncation=True,
                max_length=self._max_length,
                padding=True,
            )
            return enc["input_ids"], enc.get("attention_mask")

        raise ValueError(
            "TRMV2InferenceEngine requires either pre-tokenised input_ids "
            "or a tokenizer set at construction time."
        )
