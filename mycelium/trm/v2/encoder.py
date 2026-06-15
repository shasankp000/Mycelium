from __future__ import annotations
import inspect
import logging
import struct
from typing import List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _pack_floats(values: List[float]) -> bytes:
    return struct.pack(f"{len(values)}f", *values)


def _backbone_accepts_mask(backbone: nn.Module) -> bool:
    """
    Returns True if the backbone's forward() accepts an attention_mask kwarg.
    Plain nn.Linear / stub networks return False; HF models return True.
    """
    try:
        sig = inspect.signature(backbone.forward)
        return "attention_mask" in sig.parameters
    except (ValueError, TypeError):
        return False


class SharedEncoder(nn.Module):
    """
    Frozen shared encoder backbone for TRM v2.
    Wraps any backbone — plain nn.Linear stubs, custom TRMNetwork, or HF models.
    Exposes a single encode() method that returns a normalised float32 embedding.

    The backbone is frozen by default (requires_grad=False on all params).
    It is only partially thawed during GateState.THAWING for the affected
    domain neighbourhood — never globally.
    """

    def __init__(
        self,
        backbone: nn.Module,
        output_dim: int,
        *,
        frozen: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.output_dim = output_dim
        self._frozen = frozen
        self._backbone_accepts_mask = _backbone_accepts_mask(backbone)
        if frozen:
            self._freeze()

    # ------------------------------------------------------------------
    # Freeze / thaw
    # ------------------------------------------------------------------

    def _freeze(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        self._frozen = True
        logger.debug("SharedEncoder: backbone frozen.")

    def freeze(self) -> None:
        self._freeze()

    def thaw_layers(self, layer_names: List[str]) -> None:
        """
        Partially thaw named sub-modules only.
        Called by TRMV2Trainer during THAWING gate state — never called globally.
        """
        for name, module in self.backbone.named_modules():
            if any(name.startswith(ln) for ln in layer_names):
                for p in module.parameters():
                    p.requires_grad = True
        self._frozen = False
        logger.info("SharedEncoder: thawed layers %s", layer_names)

    def refreeze(self) -> None:
        self._freeze()

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns shape (batch, output_dim), L2-normalised.

        The backbone is called with attention_mask only if its forward()
        signature accepts it — this allows plain nn.Linear stubs and
        HF transformer backbones to be used interchangeably.

        The backbone is expected to return either:
          - a tensor of shape (batch, seq, hidden) → we mean-pool
          - a tensor of shape (batch, hidden)       → used directly
        """
        if self._backbone_accepts_mask:
            out = self.backbone(input_ids, attention_mask=attention_mask)
        else:
            inp = input_ids  # backbone receives native dtype; EmbeddingBag needs long, Linear needs float — caller's responsibility
            out = self.backbone(inp)

        # unwrap common wrapper types
        if hasattr(out, "last_hidden_state"):
            out = out.last_hidden_state
        if hasattr(out, "logits"):
            out = out.logits

        if out.dim() == 3:
            if attention_mask is not None and self._backbone_accepts_mask:
                mask = attention_mask.unsqueeze(-1).float()
                out = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                out = out.mean(dim=1)

        # L2 normalise
        out = nn.functional.normalize(out, p=2, dim=-1)
        return out

    def encode_to_list(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[List[float]]:
        return self.encode(input_ids, attention_mask).cpu().tolist()

    def encode_to_bytes(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[bytes]:
        vecs = self.encode_to_list(input_ids, attention_mask)
        return [_pack_floats(v) for v in vecs]

    # ------------------------------------------------------------------
    # Convenience: load from checkpoint path
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        backbone: nn.Module,
        output_dim: int,
        checkpoint_path: str,
        *,
        frozen: bool = True,
        map_location: str = "cpu",
    ) -> "SharedEncoder":
        state = torch.load(checkpoint_path, map_location=map_location)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        backbone.load_state_dict(state, strict=False)
        logger.info("SharedEncoder loaded backbone from %s", checkpoint_path)
        return cls(backbone, output_dim, frozen=frozen)
