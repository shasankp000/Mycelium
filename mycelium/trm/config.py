"""
Phase D — TRMConfig
====================
Hyperparameters for the Samsung TRM neural reasoner adapted for
Mycelium's domain-routing task (arXiv:2510.04871).

All architectural choices are justified by ablation results in the paper:
    n_layers=2   §4.4  — 2-layer beats 4-layer (87.4% vs 79.5%)
    n_recursions=6  Table 1 — optimal inner recursion depth
    n_supervision=3 Table 3 — T=3 outer deep supervision loops
    ema_decay=0.999 §4.7  — prevents sharp collapse on small datasets
    stable_max_loss §6    — numerically stable cross-entropy variant
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TRMConfig:
    # ------------------------------------------------------------------ #
    # Architecture                                                         #
    # ------------------------------------------------------------------ #

    hidden_size: int = 512
    """Embedding dimension D.  Should match the spectral analyser's output dim."""

    n_domains: int = 8
    """Number of expert routing domains.  Must match MultiLensRouter."""

    context_len: int = 64
    """Maximum canonical query token length L."""

    n_layers: int = 2
    """Transformer layers inside TRMCell.  Paper §4.4: 2 > 4."""

    n_recursions: int = 6
    """Inner latent recursion steps n per supervision loop.  Paper Table 1."""

    n_supervision: int = 3
    """Outer deep supervision loops T.  Paper Table 3: T=3 optimal."""

    n_heads: int = 8
    """Number of attention heads.  hidden_size must be divisible by n_heads."""

    ffn_expansion: int = 4
    """SwiGLU FFN hidden dim = hidden_size * ffn_expansion."""

    dropout: float = 0.0
    """Dropout rate.  Paper uses 0 — routing datasets are small."""

    use_rope: bool = True
    """Rotary position embeddings inside TRMCell attention."""

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #

    ema_decay: float = 0.999
    """EMA decay for model weights.  §4.7: critical for stability."""

    learning_rate: float = 1e-4

    weight_decay: float = 0.1

    betas: tuple = (0.9, 0.95)
    """AdamW beta parameters following PaLM convention."""

    warmup_steps: int = 2000

    max_train_steps: int = 100_000

    gradient_clip: float = 1.0

    # ------------------------------------------------------------------ #
    # Deep supervision                                                     #
    # ------------------------------------------------------------------ #

    max_supervision_steps: int = 16
    """N_sup upper bound at test-time adaptive computation."""

    halt_threshold: float = 0.5
    """Sigmoid(halt_logit) > threshold → early stop at inference."""

    # ------------------------------------------------------------------ #
    # Loss                                                                 #
    # ------------------------------------------------------------------ #

    stable_max_loss: bool = True
    """Use numerically stable-max cross-entropy (Prieto et al. 2025, §6)."""

    domain_loss_weight: float = 1.0
    """Weight for primary routing cross-entropy term."""

    halt_loss_weight: float = 0.1
    """Weight for BCE halt prediction term."""

    contrastive_loss_weight: float = 0.05
    """Weight for optional IR-graph contrastive term."""

    # ------------------------------------------------------------------ #
    # Inference / integration                                              #
    # ------------------------------------------------------------------ #

    device: str = "cpu"
    """Torch device string.  TRM is small enough to run on CPU."""

    model_path: str = ""
    """Path to a saved .pt checkpoint.  Empty string = untrained (uniform priors)."""

    fallback_to_router: bool = True
    """If TRM is untrained / unavailable, fall back to raw MultiLensRouter scores."""
