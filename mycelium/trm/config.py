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
                            (disabled by default: F.cross_entropy is more
                            stable on small datasets with random-init weights;
                            enable only after the model has converged)
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _count_live_domains() -> int:
    """Return the number of expert domains currently on disk.

    Resolution order:
      1. _discover_live_domains() from layer1_router  — authoritative source,
         reads whatever subdirectories exist under the experts/ directory.
      2. config_loader.layer1_domain_list()           — toml fallback.
      3. 8                                            — hard fallback of last
         resort (only if disk AND config are both unavailable).

    This is called as a dataclass default_factory so every TRMConfig()
    constructed anywhere in the codebase automatically picks up the current
    domain count without any manual synchronisation.
    """
    try:
        from mycelium.pipeline.layer1_router import _discover_live_domains
        domains = _discover_live_domains()
        if domains:
            return len(domains)
    except Exception:
        pass
    try:
        from mycelium.pipeline import config_loader as cfg
        return len(cfg.layer1_domain_list())
    except Exception:
        return 8


@dataclass
class TRMConfig:
    # ------------------------------------------------------------------ #
    # Architecture                                                         #
    # ------------------------------------------------------------------ #

    hidden_size: int = 512
    """Embedding dimension D.  Should match the spectral analyser's output dim."""

    n_domains: int = field(default_factory=_count_live_domains)
    """Number of expert routing domains.
    Auto-discovered from the experts/ directory at instantiation time via
    _count_live_domains() so this value always stays in sync with whatever
    expert subdirectories exist on disk.  Never hardcode this field."""

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

    learning_rate: float = 1e-5
    """
    Conservative LR suitable for both scratch training and fine-tuning.
    1e-4 caused loss explosions (logits → 1000s) with random-init weights;
    1e-5 + cosine warmup is stable in both regimes.
    """

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

    stable_max_loss: bool = False
    """
    Use numerically stable-max cross-entropy (Prieto et al. §6).
    Disabled by default — standard F.cross_entropy is more stable during
    early training with random-init weights.  Re-enable after convergence
    if needed for fine-tuning on harder distributions.
    """

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
    """Torch device string.  Overridden to 'cuda' at training time if available."""

    model_path: str = ""
    """Path to a saved .pt checkpoint.  Empty string = untrained (uniform priors)."""

    fallback_to_router: bool = True
    """If TRM is untrained / unavailable, fall back to raw MultiLensRouter scores."""

    # ------------------------------------------------------------------ #
    # OOD fallback (Part D)                                               #
    # ------------------------------------------------------------------ #

    ood_halt_threshold: float = 0.55
    """
    If halt_confidence < this value the heuristic OOD trigger considers
    TRM's answer unstable.  Works in conjunction with ood_divergence_threshold.
    Deliberately set slightly above halt_threshold (0.5) so that queries
    that barely triggered a halt are still forwarded to the fallback chain.
    """

    ood_divergence_threshold: float = 0.35
    """
    If (1 - spectral_vec[primary_domain_idx]) > this value the heuristic
    considers TRM's top-domain pick to be in strong disagreement with
    MultiLensRouter's spectral prior.
    Both ood_halt_threshold AND ood_divergence_threshold must be exceeded
    simultaneously for the heuristic to fire.
    """

    ood_head_hidden: int = 0
    """
    Bottleneck dimension for TRMOODHead.  0 → auto (hidden_size // 4).
    Set to a positive integer to override.
    """
