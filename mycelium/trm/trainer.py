"""
Phase D — TRMTrainer
=====================
Training loop for TRMReasoner with:
    - AdamW (β1=0.9, β2=0.95, weight_decay=0.1)
    - Cosine LR schedule with linear warmup
    - EMA (decay=0.999) — paper §4.7: critical for stability on small data
    - Deep supervision loss: domain CE/KL + halt BCE + optional contrastive
    - Stable-max cross-entropy (Prieto et al. 2025, §6 of TRM paper)
    - Soft-label KL-divergence when target_domain is a float vector
      (e.g. spectral_vec from MultiLensRouter used as multi-domain ground truth)

Dataset format
--------------
Each training sample is a dict:
    {
        "token_ids":             LongTensor  [context_len]
        "spectral_vec":          FloatTensor [n_domains]
        "predicate_family_id":   LongTensor  scalar
        "initial_domain_probs":  FloatTensor [n_domains]

        # Hard-label path (default):
        "target_domain":         LongTensor  scalar   (ground-truth expert index)

        # Soft-label path (mixed-domain queries):
        # Pass target_domain as a FloatTensor [n_domains] probability vector.
        # The loss automatically switches from stable-max cross-entropy to
        # KL-divergence(log_softmax(logits) || target_soft_label).
        # spectral_vec from MultiLensRouter is a perfect source: it already
        # encodes multi-domain probability across all registered experts.
    }

These are sourced from Mycelium's routing trace logs:
    training_data/routing_traces.jsonl
    evaluation_data/routing_ground_truth.jsonl

Usage
-----
    cfg     = TRMConfig()
    model   = TRMReasoner(cfg)
    trainer = TRMTrainer(model, cfg, train_dataset, eval_dataset)
    trainer.train()
    trainer.save("checkpoints/trm_latest.pt")
"""

from __future__ import annotations

import copy
import logging
import math
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from .config import TRMConfig
from .reasoner import TRMReasoner

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# --------------------------------------------------------------------------- #
# Stable-max cross-entropy (§6 of arXiv:2510.04871)                           #
# --------------------------------------------------------------------------- #

def stable_max_cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """
    Numerically stable cross-entropy that subtracts the max logit before
    computing softmax (Prieto et al. 2025).  Identical in value to
    F.cross_entropy but avoids overflow on large logit scales.

    logits  : [B, n_classes]
    targets : [B]  int64
    """
    shifted = logits - logits.max(dim=-1, keepdim=True).values
    log_z = shifted.exp().sum(dim=-1).log()
    log_py = shifted.gather(1, targets.unsqueeze(1)).squeeze(1)
    return (log_z - log_py).mean()


def soft_label_kl_loss(logits: Tensor, soft_targets: Tensor) -> Tensor:
    """
    KL-divergence loss for soft / multi-domain targets.

    Replaces stable_max_cross_entropy when target_domain is a float vector
    rather than a single integer.  Suitable for mixed-domain queries where
    the ground truth is a probability distribution over multiple domains
    (e.g. spectral_vec from MultiLensRouter).

    KL(P || Q) = sum( P * log(P / Q) )  where:
        P = soft_targets  (the ground-truth distribution)
        Q = softmax(logits)  (the model's predicted distribution)

    Using F.kl_div(log_Q, P, reduction="batchmean") which is numerically
    stable (log-space input, probability-space target).

    logits      : [B, n_classes]  raw domain logits from TRMReasoner
    soft_targets: [B, n_classes]  target probability distribution (∑ == 1)
    """
    log_probs = F.log_softmax(logits, dim=-1)          # [B, n_classes]
    soft_targets = soft_targets.clamp(min=1e-8)
    soft_targets = soft_targets / soft_targets.sum(dim=-1, keepdim=True)  # renorm
    return F.kl_div(log_probs, soft_targets, reduction="batchmean")


# --------------------------------------------------------------------------- #
# EMA helper                                                                  #
# --------------------------------------------------------------------------- #

class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def state_dict(self) -> dict:
        return self.shadow.state_dict()


# --------------------------------------------------------------------------- #
# Cosine LR with warmup                                                       #
# --------------------------------------------------------------------------- #

def _cosine_lr(step: int, warmup: int, total: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# --------------------------------------------------------------------------- #
# TRMTrainer                                                                  #
# --------------------------------------------------------------------------- #

class TRMTrainer:
    """
    Training harness for TRMReasoner.

    Parameters
    ----------
    model         : TRMReasoner
    cfg           : TRMConfig
    train_dataset : Dataset  each item is a dict (see module docstring)
    eval_dataset  : Dataset or None
    batch_size    : int  default 32

    Notes
    -----
    drop_last is intentionally False so that a last partial batch is never
    silently discarded.  This prevents zero-batch training when the gated
    TraceWriter produces fewer than batch_size samples on later sim runs.
    """

    def __init__(
        self,
        model: TRMReasoner,
        cfg: TRMConfig,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        batch_size: int = 32,
    ) -> None:
        self.model   = model.to(cfg.device)
        self.cfg     = cfg
        self.ema     = EMA(model, decay=cfg.ema_decay)
        self.train_dl = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )
        self.eval_dl = (
            DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
            if eval_dataset else None
        )
        self.optimizer = AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
        )
        self.step = 0

    # ------------------------------------------------------------------ #
    # Loss computation                                                     #
    # ------------------------------------------------------------------ #

    def _compute_loss(
        self,
        batch: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """
        Three-term loss:
            L = w_dom * L_domain  +  w_halt * L_halt  +  w_con * L_contrastive

        L_domain path selection
        -----------------------
        Hard label  (target_domain is a 1-D int64 vector [B]):
            stable_max_cross_entropy(logits, target_domain)

        Soft label  (target_domain is a 2-D float32 matrix [B, n_domains]):
            KL( softmax(logits) || target_domain )
            Use this path for mixed-domain queries by passing spectral_vec
            (from MultiLensRouter) as target_domain in the dataset record.
        """
        cfg = self.cfg
        device = cfg.device

        token_ids            = batch["token_ids"].to(device)
        spectral_vec         = batch["spectral_vec"].to(device)
        predicate_family_id  = batch["predicate_family_id"].to(device)
        initial_domain_probs = batch["initial_domain_probs"].to(device)
        target_domain        = batch["target_domain"].to(device)

        out = self.model(
            token_ids=token_ids,
            spectral_vec=spectral_vec,
            predicate_family_id=predicate_family_id,
            initial_domain_probs=initial_domain_probs,
        )

        # 1. Primary routing loss — auto-select hard vs soft path
        if target_domain.dim() == 2:
            l_domain = soft_label_kl_loss(
                out.domain_logits,
                target_domain.float(),
            )
            hard_target = target_domain.argmax(dim=-1)  # [B]
        else:
            if cfg.stable_max_loss:
                l_domain = stable_max_cross_entropy(out.domain_logits, target_domain)
            else:
                l_domain = F.cross_entropy(out.domain_logits, target_domain)
            hard_target = target_domain  # [B]

        # 2. Halt BCE: target = 1 if prediction correct, else 0
        correct = (out.primary_domain_idx == hard_target).float()  # [B]
        l_halt = F.binary_cross_entropy_with_logits(
            self.model.halt_head(self.model.answer_embedder(initial_domain_probs)),
            correct,
        )

        # 3. Contrastive term (optional — skipped if weight==0)
        l_contrastive = torch.tensor(0.0, device=device)
        if cfg.contrastive_loss_weight > 0 and "semantic_hash" in batch:
            emb = out.final_y.mean(dim=1)          # [B, D]
            emb = F.normalize(emb, dim=-1)
            sim = emb @ emb.T / 0.07               # [B, B] cosine sim / temp
            labels = torch.arange(emb.shape[0], device=device)
            l_contrastive = F.cross_entropy(sim, labels)

        total = (
            cfg.domain_loss_weight        * l_domain
            + cfg.halt_loss_weight        * l_halt
            + cfg.contrastive_loss_weight * l_contrastive
        )
        return {
            "loss":           total,
            "l_domain":       l_domain.detach(),
            "l_halt":         l_halt.detach(),
            "l_contrastive":  l_contrastive.detach(),
        }

    # ------------------------------------------------------------------ #
    # Training loop                                                        #
    # ------------------------------------------------------------------ #

    def train(self) -> None:
        cfg = self.cfg
        self.model.train()
        logger.info("TRMTrainer: starting training for %d steps", cfg.max_train_steps)

        while self.step < cfg.max_train_steps:
            for batch in self.train_dl:
                if self.step >= cfg.max_train_steps:
                    break

                lr = _cosine_lr(self.step, cfg.warmup_steps, cfg.max_train_steps, cfg.learning_rate)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr

                losses = self._compute_loss(batch)
                loss = losses["loss"]

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.gradient_clip)
                self.optimizer.step()
                self.ema.update(self.model)

                self.step += 1

                if self.step % 500 == 0:
                    logger.info(
                        "step=%d lr=%.2e loss=%.4f l_dom=%.4f l_halt=%.4f",
                        self.step, lr,
                        loss.item(),
                        losses["l_domain"].item(),
                        losses["l_halt"].item(),
                    )

                if self.step % 5000 == 0 and self.eval_dl:
                    acc = self.evaluate()
                    logger.info("step=%d eval_acc=%.4f", self.step, acc)

    # ------------------------------------------------------------------ #
    # Evaluation                                                           #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def evaluate(self) -> float:
        """Returns top-1 accuracy on the eval dataset using EMA weights."""
        if self.eval_dl is None:
            return float("nan")
        cfg = self.cfg
        ema_model = self.ema.shadow.to(cfg.device).eval()
        correct = total = 0
        for batch in self.eval_dl:
            target = batch["target_domain"].to(cfg.device)
            hard_target = target.argmax(dim=-1) if target.dim() == 2 else target
            out = ema_model(
                token_ids=batch["token_ids"].to(cfg.device),
                spectral_vec=batch["spectral_vec"].to(cfg.device),
                predicate_family_id=batch["predicate_family_id"].to(cfg.device),
                initial_domain_probs=batch["initial_domain_probs"].to(cfg.device),
            )
            correct += (out.primary_domain_idx == hard_target).sum().item()
            total   += hard_target.shape[0]
        return correct / max(1, total)

    # ------------------------------------------------------------------ #
    # Checkpoint                                                           #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """Save both live model and EMA shadow weights."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state":  self.model.state_dict(),
            "ema_state":    self.ema.state_dict(),
            "step":         self.step,
            "cfg":          self.cfg,
        }, path)
        logger.info("TRMTrainer: saved checkpoint to %s (step=%d)", path, self.step)

    @classmethod
    def load_checkpoint(cls, path: str, model: TRMReasoner, cfg: TRMConfig) -> "TRMTrainer":
        """Restore trainer state from a saved checkpoint."""
        ckpt = torch.load(path, map_location=cfg.device)
        model.load_state_dict(ckpt["model_state"])
        trainer = cls(model, cfg, train_dataset=_EmptyDataset(), batch_size=1)
        trainer.ema.shadow.load_state_dict(ckpt["ema_state"])
        trainer.step = ckpt["step"]
        return trainer


class _EmptyDataset(Dataset):
    """Placeholder dataset used when restoring a checkpoint without data."""
    def __len__(self) -> int: return 0
    def __getitem__(self, idx: int) -> dict: raise IndexError(idx)
