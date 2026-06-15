from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from mycelium.trm.v2.encoder import SharedEncoder
from mycelium.trm.v2.heads import DomainHead, HeadRegistry
from mycelium.trm.v2.checkpointing import (
    HeadManifest, HeadManifestEntry, save_head
)
from mycelium.trm.v2.policies import GateModeController

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    domain_id: str
    domain_version: int = 1
    head_version: int = 1
    artifact_dir: str = "artifacts/trm_v2/heads"
    epochs: int = 10
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    device: str = "cpu"
    eval_every_n_epochs: int = 2
    early_stop_patience: int = 3


@dataclass
class TrainResult:
    domain_id: str
    final_loss: float
    best_eval_f1: float
    epochs_run: int
    artifact_path: str
    meta: Dict[str, Any] = field(default_factory=dict)


class TRMV2Trainer:
    """
    Trains a single DomainHead against labelled query embeddings.

    Contract:
      - encoder weights are NEVER modified here (encoder must be frozen or
        thaw_layers() called explicitly before instantiating trainer).
      - Only the target domain's head parameters are updated.
      - Other heads in the registry are untouched.
    """

    def __init__(
        self,
        encoder: SharedEncoder,
        head_registry: HeadRegistry,
        gate_controller: GateModeController,
        manifest: Optional[HeadManifest] = None,
    ) -> None:
        self._encoder = encoder
        self._heads = head_registry
        self._gate_ctrl = gate_controller
        self._manifest = manifest or HeadManifest()

    # ------------------------------------------------------------------
    # Bootstrap: train a brand-new head from labelled data
    # ------------------------------------------------------------------

    def bootstrap(
        self,
        config: TrainConfig,
        input_ids: torch.Tensor,           # (N, seq_len)
        attention_mask: torch.Tensor,      # (N, seq_len)
        labels: torch.Tensor,              # (N,) binary: 1=in-domain, 0=out
        eval_callback: Optional[Callable[[int, float], float]] = None,
    ) -> TrainResult:
        device = torch.device(config.device)
        head = self._heads.require(config.domain_id)
        head = head.to(device)

        optimizer = AdamW(head.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()

        # Pre-compute embeddings once (encoder is frozen)
        logger.info("Pre-computing embeddings for domain '%s' (%d samples)…",
                    config.domain_id, input_ids.shape[0])
        self._encoder.eval()
        with torch.no_grad():
            embeddings = self._encoder.encode(
                input_ids.to(device),
                attention_mask.to(device) if attention_mask is not None else None,
            ).cpu()

        dataset = TensorDataset(embeddings, labels.float())
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

        best_f1 = 0.0
        best_path = ""
        no_improve = 0
        last_loss = float("inf")

        for epoch in range(1, config.epochs + 1):
            head.train()
            epoch_loss = 0.0
            for emb_batch, lbl_batch in loader:
                emb_batch = emb_batch.to(device)
                lbl_batch = lbl_batch.to(device)
                optimizer.zero_grad()
                logits = head(emb_batch)
                if logits.dim() > 1:
                    logits = logits.squeeze(-1)
                loss = loss_fn(logits, lbl_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(len(loader), 1)
            last_loss = avg_loss
            logger.debug("Epoch %d/%d — loss=%.4f", epoch, config.epochs, avg_loss)

            if epoch % config.eval_every_n_epochs == 0 and eval_callback is not None:
                f1 = eval_callback(epoch, avg_loss)
                if f1 > best_f1:
                    best_f1 = f1
                    no_improve = 0
                    entry = HeadManifestEntry(
                        domain_id=config.domain_id,
                        domain_version=config.domain_version,
                        head_version=config.head_version,
                        input_dim=head.input_dim,
                        hidden_dim=head.hidden_dim,
                        output_dim=head.output_dim,
                    )
                    self._manifest.add(entry)
                    saved = save_head(
                        head, config.artifact_dir,
                        config.domain_id, config.domain_version, config.head_version,
                        manifest=self._manifest,
                    )
                    best_path = str(saved)
                else:
                    no_improve += 1
                    if no_improve >= config.early_stop_patience:
                        logger.info("Early stop at epoch %d (no F1 improvement for %d evals).",
                                    epoch, config.early_stop_patience)
                        break

        # Final save if no eval callback provided
        if not best_path:
            entry = HeadManifestEntry(
                domain_id=config.domain_id,
                domain_version=config.domain_version,
                head_version=config.head_version,
                input_dim=head.input_dim,
                hidden_dim=head.hidden_dim,
                output_dim=head.output_dim,
            )
            self._manifest.add(entry)
            saved = save_head(
                head, config.artifact_dir,
                config.domain_id, config.domain_version, config.head_version,
                manifest=self._manifest,
            )
            best_path = str(saved)

        self._gate_ctrl.bootstrap_complete(config.domain_id)

        return TrainResult(
            domain_id=config.domain_id,
            final_loss=last_loss,
            best_eval_f1=best_f1,
            epochs_run=config.epochs,
            artifact_path=best_path,
        )
