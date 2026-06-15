from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DriftSignal:
    domain_id: str
    drift_score: float          # 0.0 = no drift, 1.0 = full distribution shift
    centroid_delta: float       # L2 distance between old and new centroid
    variance_ratio: float       # new_variance / old_variance
    sample_count: int
    triggered: bool
    reason: str = ""


@dataclass
class DriftConfig:
    centroid_delta_threshold: float = 0.15   # L2 shift that triggers drift
    variance_ratio_threshold: float = 2.0    # variance explosion threshold
    min_samples_to_evaluate: int = 20        # don't evaluate until N samples seen


class DriftDetector:
    """
    Detects distribution shift for a single domain by tracking:
      - Exponential moving average of the centroid
      - Running variance of incoming embeddings

    Call update() for every routed query embedding.
    Call evaluate() to check if drift has been triggered.

    State is entirely in-memory; persistence is the caller's responsibility
    (DomainGraphRegistry stores the canonical centroid).
    """

    def __init__(
        self,
        domain_id: str,
        initial_centroid: List[float],
        config: Optional[DriftConfig] = None,
        ema_alpha: float = 0.05,
    ) -> None:
        self.domain_id = domain_id
        self._config = config or DriftConfig()
        self._alpha = ema_alpha
        self._dim = len(initial_centroid)

        # EMA centroid
        self._centroid: List[float] = list(initial_centroid)
        self._baseline_centroid: List[float] = list(initial_centroid)

        # Welford online variance
        self._n: int = 0
        self._mean: List[float] = list(initial_centroid)
        self._M2: List[float] = [0.0] * self._dim

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, embedding: List[float]) -> None:
        """Feed one new embedding into the detector."""
        if len(embedding) != self._dim:
            logger.warning("DriftDetector[%s]: dim mismatch %d vs %d",
                           self.domain_id, len(embedding), self._dim)
            return

        # EMA centroid update
        self._centroid = [
            self._alpha * e + (1 - self._alpha) * c
            for e, c in zip(embedding, self._centroid)
        ]

        # Welford online mean + variance
        self._n += 1
        for i in range(self._dim):
            delta = embedding[i] - self._mean[i]
            self._mean[i] += delta / self._n
            delta2 = embedding[i] - self._mean[i]
            self._M2[i] += delta * delta2

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def evaluate(self) -> DriftSignal:
        if self._n < self._config.min_samples_to_evaluate:
            return DriftSignal(
                domain_id=self.domain_id,
                drift_score=0.0,
                centroid_delta=0.0,
                variance_ratio=1.0,
                sample_count=self._n,
                triggered=False,
                reason=f"Insufficient samples ({self._n} < {self._config.min_samples_to_evaluate})",
            )

        # Centroid delta (L2)
        delta = math.sqrt(sum(
            (c - b) ** 2
            for c, b in zip(self._centroid, self._baseline_centroid)
        ))

        # Variance ratio (mean per-dim variance vs baseline = 1.0 for unit sphere)
        if self._n > 1:
            variances = [m2 / (self._n - 1) for m2 in self._M2]
            mean_var = sum(variances) / max(len(variances), 1)
        else:
            mean_var = 0.0
        variance_ratio = mean_var / max(1e-9, 1.0)  # baseline variance ~1.0 for normalised embeds

        drift_score = min(1.0, (delta / max(self._config.centroid_delta_threshold, 1e-9) +
                                variance_ratio / max(self._config.variance_ratio_threshold, 1e-9)) / 2.0)

        triggered = (
            delta >= self._config.centroid_delta_threshold or
            variance_ratio >= self._config.variance_ratio_threshold
        )

        reason = ""
        if triggered:
            parts = []
            if delta >= self._config.centroid_delta_threshold:
                parts.append(f"centroid_delta={delta:.4f} >= {self._config.centroid_delta_threshold}")
            if variance_ratio >= self._config.variance_ratio_threshold:
                parts.append(f"variance_ratio={variance_ratio:.4f} >= {self._config.variance_ratio_threshold}")
            reason = "; ".join(parts)

        return DriftSignal(
            domain_id=self.domain_id,
            drift_score=drift_score,
            centroid_delta=delta,
            variance_ratio=variance_ratio,
            sample_count=self._n,
            triggered=triggered,
            reason=reason,
        )

    def reset_baseline(self) -> None:
        """Call after a successful thaw-retrain cycle."""
        self._baseline_centroid = list(self._centroid)
        logger.info("DriftDetector[%s]: baseline reset.", self.domain_id)

    @property
    def current_centroid(self) -> List[float]:
        return list(self._centroid)

    @property
    def sample_count(self) -> int:
        return self._n
