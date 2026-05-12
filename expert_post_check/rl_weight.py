"""
expert_post_check.rl_weight
============================
RL-style per-domain TRM priority weight store.

Mechanism
---------
After each post-check run the fuzzy verifier emits a ``confidence_vote``
(positive on match, negative on mismatch).  This module applies that vote
to the stored weight for the query's domain using an Exponential Moving
Average (EMA):

    w_new = alpha * (base + vote) + (1 - alpha) * w_old

where ``alpha`` controls the learning rate (default 0.1 — slow, stable).

The weight is persisted to ``post_check_weights.json`` in the project root
so it survives across server restarts.  On the next run that domain is
processed, the stored weight is read back and used to decide whether the
TRM-slot or the 6-phase pipeline should be the ``preferred`` source when
both answers agree.

Weight semantics
----------------
    w >= TRM_PREFER_THRESHOLD   → TRM-slot gets priority in routing
    w <  TRM_PREFER_THRESHOLD   → 6-phase pipeline gets priority

Bounds
------
    Weights are clamped to [0.0, 1.0] after every update.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict

logger = logging.getLogger(__name__)

_WEIGHTS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "post_check_weights.json",
)
DEFAULT_WEIGHT = 0.5
ALPHA = 0.1                 # EMA learning rate
TRM_PREFER_THRESHOLD = 0.55 # weight above which TRM-slot is preferred


@dataclass
class WeightUpdateResult:
    domain: str
    old_weight: float
    new_weight: float
    vote_applied: float
    trm_preferred: bool


class RLWeightStore:
    """Persistent EMA weight store for per-domain TRM priority.

    Thread-safety: file I/O is not locked.  For the current single-process
    architecture this is acceptable; add a filelock if concurrency is needed.
    """

    def __init__(
        self,
        weights_file: str = _WEIGHTS_FILE,
        alpha: float = ALPHA,
        prefer_threshold: float = TRM_PREFER_THRESHOLD,
    ) -> None:
        self._file = weights_file
        self._alpha = alpha
        self._prefer_threshold = prefer_threshold
        self._weights: Dict[str, float] = self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_weight(self, domain: str) -> float:
        """Return the current weight for *domain* (default 0.5)."""
        return self._weights.get(domain, DEFAULT_WEIGHT)

    def trm_preferred(self, domain: str) -> bool:
        """Return True when TRM-slot should take priority for *domain*."""
        return self.get_weight(domain) >= self._prefer_threshold

    def update(
        self, domain: str, confidence_vote: float
    ) -> WeightUpdateResult:
        """Apply *confidence_vote* to *domain*'s weight via EMA.

        Args:
            domain:          Domain being updated.
            confidence_vote: Signed delta from FuzzyVerifier (+ve = match).

        Returns:
            A ``WeightUpdateResult`` describing the update.
        """
        old = self._weights.get(domain, DEFAULT_WEIGHT)
        target = old + confidence_vote          # shift from vote
        new = self._alpha * target + (1 - self._alpha) * old
        new = max(0.0, min(1.0, new))           # clamp [0, 1]
        self._weights[domain] = new
        self._save()
        result = WeightUpdateResult(
            domain=domain,
            old_weight=round(old, 6),
            new_weight=round(new, 6),
            vote_applied=confidence_vote,
            trm_preferred=new >= self._prefer_threshold,
        )
        logger.info(
            "RL weight update — domain=%s old=%.4f vote=%.3f new=%.4f preferred=%s",
            domain, old, confidence_vote, new, result.trm_preferred,
        )
        return result

    def all_weights(self) -> Dict[str, float]:
        """Return a copy of all stored weights."""
        return dict(self._weights)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self) -> Dict[str, float]:
        if os.path.exists(self._file):
            try:
                with open(self._file, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    logger.debug("Loaded RL weights from %s", self._file)
                    return {k: float(v) for k, v in data.items()}
            except Exception as exc:
                logger.warning("Could not load weights file: %s", exc)
        return {}

    def _save(self) -> None:
        try:
            with open(self._file, "w", encoding="utf-8") as fh:
                json.dump(self._weights, fh, indent=2)
        except Exception as exc:
            logger.warning("Could not save weights file: %s", exc)
