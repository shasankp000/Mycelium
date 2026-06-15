from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class HaltState:
    step: int = 0
    confidence: float = 0.0
    halted: bool = False
    reason: str = ""


class HaltControllerV2:
    """
    Decides when iterative inference should stop.

    Halts when ANY of these conditions are met (in order):
      1. confidence >= confidence_threshold
      2. step >= max_steps
      3. Confidence delta over last `patience` steps < min_delta (plateau)
    """

    def __init__(
        self,
        confidence_threshold: float = 0.80,
        max_steps: int = 5,
        min_delta: float = 0.01,
        patience: int = 2,
    ) -> None:
        self._conf_threshold = confidence_threshold
        self._max_steps = max_steps
        self._min_delta = min_delta
        self._patience = patience
        self._history: List[float] = []

    def reset(self) -> None:
        self._history = []

    def step(self, confidence: float, step: int) -> HaltState:
        self._history.append(confidence)

        if confidence >= self._conf_threshold:
            return HaltState(
                step=step, confidence=confidence, halted=True,
                reason=f"confidence={confidence:.4f} >= threshold={self._conf_threshold}",
            )

        if step >= self._max_steps:
            return HaltState(
                step=step, confidence=confidence, halted=True,
                reason=f"max_steps={self._max_steps} reached",
            )

        if len(self._history) >= self._patience + 1:
            recent = self._history[-(self._patience + 1):]
            delta = max(recent) - min(recent)
            if delta < self._min_delta:
                return HaltState(
                    step=step, confidence=confidence, halted=True,
                    reason=f"plateau: delta={delta:.4f} < min_delta={self._min_delta} over {self._patience} steps",
                )

        return HaltState(step=step, confidence=confidence, halted=False)
