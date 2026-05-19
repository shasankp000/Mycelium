"""Phase E — Contradiction Classifier and Leverage Propagator."""

from .classifier import ContradictionClassifier, ClassificationResult
from .propagation import LeveragePropagator, PropagationResult
from .pipeline import PhaseEPipeline, ContradictionReport

__all__ = [
    "ContradictionClassifier",
    "ClassificationResult",
    "LeveragePropagator",
    "PropagationResult",
    "PhaseEPipeline",
    "ContradictionReport",
]
