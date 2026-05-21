"""
Data classes for Phase 3 Validation Layer — Layer 1 types.

Provides typed containers for contradiction analysis results,
validation decisions, and complete failure information used by
the validation orchestrator and downstream phases.

Example:
    >>> from phase3_validation.phases.layer_1_types import (
    ...     Layer1Result, ValidationDecision, CompleteFailureInfo,
    ... )
    >>> result = Layer1Result(contradiction_detected=True, severity="CRITICAL")
    >>> print(result.validation_result_class)
    'complete_failure'
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ------------------------------------------------------------------
# Layer 1 result
# ------------------------------------------------------------------


@dataclass
class Layer1Result:
    """Output of Layer 1 Contradiction Analysis.

    Captures every signal produced by the four sub-checkers
    (contradiction, OOD, reasoning chain, evidence alignment)
    and derives an overall validation result class.

    Attributes:
        contradiction_detected: True when a critical contradiction
            is detected by any sub-checker.
        severity: Overall severity level — one of
            ``CRITICAL``, ``MAJOR``, ``MINOR``, or ``NONE``.
        reason: Primary failure reason — one of
            ``answer_reasoning_mismatch``,
            ``hallucination_detected``,
            ``broken_reasoning_chain``,
            ``evidence_contradiction``, or empty string.
        answer_reasoning_similarity: Cosine similarity between
            the answer embedding and mean reasoning embedding
            (0–1, higher = more similar).
        ood_score: Mahalanobis distance from the evidence
            distribution (higher = more out-of-distribution).
        chain_validity_score: Mean cosine similarity between
            consecutive reasoning steps (0–1).
        evidence_alignment_score: Fraction of answer concepts
            found in the evidence (0–1).
        problem_description: Human-readable description of the
            detected issue (empty when no issue).
        affected_components: List of component names that
            contributed to the failure.
        validation_result_class: Final classification — one of
            ``complete_failure``, ``major_failure``,
            ``minor_discrepancy``, or ``all_pass``.
    """

    contradiction_detected: bool = False
    severity: str = "NONE"
    reason: str = ""
    answer_reasoning_similarity: float = 1.0
    ood_score: float = 0.0
    chain_validity_score: float = 1.0
    evidence_alignment_score: float = 1.0
    problem_description: str = ""
    affected_components: List[str] = field(default_factory=list)
    validation_result_class: str = "all_pass"


# ------------------------------------------------------------------
# Validation decision
# ------------------------------------------------------------------


@dataclass
class ValidationDecision:
    """Classification result produced by the ValidationResultClassifier.

    Attributes:
        result_class: One of ``complete_failure``, ``major_failure``,
            ``minor_discrepancy``, or ``all_pass``.
        action: Concrete action to take (e.g.
            ``reject_and_rerun``, ``pass_to_action_executor``).
        next_step: Human-readable description of the next step.
        confidence: Confidence of the classification in ``[0, 1]``.
        layer1_result: The underlying Layer1Result that drove this
            decision (optional, useful for debugging).
        failure_info: Populated only when result_class is
            ``complete_failure``; contains detailed failure data.
    """

    result_class: str = "all_pass"
    action: str = "pass_to_action_executor"
    next_step: str = "Proceed to action execution"
    confidence: float = 1.0
    layer1_result: Optional[Layer1Result] = None
    failure_info: Optional["CompleteFailureInfo"] = None


# ------------------------------------------------------------------
# Complete failure info
# ------------------------------------------------------------------


@dataclass
class CompleteFailureInfo:
    """Details about a complete validation failure.

    Attributes:
        original_decision: The FinalDecisionResult that failed
            validation.
        failure_reason: Short human-readable reason for failure.
        affected_layers: Phase/layer numbers affected
            (e.g. ``[1, 2, 3]``).
        recommended_action: Suggested corrective action
            (e.g. ``rerun_full_pipeline``).
        retry_count: Number of re-run attempts so far.
        max_retries: Maximum allowed re-run attempts.
    """

    original_decision: Any = None  # FinalDecisionResult
    failure_reason: str = ""
    affected_layers: List[int] = field(default_factory=list)
    recommended_action: str = "rerun_full_pipeline"
    retry_count: int = 0
    max_retries: int = 3
