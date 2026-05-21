"""
Phase 2.6 — Decision Synthesis & Recommendation Generation.

This module consolidates expert decisions, generates final
recommendations, creates explainable reasoning chains, and
assigns confidence scores for the Phase 2 reasoning pipeline.

Components:
    * ReasoningChainBuilder — step-by-step reasoning chains
    * DecisionMaker — final decision logic
    * ConfidenceScorer — composite confidence scoring
    * ActionRecommender — action recommendation generation
    * DecisionSynthesisPipeline — orchestrator

Example:
    >>> from mycelium.pipeline.phase2.phases.phase_2_6_synthesis import (
    ...     DecisionSynthesisPipeline,
    ... )
    >>> pipeline = DecisionSynthesisPipeline()
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mycelium.pipeline.phase2.config.phase2_config import Phase2Config
from mycelium.pipeline.phase2.phases.phase_2_4_inference import (
    InferenceResult,
)
from mycelium.pipeline.phase2.phases.phase_2_5_calibration import (
    CalibrationResult,
)
from mycelium.pipeline.phase2.utils.semantic_types import (
    ExpertSelectionResult,
    RankedExpert,
    SemanticResult,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Custom exceptions
# ------------------------------------------------------------------


class SynthesisError(Exception):
    """Raised when decision synthesis fails."""


# ------------------------------------------------------------------
# FinalDecisionResult dataclass
# ------------------------------------------------------------------


@dataclass
class FinalDecisionResult:
    """Output of Phase 2.6 decision synthesis.

    Attributes:
        original_text: The input text.
        final_decision: One of 'use_existing',
            'create_new_expert', 'create_patch'.
        decision_confidence: Confidence in [0, 1].
        confidence_factors: Breakdown of confidence components.
        reasoning_chain: Step-by-step reasoning list.
        reasoning_text: Human-readable explanation.
        recommended_action: Specific action to take.
        action_details: Additional action details.
        selected_experts: Names of selected experts.
        primary_expert: Best-matching expert name.
        expert_scores: Expert name to score mapping.
        expert_predictions: Expert name to prediction mapping.
        aggregated_prediction: Consensus prediction.
        total_uncertainty: Overall uncertainty value.
        uncertainty_intervals: Per-expert intervals.
        processing_time_ms: Wall-clock processing time.
        phases_executed: List of phase names that ran.
        warnings: Non-fatal issues encountered.
    """

    original_text: str = ""
    final_decision: str = "use_existing"
    decision_confidence: float = 0.0
    confidence_factors: Dict[str, float] = field(
        default_factory=dict
    )
    reasoning_chain: List[Dict] = field(default_factory=list)
    reasoning_text: str = ""
    recommended_action: str = ""
    action_details: Dict = field(default_factory=dict)
    selected_experts: List[str] = field(default_factory=list)
    primary_expert: Optional[str] = None
    expert_scores: Dict[str, float] = field(
        default_factory=dict
    )
    expert_predictions: Dict[str, Dict] = field(
        default_factory=dict
    )
    aggregated_prediction: Any = None
    total_uncertainty: float = 0.0
    uncertainty_intervals: Dict = field(default_factory=dict)
    processing_time_ms: float = 0.0
    phases_executed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ------------------------------------------------------------------
# ReasoningChainBuilder
# ------------------------------------------------------------------


class ReasoningChainBuilder:
    """Build human-readable reasoning chains."""

    def build_reasoning_chain(
        self,
        normalization_result: object,
        semantic_result: SemanticResult,
        selection_result: ExpertSelectionResult,
        inference_result: InferenceResult,
        calibration_result: CalibrationResult,
    ) -> List[Dict[str, Any]]:
        """Create step-by-step reasoning chain.

        Args:
            normalization_result: Output from Phase 2.1.
            semantic_result: Output from Phase 2.2.
            selection_result: Output from Phase 2.3.
            inference_result: Output from Phase 2.4.
            calibration_result: Output from Phase 2.5.

        Returns:
            List of reasoning step dicts.
        """
        chain: List[Dict[str, Any]] = []

        # Step 1: Input normalization
        chain.append({
            "step": 1,
            "phase": "2.1",
            "name": "Input Normalization",
            "description": "Cleaned and validated input text",
            "details": {
                "is_valid": getattr(
                    normalization_result, "is_valid", True
                ),
                "language": getattr(
                    normalization_result, "language", "en"
                ),
                "tags_extracted": len(
                    getattr(
                        normalization_result,
                        "extracted_tags",
                        [],
                    )
                ),
            },
        })

        # Step 2: Semantic understanding
        chain.append({
            "step": 2,
            "phase": "2.2",
            "name": "Semantic Understanding",
            "description": (
                "Analyzed text semantics and domain relevance"
            ),
            "details": {
                "top_domain": (
                    semantic_result.ranked_domains[0]
                    if semantic_result.ranked_domains
                    else "unknown"
                ),
                "entities_found": len(
                    semantic_result.extracted_entities
                ),
                "concepts_found": len(
                    semantic_result.key_concepts
                ),
                "confidence": semantic_result.confidence_score,
            },
        })

        # Step 3: Expert selection
        chain.append({
            "step": 3,
            "phase": "2.3",
            "name": "Expert Selection",
            "description": (
                f"Selected {selection_result.total_selected} "
                f"experts from "
                f"{selection_result.total_candidates} candidates"
            ),
            "details": {
                "strategy": selection_result.selection_strategy,
                "selected": [
                    e.name
                    for e in selection_result.selected_experts
                ],
                "confidence": (
                    selection_result.confidence_in_selection
                ),
            },
        })

        # Step 4: Multi-expert inference
        chain.append({
            "step": 4,
            "phase": "2.4",
            "name": "Multi-Expert Inference",
            "description": (
                "Collected predictions from selected experts"
            ),
            "details": {
                "ensemble_confidence": (
                    inference_result.ensemble_confidence
                ),
                "disagreement": (
                    inference_result.prediction_disagreement
                ),
                "outliers": len(
                    inference_result.outlier_indices
                ),
            },
        })

        # Step 5: Calibration
        chain.append({
            "step": 5,
            "phase": "2.5",
            "name": "Confidence Calibration",
            "description": (
                "Calibrated predictions and quantified "
                "uncertainty"
            ),
            "details": {
                "method": (
                    calibration_result.calibration_method
                ),
                "temperature": calibration_result.temperature,
                "total_uncertainty": (
                    calibration_result.total_uncertainty.get(
                        "total_uncertainty", 0.0
                    )
                ),
            },
        })

        return chain

    def generate_reasoning_text(
        self,
        reasoning_chain: List[Dict],
    ) -> str:
        """Convert chain to human-readable text.

        Args:
            reasoning_chain: List of reasoning step dicts.

        Returns:
            Formatted reasoning text.
        """
        lines = ["Reasoning Chain:"]
        for step in reasoning_chain:
            step_num = step.get("step", "?")
            phase = step.get("phase", "?")
            name = step.get("name", "Unknown")
            desc = step.get("description", "")
            lines.append(
                f"  Step {step_num} (Phase {phase}): "
                f"{name}"
            )
            lines.append(f"    → {desc}")

            details = step.get("details", {})
            for key, val in details.items():
                lines.append(f"    • {key}: {val}")

        return "\n".join(lines)

    def add_evidence(
        self,
        reasoning_chain: List[Dict],
        evidence: Dict,
    ) -> List[Dict]:
        """Add supporting evidence to reasoning steps.

        Args:
            reasoning_chain: Existing reasoning chain.
            evidence: Evidence dict to add.

        Returns:
            Updated reasoning chain.
        """
        updated = []
        for step in reasoning_chain:
            new_step = dict(step)
            phase = step.get("phase", "")
            if phase in evidence:
                new_step["evidence"] = evidence[phase]
            updated.append(new_step)
        return updated


# ------------------------------------------------------------------
# DecisionMaker
# ------------------------------------------------------------------

# Default thresholds for decision making
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "use_existing": 0.6,
    "create_patch": 0.3,
    "create_new_expert": 0.0,
}


class DecisionMaker:
    """Make final decisions based on all information."""

    def make_decision(
        self,
        calibration_result: CalibrationResult,
        expert_selection_result: ExpertSelectionResult,
        decision_thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Make final decision with confidence.

        Args:
            calibration_result: Output from Phase 2.5.
            expert_selection_result: Output from Phase 2.3.
            decision_thresholds: Custom threshold dict.

        Returns:
            Dict with decision, confidence, and details.
        """
        thresholds = decision_thresholds or DEFAULT_THRESHOLDS

        calibrated = calibration_result.calibrated_predictions
        posterior = calibration_result.bayesian_posterior

        # Apply decision rules
        decision = self.apply_decision_rules(
            calibrated, thresholds
        )

        # Check for boundary cases
        is_boundary = self.detect_decision_boundary_cases(
            calibrated, thresholds
        )
        if is_boundary:
            decision["is_boundary_case"] = True

        # Incorporate posterior if available
        if posterior:
            best_action = max(posterior, key=posterior.get)
            decision["bayesian_recommendation"] = best_action
            decision["posterior_probs"] = posterior

        # Selection quality factor
        selection_conf = (
            expert_selection_result.confidence_in_selection
        )
        decision["selection_confidence"] = selection_conf

        return decision

    def apply_decision_rules(
        self,
        calibrated_predictions: List[Dict],
        thresholds: Dict[str, float],
    ) -> Dict[str, Any]:
        """Apply rule-based decision making.

        Args:
            calibrated_predictions: Calibrated predictions.
            thresholds: Decision thresholds.

        Returns:
            Decision dict.
        """
        if not calibrated_predictions:
            return {
                "decision": "create_new_expert",
                "confidence": 0.0,
                "reason": "No predictions available",
            }

        # Count votes per action
        from collections import Counter
        actions = []
        for p in calibrated_predictions:
            pred = p.get("prediction", {})
            if isinstance(pred, dict):
                actions.append(
                    pred.get(
                        "recommended_action", "unknown"
                    )
                )
            else:
                actions.append("unknown")

        counts = Counter(actions)
        total = len(actions)

        # Average calibrated confidence
        avg_conf = float(
            np.mean(
                [
                    p.get("confidence", 0.0)
                    for p in calibrated_predictions
                ]
            )
        )

        # Decision logic
        use_threshold = thresholds.get("use_existing", 0.6)
        patch_threshold = thresholds.get("create_patch", 0.3)

        if avg_conf >= use_threshold:
            # Check if majority agrees on use_existing
            use_votes = counts.get("use_existing", 0)
            if use_votes >= total / 2:
                decision = "use_existing"
            else:
                # Majority action wins
                decision = counts.most_common(1)[0][0]
        elif avg_conf >= patch_threshold:
            decision = "create_patch"
        else:
            decision = "create_new_expert"

        return {
            "decision": decision,
            "confidence": avg_conf,
            "vote_counts": dict(counts),
            "reason": (
                f"Average confidence {avg_conf:.3f} with "
                f"{total} expert(s)"
            ),
        }

    def detect_decision_boundary_cases(
        self,
        predictions: List[Dict],
        thresholds: Dict[str, float],
    ) -> bool:
        """Identify edge cases near decision boundaries.

        Args:
            predictions: List of prediction dicts.
            thresholds: Decision thresholds.

        Returns:
            True if near a boundary.
        """
        if not predictions:
            return False

        avg_conf = float(
            np.mean(
                [
                    p.get("confidence", 0.0)
                    for p in predictions
                ]
            )
        )

        margin = 0.05
        for threshold_val in thresholds.values():
            if abs(avg_conf - threshold_val) < margin:
                return True

        return False


# ------------------------------------------------------------------
# ConfidenceScorer
# ------------------------------------------------------------------


class ConfidenceScorer:
    """Compute final confidence in decisions."""

    def score_final_confidence(
        self,
        calibration_result: CalibrationResult,
        inference_result: InferenceResult,
        expert_count: int,
    ) -> float:
        """Compute composite confidence score.

        Args:
            calibration_result: Output from Phase 2.5.
            inference_result: Output from Phase 2.4.
            expert_count: Number of experts used.

        Returns:
            Final confidence in [0, 1].
        """
        # Component 1: Ensemble confidence
        ensemble_conf = inference_result.ensemble_confidence

        # Component 2: Calibrated average
        cal_preds = calibration_result.calibrated_predictions
        if cal_preds:
            cal_avg = float(
                np.mean(
                    [
                        p.get("confidence", 0.0)
                        for p in cal_preds
                    ]
                )
            )
        else:
            cal_avg = 0.0

        # Component 3: Expert agreement
        disagreement = (
            inference_result.prediction_disagreement
        )
        agreement_factor = 1.0 - disagreement

        # Component 4: Expert count factor
        count_factor = min(expert_count / 3.0, 1.0)

        # Weighted combination
        base = (
            0.3 * ensemble_conf
            + 0.3 * cal_avg
            + 0.2 * agreement_factor
            + 0.2 * count_factor
        )

        # Apply uncertainty penalty
        total_unc = calibration_result.total_uncertainty.get(
            "total_uncertainty", 0.0
        )
        final = self.apply_confidence_penalties(
            base, total_unc, disagreement
        )

        return float(np.clip(final, 0.0, 1.0))

    def compute_confidence_factors(
        self,
        results: Dict[str, Any],
    ) -> Dict[str, float]:
        """Breakdown of confidence components.

        Args:
            results: Dict with phase results.

        Returns:
            Dict mapping factor name to value.
        """
        factors: Dict[str, float] = {}

        # Semantic confidence
        sem = results.get("semantic")
        if sem and hasattr(sem, "confidence_score"):
            factors["semantic_confidence"] = (
                sem.confidence_score
            )

        # Selection confidence
        sel = results.get("selection")
        if sel and hasattr(sel, "confidence_in_selection"):
            factors["selection_confidence"] = (
                sel.confidence_in_selection
            )

        # Ensemble confidence
        inf = results.get("inference")
        if inf and hasattr(inf, "ensemble_confidence"):
            factors["ensemble_confidence"] = (
                inf.ensemble_confidence
            )

        # Calibration quality
        cal = results.get("calibration")
        if cal and hasattr(cal, "calibration_metrics"):
            metrics = cal.calibration_metrics
            factors["mean_calibrated_confidence"] = (
                metrics.get("mean_confidence", 0.0)
            )

        return factors

    def apply_confidence_penalties(
        self,
        base_confidence: float,
        uncertainty: float,
        disagreement: float,
    ) -> float:
        """Penalize for uncertainty and disagreement.

        Args:
            base_confidence: Starting confidence.
            uncertainty: Total uncertainty value.
            disagreement: Prediction disagreement [0, 1].

        Returns:
            Penalized confidence in [0, 1].
        """
        penalty = (
            0.3 * min(uncertainty, 1.0)
            + 0.2 * disagreement
        )
        result = base_confidence * (1.0 - penalty)
        return float(np.clip(result, 0.0, 1.0))


# ------------------------------------------------------------------
# ActionRecommender
# ------------------------------------------------------------------


class ActionRecommender:
    """Recommend specific actions based on decision."""

    def recommend_action(
        self,
        decision: Dict,
        expert_selection: ExpertSelectionResult,
        domain_analysis: Dict,
    ) -> Dict[str, Any]:
        """Generate action recommendations.

        Args:
            decision: Decision dict from DecisionMaker.
            expert_selection: Phase 2.3 result.
            domain_analysis: Domain relevance information.

        Returns:
            Action recommendation dict.
        """
        action = decision.get("decision", "create_new_expert")

        if action == "use_existing":
            return self._recommend_use_existing_details(
                decision, expert_selection
            )
        elif action == "create_patch":
            return self._recommend_patch_details(
                decision, expert_selection, domain_analysis
            )
        else:
            return self._recommend_new_expert_details(
                decision, domain_analysis
            )

    def recommend_new_expert_creation(
        self,
        semantic_result: SemanticResult,
        inference_result: InferenceResult,
    ) -> Dict[str, Any]:
        """Recommend creating a new expert if needed.

        Args:
            semantic_result: Phase 2.2 result.
            inference_result: Phase 2.4 result.

        Returns:
            Recommendation dict.
        """
        return {
            "action": "create_new_expert",
            "reason": (
                "No existing expert adequately covers the "
                "input domain"
            ),
            "suggested_domain": (
                semantic_result.ranked_domains[0]
                if semantic_result.ranked_domains
                else "general"
            ),
            "key_concepts": [
                c.text
                for c in semantic_result.key_concepts[:5]
            ],
            "confidence": inference_result.ensemble_confidence,
            "priority": (
                "high"
                if inference_result.ensemble_confidence < 0.3
                else "medium"
            ),
        }

    def recommend_patch_creation(
        self,
        semantic_result: SemanticResult,
        inference_result: InferenceResult,
    ) -> Dict[str, Any]:
        """Recommend creating a patch for existing expert.

        Args:
            semantic_result: Phase 2.2 result.
            inference_result: Phase 2.4 result.

        Returns:
            Recommendation dict.
        """
        # Find closest expert
        best_expert = None
        best_conf = 0.0
        for name, conf in (
            inference_result.prediction_confidences.items()
        ):
            if conf > best_conf:
                best_conf = conf
                best_expert = name

        return {
            "action": "create_patch",
            "target_expert": best_expert,
            "reason": (
                "Existing expert partially covers the domain "
                "but needs enhancement"
            ),
            "gap_concepts": [
                c.text
                for c in semantic_result.key_concepts[:3]
            ],
            "confidence": best_conf,
            "priority": "medium",
        }

    def recommend_use_existing(
        self,
        inference_result: InferenceResult,
        expert_selection: ExpertSelectionResult,
    ) -> Dict[str, Any]:
        """Recommend using existing expert.

        Args:
            inference_result: Phase 2.4 result.
            expert_selection: Phase 2.3 result.

        Returns:
            Recommendation dict.
        """
        primary = None
        if expert_selection.selected_experts:
            primary = expert_selection.selected_experts[0]

        return {
            "action": "use_existing",
            "primary_expert": (
                primary.name if primary else None
            ),
            "domain": (
                primary.domain if primary else "unknown"
            ),
            "match_score": (
                primary.match_score if primary else 0.0
            ),
            "confidence": (
                inference_result.ensemble_confidence
            ),
            "reason": (
                "Existing expert adequately covers the input"
            ),
        }

    # ---- internals ------------------------------------------------

    @staticmethod
    def _recommend_use_existing_details(
        decision: Dict,
        expert_selection: ExpertSelectionResult,
    ) -> Dict[str, Any]:
        """Build details for use_existing recommendation."""
        primary = None
        if expert_selection.selected_experts:
            primary = expert_selection.selected_experts[0]

        return {
            "action": "use_existing",
            "primary_expert": (
                primary.name if primary else None
            ),
            "domain": (
                primary.domain if primary else "unknown"
            ),
            "confidence": decision.get("confidence", 0.0),
            "all_experts": [
                e.name
                for e in expert_selection.selected_experts
            ],
        }

    @staticmethod
    def _recommend_patch_details(
        decision: Dict,
        expert_selection: ExpertSelectionResult,
        domain_analysis: Dict,
    ) -> Dict[str, Any]:
        """Build details for create_patch recommendation."""
        primary = None
        if expert_selection.selected_experts:
            primary = expert_selection.selected_experts[0]

        return {
            "action": "create_patch",
            "target_expert": (
                primary.name if primary else None
            ),
            "domain": (
                primary.domain if primary else "unknown"
            ),
            "confidence": decision.get("confidence", 0.0),
            "domains_to_cover": list(
                domain_analysis.keys()
            )[:3],
        }

    @staticmethod
    def _recommend_new_expert_details(
        decision: Dict,
        domain_analysis: Dict,
    ) -> Dict[str, Any]:
        """Build details for create_new_expert."""
        # Find the top domain
        top_domain = "general"
        if domain_analysis:
            top_domain = max(
                domain_analysis, key=domain_analysis.get
            )

        return {
            "action": "create_new_expert",
            "suggested_domain": top_domain,
            "confidence": decision.get("confidence", 0.0),
            "domains_analyzed": list(
                domain_analysis.keys()
            ),
        }


# ------------------------------------------------------------------
# DecisionSynthesisPipeline
# ------------------------------------------------------------------


class DecisionSynthesisPipeline:
    """Orchestrates Phase 2.6 decision synthesis.

    Usage:
        >>> pipeline = DecisionSynthesisPipeline()
        >>> result = pipeline.synthesize(
        ...     norm_result, sem_result, sel_result,
        ...     inf_result, cal_result
        ... )
    """

    def __init__(
        self,
        config: Optional[Phase2Config] = None,
    ) -> None:
        self._config = config or Phase2Config()
        self._reasoning = ReasoningChainBuilder()
        self._decision_maker = DecisionMaker()
        self._confidence_scorer = ConfidenceScorer()
        self._recommender = ActionRecommender()
        self._metadata: Dict[str, Any] = {}

    def synthesize(
        self,
        normalization_result: object,
        semantic_result: SemanticResult,
        selection_result: ExpertSelectionResult,
        inference_result: InferenceResult,
        calibration_result: CalibrationResult,
        decision_thresholds: Optional[Dict] = None,
    ) -> FinalDecisionResult:
        """Run full decision synthesis pipeline.

        Args:
            normalization_result: Output from Phase 2.1.
            semantic_result: Output from Phase 2.2.
            selection_result: Output from Phase 2.3.
            inference_result: Output from Phase 2.4.
            calibration_result: Output from Phase 2.5.
            decision_thresholds: Optional decision thresholds.

        Returns:
            FinalDecisionResult with all outputs.
        """
        start = time.perf_counter()
        warnings: List[str] = []

        logger.info("Starting Phase 2.6 decision synthesis")

        # Step 1: Build reasoning chain
        reasoning_chain = (
            self._reasoning.build_reasoning_chain(
                normalization_result,
                semantic_result,
                selection_result,
                inference_result,
                calibration_result,
            )
        )
        reasoning_text = (
            self._reasoning.generate_reasoning_text(
                reasoning_chain
            )
        )

        # Step 2: Make decision
        decision = self._decision_maker.make_decision(
            calibration_result,
            selection_result,
            decision_thresholds,
        )
        final_decision = decision.get(
            "decision", "create_new_expert"
        )

        if decision.get("is_boundary_case", False):
            warnings.append(
                "Decision is near a confidence boundary"
            )

        # Step 3: Score confidence
        expert_count = len(
            selection_result.selected_experts
        )
        decision_confidence = (
            self._confidence_scorer.score_final_confidence(
                calibration_result,
                inference_result,
                expert_count,
            )
        )

        # Confidence factors
        confidence_factors = (
            self._confidence_scorer.compute_confidence_factors(
                {
                    "semantic": semantic_result,
                    "selection": selection_result,
                    "inference": inference_result,
                    "calibration": calibration_result,
                }
            )
        )

        # Step 4: Generate action recommendation
        domain_analysis = (
            semantic_result.domain_relevance_scores
        )
        action_rec = self._recommender.recommend_action(
            decision, selection_result, domain_analysis
        )

        # Determine primary expert
        primary_expert = None
        if selection_result.selected_experts:
            primary_expert = (
                selection_result.selected_experts[0].name
            )

        # Total uncertainty
        total_unc = calibration_result.total_uncertainty.get(
            "total_uncertainty", 0.0
        )

        elapsed = (time.perf_counter() - start) * 1000.0
        self._metadata = {
            "decision": final_decision,
            "confidence": decision_confidence,
            "processing_time_ms": elapsed,
        }

        logger.info(
            "Phase 2.6 complete: decision=%s "
            "confidence=%.3f time=%.1fms",
            final_decision,
            decision_confidence,
            elapsed,
        )

        return FinalDecisionResult(
            original_text=getattr(
                normalization_result,
                "original_text",
                inference_result.text,
            ),
            final_decision=final_decision,
            decision_confidence=decision_confidence,
            confidence_factors=confidence_factors,
            reasoning_chain=reasoning_chain,
            reasoning_text=reasoning_text,
            recommended_action=action_rec.get(
                "action", final_decision
            ),
            action_details=action_rec,
            selected_experts=[
                e.name
                for e in selection_result.selected_experts
            ],
            primary_expert=primary_expert,
            expert_scores=selection_result.expert_scores,
            expert_predictions=(
                inference_result.predictions_per_expert
            ),
            aggregated_prediction=(
                inference_result.aggregated_prediction
            ),
            total_uncertainty=total_unc,
            uncertainty_intervals=(
                calibration_result.uncertainty_intervals
            ),
            processing_time_ms=elapsed,
            phases_executed=[
                "2.1", "2.2", "2.3", "2.4", "2.5", "2.6",
            ],
            warnings=warnings,
        )

    def get_processing_metadata(self) -> Dict[str, Any]:
        """Return timing metadata.

        Returns:
            Dict with processing metadata.
        """
        return dict(self._metadata)
