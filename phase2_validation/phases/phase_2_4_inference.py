"""
Phase 2.4 — Multi-Expert Inference.

This module implements parallel expert inference, prediction
aggregation, outlier detection, and confidence computation for
the Phase 2 reasoning pipeline.

Components:
    * ExpertInference — single expert inference
    * ParallelInferenceExecutor — concurrent multi-expert inference
    * PredictionAggregator — ensemble aggregation strategies
    * ExpertConfidenceCompute — per-expert confidence scoring
    * MultiExpertInferencePipeline — orchestrator

Example:
    >>> from phase2_validation.phases.phase_2_4_inference import (
    ...     MultiExpertInferencePipeline,
    ... )
    >>> pipeline = MultiExpertInferencePipeline()
"""

import logging
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from phase2_validation.config.phase2_config import Phase2Config
from phase2_validation.utils.semantic_types import (
    ExpertPrediction,
    PredictionStatistics,
    RankedExpert,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Custom exceptions
# ------------------------------------------------------------------


class InferenceError(Exception):
    """Raised when inference fails for an expert."""


class AggregationError(Exception):
    """Raised when prediction aggregation fails."""


# ------------------------------------------------------------------
# InferenceResult dataclass
# ------------------------------------------------------------------


@dataclass
class InferenceResult:
    """Output of Phase 2.4 multi-expert inference.

    Attributes:
        text: The input text that was processed.
        selected_experts: Names of the experts used.
        predictions_per_expert: Mapping of expert name to
            prediction dict with keys prediction, confidence,
            metadata.
        aggregated_prediction: Consensus prediction value.
        prediction_confidences: Mapping of expert name to
            confidence score.
        ensemble_confidence: Overall ensemble confidence.
        prediction_statistics: Aggregated statistics dict.
        outlier_indices: Indices of outlier predictions.
        prediction_disagreement: Disagreement level in [0, 1].
        processing_time_ms: Wall-clock processing time.
        warnings: Non-fatal issues encountered.
    """

    text: str = ""
    selected_experts: List[str] = field(default_factory=list)
    predictions_per_expert: Dict[str, Dict] = field(
        default_factory=dict
    )
    aggregated_prediction: Any = None
    prediction_confidences: Dict[str, float] = field(
        default_factory=dict
    )
    ensemble_confidence: float = 0.0
    prediction_statistics: Dict = field(default_factory=dict)
    outlier_indices: List[int] = field(default_factory=list)
    prediction_disagreement: float = 0.0
    processing_time_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)


# ------------------------------------------------------------------
# ExpertInference
# ------------------------------------------------------------------


class ExpertInference:
    """Manage inference from a single expert.

    Since this is a validation pipeline, inference is simulated
    based on expert metadata (domain, match score).
    """

    def __init__(
        self,
        config: Optional[Phase2Config] = None,
    ) -> None:
        self._config = config or Phase2Config()

    def predict(
        self,
        text: str,
        expert: RankedExpert,
        expert_config: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Run inference for a single expert.

        Args:
            text: Input text to process.
            expert: The ranked expert to use.
            expert_config: Optional configuration for the expert.

        Returns:
            Dict with keys: prediction, confidence, metadata,
            latency_ms.

        Raises:
            InferenceError: If inference fails.
        """
        start = time.perf_counter()
        try:
            logger.debug(
                "Running inference for expert=%s domain=%s",
                expert.name,
                expert.domain,
            )

            # Simulate prediction based on expert match quality
            confidence = self._compute_raw_confidence(
                text, expert, expert_config
            )
            prediction = self._generate_prediction(
                text, expert, confidence
            )

            latency_ms = (
                (time.perf_counter() - start) * 1000.0
            )

            result = {
                "prediction": prediction,
                "confidence": confidence,
                "metadata": {
                    "expert_name": expert.name,
                    "domain": expert.domain,
                    "match_score": expert.match_score,
                    "latency_ms": latency_ms,
                },
                "latency_ms": latency_ms,
            }

            logger.info(
                "Expert %s prediction: conf=%.3f latency=%.1fms",
                expert.name,
                confidence,
                latency_ms,
            )
            return result

        except Exception as exc:
            raise InferenceError(
                f"Inference failed for expert {expert.name}: "
                f"{exc}"
            ) from exc

    def predict_batch(
        self,
        texts: List[str],
        expert: RankedExpert,
        expert_config: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Run batch inference for a single expert.

        Args:
            texts: List of input texts.
            expert: The ranked expert to use.
            expert_config: Optional expert configuration.

        Returns:
            List of prediction dicts.
        """
        return [
            self.predict(t, expert, expert_config)
            for t in texts
        ]

    def get_prediction_metadata(
        self, expert: RankedExpert,
    ) -> Dict[str, Any]:
        """Return metadata about the expert's predictions.

        Args:
            expert: The ranked expert.

        Returns:
            Dict with expert metadata.
        """
        return {
            "expert_name": expert.name,
            "domain": expert.domain,
            "match_score": expert.match_score,
            "confidence_interval": (
                expert.confidence_low,
                expert.confidence_high,
            ),
            "ranking_factors": expert.ranking_factors,
        }

    # ---- internals ------------------------------------------------

    @staticmethod
    def _compute_raw_confidence(
        text: str,
        expert: RankedExpert,
        expert_config: Optional[Dict] = None,
    ) -> float:
        """Compute raw confidence for a prediction.

        Combines expert match score with text-based heuristics.
        """
        base = expert.match_score

        # Text length factor: longer text provides more signal
        text_len = len(text.split())
        length_factor = min(text_len / 20.0, 1.0)

        # Confidence interval width factor
        ci_width = expert.confidence_high - expert.confidence_low
        ci_factor = 1.0 - (ci_width * 0.5)

        confidence = (
            0.5 * base
            + 0.3 * length_factor
            + 0.2 * ci_factor
        )
        return float(np.clip(confidence, 0.0, 1.0))

    @staticmethod
    def _generate_prediction(
        text: str,
        expert: RankedExpert,
        confidence: float,
    ) -> Dict[str, Any]:
        """Generate a simulated prediction.

        Uses a deterministic hash to produce consistent results.
        """
        hash_input = f"{text}:{expert.name}:{expert.domain}"
        hash_val = hashlib.sha256(
            hash_input.encode("utf-8")
        ).hexdigest()
        hash_int = int(hash_val[:8], 16)

        # Determine action based on confidence thresholds
        if confidence >= 0.7:
            action = "use_existing"
        elif confidence >= 0.4:
            action = "create_patch"
        else:
            action = "create_new_expert"

        return {
            "recommended_action": action,
            "domain": expert.domain,
            "relevance_score": confidence,
            "hash_signature": hash_val[:16],
            "prediction_id": hash_int % 10000,
        }


# ------------------------------------------------------------------
# ParallelInferenceExecutor
# ------------------------------------------------------------------


class ParallelInferenceExecutor:
    """Execute inference from multiple experts in parallel."""

    def __init__(
        self,
        config: Optional[Phase2Config] = None,
        max_workers: int = 4,
    ) -> None:
        self._config = config or Phase2Config()
        self._max_workers = max_workers
        self._inference = ExpertInference(config)

    def execute_parallel(
        self,
        text: str,
        selected_experts: List[RankedExpert],
        expert_configs: Optional[Dict] = None,
        use_multiprocessing: bool = False,
    ) -> Dict[str, Dict]:
        """Run inference from all experts concurrently.

        Args:
            text: Input text to process.
            selected_experts: Experts selected by Phase 2.3.
            expert_configs: Optional per-expert configurations.
            use_multiprocessing: Use threads (default) or
                processes.

        Returns:
            Dict mapping expert name to prediction dict.
        """
        if not selected_experts:
            logger.warning("No experts provided for inference")
            return {}

        configs = expert_configs or {}
        results: Dict[str, Dict] = {}

        logger.info(
            "Starting parallel inference for %d experts",
            len(selected_experts),
        )

        with ThreadPoolExecutor(
            max_workers=min(
                self._max_workers, len(selected_experts)
            )
        ) as executor:
            futures = {
                executor.submit(
                    self._inference.predict,
                    text,
                    expert,
                    configs.get(expert.name),
                ): expert
                for expert in selected_experts
            }

            for future in as_completed(futures):
                expert = futures[future]
                try:
                    result = future.result()
                    results[expert.name] = result
                except InferenceError as exc:
                    logger.error(
                        "Inference failed for %s: %s",
                        expert.name,
                        exc,
                    )
                    results[expert.name] = {
                        "prediction": None,
                        "confidence": 0.0,
                        "metadata": {
                            "expert_name": expert.name,
                            "error": str(exc),
                        },
                        "latency_ms": 0.0,
                    }

        return results

    def collect_predictions(
        self,
        inference_results: Dict[str, Dict],
    ) -> List[Dict]:
        """Gather all expert predictions into a list.

        Args:
            inference_results: Mapping of expert name to
                prediction dict.

        Returns:
            List of prediction dicts with expert_name added.
        """
        predictions = []
        for name, result in inference_results.items():
            pred = dict(result)
            pred["expert_name"] = name
            predictions.append(pred)
        return predictions

    def compute_prediction_statistics(
        self,
        predictions: List[Dict],
    ) -> Dict[str, float]:
        """Compute statistics across all predictions.

        Args:
            predictions: List of prediction dicts with
                'confidence' key.

        Returns:
            Dict with mean, std, min, max, median, iqr.
        """
        if not predictions:
            return PredictionStatistics().__dict__

        confidences = np.array(
            [p.get("confidence", 0.0) for p in predictions]
        )

        q1 = float(np.percentile(confidences, 25))
        q3 = float(np.percentile(confidences, 75))

        stats = PredictionStatistics(
            mean=float(np.mean(confidences)),
            std=float(np.std(confidences)),
            min=float(np.min(confidences)),
            max=float(np.max(confidences)),
            median=float(np.median(confidences)),
            iqr=q3 - q1,
        )
        return stats.__dict__

    def detect_prediction_outliers(
        self,
        predictions: List[Dict],
    ) -> List[int]:
        """Identify outlier predictions using IQR method.

        Args:
            predictions: List of prediction dicts.

        Returns:
            List of indices of outlier predictions.
        """
        if len(predictions) < 3:
            return []

        confidences = np.array(
            [p.get("confidence", 0.0) for p in predictions]
        )

        q1 = np.percentile(confidences, 25)
        q3 = np.percentile(confidences, 75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = [
            i
            for i, c in enumerate(confidences)
            if c < lower or c > upper
        ]
        return outliers


# ------------------------------------------------------------------
# PredictionAggregator
# ------------------------------------------------------------------


class PredictionAggregator:
    """Aggregate predictions from multiple experts."""

    def aggregate_predictions(
        self,
        predictions: List[Dict],
        strategy: str = "ensemble_voting",
    ) -> Dict[str, Any]:
        """Aggregate predictions using the specified strategy.

        Args:
            predictions: List of prediction dicts.
            strategy: One of 'ensemble_voting',
                'confidence_weighted', 'bayesian'.

        Returns:
            Aggregated prediction dict.

        Raises:
            AggregationError: If strategy is unknown.
        """
        if not predictions:
            return {"recommended_action": "create_new_expert"}

        valid = [
            p for p in predictions
            if p.get("prediction") is not None
        ]
        if not valid:
            return {"recommended_action": "create_new_expert"}

        if strategy == "ensemble_voting":
            return self._ensemble_voting(valid)
        elif strategy == "confidence_weighted":
            return self._confidence_weighted(valid)
        elif strategy == "bayesian":
            return self._bayesian_aggregation(valid)
        else:
            raise AggregationError(
                f"Unknown aggregation strategy: {strategy}"
            )

    def compute_aggregate_confidence(
        self,
        predictions: List[Dict],
    ) -> float:
        """Compute confidence in the aggregated prediction.

        Args:
            predictions: List of prediction dicts.

        Returns:
            Float confidence in [0, 1].
        """
        if not predictions:
            return 0.0

        confidences = [
            p.get("confidence", 0.0) for p in predictions
        ]
        if not confidences:
            return 0.0

        # Weighted average favoring higher-confidence experts
        weights = np.array(confidences)
        total_weight = weights.sum()
        if total_weight == 0:
            return 0.0

        weighted = float(
            np.sum(weights * weights) / total_weight
        )
        return float(np.clip(weighted, 0.0, 1.0))

    def weight_predictions_by_score(
        self,
        predictions: List[Dict],
        expert_scores: Dict[str, float],
    ) -> List[Dict]:
        """Weight predictions by expert match scores.

        Args:
            predictions: List of prediction dicts.
            expert_scores: Mapping of expert name to match score.

        Returns:
            Predictions with 'weight' key added.
        """
        result = []
        for pred in predictions:
            weighted = dict(pred)
            name = pred.get("expert_name", "")
            score = expert_scores.get(name, 0.5)
            weighted["weight"] = score
            result.append(weighted)
        return result

    def detect_prediction_disagreement(
        self,
        predictions: List[Dict],
    ) -> Tuple[float, List[int]]:
        """Quantify disagreement among expert predictions.

        Args:
            predictions: List of prediction dicts.

        Returns:
            Tuple of (disagreement_level, indices of disagreeing
            experts). Disagreement is in [0, 1].
        """
        if len(predictions) <= 1:
            return 0.0, []

        actions = []
        for p in predictions:
            pred = p.get("prediction", {})
            if isinstance(pred, dict):
                actions.append(
                    pred.get("recommended_action", "unknown")
                )
            else:
                actions.append("unknown")

        unique = set(actions)
        if len(unique) <= 1:
            return 0.0, []

        # Disagreement = 1 - (max_agreement / total)
        from collections import Counter
        counts = Counter(actions)
        max_count = counts.most_common(1)[0][1]
        disagreement = 1.0 - (max_count / len(actions))

        # Find indices of experts that disagree with majority
        majority_action = counts.most_common(1)[0][0]
        disagreeing = [
            i
            for i, a in enumerate(actions)
            if a != majority_action
        ]

        return float(disagreement), disagreeing

    # ---- internals ------------------------------------------------

    @staticmethod
    def _ensemble_voting(
        predictions: List[Dict],
    ) -> Dict[str, Any]:
        """Majority vote aggregation."""
        from collections import Counter

        actions = []
        for p in predictions:
            pred = p.get("prediction", {})
            if isinstance(pred, dict):
                actions.append(
                    pred.get("recommended_action", "unknown")
                )
            else:
                actions.append("unknown")

        counts = Counter(actions)
        winner = counts.most_common(1)[0][0]

        # Average confidence of experts voting for winner
        winner_confs = [
            p.get("confidence", 0.0)
            for p, a in zip(predictions, actions)
            if a == winner
        ]
        avg_conf = (
            float(np.mean(winner_confs)) if winner_confs else 0.0
        )

        return {
            "recommended_action": winner,
            "vote_counts": dict(counts),
            "aggregation_method": "ensemble_voting",
            "consensus_confidence": avg_conf,
        }

    @staticmethod
    def _confidence_weighted(
        predictions: List[Dict],
    ) -> Dict[str, Any]:
        """Confidence-weighted aggregation."""
        from collections import defaultdict

        action_weights: Dict[str, float] = defaultdict(float)
        for p in predictions:
            pred = p.get("prediction", {})
            conf = p.get("confidence", 0.0)
            if isinstance(pred, dict):
                action = pred.get(
                    "recommended_action", "unknown"
                )
            else:
                action = "unknown"
            action_weights[action] += conf

        winner = max(action_weights, key=action_weights.get)
        total = sum(action_weights.values())
        weighted_conf = (
            action_weights[winner] / total if total > 0 else 0.0
        )

        return {
            "recommended_action": winner,
            "action_weights": dict(action_weights),
            "aggregation_method": "confidence_weighted",
            "consensus_confidence": float(weighted_conf),
        }

    @staticmethod
    def _bayesian_aggregation(
        predictions: List[Dict],
    ) -> Dict[str, Any]:
        """Simplified Bayesian aggregation."""
        from collections import defaultdict

        # Compute log-likelihood for each action
        action_log_prob: Dict[str, float] = defaultdict(float)
        for p in predictions:
            pred = p.get("prediction", {})
            conf = max(p.get("confidence", 0.01), 0.01)
            if isinstance(pred, dict):
                action = pred.get(
                    "recommended_action", "unknown"
                )
            else:
                action = "unknown"
            action_log_prob[action] += float(np.log(conf))

        # Convert to probabilities
        log_probs = np.array(list(action_log_prob.values()))
        log_probs -= np.max(log_probs)  # numerical stability
        probs = np.exp(log_probs)
        probs /= probs.sum()

        actions = list(action_log_prob.keys())
        best_idx = int(np.argmax(probs))
        winner = actions[best_idx]

        return {
            "recommended_action": winner,
            "posterior_probs": {
                a: float(p)
                for a, p in zip(actions, probs)
            },
            "aggregation_method": "bayesian",
            "consensus_confidence": float(probs[best_idx]),
        }


# ------------------------------------------------------------------
# ExpertConfidenceCompute
# ------------------------------------------------------------------


class ExpertConfidenceCompute:
    """Compute confidence scores for expert predictions."""

    def compute_confidence(
        self,
        prediction: Dict,
        expert_metadata: Dict,
    ) -> float:
        """Compute per-prediction confidence.

        Args:
            prediction: A prediction dict.
            expert_metadata: Metadata about the expert.

        Returns:
            Confidence score in [0, 1].
        """
        raw_conf = prediction.get("confidence", 0.0)
        match_score = expert_metadata.get("match_score", 0.5)

        # Combine raw confidence with expert match quality
        combined = 0.6 * raw_conf + 0.4 * match_score
        return float(np.clip(combined, 0.0, 1.0))

    def compute_confidence_intervals(
        self,
        predictions: List[Dict],
    ) -> Dict[str, Tuple[float, float]]:
        """Compute confidence intervals per expert.

        Args:
            predictions: List of prediction dicts with
                'expert_name' and 'confidence'.

        Returns:
            Dict mapping expert name to (low, high) tuple.
        """
        intervals: Dict[str, Tuple[float, float]] = {}
        for pred in predictions:
            name = pred.get("expert_name", "unknown")
            conf = pred.get("confidence", 0.5)
            # Simple interval based on confidence
            margin = max(0.1, (1.0 - conf) * 0.3)
            low = max(0.0, conf - margin)
            high = min(1.0, conf + margin)
            intervals[name] = (round(low, 4), round(high, 4))
        return intervals

    def rank_predictions_by_confidence(
        self,
        predictions: List[Dict],
    ) -> List[Tuple[str, Dict, float]]:
        """Rank predictions by confidence score.

        Args:
            predictions: List of prediction dicts.

        Returns:
            Sorted list of (expert_name, prediction, confidence).
        """
        ranked = []
        for pred in predictions:
            name = pred.get("expert_name", "unknown")
            conf = pred.get("confidence", 0.0)
            ranked.append((name, pred, conf))
        ranked.sort(key=lambda x: x[2], reverse=True)
        return ranked


# ------------------------------------------------------------------
# MultiExpertInferencePipeline
# ------------------------------------------------------------------


class MultiExpertInferencePipeline:
    """Orchestrates Phase 2.4 multi-expert inference.

    Usage:
        >>> pipeline = MultiExpertInferencePipeline()
        >>> result = pipeline.infer(
        ...     text, selected_experts, expert_configs
        ... )
    """

    def __init__(
        self,
        config: Optional[Phase2Config] = None,
        max_workers: int = 4,
    ) -> None:
        self._config = config or Phase2Config()
        self._executor = ParallelInferenceExecutor(
            config, max_workers
        )
        self._aggregator = PredictionAggregator()
        self._confidence = ExpertConfidenceCompute()
        self._metadata: Dict[str, Any] = {}

    def infer(
        self,
        text: str,
        selected_experts: List[RankedExpert],
        expert_configs: Optional[Dict] = None,
        aggregation_strategy: str = "ensemble_voting",
    ) -> InferenceResult:
        """Run full multi-expert inference pipeline.

        Args:
            text: Input text to process.
            selected_experts: Experts from Phase 2.3.
            expert_configs: Optional per-expert configurations.
            aggregation_strategy: Aggregation method.

        Returns:
            InferenceResult with all inference outputs.
        """
        start = time.perf_counter()
        warnings: List[str] = []

        logger.info(
            "Starting Phase 2.4 inference for %d experts",
            len(selected_experts),
        )

        if not selected_experts:
            warnings.append("No experts selected for inference")
            return InferenceResult(
                text=text,
                warnings=warnings,
                processing_time_ms=(
                    (time.perf_counter() - start) * 1000.0
                ),
            )

        # Step 1: Parallel inference
        raw_results = self._executor.execute_parallel(
            text, selected_experts, expert_configs
        )

        # Step 2: Collect predictions
        predictions = self._executor.collect_predictions(
            raw_results
        )

        # Step 3: Compute statistics
        stats = self._executor.compute_prediction_statistics(
            predictions
        )

        # Step 4: Detect outliers
        outliers = self._executor.detect_prediction_outliers(
            predictions
        )
        if outliers:
            warnings.append(
                f"Detected {len(outliers)} outlier prediction(s)"
            )

        # Step 5: Aggregate predictions
        aggregated = self._aggregator.aggregate_predictions(
            predictions, aggregation_strategy
        )

        # Step 6: Compute ensemble confidence
        ensemble_conf = (
            self._aggregator.compute_aggregate_confidence(
                predictions
            )
        )

        # Step 7: Compute per-expert confidences
        pred_confidences: Dict[str, float] = {}
        for pred in predictions:
            name = pred.get("expert_name", "unknown")
            pred_confidences[name] = pred.get(
                "confidence", 0.0
            )

        # Step 8: Detect disagreement
        disagreement, _ = (
            self._aggregator.detect_prediction_disagreement(
                predictions
            )
        )
        if disagreement > 0.5:
            warnings.append(
                f"High prediction disagreement: "
                f"{disagreement:.2f}"
            )

        elapsed = (time.perf_counter() - start) * 1000.0
        self._metadata = {
            "total_experts": len(selected_experts),
            "successful_predictions": len(
                [p for p in predictions if p.get("prediction")]
            ),
            "aggregation_strategy": aggregation_strategy,
            "processing_time_ms": elapsed,
        }

        logger.info(
            "Phase 2.4 complete: ensemble_conf=%.3f "
            "disagreement=%.3f time=%.1fms",
            ensemble_conf,
            disagreement,
            elapsed,
        )

        return InferenceResult(
            text=text,
            selected_experts=[e.name for e in selected_experts],
            predictions_per_expert=raw_results,
            aggregated_prediction=aggregated,
            prediction_confidences=pred_confidences,
            ensemble_confidence=ensemble_conf,
            prediction_statistics=stats,
            outlier_indices=outliers,
            prediction_disagreement=disagreement,
            processing_time_ms=elapsed,
            warnings=warnings,
        )

    def get_processing_metadata(self) -> Dict[str, Any]:
        """Return timing and execution metadata.

        Returns:
            Dict with processing metadata.
        """
        return dict(self._metadata)
