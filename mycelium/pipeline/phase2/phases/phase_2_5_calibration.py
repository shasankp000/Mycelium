"""
Phase 2.5 — Confidence Calibration & Uncertainty Quantification.

This module implements prediction calibration, uncertainty
estimation, Bayesian aggregation, and calibration metrics for
the Phase 2 reasoning pipeline.

Components:
    * PredictionCalibrator — temperature scaling calibration
    * UncertaintyQuantifier — epistemic/aleatoric uncertainty
    * BayesianAggregator — Bayesian posterior computation
    * CalibrationMetrics — Brier, log loss, ECE metrics
    * CalibrationPipeline — orchestrator

Example:
    >>> from mycelium.pipeline.phase2.phases.phase_2_5_calibration import (
    ...     CalibrationPipeline,
    ... )
    >>> pipeline = CalibrationPipeline()
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

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Custom exceptions
# ------------------------------------------------------------------


class CalibrationError(Exception):
    """Raised when calibration fails."""


# ------------------------------------------------------------------
# CalibrationResult dataclass
# ------------------------------------------------------------------


@dataclass
class CalibrationResult:
    """Output of Phase 2.5 calibration.

    Attributes:
        original_predictions: Uncalibrated predictions.
        calibrated_predictions: Calibrated predictions.
        calibration_method: Method used for calibration.
        temperature: Temperature parameter.
        epistemic_uncertainty: Model uncertainty dict.
        aleatoric_uncertainty: Data uncertainty dict.
        total_uncertainty: Combined uncertainty dict.
        bayesian_posterior: Posterior distribution dict.
        uncertainty_intervals: Per-expert uncertainty intervals.
        calibration_metrics: ECE, Brier, log loss metrics.
        processing_time_ms: Wall-clock processing time.
        warnings: Non-fatal issues encountered.
    """

    original_predictions: List[Dict] = field(
        default_factory=list
    )
    calibrated_predictions: List[Dict] = field(
        default_factory=list
    )
    calibration_method: str = "temperature_scaling"
    temperature: float = 1.0
    epistemic_uncertainty: Dict = field(default_factory=dict)
    aleatoric_uncertainty: Dict = field(default_factory=dict)
    total_uncertainty: Dict = field(default_factory=dict)
    bayesian_posterior: Dict = field(default_factory=dict)
    uncertainty_intervals: Dict[str, Tuple[float, float]] = (
        field(default_factory=dict)
    )
    calibration_metrics: Dict = field(default_factory=dict)
    processing_time_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)


# ------------------------------------------------------------------
# PredictionCalibrator
# ------------------------------------------------------------------


class PredictionCalibrator:
    """Calibrate predictions using temperature scaling."""

    def calibrate_predictions(
        self,
        raw_predictions: List[Dict],
        calibration_method: str = "temperature_scaling",
    ) -> List[Dict]:
        """Apply calibration to raw predictions.

        Args:
            raw_predictions: List of prediction dicts with
                'confidence' key.
            calibration_method: 'temperature_scaling' or
                'isotonic'.

        Returns:
            List of calibrated prediction dicts.
        """
        if not raw_predictions:
            return []

        temperature = self.compute_calibration_temperature(
            raw_predictions
        )

        calibrated = []
        for pred in raw_predictions:
            cal_pred = dict(pred)
            raw_conf = pred.get("confidence", 0.5)

            if calibration_method == "temperature_scaling":
                cal_conf = self.apply_temperature_scaling(
                    [raw_conf], temperature
                )[0]
            else:
                # Isotonic: simple monotonic mapping
                cal_conf = self._isotonic_calibrate(raw_conf)

            cal_pred["calibrated_confidence"] = cal_conf
            cal_pred["original_confidence"] = raw_conf
            cal_pred["confidence"] = cal_conf
            calibrated.append(cal_pred)

        return calibrated

    def compute_calibration_temperature(
        self,
        predictions: List[Dict],
        ground_truth: Optional[List] = None,
    ) -> float:
        """Compute optimal temperature for scaling.

        Args:
            predictions: List of prediction dicts.
            ground_truth: Optional ground truth labels.

        Returns:
            Optimal temperature value (>= 0.1).
        """
        if not predictions:
            return 1.0

        confidences = np.array(
            [p.get("confidence", 0.5) for p in predictions]
        )

        # Without ground truth, use variance-based heuristic
        if ground_truth is None:
            variance = float(np.var(confidences))
            # Higher variance -> higher temperature (more
            # smoothing)
            temperature = 1.0 + variance * 2.0
        else:
            # With ground truth, minimize NLL
            gt = np.array(ground_truth, dtype=float)
            best_t = 1.0
            best_loss = float("inf")
            for t in np.arange(0.1, 5.0, 0.1):
                scaled = self.apply_temperature_scaling(
                    confidences.tolist(), float(t)
                )
                scaled_arr = np.array(scaled)
                loss = -float(
                    np.mean(
                        gt * np.log(
                            np.clip(scaled_arr, 1e-10, 1.0)
                        )
                        + (1 - gt) * np.log(
                            np.clip(
                                1 - scaled_arr, 1e-10, 1.0
                            )
                        )
                    )
                )
                if loss < best_loss:
                    best_loss = loss
                    best_t = float(t)
            temperature = best_t

        return max(0.1, temperature)

    @staticmethod
    def apply_temperature_scaling(
        logits: List[float],
        temperature: float,
    ) -> List[float]:
        """Apply temperature scaling formula.

        Converts confidence values using softmax-like scaling.

        Args:
            logits: List of confidence values.
            temperature: Temperature parameter (> 0).

        Returns:
            List of scaled confidence values.
        """
        temp = max(temperature, 0.01)
        arr = np.array(logits, dtype=float)

        # Convert to logit space, scale, convert back
        clipped = np.clip(arr, 0.01, 0.99)
        log_odds = np.log(clipped / (1.0 - clipped))
        scaled_log_odds = log_odds / temp
        result = 1.0 / (1.0 + np.exp(-scaled_log_odds))

        return [float(x) for x in result]

    @staticmethod
    def _isotonic_calibrate(confidence: float) -> float:
        """Simple isotonic calibration approximation."""
        # Sigmoid squashing toward 0.5 for overconfident values
        x = confidence
        if x > 0.9:
            return 0.85 + (x - 0.9) * 0.5
        elif x < 0.1:
            return 0.05 + (x - 0.0) * 0.5
        return x


# ------------------------------------------------------------------
# UncertaintyQuantifier
# ------------------------------------------------------------------


class UncertaintyQuantifier:
    """Quantify uncertainty in predictions."""

    def compute_epistemic_uncertainty(
        self,
        predictions: List[Dict],
    ) -> Dict[str, float]:
        """Compute model uncertainty (expert disagreement).

        Args:
            predictions: List of prediction dicts.

        Returns:
            Dict with uncertainty metrics.
        """
        if not predictions:
            return {
                "mean_disagreement": 0.0,
                "variance": 0.0,
                "entropy": 0.0,
            }

        confidences = np.array(
            [p.get("confidence", 0.5) for p in predictions]
        )

        variance = float(np.var(confidences))

        # Shannon entropy of confidence distribution
        clipped = np.clip(confidences, 1e-10, 1.0)
        normalized = clipped / clipped.sum()
        entropy = -float(
            np.sum(normalized * np.log(normalized))
        )

        return {
            "mean_disagreement": float(np.std(confidences)),
            "variance": variance,
            "entropy": entropy,
        }

    def compute_aleatoric_uncertainty(
        self,
        predictions: List[Dict],
    ) -> Dict[str, float]:
        """Compute data uncertainty (prediction variance).

        Args:
            predictions: List of prediction dicts.

        Returns:
            Dict with uncertainty metrics.
        """
        if not predictions:
            return {
                "mean_confidence": 0.0,
                "noise_estimate": 0.0,
                "data_entropy": 0.0,
            }

        confidences = np.array(
            [p.get("confidence", 0.5) for p in predictions]
        )

        mean_conf = float(np.mean(confidences))
        # Aleatoric: average predictive entropy
        clipped = np.clip(confidences, 1e-10, 1.0 - 1e-10)
        entropy = -float(
            np.mean(
                clipped * np.log(clipped)
                + (1 - clipped) * np.log(1 - clipped)
            )
        )

        noise = float(
            np.mean(np.abs(confidences - mean_conf))
        )

        return {
            "mean_confidence": mean_conf,
            "noise_estimate": noise,
            "data_entropy": entropy,
        }

    def compute_total_uncertainty(
        self,
        epistemic: Dict[str, float],
        aleatoric: Dict[str, float],
    ) -> Dict[str, float]:
        """Combine epistemic and aleatoric uncertainty.

        Args:
            epistemic: Epistemic uncertainty dict.
            aleatoric: Aleatoric uncertainty dict.

        Returns:
            Dict with combined uncertainty metrics.
        """
        ep_var = epistemic.get("variance", 0.0)
        al_entropy = aleatoric.get("data_entropy", 0.0)

        total = ep_var + al_entropy
        ratio = (
            ep_var / total if total > 0 else 0.5
        )

        return {
            "total_uncertainty": total,
            "epistemic_fraction": ratio,
            "aleatoric_fraction": 1.0 - ratio,
            "epistemic_variance": ep_var,
            "aleatoric_entropy": al_entropy,
        }

    def compute_uncertainty_intervals(
        self,
        predictions: List[Dict],
    ) -> Dict[str, Tuple[float, float]]:
        """Compute confidence intervals for each prediction.

        Args:
            predictions: List of prediction dicts.

        Returns:
            Dict mapping expert name to (low, high) interval.
        """
        intervals: Dict[str, Tuple[float, float]] = {}

        if not predictions:
            return intervals

        confidences = np.array(
            [p.get("confidence", 0.5) for p in predictions]
        )
        global_std = float(np.std(confidences)) if len(
            confidences
        ) > 1 else 0.1

        for pred in predictions:
            name = pred.get("expert_name", "unknown")
            conf = pred.get("confidence", 0.5)
            margin = max(global_std, 0.05)
            low = max(0.0, conf - margin)
            high = min(1.0, conf + margin)
            intervals[name] = (round(low, 4), round(high, 4))

        return intervals


# ------------------------------------------------------------------
# BayesianAggregator
# ------------------------------------------------------------------


class BayesianAggregator:
    """Bayesian aggregation of expert predictions."""

    def compute_posterior(
        self,
        predictions: List[Dict],
        prior_beliefs: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Compute posterior distribution over predictions.

        Args:
            predictions: List of prediction dicts.
            prior_beliefs: Optional prior probabilities per
                action.

        Returns:
            Dict mapping action to posterior probability.
        """
        if not predictions:
            return {}

        # Uniform prior if not specified
        actions = set()
        for p in predictions:
            pred = p.get("prediction", {})
            if isinstance(pred, dict):
                actions.add(
                    pred.get(
                        "recommended_action",
                        "unknown",
                    )
                )
            else:
                actions.add("unknown")

        if prior_beliefs is None:
            prior_beliefs = {
                a: 1.0 / max(len(actions), 1)
                for a in actions
            }

        # Compute posterior using Bayes' rule
        log_posterior: Dict[str, float] = {}
        for action in actions:
            log_prior = np.log(
                max(prior_beliefs.get(action, 0.01), 1e-10)
            )
            log_likelihood = 0.0
            for p in predictions:
                pred = p.get("prediction", {})
                conf = max(
                    p.get("confidence", 0.01), 0.01
                )
                if isinstance(pred, dict):
                    pred_action = pred.get(
                        "recommended_action", "unknown"
                    )
                else:
                    pred_action = "unknown"

                if pred_action == action:
                    log_likelihood += float(np.log(conf))
                else:
                    log_likelihood += float(
                        np.log(1.0 - conf + 0.01)
                    )

            log_posterior[action] = float(
                log_prior + log_likelihood
            )

        # Normalize
        log_vals = np.array(list(log_posterior.values()))
        log_vals -= np.max(log_vals)
        probs = np.exp(log_vals)
        probs /= probs.sum()

        return {
            a: float(p)
            for a, p in zip(log_posterior.keys(), probs)
        }

    def compute_likelihood(
        self,
        prediction: Dict,
        observation: Any,
    ) -> float:
        """Compute likelihood of observation given prediction.

        Args:
            prediction: A prediction dict.
            observation: Observed outcome.

        Returns:
            Likelihood value in [0, 1].
        """
        pred = prediction.get("prediction", {})
        conf = prediction.get("confidence", 0.5)

        if isinstance(pred, dict):
            pred_action = pred.get(
                "recommended_action", "unknown"
            )
        else:
            pred_action = str(pred)

        if str(observation) == pred_action:
            return float(conf)
        return float(1.0 - conf)

    def compute_bayesian_confidence(
        self,
        posterior: Dict[str, float],
    ) -> float:
        """Compute confidence from posterior distribution.

        Args:
            posterior: Dict mapping action to probability.

        Returns:
            Confidence score in [0, 1].
        """
        if not posterior:
            return 0.0

        probs = list(posterior.values())
        max_prob = max(probs)

        # Confidence is the gap between best and second-best
        sorted_probs = sorted(probs, reverse=True)
        if len(sorted_probs) >= 2:
            gap = sorted_probs[0] - sorted_probs[1]
            confidence = max_prob * (0.5 + 0.5 * gap)
        else:
            confidence = max_prob

        return float(np.clip(confidence, 0.0, 1.0))

    def sample_posterior(
        self,
        posterior: Dict[str, float],
        n_samples: int = 1000,
    ) -> np.ndarray:
        """Sample from posterior for uncertainty estimation.

        Args:
            posterior: Dict mapping action to probability.
            n_samples: Number of samples to draw.

        Returns:
            Array of sampled action indices.
        """
        if not posterior:
            return np.array([])

        actions = list(posterior.keys())
        probs = np.array(list(posterior.values()))
        probs = np.clip(probs, 0.0, 1.0)
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = np.ones(len(probs)) / len(probs)

        samples = np.random.choice(
            len(actions), size=n_samples, p=probs
        )
        return samples


# ------------------------------------------------------------------
# CalibrationMetrics
# ------------------------------------------------------------------


class CalibrationMetrics:
    """Compute metrics for calibration quality."""

    def compute_brier_score(
        self,
        predictions: List[Dict],
        ground_truth: List,
    ) -> float:
        """Compute Brier score (MSE of probabilities).

        Args:
            predictions: List of prediction dicts.
            ground_truth: List of binary ground truth values.

        Returns:
            Brier score (lower is better).
        """
        if not predictions or not ground_truth:
            return 0.0

        n = min(len(predictions), len(ground_truth))
        confs = np.array(
            [
                predictions[i].get("confidence", 0.5)
                for i in range(n)
            ]
        )
        gt = np.array(ground_truth[:n], dtype=float)

        return float(np.mean((confs - gt) ** 2))

    def compute_log_loss(
        self,
        predictions: List[Dict],
        ground_truth: List,
    ) -> float:
        """Compute log loss (cross-entropy).

        Args:
            predictions: List of prediction dicts.
            ground_truth: List of binary ground truth values.

        Returns:
            Log loss value (lower is better).
        """
        if not predictions or not ground_truth:
            return 0.0

        n = min(len(predictions), len(ground_truth))
        confs = np.array(
            [
                predictions[i].get("confidence", 0.5)
                for i in range(n)
            ]
        )
        gt = np.array(ground_truth[:n], dtype=float)

        clipped = np.clip(confs, 1e-10, 1.0 - 1e-10)
        loss = -float(
            np.mean(
                gt * np.log(clipped)
                + (1 - gt) * np.log(1 - clipped)
            )
        )
        return loss

    def compute_ece(
        self,
        predictions: List[Dict],
        ground_truth: List,
        n_bins: int = 10,
    ) -> float:
        """Compute Expected Calibration Error.

        Args:
            predictions: List of prediction dicts.
            ground_truth: List of binary ground truth values.
            n_bins: Number of bins for calibration.

        Returns:
            ECE value in [0, 1] (lower is better).
        """
        if not predictions or not ground_truth:
            return 0.0

        n = min(len(predictions), len(ground_truth))
        confs = np.array(
            [
                predictions[i].get("confidence", 0.5)
                for i in range(n)
            ]
        )
        gt = np.array(ground_truth[:n], dtype=float)

        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            mask = (confs >= bin_boundaries[i]) & (
                confs < bin_boundaries[i + 1]
            )
            if not np.any(mask):
                continue

            bin_conf = float(np.mean(confs[mask]))
            bin_acc = float(np.mean(gt[mask]))
            bin_size = int(np.sum(mask))

            ece += (bin_size / n) * abs(bin_acc - bin_conf)

        return float(ece)

    def compute_calibration_report(
        self,
        predictions: List[Dict],
        ground_truth: Optional[List] = None,
    ) -> Dict[str, Any]:
        """Generate full calibration metrics report.

        Args:
            predictions: List of prediction dicts.
            ground_truth: Optional ground truth labels.

        Returns:
            Dict with all calibration metrics.
        """
        report: Dict[str, Any] = {
            "n_predictions": len(predictions),
        }

        if predictions:
            confs = [
                p.get("confidence", 0.5)
                for p in predictions
            ]
            report["mean_confidence"] = float(np.mean(confs))
            report["std_confidence"] = float(np.std(confs))
            report["min_confidence"] = float(np.min(confs))
            report["max_confidence"] = float(np.max(confs))

        if ground_truth and predictions:
            report["brier_score"] = self.compute_brier_score(
                predictions, ground_truth
            )
            report["log_loss"] = self.compute_log_loss(
                predictions, ground_truth
            )
            report["ece"] = self.compute_ece(
                predictions, ground_truth
            )
        else:
            report["brier_score"] = None
            report["log_loss"] = None
            report["ece"] = None

        return report


# ------------------------------------------------------------------
# CalibrationPipeline
# ------------------------------------------------------------------


class CalibrationPipeline:
    """Orchestrates Phase 2.5 calibration and uncertainty.

    Usage:
        >>> pipeline = CalibrationPipeline()
        >>> result = pipeline.calibrate_and_quantify(
        ...     inference_result
        ... )
    """

    def __init__(
        self,
        config: Optional[Phase2Config] = None,
    ) -> None:
        self._config = config or Phase2Config()
        self._calibrator = PredictionCalibrator()
        self._uncertainty = UncertaintyQuantifier()
        self._bayesian = BayesianAggregator()
        self._metrics = CalibrationMetrics()
        self._metadata: Dict[str, Any] = {}

    def calibrate_and_quantify(
        self,
        inference_result: InferenceResult,
        calibration_method: str = "temperature_scaling",
        uncertainty_method: str = "bayesian",
    ) -> CalibrationResult:
        """Run full calibration and uncertainty pipeline.

        Args:
            inference_result: Output from Phase 2.4.
            calibration_method: Calibration approach.
            uncertainty_method: Uncertainty approach.

        Returns:
            CalibrationResult with all outputs.
        """
        start = time.perf_counter()
        warnings: List[str] = []

        logger.info("Starting Phase 2.5 calibration")

        # Build prediction list from inference result
        raw_predictions = []
        for name, pred_dict in (
            inference_result.predictions_per_expert.items()
        ):
            entry = dict(pred_dict)
            entry["expert_name"] = name
            raw_predictions.append(entry)

        if not raw_predictions:
            warnings.append("No predictions to calibrate")
            return CalibrationResult(
                warnings=warnings,
                processing_time_ms=(
                    (time.perf_counter() - start) * 1000.0
                ),
            )

        # Step 1: Calibrate predictions
        calibrated = self._calibrator.calibrate_predictions(
            raw_predictions, calibration_method
        )

        temperature = (
            self._calibrator.compute_calibration_temperature(
                raw_predictions
            )
        )

        # Step 2: Compute uncertainties
        epistemic = (
            self._uncertainty.compute_epistemic_uncertainty(
                calibrated
            )
        )
        aleatoric = (
            self._uncertainty.compute_aleatoric_uncertainty(
                calibrated
            )
        )
        total = self._uncertainty.compute_total_uncertainty(
            epistemic, aleatoric
        )

        # Step 3: Uncertainty intervals
        intervals = (
            self._uncertainty.compute_uncertainty_intervals(
                calibrated
            )
        )

        # Step 4: Bayesian posterior
        posterior = self._bayesian.compute_posterior(calibrated)

        # Step 5: Calibration metrics (no ground truth)
        metrics = self._metrics.compute_calibration_report(
            calibrated
        )

        elapsed = (time.perf_counter() - start) * 1000.0
        self._metadata = {
            "calibration_method": calibration_method,
            "uncertainty_method": uncertainty_method,
            "n_predictions": len(calibrated),
            "processing_time_ms": elapsed,
        }

        logger.info(
            "Phase 2.5 complete: temp=%.2f "
            "total_uncertainty=%.4f time=%.1fms",
            temperature,
            total.get("total_uncertainty", 0.0),
            elapsed,
        )

        return CalibrationResult(
            original_predictions=raw_predictions,
            calibrated_predictions=calibrated,
            calibration_method=calibration_method,
            temperature=temperature,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            total_uncertainty=total,
            bayesian_posterior=posterior,
            uncertainty_intervals=intervals,
            calibration_metrics=metrics,
            processing_time_ms=elapsed,
            warnings=warnings,
        )

    def get_processing_metadata(self) -> Dict[str, Any]:
        """Return timing metadata.

        Returns:
            Dict with processing metadata.
        """
        return dict(self._metadata)
