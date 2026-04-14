"""Tests for Phase 2.5 — Confidence Calibration & Uncertainty."""

import numpy as np
import pytest

from phase2_validation.phases.phase_2_4_inference import (
    InferenceResult,
)
from phase2_validation.phases.phase_2_5_calibration import (
    BayesianAggregator,
    CalibrationError,
    CalibrationMetrics,
    CalibrationPipeline,
    CalibrationResult,
    PredictionCalibrator,
    UncertaintyQuantifier,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_prediction(
    expert_name, confidence, action="use_existing",
):
    """Build a prediction dict for calibration tests."""
    return {
        "prediction": {
            "recommended_action": action,
            "domain": "test",
            "relevance_score": confidence,
        },
        "confidence": confidence,
        "metadata": {
            "expert_name": expert_name,
            "domain": "test",
            "match_score": confidence,
        },
        "latency_ms": 1.0,
        "expert_name": expert_name,
    }


def _make_inference_result(n_experts=3):
    """Build an InferenceResult with *n_experts* entries."""
    predictions_per_expert = {}
    for i in range(n_experts):
        name = f"expert_{i}"
        predictions_per_expert[name] = {
            "prediction": {
                "recommended_action": "use_existing",
                "domain": f"domain_{i}",
            },
            "confidence": 0.5 + i * 0.1,
            "metadata": {"expert_name": name},
            "latency_ms": 1.0,
        }
    return InferenceResult(
        text="test text",
        selected_experts=[
            f"expert_{i}" for i in range(n_experts)
        ],
        predictions_per_expert=predictions_per_expert,
        ensemble_confidence=0.7,
        prediction_disagreement=0.1,
    )


# ------------------------------------------------------------------
# PredictionCalibrator
# ------------------------------------------------------------------


class TestPredictionCalibrator:
    def setup_method(self):
        self.calibrator = PredictionCalibrator()

    def test_temperature_scaling(self):
        logits = [0.8, 0.5, 0.3]
        result = self.calibrator.apply_temperature_scaling(
            logits, 1.5,
        )
        assert len(result) == 3
        for val in result:
            assert isinstance(val, float)
            assert 0.0 <= val <= 1.0

    def test_calibration_temperature_computation(self):
        preds = [
            _make_prediction("a", 0.9),
            _make_prediction("b", 0.6),
            _make_prediction("c", 0.3),
        ]
        temp = (
            self.calibrator
            .compute_calibration_temperature(preds)
        )
        assert isinstance(temp, float)
        assert temp >= 0.1

    def test_calibrate_predictions(self):
        preds = [
            _make_prediction("x", 0.85),
            _make_prediction("y", 0.55),
        ]
        calibrated = self.calibrator.calibrate_predictions(
            preds,
        )
        assert len(calibrated) == 2
        for cp in calibrated:
            assert "calibrated_confidence" in cp
            assert "original_confidence" in cp
            assert 0.0 <= cp["confidence"] <= 1.0

    def test_calibrate_with_ground_truth(self):
        preds = [
            _make_prediction("a", 0.9),
            _make_prediction("b", 0.4),
            _make_prediction("c", 0.7),
        ]
        gt = [1.0, 0.0, 1.0]
        temp = (
            self.calibrator
            .compute_calibration_temperature(preds, gt)
        )
        assert isinstance(temp, float)
        assert temp >= 0.1


# ------------------------------------------------------------------
# UncertaintyQuantifier
# ------------------------------------------------------------------


class TestUncertaintyQuantifier:
    def setup_method(self):
        self.uq = UncertaintyQuantifier()

    def test_epistemic_uncertainty_computation(self):
        preds = [
            _make_prediction("a", 0.9),
            _make_prediction("b", 0.3),
            _make_prediction("c", 0.6),
        ]
        result = (
            self.uq
            .compute_epistemic_uncertainty(preds)
        )
        for key in (
            "mean_disagreement", "variance", "entropy",
        ):
            assert key in result
        assert result["variance"] >= 0.0
        assert result["entropy"] >= 0.0

    def test_aleatoric_uncertainty_computation(self):
        preds = [
            _make_prediction("a", 0.8),
            _make_prediction("b", 0.7),
        ]
        result = (
            self.uq
            .compute_aleatoric_uncertainty(preds)
        )
        for key in (
            "mean_confidence",
            "noise_estimate",
            "data_entropy",
        ):
            assert key in result
        assert result["mean_confidence"] > 0.0
        assert result["noise_estimate"] >= 0.0

    def test_total_uncertainty_computation(self):
        epistemic = {
            "mean_disagreement": 0.2,
            "variance": 0.04,
            "entropy": 0.5,
        }
        aleatoric = {
            "mean_confidence": 0.7,
            "noise_estimate": 0.1,
            "data_entropy": 0.3,
        }
        result = self.uq.compute_total_uncertainty(
            epistemic, aleatoric,
        )
        for key in (
            "total_uncertainty",
            "epistemic_fraction",
            "aleatoric_fraction",
        ):
            assert key in result
        assert result["total_uncertainty"] >= 0.0
        assert pytest.approx(
            result["epistemic_fraction"]
            + result["aleatoric_fraction"],
            abs=1e-9,
        ) == 1.0

    def test_uncertainty_intervals(self):
        preds = [
            _make_prediction("e1", 0.8),
            _make_prediction("e2", 0.6),
            _make_prediction("e3", 0.7),
        ]
        intervals = (
            self.uq
            .compute_uncertainty_intervals(preds)
        )
        assert isinstance(intervals, dict)
        assert len(intervals) == 3
        for name, (low, high) in intervals.items():
            assert low <= high
            assert 0.0 <= low
            assert high <= 1.0

    def test_empty_predictions_uncertainty(self):
        ep = (
            self.uq
            .compute_epistemic_uncertainty([])
        )
        al = (
            self.uq
            .compute_aleatoric_uncertainty([])
        )
        assert ep["mean_disagreement"] == 0.0
        assert ep["variance"] == 0.0
        assert ep["entropy"] == 0.0
        assert al["mean_confidence"] == 0.0
        assert al["noise_estimate"] == 0.0
        assert al["data_entropy"] == 0.0


# ------------------------------------------------------------------
# BayesianAggregator
# ------------------------------------------------------------------


class TestBayesianAggregator:
    def setup_method(self):
        self.ba = BayesianAggregator()

    def test_posterior_computation(self):
        preds = [
            _make_prediction("a", 0.8, "use_existing"),
            _make_prediction("b", 0.7, "use_existing"),
            _make_prediction("c", 0.5, "create_patch"),
        ]
        posterior = self.ba.compute_posterior(preds)
        assert isinstance(posterior, dict)
        assert len(posterior) > 0
        total = sum(posterior.values())
        assert pytest.approx(total, abs=1e-6) == 1.0

    def test_posterior_with_prior(self):
        preds = [
            _make_prediction("a", 0.8, "use_existing"),
            _make_prediction("b", 0.6, "create_patch"),
        ]
        prior = {
            "use_existing": 0.7,
            "create_patch": 0.3,
        }
        posterior = self.ba.compute_posterior(
            preds, prior_beliefs=prior,
        )
        assert isinstance(posterior, dict)
        total = sum(posterior.values())
        assert pytest.approx(total, abs=1e-6) == 1.0

    def test_likelihood_computation(self):
        pred = _make_prediction("a", 0.8, "use_existing")
        match = self.ba.compute_likelihood(
            pred, "use_existing",
        )
        mismatch = self.ba.compute_likelihood(
            pred, "create_patch",
        )
        assert isinstance(match, float)
        assert isinstance(mismatch, float)
        assert match > mismatch

    def test_bayesian_confidence(self):
        posterior = {
            "use_existing": 0.8,
            "create_patch": 0.2,
        }
        conf = self.ba.compute_bayesian_confidence(
            posterior,
        )
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    def test_posterior_sampling(self):
        posterior = {
            "use_existing": 0.6,
            "create_patch": 0.4,
        }
        samples = self.ba.sample_posterior(
            posterior, n_samples=500,
        )
        assert isinstance(samples, np.ndarray)
        assert len(samples) == 500


# ------------------------------------------------------------------
# CalibrationMetrics
# ------------------------------------------------------------------


class TestCalibrationMetrics:
    def setup_method(self):
        self.cm = CalibrationMetrics()

    def test_brier_score(self):
        preds = [
            _make_prediction("a", 0.9),
            _make_prediction("b", 0.3),
            _make_prediction("c", 0.7),
        ]
        gt = [1.0, 0.0, 1.0]
        score = self.cm.compute_brier_score(preds, gt)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_log_loss(self):
        preds = [
            _make_prediction("a", 0.9),
            _make_prediction("b", 0.2),
        ]
        gt = [1.0, 0.0]
        loss = self.cm.compute_log_loss(preds, gt)
        assert isinstance(loss, float)
        assert loss > 0.0

    def test_ece_computation(self):
        preds = [
            _make_prediction(f"e{i}", 0.5 + i * 0.05)
            for i in range(10)
        ]
        gt = [
            1.0 if i % 2 == 0 else 0.0
            for i in range(10)
        ]
        ece = self.cm.compute_ece(preds, gt, n_bins=10)
        assert isinstance(ece, float)
        assert 0.0 <= ece <= 1.0

    def test_full_calibration_report(self):
        preds = [
            _make_prediction("a", 0.8),
            _make_prediction("b", 0.6),
        ]
        gt = [1.0, 0.0]
        report = self.cm.compute_calibration_report(
            preds, gt,
        )
        for key in (
            "n_predictions",
            "mean_confidence",
            "std_confidence",
            "min_confidence",
            "max_confidence",
            "brier_score",
            "log_loss",
            "ece",
        ):
            assert key in report


# ------------------------------------------------------------------
# CalibrationPipeline
# ------------------------------------------------------------------


class TestCalibrationPipeline:
    def setup_method(self):
        self.pipeline = CalibrationPipeline()

    def test_full_calibration_pipeline(self):
        ir = _make_inference_result(3)
        result = self.pipeline.calibrate_and_quantify(ir)
        assert isinstance(result, CalibrationResult)
        assert len(result.calibrated_predictions) == 3
        assert len(result.original_predictions) == 3
        assert result.temperature >= 0.1
        assert result.processing_time_ms > 0.0
        assert (
            result.calibration_method
            == "temperature_scaling"
        )

    def test_calibration_with_uncertainty(self):
        ir = _make_inference_result(4)
        result = self.pipeline.calibrate_and_quantify(ir)
        assert "variance" in result.epistemic_uncertainty
        assert (
            "mean_confidence"
            in result.aleatoric_uncertainty
        )
        assert (
            "total_uncertainty"
            in result.total_uncertainty
        )
        assert isinstance(
            result.uncertainty_intervals, dict,
        )
        assert len(result.uncertainty_intervals) > 0

    def test_empty_inference_result(self):
        ir = InferenceResult()
        result = self.pipeline.calibrate_and_quantify(ir)
        assert isinstance(result, CalibrationResult)
        assert len(result.warnings) > 0
        assert result.calibrated_predictions == []

    def test_processing_metadata(self):
        ir = _make_inference_result(2)
        self.pipeline.calibrate_and_quantify(ir)
        meta = self.pipeline.get_processing_metadata()
        assert "calibration_method" in meta
        assert "n_predictions" in meta
        assert "processing_time_ms" in meta
        assert meta["n_predictions"] == 2

    def test_bayesian_posterior_populated(self):
        ir = _make_inference_result(3)
        result = self.pipeline.calibrate_and_quantify(ir)
        assert isinstance(
            result.bayesian_posterior, dict,
        )
        if result.bayesian_posterior:
            total = sum(
                result.bayesian_posterior.values()
            )
            assert pytest.approx(
                total, abs=1e-6,
            ) == 1.0

    def test_calibration_metrics_populated(self):
        ir = _make_inference_result(3)
        result = self.pipeline.calibrate_and_quantify(ir)
        assert isinstance(
            result.calibration_metrics, dict,
        )
        assert (
            "n_predictions"
            in result.calibration_metrics
        )
        assert (
            result.calibration_metrics["n_predictions"]
            == 3
        )
