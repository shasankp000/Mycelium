"""Tests for Phase 2.4 — Multi-Expert Inference."""

import pytest

from mycelium.pipeline.phase2.phases.phase_2_4_inference import (
    AggregationError,
    ExpertConfidenceCompute,
    ExpertInference,
    InferenceResult,
    MultiExpertInferencePipeline,
    ParallelInferenceExecutor,
    PredictionAggregator,
)
from mycelium.pipeline.phase2.utils.semantic_types import RankedExpert


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_ranked(name, domain, score):
    """Build a RankedExpert with defaults."""
    return RankedExpert(
        name=name,
        domain=domain,
        match_score=score,
        confidence_low=max(score - 0.1, 0.0),
        confidence_high=min(score + 0.1, 1.0),
        ranking_factors={"domain": score},
    )


def _make_prediction(
    expert_name, confidence, action="use_existing",
):
    """Build a prediction dict matching ExpertInference output."""
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


SAMPLE_TEXT = (
    "The molecular structure of aspirin interacts with "
    "cyclooxygenase enzymes to reduce inflammation"
)


# ------------------------------------------------------------------
# ExpertInference
# ------------------------------------------------------------------


class TestExpertInference:
    def setup_method(self):
        self.engine = ExpertInference()

    def test_single_expert_prediction(self):
        expert = _make_ranked("med_1", "medical", 0.85)
        result = self.engine.predict(SAMPLE_TEXT, expert)
        assert "prediction" in result
        assert "confidence" in result
        assert "metadata" in result
        assert "latency_ms" in result
        assert 0.0 <= result["confidence"] <= 1.0

    def test_expert_batch_prediction(self):
        expert = _make_ranked("chem_1", "chemistry", 0.8)
        texts = [
            "Aspirin reduces pain",
            "Benzene is aromatic",
            "Protein folding is complex",
        ]
        results = self.engine.predict_batch(texts, expert)
        assert len(results) == 3
        for r in results:
            assert "prediction" in r
            assert "confidence" in r

    def test_prediction_metadata(self):
        expert = _make_ranked("phys_1", "physics", 0.75)
        meta = self.engine.get_prediction_metadata(expert)
        assert meta["expert_name"] == "phys_1"
        assert meta["domain"] == "physics"
        assert meta["match_score"] == 0.75
        assert "confidence_interval" in meta
        assert "ranking_factors" in meta


# ------------------------------------------------------------------
# ParallelInferenceExecutor
# ------------------------------------------------------------------


class TestParallelInferenceExecutor:
    def setup_method(self):
        self.executor = ParallelInferenceExecutor()

    def test_parallel_inference_execution(self):
        experts = [
            _make_ranked("e1", "medical", 0.9),
            _make_ranked("e2", "physics", 0.7),
            _make_ranked("e3", "chemistry", 0.8),
        ]
        results = self.executor.execute_parallel(
            SAMPLE_TEXT, experts,
        )
        assert len(results) == 3
        for name in ("e1", "e2", "e3"):
            assert name in results
            assert "prediction" in results[name]

    def test_concurrent_expert_inference(self):
        experts = [
            _make_ranked(f"exp_{i}", "general", 0.5 + i * 0.05)
            for i in range(6)
        ]
        results = self.executor.execute_parallel(
            SAMPLE_TEXT, experts,
        )
        assert len(results) == 6
        for e in experts:
            assert e.name in results

    def test_prediction_collection(self):
        experts = [
            _make_ranked("a", "medical", 0.8),
            _make_ranked("b", "physics", 0.7),
        ]
        raw = self.executor.execute_parallel(
            SAMPLE_TEXT, experts,
        )
        collected = self.executor.collect_predictions(raw)
        assert isinstance(collected, list)
        assert len(collected) == 2
        for p in collected:
            assert "expert_name" in p

    def test_prediction_statistics(self):
        preds = [
            _make_prediction("x1", 0.9),
            _make_prediction("x2", 0.7),
            _make_prediction("x3", 0.5),
        ]
        stats = self.executor.compute_prediction_statistics(
            preds,
        )
        for key in ("mean", "std", "min", "max", "median", "iqr"):
            assert key in stats
        assert stats["min"] == pytest.approx(0.5)
        assert stats["max"] == pytest.approx(0.9)

    def test_outlier_detection(self):
        preds = [
            _make_prediction("n1", 0.80),
            _make_prediction("n2", 0.82),
            _make_prediction("n3", 0.78),
            _make_prediction("n4", 0.81),
            _make_prediction("outlier", 0.05),
        ]
        outliers = self.executor.detect_prediction_outliers(
            preds,
        )
        assert len(outliers) >= 1
        assert 4 in outliers


# ------------------------------------------------------------------
# PredictionAggregator
# ------------------------------------------------------------------


class TestPredictionAggregator:
    def setup_method(self):
        self.agg = PredictionAggregator()

    def test_ensemble_voting_aggregation(self):
        preds = [
            _make_prediction("a", 0.8, "use_existing"),
            _make_prediction("b", 0.7, "use_existing"),
            _make_prediction("c", 0.6, "create_patch"),
        ]
        result = self.agg.aggregate_predictions(
            preds, "ensemble_voting",
        )
        assert result["recommended_action"] == "use_existing"
        assert result["aggregation_method"] == "ensemble_voting"
        assert "vote_counts" in result

    def test_confidence_weighted_aggregation(self):
        preds = [
            _make_prediction("a", 0.9, "use_existing"),
            _make_prediction("b", 0.3, "create_patch"),
        ]
        result = self.agg.aggregate_predictions(
            preds, "confidence_weighted",
        )
        assert "recommended_action" in result
        assert (
            result["aggregation_method"]
            == "confidence_weighted"
        )
        assert "action_weights" in result

    def test_bayesian_aggregation(self):
        preds = [
            _make_prediction("a", 0.85, "use_existing"),
            _make_prediction("b", 0.80, "use_existing"),
        ]
        result = self.agg.aggregate_predictions(
            preds, "bayesian",
        )
        assert result["recommended_action"] == "use_existing"
        assert result["aggregation_method"] == "bayesian"
        assert "posterior_probs" in result

    def test_aggregate_confidence_computation(self):
        preds = [
            _make_prediction("a", 0.8),
            _make_prediction("b", 0.6),
            _make_prediction("c", 0.9),
        ]
        conf = self.agg.compute_aggregate_confidence(preds)
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    def test_prediction_disagreement_detection(self):
        preds = [
            _make_prediction("a", 0.8, "use_existing"),
            _make_prediction("b", 0.7, "create_patch"),
            _make_prediction("c", 0.6, "create_new_expert"),
        ]
        level, indices = (
            self.agg.detect_prediction_disagreement(preds)
        )
        assert level > 0.0
        assert len(indices) > 0


# ------------------------------------------------------------------
# ExpertConfidenceCompute
# ------------------------------------------------------------------


class TestExpertConfidenceCompute:
    def setup_method(self):
        self.cc = ExpertConfidenceCompute()

    def test_confidence_computation(self):
        pred = _make_prediction("e1", 0.8)
        meta = {"match_score": 0.9}
        conf = self.cc.compute_confidence(pred, meta)
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    def test_confidence_intervals(self):
        preds = [
            _make_prediction("a", 0.8),
            _make_prediction("b", 0.5),
        ]
        intervals = self.cc.compute_confidence_intervals(preds)
        assert "a" in intervals
        assert "b" in intervals
        for name, (low, high) in intervals.items():
            assert low <= high
            assert 0.0 <= low
            assert high <= 1.0

    def test_rank_predictions_by_confidence(self):
        preds = [
            _make_prediction("low", 0.3),
            _make_prediction("high", 0.95),
            _make_prediction("mid", 0.6),
        ]
        ranked = self.cc.rank_predictions_by_confidence(preds)
        assert len(ranked) == 3
        assert ranked[0][0] == "high"
        assert ranked[-1][0] == "low"
        confs = [r[2] for r in ranked]
        assert confs == sorted(confs, reverse=True)


# ------------------------------------------------------------------
# MultiExpertInferencePipeline
# ------------------------------------------------------------------


class TestMultiExpertInferencePipeline:
    def setup_method(self):
        self.pipeline = MultiExpertInferencePipeline()

    def test_full_inference_pipeline(self):
        experts = [
            _make_ranked("e1", "medical", 0.85),
            _make_ranked("e2", "chemistry", 0.75),
        ]
        result = self.pipeline.infer(SAMPLE_TEXT, experts)
        assert isinstance(result, InferenceResult)
        assert result.text == SAMPLE_TEXT
        assert len(result.selected_experts) == 2
        assert len(result.predictions_per_expert) == 2
        assert result.aggregated_prediction is not None
        assert 0.0 <= result.ensemble_confidence <= 1.0
        assert result.processing_time_ms > 0

    def test_inference_with_medical_experts(self):
        experts = [
            _make_ranked("biobert", "medical", 0.9),
        ]
        result = self.pipeline.infer(
            "Patient shows elevated troponin levels",
            experts,
        )
        assert isinstance(result, InferenceResult)
        assert "biobert" in result.selected_experts
        assert "biobert" in result.predictions_per_expert

    def test_inference_with_physics_experts(self):
        experts = [
            _make_ranked("phys_net", "physics", 0.88),
        ]
        result = self.pipeline.infer(
            "Schwarzschild radius of a black hole",
            experts,
        )
        assert isinstance(result, InferenceResult)
        assert "phys_net" in result.selected_experts
        assert result.ensemble_confidence > 0

    def test_inference_with_mixed_experts(self):
        experts = [
            _make_ranked("med", "medical", 0.8),
            _make_ranked("phys", "physics", 0.7),
            _make_ranked("chem", "chemistry", 0.6),
        ]
        result = self.pipeline.infer(
            SAMPLE_TEXT, experts,
        )
        assert len(result.selected_experts) == 3
        assert len(result.prediction_confidences) == 3
        assert isinstance(
            result.prediction_statistics, dict,
        )

    def test_inference_no_experts(self):
        result = self.pipeline.infer(SAMPLE_TEXT, [])
        assert isinstance(result, InferenceResult)
        assert len(result.warnings) > 0
        assert result.selected_experts == []
        assert result.predictions_per_expert == {}

    def test_processing_metadata(self):
        experts = [
            _make_ranked("m1", "medical", 0.8),
        ]
        self.pipeline.infer(SAMPLE_TEXT, experts)
        meta = self.pipeline.get_processing_metadata()
        assert "total_experts" in meta
        assert "processing_time_ms" in meta
        assert meta["total_experts"] == 1

    def test_inference_result_statistics_keys(self):
        experts = [
            _make_ranked("s1", "medical", 0.9),
            _make_ranked("s2", "physics", 0.6),
            _make_ranked("s3", "chemistry", 0.7),
        ]
        result = self.pipeline.infer(SAMPLE_TEXT, experts)
        stats = result.prediction_statistics
        for key in ("mean", "std", "min", "max", "median"):
            assert key in stats


# ------------------------------------------------------------------
# Edge cases and error handling
# ------------------------------------------------------------------


class TestEdgeCases:
    def test_unknown_aggregation_strategy_raises(self):
        agg = PredictionAggregator()
        preds = [_make_prediction("a", 0.8)]
        with pytest.raises(AggregationError):
            agg.aggregate_predictions(
                preds, "nonexistent_strategy",
            )

    def test_empty_predictions_aggregate(self):
        agg = PredictionAggregator()
        result = agg.aggregate_predictions([], "ensemble_voting")
        assert "recommended_action" in result

    def test_empty_predictions_confidence(self):
        agg = PredictionAggregator()
        conf = agg.compute_aggregate_confidence([])
        assert conf == 0.0

    def test_parallel_executor_empty_experts(self):
        executor = ParallelInferenceExecutor()
        results = executor.execute_parallel(SAMPLE_TEXT, [])
        assert results == {}
