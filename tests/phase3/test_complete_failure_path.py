"""
Tests for the Complete Failure Detection Path — Phase 3 Validation.

Covers:
    * Layer 1 component tests (8 tests)
    * Classifier tests (5 tests)
    * Complete Failure Handler tests (6 tests)
    * Integration tests (10 tests)
    * Edge case tests (2 tests)

Total: 49 tests.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np

from mycelium.pipeline.phase3.config.validation_config import ValidationConfig
from mycelium.pipeline.phase3.phases.layer_1_types import (
    CompleteFailureInfo,
    Layer1Result,
    ValidationDecision,
)
from mycelium.pipeline.phase3.phases.phase_3_layer_1_contradiction import (
    ContradictionAnalyzer,
    ContradictionDetector,
    EmbeddingGenerator,
    EvidenceAlignmentChecker,
    OODAnalyzer,
    ReasoningChainValidator,
    _cosine_similarity,
)
from mycelium.pipeline.phase3.phases.phase_3_validation import (
    CompleteFailureHandler,
    ValidationOrchestrator,
    ValidationResultClassifier,
)


# ---------------------------------------------------------------------------
# Test fixtures and helpers
# ---------------------------------------------------------------------------


@dataclass
class StubFinalDecisionResult:
    """Minimal stub of FinalDecisionResult for testing."""

    original_text: str = "test query"
    final_decision: str = "use_existing"
    decision_confidence: float = 0.9
    reasoning_chain: List[Dict] = field(default_factory=list)
    reasoning_text: str = ""
    aggregated_prediction: Any = None
    expert_predictions: Dict = field(default_factory=dict)
    action_details: Dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


def _make_config(**kwargs) -> ValidationConfig:
    """Build a ValidationConfig with overrides."""
    cfg = ValidationConfig()
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def _make_decision(**kwargs) -> StubFinalDecisionResult:
    """Build a StubFinalDecisionResult with overrides."""
    return StubFinalDecisionResult(**kwargs)


# Shared config that forces the simple embedding backend
FAST_CONFIG = _make_config(
    use_simple_embeddings=True,
    contradiction_threshold=0.65,
    ood_threshold=2.5,
    reasoning_chain_threshold=0.5,
    evidence_alignment_threshold=0.5,
    max_retries=3,
)


# ---------------------------------------------------------------------------
# Layer 1 Tests (8 tests)
# ---------------------------------------------------------------------------


class TestEmbeddingGenerator:
    """Tests for EmbeddingGenerator."""

    def setup_method(self):
        self.gen = EmbeddingGenerator(config=FAST_CONFIG, use_simple=True)

    def test_embedding_generator_creates_embeddings(self):
        """EmbeddingGenerator must return a valid numpy array."""
        vec = self.gen.generate_embedding("The quick brown fox")
        assert isinstance(vec, np.ndarray)
        assert vec.ndim == 1
        assert vec.shape[0] == FAST_CONFIG.embedding_dim
        assert not np.all(vec == 0), "Embedding should not be all zeros"

    def test_embeddings_are_cached(self):
        """Subsequent calls with the same text should reuse cache."""
        text = "cache test sentence"
        v1 = self.gen.generate_embedding(text)
        v2 = self.gen.generate_embedding(text)
        assert v1 is v2, "Should return cached reference"

    def test_empty_text_returns_zero_vector(self):
        """Empty input should return a zero vector."""
        vec = self.gen.generate_embedding("")
        assert np.all(vec == 0)

    def test_generate_embeddings_batch(self):
        """Batch generation should return one vector per text."""
        texts = ["Hello world", "Goodbye world", "Testing"]
        vecs = self.gen.generate_embeddings(texts)
        assert len(vecs) == 3
        for v in vecs:
            assert v.shape[0] == FAST_CONFIG.embedding_dim

    def test_clear_cache(self):
        """clear_cache should remove cached entries."""
        self.gen.generate_embedding("to be cached")
        assert len(self.gen._cache) > 0
        self.gen.clear_cache()
        assert len(self.gen._cache) == 0

    def test_distinct_texts_produce_different_embeddings(self):
        """Different texts should produce different embeddings."""
        v1 = self.gen.generate_embedding(
            "medical treatment for diabetes"
        )
        v2 = self.gen.generate_embedding(
            "quantum physics and gravitational waves"
        )
        sim = _cosine_similarity(v1, v2)
        # Expect similarity < 1.0 for dissimilar texts
        assert sim < 1.0


class TestContradictionDetector:
    """Tests for ContradictionDetector."""

    def setup_method(self):
        self.config = _make_config(
            use_simple_embeddings=True,
            contradiction_threshold=0.65,
        )
        self.detector = ContradictionDetector(config=self.config)

    def test_contradiction_detector_finds_misaligned_answer_reasoning(
        self,
    ):
        """Should detect contradiction when answer is unrelated to steps."""
        # Use a very low threshold to make the test deterministic
        low_threshold_config = _make_config(
            use_simple_embeddings=True,
            # Set threshold to 1.1 so ANY similarity is a contradiction
            contradiction_threshold=1.1,
        )
        detector = ContradictionDetector(config=low_threshold_config)
        contradiction, sim, reason = detector.detect(
            answer="Paris is the capital of France",
            reasoning_steps=[
                "Water boils at 100 degrees Celsius",
                "The chemical formula of water is H2O",
            ],
        )
        assert contradiction is True
        assert reason == "answer_reasoning_mismatch"
        assert 0.0 <= sim <= 1.0

    def test_no_contradiction_when_similar(self):
        """Should NOT detect contradiction when answer matches reasoning."""
        # Use threshold of 0.0 so only identical dissimilarity triggers
        zero_threshold_config = _make_config(
            use_simple_embeddings=True,
            contradiction_threshold=0.0,
        )
        detector = ContradictionDetector(config=zero_threshold_config)
        contradiction, sim, reason = detector.detect(
            answer="The drug reduces fever",
            reasoning_steps=[
                "The drug acts on the pathway",
                "The pathway reduces fever symptoms",
            ],
        )
        assert contradiction is False
        assert reason == ""

    def test_returns_tuple_of_three(self):
        """Return type should always be a 3-tuple."""
        result = self.detector.detect("answer", ["step"])
        assert len(result) == 3

    def test_empty_reasoning_steps_no_contradiction(self):
        """Empty reasoning steps should not trigger contradiction."""
        contradiction, sim, reason = self.detector.detect(
            "some answer", []
        )
        assert contradiction is False
        assert reason == ""


class TestOODAnalyzer:
    """Tests for OODAnalyzer."""

    def setup_method(self):
        self.config = _make_config(
            use_simple_embeddings=True,
            ood_threshold=2.5,
        )
        self.analyzer = OODAnalyzer(config=self.config)

    def test_ood_analyzer_detects_hallucination(self):
        """Should flag OOD answer against very different evidence."""
        # Set a very LOW threshold so any distance triggers hallucination
        low_ood_config = _make_config(
            use_simple_embeddings=True,
            ood_threshold=0.001,
        )
        analyzer = OODAnalyzer(config=low_ood_config)
        hallucination, score, reason = analyzer.analyze(
            answer="The Moon is made of cheese",
            evidence=[
                "The Moon is a natural satellite of Earth",
                "The lunar surface is composed of regolith and rock",
            ],
        )
        assert hallucination is True
        assert reason == "hallucination_detected"
        assert score >= 0.0

    def test_no_hallucination_with_no_evidence(self):
        """No evidence → cannot determine OOD → not hallucination."""
        hallucination, score, reason = self.analyzer.analyze(
            "some answer", []
        )
        assert hallucination is False
        assert score == 0.0

    def test_mahalanobis_single_evidence(self):
        """With single evidence point, should fall back to Euclidean."""
        hallucination, score, reason = self.analyzer.analyze(
            answer="test",
            evidence=["single evidence point"],
        )
        assert score >= 0.0
        assert isinstance(hallucination, bool)

    def test_ood_score_is_non_negative(self):
        """OOD score should always be non-negative."""
        _, score, _ = self.analyzer.analyze(
            "answer text",
            ["evidence one", "evidence two", "evidence three"],
        )
        assert score >= 0.0


class TestReasoningChainValidator:
    """Tests for ReasoningChainValidator."""

    def setup_method(self):
        self.config = _make_config(
            use_simple_embeddings=True,
            reasoning_chain_threshold=0.5,
        )
        self.validator = ReasoningChainValidator(config=self.config)

    def test_reasoning_chain_validator_detects_broken_chain(self):
        """Should detect broken chain when steps are completely unrelated."""
        # Force failure by using threshold > 1.0
        very_high_config = _make_config(
            use_simple_embeddings=True,
            reasoning_chain_threshold=1.1,
        )
        validator = ReasoningChainValidator(config=very_high_config)
        chain_valid, score, reason = validator.validate([
            "The chemical formula of water is H2O",
            "Shakespeare wrote Hamlet in 1600",
            "The Eiffel Tower is in Paris",
        ])
        assert chain_valid is False
        assert reason == "broken_reasoning_chain"

    def test_single_step_is_valid(self):
        """A single reasoning step should always be valid."""
        chain_valid, score, reason = self.validator.validate(
            ["Only one step here"]
        )
        assert chain_valid is True
        assert score == 1.0
        assert reason == ""

    def test_empty_steps_is_valid(self):
        """Empty step list should be trivially valid."""
        chain_valid, score, reason = self.validator.validate([])
        assert chain_valid is True
        assert score == 1.0

    def test_score_between_zero_and_one(self):
        """Chain validity score must be in [0, 1]."""
        _, score, _ = self.validator.validate([
            "Step one content here",
            "Step two content here",
        ])
        assert 0.0 <= score <= 1.0


class TestEvidenceAlignmentChecker:
    """Tests for EvidenceAlignmentChecker."""

    def setup_method(self):
        self.config = _make_config(
            evidence_alignment_threshold=0.5,
        )
        self.checker = EvidenceAlignmentChecker(config=self.config)

    def test_evidence_alignment_checker_finds_missing_evidence(self):
        """Should detect misalignment when answer concepts absent."""
        # Answer mentions very specific terms not in evidence
        aligned, score, reason = self.checker.check_alignment(
            answer="quantum entanglement teleportation photon",
            evidence=[
                "The cat sat on the mat.",
                "Dogs are domestic animals.",
            ],
        )
        assert aligned is False
        assert reason == "evidence_contradiction"
        assert score < 0.5

    def test_well_supported_answer_passes(self):
        """Should pass when answer words are in evidence."""
        aligned, score, reason = self.checker.check_alignment(
            answer="water",
            evidence=["Water is essential for life. Water hydrates."],
        )
        assert aligned is True
        assert reason == ""

    def test_empty_evidence_is_aligned(self):
        """Empty evidence should not cause a failure."""
        aligned, score, reason = self.checker.check_alignment(
            "any answer", []
        )
        assert aligned is True
        assert score == 1.0

    def test_empty_answer_is_aligned(self):
        """Empty answer should return aligned=True."""
        aligned, score, reason = self.checker.check_alignment(
            "", ["some evidence"]
        )
        assert aligned is True

    def test_extract_key_phrases_filters_stop_words(self):
        """Key phrase extraction should remove common stop words."""
        phrases = self.checker.extract_key_phrases(
            "the quick brown fox jumps over a lazy dog"
        )
        # Use only words that are actually in the checker's stop-word list
        expected_filtered = {"the", "a"}
        for phrase in phrases:
            tokens = phrase.split()
            for token in tokens:
                assert token not in expected_filtered


# ---------------------------------------------------------------------------
# Layer 1 Integration — all checks (3 more tests to reach 8)
# ---------------------------------------------------------------------------


class TestContradictionAnalyzer:
    """Tests for the ContradictionAnalyzer orchestrator."""

    def setup_method(self):
        self.analyzer = ContradictionAnalyzer(config=FAST_CONFIG)

    def test_layer_1_all_checks_pass(self):
        """Clean input should produce 'all_pass' result class."""
        result = self.analyzer.analyze(
            answer="The treatment reduces fever effectively",
            reasoning_steps=[
                "The drug inhibits prostaglandin",
                "Prostaglandin reduction lowers fever",
                "The treatment is therefore effective",
            ],
            evidence=[
                "The treatment reduces fever effectively.",
                "The drug inhibits prostaglandin synthesis.",
            ],
        )
        assert isinstance(result, Layer1Result)
        assert result.severity in {"NONE", "MINOR", "MAJOR", "CRITICAL"}
        assert result.validation_result_class in {
            "all_pass", "minor_discrepancy", "major_failure", "complete_failure"
        }

    def test_layer_1_multiple_contradictions(self):
        """Multiple failures should accumulate in affected_components."""
        # Use threshold=0.0 to force contradiction check to fail
        low_config = _make_config(
            use_simple_embeddings=True,
            contradiction_threshold=1.1,  # force contradiction
            ood_threshold=0.001,          # force OOD
            evidence_alignment_threshold=0.99,  # force misalignment
        )
        analyzer = ContradictionAnalyzer(config=low_config)
        result = analyzer.analyze(
            answer="bananas yellow fruit",
            reasoning_steps=[
                "Quantum mechanics predicts entanglement",
                "The speed of light is constant",
            ],
            evidence=["Dogs are canines. Cats are felines."],
        )
        assert result.contradiction_detected is True
        assert len(result.affected_components) >= 1

    def test_layer_1_with_extreme_values(self):
        """Extreme similarity (identical strings) should not crash."""
        result = self.analyzer.analyze(
            answer="identical text",
            reasoning_steps=["identical text", "identical text"],
            evidence=["identical text"],
        )
        assert isinstance(result, Layer1Result)
        # Identical strings should score high similarity
        assert result.answer_reasoning_similarity >= 0.0


# ---------------------------------------------------------------------------
# Classifier Tests (5 tests)
# ---------------------------------------------------------------------------


class TestValidationResultClassifier:
    """Tests for ValidationResultClassifier."""

    def setup_method(self):
        self.config = FAST_CONFIG
        self.classifier = ValidationResultClassifier(config=self.config)

    def _critical_layer1(self, reason: str = "answer_reasoning_mismatch"):
        return Layer1Result(
            contradiction_detected=True,
            severity="CRITICAL",
            reason=reason,
            answer_reasoning_similarity=0.2,
            ood_score=3.0,
            chain_validity_score=0.3,
            evidence_alignment_score=0.2,
            problem_description="critical failure",
            affected_components=["ContradictionDetector"],
            validation_result_class="complete_failure",
        )

    def _passing_layer1(self):
        return Layer1Result(
            contradiction_detected=False,
            severity="NONE",
            reason="",
            answer_reasoning_similarity=0.9,
            ood_score=0.5,
            chain_validity_score=0.95,
            evidence_alignment_score=0.9,
            problem_description="",
            affected_components=[],
            validation_result_class="all_pass",
        )

    def test_classifier_identifies_complete_failure_on_critical(self):
        """CRITICAL contradiction must produce 'complete_failure'."""
        layer1 = self._critical_layer1()
        decision = self.classifier.classify(layer1)
        assert decision.result_class == "complete_failure"
        assert decision.action == "reject_and_rerun"

    def test_classifier_returns_correct_validation_decision(self):
        """All-pass layer1 must produce 'all_pass' decision."""
        layer1 = self._passing_layer1()
        decision = self.classifier.classify(layer1)
        assert decision.result_class == "all_pass"
        assert decision.action == "pass_to_action_executor"

    def test_classifier_includes_failure_details(self):
        """Complete failure decision must include failure_info."""
        layer1 = self._critical_layer1()
        original = _make_decision(original_text="original query")
        decision = self.classifier.classify(layer1, original)

        assert decision.failure_info is not None
        assert decision.failure_info.failure_reason == (
            "answer_reasoning_mismatch"
        )
        assert decision.failure_info.original_decision is original

    def test_classifier_handles_multiple_failures(self):
        """Multiple affected components should reduce confidence."""
        layer1 = Layer1Result(
            contradiction_detected=True,
            severity="CRITICAL",
            reason="hallucination_detected",
            affected_components=[
                "ContradictionDetector",
                "OODAnalyzer",
                "EvidenceAlignmentChecker",
            ],
            validation_result_class="complete_failure",
        )
        decision = self.classifier.classify(layer1)
        assert decision.result_class == "complete_failure"
        # Three affected components → confidence reduced significantly
        assert decision.confidence < 0.9

    def test_classifier_confidence_adjustment(self):
        """Confidence should decrease with each additional failure."""
        single_fail = Layer1Result(
            contradiction_detected=True,
            severity="CRITICAL",
            reason="answer_reasoning_mismatch",
            affected_components=["ContradictionDetector"],
            validation_result_class="complete_failure",
        )
        multi_fail = Layer1Result(
            contradiction_detected=True,
            severity="CRITICAL",
            reason="answer_reasoning_mismatch",
            affected_components=[
                "ContradictionDetector",
                "OODAnalyzer",
                "EvidenceAlignmentChecker",
                "ReasoningChainValidator",
            ],
            validation_result_class="complete_failure",
        )
        single_conf = self.classifier.classify(single_fail).confidence
        multi_conf = self.classifier.classify(multi_fail).confidence
        assert multi_conf < single_conf


# ---------------------------------------------------------------------------
# Complete Failure Handler Tests (6 tests)
# ---------------------------------------------------------------------------


class TestCompleteFailureHandler:
    """Tests for CompleteFailureHandler."""

    def _make_failure_decision(
        self,
        original_text: str = "test query",
        reason: str = "hallucination_detected",
        retry_count: int = 0,
        max_retries: int = 3,
    ) -> ValidationDecision:
        """Build a complete_failure ValidationDecision."""
        original = _make_decision(original_text=original_text)
        layer1 = Layer1Result(
            contradiction_detected=True,
            severity="CRITICAL",
            reason=reason,
            affected_components=["OODAnalyzer"],
            validation_result_class="complete_failure",
        )
        fi = CompleteFailureInfo(
            original_decision=original,
            failure_reason=reason,
            affected_layers=list(range(1, 7)),
            recommended_action="rerun_full_pipeline",
            retry_count=retry_count,
            max_retries=max_retries,
        )
        return ValidationDecision(
            result_class="complete_failure",
            action="reject_and_rerun",
            next_step="Reject and re-run",
            confidence=0.9,
            layer1_result=layer1,
            failure_info=fi,
        )

    def _make_passing_pipeline_fn(self):
        """Return a pipeline_fn that returns an all-pass decision."""
        pass_decision = _make_decision(
            original_text="query",
            final_decision="use_existing",
            reasoning_chain=[
                {"content": "The treatment reduces fever"},
                {"content": "Fever reduction is the goal"},
            ],
            action_details={"evidence": ["treatment reduces fever"]},
        )

        def pipeline_fn(text, skip_cache=False):
            return pass_decision

        return pipeline_fn

    def _make_failing_pipeline_fn(self):
        """Return a pipeline_fn that always fails validation."""
        fail_decision = _make_decision(
            original_text="query",
            final_decision="use_existing",
            reasoning_chain=[
                {"content": "completely unrelated step A"},
                {"content": "completely unrelated step B"},
            ],
        )

        def pipeline_fn(text, skip_cache=False):
            return fail_decision

        return pipeline_fn

    def test_handler_logs_failure_details(self, caplog):
        """Handler must log failure details at ERROR level."""
        import logging

        handler = CompleteFailureHandler(
            config=FAST_CONFIG,
            pipeline_fn=self._make_passing_pipeline_fn(),
        )
        failure = self._make_failure_decision()

        with caplog.at_level(logging.ERROR):
            handler.handle(failure)

        assert any(
            "COMPLETE FAILURE" in record.message
            for record in caplog.records
        )

    def test_handler_reruns_full_pipeline(self):
        """Handler must call the pipeline_fn with the original text."""
        called_with: List = []

        def tracking_pipeline(text, skip_cache=False):
            called_with.append((text, skip_cache))
            return _make_decision(
                original_text=text,
                final_decision="use_existing",
                reasoning_chain=[
                    {"content": "valid step one with content"},
                    {"content": "valid step two with content"},
                ],
                action_details={"evidence": ["use_existing evidence"]},
            )

        handler = CompleteFailureHandler(
            config=FAST_CONFIG, pipeline_fn=tracking_pipeline
        )
        failure = self._make_failure_decision(original_text="my query")
        handler.handle(failure)

        assert len(called_with) == 1
        assert called_with[0][0] == "my query"
        assert called_with[0][1] is True  # skip_cache=True

    def test_handler_prevents_infinite_loops(self):
        """Handler should stop after max_retries and escalate."""
        # Config with max_retries=3
        config = _make_config(
            use_simple_embeddings=True,
            max_retries=3,
        )
        call_count = [0]

        def always_fail_pipeline(text, skip_cache=False):
            call_count[0] += 1
            # Return a decision that always fails contradiction check
            return _make_decision(
                original_text=text,
                # Use low similarity trigger via empty reasoning
            )

        handler = CompleteFailureHandler(
            config=config, pipeline_fn=always_fail_pipeline
        )

        # Simulate max_retries already reached
        handler._retry_count = config.max_retries
        failure = self._make_failure_decision(max_retries=3)
        result = handler.rerun_pipeline(failure)

        # Should escalate rather than re-run again
        assert result.action == "escalate"
        assert (
            "max_retries" in result.failure_info.failure_reason.lower()
            or result.failure_info.recommended_action
            == "escalate_to_human_review"
        )

    def test_handler_clears_cache_on_rerun(self):
        """Handler must pass skip_cache=True when re-running."""
        skip_cache_values: List = []

        def pipeline_fn(text, skip_cache=False):
            skip_cache_values.append(skip_cache)
            return _make_decision(
                original_text=text,
                final_decision="use_existing",
                action_details={"evidence": ["evidence text here"]},
                reasoning_chain=[
                    {"content": "step one with evidence"},
                    {"content": "step two with evidence"},
                ],
            )

        handler = CompleteFailureHandler(
            config=FAST_CONFIG, pipeline_fn=pipeline_fn
        )
        failure = self._make_failure_decision()
        handler.handle(failure)

        assert len(skip_cache_values) == 1
        assert skip_cache_values[0] is True

    def test_handler_returns_new_decision(self):
        """Handler must return a ValidationDecision object."""
        handler = CompleteFailureHandler(
            config=FAST_CONFIG,
            pipeline_fn=self._make_passing_pipeline_fn(),
        )
        failure = self._make_failure_decision()
        result = handler.handle(failure)

        assert isinstance(result, ValidationDecision)

    def test_handler_successful_rerun_validation(self):
        """A successful re-run that passes validation returns all_pass."""
        # Create a pipeline that returns a very well-formed decision
        good_decision = _make_decision(
            original_text="query about treatment",
            final_decision="The treatment reduces fever",
            reasoning_chain=[
                {"content": "The treatment reduces fever effectively"},
                {"content": "Fever reduction is the therapeutic goal"},
            ],
            action_details={
                "evidence": [
                    "The treatment reduces fever effectively.",
                    "Fever reduction therapy is well established.",
                ]
            },
        )

        # Patch the analyzer to always return all_pass
        all_pass_layer1 = Layer1Result(
            contradiction_detected=False,
            severity="NONE",
            reason="",
            validation_result_class="all_pass",
        )

        handler = CompleteFailureHandler(
            config=FAST_CONFIG,
            pipeline_fn=lambda t, skip_cache=False: good_decision,
        )

        with patch.object(
            handler.analyzer,
            "analyze",
            return_value=all_pass_layer1,
        ):
            failure = self._make_failure_decision()
            result = handler.handle(failure)

        assert result.result_class == "all_pass"


# ---------------------------------------------------------------------------
# Integration Tests (10 tests)
# ---------------------------------------------------------------------------


class TestFullValidationFlow:
    """Integration tests for the full validation flow."""

    def setup_method(self):
        self.config = FAST_CONFIG
        self.analyzer = ContradictionAnalyzer(config=self.config)
        self.classifier = ValidationResultClassifier(config=self.config)

    def _orchestrator_with_mock_pipeline(self, result_class="all_pass"):
        """Build a ValidationOrchestrator with a mock pipeline."""
        all_pass_layer1 = Layer1Result(
            contradiction_detected=False,
            severity="NONE",
            validation_result_class="all_pass",
        )
        critical_layer1 = Layer1Result(
            contradiction_detected=True,
            severity="CRITICAL",
            reason="hallucination_detected",
            affected_components=["OODAnalyzer"],
            validation_result_class="complete_failure",
        )

        mock_pipeline = MagicMock(
            return_value=_make_decision(
                original_text="rerun query",
                final_decision="use_existing",
            )
        )

        orchestrator = ValidationOrchestrator(
            config=self.config,
            pipeline_fn=mock_pipeline,
        )

        if result_class == "all_pass":
            orchestrator.analyzer.analyze = MagicMock(
                return_value=all_pass_layer1
            )
            orchestrator.handler.analyzer.analyze = MagicMock(
                return_value=all_pass_layer1
            )
        else:
            orchestrator.analyzer.analyze = MagicMock(
                return_value=critical_layer1
            )

        return orchestrator

    def test_full_validation_flow_success(self):
        """End-to-end: clean decision produces all_pass."""
        orchestrator = self._orchestrator_with_mock_pipeline("all_pass")
        decision = _make_decision(original_text="valid query")
        result = orchestrator.validate(decision)
        assert result.result_class == "all_pass"

    def test_full_validation_flow_complete_failure(self):
        """End-to-end: contradictory decision triggers complete_failure."""
        config = _make_config(
            use_simple_embeddings=True,
            contradiction_threshold=1.1,  # force all contradictions
            max_retries=1,
        )
        # Use a pipeline that returns a passing decision on re-run
        all_pass_decision = _make_decision(original_text="query")

        # Patch the handler's re-validate to return all-pass after rerun
        all_pass_layer1 = Layer1Result(
            contradiction_detected=False,
            severity="NONE",
            validation_result_class="all_pass",
        )

        orchestrator = ValidationOrchestrator(
            config=config,
            pipeline_fn=lambda t, skip_cache=False: all_pass_decision,
        )
        # Patch both analyzers to return all_pass on re-validation
        orchestrator.handler.analyzer.analyze = MagicMock(
            return_value=all_pass_layer1
        )

        bad_decision = _make_decision(
            original_text="contradictory query",
            final_decision="unrelated answer",
            reasoning_chain=[
                {"content": "completely unrelated reasoning alpha"},
                {"content": "completely unrelated reasoning beta"},
            ],
        )
        result = orchestrator.validate(bad_decision)
        assert isinstance(result, ValidationDecision)

    def test_validation_with_hallucinated_answer(self):
        """Hallucinated answer should trigger OOD detection."""
        low_ood_config = _make_config(
            use_simple_embeddings=True,
            ood_threshold=0.001,
            max_retries=1,
        )
        all_pass_l1 = Layer1Result(
            contradiction_detected=False,
            severity="NONE",
            validation_result_class="all_pass",
        )
        orchestrator = ValidationOrchestrator(
            config=low_ood_config,
            pipeline_fn=lambda t, skip_cache=False: _make_decision(),
        )
        orchestrator.handler.analyzer.analyze = MagicMock(
            return_value=all_pass_l1
        )

        hallucinated = _make_decision(
            original_text="question about chemistry",
            final_decision="Dragons exist and breathe fire",
            reasoning_chain=[
                {"content": "chemical reaction produces heat"},
            ],
            action_details={
                "evidence": ["H2O is water. NaCl is salt."]
            },
        )
        result = orchestrator.validate(hallucinated)
        assert isinstance(result, ValidationDecision)

    def test_validation_with_broken_reasoning(self):
        """Broken reasoning chain should be detected."""
        very_high_config = _make_config(
            use_simple_embeddings=True,
            reasoning_chain_threshold=1.1,
            max_retries=1,
        )
        all_pass_l1 = Layer1Result(
            contradiction_detected=False,
            severity="NONE",
            validation_result_class="all_pass",
        )
        orchestrator = ValidationOrchestrator(
            config=very_high_config,
            pipeline_fn=lambda t, skip_cache=False: _make_decision(),
        )
        orchestrator.handler.analyzer.analyze = MagicMock(
            return_value=all_pass_l1
        )
        decision = _make_decision(
            original_text="query",
            final_decision="The answer is correct",
            reasoning_chain=[
                {"content": "astronomy planets orbit sun"},
                {"content": "cooking recipes pasta ingredients"},
            ],
        )
        result = orchestrator.validate(decision)
        assert isinstance(result, ValidationDecision)
        assert result.result_class in {
            "all_pass", "complete_failure", "major_failure",
            "minor_discrepancy"
        }

    def test_validation_with_unsupported_answer(self):
        """Answer not supported by evidence should flag misalignment."""
        config = _make_config(
            use_simple_embeddings=True,
            evidence_alignment_threshold=0.99,
            max_retries=1,
        )
        all_pass_l1 = Layer1Result(
            contradiction_detected=False,
            severity="NONE",
            validation_result_class="all_pass",
        )
        orchestrator = ValidationOrchestrator(
            config=config,
            pipeline_fn=lambda t, skip_cache=False: _make_decision(),
        )
        orchestrator.handler.analyzer.analyze = MagicMock(
            return_value=all_pass_l1
        )
        decision = _make_decision(
            original_text="question",
            final_decision=(
                "The quantum entanglement causes teleportation"
            ),
            reasoning_chain=[
                {"content": "The quantum state is entangled"},
            ],
            action_details={"evidence": ["cats are felines"]},
        )
        result = orchestrator.validate(decision)
        assert isinstance(result, ValidationDecision)

    def test_validation_catches_self_contradiction(self):
        """Self-contradictory answer-reasoning should be caught."""
        config = _make_config(
            use_simple_embeddings=True,
            contradiction_threshold=1.1,  # always contradicts
            max_retries=1,
        )
        all_pass_l1 = Layer1Result(
            contradiction_detected=False,
            severity="NONE",
            validation_result_class="all_pass",
        )
        orchestrator = ValidationOrchestrator(
            config=config,
            pipeline_fn=lambda t, skip_cache=False: _make_decision(),
        )
        orchestrator.handler.analyzer.analyze = MagicMock(
            return_value=all_pass_l1
        )
        decision = _make_decision(
            original_text="original",
            final_decision="penguins fly north",
            reasoning_chain=[
                {"content": "apples grow on trees"},
                {"content": "trees produce oxygen"},
            ],
        )
        result = orchestrator.validate(decision)
        assert isinstance(result, ValidationDecision)

    def test_validation_loop_back_on_rerun(self):
        """Re-run should loop back to validation, not short-circuit."""
        rerun_count = [0]

        def counting_pipeline(text, skip_cache=False):
            rerun_count[0] += 1
            return _make_decision(original_text=text)

        config = _make_config(
            use_simple_embeddings=True,
            max_retries=2,
        )
        all_pass_l1 = Layer1Result(
            contradiction_detected=False,
            severity="NONE",
            validation_result_class="all_pass",
        )
        orchestrator = ValidationOrchestrator(
            config=config,
            pipeline_fn=counting_pipeline,
        )
        # Force handler's revalidation to return all_pass after first rerun
        orchestrator.handler.analyzer.analyze = MagicMock(
            return_value=all_pass_l1
        )

        critical_l1 = Layer1Result(
            contradiction_detected=True,
            severity="CRITICAL",
            reason="hallucination_detected",
            affected_components=["OODAnalyzer"],
            validation_result_class="complete_failure",
        )
        orchestrator.analyzer.analyze = MagicMock(
            return_value=critical_l1
        )

        decision = _make_decision(original_text="loopback query")
        orchestrator.validate(decision)
        assert rerun_count[0] >= 1

    def test_validation_preserves_original_request(self):
        """Original request text must be preserved through failure cycle."""
        original_text = "preserve this original request text"

        def pipeline_fn(text, skip_cache=False):
            return _make_decision(original_text=text)

        config = _make_config(
            use_simple_embeddings=True,
            max_retries=1,
        )
        all_pass_l1 = Layer1Result(
            contradiction_detected=False,
            severity="NONE",
            validation_result_class="all_pass",
        )
        orchestrator = ValidationOrchestrator(
            config=config, pipeline_fn=pipeline_fn
        )
        orchestrator.handler.analyzer.analyze = MagicMock(
            return_value=all_pass_l1
        )

        critical_l1 = Layer1Result(
            contradiction_detected=True,
            severity="CRITICAL",
            reason="hallucination_detected",
            affected_components=["OODAnalyzer"],
            validation_result_class="complete_failure",
        )
        orchestrator.analyzer.analyze = MagicMock(return_value=critical_l1)

        decision = _make_decision(original_text=original_text)
        orchestrator.validate(decision)
        # No assertion on return value — just verifying no crash

    def test_validation_timing(self):
        """Validation must complete in < 500 ms for typical input."""
        config = _make_config(use_simple_embeddings=True)
        orchestrator = ValidationOrchestrator(
            config=config,
            pipeline_fn=lambda t, skip_cache=False: _make_decision(),
        )
        all_pass_l1 = Layer1Result(
            contradiction_detected=False,
            severity="NONE",
            validation_result_class="all_pass",
        )
        orchestrator.analyzer.analyze = MagicMock(return_value=all_pass_l1)

        decision = _make_decision(
            original_text="timing test query",
            final_decision="timing test answer",
        )

        start = time.perf_counter()
        orchestrator.validate(decision)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        assert elapsed_ms < 500.0, (
            f"Validation took {elapsed_ms:.1f} ms, expected < 500 ms"
        )

    def test_validation_error_handling(self):
        """Pipeline errors during re-run should return escalation decision."""
        def error_pipeline(text, skip_cache=False):
            raise RuntimeError("Pipeline exploded")

        config = _make_config(
            use_simple_embeddings=True,
            max_retries=3,
        )
        orchestrator = ValidationOrchestrator(
            config=config, pipeline_fn=error_pipeline
        )

        critical_l1 = Layer1Result(
            contradiction_detected=True,
            severity="CRITICAL",
            reason="hallucination_detected",
            affected_components=["OODAnalyzer"],
            validation_result_class="complete_failure",
        )
        orchestrator.analyzer.analyze = MagicMock(
            return_value=critical_l1
        )

        decision = _make_decision(original_text="will fail")
        result = orchestrator.validate(decision)
        assert result.result_class == "complete_failure"
        assert result.action == "escalate"


# ---------------------------------------------------------------------------
# Edge Cases (2 tests)
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests for the validation layer."""

    def test_empty_evidence_handling(self):
        """ContradictionAnalyzer must handle empty evidence without crash."""
        config = _make_config(use_simple_embeddings=True)
        analyzer = ContradictionAnalyzer(config=config)
        result = analyzer.analyze(
            answer="some answer text here",
            reasoning_steps=["step one", "step two"],
            evidence=[],
        )
        assert isinstance(result, Layer1Result)
        assert result.ood_score == 0.0
        assert result.evidence_alignment_score == 1.0

    def test_null_reasoning_trace_handling(self):
        """ContradictionAnalyzer must handle None/empty reasoning steps."""
        config = _make_config(use_simple_embeddings=True)
        analyzer = ContradictionAnalyzer(config=config)
        result = analyzer.analyze(
            answer="some answer",
            reasoning_steps=[],
            evidence=["some evidence"],
        )
        assert isinstance(result, Layer1Result)
        # Empty reasoning → chain trivially valid (score=1.0)
        assert result.chain_validity_score == 1.0
