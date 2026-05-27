# tests/test_predicate_pipeline.py
# Integration tests for PredicatePipeline (Stage 6).
#
# These tests exercise the full wired pipeline:
#   NLPPreprocessor (mocked/bypassed) -> PredicateExtractor -> PredicateNegator
#
# spaCy subprocess is NOT invoked.  NLPPreprocessor is replaced with a
# lightweight stub that returns a hand-crafted SentenceAnalysis so tests
# are fully deterministic and offline.
#
# Run with: pytest tests/test_predicate_pipeline.py -v

from __future__ import annotations

import pytest

from mycelium.pipeline.layer0.nlp_preprocessor import SentenceAnalysis, TokenInfo
from mycelium.pipeline.predicates.predicate_extractor import PredicateExtractor
from mycelium.pipeline.predicates.predicate_negator import PredicateNegator
from mycelium.pipeline.predicates.predicate_pipeline import (
    PredicatePipeline,
    get_pipeline,
)
from mycelium.pipeline.predicates.predicate_relations import PredicateRelationType
from mycelium.pipeline.predicates.predicate_store import PredicateStore
from mycelium.pipeline.predicates.predicate_types import (
    ModalCertainty,
    PredicateType,
    QuantifierScope,
)


# ---------------------------------------------------------------------------
# Stub preprocessor
# ---------------------------------------------------------------------------

class _StubPreprocessor:
    """Replaces NLPPreprocessor.  Returns a pre-built SentenceAnalysis.

    Stores the last 'analyse' call args for assertion.
    """

    def __init__(self, analysis: SentenceAnalysis) -> None:
        self._analysis = analysis
        self.last_text: str = ""
        self.last_model: str = ""

    def analyse(self, text: str, model: str = "en_core_web_sm") -> SentenceAnalysis:
        self.last_text = text
        self.last_model = model
        return self._analysis


# ---------------------------------------------------------------------------
# TokenInfo / SentenceAnalysis helpers  (mirrors test_predicate_extractor.py)
# ---------------------------------------------------------------------------

def _tok(
    token: str, lemma: str = "", pos: str = "NOUN",
    tag: str = "", dep: str = "dep", is_stop: bool = False,
) -> TokenInfo:
    return TokenInfo(
        token=token, lemma=lemma or token.lower(),
        pos=pos, tag=tag, dep=dep, is_stop=is_stop,
    )


def _factive_analysis(raw: str = "Paris is the capital of France") -> SentenceAnalysis:
    return SentenceAnalysis(
        raw_text=raw,
        tokens=[_tok("Paris", pos="PROPN"), _tok("is", lemma="be"), _tok("capital", pos="NOUN")],
        sentence_type="DECLARATIVE",
        dep_triples=[("paris", "be", "capital")],
        evaluative_words=[],
        presupposition_triggers=[],
        coercive_signals=[],
        spacy_available=True,
    )


def _modal_analysis() -> SentenceAnalysis:
    return SentenceAnalysis(
        raw_text="AI might displace workers",
        tokens=[
            _tok("AI", pos="PROPN"),
            _tok("might", lemma="might", pos="AUX"),
            _tok("displace", pos="VERB"),
            _tok("workers", pos="NOUN"),
        ],
        sentence_type="DECLARATIVE",
        dep_triples=[("ai", "displace", "worker")],
        evaluative_words=[],
        presupposition_triggers=[],
        coercive_signals=[],
        spacy_available=True,
    )


def _normative_analysis() -> SentenceAnalysis:
    return SentenceAnalysis(
        raw_text="Democracy is the best system",
        tokens=[
            _tok("Democracy", pos="NOUN"),
            _tok("is", lemma="be", pos="AUX"),
            _tok("best", lemma="best", pos="ADJ"),
        ],
        sentence_type="DECLARATIVE",
        dep_triples=[("democracy", "be", "system")],
        evaluative_words=["best"],
        presupposition_triggers=[],
        coercive_signals=[],
        spacy_available=True,
    )


def _presup_analysis() -> SentenceAnalysis:
    return SentenceAnalysis(
        raw_text="Why does inflation cause unemployment?",
        tokens=[
            _tok("inflation", pos="NOUN"),
            _tok("cause", lemma="cause", pos="VERB"),
            _tok("unemployment", pos="NOUN"),
        ],
        sentence_type="INTERROGATIVE",
        dep_triples=[("inflation", "cause", "unemployment")],
        evaluative_words=[],
        presupposition_triggers=["why_causal_presupposition"],
        coercive_signals=[],
        spacy_available=True,
    )


def _spacy_down_analysis() -> SentenceAnalysis:
    """Simulates preprocessor fallback when spaCy subprocess is unavailable."""
    return SentenceAnalysis(
        raw_text="Vaccines cause autism",
        tokens=[],
        sentence_type="DECLARATIVE",
        dep_triples=[],
        evaluative_words=[],
        presupposition_triggers=[],
        coercive_signals=[],
        spacy_available=False,
    )


def _make_pipeline(analysis: SentenceAnalysis) -> PredicatePipeline:
    stub = _StubPreprocessor(analysis)
    return PredicatePipeline(
        preprocessor=stub,
        extractor=PredicateExtractor(),
        negator=PredicateNegator(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRunReturnType:
    def test_run_returns_predicate_store(self):
        pipeline = _make_pipeline(_factive_analysis())
        result = pipeline.run("Paris is the capital of France")
        assert isinstance(result, PredicateStore)

    def test_run_store_is_non_empty(self):
        pipeline = _make_pipeline(_factive_analysis())
        result = pipeline.run("Paris is the capital of France")
        assert len(result) > 0


class TestPredicateTypeRouting:
    def test_modal_text_produces_modal_frame(self):
        pipeline = _make_pipeline(_modal_analysis())
        store = pipeline.run("AI might displace workers")
        assert store.primary().predicate_type == PredicateType.MODAL

    def test_normative_text_produces_normative_frame(self):
        pipeline = _make_pipeline(_normative_analysis())
        store = pipeline.run("Democracy is the best system")
        assert store.primary().predicate_type == PredicateType.NORMATIVE

    def test_factive_text_produces_factive_frame(self):
        pipeline = _make_pipeline(_factive_analysis())
        store = pipeline.run("Paris is the capital of France")
        assert store.primary().predicate_type == PredicateType.FACTIVE


class TestNegationWiring:
    def test_run_populates_negated_frames(self):
        """negate_store() must be called: store should contain negated frames."""
        pipeline = _make_pipeline(_factive_analysis())
        store = pipeline.run("Paris is the capital of France")
        negated = [f for f in store if f.negated]
        assert len(negated) >= 1

    def test_run_populates_contradicts_edges(self):
        pipeline = _make_pipeline(_factive_analysis())
        store = pipeline.run("Paris is the capital of France")
        primary = store.primary()
        related = store.related(primary.predicate_id, PredicateRelationType.CONTRADICTS)
        assert len(related) == 1

    def test_normative_frame_has_no_negated_counterpart(self):
        """NORMATIVE frames are non-falsifiable: no negation should be produced."""
        pipeline = _make_pipeline(_normative_analysis())
        store = pipeline.run("Democracy is the best system")
        negated = [f for f in store if f.negated]
        assert len(negated) == 0


class TestPresuppositionWiring:
    def test_presupposition_frame_injected(self):
        pipeline = _make_pipeline(_presup_analysis())
        store = pipeline.run("Why does inflation cause unemployment?")
        # primary + presupposition frame
        assert len(store) >= 2

    def test_presupposes_edge_created(self):
        pipeline = _make_pipeline(_presup_analysis())
        store = pipeline.run("Why does inflation cause unemployment?")
        primary = store.primary()
        related = store.related(primary.predicate_id, PredicateRelationType.PRESUPPOSES)
        assert len(related) == 1


class TestRunFromAnalysis:
    def test_run_from_analysis_returns_store(self):
        analysis = _factive_analysis()
        pipeline = _make_pipeline(analysis)
        store = pipeline.run_from_analysis(analysis)
        assert isinstance(store, PredicateStore)

    def test_run_from_analysis_result_matches_run(self):
        """Both entry points must produce the same primary frame type."""
        analysis = _factive_analysis()
        pipeline = _make_pipeline(analysis)
        store_via_run = pipeline.run("Paris is the capital of France")
        store_via_analysis = pipeline.run_from_analysis(analysis)
        assert (
            store_via_run.primary().predicate_type
            == store_via_analysis.primary().predicate_type
        )


class TestTiming:
    def test_last_run_ms_is_positive_after_run(self):
        pipeline = _make_pipeline(_factive_analysis())
        pipeline.run("Paris is the capital of France")
        assert pipeline.last_run_ms > 0.0

    def test_last_run_ms_updated_after_run_from_analysis(self):
        analysis = _factive_analysis()
        pipeline = _make_pipeline(analysis)
        pipeline.run_from_analysis(analysis)
        assert pipeline.last_run_ms > 0.0


class TestSpacyDownFallback:
    def test_spacy_down_still_returns_valid_store(self):
        """If spaCy is unavailable the pipeline must not raise."""
        pipeline = _make_pipeline(_spacy_down_analysis())
        store = pipeline.run("Vaccines cause autism")
        assert isinstance(store, PredicateStore)
        assert len(store) >= 1

    def test_spacy_down_produces_factive_fallback(self):
        pipeline = _make_pipeline(_spacy_down_analysis())
        store = pipeline.run("Vaccines cause autism")
        assert store.primary().predicate_type == PredicateType.FACTIVE
        assert store.primary().generated_by_phase == "PredicateExtractor/fallback"


class TestDomainHintPropagation:
    def test_domain_hint_propagated_to_all_frames(self):
        pipeline = _make_pipeline(_factive_analysis())
        store = pipeline.run("Paris is the capital of France", domain_hint="history")
        for frame in store:
            if frame.generated_by_phase != "PredicateExtractor/presupposition":
                assert frame.domain_hint == "history"


class TestSingleton:
    def test_get_pipeline_returns_singleton(self):
        a = get_pipeline()
        b = get_pipeline()
        assert a is b

    def test_singleton_is_predicate_pipeline_instance(self):
        assert isinstance(get_pipeline(), PredicatePipeline)
