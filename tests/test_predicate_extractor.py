# tests/test_predicate_extractor.py
# Unit tests for predicate_extractor.py.
#
# All tests are pure: no LLM calls, no spaCy subprocess, no I/O.
# SentenceAnalysis objects are built from hand-crafted TokenInfo fixtures
# to cover each classification branch without requiring .venv2.
#
# Run with: pytest tests/test_predicate_extractor.py -v

from __future__ import annotations

import pytest

from mycelium.pipeline.layer0.nlp_preprocessor import SentenceAnalysis, TokenInfo
from mycelium.pipeline.predicates.predicate_extractor import (
    PredicateExtractor,
    get_extractor,
)
from mycelium.pipeline.predicates.predicate_relations import PredicateRelationType
from mycelium.pipeline.predicates.predicate_types import (
    ModalCertainty,
    PredicateType,
    QuantifierScope,
    RefutationBurden,
    make_predicate_id,
)


# ---------------------------------------------------------------------------
# TokenInfo helpers
# ---------------------------------------------------------------------------

def _tok(
    token: str,
    lemma: str = "",
    pos: str = "NOUN",
    tag: str = "",
    dep: str = "dep",
    is_stop: bool = False,
) -> TokenInfo:
    return TokenInfo(
        token=token,
        lemma=lemma or token.lower(),
        pos=pos,
        tag=tag,
        dep=dep,
        is_stop=is_stop,
    )


def _analysis(
    raw: str,
    triples=None,
    tokens=None,
    evaluative=None,
    presup_triggers=None,
    coercive=None,
    spacy_available: bool = True,
    sentence_type: str = "DECLARATIVE",
) -> SentenceAnalysis:
    return SentenceAnalysis(
        raw_text=raw,
        tokens=tokens or [],
        sentence_type=sentence_type,
        dep_triples=triples or [],
        evaluative_words=evaluative or [],
        presupposition_triggers=presup_triggers or [],
        coercive_signals=coercive or [],
        spacy_available=spacy_available,
    )


@pytest.fixture
def extractor() -> PredicateExtractor:
    return PredicateExtractor()


# ---------------------------------------------------------------------------
# PredicateType classification
# ---------------------------------------------------------------------------

class TestPredicateTypeClassification:
    def test_modal_detected_from_auxiliary(self, extractor):
        """'might' in tokens -> MODAL."""
        tokens = [
            _tok("AI", pos="PROPN"),
            _tok("might", lemma="might", pos="AUX"),
            _tok("displace", pos="VERB"),
            _tok("workers", pos="NOUN"),
        ]
        analysis = _analysis(
            "AI might displace workers",
            triples=[("AI", "displace", "workers")],
            tokens=tokens,
        )
        store = extractor.extract(analysis)
        assert store.primary().predicate_type == PredicateType.MODAL

    def test_normative_detected_from_evaluative_word(self, extractor):
        """'best' in evaluative_words -> NORMATIVE."""
        tokens = [
            _tok("Democracy", pos="NOUN"),
            _tok("is", lemma="be", pos="AUX"),
            _tok("best", lemma="best", pos="ADJ"),
        ]
        analysis = _analysis(
            "Democracy is the best system",
            triples=[("democracy", "be", "system")],
            tokens=tokens,
            evaluative=["best"],
        )
        store = extractor.extract(analysis)
        assert store.primary().predicate_type == PredicateType.NORMATIVE

    def test_causal_detected_from_causal_verb(self, extractor):
        """'cause' as predicate lemma -> CAUSAL."""
        tokens = [
            _tok("Smoking", pos="NOUN"),
            _tok("causes", lemma="cause", pos="VERB"),
            _tok("cancer", pos="NOUN"),
        ]
        analysis = _analysis(
            "Smoking causes cancer",
            triples=[("smoking", "cause", "cancer")],
            tokens=tokens,
        )
        store = extractor.extract(analysis)
        assert store.primary().predicate_type == PredicateType.CAUSAL

    def test_factive_default_for_plain_declarative(self, extractor):
        """No modal, causal, comparative signals -> FACTIVE."""
        tokens = [
            _tok("Paris", pos="PROPN"),
            _tok("is", lemma="be", pos="AUX"),
            _tok("capital", pos="NOUN"),
        ]
        analysis = _analysis(
            "Paris is the capital of France",
            triples=[("paris", "be", "capital")],
            tokens=tokens,
        )
        store = extractor.extract(analysis)
        assert store.primary().predicate_type == PredicateType.FACTIVE


# ---------------------------------------------------------------------------
# modal_certainty
# ---------------------------------------------------------------------------

class TestModalCertainty:
    def test_modal_certainty_set_for_modal_frames(self, extractor):
        tokens = [
            _tok("it", pos="PRON"),
            _tok("might", lemma="might", pos="AUX"),
            _tok("cause", lemma="cause", pos="VERB"),
            _tok("inflation", pos="NOUN"),
        ]
        analysis = _analysis(
            "This might cause inflation",
            triples=[("it", "cause", "inflation")],
            tokens=tokens,
        )
        store = extractor.extract(analysis)
        frame = store.primary()
        assert frame.predicate_type == PredicateType.MODAL
        assert frame.modal_certainty == pytest.approx(float(ModalCertainty.POSSIBLE))

    def test_modal_certainty_none_for_factive(self, extractor):
        tokens = [
            _tok("Paris", pos="PROPN"),
            _tok("is", lemma="be", pos="AUX"),
            _tok("capital", pos="NOUN"),
        ]
        analysis = _analysis(
            "Paris is the capital",
            triples=[("paris", "be", "capital")],
            tokens=tokens,
        )
        store = extractor.extract(analysis)
        assert store.primary().modal_certainty is None


# ---------------------------------------------------------------------------
# Falsifiability
# ---------------------------------------------------------------------------

class TestFalsifiability:
    def test_normative_is_not_falsifiable(self, extractor):
        tokens = [
            _tok("war", pos="NOUN"),
            _tok("is", lemma="be", pos="AUX"),
            _tok("wrong", lemma="wrong", pos="ADJ"),
        ]
        analysis = _analysis(
            "War is wrong",
            triples=[("war", "be", "wrong")],
            tokens=tokens,
            evaluative=["wrong"],
        )
        store = extractor.extract(analysis)
        frame = store.primary()
        assert frame.falsifiable is False
        assert frame.refutation_burden == RefutationBurden.NON_FALSIFIABLE


# ---------------------------------------------------------------------------
# QuantifierScope
# ---------------------------------------------------------------------------

class TestQuantifierScope:
    def test_universal_scope_from_all(self, extractor):
        tokens = [
            _tok("all", lemma="all", pos="DET"),
            _tok("swans", pos="NOUN"),
            _tok("are", lemma="be", pos="AUX"),
            _tok("white", pos="ADJ"),
        ]
        analysis = _analysis(
            "All swans are white",
            triples=[("swan", "be", "white")],
            tokens=tokens,
        )
        store = extractor.extract(analysis)
        assert store.primary().quantifier_scope == QuantifierScope.UNIVERSAL

    def test_existential_scope_from_some(self, extractor):
        tokens = [
            _tok("some", lemma="some", pos="DET"),
            _tok("birds", pos="NOUN"),
            _tok("fly", lemma="fly", pos="VERB"),
        ]
        analysis = _analysis(
            "Some birds cannot fly",
            triples=[("bird", "fly", "")],
            tokens=tokens,
        )
        store = extractor.extract(analysis)
        assert store.primary().quantifier_scope == QuantifierScope.EXISTENTIAL


# ---------------------------------------------------------------------------
# Negation detection
# ---------------------------------------------------------------------------

class TestNegationDetection:
    def test_negation_detected_via_dep_neg(self, extractor):
        tokens = [
            _tok("AI", pos="NOUN"),
            _tok("does", lemma="do", pos="AUX"),
            _tok("not", lemma="not", pos="PART", dep="neg"),
            _tok("think", lemma="think", pos="VERB"),
        ]
        analysis = _analysis(
            "AI does not think",
            triples=[("ai", "think", "")],
            tokens=tokens,
        )
        store = extractor.extract(analysis)
        assert store.primary().negated is True

    def test_negation_regex_fallback_when_spacy_unavailable(self, extractor):
        analysis = _analysis(
            "Vaccines do not cause autism",
            triples=[("vaccine", "cause", "autism")],
            tokens=[],
            spacy_available=False,
        )
        store = extractor.extract(analysis)
        assert store.primary().negated is True


# ---------------------------------------------------------------------------
# Fallback frame
# ---------------------------------------------------------------------------

class TestFallbackFrame:
    def test_fallback_frame_created_when_no_triples(self, extractor):
        analysis = _analysis("An interesting claim.", triples=[], tokens=[])
        store = extractor.extract(analysis)
        assert len(store) >= 1
        assert store.primary().predicate_type == PredicateType.FACTIVE
        assert store.primary().generated_by_phase == "PredicateExtractor/fallback"


# ---------------------------------------------------------------------------
# Presupposition frame injection
# ---------------------------------------------------------------------------

class TestPresuppositionInjection:
    def test_presupposition_frame_injected(self, extractor):
        tokens = [
            _tok("Why", pos="ADV"),
            _tok("does", pos="AUX"),
            _tok("inflation", pos="NOUN"),
            _tok("cause", lemma="cause", pos="VERB"),
            _tok("unemployment", pos="NOUN"),
        ]
        analysis = _analysis(
            "Why does inflation cause unemployment?",
            triples=[("inflation", "cause", "unemployment")],
            tokens=tokens,
            presup_triggers=["why_causal_presupposition"],
        )
        store = extractor.extract(analysis)
        # Should have primary frame + presupposition frame
        assert len(store) == 2

    def test_presupposes_edge_created(self, extractor):
        tokens = [
            _tok("inflation", pos="NOUN"),
            _tok("cause", lemma="cause", pos="VERB"),
            _tok("unemployment", pos="NOUN"),
        ]
        analysis = _analysis(
            "Why does inflation cause unemployment?",
            triples=[("inflation", "cause", "unemployment")],
            tokens=tokens,
            presup_triggers=["why_causal_presupposition"],
        )
        store = extractor.extract(analysis)
        primary = store.primary()
        related = store.related(primary.predicate_id, PredicateRelationType.PRESUPPOSES)
        assert len(related) == 1


# ---------------------------------------------------------------------------
# PredicateStore integration
# ---------------------------------------------------------------------------

class TestStoreIntegration:
    def test_store_length_matches_triples(self, extractor):
        tokens = [
            _tok("Paris", pos="PROPN"),
            _tok("is", lemma="be", pos="AUX"),
            _tok("capital", pos="NOUN"),
        ]
        analysis = _analysis(
            "Paris is the capital",
            triples=[("paris", "be", "capital")],
            tokens=tokens,
        )
        store = extractor.extract(analysis)
        assert len(store) == 1

    def test_primary_is_first_inserted(self, extractor):
        tokens = [_tok("Paris", pos="PROPN"), _tok("is", lemma="be")]
        analysis = _analysis(
            "Paris is the capital",
            triples=[("paris", "be", "capital")],
            tokens=tokens,
        )
        store = extractor.extract(analysis)
        assert store.primary() is not None

    def test_predicate_id_deterministic(self, extractor):
        tokens = [_tok("Paris", pos="PROPN"), _tok("is", lemma="be")]
        analysis = _analysis(
            "Paris is the capital",
            triples=[("paris", "be", "capital")],
            tokens=tokens,
        )
        store1 = extractor.extract(analysis)
        store2 = extractor.extract(analysis)
        assert store1.primary().predicate_id == store2.primary().predicate_id

    def test_domain_hint_propagated(self, extractor):
        tokens = [_tok("Paris", pos="PROPN"), _tok("is", lemma="be")]
        analysis = _analysis(
            "Paris is the capital",
            triples=[("paris", "be", "capital")],
            tokens=tokens,
        )
        store = extractor.extract(analysis, domain_hint="history")
        assert store.primary().domain_hint == "history"


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

def test_get_extractor_returns_singleton():
    a = get_extractor()
    b = get_extractor()
    assert a is b
