# tests/test_predicate_negator.py
# Unit tests for predicate_negator.py — Rules N1-N8 and negate_store().
#
# All tests are pure: no LLM calls, no subprocess, no I/O.
# PredicateFrames are built directly from predicate_types primitives.
#
# Run with: pytest tests/test_predicate_negator.py -v

from __future__ import annotations

import pytest

from mycelium.pipeline.predicates.predicate_negator import PredicateNegator, get_negator
from mycelium.pipeline.predicates.predicate_relations import PredicateRelationType
from mycelium.pipeline.predicates.predicate_store import PredicateStore
from mycelium.pipeline.predicates.predicate_types import (
    ModalCertainty,
    PredicateEntity,
    PredicateFrame,
    PredicateRelation,
    PredicateType,
    QuantifierScope,
    RefutationBurden,
    make_predicate_id,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _entity(text: str = "entity") -> PredicateEntity:
    return PredicateEntity(text=text)


def _relation(lemma: str = "be", polarity: str = "POSITIVE") -> PredicateRelation:
    return PredicateRelation(lemma=lemma, polarity=polarity)


def _frame(
    surface: str,
    ptype: PredicateType,
    polarity: str = "POSITIVE",
    scope: QuantifierScope = QuantifierScope.AMBIGUOUS,
    modal_certainty: float | None = None,
    negated: bool = False,
    falsifiable: bool = True,
    domain: str = "general",
    relation_lemma: str = "be",
) -> PredicateFrame:
    pid = make_predicate_id(surface, ptype, domain)
    return PredicateFrame(
        predicate_id=pid,
        surface_form=surface,
        predicate_type=ptype,
        subject=_entity("subject"),
        relation=_relation(relation_lemma, polarity),
        object=_entity("object"),
        quantifier_scope=scope,
        modal_certainty=modal_certainty,
        negated=negated,
        falsifiable=falsifiable,
        domain_hint=domain,
    )


@pytest.fixture
def negator() -> PredicateNegator:
    return PredicateNegator()


# ---------------------------------------------------------------------------
# Rule N1 — FACTIVE
# ---------------------------------------------------------------------------

class TestRuleN1:
    def test_polarity_flipped(self, negator):
        src = _frame("Paris is the capital", PredicateType.FACTIVE, polarity="POSITIVE")
        result = negator.negate(src)
        assert result.relation.polarity == "NEGATIVE"

    def test_negated_flag_toggled(self, negator):
        src = _frame("Paris is the capital", PredicateType.FACTIVE)
        result = negator.negate(src)
        assert result.negated is True

    def test_transformation_record_appended(self, negator):
        src = _frame("Paris is the capital", PredicateType.FACTIVE)
        result = negator.negate(src)
        assert len(result.transformation_history) == 1
        assert result.transformation_history[0].transformation_type == "POLARITY_FLIP"
        assert result.transformation_history[0].triggered_by == "PredicateNegator/N1"

    def test_semantic_revision_incremented(self, negator):
        src = _frame("Paris is the capital", PredicateType.FACTIVE)
        result = negator.negate(src)
        assert result.semantic_revision == src.semantic_revision + 1

    def test_derived_from_set(self, negator):
        src = _frame("Paris is the capital", PredicateType.FACTIVE)
        result = negator.negate(src)
        assert src.predicate_id in result.derived_from

    def test_source_not_mutated(self, negator):
        src = _frame("Paris is the capital", PredicateType.FACTIVE)
        original_pid = src.predicate_id
        original_polarity = src.relation.polarity
        negator.negate(src)
        assert src.predicate_id == original_pid
        assert src.relation.polarity == original_polarity

    def test_double_negation_flips_back(self, negator):
        src = _frame("Paris is the capital", PredicateType.FACTIVE)
        once = negator.negate(src)
        twice = negator.negate(once)
        assert twice.relation.polarity == src.relation.polarity


# ---------------------------------------------------------------------------
# Rule N2 — COMPARATIVE
# ---------------------------------------------------------------------------

class TestRuleN2:
    def test_polarity_flipped(self, negator):
        src = _frame("Nuclear is safer than coal", PredicateType.COMPARATIVE)
        result = negator.negate(src)
        assert result.relation.polarity == "NEGATIVE"

    def test_refutation_burden_scope_bounded(self, negator):
        src = _frame("Nuclear is safer than coal", PredicateType.COMPARATIVE)
        result = negator.negate(src)
        assert result.refutation_burden == RefutationBurden.SCOPE_BOUNDED

    def test_transformation_type(self, negator):
        src = _frame("Nuclear is safer than coal", PredicateType.COMPARATIVE)
        result = negator.negate(src)
        assert result.transformation_history[-1].transformation_type == "DE_MORGAN_INVERSION"
        assert result.transformation_history[-1].triggered_by == "PredicateNegator/N2"


# ---------------------------------------------------------------------------
# Rule N3 — CAUSAL
# ---------------------------------------------------------------------------

class TestRuleN3:
    def test_polarity_flipped(self, negator):
        src = _frame("Smoking causes cancer", PredicateType.CAUSAL, relation_lemma="cause")
        result = negator.negate(src)
        assert result.relation.polarity == "NEGATIVE"

    def test_refutation_burden_scope_bounded(self, negator):
        src = _frame("Smoking causes cancer", PredicateType.CAUSAL, relation_lemma="cause")
        result = negator.negate(src)
        assert result.refutation_burden == RefutationBurden.SCOPE_BOUNDED

    def test_triggered_by_n3(self, negator):
        src = _frame("Smoking causes cancer", PredicateType.CAUSAL)
        result = negator.negate(src)
        assert result.transformation_history[-1].triggered_by == "PredicateNegator/N3"


# ---------------------------------------------------------------------------
# Rule N4 — EXISTENTIAL
# ---------------------------------------------------------------------------

class TestRuleN4:
    def test_existential_scope_flips_to_universal(self, negator):
        src = _frame(
            "Some black holes are small", PredicateType.EXISTENTIAL,
            scope=QuantifierScope.EXISTENTIAL,
        )
        result = negator.negate(src)
        assert result.quantifier_scope == QuantifierScope.UNIVERSAL

    def test_existential_burden_becomes_exhaustive(self, negator):
        src = _frame(
            "Some black holes are small", PredicateType.EXISTENTIAL,
            scope=QuantifierScope.EXISTENTIAL,
        )
        result = negator.negate(src)
        assert result.refutation_burden == RefutationBurden.EXHAUSTIVE

    def test_universal_scope_flips_to_existential(self, negator):
        src = _frame(
            "All swans are white", PredicateType.EXISTENTIAL,
            scope=QuantifierScope.UNIVERSAL,
        )
        result = negator.negate(src)
        assert result.quantifier_scope == QuantifierScope.EXISTENTIAL

    def test_universal_burden_becomes_single_counterexample(self, negator):
        src = _frame(
            "All swans are white", PredicateType.EXISTENTIAL,
            scope=QuantifierScope.UNIVERSAL,
        )
        result = negator.negate(src)
        assert result.refutation_burden == RefutationBurden.SINGLE_COUNTEREXAMPLE

    def test_triggered_by_n4(self, negator):
        src = _frame("Some X exist", PredicateType.EXISTENTIAL)
        result = negator.negate(src)
        assert result.transformation_history[-1].triggered_by == "PredicateNegator/N4"


# ---------------------------------------------------------------------------
# Rule N5 — TEMPORAL
# ---------------------------------------------------------------------------

class TestRuleN5:
    def test_polarity_flipped(self, negator):
        src = _frame("Policy enacted in 1994", PredicateType.TEMPORAL)
        result = negator.negate(src)
        assert result.relation.polarity == "NEGATIVE"

    def test_scope_conditions_preserved(self, negator):
        src = _frame("Policy enacted in 1994", PredicateType.TEMPORAL)
        src.__dict__["scope_conditions"] = ["post-1990", "United States"]
        result = negator.negate(src)
        assert result.scope_conditions == ["post-1990", "United States"]

    def test_triggered_by_n5(self, negator):
        src = _frame("Policy enacted in 1994", PredicateType.TEMPORAL)
        result = negator.negate(src)
        assert result.transformation_history[-1].triggered_by == "PredicateNegator/N5"


# ---------------------------------------------------------------------------
# Rule N6 — MODAL (most critical — spec v0.2.1 formal definition)
# ---------------------------------------------------------------------------

class TestRuleN6:
    def test_type_recast_to_factive(self, negator):
        src = _frame(
            "AI might displace workers", PredicateType.MODAL,
            modal_certainty=float(ModalCertainty.POSSIBLE),
        )
        result = negator.negate(src)
        assert result.predicate_type == PredicateType.FACTIVE

    def test_polarity_flipped_on_embedded_proposition(self, negator):
        src = _frame(
            "AI might displace workers", PredicateType.MODAL,
            polarity="POSITIVE",
            modal_certainty=float(ModalCertainty.POSSIBLE),
        )
        result = negator.negate(src)
        assert result.relation.polarity == "NEGATIVE"

    def test_modal_auxiliary_stripped_from_relation(self, negator):
        src = _frame(
            "AI might displace workers", PredicateType.MODAL,
            modal_certainty=float(ModalCertainty.POSSIBLE),
        )
        result = negator.negate(src)
        assert result.relation.modality == "CERTAIN"

    def test_modal_certainty_ceiling_inherited(self, negator):
        """Ceiling on negated frame = source.modal_certainty."""
        src = _frame(
            "AI might displace workers", PredicateType.MODAL,
            modal_certainty=float(ModalCertainty.POSSIBLE),  # 0.5
        )
        result = negator.negate(src)
        assert result.modal_certainty == pytest.approx(0.5)

    def test_modal_certainty_ceiling_probable(self, negator):
        """Ceiling is 0.75 for PROBABLE modals."""
        src = _frame(
            "Nuclear power should replace coal", PredicateType.MODAL,
            modal_certainty=float(ModalCertainty.PROBABLE),  # 0.75
        )
        result = negator.negate(src)
        assert result.modal_certainty == pytest.approx(0.75)

    def test_refutation_burden_scope_bounded(self, negator):
        src = _frame(
            "AI might displace workers", PredicateType.MODAL,
            modal_certainty=float(ModalCertainty.POSSIBLE),
        )
        result = negator.negate(src)
        assert result.refutation_burden == RefutationBurden.SCOPE_BOUNDED

    def test_falsifiable_remains_true(self, negator):
        src = _frame(
            "AI might displace workers", PredicateType.MODAL,
            modal_certainty=float(ModalCertainty.POSSIBLE),
        )
        result = negator.negate(src)
        assert result.falsifiable is True

    def test_transformation_type_modal_factive_embed(self, negator):
        src = _frame(
            "AI might displace workers", PredicateType.MODAL,
            modal_certainty=float(ModalCertainty.POSSIBLE),
        )
        result = negator.negate(src)
        assert result.transformation_history[-1].transformation_type == "MODAL_FACTIVE_EMBED"
        assert result.transformation_history[-1].triggered_by == "PredicateNegator/N6"

    def test_source_frame_not_mutated_by_n6(self, negator):
        """N6 must not change the source frame's type or certainty."""
        src = _frame(
            "AI might displace workers", PredicateType.MODAL,
            modal_certainty=float(ModalCertainty.POSSIBLE),
        )
        negator.negate(src)
        assert src.predicate_type == PredicateType.MODAL
        assert src.modal_certainty == pytest.approx(0.5)

    def test_n6_default_ceiling_when_no_modal_certainty(self, negator):
        """If source.modal_certainty is None (should not happen normally),
        default ceiling is POSSIBLE (0.5)."""
        src = _frame(
            "This might happen", PredicateType.MODAL,
            modal_certainty=None,
        )
        result = negator.negate(src)
        assert result.modal_certainty == pytest.approx(float(ModalCertainty.POSSIBLE))


# ---------------------------------------------------------------------------
# Rule N7 — NORMATIVE
# ---------------------------------------------------------------------------

class TestRuleN7:
    def test_returns_none(self, negator):
        src = _frame(
            "Democracy is the best system", PredicateType.NORMATIVE,
            falsifiable=False,
        )
        result = negator.negate(src)
        assert result is None


# ---------------------------------------------------------------------------
# Rule N8 — DEFINITIONAL
# ---------------------------------------------------------------------------

class TestRuleN8:
    def test_polarity_flipped(self, negator):
        src = _frame("Mammals are warm-blooded", PredicateType.DEFINITIONAL)
        result = negator.negate(src)
        assert result.relation.polarity == "NEGATIVE"

    def test_burden_exhaustive(self, negator):
        src = _frame("Mammals are warm-blooded", PredicateType.DEFINITIONAL)
        result = negator.negate(src)
        assert result.refutation_burden == RefutationBurden.EXHAUSTIVE

    def test_triggered_by_n8(self, negator):
        src = _frame("Mammals are warm-blooded", PredicateType.DEFINITIONAL)
        result = negator.negate(src)
        assert result.transformation_history[-1].triggered_by == "PredicateNegator/N8"


# ---------------------------------------------------------------------------
# negate_store() — store-level operation
# ---------------------------------------------------------------------------

class TestNegateStore:
    def test_falsifiable_frames_are_negated(self, negator):
        store = PredicateStore()
        f1 = _frame("Paris is the capital", PredicateType.FACTIVE)
        f2 = _frame("Smoking causes cancer", PredicateType.CAUSAL, relation_lemma="cause")
        store.add(f1)
        store.add(f2)
        negator.negate_store(store)
        assert len(store) == 4  # 2 originals + 2 negated

    def test_contradicts_edges_added(self, negator):
        store = PredicateStore()
        f1 = _frame("Paris is the capital", PredicateType.FACTIVE)
        store.add(f1)
        negator.negate_store(store)
        related = store.related(f1.predicate_id, PredicateRelationType.CONTRADICTS)
        assert len(related) == 1

    def test_negated_form_reference_set(self, negator):
        store = PredicateStore()
        f1 = _frame("Paris is the capital", PredicateType.FACTIVE)
        store.add(f1)
        negator.negate_store(store)
        assert f1.negated_form is not None

    def test_normative_frames_skipped(self, negator):
        store = PredicateStore()
        norm = _frame(
            "Democracy is best", PredicateType.NORMATIVE, falsifiable=False
        )
        store.add(norm)
        negator.negate_store(store)
        assert len(store) == 1  # unchanged

    def test_idempotent_on_re_run(self, negator):
        store = PredicateStore()
        f1 = _frame("Paris is the capital", PredicateType.FACTIVE)
        store.add(f1)
        negator.negate_store(store)
        size_after_first = len(store)
        negator.negate_store(store)  # second run
        assert len(store) == size_after_first  # no new frames added


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

def test_get_negator_returns_singleton():
    a = get_negator()
    b = get_negator()
    assert a is b
