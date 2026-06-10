# mycelium/pipeline/predicates/predicate_negator.py
# Deterministic structural negation of PredicateFrames.
#
# Implements negation rules N1-N8 from implementation spec v0.2.1.
# Each rule produces a new PredicateFrame; the source frame is NEVER mutated.
# Every produced frame carries a full TransformationRecord for lineage.
#
# Key spec contract (v0.2.1, Rule N6):
#   MODAL negation rule:
#     negated_form = FACTIVE negation of the embedded proposition
#     net_confidence ceiling = modal_certainty_score x base_confidence
#     refutation_burden = SCOPE_BOUNDED
#   "might cause X" -> negated: "does not cause X"
#   but net_confidence(negated) <= modal_certainty regardless of evidence
#   because the original only claimed possibility, not actuality.
#
# Spec ref: implementation spec v0.2.1 - Section 6 (Negation Rules)

from __future__ import annotations

import copy
from typing import Optional

from mycelium.pipeline.predicates.predicate_relations import PredicateRelationType
from mycelium.pipeline.predicates.predicate_store import PredicateStore
from mycelium.pipeline.predicates.predicate_types import (
    ModalCertainty,
    PredicateFrame,
    PredicateRelation,
    PredicateType,
    QuantifierScope,
    RefutationBurden,
    make_predicate_id,
    make_transformation_record,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _flip_polarity(polarity: str) -> str:
    return "NEGATIVE" if polarity == "POSITIVE" else "POSITIVE"


def _negate_predicate_id(source: PredicateFrame) -> str:
    """Generate a stable id for the negated frame.

    Prefix 'NEG:' distinguishes negated ids from source ids without
    recomputing the hash from a different surface form.
    """
    return make_predicate_id(
        f"NEG:{source.surface_form}",
        source.predicate_type,
        source.domain_hint,
    )


def _base_negated_frame(source: PredicateFrame, new_pid: str) -> PredicateFrame:
    """Deep-copy source into a new frame with identity fields reset for negation.

    Callers are responsible for setting:
    - relation (polarity)
    - predicate_type (if changed by the rule)
    - quantifier_scope (if changed by the rule)
    - refutation_burden
    - transformation_history (append the record)
    - semantic_revision
    - generated_by_phase
    - derived_from
    - modal_certainty (for N6)
    """
    frame = copy.deepcopy(source)
    # Reset identity
    object.__setattr__(frame, "predicate_id", new_pid) if False else None
    frame.__dict__["predicate_id"] = new_pid
    frame.__dict__["derived_from"] = [source.predicate_id]
    frame.__dict__["negated"] = not source.negated
    frame.__dict__["negated_form"] = None   # negations of negations not chained
    frame.__dict__["presupposition_frame"] = None
    return frame


# ---------------------------------------------------------------------------
# PredicateNegator
# ---------------------------------------------------------------------------

class PredicateNegator:
    """Applies spec-defined negation rules N1-N8 to PredicateFrames.

    All methods are pure: they return new frames and never mutate inputs.
    The negator is stateless and thread-safe.

    Typical usage
    -------------
    ::
        negator = PredicateNegator()

        # Negate a single frame
        negated = negator.negate(frame)

        # Negate all falsifiable frames in a store
        negator.negate_store(store)
    """

    def negate(self, source: PredicateFrame) -> Optional[PredicateFrame]:
        """Apply the correct negation rule for source.predicate_type.

        Returns None for NORMATIVE frames (Rule N7 — non-falsifiable).
        Returns a new PredicateFrame for all other types.
        """
        rule_map = {
            PredicateType.FACTIVE:      self._rule_n1,
            PredicateType.COMPARATIVE:  self._rule_n2,
            PredicateType.CAUSAL:       self._rule_n3,
            PredicateType.EXISTENTIAL:  self._rule_n4,
            PredicateType.TEMPORAL:     self._rule_n5,
            PredicateType.MODAL:        self._rule_n6,
            PredicateType.NORMATIVE:    self._rule_n7,
            PredicateType.DEFINITIONAL: self._rule_n8,
        }
        rule = rule_map.get(source.predicate_type, self._rule_n1)
        return rule(source)

    def negate_store(self, store: PredicateStore) -> None:
        """Negate every falsifiable frame in the store in-place.

        For each frame where falsifiable=True:
        - Calls negate() to produce a negated frame
        - Inserts the negated frame into the store
        - Adds a CONTRADICTS edge from source -> negated
        - Sets source.negated_form to the negated frame

        NORMATIVE frames (falsifiable=False) are silently skipped.
        Frames that already have a negated_form are skipped (idempotent).
        Frames that are themselves already negated (negated=True) are also
        skipped — we do not produce negations-of-negations. This ensures
        negate_store() is safe to call multiple times on the same store.
        """
        for frame in list(store.falsifiable()):
            # Skip if already processed (has a negated_form) OR if this
            # frame is itself a negated product of a prior negate_store() call.
            if frame.negated_form is not None or frame.negated:
                continue
            negated = self.negate(frame)
            if negated is None:
                continue
            store.add(negated)
            store.add_relation(
                source_id=frame.predicate_id,
                target_id=negated.predicate_id,
                relation_type=PredicateRelationType.CONTRADICTS,
                confidence=1.0,
            )
            # Patch negated_form reference onto source (post-construction link)
            frame.__dict__["negated_form"] = negated

    # ------------------------------------------------------------------
    # Rule N1 — FACTIVE: polarity flip
    # ------------------------------------------------------------------

    def _rule_n1(self, source: PredicateFrame) -> PredicateFrame:
        """N1: Flip the polarity of the relation.

        'Paris is the capital of France'
        -> 'Paris is NOT the capital of France'

        quantifier_scope is preserved unchanged.
        refutation_burden is inherited from source.
        """
        new_pid = _negate_predicate_id(source)
        frame = _base_negated_frame(source, new_pid)
        frame.__dict__["relation"] = PredicateRelation(
            lemma=source.relation.lemma,
            polarity=_flip_polarity(source.relation.polarity),
            tense=source.relation.tense,
            modality=source.relation.modality,
        )
        record = make_transformation_record(
            transformation_type="POLARITY_FLIP",
            source_predicate_id=source.predicate_id,
            resulting_predicate_id=new_pid,
            reason="Rule N1: FACTIVE polarity flip",
            triggered_by="PredicateNegator/N1",
        )
        frame.__dict__["transformation_history"] = list(source.transformation_history) + [record]
        frame.__dict__["semantic_revision"] = source.semantic_revision + 1
        frame.__dict__["generated_by_phase"] = "PredicateNegator/N1"
        return frame

    # ------------------------------------------------------------------
    # Rule N2 — COMPARATIVE: De Morgan inversion
    # ------------------------------------------------------------------

    def _rule_n2(self, source: PredicateFrame) -> PredicateFrame:
        """N2: Invert the direction of the comparison.

        'Nuclear is safer than coal' -> 'Nuclear is NOT safer than coal'
        (which by De Morgan does NOT assert 'coal is safer than nuclear';
        it asserts the comparison is false or indeterminate.)

        refutation_burden = SCOPE_BOUNDED (comparison must hold in same scope).
        """
        new_pid = _negate_predicate_id(source)
        frame = _base_negated_frame(source, new_pid)
        frame.__dict__["relation"] = PredicateRelation(
            lemma=source.relation.lemma,
            polarity=_flip_polarity(source.relation.polarity),
            tense=source.relation.tense,
            modality=source.relation.modality,
        )
        frame.__dict__["refutation_burden"] = RefutationBurden.SCOPE_BOUNDED
        record = make_transformation_record(
            transformation_type="DE_MORGAN_INVERSION",
            source_predicate_id=source.predicate_id,
            resulting_predicate_id=new_pid,
            reason="Rule N2: COMPARATIVE De Morgan inversion",
            triggered_by="PredicateNegator/N2",
        )
        frame.__dict__["transformation_history"] = list(source.transformation_history) + [record]
        frame.__dict__["semantic_revision"] = source.semantic_revision + 1
        frame.__dict__["generated_by_phase"] = "PredicateNegator/N2"
        return frame

    # ------------------------------------------------------------------
    # Rule N3 — CAUSAL: polarity flip + SCOPE_BOUNDED
    # ------------------------------------------------------------------

    def _rule_n3(self, source: PredicateFrame) -> PredicateFrame:
        """N3: Flip polarity on the causal relation.

        'Smoking causes cancer' -> 'Smoking does not cause cancer'

        refutation_burden = SCOPE_BOUNDED: a single study is insufficient;
        evidence must apply within the same causal scope.
        """
        new_pid = _negate_predicate_id(source)
        frame = _base_negated_frame(source, new_pid)
        frame.__dict__["relation"] = PredicateRelation(
            lemma=source.relation.lemma,
            polarity=_flip_polarity(source.relation.polarity),
            tense=source.relation.tense,
            modality=source.relation.modality,
        )
        frame.__dict__["refutation_burden"] = RefutationBurden.SCOPE_BOUNDED
        record = make_transformation_record(
            transformation_type="POLARITY_FLIP",
            source_predicate_id=source.predicate_id,
            resulting_predicate_id=new_pid,
            reason="Rule N3: CAUSAL polarity flip with SCOPE_BOUNDED burden",
            triggered_by="PredicateNegator/N3",
        )
        frame.__dict__["transformation_history"] = list(source.transformation_history) + [record]
        frame.__dict__["semantic_revision"] = source.semantic_revision + 1
        frame.__dict__["generated_by_phase"] = "PredicateNegator/N3"
        return frame

    # ------------------------------------------------------------------
    # Rule N4 — EXISTENTIAL: De Morgan (some <-> none)
    # ------------------------------------------------------------------

    def _rule_n4(self, source: PredicateFrame) -> PredicateFrame:
        """N4: Invert existential quantifier scope.

        'Some black holes exist smaller than 1km'
        -> 'No black holes exist smaller than 1km'
        (and vice versa for UNIVERSAL scope)

        refutation_burden flips:
          EXISTENTIAL scope -> EXHAUSTIVE (must check all instances)
          UNIVERSAL scope   -> SINGLE_COUNTEREXAMPLE
        """
        new_pid = _negate_predicate_id(source)
        frame = _base_negated_frame(source, new_pid)
        frame.__dict__["relation"] = PredicateRelation(
            lemma=source.relation.lemma,
            polarity=_flip_polarity(source.relation.polarity),
            tense=source.relation.tense,
            modality=source.relation.modality,
        )
        # Flip quantifier scope
        if source.quantifier_scope == QuantifierScope.EXISTENTIAL:
            frame.__dict__["quantifier_scope"] = QuantifierScope.UNIVERSAL
            frame.__dict__["refutation_burden"] = RefutationBurden.EXHAUSTIVE
        elif source.quantifier_scope == QuantifierScope.UNIVERSAL:
            frame.__dict__["quantifier_scope"] = QuantifierScope.EXISTENTIAL
            frame.__dict__["refutation_burden"] = RefutationBurden.SINGLE_COUNTEREXAMPLE
        else:
            frame.__dict__["refutation_burden"] = RefutationBurden.EXHAUSTIVE
        record = make_transformation_record(
            transformation_type="DE_MORGAN_INVERSION",
            source_predicate_id=source.predicate_id,
            resulting_predicate_id=new_pid,
            reason="Rule N4: EXISTENTIAL De Morgan quantifier inversion",
            triggered_by="PredicateNegator/N4",
        )
        frame.__dict__["transformation_history"] = list(source.transformation_history) + [record]
        frame.__dict__["semantic_revision"] = source.semantic_revision + 1
        frame.__dict__["generated_by_phase"] = "PredicateNegator/N4"
        return frame

    # ------------------------------------------------------------------
    # Rule N5 — TEMPORAL: polarity flip, scope_conditions preserved
    # ------------------------------------------------------------------

    def _rule_n5(self, source: PredicateFrame) -> PredicateFrame:
        """N5: Flip polarity; preserve temporal scope conditions.

        'The policy was enacted in 1994'
        -> 'The policy was NOT enacted in 1994'

        scope_conditions are carried through unchanged so evidence
        search remains anchored to the same time window.
        """
        new_pid = _negate_predicate_id(source)
        frame = _base_negated_frame(source, new_pid)
        frame.__dict__["relation"] = PredicateRelation(
            lemma=source.relation.lemma,
            polarity=_flip_polarity(source.relation.polarity),
            tense=source.relation.tense,
            modality=source.relation.modality,
        )
        # Preserve scope_conditions (temporal anchor must not be lost)
        frame.__dict__["scope_conditions"] = list(source.scope_conditions)
        frame.__dict__["refutation_burden"] = RefutationBurden.SCOPE_BOUNDED
        record = make_transformation_record(
            transformation_type="POLARITY_FLIP",
            source_predicate_id=source.predicate_id,
            resulting_predicate_id=new_pid,
            reason="Rule N5: TEMPORAL polarity flip, scope_conditions preserved",
            triggered_by="PredicateNegator/N5",
        )
        frame.__dict__["transformation_history"] = list(source.transformation_history) + [record]
        frame.__dict__["semantic_revision"] = source.semantic_revision + 1
        frame.__dict__["generated_by_phase"] = "PredicateNegator/N5"
        return frame

    # ------------------------------------------------------------------
    # Rule N6 — MODAL: FACTIVE-embed with confidence ceiling
    # ------------------------------------------------------------------

    def _rule_n6(self, source: PredicateFrame) -> PredicateFrame:
        """N6: MODAL negation via FACTIVE embedding of the stripped proposition.

        Formal definition (spec v0.2.1):
            negated_form  = FACTIVE negation of the embedded proposition
            net_confidence ceiling = modal_certainty_score x base_confidence
            refutation_burden = SCOPE_BOUNDED

        Example:
            'AI might displace workers'
            -> negated: 'AI does not displace workers'
            -> BUT net_confidence(negated) <= modal_certainty (e.g. 0.5)
               regardless of how strong the evidence is.

        Rationale: disproving 'X happens' does NOT disprove 'X might happen'.
        The ceiling is stored in modal_certainty on the negated frame so
        EvidenceFinder can enforce it without re-parsing the predicate type.

        IMPORTANT: refutation_score on the EvidenceBundle is NOT capped.
        Only confirmation_score and net_confidence are capped by modal_certainty.
        Strong refuting evidence is never suppressed.
        """
        new_pid = _negate_predicate_id(source)
        frame = _base_negated_frame(source, new_pid)

        # Recast as FACTIVE (the embedded stripped proposition)
        frame.__dict__["predicate_type"] = PredicateType.FACTIVE

        # Polarity flip on the stripped proposition
        frame.__dict__["relation"] = PredicateRelation(
            lemma=source.relation.lemma,
            polarity=_flip_polarity(source.relation.polarity),
            tense=source.relation.tense,
            modality="CERTAIN",  # modal auxiliary stripped from negated form
        )

        # Inherit the confidence ceiling from source modal_certainty.
        # If source somehow has no modal_certainty, default to POSSIBLE (0.5).
        ceiling = source.modal_certainty if source.modal_certainty is not None \
            else float(ModalCertainty.POSSIBLE)
        frame.__dict__["modal_certainty"] = ceiling

        frame.__dict__["refutation_burden"] = RefutationBurden.SCOPE_BOUNDED
        frame.__dict__["falsifiable"] = True

        record = make_transformation_record(
            transformation_type="MODAL_FACTIVE_EMBED",
            source_predicate_id=source.predicate_id,
            resulting_predicate_id=new_pid,
            reason=(
                f"Rule N6: MODAL embedded as FACTIVE; "
                f"confidence ceiling={ceiling:.2f} (modal_certainty from source)"
            ),
            triggered_by="PredicateNegator/N6",
        )
        frame.__dict__["transformation_history"] = list(source.transformation_history) + [record]
        frame.__dict__["semantic_revision"] = source.semantic_revision + 1
        frame.__dict__["generated_by_phase"] = "PredicateNegator/N6"
        return frame

    # ------------------------------------------------------------------
    # Rule N7 — NORMATIVE: non-falsifiable, no negation
    # ------------------------------------------------------------------

    def _rule_n7(self, source: PredicateFrame) -> None:  # type: ignore[return]
        """N7: NORMATIVE predicates cannot be negated by evidence.

        Returns None. Callers (negate_store) silently skip None returns.

        Rationale: 'Democracy is the best system' is a value judgement.
        No evidence can falsify it. It must not enter the proof tree.
        """
        return None

    # ------------------------------------------------------------------
    # Rule N8 — DEFINITIONAL: polarity flip on copula
    # ------------------------------------------------------------------

    def _rule_n8(self, source: PredicateFrame) -> PredicateFrame:
        """N8: Flip polarity on the definitional copula.

        'Mammals are warm-blooded' -> 'Mammals are NOT warm-blooded'

        refutation_burden = EXHAUSTIVE: to refute a definition you must
        show it fails universally, not just for one instance.
        """
        new_pid = _negate_predicate_id(source)
        frame = _base_negated_frame(source, new_pid)
        frame.__dict__["relation"] = PredicateRelation(
            lemma=source.relation.lemma,
            polarity=_flip_polarity(source.relation.polarity),
            tense=source.relation.tense,
            modality=source.relation.modality,
        )
        frame.__dict__["refutation_burden"] = RefutationBurden.EXHAUSTIVE
        record = make_transformation_record(
            transformation_type="POLARITY_FLIP",
            source_predicate_id=source.predicate_id,
            resulting_predicate_id=new_pid,
            reason="Rule N8: DEFINITIONAL polarity flip, EXHAUSTIVE burden",
            triggered_by="PredicateNegator/N8",
        )
        frame.__dict__["transformation_history"] = list(source.transformation_history) + [record]
        frame.__dict__["semantic_revision"] = source.semantic_revision + 1
        frame.__dict__["generated_by_phase"] = "PredicateNegator/N8"
        return frame


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_negator_instance: Optional[PredicateNegator] = None


def get_negator() -> PredicateNegator:
    """Return the module-level PredicateNegator singleton."""
    global _negator_instance
    if _negator_instance is None:
        _negator_instance = PredicateNegator()
    return _negator_instance
