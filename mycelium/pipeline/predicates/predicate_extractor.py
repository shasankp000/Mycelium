# mycelium/pipeline/predicates/predicate_extractor.py
# Deterministic rule-based extraction of PredicateFrames from a
# SentenceAnalysis produced by layer0/nlp_preprocessor.py.
#
# No LLM calls. No retrieval. Fully synchronous.
# Consumes: SentenceAnalysis (dep_triples, tokens, NER, presupposition_triggers)
# Produces: PredicateStore (populated, with relation graph edges)
#
# Spec ref: implementation spec v0.2.1 — Milestone B, Section 5

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from mycelium.pipeline.layer0.nlp_preprocessor import SentenceAnalysis
from mycelium.pipeline.predicates.predicate_relations import PredicateRelationType
from mycelium.pipeline.predicates.predicate_store import PredicateStore
from mycelium.pipeline.predicates.predicate_types import (
    MODAL_CERTAINTY_MAP,
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
# Vocabulary sets for classification rules
# ---------------------------------------------------------------------------

_MODAL_AUXILIARIES = frozenset(MODAL_CERTAINTY_MAP.keys())

_NORMATIVE_SIGNALS = frozenset([
    "should", "ought", "must", "best", "worst", "better", "worse",
    "right", "wrong", "good", "bad", "fair", "unfair", "just", "unjust",
    "moral", "immoral", "ethical", "unethical", "ideal", "perfect",
])

_TEMPORAL_SIGNALS = frozenset([
    "before", "after", "during", "when", "since", "until", "while",
    "ago", "year", "century", "decade", "recently", "formerly",
    "historically", "previously", "once", "then",
])

_CAUSAL_VERBS = frozenset([
    "cause", "lead", "result", "produce", "trigger", "create",
    "generate", "induce", "force", "enable", "prevent", "allow",
    "make", "drive", "bring",
])

_COMPARATIVE_SIGNALS = frozenset([
    "than", "more", "less", "fewer", "greater", "higher", "lower",
    "better", "worse", "faster", "slower", "safer", "riskier",
    "larger", "smaller", "bigger", "cheaper", "costlier",
])

_EXISTENTIAL_SIGNALS = frozenset([
    "exist", "there", "some", "any", "certain", "few", "several",
    "number", "instance", "case", "example",
])

_DEFINITIONAL_COPULAS = frozenset(["be", "define", "mean", "refer", "constitute", "comprise"])

_UNIVERSAL_QUANTIFIERS = frozenset(["all", "every", "each", "always", "never", "no", "none"])
_EXISTENTIAL_QUANTIFIERS = frozenset(["some", "many", "few", "several", "most", "often"])

# NER labels we map to entity_type
_NER_LABEL_MAP: Dict[str, str] = {
    "PERSON": "PERSON", "ORG": "ORG", "GPE": "GPE",
    "LOC": "LOC", "FAC": "FAC", "NORP": "NORP",
    "PRODUCT": "PRODUCT", "EVENT": "EVENT",
    "WORK_OF_ART": "WORK_OF_ART", "LAW": "LAW",
    "DATE": "DATE", "TIME": "TIME",
    "PERCENT": "PERCENT", "MONEY": "MONEY",
    "QUANTITY": "QUANTITY", "ORDINAL": "ORDINAL",
    "CARDINAL": "CARDINAL",
}


# ---------------------------------------------------------------------------
# PredicateExtractor
# ---------------------------------------------------------------------------

class PredicateExtractor:
    """Converts a SentenceAnalysis into a populated PredicateStore.

    Usage
    -----
    ::
        extractor = PredicateExtractor()
        store = extractor.extract(analysis, domain_hint="science")

    The extractor is stateless and reusable across requests.
    Thread-safe (no mutable class state).
    """

    def extract(
        self,
        analysis: SentenceAnalysis,
        domain_hint: str = "general",
    ) -> PredicateStore:
        """Extract PredicateFrames from a SentenceAnalysis.

        Parameters
        ----------
        analysis:
            Output of NLPPreprocessor.analyse().
        domain_hint:
            Domain string from L1 routing, e.g. 'science', 'history'.
            Injected into every PredicateFrame for downstream routing.

        Returns
        -------
        PredicateStore populated with one frame per dep_triple, plus any
        presupposition frames and CONTRADICTS relation edges.
        """
        store = PredicateStore()

        # Build NER lookup: surface_text (lowercased) -> NER label
        ner_lookup = self._build_ner_lookup(analysis)

        # Build token lemma set for fast membership tests
        lemma_set = {t.lemma.lower() for t in analysis.tokens}

        if analysis.dep_triples:
            for triple in analysis.dep_triples:
                frame = self._frame_from_triple(
                    triple=triple,
                    analysis=analysis,
                    lemma_set=lemma_set,
                    ner_lookup=ner_lookup,
                    domain_hint=domain_hint,
                )
                store.add(frame)
        else:
            # Fallback: no dep_triples available — create one FACTIVE frame
            # from the full surface text. Happens when spaCy is unavailable
            # or the sentence has no clear SVO structure.
            frame = self._frame_from_raw(
                analysis=analysis,
                lemma_set=lemma_set,
                domain_hint=domain_hint,
            )
            store.add(frame)

        # Add presupposition frames and CONTRADICTS edges
        self._inject_presupposition_frames(store, analysis, domain_hint)

        return store

    # ------------------------------------------------------------------
    # Frame construction
    # ------------------------------------------------------------------

    def _frame_from_triple(
        self,
        triple: Tuple[str, str, str],
        analysis: SentenceAnalysis,
        lemma_set: frozenset,
        ner_lookup: Dict[str, str],
        domain_hint: str,
    ) -> PredicateFrame:
        """Build a PredicateFrame from a (subj, pred, obj) dep_triple."""
        subj_lemma, pred_lemma, obj_lemma = triple

        ptype = self._classify_type(analysis, lemma_set, pred_lemma)
        scope = self._resolve_scope(lemma_set)
        negated = self._detect_negation(analysis)
        polarity = "NEGATIVE" if negated else "POSITIVE"
        modal_cert = self._resolve_modal_certainty(ptype, lemma_set)
        falsifiable = ptype != PredicateType.NORMATIVE
        burden = self._resolve_burden(ptype, scope)
        presup_attack = len(analysis.presupposition_triggers) > 0

        subj = PredicateEntity(
            text=subj_lemma,
            entity_type=ner_lookup.get(subj_lemma.lower(), ""),
        )
        obj = PredicateEntity(
            text=obj_lemma,
            entity_type=ner_lookup.get(obj_lemma.lower(), ""),
        )
        relation = PredicateRelation(
            lemma=pred_lemma,
            polarity=polarity,
            modality=self._surface_modality(ptype, lemma_set),
        )

        pid = make_predicate_id(analysis.raw_text, ptype, domain_hint)

        return PredicateFrame(
            predicate_id=pid,
            surface_form=analysis.raw_text,
            predicate_type=ptype,
            subject=subj,
            relation=relation,
            object=obj,
            negated=negated,
            quantifier_scope=scope,
            falsifiable=falsifiable,
            refutation_burden=burden,
            presupposition_attack=presup_attack,
            domain_hint=domain_hint,
            modal_certainty=modal_cert,
            generated_by_phase="PredicateExtractor",
        )

    def _frame_from_raw(
        self,
        analysis: SentenceAnalysis,
        lemma_set: frozenset,
        domain_hint: str,
    ) -> PredicateFrame:
        """Fallback: build a FACTIVE frame directly from the raw surface text."""
        ptype = PredicateType.FACTIVE
        pid = make_predicate_id(analysis.raw_text, ptype, domain_hint)
        negated = self._detect_negation(analysis)
        scope = self._resolve_scope(lemma_set)
        presup_attack = len(analysis.presupposition_triggers) > 0

        subj = PredicateEntity(text=analysis.raw_text[:40])
        obj = PredicateEntity(text="")
        relation = PredicateRelation(
            lemma="be",
            polarity="NEGATIVE" if negated else "POSITIVE",
        )

        return PredicateFrame(
            predicate_id=pid,
            surface_form=analysis.raw_text,
            predicate_type=ptype,
            subject=subj,
            relation=relation,
            object=obj,
            negated=negated,
            quantifier_scope=scope,
            falsifiable=True,
            refutation_burden=RefutationBurden.SCOPE_BOUNDED,
            presupposition_attack=presup_attack,
            domain_hint=domain_hint,
            generated_by_phase="PredicateExtractor/fallback",
        )

    # ------------------------------------------------------------------
    # PredicateType classification — 5-pass rule cascade
    # ------------------------------------------------------------------

    def _classify_type(
        self,
        analysis: SentenceAnalysis,
        lemma_set: frozenset,
        pred_lemma: str,
    ) -> PredicateType:
        """Classify a predicate type using a deterministic priority cascade.

        Priority order (first match wins):
        MODAL > NORMATIVE > TEMPORAL > CAUSAL > COMPARATIVE
        > EXISTENTIAL > DEFINITIONAL > FACTIVE

        MODAL is tested first to prevent 'might be better' from being
        mis-classified as NORMATIVE or COMPARATIVE.
        """
        # 1. MODAL: any modal auxiliary in the token stream
        if lemma_set & _MODAL_AUXILIARIES:
            # Exclude CERTAIN-level modals that signal a definite future
            # (those are effectively FACTIVE in intent)
            modal_lemmas = lemma_set & _MODAL_AUXILIARIES
            if any(MODAL_CERTAINTY_MAP.get(m, ModalCertainty.POSSIBLE)
                   < ModalCertainty.CERTAIN for m in modal_lemmas):
                return PredicateType.MODAL

        # 2. NORMATIVE: opinion/value-laden adjective in evaluative words
        if analysis.evaluative_words:
            norm_signals = {
                w.lower() for w in analysis.evaluative_words
            } & _NORMATIVE_SIGNALS
            if norm_signals:
                return PredicateType.NORMATIVE

        # 3. TEMPORAL: temporal adverb or date NER in token stream
        if lemma_set & _TEMPORAL_SIGNALS:
            return PredicateType.TEMPORAL

        # 4. CAUSAL: causal verb is the root predicate
        if pred_lemma.lower() in _CAUSAL_VERBS:
            return PredicateType.CAUSAL
        if lemma_set & _CAUSAL_VERBS:
            return PredicateType.CAUSAL

        # 5. COMPARATIVE: comparative signal word present
        if lemma_set & _COMPARATIVE_SIGNALS:
            return PredicateType.COMPARATIVE

        # 6. EXISTENTIAL: existential signal or weak quantifier
        if lemma_set & _EXISTENTIAL_SIGNALS:
            return PredicateType.EXISTENTIAL

        # 7. DEFINITIONAL: copula 'be' / 'define' as predicate with no causal meaning
        if pred_lemma.lower() in _DEFINITIONAL_COPULAS:
            return PredicateType.DEFINITIONAL

        # 8. Default: FACTIVE
        return PredicateType.FACTIVE

    # ------------------------------------------------------------------
    # QuantifierScope resolution
    # ------------------------------------------------------------------

    def _resolve_scope(self, lemma_set: frozenset) -> QuantifierScope:
        """Determine quantifier scope from token lemmas."""
        if lemma_set & _UNIVERSAL_QUANTIFIERS:
            return QuantifierScope.UNIVERSAL
        if lemma_set & _EXISTENTIAL_QUANTIFIERS:
            return QuantifierScope.EXISTENTIAL
        # Check for a numeric scope signal (e.g. 'per TWh', '90%')
        # treated as SCOPED
        for lemma in lemma_set:
            if re.search(r"\d", lemma):
                return QuantifierScope.SCOPED
        return QuantifierScope.AMBIGUOUS

    # ------------------------------------------------------------------
    # Negation detection
    # ------------------------------------------------------------------

    def _detect_negation(self, analysis: SentenceAnalysis) -> bool:
        """Return True if the sentence contains an explicit negation marker."""
        neg_deps = {"neg"}
        for token in analysis.tokens:
            if token.dep in neg_deps:
                return True
        # Regex fallback for when spaCy is unavailable
        if not analysis.spacy_available:
            return bool(re.search(
                r"\b(not|never|no|neither|nor|don't|doesn't|didn't|isn't|"
                r"aren't|wasn't|weren't|won't|wouldn't|can't|cannot)\b",
                analysis.raw_text, re.IGNORECASE,
            ))
        return False

    # ------------------------------------------------------------------
    # Modal certainty resolution
    # ------------------------------------------------------------------

    def _resolve_modal_certainty(
        self, ptype: PredicateType, lemma_set: frozenset
    ) -> Optional[float]:
        """Return the ModalCertainty float for MODAL frames, else None."""
        if ptype != PredicateType.MODAL:
            return None
        modal_lemmas = lemma_set & _MODAL_AUXILIARIES
        if not modal_lemmas:
            return float(ModalCertainty.POSSIBLE)
        # Return the minimum certainty of all modals found (most conservative)
        return float(min(
            MODAL_CERTAINTY_MAP.get(m, ModalCertainty.POSSIBLE)
            for m in modal_lemmas
        ))

    # ------------------------------------------------------------------
    # Surface modality string
    # ------------------------------------------------------------------

    def _surface_modality(self, ptype: PredicateType, lemma_set: frozenset) -> str:
        """Return the modal auxiliary string for MODAL frames, else 'CERTAIN'."""
        if ptype != PredicateType.MODAL:
            return "CERTAIN"
        modal_lemmas = lemma_set & _MODAL_AUXILIARIES
        return next(iter(modal_lemmas)) if modal_lemmas else "CERTAIN"

    # ------------------------------------------------------------------
    # RefutationBurden resolution
    # ------------------------------------------------------------------

    def _resolve_burden(
        self, ptype: PredicateType, scope: QuantifierScope
    ) -> RefutationBurden:
        """Assign refutation burden from predicate type and scope."""
        if ptype == PredicateType.NORMATIVE:
            return RefutationBurden.NON_FALSIFIABLE
        if scope == QuantifierScope.UNIVERSAL:
            return RefutationBurden.SINGLE_COUNTEREXAMPLE
        if scope == QuantifierScope.EXISTENTIAL:
            return RefutationBurden.EXHAUSTIVE
        # MODAL always gets SCOPE_BOUNDED (per spec v0.2.1)
        if ptype == PredicateType.MODAL:
            return RefutationBurden.SCOPE_BOUNDED
        return RefutationBurden.SCOPE_BOUNDED

    # ------------------------------------------------------------------
    # NER alignment
    # ------------------------------------------------------------------

    def _build_ner_lookup(self, analysis: SentenceAnalysis) -> Dict[str, str]:
        """Build a lowercased surface-text -> NER label dict from token stream."""
        lookup: Dict[str, str] = {}
        for token in analysis.tokens:
            # spaCy NER labels come through via pos-tag's tag field for named
            # entities in some configurations; check both dep and tag.
            tag = token.tag.upper()
            if tag in _NER_LABEL_MAP:
                lookup[token.token.lower()] = _NER_LABEL_MAP[tag]
                lookup[token.lemma.lower()] = _NER_LABEL_MAP[tag]
        return lookup

    # ------------------------------------------------------------------
    # Presupposition frame injection
    # ------------------------------------------------------------------

    def _inject_presupposition_frames(
        self,
        store: PredicateStore,
        analysis: SentenceAnalysis,
        domain_hint: str,
    ) -> None:
        """For frames with presupposition_attack=True, create and link a
        presupposition frame.

        The presupposition frame is a FACTIVE frame representing the hidden
        embedded assumption. It is inserted into the store and connected to
        the parent frame via a PRESUPPOSES edge.

        This is a lightweight structural injection — no semantic reasoning
        about the presupposition content is performed here.
        """
        primary = store.primary()
        if primary is None or not primary.presupposition_attack:
            return

        # Build a short presupposition surface form from the triggers
        trigger_desc = ", ".join(analysis.presupposition_triggers[:2])
        presup_surface = f"[presupposition of: {analysis.raw_text[:60]}] ({trigger_desc})"

        presup_pid = make_predicate_id(
            presup_surface, PredicateType.FACTIVE, domain_hint
        )

        presup_frame = PredicateFrame(
            predicate_id=presup_pid,
            surface_form=presup_surface,
            predicate_type=PredicateType.FACTIVE,
            subject=primary.subject,
            relation=PredicateRelation(lemma="be", polarity="POSITIVE"),
            object=PredicateEntity(text="[presupposed]"),
            falsifiable=True,
            refutation_burden=RefutationBurden.SCOPE_BOUNDED,
            derived_from=[primary.predicate_id],
            generated_by_phase="PredicateExtractor/presupposition",
            domain_hint=domain_hint,
        )

        store.add(presup_frame)
        store.add_relation(
            source_id=primary.predicate_id,
            target_id=presup_pid,
            relation_type=PredicateRelationType.PRESUPPOSES,
            confidence=0.8,
        )

        # Patch presupposition_frame reference onto the primary (in-place)
        # This is the one allowed in-place mutation post-construction;
        # it does NOT increment semantic_revision (it's not a transform).
        object.__setattr__ if False else None  # noqa: dataclass field update
        primary.presupposition_frame = presup_frame


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_extractor_instance: Optional[PredicateExtractor] = None


def get_extractor() -> PredicateExtractor:
    """Return the module-level PredicateExtractor singleton."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = PredicateExtractor()
    return _extractor_instance
