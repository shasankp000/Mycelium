# mycelium/pipeline/predicates/predicate_types.py
# Semantic IR types for the Mycelium predicate engine.
#
# These are first-class runtime objects, NOT temporary parsing artifacts.
# PredicateFrames survive across retrieval, contradiction, stabilization,
# replay, and patch propagation.
#
# Spec ref: implementation spec v0.2.1 — Section 4.1

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Literal, Optional, Tuple


# ---------------------------------------------------------------------------
# Core enums
# ---------------------------------------------------------------------------


class PredicateType(str, Enum):
    """Semantic type of a predicate claim."""

    FACTIVE = "FACTIVE"
    """Asserts a fact directly: 'Paris is the capital of France'."""

    COMPARATIVE = "COMPARATIVE"
    """Asserts a comparison: 'Nuclear is safer than coal per TWh'."""

    CAUSAL = "CAUSAL"
    """Asserts causation: 'Smoking causes cancer'."""

    EXISTENTIAL = "EXISTENTIAL"
    """Asserts existence: 'There exist black holes smaller than 1 km'."""

    TEMPORAL = "TEMPORAL"
    """Asserts a time-bound fact: 'The policy was enacted in 1994'."""

    MODAL = "MODAL"
    """Asserts possibility/probability: 'AI might displace workers'."""

    NORMATIVE = "NORMATIVE"
    """Asserts a value judgement: 'Democracy is the best system'.
    NOT falsifiable by evidence. Never enters the proof tree."""

    DEFINITIONAL = "DEFINITIONAL"
    """Asserts a definition: 'Mammals are warm-blooded'."""


class QuantifierScope(str, Enum):
    """Implicit quantification scope of the predicate.

    Determines the De Morgan negation strategy and refutation burden.
    """

    UNIVERSAL = "UNIVERSAL"
    """'All X are P' — one counterexample suffices to refute."""

    EXISTENTIAL = "EXISTENTIAL"
    """'Some X is P' — exhaustive refutation required to disprove."""

    SCOPED = "SCOPED"
    """'All X in context C are P' — counterexample must be within scope."""

    AMBIGUOUS = "AMBIGUOUS"
    """Scope unclear from surface form; treated as SCOPED at runtime."""

    UNSCOPED = "UNSCOPED"
    """No quantification signal present in the surface form.
    Used by pre-v0.2.1 extractor path and older test fixtures.
    Treated identically to AMBIGUOUS at runtime."""


class RefutationBurden(str, Enum):
    """How hard it is to refute this predicate.

    Set by PredicateNegator based on predicate type and quantifier scope.
    Read by EvidenceFinder to gate search depth and early-halt logic.
    """

    SINGLE_COUNTEREXAMPLE = "SINGLE_COUNTEREXAMPLE"
    """One strong contradicting piece of evidence is sufficient."""

    EXHAUSTIVE = "EXHAUSTIVE"
    """All relevant instances must be checked before refutation is possible."""

    SCOPE_BOUNDED = "SCOPE_BOUNDED"
    """Refutation evidence must fall within the predicate's scope conditions."""

    NON_FALSIFIABLE = "NON_FALSIFIABLE"
    """Cannot be refuted by evidence (NORMATIVE). Skip retrieval entirely."""


class EpistemicBurden(str, Enum):
    """Epistemic difficulty of establishing or scoring the truth of a predicate claim.

    Spec ref: implementation spec v0.2.1 — Section 4.1 (burden stratification).

    Two vocabularies coexist here:

    Scorer multiplier vocabulary (used by EvidenceScorer and test_evidence_scorer):
      NONE          — no burden adjustment; multiplier = 1.0
      SCOPE_BOUNDED — scope-restricted claim;  multiplier = 0.85
      REBUTTAL      — rebuttal-class claim;     multiplier = 1.10 (capped at 1.0)
      EXHAUSTIVE    — exhaustive check needed;  multiplier = 0.90
      MODAL         — modal/possibility claim;  multiplier = 0.70 * ModalCertainty

    Retrieval-depth vocabulary (used by EvidenceFinder and DomainToolPlanner):
      EMPIRICAL     — directly verifiable; lowest search depth
      INFERENTIAL   — requires inference across sources; medium depth
      INTERPRETIVE  — requires theoretical framing; high depth
      NORMATIVE     — value judgement; skip retrieval entirely
      CONTESTED     — factually contested; force multi-perspective retrieval

    TRM v2 will unify these into a single vocabulary.  Until then both sets
    are present so both scorer tests and finder tests can import from here.
    """

    # --- Scorer multiplier vocabulary ---
    NONE = "NONE"
    """No burden adjustment. EvidenceScorer multiplier = 1.0."""

    SCOPE_BOUNDED = "SCOPE_BOUNDED"
    """Scope-restricted claim. EvidenceScorer multiplier = 0.85."""

    REBUTTAL = "REBUTTAL"
    """Rebuttal-class claim (high-value counter-evidence). Multiplier = 1.10, capped at 1.0."""

    EXHAUSTIVE = "EXHAUSTIVE"
    """Exhaustive-check required claim. EvidenceScorer multiplier = 0.90."""

    MODAL = "MODAL"
    """Modal/possibility claim. EvidenceScorer multiplier = 0.70 * ModalCertainty value."""

    # --- Retrieval-depth vocabulary ---
    EMPIRICAL = "EMPIRICAL"
    """Claim is directly verifiable by observation or measurement.
    E.g. 'Water boils at 100°C at sea level'. Lowest search depth."""

    INFERENTIAL = "INFERENTIAL"
    """Claim requires inference across multiple sources.
    E.g. 'Smoking causes lung cancer'. Medium search depth."""

    INTERPRETIVE = "INTERPRETIVE"
    """Claim requires interpretation under a theoretical frame.
    E.g. 'The French Revolution accelerated secularisation'. High search depth."""

    NORMATIVE = "NORMATIVE"
    """Claim is a value judgement. EvidenceFinder skips retrieval entirely."""

    CONTESTED = "CONTESTED"
    """Claim is factually contested across credible sources.
    Forces multi-perspective retrieval and disables early-halt."""


class ModalCertainty(float, Enum):
    """Certainty level implied by a modal auxiliary.

    Used exclusively by Rule N6 (MODAL negation) to cap net_confidence
    on EvidenceBundle. Not used anywhere else in the pipeline.

    Rationale: disproving 'X happens' does not disprove 'X might happen'.
    The ceiling prevents the evidence layer from overclaiming certainty
    about a possibility-class statement.
    """

    CERTAIN = 1.00
    """'will', 'shall' — treated as FACTIVE in practice."""

    PROBABLE = 0.75
    """'should', 'ought', 'likely'."""

    POSSIBLE = 0.50
    """'may', 'might', 'could', 'can'."""

    SPECULATIVE = 0.25
    """'conceivably', 'possibly'."""


# Map of modal auxiliary/adverb surface lemmas to ModalCertainty.
# Used by PredicateExtractor to set modal_certainty on MODAL frames,
# and by Rule N6 in PredicateNegator.
# If a detected auxiliary is absent from this map, default to POSSIBLE.
MODAL_CERTAINTY_MAP: dict[str, ModalCertainty] = {
    "will":        ModalCertainty.CERTAIN,
    "shall":       ModalCertainty.CERTAIN,
    "should":      ModalCertainty.PROBABLE,
    "ought":       ModalCertainty.PROBABLE,
    "likely":      ModalCertainty.PROBABLE,
    "may":         ModalCertainty.POSSIBLE,
    "might":       ModalCertainty.POSSIBLE,
    "could":       ModalCertainty.POSSIBLE,
    "can":         ModalCertainty.POSSIBLE,
    "conceivably": ModalCertainty.SPECULATIVE,
    "possibly":    ModalCertainty.SPECULATIVE,
}


# ---------------------------------------------------------------------------
# Sub-structures
# ---------------------------------------------------------------------------


@dataclass
class PredicateEntity:
    """A subject or object participant in a predicate."""

    text: str
    """Surface text span of the entity."""

    entity_type: str = ""
    """spaCy NER label, e.g. 'ORG', 'GPE', 'PERSON'. Empty if unknown."""

    confidence: float = 0.8
    """Extractor confidence in the entity boundary and type."""


@dataclass
class PredicateRelation:
    """The relation (verb/copula/adjective) linking subject to object."""

    lemma: str
    """Lemmatised relation, e.g. 'be', 'cause', 'safer_than', 'displace'."""

    polarity: Literal["POSITIVE", "NEGATIVE"] = "POSITIVE"
    """NEGATIVE if the surface form contains explicit negation ('does not')."""

    tense: str = "PRESENT"
    """Tense of the verb: PRESENT | PAST | FUTURE | GNOMIC."""

    modality: str = "CERTAIN"
    """Surface modality string, e.g. 'CERTAIN', 'might', 'should'."""


@dataclass
class TransformationRecord:
    """Immutable record of a single semantic transform applied to a PredicateFrame.

    Appended to PredicateFrame.transformation_history on every structural
    change (negation, scope narrowing, ontology rewrite, patch application).
    Never mutated after creation.
    """

    timestamp: float
    """Unix timestamp of the transformation."""

    transformation_type: str
    """Short label, e.g. 'POLARITY_FLIP', 'MODAL_FACTIVE_EMBED',
    'SCOPE_NARROWING', 'DE_MORGAN_INVERSION', 'PRESUPPOSITION_ATTACK'."""

    source_predicate_id: str
    """predicate_id of the frame this was derived from."""

    resulting_predicate_id: str
    """predicate_id of the frame produced by this transformation."""

    reason: str
    """Human-readable explanation, e.g. 'Rule N1: polarity flip on FACTIVE'."""

    triggered_by: str
    """Component that triggered this transform, e.g. 'PredicateNegator/N1'."""


# ---------------------------------------------------------------------------
# PredicateFrame — the core semantic IR object
# ---------------------------------------------------------------------------


@dataclass
class PredicateFrame:
    """A first-class semantic reasoning unit representing a single structured claim.

    PredicateFrames are NOT temporary parsing artifacts. They are created by
    PredicateExtractor, transformed by PredicateNegator, stored in
    PredicateStore, and referenced across the full pipeline including
    retrieval, contradiction, stabilization, replay, and patch propagation.

    Two construction paths coexist:

    v0.2.1 structured path (preferred):
        subject: PredicateEntity
        relation: PredicateRelation
        object: PredicateEntity | str

    Pre-v0.2.1 flat path (used by older tests and extractor code):
        raw_text: str
        subject: str
        predicate_verb: str
        object_concept: str

    TRM v2 will migrate all callers to the structured path.

    Identity
    --------
    predicate_id is a deterministic SHA-256 digest of (surface_form +
    predicate_type + domain_hint), computed at construction time via
    `make_predicate_id()`. This guarantees replay-stable identity.
    """

    # -- Identity --
    predicate_id: str
    """Stable deterministic identifier. Use make_predicate_id() to generate."""

    surface_form: str = ""
    """Original sentence span that produced this predicate (v0.2.1 path)."""

    # -- Semantic type --
    predicate_type: PredicateType = PredicateType.FACTIVE

    # -- v0.2.1 structured path --
    subject: object = None  # PredicateEntity | str | None
    relation: Optional["PredicateRelation"] = None
    object: object = None   # PredicateEntity | str | None

    # -- Pre-v0.2.1 flat path (older tests / extractor) --
    raw_text: Optional[str] = None
    """Alias for surface_form used by pre-v0.2.1 test fixtures."""

    predicate_verb: Optional[str] = None
    """Flat verb string; pre-v0.2.1 alternative to relation.lemma."""

    object_concept: Optional[str] = None
    """Flat object string; pre-v0.2.1 alternative to object."""

    # -- Polarity and scope --
    negated: bool = False
    """True if the surface form already contains explicit negation."""

    quantifier_scope: QuantifierScope = QuantifierScope.AMBIGUOUS

    scope_conditions: List[str] = field(default_factory=list)
    """Explicit scope qualifiers extracted from text, e.g. ['per TWh', 'post-1990']."""

    # -- Presupposition --
    presupposition_attack: bool = False
    """True when refutation should target the hidden presupposition, not the
    surface claim. See Rule N6 (presupposition attack) in the negator."""

    presupposition_frame: Optional["PredicateFrame"] = None
    """The extracted presupposition frame, if presupposition_attack is True."""

    # -- Falsifiability --
    falsifiable: bool = True
    """False for NORMATIVE predicates. EvidenceFinder skips non-falsifiable frames."""

    refutation_burden: RefutationBurden = RefutationBurden.SCOPE_BOUNDED

    # -- Epistemic burden (v0.2.1 + scorer vocabulary) --
    epistemic_burden: Optional[EpistemicBurden] = None
    """Governs EvidenceFinder search depth and EvidenceScorer multiplier.
    None means not yet classified (scorer treats as NONE)."""

    # -- Negated form (set by PredicateNegator) --
    negated_form: Optional["PredicateFrame"] = None
    """Deterministic structural negation of this frame. None for NORMATIVE."""

    # -- Routing hint --
    domain_hint: str = "general"
    """Domain string from L1 MultiLens routing, e.g. 'science', 'history'."""

    # -- Source location --
    source_span: Tuple[int, int] = (0, 0)
    """Character offsets (start, end) in the original raw query string."""

    # -- Provenance (hardened additions) --
    derived_from: List[str] = field(default_factory=list)
    """predicate_ids of parent frames this was derived from."""

    generated_by_phase: str = "PredicateExtractor"
    """Name of the pipeline component that created or last transformed this frame."""

    transformation_history: List[TransformationRecord] = field(default_factory=list)
    """Ordered log of every structural transform applied. Never mutated in-place;
    new records are appended."""

    semantic_revision: int = 0
    """Incremented on every structural transformation. Starts at 0 at extraction."""

    provenance_confidence: float = 1.0
    """Confidence that this frame accurately represents the original claim.
    Decreases when scope is narrowed or presuppositions are attacked."""

    graph_links: List[str] = field(default_factory=list)
    """predicate_ids of related frames in the PredicateStore relation graph."""

    # -- Modal-specific (v0.2.1) --
    modal_certainty: Optional[float] = None
    """Set only for MODAL predicates. Sourced from MODAL_CERTAINTY_MAP via
    the detected auxiliary. EvidenceFinder reads this to cap net_confidence
    on the resulting EvidenceBundle. Not set or read for any other type."""


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def make_predicate_id(surface_form: str, predicate_type: PredicateType, domain_hint: str = "general") -> str:
    """Produce a deterministic, replay-stable predicate_id.

    The id is a 16-character hex prefix of SHA-256(surface_form + predicate_type + domain_hint).
    Collisions are astronomically unlikely for distinct claims within a single request.
    """
    raw = f"{surface_form}|{predicate_type.value}|{domain_hint}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def make_transformation_record(
    transformation_type: str,
    source_predicate_id: str,
    resulting_predicate_id: str,
    reason: str,
    triggered_by: str,
) -> TransformationRecord:
    """Convenience constructor that injects the current timestamp."""
    return TransformationRecord(
        timestamp=time.time(),
        transformation_type=transformation_type,
        source_predicate_id=source_predicate_id,
        resulting_predicate_id=resulting_predicate_id,
        reason=reason,
        triggered_by=triggered_by,
    )
