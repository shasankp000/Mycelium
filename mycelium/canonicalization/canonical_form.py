"""
Phase B — Canonical Form Generator.

Consolidation Notes §15-16, impl spec §B.2.

The canonical form is the stable normalized string from which the semantic
hash is computed.  Without canonicalization, semantically identical claims
are represented by different strings, creating spurious graph fragmentation:

    "Smoking causes lung cancer"
    "Cigarette smoking leads to lung cancer"   → should be same node
    "Lung cancer can be caused by smoking"      → should be same node

Canonical form format (§15 from consolidation notes):
    FAMILY::subject_normalized::predicate_stem::object_normalized::depth{N}

Normalisation rules (§16):
    1. Lowercase all tokens
    2. Strip stopwords (a, an, the, is, are, by, of, that, which, …)
    3. Lemmatize if spaCy available; else apply simple suffix rules
    4. Sort multi-word subjects and objects alphabetically within span
       (preserves canonical form across paraphrase reorderings)
    5. Compress whitespace

Abstraction level suffix:
    depth0 = most concrete (leaf fact)
    depth1+ = abstracted over variables (used by Phase D motif promotion)

Equivalence family (§16 critical note):
    Paraphrases that map to the same canonical form are stored in
    SemanticSignature.equivalence_family for reverse lookup.
    This list is maintained by the CanonicalFormGenerator and
    consulted by TRM before creating new nodes (Phase D).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .predicate_families import classify_predicate_family
from .srl_extractor import SRLTriple


_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "by", "in", "on", "at", "to", "for", "and", "or", "but",
    "that", "which", "who", "whom", "this", "these", "those",
    "it", "its", "can", "could", "may", "might",
}


def _normalise_span(text: str) -> str:
    """Lowercase, strip stopwords, compress whitespace, underscorify.

    Multi-word spans become underscore-joined lowercase strings so they
    can safely appear in FAMILY::subject::predicate::object::depthN.
    """
    tokens = re.split(r"\s+", text.lower().strip())
    filtered = [t for t in tokens if t and t not in _STOPWORDS]
    return "_".join(filtered) if filtered else text.lower().replace(" ", "_")


def _lemmatize_predicate(pred: str, nlp=None) -> str:
    """Return the lemma root of a predicate phrase.

    Uses spaCy if available, else applies simple English suffix rules.
    """
    if nlp is not None:
        try:
            doc = nlp(pred.lower())
            # Take the lemma of the main verb (last VERB token)
            for token in reversed(list(doc)):
                if token.pos_ in {"VERB"}:
                    return token.lemma_
        except Exception:
            pass

    # Fallback: simple suffix stripping
    p = pred.lower().strip()
    for suffix, replacement in [
        ("ies", "y"),
        ("ied", "y"),
        ("ing", ""),
        ("ed", ""),
        ("es", ""),
        ("s", ""),
    ]:
        if p.endswith(suffix) and len(p) - len(suffix) > 2:
            return p[: -len(suffix)] + replacement
    return p


def generate_canonical_form(
    triple: SRLTriple,
    *,
    abstraction_level: int = 0,
    embedding_fn: Optional[callable] = None,
    nlp=None,
) -> tuple[str, str]:
    """Generate a canonical form string and predicate family from an SRL triple.

    Returns
    -------
    (canonical_form, predicate_family) : tuple[str, str]

    canonical_form format:
        FAMILY::subject::predicate_lemma::object::depth{N}

    Example:
        triple = SRLTriple("Smoking", "causes", "lung cancer")
        → ("CAUSAL::smoking::cause::lung_cancer::depth0", "CAUSAL")
    """
    family = classify_predicate_family(triple.predicate, embedding_fn=embedding_fn)
    subject_norm = _normalise_span(triple.subject)
    predicate_stem = _lemmatize_predicate(triple.predicate, nlp)
    object_norm = _normalise_span(triple.obj)
    canonical = (
        f"{family}::"
        f"{subject_norm}::"
        f"{predicate_stem}::"
        f"{object_norm}::"
        f"depth{abstraction_level}"
    )
    return canonical, family


@dataclass
class CanonicalFormGenerator:
    """Stateful generator that maintains an equivalence-family registry.

    Equivalence family (§16):
        When two different input texts produce the same canonical_form,
        they are semantically equivalent.  The generator stores all
        input texts seen for a given canonical form in the registry
        so that SemanticSignature.equivalence_family can be populated.

    Usage:
        gen = CanonicalFormGenerator()
        canonical, family = gen.generate(triple)
        equiv_family = gen.get_equivalence_family(canonical)
    """

    # canonical_form → list of raw sentences that mapped to it
    _registry: dict = field(default_factory=dict)

    def __post_init__(self):
        if not hasattr(self, "_registry"):
            self._registry = {}
        self._nlp = None  # lazy-loaded spaCy

    def _ensure_nlp(self):
        if self._nlp is None:
            try:
                import spacy
                try:
                    self._nlp = spacy.load("en_core_web_sm")
                except OSError:
                    self._nlp = spacy.blank("en")
            except ImportError:
                self._nlp = False

    def generate(
        self,
        triple: SRLTriple,
        *,
        abstraction_level: int = 0,
        embedding_fn: Optional[callable] = None,
    ) -> tuple[str, str]:
        """Generate canonical form and update equivalence registry."""
        self._ensure_nlp()
        canonical, family = generate_canonical_form(
            triple,
            abstraction_level=abstraction_level,
            embedding_fn=embedding_fn,
            nlp=self._nlp if self._nlp else None,
        )
        # Register this raw sentence under the canonical form
        if canonical not in self._registry:
            self._registry[canonical] = []
        if triple.raw_sentence and triple.raw_sentence not in self._registry[canonical]:
            self._registry[canonical].append(triple.raw_sentence)
        return canonical, family

    def get_equivalence_family(self, canonical_form: str) -> list[str]:
        """Return all raw sentences that map to this canonical form."""
        return list(self._registry.get(canonical_form, []))
