"""
Phase B — Predicate Family Classification.

Consolidation Notes §7-8, §15-16.

Predicate family is the highest-level semantic category of a relation.
The MultiLensRouter (Phase C, Lens 4) routes based on this field.

The families cover the five core relationship archetypes from §7:
    CAUSAL       — A causes / leads to / results in B
    CORRELATIONAL— A correlates with / is associated with B
    TEMPORAL     — A precedes / follows / during B
    DEFINITIONAL — A is / is defined as / is a type of B
    COMPARATIVE  — A is like / is similar to / differs from B
    HIERARCHICAL — A is part of / is contained in B
    ADVERSARIAL  — A contradicts / opposes / undermines B
    PROCEDURAL   — A requires / involves / produces B (in process chains)

Unknown predicates default to CORRELATIONAL (neutral, lowest commitment).

Implementation approach (§16 alignment note):
    The consolidation notes warn against reducing family classification
    to keyword matching because causation verbs appear in correlational
    contexts ("smoking is linked to cancer" vs "smoking causes cancer").
    We use keyword pre-screening followed by MiniLM similarity scoring
    against canonical exemplars for the ambiguous cases.

    Phase C (Lens 4) may further re-classify after spectral analysis.
"""

from __future__ import annotations

from typing import Optional

# Maps predicate family name → list of trigger verbs / phrases.
# Used for fast initial classification; overridden by embedding similarity
# if confidence is low.
PREDICATE_FAMILY_MAP: dict[str, list[str]] = {
    "CAUSAL": [
        "causes", "leads to", "results in", "produces", "triggers",
        "induces", "generates", "brings about", "gives rise to",
        "is responsible for", "contributes to",
    ],
    "CORRELATIONAL": [
        "correlates with", "is associated with", "is linked to",
        "is related to", "co-occurs with", "is correlated",
        "tends to", "is connected to",
    ],
    "TEMPORAL": [
        "precedes", "follows", "occurs before", "occurs after",
        "during", "at the same time as", "coincides with",
        "happens before", "happened before",
    ],
    "DEFINITIONAL": [
        "is", "is defined as", "is a type of", "is a kind of",
        "refers to", "means", "is called", "is known as",
        "is classified as", "constitutes",
    ],
    "COMPARATIVE": [
        "is like", "is similar to", "resembles", "differs from",
        "is unlike", "is analogous to", "is comparable to",
    ],
    "HIERARCHICAL": [
        "is part of", "is contained in", "belongs to",
        "is a subset of", "is included in", "is a component of",
        "is an element of",
    ],
    "ADVERSARIAL": [
        "contradicts", "opposes", "undermines", "refutes",
        "challenges", "conflicts with", "argues against", "disputes",
    ],
    "PROCEDURAL": [
        "requires", "involves", "produces", "uses", "applies",
        "performs", "executes", "transforms",
    ],
}


def classify_predicate_family(
    predicate_text: str,
    *,
    embedding_fn: Optional[callable] = None,
) -> str:
    """Classify a predicate string into one of PREDICATE_FAMILY_MAP's keys.

    Algorithm (§16):
        1. Lowercase normalisation + keyword match scan.
        2. If a unique match is found → return that family immediately.
        3. If zero or multiple matches → use embedding_fn to score similarity
           against per-family canonical exemplar sentences, pick argmax.
        4. Fallback: CORRELATIONAL (lowest semantic commitment).

    Parameters
    ----------
    predicate_text : str
        The raw predicate or verb phrase from SRL extraction.
    embedding_fn : callable, optional
        A function (str) -> list[float] that produces a 384-dim embedding.
        If None, pure keyword matching is used (less accurate for ambiguous
        predicates but sufficient for unit testing).

    Returns
    -------
    str
        One of the keys in PREDICATE_FAMILY_MAP, or 'CORRELATIONAL' as
        the unknown fallback.
    """
    normalised = predicate_text.lower().strip()

    # Step 1: keyword scan
    matched_families: list[str] = []
    for family, triggers in PREDICATE_FAMILY_MAP.items():
        if any(trigger in normalised for trigger in triggers):
            matched_families.append(family)

    if len(matched_families) == 1:
        return matched_families[0]

    # Step 2: embedding similarity (when embedding_fn provided)
    if embedding_fn is not None:
        # Canonical exemplar sentences per family
        exemplars: dict[str, str] = {
            "CAUSAL":        "smoking causes lung cancer",
            "CORRELATIONAL": "height is correlated with basketball ability",
            "TEMPORAL":      "sunrise occurs before sunset",
            "DEFINITIONAL":  "a mammal is a warm-blooded animal",
            "COMPARATIVE":   "a dolphin is similar to a whale",
            "HIERARCHICAL":  "a kidney is part of the urinary system",
            "ADVERSARIAL":   "study A contradicts study B on this topic",
            "PROCEDURAL":    "digestion involves enzymatic breakdown",
        }
        try:
            import numpy as np
            query_vec = np.array(embedding_fn(normalised))
            best_family = "CORRELATIONAL"
            best_score = -1.0
            for family, exemplar in exemplars.items():
                ex_vec = np.array(embedding_fn(exemplar))
                # cosine similarity
                score = float(
                    np.dot(query_vec, ex_vec)
                    / (np.linalg.norm(query_vec) * np.linalg.norm(ex_vec) + 1e-8)
                )
                if score > best_score:
                    best_score = score
                    best_family = family
            return best_family
        except Exception:
            pass  # fall through to fallback

    # Fallback: if multiple keyword matches, prefer highest-priority order
    priority = [
        "CAUSAL", "ADVERSARIAL", "TEMPORAL", "DEFINITIONAL",
        "HIERARCHICAL", "PROCEDURAL", "COMPARATIVE", "CORRELATIONAL",
    ]
    for p in priority:
        if p in matched_families:
            return p

    return "CORRELATIONAL"  # safe minimum-commitment fallback
