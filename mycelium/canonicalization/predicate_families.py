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

Optimisations (§5.2):
    - Module-level _family_cache (OrderedDict, max 4096 entries) memoises
      per-lemma results.  Predicate lemmas from domain text are finite and
      highly repetitive so the hit rate after warm-up is close to 100 %.
    - classify_batch() batches all *uncached* lemmas into a single
      zero-shot pipeline call (batch_size=32) rather than one call per
      lemma, eliminating N-1 pipeline round-trips for multi-predicate
      sentences.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import List, Optional

# ---------------------------------------------------------------------------
# Predicate family keyword map
# ---------------------------------------------------------------------------

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

FAMILY_LABELS = list(PREDICATE_FAMILY_MAP.keys())

# ---------------------------------------------------------------------------
# Module-level LRU cache for predicate family results (§5.2)
# Max 4096 entries — predicate lemma vocabulary is bounded.
# ---------------------------------------------------------------------------
_FAMILY_CACHE_MAX = 4096
_family_cache: OrderedDict[str, str] = OrderedDict()


def _cache_get(key: str) -> Optional[str]:
    """O(1) cache lookup with LRU promotion."""
    if key in _family_cache:
        _family_cache.move_to_end(key)
        return _family_cache[key]
    return None


def _cache_put(key: str, value: str) -> None:
    """Insert into cache, evicting oldest entry when full."""
    if key in _family_cache:
        _family_cache.move_to_end(key)
    else:
        if len(_family_cache) >= _FAMILY_CACHE_MAX:
            _family_cache.popitem(last=False)
    _family_cache[key] = value


# ---------------------------------------------------------------------------
# Batch classification (§5.2)
# ---------------------------------------------------------------------------

def classify_batch(
    predicate_lemmas: List[str],
    *,
    embedding_fn: Optional[callable] = None,
) -> List[str]:
    """Classify a list of predicate lemmas in a single batched pipeline call.

    Algorithm (§5.2):
        1. Return cached results for lemmas already in _family_cache.
        2. Attempt keyword classification for remaining lemmas.
        3. For any still-ambiguous lemmas (0 or multiple keyword matches),
           run a single zero-shot NLI pipeline call with batch_size=32.
        4. Store all results in _family_cache before returning.

    Parameters
    ----------
    predicate_lemmas : list[str]
        Raw predicate or verb phrases from SRL extraction.
    embedding_fn : callable, optional
        Forwarded to classify_predicate_family() for ambiguous cases when
        the NLI pipeline is unavailable.

    Returns
    -------
    list[str]
        Family label for each lemma, in the same order as the input.
    """
    results: list[Optional[str]] = [None] * len(predicate_lemmas)
    needs_nli: list[int] = []  # indices still unresolved after keyword scan

    for idx, lemma in enumerate(predicate_lemmas):
        normalised = lemma.lower().strip()
        cached = _cache_get(normalised)
        if cached is not None:
            results[idx] = cached
            continue

        # Fast keyword pass
        matched: list[str] = [
            family
            for family, triggers in PREDICATE_FAMILY_MAP.items()
            if any(trigger in normalised for trigger in triggers)
        ]
        if len(matched) == 1:
            _cache_put(normalised, matched[0])
            results[idx] = matched[0]
        else:
            needs_nli.append(idx)

    if needs_nli:
        texts_to_classify = [predicate_lemmas[i].lower().strip() for i in needs_nli]
        nli_results: list[str] = []

        try:
            from transformers import pipeline as hf_pipeline  # type: ignore
            zsl = hf_pipeline(
                "zero-shot-classification",
                model="typeform/distilbert-base-uncased-mnli",
            )
            # Single batched call — eliminates N-1 pipeline round-trips (§5.2)
            outputs = zsl(texts_to_classify, candidate_labels=FAMILY_LABELS, batch_size=32)
            if not isinstance(outputs, list):
                outputs = [outputs]
            nli_results = [out["labels"][0] for out in outputs]
        except Exception:
            # Fallback: per-lemma keyword classification
            for t in texts_to_classify:
                nli_results.append(
                    classify_predicate_family(t, embedding_fn=embedding_fn)
                )

        for list_pos, original_idx in enumerate(needs_nli):
            family = nli_results[list_pos] if list_pos < len(nli_results) else "CORRELATIONAL"
            normalised = predicate_lemmas[original_idx].lower().strip()
            _cache_put(normalised, family)
            results[original_idx] = family

    return [r or "CORRELATIONAL" for r in results]


# ---------------------------------------------------------------------------
# Single-lemma classification (unchanged public API — uses cache)
# ---------------------------------------------------------------------------

def classify_predicate_family(
    predicate_text: str,
    *,
    embedding_fn: Optional[callable] = None,
) -> str:
    """Classify a predicate string into one of PREDICATE_FAMILY_MAP's keys.

    Algorithm (§16, augmented with §5.2 caching):
        0. Check _family_cache first — O(1) return for known lemmas.
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

    # Step 0: cache hit (§5.2)
    cached = _cache_get(normalised)
    if cached is not None:
        return cached

    # Step 1: keyword scan
    matched_families: list[str] = []
    for family, triggers in PREDICATE_FAMILY_MAP.items():
        if any(trigger in normalised for trigger in triggers):
            matched_families.append(family)

    if len(matched_families) == 1:
        _cache_put(normalised, matched_families[0])
        return matched_families[0]

    # Step 2: embedding similarity (when embedding_fn provided)
    if embedding_fn is not None:
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
                score = float(
                    np.dot(query_vec, ex_vec)
                    / (np.linalg.norm(query_vec) * np.linalg.norm(ex_vec) + 1e-8)
                )
                if score > best_score:
                    best_score = score
                    best_family = family
            _cache_put(normalised, best_family)
            return best_family
        except Exception:
            pass

    # Fallback: if multiple keyword matches, prefer highest-priority order
    priority = [
        "CAUSAL", "ADVERSARIAL", "TEMPORAL", "DEFINITIONAL",
        "HIERARCHICAL", "PROCEDURAL", "COMPARATIVE", "CORRELATIONAL",
    ]
    for p in priority:
        if p in matched_families:
            _cache_put(normalised, p)
            return p

    _cache_put(normalised, "CORRELATIONAL")
    return "CORRELATIONAL"
