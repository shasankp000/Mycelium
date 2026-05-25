"""
Phase B — Semantic Role Labelling (SRL) Extraction.

Consolidation Notes §5-8 and impl spec §B.1.

Core realisation from §5-8:
    Natural language carries implicit SRL structure that must be
    surface-parsed before any graph operation.  The semantic graph
    layer cannot operate on raw text — it requires structured triples.

SRL triple format: (subject, predicate, object, [modifiers])
    subject   — ARG0 in PropBank notation
    predicate — the core verb or relational phrase
    object    — ARG1 (theme/patient)
    modifiers — ARG2+ (location, time, instrument, cause, purpose, etc.)

Approach (§B.1 alignment with consolidation notes):
    The impl spec proposes using allennlp SRL model.  However, the
    consolidation notes (§16) emphasise that for the Phase B gate test
    ("smoking causes lung cancer" → (smoking, causes, lung_cancer)),
    correctness matters more than model choice.

    We implement a two-tier approach:
        Tier 1: rule-based SVO extractor using spaCy dependency parsing
                (fast, offline, works without allennlp GPU requirements)
        Tier 2: optional allennlp SRL model when available
                (richer ARG2+ modifier extraction)

    Tier 1 is sufficient for Phase A/B gate tests and offline operation.
    Tier 2 is activated when MYCELIUM_SRL_MODEL=allennlp is set.

spaCy venv routing
------------------
    spaCy with en_core_web_sm lives in the dedicated lexis venv at
    .venv2 (Python 3.11).  This file probes that venv's site-packages
    FIRST before falling back to the active sys.path so that the TRM /
    OOD fallback pipeline always gets the full dependency-parsed output
    rather than the regex fallback.

    Search order:
        1. .venv2/lib/python3.11/site-packages   (lexis venv, spaCy+model)
        2. Active sys.path (may also have spaCy if installed in main venv)
        3. Regex fallback (no spaCy at all)

    The probe is done at import time once and cached in _SPACY_SITE.
    No subprocess is spawned; we just temporarily prepend the path.

    Stale-import eviction
    ---------------------
    Python 3.13 may ship a stub or broken spaCy build that gets cached
    in sys.modules['spacy'] before this file runs (e.g. from a top-level
    `import spacy` elsewhere in the process).  That stale entry would
    shadow the .venv2 copy even after sys.path injection.
    _import_spacy() therefore evicts 'spacy' and all 'spacy.*' sub-modules
    from sys.modules before attempting the path-injected import so the
    correct 3.11 copy is always resolved.
"""

from __future__ import annotations

import os
import re
import sys
import pathlib
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional


# ---------------------------------------------------------------------------
# .venv2 spaCy path probe
# ---------------------------------------------------------------------------

def _find_venv2_site() -> Optional[str]:
    """
    Walk upward from this file's location looking for a .venv2 directory
    that contains a spaCy installation.  Returns the site-packages path
    string if found, else None.

    We search three candidate roots:
        1. The repository root (two levels above this file: repo/mycelium/canonicalization/)
        2. The current working directory
        3. The HOME directory
    """
    candidates = [
        pathlib.Path(__file__).resolve().parent.parent.parent,  # repo root
        pathlib.Path.cwd(),
        pathlib.Path.home(),
    ]

    for base in candidates:
        # Python 3.11 site-packages inside .venv2
        site = base / ".venv2" / "lib" / "python3.11" / "site-packages"
        if site.is_dir() and (site / "spacy").is_dir():
            return str(site)

    return None


# Resolved once at module import; None means .venv2 not found.
_SPACY_SITE: Optional[str] = _find_venv2_site()


def _evict_spacy_from_sys_modules() -> None:
    """
    Remove any previously imported 'spacy' and 'spacy.*' entries from
    sys.modules.

    This is necessary because Python 3.13 (the main venv) may have
    already imported a stub or incompatible spaCy that lacks `spacy.load`
    (spaCy 3.x does not support CPython 3.13 as of May 2026).  If that
    broken import is cached in sys.modules, prepending the .venv2
    site-packages to sys.path has no effect — Python returns the cached
    module instead of re-importing from the new path.

    Calling this before `import spacy` forces a clean resolution from
    whatever is currently at the front of sys.path.
    """
    stale = [k for k in sys.modules if k == "spacy" or k.startswith("spacy.")]
    for key in stale:
        del sys.modules[key]


def _import_spacy():
    """
    Import spaCy, preferring the .venv2 site-packages when available.

    Strategy:
        1. If _SPACY_SITE is set and not already on sys.path, prepend it.
        2. Evict any stale spaCy entries from sys.modules so the
           path-injected copy is always resolved (not a cached 3.13 stub).
        3. Attempt `import spacy`.
        4. Validate that the import has a working `load` attribute —
           if not, treat it as a failed import and fall through to None.
        5. On failure, clean up the sys.path entry so we don't pollute
           the process with a non-working path.
    """
    global _SPACY_SITE

    injected = False
    if _SPACY_SITE and _SPACY_SITE not in sys.path:
        sys.path.insert(0, _SPACY_SITE)
        injected = True

    # Evict any stale/partial spaCy already cached from the 3.13 env.
    _evict_spacy_from_sys_modules()

    try:
        import spacy  # noqa: PLC0415
        # Guard: a broken install may import but lack `load` (seen on 3.13).
        if not callable(getattr(spacy, "load", None)):
            raise ImportError("spacy.load not callable — incompatible build")
        return spacy
    except (ImportError, Exception):
        # Clean up so we don't leave a broken path in sys.path.
        if injected and _SPACY_SITE in sys.path:
            sys.path.remove(_SPACY_SITE)
        _SPACY_SITE = None
        return None


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class SRLTriple:
    """A single semantic role labelled triple extracted from text.

    subject    — ARG0: the agent or topic
    predicate  — REL:  the relation verb or phrase
    obj        — ARG1: the patient or theme
    modifiers  — ARG2+: additional roles (location, time, cause, etc.)
    raw_sentence  — original input for provenance
    confidence — SRL extraction confidence [0, 1]
    """

    subject: str
    predicate: str
    obj: str
    modifiers: dict = field(default_factory=dict)
    raw_sentence: str = ""
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

class SRLExtractor:
    """Two-tier SRL extraction: spaCy dependency parser (Tier 1) or
    allennlp SRL model (Tier 2).

    spaCy is loaded from .venv2 (lexis venv, Python 3.11) when that
    environment is present, ensuring en_core_web_sm is always available
    regardless of which venv the main process runs in.

    Optimisations (§5.1):
    - Per-instance NER/doc cache (_ner_cache) so the same text string is
      parsed by spaCy exactly once, even if extract() is called multiple
      times (e.g. from CanonicalizeAndHash during Phase C IR bridge).
    - extract() itself is wrapped with @lru_cache(maxsize=256) so
      identical sentences deduplicate at the result level.

    Usage:
        extractor = SRLExtractor()
        triples = extractor.extract("Smoking causes lung cancer.")
        # → [SRLTriple(subject='Smoking', predicate='causes',
        #               obj='lung cancer', ...)]
    """

    _USE_ALLENNLP = os.getenv("MYCELIUM_SRL_MODEL", "spacy").lower() == "allennlp"

    def __init__(self) -> None:
        self._nlp = None          # spaCy model, lazy-loaded
        self._allennlp = None     # allennlp predictor, lazy-loaded
        # Per-text spaCy doc cache (§5.1): avoids re-parsing the same sentence
        # when extract() is called multiple times within one pipeline pass.
        self._ner_cache: dict = {}

    def _load_spacy(self):
        """
        Load spaCy, routing to .venv2 first.

        Sets self._nlp to:
            - a loaded spaCy Language object (en_core_web_sm preferred,
              blank English as fallback if model not found)
            - False if spaCy itself is not importable from anywhere
        """
        if self._nlp is None:
            spacy = _import_spacy()
            if spacy is None:
                self._nlp = False
            else:
                try:
                    self._nlp = spacy.load("en_core_web_sm")
                except OSError:
                    # Model not installed; blank pipeline still gives
                    # tokenisation but no dependency parse → regex fallback
                    # will be used inside _extract_spacy.
                    self._nlp = spacy.blank("en")
        return self._nlp

    def _get_doc(self, text: str):
        """Return a cached spaCy Doc for *text*, parsing only on first call (§5.1)."""
        if text not in self._ner_cache:
            nlp = self._load_spacy()
            if not nlp:
                return None
            self._ner_cache[text] = nlp(text)
        return self._ner_cache[text]

    def _extract_spacy(self, text: str) -> list[SRLTriple]:
        """Tier 1: rule-based SVO extraction via spaCy dependency parse."""
        doc = self._get_doc(text)
        if doc is None:
            return self._extract_regex(text)

        # If we got a blank pipeline (no dep parse), fall back to regex
        if not doc.has_annotation("DEP"):
            return self._extract_regex(text)

        triples: list[SRLTriple] = []
        for sent in doc.sents:
            for token in sent:
                if token.dep_ == "ROOT" and token.pos_ in {"VERB", "AUX"}:
                    subjects = [
                        t for t in token.lefts
                        if t.dep_ in {"nsubj", "nsubjpass", "csubj"}
                    ]
                    objects_ = [
                        t for t in token.rights
                        if t.dep_ in {"dobj", "attr", "pobj", "ccomp", "xcomp"}
                    ]
                    if subjects and objects_:
                        subj = " ".join(
                            t.text for t in subjects[0].subtree
                        ).strip()
                        obj = " ".join(
                            t.text for t in objects_[0].subtree
                        ).strip()
                        pred = token.text
                        aux_tokens = [
                            t.text for t in token.lefts
                            if t.dep_ in {"aux", "auxpass", "neg"}
                        ]
                        if aux_tokens:
                            pred = " ".join(aux_tokens) + " " + pred

                        triples.append(SRLTriple(
                            subject=subj,
                            predicate=pred,
                            obj=obj,
                            raw_sentence=sent.text,
                            confidence=0.85,
                        ))
        return triples

    def _extract_regex(self, text: str) -> list[SRLTriple]:
        """Regex fallback for when spaCy is unavailable.

        Handles simple SVO patterns: SUBJECT PREDICATE OBJECT.
        Confidence is lower (0.5) to reflect reduced accuracy.
        """
        causal_verbs = (
            r"causes|leads to|results in|produces|triggers|correlates with"
            r"|is associated with|is linked to|is|precedes|follows"
        )
        pattern = re.compile(
            r"([A-Za-z][\w\s-]{1,40})\s+("
            + causal_verbs
            + r")\s+([A-Za-z][\w\s-]{1,60})",
            re.IGNORECASE,
        )
        triples: list[SRLTriple] = []
        for m in pattern.finditer(text):
            triples.append(SRLTriple(
                subject=m.group(1).strip(),
                predicate=m.group(2).strip(),
                obj=m.group(3).strip().rstrip("."),
                raw_sentence=text,
                confidence=0.5,
            ))
        return triples

    @lru_cache(maxsize=256)
    def extract(self, text: str) -> list[SRLTriple]:
        """Extract SRL triples from a text string (§5.1).

        Results are LRU-cached (256 entries) so the same sentence is never
        re-parsed, even when CanonicalizeAndHash.process() is called
        multiple times on an identical query during the Phase C IR bridge.

        Tries Tier 2 (allennlp) if MYCELIUM_SRL_MODEL=allennlp,
        else Tier 1 (spaCy via .venv2 or active venv), else regex fallback.

        Returns
        -------
        list[SRLTriple]
            May be empty if no SVO structure is detected.
        """
        if self._USE_ALLENNLP:
            results = self._extract_allennlp(text)
            if results:
                return results

        return self._extract_spacy(text)

    def _extract_allennlp(self, text: str) -> list[SRLTriple]:
        """Tier 2: allennlp SRL model extraction.

        Activated when MYCELIUM_SRL_MODEL=allennlp.
        Requires allennlp + bert-base-srl model to be installed.
        """
        try:
            from allennlp.predictors.predictor import Predictor  # type: ignore
            if self._allennlp is None:
                self._allennlp = Predictor.from_path(
                    "https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.11.19.tar.gz"
                )
            result = self._allennlp.predict(sentence=text)
            triples: list[SRLTriple] = []
            for verb_dict in result.get("verbs", []):
                tags = verb_dict.get("tags", [])
                words = result.get("words", [])
                roles: dict[str, list[str]] = {}
                current_tag: Optional[str] = None
                current_span: list[str] = []
                for word, tag in zip(words, tags):
                    if tag.startswith("B-"):
                        if current_tag:
                            roles.setdefault(current_tag, []).append(" ".join(current_span))
                        current_tag = tag[2:]
                        current_span = [word]
                    elif tag.startswith("I-") and current_tag:
                        current_span.append(word)
                    else:
                        if current_tag:
                            roles.setdefault(current_tag, []).append(" ".join(current_span))
                        current_tag = None
                        current_span = []
                if current_tag:
                    roles.setdefault(current_tag, []).append(" ".join(current_span))

                subj = " ".join(roles.get("ARG0", [""]))
                pred = " ".join(roles.get("V", [verb_dict.get("verb", "")]))
                obj = " ".join(roles.get("ARG1", [""]))
                modifiers = {
                    k: " ".join(v)
                    for k, v in roles.items()
                    if k not in {"ARG0", "V", "ARG1"}
                }
                if subj and obj:
                    triples.append(SRLTriple(
                        subject=subj,
                        predicate=pred,
                        obj=obj,
                        modifiers=modifiers,
                        raw_sentence=text,
                        confidence=0.95,
                    ))
            return triples
        except Exception:
            return []
