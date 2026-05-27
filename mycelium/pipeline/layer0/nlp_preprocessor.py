"""
Layer 0 — Shared NLP Preprocessor.

The Python equivalent of AI-Player's OpenNLPProcessor.java (which ran
Apache OpenNLP: sentence detector → tokenizer → POS tagger → lemmatizer).
Here we use spaCy (en_core_web_sm) to produce the same per-token metadata
struct: {token, lemma, pos, dep, is_stop}.

spaCy venv routing
------------------
spaCy with en_core_web_sm lives in the dedicated lexis venv at .venv2
(Python 3.11).  This file probes that venv's site-packages FIRST before
falling back to the active sys.path — identical strategy to
mycelium/canonicalization/srl_extractor.py.

    Search order:
        1. .venv2/lib/python3.11/site-packages   (lexis venv, spaCy+model)
        2. Active sys.path (may also have spaCy if installed in main venv)
        3. Regex / rule-only fallback (no spaCy at all)

Fallback behaviour
------------------
When spaCy is unavailable every downstream Layer 0 component degrades
gracefully:
  - Layers A/B/C fall back to pure lexical rules (regex + keyword sets).
  - Layer D (LLM arbiter) is triggered unconditionally because there are
    no model scores to pass to it — this matches the AI-Player pattern
    where getIntentionFromLLM() fires when the local classifiers are absent.

Public API
----------
    preprocessor = NLPPreprocessor()
    analysis = preprocessor.analyse("Why does smoking cause cancer?")
    # analysis.sentence_type  → "INTERROGATIVE"
    # analysis.dep_triples    → [("smoking", "causes", "cancer")]
    # analysis.evaluative_words → []
    # analysis.presupposition_triggers → ["factive_verb:cause"]
"""

from __future__ import annotations

import os
import re
import sys
import pathlib
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# .venv2 spaCy path probe  (same logic as srl_extractor.py)
# ---------------------------------------------------------------------------

def _find_venv2_site() -> Optional[str]:
    """Walk upward from this file looking for .venv2 with a spaCy install."""
    candidates = [
        pathlib.Path(__file__).resolve().parent.parent.parent.parent,  # repo root
        pathlib.Path(__file__).resolve().parent.parent.parent,         # mycelium/
        pathlib.Path.cwd(),
        pathlib.Path.home(),
    ]
    for base in candidates:
        site = base / ".venv2" / "lib" / "python3.11" / "site-packages"
        if site.is_dir() and (site / "spacy").is_dir():
            return str(site)
    return None


_SPACY_SITE: Optional[str] = _find_venv2_site()


def _evict_spacy_from_sys_modules() -> None:
    """Evict stale spacy / spacy.* entries from sys.modules.

    Python 3.13 may have cached a broken spaCy stub before this module
    runs.  Evicting it forces a clean resolution from the .venv2 path.
    """
    stale = [k for k in sys.modules if k == "spacy" or k.startswith("spacy.")]
    for key in stale:
        del sys.modules[key]


def _import_spacy():
    """Import spaCy, preferring .venv2 site-packages.

    Returns the spacy module on success, None on failure.
    """
    global _SPACY_SITE

    injected = False
    if _SPACY_SITE and _SPACY_SITE not in sys.path:
        sys.path.insert(0, _SPACY_SITE)
        injected = True

    _evict_spacy_from_sys_modules()

    try:
        import spacy  # noqa: PLC0415
        if not callable(getattr(spacy, "load", None)):
            raise ImportError("spacy.load not callable — incompatible build")
        return spacy
    except (ImportError, Exception):
        if injected and _SPACY_SITE in sys.path:
            sys.path.remove(_SPACY_SITE)
        _SPACY_SITE = None
        return None


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class TokenInfo:
    """Per-token NLP metadata — mirrors AI-Player's OpenNLPProcessor.TokenInfo.

    token   — surface form
    lemma   — base form (e.g. "running" → "run")
    pos     — coarse POS tag (NOUN, VERB, ADJ, ADV, AUX, PRON, …)
    tag     — fine-grained Penn Treebank tag (NNS, VBZ, JJR, …)
    dep     — dependency relation to head (nsubj, dobj, ROOT, advmod, …)
    head    — surface form of the syntactic head
    is_stop — True if spaCy marks token as a stop word
    """
    token: str
    lemma: str
    pos: str
    tag: str
    dep: str
    head: str
    is_stop: bool

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.token}({self.pos}/{self.dep})→{self.lemma}"


@dataclass
class SentenceAnalysis:
    """Full structural analysis of a single input text.

    Produced by NLPPreprocessor.analyse().  All three Layer 0 components
    consume this struct — they never call spaCy directly.

    sentence_type
        DECLARATIVE  — assertive statement ("The sky is blue.")
        INTERROGATIVE — question (starts with WH-word or auxiliary inversion)
        IMPERATIVE   — bare-verb command ("Mine some stone.")
        EXCLAMATORY  — exclamation ("What a result!")
        UNKNOWN      — could not be determined

    dep_triples
        List of (subject_lemma, root_verb_lemma, object_lemma) extracted from
        the dependency parse.  Equivalent to SRLTriple lightweight output.

    evaluative_words
        Adjectives/adverbs with opinion polarity (good, best, worst, terrible,
        obviously, fortunately …).

    presupposition_triggers
        Strings describing structural presupposition hooks found in the text,
        e.g. "factive_verb:know", "definite_superlative", "change_of_state:stop".

    coercive_signals
        Strings flagging coercion-pattern matches, e.g.
        "absolutist_quantifier:always", "forced_agreement:everyone_knows",
        "imperative_urgency:VB+immediately".

    tokens
        Full flat list of TokenInfo objects (all sentences concatenated).

    spacy_available
        False when spaCy could not be loaded — signals downstream that only
        lexical/regex layers fired and Layer D (LLM) must run unconditionally.
    """
    sentence_type: str = "UNKNOWN"
    dep_triples: List[Tuple[str, str, str]] = field(default_factory=list)
    evaluative_words: List[str] = field(default_factory=list)
    presupposition_triggers: List[str] = field(default_factory=list)
    coercive_signals: List[str] = field(default_factory=list)
    tokens: List[TokenInfo] = field(default_factory=list)
    spacy_available: bool = False
    raw_text: str = ""


# ---------------------------------------------------------------------------
# NLPPreprocessor
# ---------------------------------------------------------------------------

# Evaluative adjectives / adverbs with opinion load.
_EVALUATIVE_ADJ = frozenset([
    "good", "bad", "best", "worst", "better", "worse", "great", "terrible",
    "excellent", "poor", "superior", "inferior", "correct", "incorrect",
    "right", "wrong", "proper", "improper", "ideal", "perfect", "awful",
    "wonderful", "horrible", "fantastic", "dreadful", "magnificent",
])
_EVALUATIVE_ADV = frozenset([
    "obviously", "clearly", "certainly", "unfortunately", "fortunately",
    "sadly", "happily", "evidently", "undoubtedly", "surely", "definitely",
    "absolutely", "naturally", "needlessly", "unfairly",
])

# Factive / change-of-state verbs that presuppose their complement.
_FACTIVE_VERBS = frozenset([
    "know", "realize", "discover", "notice", "find", "see", "understand",
    "remember", "forget", "regret", "admit", "deny", "prove", "confirm",
])
_CHANGE_OF_STATE_VERBS = frozenset([
    "start", "stop", "begin", "cease", "continue", "resume", "finish",
    "end", "quit", "avoid", "fail",
])
_CAUSAL_VERBS = frozenset([
    "cause", "lead", "result", "produce", "trigger", "generate", "induce",
    "create", "cause",
])

# Absolutist / overgeneralisation quantifiers → coercion signal.
_ABSOLUTIST = frozenset(["always", "never", "everyone", "nobody", "no one",
                          "all", "none", "every", "any"])

# Hedge-removal / forced-agreement phrases → coercion signal.
_FORCED_AGREEMENT = [
    r"don'?t you agree",
    r"everyone knows",
    r"it'?s obvious",
    r"as we all know",
    r"you must agree",
    r"surely you",
    r"clearly you",
]
_FORCED_AGREEMENT_RE = re.compile(
    "|".join(_FORCED_AGREEMENT), re.IGNORECASE
)

# WH-interrogative starters.
_WH_WORDS = frozenset(["what", "why", "who", "whom", "which", "whose",
                        "when", "where", "how"])
# Auxiliary inversion starters.
_AUX_STARTERS = frozenset(["is", "are", "was", "were", "do", "does", "did",
                             "can", "could", "will", "would", "shall",
                             "should", "may", "might", "must", "have",
                             "has", "had"])


class NLPPreprocessor:
    """Shared NLP preprocessing bridge for all Layer 0 components.

    Wraps spaCy (loaded from .venv2 / lexis venv) and extracts the
    structural signals that ManipulationDetector, ObjectivityClassifier
    and ValueAssumptionExtractor need.  This is the direct Python
    equivalent of OpenNLPProcessor.java + the LIDSNet feature builder.

    Usage::

        pre = NLPPreprocessor()
        analysis = pre.analyse("Why does everyone know the vaccine is dangerous?")
        # analysis.sentence_type           → "INTERROGATIVE"
        # analysis.presupposition_triggers → ["factive_verb:know"]
        # analysis.coercive_signals        → ["absolutist_quantifier:everyone"]

    The analyse() result is LRU-cached (256 entries) so the same text is
    never re-parsed within one pipeline pass — same optimisation as
    SRLExtractor._get_doc().
    """

    def __init__(self) -> None:
        self._nlp = None   # spaCy Language, lazy-loaded; False if unavailable

    # ------------------------------------------------------------------
    # spaCy loading
    # ------------------------------------------------------------------

    def _load_spacy(self):
        """Lazy-load spaCy from .venv2.  Sets self._nlp to model or False."""
        if self._nlp is None:
            spacy = _import_spacy()
            if spacy is None:
                self._nlp = False
            else:
                try:
                    self._nlp = spacy.load("en_core_web_sm")
                except OSError:
                    # Model not installed — blank pipeline gives tokens but
                    # no dep parse; structural layers will degrade to lexical.
                    try:
                        self._nlp = spacy.blank("en")
                    except Exception:
                        self._nlp = False
        return self._nlp

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @lru_cache(maxsize=256)
    def analyse(self, text: str) -> SentenceAnalysis:
        """Run the full NLP pre-analysis pipeline on *text*.

        Returns a SentenceAnalysis.  Results are cached (LRU-256) so
        identical queries are never re-parsed.

        When spaCy is unavailable (self._nlp is False after load attempt)
        the structural layers still run in lexical/regex-only mode and
        SentenceAnalysis.spacy_available is set to False — signalling
        downstream components that Layer D (LLM) must fire unconditionally.
        """
        nlp = self._load_spacy()
        if not nlp:
            return self._analyse_lexical_only(text)
        return self._analyse_with_spacy(nlp, text)

    # ------------------------------------------------------------------
    # spaCy path
    # ------------------------------------------------------------------

    def _analyse_with_spacy(self, nlp, text: str) -> SentenceAnalysis:
        doc = nlp(text)
        spacy_ok = doc.has_annotation("DEP")

        tokens: List[TokenInfo] = [
            TokenInfo(
                token=t.text,
                lemma=t.lemma_.lower(),
                pos=t.pos_,
                tag=t.tag_,
                dep=t.dep_,
                head=t.head.text,
                is_stop=t.is_stop,
            )
            for t in doc
            if not t.is_space
        ]

        sentence_type = self._detect_sentence_type(doc, tokens)
        dep_triples = self._extract_dep_triples(doc) if spacy_ok else []
        evaluative_words = self._find_evaluative_words(tokens)
        presupposition_triggers = self._find_presupposition_triggers(doc, tokens, spacy_ok)
        coercive_signals = self._find_coercive_signals(doc, tokens, text, spacy_ok)

        return SentenceAnalysis(
            sentence_type=sentence_type,
            dep_triples=dep_triples,
            evaluative_words=evaluative_words,
            presupposition_triggers=presupposition_triggers,
            coercive_signals=coercive_signals,
            tokens=tokens,
            spacy_available=True,
            raw_text=text,
        )

    # ------------------------------------------------------------------
    # Lexical-only fallback (no spaCy)
    # ------------------------------------------------------------------

    def _analyse_lexical_only(self, text: str) -> SentenceAnalysis:
        """Pure regex / keyword analysis when spaCy is unavailable."""
        lower = text.lower().strip()
        words = re.findall(r"[a-z']+", lower)

        sentence_type = self._detect_sentence_type_lexical(lower, words)
        evaluative_words = [w for w in words
                            if w in _EVALUATIVE_ADJ or w in _EVALUATIVE_ADV]
        presupposition_triggers = self._find_presupposition_triggers_lexical(words)
        coercive_signals = self._find_coercive_signals_lexical(lower, words)

        return SentenceAnalysis(
            sentence_type=sentence_type,
            dep_triples=[],
            evaluative_words=evaluative_words,
            presupposition_triggers=presupposition_triggers,
            coercive_signals=coercive_signals,
            tokens=[],
            spacy_available=False,
            raw_text=text,
        )

    # ------------------------------------------------------------------
    # Sentence-type detection
    # ------------------------------------------------------------------

    def _detect_sentence_type(
        self, doc, tokens: List[TokenInfo]
    ) -> str:
        """Classify sentence type from dep-parse + first-token heuristics."""
        if not tokens:
            return "UNKNOWN"

        first_lemma = tokens[0].lemma
        first_pos = tokens[0].pos
        first_tag = tokens[0].tag

        # Interrogative: starts with WH-word OR auxiliary inversion
        if first_lemma in _WH_WORDS:
            return "INTERROGATIVE"
        if first_pos in {"AUX", "VERB"} and first_lemma in _AUX_STARTERS:
            return "INTERROGATIVE"

        # Imperative: ROOT is bare VB (base form verb, no subject)
        for tok in tokens:
            if tok.dep == "ROOT" and tok.tag == "VB":
                # Check no nsubj child exists
                subjects = [t for t in tokens if t.dep == "nsubj" and t.head == tok.token]
                if not subjects:
                    return "IMPERATIVE"

        # Exclamatory: leading exclamative word or trailing !
        excl_words = {"what", "how"}
        if first_lemma in excl_words and "!" in doc.text:
            return "EXCLAMATORY"

        return "DECLARATIVE"

    def _detect_sentence_type_lexical(self, lower: str, words: List[str]) -> str:
        """Lexical-only sentence-type detection (no dep parse)."""
        if not words:
            return "UNKNOWN"
        if words[0] in _WH_WORDS:
            return "INTERROGATIVE"
        if words[0] in _AUX_STARTERS and lower.strip().endswith("?"):
            return "INTERROGATIVE"
        if lower.strip().endswith("!"):
            return "EXCLAMATORY"
        # Bare-verb imperative heuristic: starts with a known action verb
        _COMMON_IMPERATIVES = frozenset([
            "go", "do", "make", "take", "find", "get", "move", "run",
            "build", "create", "show", "tell", "give", "explain",
            "list", "compare", "describe", "define", "calculate",
        ])
        if words[0] in _COMMON_IMPERATIVES:
            return "IMPERATIVE"
        return "DECLARATIVE"

    # ------------------------------------------------------------------
    # Dep-triple extraction (subject, root-verb, object)
    # ------------------------------------------------------------------

    def _extract_dep_triples(
        self, doc
    ) -> List[Tuple[str, str, str]]:
        """Extract (subject_lemma, predicate_lemma, object_lemma) triples."""
        triples: List[Tuple[str, str, str]] = []
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
                        subj_lemma = subjects[0].lemma_.lower()
                        pred_lemma = token.lemma_.lower()
                        obj_lemma = objects_[0].lemma_.lower()
                        triples.append((subj_lemma, pred_lemma, obj_lemma))
        return triples

    # ------------------------------------------------------------------
    # Evaluative word extraction
    # ------------------------------------------------------------------

    def _find_evaluative_words(self, tokens: List[TokenInfo]) -> List[str]:
        result = []
        for t in tokens:
            if t.is_stop:
                continue
            if t.pos in {"ADJ", "JJR", "JJS"} and t.lemma in _EVALUATIVE_ADJ:
                result.append(t.token)
            elif t.pos == "ADV" and t.lemma in _EVALUATIVE_ADV:
                result.append(t.token)
            # Fine-grained tag: JJR=comparative, JJS=superlative
            elif t.tag in {"JJR", "JJS"}:
                result.append(t.token)
        return result

    # ------------------------------------------------------------------
    # Presupposition trigger extraction
    # ------------------------------------------------------------------

    def _find_presupposition_triggers(
        self, doc, tokens: List[TokenInfo], dep_ok: bool
    ) -> List[str]:
        triggers: List[str] = []
        lemmas = {t.lemma for t in tokens}

        # Factive verbs: presuppose truth of their complement
        for fv in _FACTIVE_VERBS & lemmas:
            triggers.append(f"factive_verb:{fv}")

        # Change-of-state verbs: presuppose the prior state
        for cv in _CHANGE_OF_STATE_VERBS & lemmas:
            triggers.append(f"change_of_state:{cv}")

        # Definite superlative NPs: "the best X" presupposes ranking exists
        if dep_ok:
            for token in doc:
                if token.tag_ in {"JJS"} and any(
                    t.lower_ == "the" for t in token.lefts
                ):
                    triggers.append("definite_superlative")
                    break
        else:
            # Lexical fallback
            lower = doc.text.lower() if hasattr(doc, "text") else " ".join(
                t.token for t in tokens
            )
            if re.search(r"\bthe\s+(best|worst|most|least)\b", lower):
                triggers.append("definite_superlative")

        # Cleft construction: "It is X that …" presupposes X is the focus
        text_lower = doc.text.lower() if hasattr(doc, "text") else ""
        if re.search(r"\bit\s+is\b.{1,40}\bthat\b", text_lower):
            triggers.append("cleft_construction")

        # Additive particle: "also" implies prior activity
        if "also" in {t.lemma for t in tokens}:
            triggers.append("additive_particle:also")

        # WH + causal verb: "Why does X cause Y?" presupposes X causes Y
        if tokens and tokens[0].lemma == "why":
            if _CAUSAL_VERBS & lemmas:
                triggers.append("why_causal_presupposition")

        return triggers

    def _find_presupposition_triggers_lexical(
        self, words: List[str]
    ) -> List[str]:
        """Lexical-only presupposition trigger detection."""
        triggers: List[str] = []
        word_set = set(words)
        for fv in _FACTIVE_VERBS & word_set:
            triggers.append(f"factive_verb:{fv}")
        for cv in _CHANGE_OF_STATE_VERBS & word_set:
            triggers.append(f"change_of_state:{cv}")
        text = " ".join(words)
        if re.search(r"\bthe\s+(best|worst|most|least)\b", text):
            triggers.append("definite_superlative")
        if re.search(r"\bit\s+is\b.{1,40}\bthat\b", text):
            triggers.append("cleft_construction")
        if "also" in word_set:
            triggers.append("additive_particle:also")
        if words and words[0] == "why" and (_CAUSAL_VERBS & word_set):
            triggers.append("why_causal_presupposition")
        return triggers

    # ------------------------------------------------------------------
    # Coercive signal extraction
    # ------------------------------------------------------------------

    def _find_coercive_signals(
        self, doc, tokens: List[TokenInfo], raw_text: str, dep_ok: bool
    ) -> List[str]:
        signals: List[str] = []
        lemmas = [t.lemma for t in tokens]
        lemma_set = set(lemmas)
        pos_seq = [(t.lemma, t.pos, t.tag, t.dep) for t in tokens]

        # Absolutist quantifiers
        for word in _ABSOLUTIST & lemma_set:
            signals.append(f"absolutist_quantifier:{word}")

        # Forced-agreement phrases (regex over raw text)
        if _FORCED_AGREEMENT_RE.search(raw_text):
            signals.append("forced_agreement_phrase")

        # Modal imperative: MODAL + you → coercive frame
        # e.g. "You must/should/have to agree"
        modal_lemmas = {"must", "should", "shall", "need", "have"}
        if modal_lemmas & lemma_set:
            # Check if second-person pronoun is nearby
            you_present = any(t.lemma in {"you", "your"} for t in tokens)
            if you_present:
                matching = modal_lemmas & lemma_set
                for m in matching:
                    signals.append(f"modal_imperative:{m}")

        # Urgency adverb after action verb: VB + immediately/now/quickly
        urgency_adverbs = {"immediately", "now", "quickly", "right away",
                           "at once", "instantly", "urgently"}
        for i, (lemma, pos, tag, dep) in enumerate(pos_seq):
            if pos == "VERB" and i + 1 < len(pos_seq):
                next_lemma = pos_seq[i + 1][0]
                if next_lemma in urgency_adverbs:
                    signals.append(f"imperative_urgency:{lemma}+{next_lemma}")

        # Passive agency-hiding: passive ROOT verb
        if dep_ok:
            for token in doc:
                if token.dep_ == "ROOT" and any(
                    c.dep_ == "auxpass" for c in token.children
                ):
                    signals.append("passive_agency_hiding")
                    break

        # Jailbreak structural markers
        jb_patterns = [
            r"ignore (all )?(previous |prior )?instructions",
            r"pretend you (are|have no)",
            r"you are now",
            r"DAN mode",
            r"developer mode",
            r"bypass (your )?(safety|filter|guardrail)",
            r"disregard (your )?(training|guidelines)",
            r"act as if you (have no|were)",
        ]
        jb_re = re.compile("|".join(jb_patterns), re.IGNORECASE)
        if jb_re.search(raw_text):
            signals.append("jailbreak_structural")

        return signals

    def _find_coercive_signals_lexical(
        self, lower: str, words: List[str]
    ) -> List[str]:
        """Lexical-only coercive signal detection."""
        signals: List[str] = []
        word_set = set(words)
        for w in _ABSOLUTIST & word_set:
            signals.append(f"absolutist_quantifier:{w}")
        if _FORCED_AGREEMENT_RE.search(lower):
            signals.append("forced_agreement_phrase")
        jb_patterns = [
            r"ignore.*instructions",
            r"pretend you",
            r"you are now",
            r"dan mode",
            r"developer mode",
            r"bypass.*safety",
            r"disregard.*training",
        ]
        jb_re = re.compile("|".join(jb_patterns), re.IGNORECASE)
        if jb_re.search(lower):
            signals.append("jailbreak_structural")
        modal_lemmas = {"must", "should", "shall"}
        if modal_lemmas & word_set and ("you" in word_set or "your" in word_set):
            for m in modal_lemmas & word_set:
                signals.append(f"modal_imperative:{m}")
        return signals


# ---------------------------------------------------------------------------
# Module-level singleton (shared across all Layer 0 components)
# ---------------------------------------------------------------------------

_PREPROCESSOR: Optional[NLPPreprocessor] = None


def get_preprocessor() -> NLPPreprocessor:
    """Return the module-level NLPPreprocessor singleton.

    Created on first call.  Thread-safety: GIL is sufficient for
    single-process inference; no locking needed.
    """
    global _PREPROCESSOR
    if _PREPROCESSOR is None:
        _PREPROCESSOR = NLPPreprocessor()
    return _PREPROCESSOR
