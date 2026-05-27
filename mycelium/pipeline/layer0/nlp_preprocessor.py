"""
Layer 0 — NLP Preprocessor.

Shared preprocessing component used by all three Layer 0 classifiers:
    ManipulationDetector, ObjectivityClassifier, ValueAssumptionExtractor.

All spaCy work is delegated to the pipeline-level spacy_bridge / spacy_worker
subprocess pair (which already handles .venv2 path resolution, config.toml
[lexis] section, and the Py3.14 → Py3.11 subprocess protocol).  No inline
venv patching is needed here.

Output: SentenceAnalysis — a flat dataclass consumed by the three classifiers.

Caching: LRU(256) on (text, model) so repeated calls within a conversation
turn are free.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Canonical pipeline-level bridge — already handles all .venv2 routing.
from mycelium.pipeline.spacy_bridge import pipeline as _spacy_pipeline


# ---------------------------------------------------------------------------
# TokenInfo
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TokenInfo:
    """Lightweight token descriptor built from spacy_worker pipeline output."""
    token:   str
    lemma:   str
    pos:     str   # coarse POS (spaCy .pos_)
    tag:     str   # fine-grained POS (spaCy .tag_)
    dep:     str   # dependency relation (spaCy .dep_)
    is_stop: bool

    @classmethod
    def from_worker(cls, pos_entry: dict, lemma: str) -> "TokenInfo":
        """Construct from one entry in the worker's 'pos' list + parallel lemma."""
        text = pos_entry["text"]
        return cls(
            token=text,
            lemma=lemma,
            pos=pos_entry.get("pos", ""),
            tag=pos_entry.get("tag", ""),
            dep=pos_entry.get("dep", ""),
            is_stop=text.lower() in _STOP_WORDS,
        )


# Minimal stop-word set (no NLTK dependency).
_STOP_WORDS = frozenset([
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "shall", "may", "might", "must", "can", "to", "of", "in",
    "on", "at", "by", "for", "with", "about", "as", "into", "through",
    "and", "but", "or", "nor", "so", "yet", "both", "either", "neither",
    "not", "no", "nor", "very", "just", "than", "then", "also", "i",
    "me", "my", "we", "our", "you", "your", "he", "she", "it", "they",
    "them", "this", "that", "these", "those", "what", "which", "who",
    "whom", "when", "where", "how", "if", "because", "while", "although",
])


# ---------------------------------------------------------------------------
# SentenceAnalysis
# ---------------------------------------------------------------------------

@dataclass
class SentenceAnalysis:
    """Flat structural analysis of a single sentence / query.

    Fields
    ------
    raw_text             : Original input string.
    tokens               : List[TokenInfo] — full token sequence.
    sentence_type        : DECLARATIVE | INTERROGATIVE | IMPERATIVE | UNKNOWN.
    dep_triples          : Subject-predicate-object triples extracted from the
                           dependency parse.  Each triple is (subj_lemma,
                           pred_lemma, obj_lemma).  May be empty.
    evaluative_words     : Adjectives / adverbs with opinion polarity.
    presupposition_triggers : Detected presupposition hooks (see extractor below).
    coercive_signals     : Detected coercion patterns (see extractor below).
    spacy_available      : False if the bridge call failed — callers should
                           demote confidence accordingly.
    """
    raw_text:                 str
    tokens:                   List[TokenInfo]       = field(default_factory=list)
    sentence_type:            str                   = "UNKNOWN"
    dep_triples:              List[Tuple[str,str,str]] = field(default_factory=list)
    evaluative_words:         List[str]             = field(default_factory=list)
    presupposition_triggers:  List[str]             = field(default_factory=list)
    coercive_signals:         List[str]             = field(default_factory=list)
    spacy_available:          bool                  = True


# ---------------------------------------------------------------------------
# Vocabulary sets (used by signal extractors)
# ---------------------------------------------------------------------------

_FACTIVE_VERBS = frozenset([
    "know", "realize", "discover", "notice", "see", "understand",
    "remember", "forget", "regret", "acknowledge", "recognise", "recognize",
])

_CHANGE_OF_STATE_VERBS = frozenset([
    "stop", "start", "begin", "cease", "continue", "resume", "finish",
    "end", "fail", "succeed", "become", "turn", "change",
])

_ABSOLUTIST_QUANTIFIERS = frozenset([
    "everyone", "everybody", "everything", "nobody", "nothing", "nowhere",
    "always", "never", "all", "none", "every",
])

_FORCED_AGREEMENT_PHRASES = [
    r"don't you (think|agree|believe)",
    r"everyone knows",
    r"it's (obvious|clear|plain) that",
    r"(surely|certainly|obviously) you",
    r"as (everyone|we all) know",
    r"you (must|have to|need to) agree",
    r"isn't it (obvious|clear)",
]
_FORCED_AGREEMENT_RE = re.compile(
    "|".join(_FORCED_AGREEMENT_PHRASES), re.IGNORECASE
)

_URGENCY_WORDS = frozenset([
    "immediately", "instantly", "urgently", "right now", "asap",
    "without delay", "at once",
])

_JAILBREAK_PHRASES = [
    r"ignore (all )?(previous|prior|above) instructions",
    r"disregard (your )?(previous|prior|all) (instructions|training|guidelines)",
    r"you are now (a |an )?(different|new|unrestricted|jailbroken)",
    r"forget (everything|all) (you (were )?trained|your (training|instructions))",
    r"pretend (you (are|have no)|there are no) (rules|restrictions|guidelines|training)",
    r"(act|behave) as (if you (have|had) no|though you (are|were) not)",
    r"(do not|don't) follow (your )?(training|guidelines|instructions|rules)",
    r"override (your )?(previous |all )?(instructions|programming|training)",
    r"new (system )?(prompt|instruction|directive):",
    r"system:",
    r"\[system\]",
    r"<\|im_start\|>system",
]
_JAILBREAK_RE = re.compile(
    "|".join(_JAILBREAK_PHRASES), re.IGNORECASE
)

_EVALUATIVE_ADJS = frozenset([
    "good", "bad", "great", "terrible", "wonderful", "awful", "excellent",
    "poor", "best", "worst", "better", "worse", "beautiful", "ugly",
    "smart", "stupid", "intelligent", "foolish", "right", "wrong",
    "correct", "incorrect", "fair", "unfair", "just", "unjust",
    "important", "useless", "valuable", "worthless", "harmful", "beneficial",
    "dangerous", "safe", "effective", "ineffective", "superior", "inferior",
    "strong", "weak", "powerful", "powerless",
])


# ---------------------------------------------------------------------------
# NLPPreprocessor
# ---------------------------------------------------------------------------

class NLPPreprocessor:
    """Analyses a text string and returns a SentenceAnalysis.

    Delegates all spaCy work to the pipeline-level spacy_bridge.pipeline()
    which in turn shells out to spacy_worker.py inside .venv2.
    If the bridge call fails for any reason, falls back to regex-only
    analysis with spacy_available=False so callers can degrade gracefully.
    """

    @functools.lru_cache(maxsize=256)
    def analyse(self, text: str, model: str = "en_core_web_sm") -> SentenceAnalysis:
        """Full structural analysis.  LRU-cached on (text, model)."""
        try:
            raw = _spacy_pipeline(text, model=model)
            return self._build_from_bridge(text, raw)
        except Exception:
            return self._build_fallback(text)

    # ------------------------------------------------------------------
    # Bridge-backed analysis path
    # ------------------------------------------------------------------

    def _build_from_bridge(
        self, text: str, raw: dict
    ) -> SentenceAnalysis:
        """Build SentenceAnalysis from spacy_bridge.pipeline() output dict."""
        pos_list  = raw.get("pos", [])      # [{text, pos, tag, dep}, ...]
        lemmas    = raw.get("lemmas", [])   # [str, ...]

        # Pad lemmas to match pos_list length (safety)
        while len(lemmas) < len(pos_list):
            lemmas.append("")

        tokens = [
            TokenInfo.from_worker(p, l)
            for p, l in zip(pos_list, lemmas)
        ]

        stype       = self._detect_sentence_type(tokens, text)
        triples     = self._extract_dep_triples(tokens)
        evaluative  = self._extract_evaluative(tokens)
        presup      = self._extract_presupposition_triggers(tokens, text, stype)
        coercive    = self._extract_coercive_signals(tokens, text)

        return SentenceAnalysis(
            raw_text=text,
            tokens=tokens,
            sentence_type=stype,
            dep_triples=triples,
            evaluative_words=evaluative,
            presupposition_triggers=presup,
            coercive_signals=coercive,
            spacy_available=True,
        )

    # ------------------------------------------------------------------
    # Regex-only fallback
    # ------------------------------------------------------------------

    def _build_fallback(self, text: str) -> SentenceAnalysis:
        """Minimal analysis without spaCy — regex signals only."""
        stype    = self._sentence_type_regex(text)
        coercive = self._coercive_signals_regex(text)
        presup   = self._presupposition_triggers_regex(text, stype)
        return SentenceAnalysis(
            raw_text=text,
            tokens=[],
            sentence_type=stype,
            dep_triples=[],
            evaluative_words=self._evaluative_regex(text),
            presupposition_triggers=presup,
            coercive_signals=coercive,
            spacy_available=False,
        )

    # ------------------------------------------------------------------
    # Sentence type detection
    # ------------------------------------------------------------------

    def _detect_sentence_type(
        self, tokens: List[TokenInfo], text: str
    ) -> str:
        if not tokens:
            return self._sentence_type_regex(text)
        stripped = text.strip()
        if stripped.endswith("?"):
            return "INTERROGATIVE"
        first = tokens[0]
        if first.dep == "ROOT" and first.pos == "VERB" and first.tag in {"VB", "VBP"}:
            return "IMPERATIVE"
        wh_words = {"what", "when", "where", "who", "whom", "which", "why", "how"}
        if tokens[0].lemma.lower() in wh_words:
            return "INTERROGATIVE"
        aux_inverted = (len(tokens) >= 2 and
                        tokens[0].pos == "AUX" and
                        tokens[1].pos in {"PRON", "NOUN", "PROPN"})
        if aux_inverted:
            return "INTERROGATIVE"
        return "DECLARATIVE"

    def _sentence_type_regex(self, text: str) -> str:
        stripped = text.strip()
        if stripped.endswith("?"):
            return "INTERROGATIVE"
        if re.match(
            r"^(what|when|where|who|whom|which|why|how|is|are|was|were|do|does|did|"
            r"can|could|would|should|shall|will|have|has|had)\b",
            stripped, re.IGNORECASE
        ):
            return "INTERROGATIVE"
        if re.match(r"^(please\s+)?[a-z]+(\s+\w+){0,3}\.?$", stripped, re.IGNORECASE):
            return "IMPERATIVE"
        return "DECLARATIVE"

    # ------------------------------------------------------------------
    # Dependency triple extraction
    # ------------------------------------------------------------------

    def _extract_dep_triples(
        self, tokens: List[TokenInfo]
    ) -> List[Tuple[str, str, str]]:
        """Extract (subject_lemma, predicate_lemma, object_lemma) triples."""
        triples: List[Tuple[str, str, str]] = []
        subj_deps = {"nsubj", "nsubjpass", "csubj"}
        obj_deps  = {"dobj", "obj", "pobj", "iobj", "attr"}

        # Build lemma lookup by position for head resolution
        lemma_by_idx = {i: t.lemma for i, t in enumerate(tokens)}

        # Find all ROOT / VERB tokens as predicate candidates
        for i, t in enumerate(tokens):
            if t.pos not in {"VERB", "AUX"} and t.dep != "ROOT":
                continue
            pred_lemma = t.lemma
            subj_lemma = ""
            obj_lemma  = ""
            for j, other in enumerate(tokens):
                if other.dep in subj_deps and not subj_lemma:
                    subj_lemma = other.lemma
                if other.dep in obj_deps and not obj_lemma:
                    obj_lemma = other.lemma
            if subj_lemma or obj_lemma:
                triples.append((subj_lemma, pred_lemma, obj_lemma))
                break  # one primary triple per sentence is enough

        return triples

    # ------------------------------------------------------------------
    # Evaluative word extraction
    # ------------------------------------------------------------------

    def _extract_evaluative(
        self, tokens: List[TokenInfo]
    ) -> List[str]:
        return [
            t.token for t in tokens
            if t.lemma.lower() in _EVALUATIVE_ADJS
            and t.pos in {"ADJ", "ADV"}
        ]

    def _evaluative_regex(self, text: str) -> List[str]:
        lower = text.lower()
        return [w for w in _EVALUATIVE_ADJS if re.search(rf"\b{re.escape(w)}\b", lower)]

    # ------------------------------------------------------------------
    # Presupposition trigger extraction
    # ------------------------------------------------------------------

    def _extract_presupposition_triggers(
        self, tokens: List[TokenInfo], text: str, stype: str
    ) -> List[str]:
        found: List[str] = []
        lemma_set = {t.lemma.lower() for t in tokens}

        # Factive verbs
        for v in _FACTIVE_VERBS & lemma_set:
            found.append(f"factive_verb:{v}")

        # Change-of-state verbs
        for v in _CHANGE_OF_STATE_VERBS & lemma_set:
            found.append(f"change_of_state:{v}")

        # Definite superlative: the + JJS
        pos_tags = [t.tag for t in tokens]
        for i, t in enumerate(tokens):
            if t.tag == "JJS" and i > 0 and tokens[i - 1].lemma.lower() == "the":
                found.append("definite_superlative")
                break

        # Cleft construction: "It is/was X that"
        if re.search(r"\bit (is|was|were)\b.{1,40}\bthat\b", text, re.IGNORECASE):
            found.append("cleft_construction")

        # Additive 'also'
        if any(t.lemma.lower() == "also" for t in tokens):
            found.append("additive_particle:also")

        # Why + causal verb
        if stype == "INTERROGATIVE":
            lower = text.lower()
            if lower.startswith("why") or "why does" in lower or "why is" in lower:
                causal_verbs = {"cause", "lead", "result", "make", "force",
                                "prevent", "allow", "enable", "trigger"}
                if causal_verbs & lemma_set:
                    found.append("why_causal_presupposition")

        return found

    def _presupposition_triggers_regex(self, text: str, stype: str) -> List[str]:
        """Regex-only presupposition detection (fallback)."""
        found: List[str] = []
        lower = text.lower()
        words = set(re.findall(r"\b\w+\b", lower))

        for v in _FACTIVE_VERBS & words:
            found.append(f"factive_verb:{v}")
        for v in _CHANGE_OF_STATE_VERBS & words:
            found.append(f"change_of_state:{v}")
        if re.search(r"\bthe (best|worst|most|least|biggest|smallest)\b", lower):
            found.append("definite_superlative")
        if re.search(r"\bit (is|was)\b.{1,40}\bthat\b", lower):
            found.append("cleft_construction")
        if " also " in lower or lower.startswith("also "):
            found.append("additive_particle:also")
        if lower.startswith("why") and any(
            v in words for v in {"cause", "lead", "make", "force", "allow"}
        ):
            found.append("why_causal_presupposition")
        return found

    # ------------------------------------------------------------------
    # Coercive signal extraction
    # ------------------------------------------------------------------

    def _extract_coercive_signals(
        self, tokens: List[TokenInfo], text: str
    ) -> List[str]:
        found: List[str] = []
        lemma_set = {t.lemma.lower() for t in tokens}

        # Jailbreak structural patterns (regex — most reliable)
        if _JAILBREAK_RE.search(text):
            found.append("jailbreak_structural")
            return found  # No need to score further

        # Forced agreement phrases
        if _FORCED_AGREEMENT_RE.search(text):
            found.append("forced_agreement_phrase")

        # Modal imperatives: must/need/have to + agree/accept/comply
        for t in tokens:
            if t.pos == "AUX" and t.lemma.lower() in {"must", "need", "have"}:
                found.append(f"modal_imperative:{t.lemma.lower()}")
                break

        # Absolutist quantifiers
        for word in _ABSOLUTIST_QUANTIFIERS & lemma_set:
            found.append(f"absolutist_quantifier:{word}")

        # Urgency signals
        lower = text.lower()
        for w in _URGENCY_WORDS:
            if w in lower:
                found.append(f"imperative_urgency:{w}")

        # Passive agency hiding: auxiliary 'be' + past participle without agent
        passive_count = sum(
            1 for t in tokens if t.dep == "auxpass"
        )
        if passive_count >= 2:
            found.append("passive_agency_hiding")

        return found

    def _coercive_signals_regex(self, text: str) -> List[str]:
        """Regex-only coercive signal detection (fallback)."""
        found: List[str] = []
        if _JAILBREAK_RE.search(text):
            found.append("jailbreak_structural")
            return found
        if _FORCED_AGREEMENT_RE.search(text):
            found.append("forced_agreement_phrase")
        lower = text.lower()
        words = set(re.findall(r"\b\w+\b", lower))
        for word in _ABSOLUTIST_QUANTIFIERS & words:
            found.append(f"absolutist_quantifier:{word}")
        for w in _URGENCY_WORDS:
            if w in lower:
                found.append(f"imperative_urgency:{w}")
        if re.search(r"\b(must|need to|have to)\b", lower):
            found.append("modal_imperative:must")
        return found


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_preprocessor_instance: Optional[NLPPreprocessor] = None


def get_preprocessor() -> NLPPreprocessor:
    """Return the module-level NLPPreprocessor singleton."""
    global _preprocessor_instance
    if _preprocessor_instance is None:
        _preprocessor_instance = NLPPreprocessor()
    return _preprocessor_instance
