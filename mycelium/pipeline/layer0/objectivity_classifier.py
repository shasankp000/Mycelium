"""
Layer 0 — Objectivity Classifier.

Four-layer classification stack:

    Layer A — Sentence-type gate (mirrors AI-Player's grammar rules in
               buildPrompt: DECLARATIVE/INTERROGATIVE/IMPERATIVE).
    Layer B — Evaluative vs. descriptive word detection (POS: JJR/JJS,
               ADV-opinion, modal verbs).
    Layer C — Predicate type check (stative vs. action vs. metalinguistic
               root verb from dep-parse triples).
    Layer D — LLM arbiter: fires when A–C score is ambiguous or spaCy
               was unavailable (same gate logic as ManipulationDetector).

Output labels:
    OBJECTIVE    — factual, answerable with evidence.
    SUBJECTIVE   — opinion / preference, no single correct answer.
    VALUE_LADEN  — normative, ethical, or political framing.
    AMBIGUOUS    — cannot be confidently classified.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

from mycelium.pipeline.layer0.nlp_preprocessor import (
    SentenceAnalysis,
    get_preprocessor,
)

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

QuestionType = Literal["OBJECTIVE", "SUBJECTIVE", "VALUE_LADEN", "AMBIGUOUS"]
_VALID_LABELS: List[str] = ["OBJECTIVE", "SUBJECTIVE", "VALUE_LADEN", "AMBIGUOUS"]


# ---------------------------------------------------------------------------
# Vocabulary sets
# ---------------------------------------------------------------------------

# Stative verbs (be, seem, feel) as main predicate → SUBJECTIVE signal
_STATIVE_VERBS = frozenset([
    "be", "seem", "appear", "feel", "believe", "think", "suppose",
    "assume", "consider", "find", "look", "sound", "taste", "smell",
])

# Metalinguistic verbs as main predicate → VALUE_LADEN signal
_METALINGUISTIC_VERBS = frozenset([
    "argue", "claim", "suggest", "assert", "contend", "posit", "maintain",
    "allege", "imply", "insinuate", "propose", "recommend", "advocate",
])

# Modal verbs → normative / VALUE_LADEN frame
_NORMATIVE_MODALS = frozenset(["should", "ought", "must", "shall"])

# WH-words that strongly signal OBJECTIVE information requests
_FACTUAL_WH = frozenset(["when", "where", "who", "what", "how"])

# WH-words that lean VALUE_LADEN when combined with evaluative words
_VALUE_WH = frozenset(["why"])

# Comparative / superlative POS tags
_COMP_SUPER_TAGS = frozenset(["JJR", "JJS", "RBR", "RBS"])

# Threshold band for LLM gate (score range → ambiguous)
_LOW_THRESH = 0.20
_HIGH_THRESH = 0.65


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ObjectivityResult:
    """Result returned by ObjectivityClassifier.classify().

    question_type — final label.
    confidence    — 0–1 confidence of the assigned label.
    rule_score    — raw score from Layers A–C.
    llm_used      — True if Layer D fired.
    signals       — list of signals that contributed to the decision.
    """
    question_type: QuestionType
    confidence: float
    rule_score: float = 0.0
    llm_used: bool = False
    signals: List[str] = None  # type: ignore

    def __post_init__(self):
        if self.signals is None:
            self.signals = []


# ---------------------------------------------------------------------------
# ObjectivityClassifier
# ---------------------------------------------------------------------------

class ObjectivityClassifier:
    """Classify whether a question is objective, subjective, value-laden, or ambiguous.

    Usage::

        clf = ObjectivityClassifier()
        result = clf.classify("What is the capital of France?")
        # result.question_type → "OBJECTIVE"

        result = clf.classify("Should we prioritise freedom over equality?")
        # result.question_type → "VALUE_LADEN"
    """

    def __init__(self) -> None:
        self._pre = get_preprocessor()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def classify(self, question: str) -> ObjectivityResult:
        """Run the 4-layer objectivity classification pipeline."""
        text = (question or "").strip()
        if not text:
            return ObjectivityResult(
                question_type="AMBIGUOUS", confidence=0.6, signals=[]
            )

        analysis: SentenceAnalysis = self._pre.analyse(text)

        # Layers A + B + C — rule scoring
        label, score, signals = self._rule_classify(analysis, text)

        # Layer D gate
        if self._needs_llm(score, analysis.spacy_available, label):
            return self._layer_d(text, analysis, score, signals, label)

        confidence = self._score_to_confidence(score, label)
        return ObjectivityResult(
            question_type=label,
            confidence=confidence,
            rule_score=score,
            llm_used=False,
            signals=signals,
        )

    # ------------------------------------------------------------------
    # Layers A + B + C
    # ------------------------------------------------------------------

    def _rule_classify(
        self, analysis: SentenceAnalysis, raw_text: str
    ) -> Tuple[QuestionType, float, List[str]]:
        """Score the three structural layers and return a tentative label."""
        score_obj = 0.0
        score_subj = 0.0
        score_value = 0.0
        signals: List[str] = []
        tokens = analysis.tokens
        lemma_set = {t.lemma for t in tokens}

        # ---- Layer A: Sentence type gate ----
        stype = analysis.sentence_type
        if stype == "INTERROGATIVE":
            first_lemma = tokens[0].lemma if tokens else ""
            if first_lemma in _FACTUAL_WH:
                score_obj += 0.35
                signals.append(f"wh_factual:{first_lemma}")
            elif first_lemma in _VALUE_WH:
                score_value += 0.30
                signals.append(f"wh_value:{first_lemma}")
        elif stype == "IMPERATIVE":
            # Direct commands are usually OBJECTIVE task requests
            score_obj += 0.40
            signals.append("imperative_task_request")
        elif stype == "DECLARATIVE":
            # Declarative by default leans SUBJECTIVE (assertion without query)
            score_subj += 0.10
            signals.append("declarative_assertion")

        # ---- Layer B: Evaluative word detection ----
        if analysis.evaluative_words:
            n = len(analysis.evaluative_words)
            score_subj += 0.20 * min(n, 3)   # up to 0.60
            signals.append(f"evaluative_words:{','.join(analysis.evaluative_words[:3])}")

        # Comparative / superlative fine-grained tags
        comp_super = [t for t in tokens if t.tag in _COMP_SUPER_TAGS]
        if comp_super:
            score_subj += 0.20
            signals.append(f"comparative_superlative:{comp_super[0].token}")

        # Normative modal verbs
        norm_modals = _NORMATIVE_MODALS & lemma_set
        if norm_modals:
            score_value += 0.35 * min(len(norm_modals), 2)
            signals.append(f"normative_modal:{','.join(norm_modals)}")

        # ---- Layer C: Predicate type check ----
        if analysis.dep_triples:
            root_pred = analysis.dep_triples[0][1]  # (subj, pred, obj)
            if root_pred in _STATIVE_VERBS:
                score_subj += 0.25
                signals.append(f"stative_predicate:{root_pred}")
            elif root_pred in _METALINGUISTIC_VERBS:
                score_value += 0.30
                signals.append(f"metalinguistic_predicate:{root_pred}")
            else:
                score_obj += 0.15
                signals.append(f"action_predicate:{root_pred}")
        elif analysis.spacy_available:
            # Dep parse available but no SVO — likely copular or complex
            score_subj += 0.10
            signals.append("no_svo_triple")

        # Presupposition triggers bump VALUE_LADEN
        if analysis.presupposition_triggers:
            score_value += 0.15 * min(len(analysis.presupposition_triggers), 2)
            signals.append(
                f"presupposition:{','.join(analysis.presupposition_triggers[:2])}"
            )

        # ---- Decide label from highest score ----
        total = score_obj + score_subj + score_value
        if total == 0:
            return "AMBIGUOUS", 0.0, signals

        norm_obj = score_obj / total
        norm_subj = score_subj / total
        norm_value = score_value / total

        max_score = max(norm_obj, norm_subj, norm_value)
        raw_score = max_score  # "confidence" signal for Layer D gate

        if norm_value == max_score:
            return "VALUE_LADEN", raw_score, signals
        if norm_subj == max_score:
            return "SUBJECTIVE", raw_score, signals
        return "OBJECTIVE", raw_score, signals

    def _score_to_confidence(self, score: float, label: str) -> float:
        """Map normalised score to a confidence value."""
        # Add a small boost when spaCy structural parse was used
        return round(min(1.0, score + 0.05), 3)

    # ------------------------------------------------------------------
    # Layer D — LLM arbiter
    # ------------------------------------------------------------------

    def _needs_llm(
        self, score: float, spacy_available: bool, tentative_label: str
    ) -> bool:
        """True when Layer D should fire."""
        if not spacy_available:
            return True
        # Ambiguous band or label already AMBIGUOUS
        if tentative_label == "AMBIGUOUS":
            return True
        return _LOW_THRESH <= score <= _HIGH_THRESH

    def _layer_d(
        self,
        text: str,
        analysis: SentenceAnalysis,
        rule_score: float,
        signals: List[str],
        fallback_label: QuestionType,
    ) -> ObjectivityResult:
        prompt = self._build_llm_prompt(text, analysis, signals)
        response = self._call_llm(prompt)

        if response is None:
            return ObjectivityResult(
                question_type=fallback_label,
                confidence=0.5,
                rule_score=rule_score,
                llm_used=False,
                signals=signals,
            )

        label, confidence = self._parse_llm_response(response)
        return ObjectivityResult(
            question_type=label,
            confidence=confidence,
            rule_score=rule_score,
            llm_used=True,
            signals=signals + [f"llm:{label}"],
        )

    def _build_llm_prompt(
        self, text: str, analysis: SentenceAnalysis, signals: List[str]
    ) -> str:
        triples = [
            f"({s}, {p}, {o})" for s, p, o in analysis.dep_triples
        ] or ["none"]
        evaluative = analysis.evaluative_words or ["none"]
        presuppositions = analysis.presupposition_triggers or ["none"]

        return (
            "You are a linguistic objectivity classification function.\n"
            "The following structural signals were extracted from the query:\n"
            f"  Sentence type          : {analysis.sentence_type}\n"
            f"  Dep-parse triples      : {', '.join(triples)}\n"
            f"  Evaluative words found : {', '.join(evaluative)}\n"
            f"  Presupposition hooks   : {', '.join(presuppositions)}\n"
            f"  Rule signals           : {', '.join(signals) or 'none'}\n"
            "\n"
            f'Raw query: "{text}"\n'
            "\n"
            "Based on these structural signals and the raw query, classify as:\n"
            f"  {' | '.join(_VALID_LABELS)}\n"
            "  OBJECTIVE   = answerable with factual evidence\n"
            "  SUBJECTIVE  = opinion / preference, no single right answer\n"
            "  VALUE_LADEN = involves normative, ethical or political judgment\n"
            "  AMBIGUOUS   = cannot be confidently classified\n"
            "Also provide a confidence score from 0.0 to 1.0.\n"
            "Respond ONLY in this exact format:\n"
            "LABEL: <label>\n"
            "CONFIDENCE: <0.0-1.0>\n"
        )

    def _call_llm(self, prompt: str) -> Optional[str]:
        try:
            from mycelium.core.llm_interface import call_llm  # type: ignore
            return call_llm(
                system="Classify as directed. Return ONLY the two-line format.",
                user=prompt,
            )
        except Exception:
            pass
        try:
            import requests  # type: ignore
            model = os.getenv("MYCELIUM_LLM_MODEL", "llama3")
            resp = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system",
                         "content": "Classify as directed. Return ONLY the two-line format."},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                },
                timeout=120,
            )
            if resp.ok:
                return resp.json().get("message", {}).get("content", "")
        except Exception:
            pass
        return None

    def _parse_llm_response(
        self, response: str
    ) -> Tuple[QuestionType, float]:
        label: QuestionType = "AMBIGUOUS"
        confidence = 0.5

        response = re.sub(r"(?s)<think>.*?</think>", "", response).strip()

        for line in response.splitlines():
            line = line.strip()
            if line.upper().startswith("LABEL:"):
                raw = line.split(":", 1)[1].strip().upper()
                if raw in _VALID_LABELS:
                    label = raw  # type: ignore
            elif line.upper().startswith("CONFIDENCE:"):
                try:
                    confidence = max(0.0, min(1.0,
                        float(line.split(":", 1)[1].strip())
                    ))
                except ValueError:
                    pass

        if label == "AMBIGUOUS":
            for valid in _VALID_LABELS:
                if valid in response.upper():
                    label = valid  # type: ignore
                    break

        return label, confidence
