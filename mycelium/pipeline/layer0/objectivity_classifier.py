"""
Layer 0 — Objectivity Classifier.

Four-layer classification stack:
    Layer A — Sentence-type gate.
    Layer B — Evaluative word / POS / modal detection.
    Layer C — Predicate type check from dep-parse triples.
    Layer D — sklearn classifier (if trained) else LLM arbiter.
               Every LLM call is logged to dataset_logger.
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

QuestionType = Literal["OBJECTIVE", "SUBJECTIVE", "VALUE_LADEN", "AMBIGUOUS"]
_VALID_LABELS: List[str] = ["OBJECTIVE", "SUBJECTIVE", "VALUE_LADEN", "AMBIGUOUS"]

_STATIVE_VERBS = frozenset([
    "be", "seem", "appear", "feel", "believe", "think", "suppose",
    "assume", "consider", "find", "look", "sound", "taste", "smell",
])
_METALINGUISTIC_VERBS = frozenset([
    "argue", "claim", "suggest", "assert", "contend", "posit", "maintain",
    "allege", "imply", "insinuate", "propose", "recommend", "advocate",
])
_NORMATIVE_MODALS  = frozenset(["should", "ought", "must", "shall"])
_FACTUAL_WH        = frozenset(["when", "where", "who", "what", "how"])
_VALUE_WH          = frozenset(["why"])
_COMP_SUPER_TAGS   = frozenset(["JJR", "JJS", "RBR", "RBS"])
_LOW_THRESH  = 0.20
_HIGH_THRESH = 0.65


@dataclass
class ObjectivityResult:
    question_type: QuestionType
    confidence: float
    rule_score: float = 0.0
    llm_used: bool = False
    signals: List[str] = None  # type: ignore
    def __post_init__(self):
        if self.signals is None:
            self.signals = []


class ObjectivityClassifier:
    def __init__(self) -> None:
        self._pre = get_preprocessor()

    def classify(self, question: str) -> ObjectivityResult:
        text = (question or "").strip()
        if not text:
            return ObjectivityResult(question_type="AMBIGUOUS", confidence=0.6, signals=[])
        analysis: SentenceAnalysis = self._pre.analyse(text)
        label, score, signals = self._rule_classify(analysis, text)

        # --- sklearn model first ---
        clf_result = self._try_classifier(text, analysis)
        if clf_result is not None:
            clf_label, clf_confidence = clf_result
            return ObjectivityResult(
                question_type=clf_label,
                confidence=clf_confidence,
                rule_score=score,
                llm_used=False,
                signals=signals,
            )

        if self._needs_llm(score, analysis.spacy_available, label):
            return self._layer_d(text, analysis, score, signals, label)
        confidence = round(min(1.0, score + 0.05), 3)
        return ObjectivityResult(
            question_type=label, confidence=confidence,
            rule_score=score, llm_used=False, signals=signals,
        )

    def _try_classifier(
        self, text: str, analysis: SentenceAnalysis
    ) -> Optional[Tuple[QuestionType, float]]:
        try:
            from mycelium.pipeline.model_registry import get_layer0_classifier
            from mycelium.pipeline.layer0.train_layer0_models import _rule_signal_vector
            artifact = get_layer0_classifier("objectivity_classifier")
            if artifact is None:
                return None
            from sentence_transformers import SentenceTransformer  # type: ignore
            import numpy as np
            encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            emb = encoder.encode([text], convert_to_numpy=True)
            rule_vec = _rule_signal_vector(text).reshape(1, -1)
            X = np.hstack([emb, rule_vec])
            clf = artifact["model"]
            le  = artifact["label_encoder"]
            proba = clf.predict_proba(X)[0]
            idx = int(proba.argmax())
            label: QuestionType = le.inverse_transform([idx])[0]  # type: ignore
            return label, float(proba[idx])
        except Exception:
            return None

    def _rule_classify(
        self, analysis: SentenceAnalysis, raw_text: str
    ) -> Tuple[QuestionType, float, List[str]]:
        score_obj = score_subj = score_value = 0.0
        signals: List[str] = []
        tokens = analysis.tokens
        lemma_set = {t.lemma for t in tokens}
        stype = analysis.sentence_type
        if stype == "INTERROGATIVE":
            first_lemma = tokens[0].lemma if tokens else ""
            if first_lemma in _FACTUAL_WH:
                score_obj += 0.35; signals.append(f"wh_factual:{first_lemma}")
            elif first_lemma in _VALUE_WH:
                score_value += 0.30; signals.append(f"wh_value:{first_lemma}")
        elif stype == "IMPERATIVE":
            score_obj += 0.40; signals.append("imperative_task_request")
        elif stype == "DECLARATIVE":
            score_subj += 0.10; signals.append("declarative_assertion")
        if analysis.evaluative_words:
            n = len(analysis.evaluative_words)
            score_subj += 0.20 * min(n, 3)
            signals.append(f"evaluative_words:{','.join(analysis.evaluative_words[:3])}")
        comp_super = [t for t in tokens if t.tag in _COMP_SUPER_TAGS]
        if comp_super:
            score_subj += 0.20; signals.append(f"comparative_superlative:{comp_super[0].token}")
        norm_modals = _NORMATIVE_MODALS & lemma_set
        if norm_modals:
            score_value += 0.35 * min(len(norm_modals), 2)
            signals.append(f"normative_modal:{','.join(norm_modals)}")
        if analysis.dep_triples:
            root_pred = analysis.dep_triples[0][1]
            if root_pred in _STATIVE_VERBS:
                score_subj += 0.25; signals.append(f"stative_predicate:{root_pred}")
            elif root_pred in _METALINGUISTIC_VERBS:
                score_value += 0.30; signals.append(f"metalinguistic_predicate:{root_pred}")
            else:
                score_obj += 0.15; signals.append(f"action_predicate:{root_pred}")
        elif analysis.spacy_available:
            score_subj += 0.10; signals.append("no_svo_triple")
        if analysis.presupposition_triggers:
            score_value += 0.15 * min(len(analysis.presupposition_triggers), 2)
            signals.append(f"presupposition:{','.join(analysis.presupposition_triggers[:2])}")
        total = score_obj + score_subj + score_value
        if total == 0:
            return "AMBIGUOUS", 0.0, signals
        norm_obj   = score_obj   / total
        norm_subj  = score_subj  / total
        norm_value = score_value / total
        max_score  = max(norm_obj, norm_subj, norm_value)
        if norm_value == max_score:
            return "VALUE_LADEN", max_score, signals
        if norm_subj == max_score:
            return "SUBJECTIVE",  max_score, signals
        return "OBJECTIVE", max_score, signals

    def _needs_llm(self, score: float, spacy_available: bool, tentative_label: str) -> bool:
        if not spacy_available:
            return True
        if tentative_label == "AMBIGUOUS":
            return True
        return _LOW_THRESH <= score <= _HIGH_THRESH

    def _layer_d(
        self, text: str, analysis: SentenceAnalysis,
        rule_score: float, signals: List[str], fallback_label: QuestionType,
    ) -> ObjectivityResult:
        prompt   = self._build_llm_prompt(text, analysis, signals)
        response = self._call_llm(prompt)
        if response is None:
            return ObjectivityResult(
                question_type=fallback_label, confidence=0.5,
                rule_score=rule_score, llm_used=False, signals=signals,
            )
        label, confidence = self._parse_llm_response(response)
        try:
            from mycelium.pipeline.layer0.dataset_logger import log_entry
            log_entry(
                component="objectivity",
                text=text,
                llm_label=label,
                confidence=confidence,
                rule_score=rule_score,
                matched_signals=signals,
                sentence_type=analysis.sentence_type,
            )
        except Exception:
            pass
        return ObjectivityResult(
            question_type=label, confidence=confidence,
            rule_score=rule_score, llm_used=True, signals=signals + [f"llm:{label}"],
        )

    def _build_llm_prompt(self, text: str, analysis: SentenceAnalysis, signals: List[str]) -> str:
        triples = [f"({s}, {p}, {o})" for s, p, o in analysis.dep_triples] or ["none"]
        return (
            "You are a linguistic objectivity classification function.\n"
            f"  Sentence type          : {analysis.sentence_type}\n"
            f"  Dep-parse triples      : {', '.join(triples)}\n"
            f"  Evaluative words found : {', '.join(analysis.evaluative_words or ['none'])}\n"
            f"  Presupposition hooks   : {', '.join(analysis.presupposition_triggers or ['none'])}\n"
            f"  Rule signals           : {', '.join(signals) or 'none'}\n"
            f'Raw query: "{text}"\n\n'
            f"Classify as: {' | '.join(_VALID_LABELS)}\n"
            "OBJECTIVE=factual evidence, SUBJECTIVE=opinion, VALUE_LADEN=normative, AMBIGUOUS=unclear\n"
            "ONLY:\nLABEL: <label>\nCONFIDENCE: <0.0-1.0>\n"
        )

    def _call_llm(self, prompt: str) -> Optional[str]:
        try:
            from mycelium.core.llm_interface import call_llm  # type: ignore
            return call_llm(system="Classify as directed. Return ONLY the two-line format.", user=prompt)
        except Exception:
            pass
        try:
            import requests  # type: ignore
            resp = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": os.getenv("MYCELIUM_LLM_MODEL", "llama3"),
                    "messages": [
                        {"role": "system", "content": "Classify. Return ONLY the two-line format."},
                        {"role": "user",   "content": prompt},
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

    def _parse_llm_response(self, response: str) -> Tuple[QuestionType, float]:
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
                    confidence = max(0.0, min(1.0, float(line.split(":", 1)[1].strip())))
                except ValueError:
                    pass
        if label == "AMBIGUOUS":
            for valid in _VALID_LABELS:
                if valid in response.upper():
                    label = valid  # type: ignore
                    break
        return label, confidence
