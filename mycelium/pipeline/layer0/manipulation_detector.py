"""
Layer 0 — Manipulation Detector.

Four-layer detection stack:

    Layer A — Syntactic presupposition extraction (dep parse via spaCy).
    Layer B — Lexical coercion signals (keyword + POS pattern matching).
    Layer C — Semantic role decomposition (subject-predicate-object triples).
    Layer D — LLM arbiter:
               • If a trained ManipulationClassifier is present in the
                 model registry, use it directly (no LLM call).
               • Otherwise fire the LLM when score is in the ambiguous band
                 [LOW_THRESH, HIGH_THRESH] OR spaCy was unavailable.
               • Every LLM call is logged to dataset_logger for future training.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

from mycelium.pipeline.layer0.nlp_preprocessor import (
    SentenceAnalysis,
    get_preprocessor,
)

_LOW_THRESH  = 0.25
_HIGH_THRESH = 0.60

_W_JAILBREAK         = 1.0
_W_FORCED_AGREEMENT  = 0.55
_W_MODAL_IMPERATIVE  = 0.35
_W_ABSOLUTIST        = 0.20
_W_URGENCY           = 0.25
_W_PASSIVE_HIDING    = 0.20
_W_LOADED_QUESTION   = 0.40
_W_PRESUPPOSITION    = 0.30

_VALID_LABELS = [
    "NOT_MANIPULATIVE",
    "COERCIVE",
    "LOADED_QUESTION",
    "JAILBREAK_ATTEMPT",
    "PRESUPPOSITION_INJECTION",
]


@dataclass
class ManipulationDetectionResult:
    is_manipulative: bool
    label: str = "NOT_MANIPULATIVE"
    matched_patterns: List[str] = field(default_factory=list)
    rule_score: float = 0.0
    llm_used: bool = False
    confidence: float = 0.0


class ManipulationDetector:
    def __init__(self) -> None:
        self._pre = get_preprocessor()
        self._llm_client = None

    def detect(self, question: str) -> ManipulationDetectionResult:
        text = (question or "").strip()
        if not text:
            return ManipulationDetectionResult(
                is_manipulative=False, label="NOT_MANIPULATIVE", confidence=1.0
            )
        analysis: SentenceAnalysis = self._pre.analyse(text)
        score, matched = self._score_signals(analysis)

        # --- Try trained sklearn model first (Layer D, no LLM) ---
        clf_result = self._try_classifier(text, analysis)
        if clf_result is not None:
            label, confidence = clf_result
            return ManipulationDetectionResult(
                is_manipulative=(label != "NOT_MANIPULATIVE"),
                label=label,
                matched_patterns=matched,
                rule_score=score,
                llm_used=False,
                confidence=confidence,
            )

        # --- LLM gate (only when no trained model) ---
        if self._needs_llm(score, analysis.spacy_available):
            return self._layer_d(text, analysis, score, matched)

        label = self._score_to_label(score, matched)
        return ManipulationDetectionResult(
            is_manipulative=(label != "NOT_MANIPULATIVE"),
            label=label,
            matched_patterns=matched,
            rule_score=score,
            llm_used=False,
            confidence=min(1.0, score) if label != "NOT_MANIPULATIVE" else 1.0 - score,
        )

    # ------------------------------------------------------------------
    # sklearn classifier inference
    # ------------------------------------------------------------------

    def _try_classifier(
        self, text: str, analysis: SentenceAnalysis
    ) -> Optional[tuple[str, float]]:
        """Return (label, confidence) from the trained model, or None."""
        try:
            from mycelium.pipeline.model_registry import get_layer0_classifier
            from mycelium.pipeline.layer0.train_layer0_models import _rule_signal_vector
            artifact = get_layer0_classifier("manipulation_classifier")
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
            label = le.inverse_transform([idx])[0]
            confidence = float(proba[idx])
            return label, confidence
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Layers A + B + C
    # ------------------------------------------------------------------

    def _score_signals(self, analysis: SentenceAnalysis) -> tuple[float, List[str]]:
        score = 0.0
        matched: List[str] = []
        for sig in analysis.coercive_signals:
            if sig == "jailbreak_structural":
                score += _W_JAILBREAK; matched.append(sig)
            elif sig == "forced_agreement_phrase":
                score += _W_FORCED_AGREEMENT; matched.append(sig)
            elif sig.startswith("modal_imperative:"):
                score += _W_MODAL_IMPERATIVE; matched.append(sig)
            elif sig.startswith("absolutist_quantifier:"):
                score += _W_ABSOLUTIST; matched.append(sig)
            elif sig.startswith("imperative_urgency:"):
                score += _W_URGENCY; matched.append(sig)
            elif sig == "passive_agency_hiding":
                score += _W_PASSIVE_HIDING; matched.append(sig)
        for trig in analysis.presupposition_triggers:
            if trig.startswith("factive_verb:") or trig in {"cleft_construction", "additive_particle:also"}:
                score += _W_PRESUPPOSITION; matched.append(trig)
            if trig == "why_causal_presupposition":
                score += _W_LOADED_QUESTION; matched.append(trig)
        if (
            analysis.sentence_type == "INTERROGATIVE"
            and analysis.presupposition_triggers
            and analysis.spacy_available is not False
        ):
            score += _W_LOADED_QUESTION * 0.5
            matched.append("interrogative_with_presupposition")
        return min(score, 1.5), matched

    def _score_to_label(self, score: float, matched: List[str]) -> str:
        if score < _LOW_THRESH:
            return "NOT_MANIPULATIVE"
        if any("jailbreak" in m for m in matched):
            return "JAILBREAK_ATTEMPT"
        if any("forced_agreement" in m for m in matched):
            return "COERCIVE"
        if any("modal_imperative" in m for m in matched):
            return "COERCIVE"
        if "why_causal_presupposition" in matched or "interrogative_with_presupposition" in matched:
            return "LOADED_QUESTION"
        if any("factive_verb" in m or "cleft_construction" in m for m in matched):
            return "PRESUPPOSITION_INJECTION"
        if score >= _HIGH_THRESH:
            return "COERCIVE"
        return "NOT_MANIPULATIVE"

    # ------------------------------------------------------------------
    # Layer D — LLM arbiter
    # ------------------------------------------------------------------

    def _needs_llm(self, score: float, spacy_available: bool) -> bool:
        if not spacy_available:
            return True
        return _LOW_THRESH <= score <= _HIGH_THRESH

    def _layer_d(
        self, text: str, analysis: SentenceAnalysis,
        rule_score: float, matched: List[str],
    ) -> ManipulationDetectionResult:
        prompt = self._build_llm_prompt(text, analysis, matched)
        llm_response = self._call_llm(prompt)
        if llm_response is None:
            label = self._score_to_label(rule_score, matched)
            return ManipulationDetectionResult(
                is_manipulative=(label != "NOT_MANIPULATIVE"),
                label=label, matched_patterns=matched,
                rule_score=rule_score, llm_used=False, confidence=0.5,
            )
        label, confidence = self._parse_llm_response(llm_response)
        # --- Log for training ---
        try:
            from mycelium.pipeline.layer0.dataset_logger import log_entry
            log_entry(
                component="manipulation",
                text=text,
                llm_label=label,
                confidence=confidence,
                rule_score=rule_score,
                matched_signals=matched,
                sentence_type=analysis.sentence_type,
            )
        except Exception:
            pass
        return ManipulationDetectionResult(
            is_manipulative=(label != "NOT_MANIPULATIVE"),
            label=label,
            matched_patterns=matched + [f"llm:{label}"],
            rule_score=rule_score, llm_used=True, confidence=confidence,
        )

    def _build_llm_prompt(self, text: str, analysis: SentenceAnalysis, matched: List[str]) -> str:
        coercive      = analysis.coercive_signals or ["none"]
        presuppositions = analysis.presupposition_triggers or ["none"]
        evaluative    = analysis.evaluative_words or ["none"]
        triples = [f"({s}, {p}, {o})" for s, p, o in analysis.dep_triples] or ["none"]
        return (
            "You are a linguistic manipulation analysis function.\n"
            "The following structural signals were automatically extracted from the query:\n"
            f"  Sentence type        : {analysis.sentence_type}\n"
            f"  Coercive signals     : {', '.join(coercive)}\n"
            f"  Presupposition hooks : {', '.join(presuppositions)}\n"
            f"  Evaluative words     : {', '.join(evaluative)}\n"
            f"  Dep-parse triples    : {', '.join(triples)}\n"
            f"  Rule-based score     : {matched!r}\n"
            "\n"
            f'Raw query: "{text}"\n\n'
            f"Classify as one of: {' | '.join(_VALID_LABELS)}\n"
            "Also provide confidence 0.0-1.0.\n"
            "Respond ONLY:\nLABEL: <label>\nCONFIDENCE: <0.0-1.0>\n"
        )

    def _call_llm(self, prompt: str) -> Optional[str]:
        try:
            from mycelium.core.llm_interface import call_llm  # type: ignore
            return call_llm(
                system="Analyze structural signals and classify. Return ONLY the two-line format.",
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
                        {"role": "system", "content": "Classify as directed. Return ONLY the two-line format."},
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

    def _parse_llm_response(self, response: str) -> tuple[str, float]:
        import re
        label = "NOT_MANIPULATIVE"
        confidence = 0.5
        response = re.sub(r"(?s)<think>.*?</think>", "", response).strip()
        for line in response.splitlines():
            line = line.strip()
            if line.upper().startswith("LABEL:"):
                raw = line.split(":", 1)[1].strip().upper()
                if raw in _VALID_LABELS:
                    label = raw
            elif line.upper().startswith("CONFIDENCE:"):
                try:
                    confidence = max(0.0, min(1.0, float(line.split(":", 1)[1].strip())))
                except ValueError:
                    pass
        if label == "NOT_MANIPULATIVE":
            for valid in _VALID_LABELS:
                if valid in response.upper():
                    label = valid
                    break
        return label, confidence
