"""
Layer 0 — Value Assumption Extractor.

Four-layer extraction stack.
    Layer D fires the LLM only when spaCy was absent or assumption count < 2.
    If a trained AssumptionTyper is present in the model registry,
    it replaces the LLM for the typing step.
    Every LLM call is logged to dataset_logger.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from mycelium.pipeline.layer0.nlp_preprocessor import (
    SentenceAnalysis,
    get_preprocessor,
)

_ASSUMPTION_TYPES = [
    "FACTIVE_PRESUPPOSITION",
    "EXISTENTIAL_PRESUPPOSITION",
    "CHANGE_OF_STATE",
    "ADDITIVE_PRESUPPOSITION",
    "CLEFT_FOCUS",
    "FALSE_DICHOTOMY",
    "VALUE_FRAME",
    "NORMATIVE_UNIVERSAL",
    "CAUSAL_PRESUPPOSITION",
]
_LLM_THRESHOLD_TRIGGERS = 2


@dataclass
class Assumption:
    text:       str
    atype:      str
    evidence:   str  = ""
    confidence: float = 0.8


@dataclass
class ValueAssumptionResult:
    assumptions: List[Assumption] = field(default_factory=list)
    llm_used:    bool = False

    @property
    def assumption_strings(self) -> List[str]:
        return [f"[{a.atype}] {a.text}" for a in self.assumptions]


class ValueAssumptionExtractor:
    def __init__(self) -> None:
        self._pre = get_preprocessor()

    def extract(self, question: str) -> ValueAssumptionResult:
        text = (question or "").strip()
        if not text:
            return ValueAssumptionResult()
        analysis: SentenceAnalysis = self._pre.analyse(text)
        assumptions: List[Assumption] = []
        assumptions.extend(self._layer_a(analysis, text))
        assumptions.extend(self._layer_b(analysis, text))
        assumptions.extend(self._layer_c(analysis))

        # Try sklearn typer first
        clf_extras = self._try_classifier(text, analysis)
        if clf_extras is not None:
            assumptions.extend(clf_extras)
            return ValueAssumptionResult(assumptions=assumptions, llm_used=False)

        if self._needs_llm(assumptions, analysis.spacy_available):
            llm_extras = self._layer_d(text, analysis, assumptions)
            if llm_extras:
                assumptions.extend(llm_extras)
                return ValueAssumptionResult(assumptions=assumptions, llm_used=True)
        return ValueAssumptionResult(assumptions=assumptions, llm_used=False)

    def _try_classifier(
        self, text: str, analysis: SentenceAnalysis
    ) -> Optional[List[Assumption]]:
        """Return list of additional typed Assumptions from the sklearn model, or None.

        After running clf.predict(), the three regex heuristics from
        train_layer0_models are applied to OR in any types the model may have
        under-weighted — particularly FALSE_DICHOTOMY, VALUE_FRAME, and
        CAUSAL_PRESUPPOSITION which are trained from regex-derived labels and
        should fire consistently at inference time too.
        """
        try:
            from mycelium.pipeline.model_registry import get_layer0_classifier, get_model
            from mycelium.pipeline.layer0.train_layer0_models import (
                _rule_signal_vector,
                _FALSE_DICHOTOMY_RE,
                _VALUE_FRAME_RE,
                _CAUSAL_PRESUPPOSITION_RE,
            )
            artifact = get_layer0_classifier("assumption_typer")
            if artifact is None:
                return None
            import numpy as np
            # Pull the already-warmed encoder from the registry — never reload from disk.
            encoder = get_model(
                "sentence-transformers/all-MiniLM-L6-v2",
                model_type="sentence_transformer",
                device="cpu",
            )
            emb = encoder.encode([text], convert_to_numpy=True)
            rule_vec = _rule_signal_vector(text).reshape(1, -1)
            X = np.hstack([emb, rule_vec])
            clf          = artifact["model"]
            active_types = artifact["active_types"]
            preds = clf.predict(X)[0].tolist()  # multi-hot list

            # --- Regex heuristic override ---
            # These three types were trained from regex-derived labels, so the
            # same regexes must fire at inference time to guarantee consistency.
            def _set(atype: str) -> None:
                if atype in active_types:
                    preds[active_types.index(atype)] = 1

            if _FALSE_DICHOTOMY_RE.search(text):
                _set("FALSE_DICHOTOMY")
            if _VALUE_FRAME_RE.search(text):
                _set("VALUE_FRAME")
            if _CAUSAL_PRESUPPOSITION_RE.search(text):
                _set("CAUSAL_PRESUPPOSITION")

            extras: List[Assumption] = []
            for i, active in enumerate(preds):
                if active:
                    atype = active_types[i]
                    extras.append(Assumption(
                        text=f"Model-detected: {atype}",
                        atype=atype,
                        evidence="sklearn:assumption_typer",
                        confidence=0.75,
                    ))
            return extras
        except Exception:
            return None

    # Layers A, B, C, D — unchanged

    def _layer_a(self, analysis: SentenceAnalysis, raw_text: str) -> List[Assumption]:
        found: List[Assumption] = []
        for trigger in analysis.presupposition_triggers:
            if trigger.startswith("factive_verb:"):
                verb = trigger.split(":", 1)[1]
                found.append(Assumption(text=f"Factive verb '{verb}' presupposes truth of complement", atype="FACTIVE_PRESUPPOSITION", evidence=verb, confidence=0.85))
            elif trigger.startswith("change_of_state:"):
                verb = trigger.split(":", 1)[1]
                found.append(Assumption(text=f"Change-of-state verb '{verb}' presupposes prior state", atype="CHANGE_OF_STATE", evidence=verb, confidence=0.80))
            elif trigger == "definite_superlative":
                found.append(Assumption(text="Definite superlative presupposes a ranking exists", atype="EXISTENTIAL_PRESUPPOSITION", evidence="the best/worst", confidence=0.80))
            elif trigger == "cleft_construction":
                found.append(Assumption(text="Cleft construction presupposes X is the unique focus", atype="CLEFT_FOCUS", evidence="it is … that", confidence=0.75))
            elif trigger == "additive_particle:also":
                found.append(Assumption(text="'also' presupposes prior related activity", atype="ADDITIVE_PRESUPPOSITION", evidence="also", confidence=0.70))
            elif trigger == "why_causal_presupposition":
                found.append(Assumption(text="'Why' + causal verb presupposes the causal relationship", atype="CAUSAL_PRESUPPOSITION", evidence="why+causal_verb", confidence=0.85))
        for sig in analysis.coercive_signals:
            if sig.startswith("absolutist_quantifier:"):
                word = sig.split(":", 1)[1]
                found.append(Assumption(text=f"Absolutist quantifier '{word}' embeds a universal claim", atype="NORMATIVE_UNIVERSAL", evidence=word, confidence=0.75))
        return found

    def _layer_b(self, analysis: SentenceAnalysis, raw_text: str) -> List[Assumption]:
        found: List[Assumption] = []
        or_pattern = re.compile(
            r"\b(is it|was it|should (it|we|i)|do you think it(?:'s| is)?)\b"
            r".{3,60}\bor\b.{3,60}[?]", re.IGNORECASE,
        )
        if or_pattern.search(raw_text):
            found.append(Assumption(text="Binary 'is it A or B?' framing may exclude other options", atype="FALSE_DICHOTOMY", evidence="or", confidence=0.70))
            return found
        if analysis.dep_triples and analysis.spacy_available:
            tokens = analysis.tokens
            or_tokens = [t for t in tokens if t.dep == "cc" and t.lemma == "or"]
            if or_tokens:
                eval_lemmas = {t.lemma for t in tokens if t.pos in {"ADJ", "NOUN"} and not t.is_stop}
                if len(eval_lemmas) >= 2:
                    found.append(Assumption(text="Coordinating 'or' between evaluative terms may be a false dichotomy", atype="FALSE_DICHOTOMY", evidence="or (dep:cc)", confidence=0.65))
        return found

    def _layer_c(self, analysis: SentenceAnalysis) -> List[Assumption]:
        found: List[Assumption] = []
        _VALUE_FRAME_ADJ = frozenset([
            "correct", "incorrect", "real", "true", "false", "obvious",
            "natural", "proper", "improper", "legitimate", "illegitimate",
            "normal", "abnormal", "rational", "irrational", "reasonable",
            "unreasonable", "valid", "invalid", "genuine", "fake",
        ])
        tokens = analysis.tokens
        for i, t in enumerate(tokens):
            if t.pos == "ADJ" and t.lemma in _VALUE_FRAME_ADJ:
                if i + 1 < len(tokens) and tokens[i + 1].pos in {"NOUN", "PROPN"}:
                    phrase = f"{t.token} {tokens[i + 1].token}"
                    found.append(Assumption(text=f"Embedded value frame: '{phrase}' presupposes normative standard", atype="VALUE_FRAME", evidence=phrase, confidence=0.75))
        if not tokens:
            lower = analysis.raw_text.lower()
            for adj in _VALUE_FRAME_ADJ:
                if re.search(rf"\b{adj}\b", lower):
                    found.append(Assumption(text=f"Evaluative adjective '{adj}' embeds a value frame", atype="VALUE_FRAME", evidence=adj, confidence=0.60))
        return found

    def _needs_llm(self, assumptions: List[Assumption], spacy_available: bool) -> bool:
        if not spacy_available:
            return True
        return len(assumptions) < _LLM_THRESHOLD_TRIGGERS

    def _layer_d(
        self, text: str, analysis: SentenceAnalysis,
        rule_assumptions: List[Assumption],
    ) -> List[Assumption]:
        prompt   = self._build_llm_prompt(text, analysis, rule_assumptions)
        response = self._call_llm(prompt)
        if response is None:
            return []
        results = self._parse_llm_response(response)
        try:
            from mycelium.pipeline.layer0.dataset_logger import log_entry
            detected_types = [a.atype for a in results]
            log_entry(
                component="assumption",
                text=text,
                llm_label="MULTI_LABEL",
                confidence=0.75,
                matched_signals=[t.atype for t in rule_assumptions],
                sentence_type=analysis.sentence_type,
                assumption_types=detected_types,
            )
        except Exception:
            pass
        return results

    def _build_llm_prompt(self, text: str, analysis: SentenceAnalysis, existing: List[Assumption]) -> str:
        triples = [f"({s}, {p}, {o})" for s, p, o in analysis.dep_triples] or ["none"]
        already_found = [f"[{a.atype}] {a.text}" for a in existing] or ["none"]
        return (
            "You are a linguistic presupposition analysis function.\n"
            f"  Sentence type: {analysis.sentence_type}\n"
            f"  Dep-parse triples: {', '.join(triples)}\n"
            f"  Presupposition hooks: {', '.join(analysis.presupposition_triggers or ['none'])}\n"
            f"  Coercive signals: {', '.join(analysis.coercive_signals or ['none'])}\n"
            "\nAssumptions already found:\n"
            + "\n".join(f"  - {a}" for a in already_found)
            + f"\n\nRaw query: \"{text}\"\n\n"
            "Identify ADDITIONAL hidden presuppositions not already listed.\n"
            "For each:\nASSUMPTION: <description>\n"
            f"TYPE: <one of: {' | '.join(_ASSUMPTION_TYPES)}>\n"
            "EVIDENCE: <word/phrase>\nCONFIDENCE: <0.0-1.0>\n---\n"
            "If none, output NONE. No other text."
        )

    def _call_llm(self, prompt: str) -> Optional[str]:
        try:
            from mycelium.core.llm_interface import call_llm  # type: ignore
            return call_llm(system="Extract presuppositions. Return ONLY the block format.", user=prompt)
        except Exception:
            pass
        try:
            import requests  # type: ignore
            resp = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": os.getenv("MYCELIUM_LLM_MODEL", "llama3"),
                    "messages": [
                        {"role": "system", "content": "Extract presuppositions. Return ONLY the block format."},
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

    def _parse_llm_response(self, response: str) -> List[Assumption]:
        response = re.sub(r"(?s)<think>.*?</think>", "", response).strip()
        if not response or response.strip().upper() == "NONE":
            return []
        assumptions: List[Assumption] = []
        for block in re.split(r"---+", response):
            block = block.strip()
            if not block:
                continue
            a_text = ""; a_type = "VALUE_FRAME"; a_evidence = ""; a_conf = 0.70
            for line in block.splitlines():
                line = line.strip()
                if line.upper().startswith("ASSUMPTION:"):
                    a_text = line.split(":", 1)[1].strip()
                elif line.upper().startswith("TYPE:"):
                    raw = line.split(":", 1)[1].strip().upper()
                    if raw in _ASSUMPTION_TYPES:
                        a_type = raw
                elif line.upper().startswith("EVIDENCE:"):
                    a_evidence = line.split(":", 1)[1].strip()
                elif line.upper().startswith("CONFIDENCE:"):
                    try:
                        a_conf = max(0.0, min(1.0, float(line.split(":", 1)[1].strip())))
                    except ValueError:
                        pass
            if a_text:
                assumptions.append(Assumption(text=a_text, atype=a_type, evidence=a_evidence, confidence=a_conf))
        return assumptions
