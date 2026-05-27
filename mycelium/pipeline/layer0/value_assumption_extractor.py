"""
Layer 0 — Value Assumption Extractor.

Extracts hidden presuppositions and embedded value frames from a query.
This is the most linguistically detailed of the three Layer 0 components.

Four-layer extraction stack:

    Layer A — Presupposition trigger taxonomy (dep parse):
               factive verbs, change-of-state verbs, definite superlatives,
               cleft constructions, additive particles, WH-causal hooks.
    Layer B — False dichotomy detector:
               CONJ[or] between two evaluatively opposite NPs.
    Layer C — Value-frame ADJ+NOUN extraction:
               evaluative adjectives attached to concrete nouns.
    Layer D — LLM arbiter: fires when spaCy unavailable or assumption
               count is low but signals suggest more complexity.

Output: ValueAssumptionResult with a typed List[Assumption] each carrying
    {text, type, confidence}.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from mycelium.pipeline.layer0.nlp_preprocessor import (
    SentenceAnalysis,
    get_preprocessor,
)

# ---------------------------------------------------------------------------
# Assumption types
# ---------------------------------------------------------------------------

_ASSUMPTION_TYPES = [
    "FACTIVE_PRESUPPOSITION",     # "Why does X cause Y?" → X causes Y
    "EXISTENTIAL_PRESUPPOSITION", # "The best approach" → there is a best approach
    "CHANGE_OF_STATE",            # "When did X start failing?" → X was functioning
    "ADDITIVE_PRESUPPOSITION",    # "Can X also do Y?" → X already does other things
    "CLEFT_FOCUS",                # "It is X that matters" → something matters
    "FALSE_DICHOTOMY",            # "Is it A or B?" ignoring other options
    "VALUE_FRAME",                # "the correct way", "the obvious solution"
    "NORMATIVE_UNIVERSAL",        # "always", "never", "everyone" frames
    "CAUSAL_PRESUPPOSITION",      # "Why does X cause Y?"
]

_LLM_THRESHOLD_TRIGGERS = 2  # If rule extraction yields < this, try LLM


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Assumption:
    """A single extracted presupposition or embedded value frame.

    text        — human-readable description of the assumption.
    atype       — one of _ASSUMPTION_TYPES.
    evidence    — the word/phrase that triggered the extraction.
    confidence  — 0–1.
    """
    text: str
    atype: str
    evidence: str = ""
    confidence: float = 0.8


@dataclass
class ValueAssumptionResult:
    """Result returned by ValueAssumptionExtractor.extract().

    assumptions — list of Assumption objects.
    llm_used    — True if Layer D fired.
    """
    assumptions: List[Assumption] = field(default_factory=list)
    llm_used: bool = False

    # Back-compat: expose assumptions as plain strings for router.py
    @property
    def assumption_strings(self) -> List[str]:
        return [f"[{a.atype}] {a.text}" for a in self.assumptions]


# ---------------------------------------------------------------------------
# ValueAssumptionExtractor
# ---------------------------------------------------------------------------

class ValueAssumptionExtractor:
    """Extract hidden presuppositions and value frames from a query.

    Usage::

        vae = ValueAssumptionExtractor()
        result = vae.extract("Why does everyone know the vaccine is dangerous?")
        for a in result.assumptions:
            print(a.atype, '→', a.text)
        # FACTIVE_PRESUPPOSITION → 'everyone' knows something (presupposes truth)
        # NORMATIVE_UNIVERSAL    → uses absolutist quantifier 'everyone'
        # VALUE_FRAME            → embedded frame: 'dangerous vaccine'
    """

    def __init__(self) -> None:
        self._pre = get_preprocessor()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def extract(self, question: str) -> ValueAssumptionResult:
        """Run the 4-layer value assumption extraction pipeline."""
        text = (question or "").strip()
        if not text:
            return ValueAssumptionResult()

        analysis: SentenceAnalysis = self._pre.analyse(text)

        # Layers A + B + C
        assumptions: List[Assumption] = []
        assumptions.extend(self._layer_a(analysis, text))
        assumptions.extend(self._layer_b(analysis, text))
        assumptions.extend(self._layer_c(analysis))

        # Layer D gate
        if self._needs_llm(assumptions, analysis.spacy_available):
            llm_extras = self._layer_d(text, analysis, assumptions)
            if llm_extras:
                assumptions.extend(llm_extras)
                return ValueAssumptionResult(assumptions=assumptions, llm_used=True)

        return ValueAssumptionResult(assumptions=assumptions, llm_used=False)

    # ------------------------------------------------------------------
    # Layer A — Presupposition trigger taxonomy
    # ------------------------------------------------------------------

    def _layer_a(
        self, analysis: SentenceAnalysis, raw_text: str
    ) -> List[Assumption]:
        found: List[Assumption] = []

        for trigger in analysis.presupposition_triggers:

            if trigger.startswith("factive_verb:"):
                verb = trigger.split(":", 1)[1]
                found.append(Assumption(
                    text=f"Factive verb '{verb}' presupposes the truth of its complement clause",
                    atype="FACTIVE_PRESUPPOSITION",
                    evidence=verb,
                    confidence=0.85,
                ))

            elif trigger.startswith("change_of_state:"):
                verb = trigger.split(":", 1)[1]
                found.append(Assumption(
                    text=f"Change-of-state verb '{verb}' presupposes the prior state was different",
                    atype="CHANGE_OF_STATE",
                    evidence=verb,
                    confidence=0.80,
                ))

            elif trigger == "definite_superlative":
                found.append(Assumption(
                    text="Definite superlative NP presupposes a ranking or ordering exists",
                    atype="EXISTENTIAL_PRESUPPOSITION",
                    evidence="the best/worst/most/least",
                    confidence=0.80,
                ))

            elif trigger == "cleft_construction":
                found.append(Assumption(
                    text="Cleft construction ('It is X that…') presupposes X is the unique focus",
                    atype="CLEFT_FOCUS",
                    evidence="it is … that",
                    confidence=0.75,
                ))

            elif trigger == "additive_particle:also":
                found.append(Assumption(
                    text="Additive particle 'also' presupposes prior related activity",
                    atype="ADDITIVE_PRESUPPOSITION",
                    evidence="also",
                    confidence=0.70,
                ))

            elif trigger == "why_causal_presupposition":
                found.append(Assumption(
                    text="'Why' + causal verb presupposes the causal relationship is established",
                    atype="CAUSAL_PRESUPPOSITION",
                    evidence="why + causal_verb",
                    confidence=0.85,
                ))

        # Normative universal from coercive signals
        for sig in analysis.coercive_signals:
            if sig.startswith("absolutist_quantifier:"):
                word = sig.split(":", 1)[1]
                found.append(Assumption(
                    text=f"Absolutist quantifier '{word}' embeds a universal claim",
                    atype="NORMATIVE_UNIVERSAL",
                    evidence=word,
                    confidence=0.75,
                ))

        return found

    # ------------------------------------------------------------------
    # Layer B — False dichotomy detector
    # ------------------------------------------------------------------

    def _layer_b(
        self, analysis: SentenceAnalysis, raw_text: str
    ) -> List[Assumption]:
        found: List[Assumption] = []

        # Heuristic: "Is it A or B?" where A and B are short NPs
        # Use both dep-parse (if available) and regex
        or_pattern = re.compile(
            r"\b(is it|was it|should (it|we|i)|do you think it(?:'s| is)?)\b"
            r".{3,60}\bor\b.{3,60}[?]",
            re.IGNORECASE,
        )
        if or_pattern.search(raw_text):
            found.append(Assumption(
                text="Binary 'is it A or B?' framing may exclude other options (false dichotomy)",
                atype="FALSE_DICHOTOMY",
                evidence="or",
                confidence=0.70,
            ))
            return found  # Don't double-count with dep parse

        # Dep-parse: CONJ[or] between two adjective/noun tokens with
        # opposite-polarity evaluative words
        if analysis.dep_triples and analysis.spacy_available:
            tokens = analysis.tokens
            # Find tokens with dep=cc and lemma=or
            or_tokens = [t for t in tokens if t.dep == "cc" and t.lemma == "or"]
            if or_tokens:
                # Check if siblings are evaluative
                eval_lemmas = {t.lemma for t in tokens
                               if t.pos in {"ADJ", "NOUN"} and not t.is_stop}
                if len(eval_lemmas) >= 2:
                    found.append(Assumption(
                        text="Coordinating 'or' between evaluative terms may frame a false dichotomy",
                        atype="FALSE_DICHOTOMY",
                        evidence="or (dep:cc)",
                        confidence=0.65,
                    ))

        return found

    # ------------------------------------------------------------------
    # Layer C — Value-frame ADJ+NOUN extraction
    # ------------------------------------------------------------------

    def _layer_c(self, analysis: SentenceAnalysis) -> List[Assumption]:
        found: List[Assumption] = []

        # Strong value-frame noun phrases: evaluative_adj + noun
        # e.g. "the correct way", "the real problem", "the obvious solution"
        _VALUE_FRAME_ADJ = frozenset([
            "correct", "incorrect", "real", "true", "false", "obvious",
            "natural", "proper", "improper", "legitimate", "illegitimate",
            "normal", "abnormal", "rational", "irrational", "reasonable",
            "unreasonable", "valid", "invalid", "genuine", "fake",
        ])

        tokens = analysis.tokens
        for i, t in enumerate(tokens):
            if t.pos == "ADJ" and t.lemma in _VALUE_FRAME_ADJ:
                # Look for an adjacent noun (right neighbour)
                if i + 1 < len(tokens) and tokens[i + 1].pos in {"NOUN", "PROPN"}:
                    phrase = f"{t.token} {tokens[i + 1].token}"
                    found.append(Assumption(
                        text=f"Embedded value frame: '{phrase}' presupposes normative standard",
                        atype="VALUE_FRAME",
                        evidence=phrase,
                        confidence=0.75,
                    ))

        # Lexical fallback when no spaCy tokens
        if not tokens:
            lower = analysis.raw_text.lower()
            for adj in _VALUE_FRAME_ADJ:
                if re.search(rf"\b{adj}\b", lower):
                    found.append(Assumption(
                        text=f"Evaluative adjective '{adj}' embeds a value frame",
                        atype="VALUE_FRAME",
                        evidence=adj,
                        confidence=0.60,
                    ))

        return found

    # ------------------------------------------------------------------
    # Layer D — LLM arbiter
    # ------------------------------------------------------------------

    def _needs_llm(
        self, assumptions: List[Assumption], spacy_available: bool
    ) -> bool:
        """Fire LLM when spaCy was absent or assumption count is suspiciously low."""
        if not spacy_available:
            return True
        # If we found very few assumptions but presupposition signals exist,
        # the LLM may surface subtler ones.
        return len(assumptions) < _LLM_THRESHOLD_TRIGGERS

    def _layer_d(
        self,
        text: str,
        analysis: SentenceAnalysis,
        rule_assumptions: List[Assumption],
    ) -> List[Assumption]:
        """Layer D: LLM extracts additional assumptions not caught by rules."""
        prompt = self._build_llm_prompt(text, analysis, rule_assumptions)
        response = self._call_llm(prompt)
        if response is None:
            return []
        return self._parse_llm_response(response)

    def _build_llm_prompt(
        self,
        text: str,
        analysis: SentenceAnalysis,
        existing: List[Assumption],
    ) -> str:
        triples = [
            f"({s}, {p}, {o})" for s, p, o in analysis.dep_triples
        ] or ["none"]
        presuppositions = analysis.presupposition_triggers or ["none"]
        coercive = analysis.coercive_signals or ["none"]
        already_found = [f"[{a.atype}] {a.text}" for a in existing] or ["none"]

        return (
            "You are a linguistic presupposition analysis function.\n"
            "The following structural signals were extracted from the query:\n"
            f"  Sentence type          : {analysis.sentence_type}\n"
            f"  Dep-parse triples      : {', '.join(triples)}\n"
            f"  Presupposition hooks   : {', '.join(presuppositions)}\n"
            f"  Coercive signals       : {', '.join(coercive)}\n"
            "\n"
            f"Assumptions already found by rules:\n"
            + "\n".join(f"  - {a}" for a in already_found)
            + "\n\n"
            f'Raw query: "{text}"\n'
            "\n"
            "Identify any ADDITIONAL hidden presuppositions or embedded value frames\n"
            "not already listed above.  For each one output:\n"
            "ASSUMPTION: <human-readable description of what is presupposed>\n"
            f"TYPE: <one of: {' | '.join(_ASSUMPTION_TYPES)}>\n"
            "EVIDENCE: <word or phrase that triggers it>\n"
            "CONFIDENCE: <0.0-1.0>\n"
            "---\n"
            "Repeat the block for each additional assumption.  If none, output NONE.\n"
            "Do NOT repeat assumptions already found.  No other text."
        )

    def _call_llm(self, prompt: str) -> Optional[str]:
        try:
            from mycelium.core.llm_interface import call_llm  # type: ignore
            return call_llm(
                system="Extract presuppositions as directed. Return ONLY the block format.",
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
                         "content": "Extract presuppositions. Return ONLY the block format."},
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

    def _parse_llm_response(self, response: str) -> List[Assumption]:
        """Parse repeated ASSUMPTION/TYPE/EVIDENCE/CONFIDENCE blocks."""
        # Strip think tags
        response = re.sub(r"(?s)<think>.*?</think>", "", response).strip()

        if not response or response.strip().upper() == "NONE":
            return []

        assumptions: List[Assumption] = []
        # Split on --- separator
        blocks = re.split(r"---+", response)
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            a_text = ""
            a_type = "VALUE_FRAME"
            a_evidence = ""
            a_conf = 0.70
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
                        a_conf = max(0.0, min(1.0,
                            float(line.split(":", 1)[1].strip())
                        ))
                    except ValueError:
                        pass
            if a_text:
                assumptions.append(Assumption(
                    text=a_text,
                    atype=a_type,
                    evidence=a_evidence,
                    confidence=a_conf,
                ))
        return assumptions
