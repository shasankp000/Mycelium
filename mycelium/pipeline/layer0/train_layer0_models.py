"""
train_layer0_models.py
=======================
Offline trainer for the three Layer 0 sklearn classifiers.

    ManipulationClassifier  →  models/layer0/manipulation_classifier.joblib
    ObjectivityClassifier   →  models/layer0/objectivity_classifier.joblib
    AssumptionTyper         →  models/layer0/assumption_typer.joblib

Dataset sources (auto-downloaded on first run, skipped if already present):
    training_data/jailbreakbench/   — JailbreakBench jailbreak prompts
    training_data/liar_train.csv    — LIAR dataset
    training_data/trivia_qa.csv     — TriviaQA rc sample
    training_data/ethics_qa.csv     — Hendrycks ETHICS commonsense
    training_data/anthropic_hh.csv  — Anthropic HH-RLHF harmless-base
    training_data/benign_prompts.jsonl — ~1000 normal questions

Feature vector per sample:
    [sentence_embedding (384-dim, all-MiniLM-L6-v2)]
    + [rule_signal_vector (one-hot, ~17 dims)]

Model architecture:
    ManipulationClassifier  — CalibratedClassifierCV(
                                LinearSVC(C=1.0, balanced),
                                method='isotonic', cv=5)
    ObjectivityClassifier   — CalibratedClassifierCV(
                                LogisticRegression(C=1.0, balanced),
                                method='isotonic', cv=5)
    AssumptionTyper         — MultiOutputClassifier(
                                CalibratedClassifierCV(
                                  LogisticRegression, method='isotonic', cv=5))

Class balance targets (ObjectivityClassifier):
    OBJECTIVE    ~2000  (trivia_qa factual questions)
    SUBJECTIVE   ~800   (liar statements)
    VALUE_LADEN  ~1500  (ethics, capped) + ~1500 (anthropic_hh, capped)
                 → merged to ~2000 after dedup/cap
    AMBIGUOUS    ~300   (auto-labelled via ambiguity markers)

    Previously the dataset had 5000+ VALUE_LADEN vs 2000 OBJECTIVE and
    0 AMBIGUOUS, causing the classifier to pull every science question
    containing evaluative-sounding words toward VALUE_LADEN →
    MULTI_PERSPECTIVE.

Usage:
    python -m mycelium.pipeline.layer0.train_layer0_models
    python -m mycelium.pipeline.layer0.train_layer0_models --models manipulation
    python -m mycelium.pipeline.layer0.train_layer0_models --skip-download
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import re
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np

csv.field_size_limit(sys.maxsize)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent.parent
_DATA_DIR   = _ROOT / "training_data"
_MODELS_DIR = _ROOT / "models" / "layer0"

_DATA_DIR.mkdir(parents=True, exist_ok=True)
_MODELS_DIR.mkdir(parents=True, exist_ok=True)

_MIN_TRAIN_SAMPLES = 50

# ---------------------------------------------------------------------------
# URL constants
# ---------------------------------------------------------------------------

_ETHICS_PARQUET_URL = (
    "https://huggingface.co/datasets/lighteval/hendrycks_ethics"
    "/resolve/main/commonsense/train-00000-of-00001.parquet"
)

_ANTHROPIC_HH_JSONL_URL = (
    "https://huggingface.co/datasets/Anthropic/hh-rlhf"
    "/resolve/main/harmless-base/train.jsonl.gz"
)

_LIAR_PARQUET_COMMIT = "110b00c693ef1844bf3c59637a1b46e0d61389c2"
_LIAR_TRAIN_PARQUET_URL = (
    f"https://huggingface.co/datasets/ucsbnlp/liar"
    f"/resolve/{_LIAR_PARQUET_COMMIT}/default/liar-train.parquet"
)

# ---------------------------------------------------------------------------
# LIAR label mapping
# ---------------------------------------------------------------------------

_LIAR_MANIP_MAP: dict[str, str] = {
    "0": "COERCIVE",
    "1": "COERCIVE",
    "2": "COERCIVE",
    "3": "NOT_MANIPULATIVE",
    "4": "NOT_MANIPULATIVE",
    "5": "NOT_MANIPULATIVE",
    "pants-fire":   "COERCIVE",
    "false":        "COERCIVE",
    "barely-true":  "COERCIVE",
    "half-true":    "NOT_MANIPULATIVE",
    "mostly-true":  "NOT_MANIPULATIVE",
    "true":         "NOT_MANIPULATIVE",
}
_LIAR_OBJ_LABEL = "SUBJECTIVE"

# ---------------------------------------------------------------------------
# Ambiguity markers (for auto-labelling AMBIGUOUS objectivity samples)
# ---------------------------------------------------------------------------
# These patterns indicate genuine ambiguity rather than value-ladenness:
# unclear referent, context-dependent meaning, either/or without presupposition.

_AMBIGUITY_MARKERS_RE = re.compile(
    r"\b("
    r"depends\s+on\s+(the|your|how|what|where|who|context)"
    r"|it(?:'s|\s+is)\s+(unclear|not\s+clear|ambiguous|debatable|uncertain)"
    r"|in\s+(some|many|certain|various)\s+(contexts?|cases?|situations?|circumstances?)"
    r"|(?:can|could|may|might)\s+(?:mean|refer\s+to|be\s+interpreted)"
    r"|(?:what\s+(?:exactly|specifically)\s+(?:do\s+you|does\s+\w+)\s+mean)"
    r"|(?:clarify|clarification|specify|more\s+(?:specific|precise|detail))"
    r"|(?:which|what)\s+(?:type|kind|sort|form)\s+of\b"
    r"|\bvague\b|\bimprecise\b|\bopen(?:-|\s+)ended\b"
    r")",
    re.IGNORECASE,
)

# Science / factual topic markers used to confirm a question is OBJECTIVE.
# Presence of these words in combination with interrogative structure strongly
# indicates a factual science question, not a value-laden one.
_SCIENCE_TOPIC_RE = re.compile(
    r"\b("
    r"physics?|chemistry|chemical|biolog(?:y|ical)|math(?:ematics?|ematical)?"
    r"|quantum|particle(?:s)?|atom(?:ic|s)?|molecul(?:e|ar|es)"
    r"|electron(?:s|ic)?|proton(?:s)?|neutron(?:s)?"
    r"|gravity|gravitational|magnetic|electro(?:magnetic|static)"
    r"|thermodynamics?|entropy|energy|force(?:s)?|momentum"
    r"|velocity|acceleration|wavelength|frequency|photon(?:s)?"
    r"|nucleus|nuclear|radioactive|isotope"
    r"|gene(?:tic|tics|s)?|cell(?:s|ular)?|protein(?:s)?"
    r"|evolution|organism(?:s)?|species"
    r"|temperature|pressure|volume|density|mass"
    r"|equation(?:s)?|theorem(?:s)?|formula(?:e|s)?"
    r"|element(?:s|al)?|compound(?:s)?|reaction(?:s)?"
    r"|solar|planet(?:s|ary)?|star(?:s)?|galaxy|universe|cosmic"
    r")",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Regex heuristics for AssumptionTyper
# ---------------------------------------------------------------------------

_FALSE_DICHOTOMY_RE = re.compile(
    r"\b("
    r"either\s+\w[\w\s,]+\s+or\b"
    r"|with\s+us\s+or\s+against\s+us"
    r"|you(?:'re|re|\s+are)\s+(either|only)\b"
    r"|only\s+two\s+(options|choices|paths|ways)"
    r"|if\s+you(?:'re|re|\s+are)\s+not\s+\w+,?\s+you(?:'re|re|\s+are)"
    r"|there(?:'s|\s+is)\s+no\s+(middle|third|other)\s+(ground|option|choice|way)"
    r")",
    re.IGNORECASE,
)

_VALUE_FRAME_RE = re.compile(
    r"\b("
    r"(?:must|have\s+to|need\s+to)\s+(?:protect|preserve|defend|uphold|safeguard)"
    r"|our\s+(?:values|way\s+of\s+life|traditions?|heritage|culture|freedom|rights?)"
    r"|(?:threatens?|undermines?|destroys?|erodes?)\s+our\s+\w+"
    r"|(?:sacred|fundamental|core|cherished)\s+(?:values?|rights?|principles?|beliefs?)"
    r"|(?:moral|ethical)\s+(?:duty|obligation|imperative|responsibility)"
    r"|(?:true|real|genuine)\s+(?:freedom|justice|equality|democracy)"
    r")",
    re.IGNORECASE,
)

_CAUSAL_PRESUPPOSITION_RE = re.compile(
    r"\b("
    r"why\s+(?:did|does|do|has|have|would|will|is|are|was|were)\b"
    r"|why\s+(?:can't|cannot|won't|wouldn't|didn't|doesn't|don't)\b"
    r"|what\s+(?:caused|made|led\s+to|resulted\s+in)\b"
    r"|(?:because\s+of|due\s+to|as\s+a\s+result\s+of|owing\s+to)\s+\w+"
    r"|(?:caused|triggered|produced|brought\s+about)\s+(?:the|a|an|this|that)\b"
    r")",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Parquet download helper
# ---------------------------------------------------------------------------

def _fetch_parquet(url: str, dest: Path, label: str) -> bool:
    import requests
    import pandas as pd

    log.info("[download] Fetching %s parquet from %s ...", label, url)
    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        df = pd.read_parquet(BytesIO(resp.content))
        df.to_csv(str(dest), index=False)
        log.info("[download] %s: %d rows written to %s", label, len(df), dest)
        return True
    except Exception as exc:
        log.warning("[download] Failed to fetch %s: %s", label, exc)
        return False


# ---------------------------------------------------------------------------
# Dataset download helpers
# ---------------------------------------------------------------------------

def _pull_jailbreakbench() -> Path:
    dest    = _DATA_DIR / "jailbreakbench"
    out_csv = _DATA_DIR / "jailbreakbench_train.csv"

    if out_csv.exists():
        log.info("[download] jailbreakbench_train.csv already present, skipping.")
        return out_csv

    if not dest.exists() or not any(dest.iterdir()):
        log.info("[download] Cloning JailbreakBench/artifacts ...")
        subprocess.run(
            ["git", "clone", "--depth=1",
             "https://github.com/JailbreakBench/artifacts",
             str(dest)],
            check=True,
        )

    import glob as _glob
    rows: List[dict] = []
    for json_path in _glob.glob(str(dest / "**" / "*.json"), recursive=True):
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            for entry in data.get("jailbreaks", []):
                prompt = (
                    entry.get("prompt") or entry.get("goal") or ""
                ).strip()
                if not prompt:
                    continue
                label = "JAILBREAK_ATTEMPT" if entry.get("jailbroken") else "BENIGN"
                rows.append({"text": prompt, "label": label})
        except Exception:
            pass

    if rows:
        import csv as _csv
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = _csv.DictWriter(f, fieldnames=["text", "label"])
            writer.writeheader()
            writer.writerows(rows)
        log.info("[download] jailbreakbench_train.csv: %d rows written.", len(rows))
    else:
        log.warning("[download] No jailbreak entries extracted from artifacts repo.")

    return out_csv


def _pull_liar() -> Path:
    dest = _DATA_DIR / "liar_train.csv"
    if dest.exists():
        log.info("[download] liar_train.csv already present, skipping.")
        return dest
    _fetch_parquet(_LIAR_TRAIN_PARQUET_URL, dest, "LIAR")
    return dest


def _pull_ethics() -> Path:
    dest = _DATA_DIR / "ethics_qa.csv"
    if dest.exists():
        log.info("[download] ethics_qa.csv already present, skipping.")
        return dest
    _fetch_parquet(_ETHICS_PARQUET_URL, dest, "ETHICS")
    return dest


def _pull_anthropic_hh() -> Path:
    dest = _DATA_DIR / "anthropic_hh.csv"
    if dest.exists():
        log.info("[download] anthropic_hh.csv already present, skipping.")
        return dest

    import requests

    log.info("[download] Fetching Anthropic HH-RLHF jsonl.gz from %s ...",
             _ANTHROPIC_HH_JSONL_URL)
    try:
        resp = requests.get(_ANTHROPIC_HH_JSONL_URL, timeout=180)
        resp.raise_for_status()

        def _first_human(text: str) -> str:
            m = re.search(r"Human:\s*(.+?)(?:\n\nAssistant:|$)", text, re.DOTALL)
            return m.group(1).strip() if m else ""

        rows: List[str] = []
        with gzip.open(BytesIO(resp.content), "rt", encoding="utf-8") as gz:
            for line in gz:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    text = _first_human(entry.get("chosen", ""))
                    if text:
                        rows.append(text)
                except Exception:
                    pass
                if len(rows) >= 3000:
                    break

        with open(dest, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["text"])
            for r in rows:
                writer.writerow([r])
        log.info("[download] anthropic_hh.csv: %d rows written.", len(rows))
    except Exception as exc:
        log.warning("[download] Failed to fetch Anthropic HH-RLHF: %s", exc)
    return dest


def _pull_hf_dataset(hf_name: str, config: str, split: str, dest: Path, **kwargs) -> Path:
    if dest.exists():
        log.info("[download] %s already present, skipping.", dest.name)
        return dest
    log.info("[download] Pulling %s (%s) ...", hf_name, split)
    try:
        from datasets import load_dataset
        ds = load_dataset(hf_name, config, split=split, **kwargs)
        ds.to_csv(str(dest))
        log.info("[download] Saved %d rows to %s", len(ds), dest)
    except Exception as exc:
        log.warning("[download] Failed to pull %s: %s", hf_name, exc)
    return dest


def pull_all_datasets() -> None:
    _pull_jailbreakbench()
    _pull_liar()
    _pull_hf_dataset(
        "trivia_qa", "rc", "train[:3000]",
        _DATA_DIR / "trivia_qa.csv",
    )
    _pull_ethics()
    _pull_anthropic_hh()


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

MANIP_LABELS = [
    "NOT_MANIPULATIVE",
    "COERCIVE",
    "LOADED_QUESTION",
    "JAILBREAK_ATTEMPT",
    "PRESUPPOSITION_INJECTION",
]

OBJ_LABELS = ["OBJECTIVE", "SUBJECTIVE", "VALUE_LADEN", "AMBIGUOUS"]

ASSUMPTION_TYPES = [
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

# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def _load_manipulation_data() -> List[Tuple[str, str]]:
    samples: List[Tuple[str, str]] = []

    jbb_csv = _DATA_DIR / "jailbreakbench_train.csv"
    jbb_count = 0
    if jbb_csv.exists():
        try:
            with open(jbb_csv, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text  = row.get("text", "").strip()
                    label = row.get("label", "").strip()
                    if not text or not label:
                        continue
                    mapped = label if label != "BENIGN" else "NOT_MANIPULATIVE"
                    samples.append((text, mapped))
                    jbb_count += 1
        except Exception as exc:
            log.warning("[manip] JailbreakBench CSV load error: %s", exc)
    log.info("[manip] jailbreakbench: %d samples", jbb_count)

    liar_path = _DATA_DIR / "liar_train.csv"
    liar_count = 0
    if liar_path.exists():
        try:
            with open(liar_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    stmt = row.get("statement", "").strip()
                    lbl  = row.get("label", "").strip().lower()
                    if not stmt or lbl not in _LIAR_MANIP_MAP:
                        continue
                    samples.append((stmt, _LIAR_MANIP_MAP[lbl]))
                    liar_count += 1
        except Exception as exc:
            log.warning("[manip] LIAR load error: %s", exc)
    log.info("[manip] liar: %d samples", liar_count)

    tqa_path = _DATA_DIR / "trivia_qa.csv"
    tqa_count = 0
    if tqa_path.exists():
        try:
            with open(tqa_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if i >= 1500:
                        break
                    q = row.get("question", "").strip()
                    if q:
                        samples.append((q, "NOT_MANIPULATIVE"))
                        tqa_count += 1
        except Exception as exc:
            log.warning("[manip] TriviaQA load error: %s", exc)
    log.info("[manip] trivia_qa: %d samples", tqa_count)

    _append_from_logger(samples, "manipulation")
    log.info("[manip] total samples: %d", len(samples))
    return samples


def _load_objectivity_data() -> List[Tuple[str, str]]:
    """
    Build a balanced objectivity training dataset.

    Target class distribution:
        OBJECTIVE    ~2000  (trivia_qa factual/science questions)
        VALUE_LADEN  ~2000  (ethics capped at 1500 + anthropic capped at 1500,
                             deduplicated and trimmed to 2000)
        SUBJECTIVE   ~800   (liar statements)
        AMBIGUOUS    ~300   (auto-labelled via ambiguity marker regex)

    The previous imbalance (5000+ VALUE_LADEN vs 2000 OBJECTIVE, 0 AMBIGUOUS)
    caused every science question to be pulled toward VALUE_LADEN because
    causal/evaluative vocabulary in physics questions overlaps with
    ethics/opinion text.
    """
    samples: List[Tuple[str, str]] = []
    seen_texts: set = set()

    def _add(text: str, label: str) -> bool:
        key = text.lower().strip()
        if key in seen_texts or not key:
            return False
        seen_texts.add(key)
        samples.append((text, label))
        return True

    # --- OBJECTIVE: trivia_qa science + general factual questions ---
    tqa_path = _DATA_DIR / "trivia_qa.csv"
    tqa_count = 0
    if tqa_path.exists():
        try:
            with open(tqa_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if tqa_count >= 2000:
                        break
                    q = row.get("question", "").strip()
                    if q and _add(q, "OBJECTIVE"):
                        tqa_count += 1
        except Exception as exc:
            log.warning("[obj] TriviaQA load error: %s", exc)
    else:
        log.warning("[obj] trivia_qa.csv not found")
    log.info("[obj] trivia_qa (OBJECTIVE): %d samples", tqa_count)

    # --- VALUE_LADEN: ethics (cap at 1500) ---
    ethics_path = _DATA_DIR / "ethics_qa.csv"
    ethics_count = 0
    _VALUE_LADEN_CAP = 1500
    if ethics_path.exists():
        try:
            with open(ethics_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if ethics_count >= _VALUE_LADEN_CAP:
                        break
                    inp = (
                        row.get("input")
                        or row.get("sentence")
                        or row.get("text")
                        or ""
                    ).strip()
                    if inp and _add(inp, "VALUE_LADEN"):
                        ethics_count += 1
        except Exception as exc:
            log.warning("[obj] Ethics load error: %s", exc)
    else:
        log.warning("[obj] ethics_qa.csv not found")
    log.info("[obj] ethics_qa (VALUE_LADEN): %d samples", ethics_count)

    # --- VALUE_LADEN: anthropic_hh (cap at 1500, combined budget ~2000) ---
    hh_path = _DATA_DIR / "anthropic_hh.csv"
    hh_count = 0
    # Remaining budget to VALUE_LADEN so combined <= ~2000 total
    hh_cap = max(0, 2000 - ethics_count)
    if hh_path.exists():
        try:
            with open(hh_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if hh_count >= hh_cap:
                        break
                    text = row.get("text", "").strip()
                    if text and _add(text, "VALUE_LADEN"):
                        hh_count += 1
        except Exception as exc:
            log.warning("[obj] Anthropic HH load error: %s", exc)
    else:
        log.warning("[obj] anthropic_hh.csv not found")
    log.info("[obj] anthropic_hh (VALUE_LADEN): %d samples (cap=%d)", hh_count, hh_cap)

    # --- SUBJECTIVE: liar statements (cap at 800) ---
    liar_path = _DATA_DIR / "liar_train.csv"
    liar_count = 0
    if liar_path.exists():
        try:
            with open(liar_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if liar_count >= 800:
                        break
                    stmt = row.get("statement", "").strip()
                    if stmt and _add(stmt, _LIAR_OBJ_LABEL):
                        liar_count += 1
        except Exception as exc:
            log.warning("[obj] LIAR load error: %s", exc)
    else:
        log.warning("[obj] liar_train.csv not found")
    log.info("[obj] liar (SUBJECTIVE): %d samples", liar_count)

    # --- AMBIGUOUS: auto-label via ambiguity markers from all sources ---
    # Scan trivia_qa, ethics, and liar for sentences matching the ambiguity
    # marker regex. Cap at 400 to avoid over-representing the class.
    ambig_count = 0
    _AMBIG_CAP = 400
    ambig_sources = []
    if (tqa_path := _DATA_DIR / "trivia_qa.csv").exists():
        ambig_sources.append((tqa_path, "question"))
    if (ethics_path := _DATA_DIR / "ethics_qa.csv").exists():
        ambig_sources.append((ethics_path, "input"))
    if (liar_path := _DATA_DIR / "liar_train.csv").exists():
        ambig_sources.append((liar_path, "statement"))

    for src_path, col in ambig_sources:
        if ambig_count >= _AMBIG_CAP:
            break
        try:
            with open(src_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if ambig_count >= _AMBIG_CAP:
                        break
                    text = (row.get(col) or row.get("text") or "").strip()
                    if text and _AMBIGUITY_MARKERS_RE.search(text):
                        # Only label as AMBIGUOUS if not already labelled
                        key = text.lower().strip()
                        if key not in seen_texts:
                            seen_texts.add(key)
                            samples.append((text, "AMBIGUOUS"))
                            ambig_count += 1
        except Exception as exc:
            log.warning("[obj] AMBIGUOUS auto-label error for %s: %s", src_path, exc)
    log.info("[obj] auto-labelled AMBIGUOUS: %d samples", ambig_count)

    _append_from_logger(samples, "objectivity")

    # Log final class distribution
    from collections import Counter
    dist = Counter(label for _, label in samples)
    log.info(
        "[obj] final distribution: %s  total=%d",
        dict(dist), len(samples),
    )
    return samples


def _load_assumption_data() -> List[Tuple[str, List[int]]]:
    from mycelium.pipeline.layer0.nlp_preprocessor import get_preprocessor
    pre = get_preprocessor()
    samples: List[Tuple[str, List[int]]] = []

    def _auto_label(text: str) -> List[int]:
        try:
            a = pre.analyse(text)
        except Exception:
            a = None

        vec = [0] * len(ASSUMPTION_TYPES)

        if a is not None:
            for trig in a.presupposition_triggers:
                if trig.startswith("factive_verb:"):
                    vec[ASSUMPTION_TYPES.index("FACTIVE_PRESUPPOSITION")] = 1
                elif trig.startswith("change_of_state:"):
                    vec[ASSUMPTION_TYPES.index("CHANGE_OF_STATE")] = 1
                elif trig == "definite_superlative":
                    vec[ASSUMPTION_TYPES.index("EXISTENTIAL_PRESUPPOSITION")] = 1
                elif trig == "cleft_construction":
                    vec[ASSUMPTION_TYPES.index("CLEFT_FOCUS")] = 1
                elif trig == "additive_particle:also":
                    vec[ASSUMPTION_TYPES.index("ADDITIVE_PRESUPPOSITION")] = 1
                elif trig == "why_causal_presupposition":
                    vec[ASSUMPTION_TYPES.index("CAUSAL_PRESUPPOSITION")] = 1
            for sig in a.coercive_signals:
                if sig.startswith("absolutist_quantifier:"):
                    vec[ASSUMPTION_TYPES.index("NORMATIVE_UNIVERSAL")] = 1

        if _FALSE_DICHOTOMY_RE.search(text):
            vec[ASSUMPTION_TYPES.index("FALSE_DICHOTOMY")] = 1
        if _VALUE_FRAME_RE.search(text):
            vec[ASSUMPTION_TYPES.index("VALUE_FRAME")] = 1
        if _CAUSAL_PRESUPPOSITION_RE.search(text):
            vec[ASSUMPTION_TYPES.index("CAUSAL_PRESUPPOSITION")] = 1

        return vec

    tqa_path = _DATA_DIR / "trivia_qa.csv"
    if tqa_path.exists():
        try:
            with open(tqa_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if i >= 1500:
                        break
                    q = row.get("question", "").strip()
                    if q:
                        samples.append((q, _auto_label(q)))
        except Exception as exc:
            log.warning("[assumption] TriviaQA load error: %s", exc)

    ethics_path = _DATA_DIR / "ethics_qa.csv"
    if ethics_path.exists():
        try:
            with open(ethics_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    inp = (
                        row.get("input")
                        or row.get("sentence")
                        or row.get("text")
                        or ""
                    ).strip()
                    if inp:
                        samples.append((inp, _auto_label(inp)))
        except Exception as exc:
            log.warning("[assumption] Ethics load error: %s", exc)

    liar_path = _DATA_DIR / "liar_train.csv"
    liar_assumption_count = 0
    if liar_path.exists():
        try:
            with open(liar_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    stmt = row.get("statement", "").strip()
                    if stmt:
                        vec = _auto_label(stmt)
                        samples.append((stmt, vec))
                        if any(vec):
                            liar_assumption_count += 1
        except Exception as exc:
            log.warning("[assumption] LIAR load error: %s", exc)
        log.info("[assumption] liar: %d samples with ≥1 active type", liar_assumption_count)

    logger_path = _ROOT / "training_data" / "dataset_log.jsonl"
    if logger_path.exists():
        with open(logger_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("component") != "assumption":
                        continue
                    text = entry.get("text", "")
                    types_found = entry.get("assumption_types", [])
                    vec = [1 if t in types_found else 0 for t in ASSUMPTION_TYPES]
                    if text:
                        samples.append((text, vec))
                except Exception:
                    pass

    log.info("[assumption] total samples: %d", len(samples))
    return samples


def _append_from_logger(
    samples: List[Tuple[str, str]], component: str
) -> None:
    logger_path = _ROOT / "training_data" / "dataset_log.jsonl"
    if not logger_path.exists():
        return
    count = 0
    with open(logger_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("component") != component:
                    continue
                text  = entry.get("text", "")
                label = entry.get("llm_label", "")
                if text and label:
                    samples.append((text, label))
                    count += 1
            except Exception:
                pass
    if count:
        log.info("[%s] +%d samples from dataset_logger", component, count)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _rule_signal_vector(text: str) -> np.ndarray:
    """
    17-dim rule signal vector from NLPPreprocessor:
      [0]  jailbreak_structural
      [1]  forced_agreement_phrase
      [2]  modal_imperative
      [3]  absolutist_quantifier
      [4]  imperative_urgency
      [5]  passive_agency_hiding
      [6]  factive_verb
      [7]  change_of_state
      [8]  definite_superlative
      [9]  cleft_construction
      [10] additive_particle
      [11] why_causal_presupposition
      [12] evaluative_word_count (0-5, normalised)
      [13] INTERROGATIVE
      [14] DECLARATIVE
      [15] IMPERATIVE
      [16] dep_triple_present
    """
    from mycelium.pipeline.layer0.nlp_preprocessor import get_preprocessor
    try:
        a = get_preprocessor().analyse(text)
    except Exception:
        return np.zeros(17, dtype=np.float32)

    v = np.zeros(17, dtype=np.float32)
    sigs  = set(a.coercive_signals)
    preps = set(a.presupposition_triggers)

    v[0]  = 1.0 if "jailbreak_structural"      in sigs  else 0.0
    v[1]  = 1.0 if "forced_agreement_phrase"   in sigs  else 0.0
    v[2]  = 1.0 if any("modal_imperative"      in s for s in sigs)  else 0.0
    v[3]  = 1.0 if any("absolutist_quantifier" in s for s in sigs)  else 0.0
    v[4]  = 1.0 if any("imperative_urgency"    in s for s in sigs)  else 0.0
    v[5]  = 1.0 if "passive_agency_hiding"     in sigs  else 0.0
    v[6]  = 1.0 if any("factive_verb"          in p for p in preps) else 0.0
    v[7]  = 1.0 if any("change_of_state"       in p for p in preps) else 0.0
    v[8]  = 1.0 if "definite_superlative"      in preps else 0.0
    v[9]  = 1.0 if "cleft_construction"        in preps else 0.0
    v[10] = 1.0 if "additive_particle:also"    in preps else 0.0
    v[11] = 1.0 if "why_causal_presupposition" in preps else 0.0
    v[12] = min(len(a.evaluative_words), 5) / 5.0
    v[13] = 1.0 if a.sentence_type == "INTERROGATIVE" else 0.0
    v[14] = 1.0 if a.sentence_type == "DECLARATIVE"   else 0.0
    v[15] = 1.0 if a.sentence_type == "IMPERATIVE"    else 0.0
    v[16] = 1.0 if a.dep_triples else 0.0
    return v


def _build_feature_matrix(
    texts: List[str],
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> np.ndarray:
    """Return (N, 401) feature matrix: 384-dim embedding + 17 rule signals."""
    from sentence_transformers import SentenceTransformer
    log.info(
        "[features] encoding %d texts with %s on device=%s ...",
        len(texts), embed_model_name, _DEVICE,
    )
    encoder = SentenceTransformer(embed_model_name, device=_DEVICE)
    embeddings = encoder.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    log.info("[features] building rule signal vectors ...")
    rule_vecs = np.array([_rule_signal_vector(t) for t in texts], dtype=np.float32)
    return np.hstack([embeddings, rule_vecs])


# ---------------------------------------------------------------------------
# Calibration eval helper
# ---------------------------------------------------------------------------

def _log_brier_score(
    clf, X: np.ndarray, y: np.ndarray, label_encoder, name: str
) -> None:
    from sklearn.metrics import brier_score_loss
    from sklearn.preprocessing import label_binarize
    try:
        classes = list(range(len(label_encoder.classes_)))
        proba = clf.predict_proba(X)
        Y_bin = label_binarize(y, classes=classes)
        if Y_bin.shape[1] == 1:
            Y_bin = np.hstack([1 - Y_bin, Y_bin])
        scores = [
            brier_score_loss(Y_bin[:, i], proba[:, i])
            for i in range(len(classes))
        ]
        mean_bs = float(np.mean(scores))
        per_class = ", ".join(
            f"{label_encoder.classes_[i]}={scores[i]:.4f}"
            for i in range(len(classes))
        )
        log.info(
            "[calibration] %s Brier score — mean=%.4f  per-class: %s",
            name, mean_bs, per_class,
        )
    except Exception as exc:
        log.warning("[calibration] Brier score computation failed for %s: %s", name, exc)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def _get_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            dev = "cuda"
            log.info("[device] CUDA available — using GPU: %s", torch.cuda.get_device_name(0))
            return dev
    except Exception:
        pass
    log.info("[device] CUDA not available — using CPU")
    return "cpu"


_DEVICE: str = _get_device()


# ---------------------------------------------------------------------------
# Model trainers
# ---------------------------------------------------------------------------

def train_manipulation_classifier(skip_if_exists: bool = False) -> Path:
    out_path = _MODELS_DIR / "manipulation_classifier.joblib"
    if skip_if_exists and out_path.exists():
        log.info("[train] manipulation_classifier already trained, skipping.")
        return out_path

    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import LabelEncoder
    from sklearn.calibration import CalibratedClassifierCV

    samples = _load_manipulation_data()
    if not samples:
        log.error("[train] No manipulation training data found.")
        return out_path

    n = len(samples)
    if n < _MIN_TRAIN_SAMPLES:
        log.error("[train] ManipulationClassifier: only %d samples — need at least %d.", n, _MIN_TRAIN_SAMPLES)
        return out_path

    texts  = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    le = LabelEncoder()
    le.fit(MANIP_LABELS)
    y = le.transform(labels)

    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        log.error("[train] ManipulationClassifier: only 1 class present.")
        return out_path

    X = _build_feature_matrix(texts)
    min_class_count = int(np.bincount(y).min())
    cv_folds = max(2, min(5, min_class_count))
    log.info("[train] ManipulationClassifier: X=%s  classes=%s  cv=%d", X.shape, le.classes_, cv_folds)
    clf = CalibratedClassifierCV(
        LinearSVC(C=1.0, max_iter=2000, class_weight="balanced"),
        method="isotonic",
        cv=cv_folds,
    )
    clf.fit(X, y)
    _log_brier_score(clf, X, y, le, "ManipulationClassifier")

    joblib.dump({"model": clf, "label_encoder": le, "version": "1.2"}, out_path)
    log.info("[train] Saved → %s", out_path)
    return out_path


def train_objectivity_classifier(skip_if_exists: bool = False) -> Path:
    out_path = _MODELS_DIR / "objectivity_classifier.joblib"
    if skip_if_exists and out_path.exists():
        log.info("[train] objectivity_classifier already trained, skipping.")
        return out_path

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    from sklearn.calibration import CalibratedClassifierCV

    samples = _load_objectivity_data()
    if not samples:
        log.error("[train] No objectivity training data found.")
        return out_path

    n = len(samples)
    if n < _MIN_TRAIN_SAMPLES:
        log.error("[train] ObjectivityClassifier: only %d samples — need at least %d.", n, _MIN_TRAIN_SAMPLES)
        return out_path

    texts  = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    le = LabelEncoder()
    le.fit(OBJ_LABELS)
    y = le.transform(labels)

    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        log.error(
            "[train] ObjectivityClassifier: only 1 class present: %s. "
            "Need ethics_qa.csv, anthropic_hh.csv, and liar_train.csv.",
            [OBJ_LABELS[c] for c in unique_classes.tolist()],
        )
        return out_path

    X = _build_feature_matrix(texts)
    min_class_count = int(np.bincount(y).min())
    cv_folds = max(2, min(5, min_class_count))
    log.info("[train] ObjectivityClassifier: X=%s  classes=%s  cv=%d", X.shape, le.classes_, cv_folds)
    clf = CalibratedClassifierCV(
        LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced", solver="lbfgs",
            multi_class="multinomial",
        ),
        method="isotonic",
        cv=cv_folds,
    )
    clf.fit(X, y)
    _log_brier_score(clf, X, y, le, "ObjectivityClassifier")

    joblib.dump({"model": clf, "label_encoder": le, "version": "1.2"}, out_path)
    log.info("[train] Saved → %s", out_path)
    return out_path


def train_assumption_typer(skip_if_exists: bool = False) -> Path:
    out_path = _MODELS_DIR / "assumption_typer.joblib"
    if skip_if_exists and out_path.exists():
        log.info("[train] assumption_typer already trained, skipping.")
        return out_path

    from sklearn.linear_model import LogisticRegression
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.calibration import CalibratedClassifierCV

    samples = _load_assumption_data()
    if not samples:
        log.error("[train] No assumption training data found.")
        return out_path

    texts  = [s[0] for s in samples]
    y_list = [s[1] for s in samples]
    Y = np.array(y_list, dtype=np.int32)

    active_cols = [i for i in range(Y.shape[1]) if Y[:, i].sum() > 0]
    if not active_cols:
        log.warning("[train] assumption_typer: no positive examples for any type, skipping.")
        return out_path

    active_types = [ASSUMPTION_TYPES[i] for i in active_cols]
    Y_active = Y[:, active_cols]

    X = _build_feature_matrix(texts)
    log.info("[train] AssumptionTyper: X=%s  active_types=%s", X.shape, active_types)
    clf = MultiOutputClassifier(
        CalibratedClassifierCV(
            LogisticRegression(C=1.0, max_iter=500, solver="lbfgs"),
            method="isotonic",
            cv=3,
        ),
        n_jobs=-1,
    )
    clf.fit(X, Y_active)

    joblib.dump({
        "model": clf,
        "active_types": active_types,
        "all_types": ASSUMPTION_TYPES,
        "version": "1.2",
    }, out_path)
    log.info("[train] Saved → %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Layer 0 classifiers for Mycelium.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["manipulation", "objectivity", "assumption", "all"],
        default=["all"],
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-if-exists", action="store_true")
    args = parser.parse_args()

    targets = args.models
    if "all" in targets:
        targets = ["manipulation", "objectivity", "assumption"]

    if not args.skip_download:
        pull_all_datasets()

    if "manipulation" in targets:
        train_manipulation_classifier(skip_if_exists=args.skip_if_exists)
    if "objectivity" in targets:
        train_objectivity_classifier(skip_if_exists=args.skip_if_exists)
    if "assumption" in targets:
        train_assumption_typer(skip_if_exists=args.skip_if_exists)

    log.info("[train] All done. Models in: %s", _MODELS_DIR)


if __name__ == "__main__":
    main()
