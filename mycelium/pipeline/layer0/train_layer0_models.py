"""
train_layer0_models.py
=======================
Offline trainer for the three Layer 0 sklearn classifiers.

    ManipulationClassifier  →  models/layer0/manipulation_classifier.joblib
    ObjectivityClassifier   →  models/layer0/objectivity_classifier.joblib
    AssumptionTyper         →  models/layer0/assumption_typer.joblib

Dataset sources (auto-downloaded on first run, skipped if already present):
    training_data/jailbreakbench/   — JailbreakBench jailbreak prompts
                                       (cloned from JailbreakBench/artifacts)
    training_data/liar_train.csv    — LIAR dataset
                                       (ucsbnlp/liar via direct Parquet fetch,
                                        bypasses broken liar.py loading script)
    training_data/trivia_qa.csv     — TriviaQA rc sample (HuggingFace datasets)
    training_data/ethics_qa.csv     — Hendrycks ETHICS commonsense
                                       (direct Parquet from
                                        lighteval/hendrycks_ethics mirror)
    training_data/anthropic_hh.csv  — Anthropic HH-RLHF harmless-base
                                       (streamed from train.jsonl.gz,
                                        VALUE_LADEN source)
    training_data/benign_prompts.jsonl — ~1000 normal questions (auto-generated
                                         from TriviaQA questions, relabelled)

Feature vector per sample:
    [sentence_embedding (384-dim, all-MiniLM-L6-v2)]
    + [rule_signal_vector (one-hot, ~17 dims)]

Model architecture:
    ManipulationClassifier  — CalibratedClassifierCV(
                                LinearSVC(C=1.0, balanced),
                                method='isotonic', cv=5)
                              Falls back to uncalibrated LinearSVC when any
                              class has fewer than 2*cv_folds samples.
    ObjectivityClassifier   — CalibratedClassifierCV(
                                LogisticRegression(C=1.0, balanced),
                                method='isotonic', cv=5)
                              Falls back to uncalibrated LR when any class
                              has fewer than 2*cv_folds samples.
    AssumptionTyper         — MultiOutputClassifier(
                                CalibratedClassifierCV(
                                  LogisticRegression, method='isotonic', cv=5))

Calibration notes:
    Isotonic regression is preferred over sigmoid (Platt) for:
      * LinearSVC: decision margins are not monotonically related to
        posterior probabilities, so sigmoid under-fits.
      * Minority classes (JAILBREAK_ATTEMPT, AMBIGUOUS): sigmoid tends to
        over-compress probability mass near 0.5; isotonic handles the
        irregular calibration curves better.
    Brier score is logged after each fit so calibration quality is
    visible at train time without a separate eval step.

    CalibratedClassifierCV silently drops classes that have zero samples in
    any CV fold, producing a model with fewer output columns than expected.
    _safe_calibrated_clf() prevents this by checking per-class counts before
    wrapping: if any class has fewer than 2*cv_folds samples it logs a WARNING
    and returns the base estimator unwrapped (sklearn's predict_proba is still
    available via the solver, just uncalibrated).

Device selection:
    SentenceTransformer encoding runs on CUDA when torch.cuda.is_available(),
    otherwise falls back to CPU.  All sklearn estimators are CPU-only.

Usage:
    python -m mycelium.pipeline.layer0.train_layer0_models
    python -m mycelium.pipeline.layer0.train_layer0_models --models manipulation
    python -m mycelium.pipeline.layer0.train_layer0_models --skip-download
"""

# __future__ imports MUST be first executable statement after the module docstring
from __future__ import annotations

import os as _os
import sys as _sys


def _set_hf_cache() -> None:
    try:
        from mycelium.pipeline.config_loader import hf_cache_dir
        _os.environ.setdefault("HF_HOME", hf_cache_dir())
    except Exception:
        _root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
        _os.environ.setdefault("HF_HOME", _os.path.join(_root, "hf_cache"))


_set_hf_cache()

import argparse
import csv
import gzip
import json
import logging
import re
import subprocess
import sys
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np

# Raise the CSV field-size limit once at import time so TriviaQA's large
# Wikipedia-passage context fields never hit the 128 KB default cap.
csv.field_size_limit(sys.maxsize)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent          # layer0/
_ROOT = _HERE.parent.parent.parent               # project root
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
# Regex heuristics for AssumptionTyper auto-labelling
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
                prompt = (entry.get("prompt") or entry.get("goal") or "").strip()
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
    ok = _fetch_parquet(_LIAR_TRAIN_PARQUET_URL, dest, "LIAR")
    if not ok:
        log.warning("[download] LIAR Parquet fetch failed — liar_train.csv will be absent.")
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

    log.info("[download] Fetching Anthropic HH-RLHF jsonl.gz ...")
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
# Dataset loaders → (text, label) lists
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
    else:
        log.warning("[manip] jailbreakbench_train.csv not found — re-run without --skip-download")
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
    else:
        log.warning("[manip] liar_train.csv not found — re-run without --skip-download")
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
    else:
        log.warning("[manip] trivia_qa.csv not found — re-run without --skip-download")
    log.info("[manip] trivia_qa: %d samples", tqa_count)

    _append_from_logger(samples, "manipulation")
    log.info("[manip] total samples: %d", len(samples))
    return samples


def _load_objectivity_data() -> List[Tuple[str, str]]:
    """
    Returns (text, obj_label) pairs with a balanced class distribution.

    Target distribution:
      OBJECTIVE   → TriviaQA questions (factual, unambiguous)
      VALUE_LADEN → ethics_qa + anthropic_hh, capped to match OBJECTIVE count
      SUBJECTIVE  → LIAR statements, capped
      AMBIGUOUS   → auto-generated hedged questions from TriviaQA stems

    Caps prevent VALUE_LADEN (the largest raw source) from dominating and
    causing the classifier to label all physics questions as VALUE_LADEN.
    """
    samples: List[Tuple[str, str]] = []
    objective_texts: List[str] = []

    # --- OBJECTIVE: TriviaQA questions ---
    tqa_path = _DATA_DIR / "trivia_qa.csv"
    tqa_count = 0
    if tqa_path.exists():
        try:
            with open(tqa_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if i >= 2000:
                        break
                    q = row.get("question", "").strip()
                    if q:
                        samples.append((q, "OBJECTIVE"))
                        objective_texts.append(q)
                        tqa_count += 1
        except Exception as exc:
            log.warning("[obj] TriviaQA load error: %s", exc)
    else:
        log.warning("[obj] trivia_qa.csv not found — re-run without --skip-download")
    log.info("[obj] trivia_qa (OBJECTIVE): %d samples", tqa_count)

    obj_count = tqa_count  # use actual OBJECTIVE count as the cap
    value_laden_cap = max(obj_count, 500)  # at least 500 even if TriviaQA is empty

    # --- VALUE_LADEN: ethics_qa (capped) ---
    ethics_path = _DATA_DIR / "ethics_qa.csv"
    ethics_count = 0
    if ethics_path.exists():
        try:
            with open(ethics_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if ethics_count >= min(1500, value_laden_cap // 2):
                        break
                    inp = (
                        row.get("input")
                        or row.get("sentence")
                        or row.get("text")
                        or ""
                    ).strip()
                    if inp:
                        samples.append((inp, "VALUE_LADEN"))
                        ethics_count += 1
        except Exception as exc:
            log.warning("[obj] Ethics load error: %s", exc)
    else:
        log.warning("[obj] ethics_qa.csv not found — re-run without --skip-download")
    log.info("[obj] ethics_qa (VALUE_LADEN): %d samples", ethics_count)

    # --- VALUE_LADEN: anthropic_hh (capped to fill remainder up to value_laden_cap) ---
    hh_path = _DATA_DIR / "anthropic_hh.csv"
    hh_count = 0
    hh_cap = max(0, value_laden_cap - ethics_count)
    if hh_path.exists():
        try:
            with open(hh_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if hh_count >= min(500, hh_cap):
                        break
                    text = row.get("text", "").strip()
                    if text:
                        samples.append((text, "VALUE_LADEN"))
                        hh_count += 1
        except Exception as exc:
            log.warning("[obj] Anthropic HH load error: %s", exc)
    else:
        log.warning("[obj] anthropic_hh.csv not found — re-run without --skip-download")
    log.info("[obj] anthropic_hh (VALUE_LADEN): %d samples (cap=%d)", hh_count, min(500, hh_cap))

    # --- SUBJECTIVE: LIAR (capped) ---
    liar_path = _DATA_DIR / "liar_train.csv"
    liar_count = 0
    if liar_path.exists():
        try:
            with open(liar_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if i >= 800:
                        break
                    stmt = row.get("statement", "").strip()
                    if stmt:
                        samples.append((stmt, _LIAR_OBJ_LABEL))
                        liar_count += 1
        except Exception as exc:
            log.warning("[obj] LIAR load error: %s", exc)
    else:
        log.warning("[obj] liar_train.csv not found — re-run without --skip-download")
    log.info("[obj] liar (SUBJECTIVE): %d samples", liar_count)

    # --- AMBIGUOUS: auto-generate hedged variants of TriviaQA stems ---
    # These are questions like "Could it be that X?", "Is it possible that X?"
    # which are the exact forms that were being misclassified.
    ambiguous_count = 0
    _HEDGES = [
        "Could it be that {q}?",
        "Is it possible that {q}?",
        "Might it be the case that {q}?",
        "Some people think {q} — is this actually true?",
        "Would you say that {q}?",
        "Do you think {q}?",
        "Is there any chance that {q}?",
        "Can we say for certain that {q}?",
    ]
    ambig_target = min(400, obj_count // 5)  # ~20% of OBJECTIVE, up to 400
    for i, q in enumerate(objective_texts):
        if ambiguous_count >= ambig_target:
            break
        # Strip trailing "?" and lowercase for clean stem
        stem = q.rstrip("?").strip()
        hedge = _HEDGES[i % len(_HEDGES)].format(q=stem)
        samples.append((hedge, "AMBIGUOUS"))
        ambiguous_count += 1
    log.info("[obj] auto-labelled AMBIGUOUS: %d samples", ambiguous_count)

    # Log final distribution
    dist = Counter(lbl for _, lbl in samples)
    log.info("[obj] final distribution: %s  total=%d", dict(dist), len(samples))

    _append_from_logger(samples, "objectivity")
    log.info("[obj] total samples: %d", len(samples))
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
    else:
        log.warning("[assumption] liar_train.csv not found")

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
    Build a sparse one-hot feature vector from rule signals extracted
    by NLPPreprocessor.  Used alongside the sentence embedding.
    Dimensions (17):
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
    local_files_only: bool = False,
) -> np.ndarray:
    """Return (N, 401) feature matrix: 384-dim embedding + 17 rule signals.

    When *local_files_only* is True (i.e. ``--skip-download`` was passed),
    SentenceTransformer is initialised with ``local_files_only=True`` so it
    never attempts to reach huggingface.co and instead loads the model
    straight from the local HF cache.  This avoids spurious
    "couldn't connect to huggingface.co" warnings during offline re-training.
    """
    from sentence_transformers import SentenceTransformer
    log.info(
        "[features] encoding %d texts with %s on device=%s%s ...",
        len(texts), embed_model_name, _DEVICE,
        " (local_files_only)" if local_files_only else "",
    )
    encoder = SentenceTransformer(
        embed_model_name,
        device=_DEVICE,
        local_files_only=local_files_only,
    )
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
# Calibration helpers
# ---------------------------------------------------------------------------

def _safe_calibrated_clf(base_estimator, y: np.ndarray, cv_folds: int, method: str = "isotonic"):
    """
    Wrap *base_estimator* in CalibratedClassifierCV only when every class
    has at least ``2 * cv_folds`` samples — the minimum required for each
    fold to contain at least 2 examples of that class.

    When the condition is not met, CalibratedClassifierCV silently drops the
    minority class from the calibrated model's output, producing a model
    whose predict_proba has fewer columns than there are known classes.  This
    causes downstream IndexErrors (e.g. in Brier score computation and in the
    DST mass-function builder).

    If any class is below the threshold, logs a WARNING and returns the base
    estimator unwrapped (sklearn's predict_proba is still available via the
    solver, just uncalibrated).
    """
    from sklearn.calibration import CalibratedClassifierCV
    counts = Counter(y)
    min_count = min(counts.values())
    threshold = 2 * cv_folds
    if min_count < threshold:
        log.warning(
            "_safe_calibrated_clf: class %r has only %d samples "
            "(need %d for cv=%d) — skipping calibration wrapper",
            min(counts, key=counts.get), min_count, threshold, cv_folds,
        )
        return base_estimator
    return CalibratedClassifierCV(base_estimator, method=method, cv=cv_folds)


# ---------------------------------------------------------------------------
# Model trainers
# ---------------------------------------------------------------------------

def train_manipulation_classifier(X: np.ndarray, y: np.ndarray):
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    base = LinearSVC(C=1.0, class_weight="balanced", max_iter=2000)
    clf = _safe_calibrated_clf(base, y_enc, cv_folds=5)
    clf.fit(X, y_enc)

    # Brier score (macro-average over classes)
    try:
        from sklearn.metrics import brier_score_loss
        proba = clf.predict_proba(X)
        scores = []
        for i, cls in enumerate(le.classes_):
            scores.append(brier_score_loss((y_enc == i).astype(int), proba[:, i]))
        log.info("[manip] Brier score (macro): %.4f", float(np.mean(scores)))
    except Exception as exc:
        log.warning("[manip] Brier score failed: %s", exc)

    return clf, le


def train_objectivity_classifier(X: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # multi_class was removed in scikit-learn 1.5; lbfgs is inherently
    # multinomial for multi-class problems so the argument was always a no-op.
    base = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, solver="lbfgs")
    clf = _safe_calibrated_clf(base, y_enc, cv_folds=5)
    clf.fit(X, y_enc)

    try:
        from sklearn.metrics import brier_score_loss
        proba = clf.predict_proba(X)
        scores = []
        for i in range(len(le.classes_)):
            scores.append(brier_score_loss((y_enc == i).astype(int), proba[:, i]))
        log.info("[obj] Brier score (macro): %.4f", float(np.mean(scores)))
    except Exception as exc:
        log.warning("[obj] Brier score failed: %s", exc)

    return clf, le


def train_assumption_typer(X: np.ndarray, Y: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    from sklearn.multioutput import MultiOutputClassifier

    base = LogisticRegression(C=1.0, class_weight="balanced", max_iter=500, solver="lbfgs")
    clf = MultiOutputClassifier(base, n_jobs=-1)
    clf.fit(X, Y)
    return clf


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def _save_model(obj, path: Path, label: str) -> None:
    joblib.dump(obj, path)
    log.info("[save] %s → %s", label, path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Layer 0 classifiers")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["manipulation", "objectivity", "assumption", "all"],
        default=["all"],
        help="Which model(s) to train (default: all)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip dataset downloads (use cached data only)",
    )
    args = parser.parse_args()

    train_all = "all" in args.models
    do_manip  = train_all or "manipulation" in args.models
    do_obj    = train_all or "objectivity"  in args.models
    do_assump = train_all or "assumption"   in args.models

    if not args.skip_download:
        log.info("=== Downloading datasets ===")
        pull_all_datasets()
    else:
        log.info("=== Skipping dataset download (--skip-download) ===")

    if do_manip:
        log.info("=== Training ManipulationClassifier ===")
        manip_data = _load_manipulation_data()
        if len(manip_data) < _MIN_TRAIN_SAMPLES:
            log.error("[manip] Not enough training data (%d samples). Aborting.", len(manip_data))
        else:
            texts, labels = zip(*manip_data)
            X = _build_feature_matrix(list(texts), local_files_only=args.skip_download)
            y = np.array(labels)
            clf, le = train_manipulation_classifier(X, y)
            _save_model(clf, _MODELS_DIR / "manipulation_classifier.joblib", "ManipulationClassifier")
            _save_model(le,  _MODELS_DIR / "manipulation_label_encoder.joblib", "ManipulationLabelEncoder")
            log.info("[manip] classes: %s", list(le.classes_))

    if do_obj:
        log.info("=== Training ObjectivityClassifier ===")
        obj_data = _load_objectivity_data()
        if len(obj_data) < _MIN_TRAIN_SAMPLES:
            log.error("[obj] Not enough training data (%d samples). Aborting.", len(obj_data))
        else:
            texts, labels = zip(*obj_data)
            X = _build_feature_matrix(list(texts), local_files_only=args.skip_download)
            y = np.array(labels)
            clf, le = train_objectivity_classifier(X, y)
            _save_model(clf, _MODELS_DIR / "objectivity_classifier.joblib", "ObjectivityClassifier")
            _save_model(le,  _MODELS_DIR / "objectivity_label_encoder.joblib", "ObjectivityLabelEncoder")
            log.info("[obj] classes: %s", list(le.classes_))

    if do_assump:
        log.info("=== Training AssumptionTyper ===")
        assump_data = _load_assumption_data()
        if len(assump_data) < _MIN_TRAIN_SAMPLES:
            log.error("[assumption] Not enough training data (%d samples). Aborting.", len(assump_data))
        else:
            texts, label_vecs = zip(*assump_data)
            X = _build_feature_matrix(list(texts), local_files_only=args.skip_download)
            Y = np.array(label_vecs, dtype=int)
            clf = train_assumption_typer(X, Y)
            _save_model(clf, _MODELS_DIR / "assumption_typer.joblib", "AssumptionTyper")

    log.info("=== Done ===")


if __name__ == "__main__":
    main()
