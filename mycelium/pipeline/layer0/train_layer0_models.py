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
    training_data/liar_train.csv    — LIAR dataset (ucsbnlp/liar, default/ shard)
    training_data/trivia_qa.csv     — TriviaQA rc sample (HuggingFace datasets)
    training_data/ethics_qa.csv     — Hendrycks ETHICS commonsense (raw GitHub CSV)
    training_data/benign_prompts.jsonl — ~1000 normal questions (auto-generated
                                         from TriviaQA questions, relabelled)

Feature vector per sample:
    [sentence_embedding (384-dim, all-MiniLM-L6-v2)]
    + [rule_signal_vector (one-hot, ~30 dims)]

Model architecture:
    ManipulationClassifier  — LinearSVC (multi-class, C=1.0)
    ObjectivityClassifier   — LogisticRegression (multi-class, C=1.0)
    AssumptionTyper         — MultiOutputClassifier(LogisticRegression)
                               (multi-label: one binary clf per assumption type)

Device selection:
    SentenceTransformer encoding runs on CUDA when torch.cuda.is_available(),
    otherwise falls back to CPU.  All sklearn estimators are CPU-only and are
    not affected by device selection.

Usage:
    python -m mycelium.pipeline.layer0.train_layer0_models
    python -m mycelium.pipeline.layer0.train_layer0_models --models manipulation
    python -m mycelium.pipeline.layer0.train_layer0_models --skip-download
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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

_HERE = Path(__file__).resolve().parent                    # layer0/
_ROOT = _HERE.parent.parent.parent                         # project root
_DATA_DIR   = _ROOT / "training_data"
_MODELS_DIR = _ROOT / "models" / "layer0"

_DATA_DIR.mkdir(parents=True, exist_ok=True)
_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Minimum samples required before attempting to fit a classifier.
_MIN_TRAIN_SAMPLES = 50

# ---------------------------------------------------------------------------
# Device selection — resolved once at import time
# ---------------------------------------------------------------------------

def _get_device() -> str:
    """Return 'cuda' when a CUDA-capable GPU is available, else 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available():
            dev = "cuda"
            log.info(
                "[device] CUDA available — using GPU: %s",
                torch.cuda.get_device_name(0),
            )
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
# Dataset download helpers
# ---------------------------------------------------------------------------

def _pull_jailbreakbench() -> Path:
    """
    Clone JailbreakBench/artifacts — this is the repo that actually contains
    the adversarial prompt CSVs, not the main jailbreakbench repo.
    """
    dest = _DATA_DIR / "jailbreakbench"
    if dest.exists() and any(dest.iterdir()):
        log.info("[download] jailbreakbench already present, skipping.")
        return dest
    log.info("[download] Cloning JailbreakBench/artifacts ...")
    subprocess.run(
        ["git", "clone", "--depth=1",
         "https://github.com/JailbreakBench/artifacts",
         str(dest)],
        check=True,
    )
    return dest


def _pull_hf_dataset(hf_name: str, config: str, split: str, dest: Path, **kwargs) -> Path:
    """Download a HuggingFace dataset split to a CSV if not already present."""
    if dest.exists():
        log.info("[download] %s already present, skipping.", dest.name)
        return dest
    log.info("[download] Pulling %s (%s) ...", hf_name, split)
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset(hf_name, config, split=split, **kwargs)
        ds.to_csv(str(dest))
        log.info("[download] Saved %d rows to %s", len(ds), dest)
    except Exception as exc:
        log.warning("[download] Failed to pull %s: %s", hf_name, exc)
    return dest


def _pull_liar() -> Path:
    """
    Download the LIAR train split directly from Parquet without invoking
    any legacy dataset script.

    ucsbnlp/liar stores its Parquet shards under the 'default/' subdirectory
    (not 'data/').  We pass the resolved URL directly via data_files= so
    HuggingFace datasets never invokes liar.py.
    """
    dest = _DATA_DIR / "liar_train.csv"
    if dest.exists():
        log.info("[download] liar_train.csv already present, skipping.")
        return dest
    log.info("[download] Pulling LIAR via direct Parquet (default/ shard) ...")
    try:
        from datasets import load_dataset  # type: ignore
        parquet_url = (
            "https://huggingface.co/datasets/ucsbnlp/liar/resolve/main/"
            "default/train-00000-of-00001.parquet"
        )
        ds = load_dataset("parquet", data_files={"train": parquet_url}, split="train")
        ds.to_csv(str(dest))
        log.info("[download] LIAR saved %d rows to %s", len(ds), dest)
    except Exception as exc:
        log.warning("[download] Failed to pull LIAR: %s", exc)
    return dest


def _pull_ethics() -> Path:
    """
    Download the Hendrycks ETHICS commonsense split from the raw GitHub repo.

    hendrycks/ethics on HuggingFace uses a legacy ethics.py script that
    datasets>=2.21 refuses to execute.  The identical CSV lives in the
    GitHub source repo at hendrycks/ethics and is freely accessible.
    We fetch it directly with requests (no datasets library needed).
    """
    dest = _DATA_DIR / "ethics_qa.csv"
    if dest.exists():
        log.info("[download] ethics_qa.csv already present, skipping.")
        return dest
    log.info("[download] Pulling ETHICS commonsense from GitHub raw ...")
    try:
        import requests  # type: ignore
        url = (
            "https://raw.githubusercontent.com/hendrycks/ethics/master/"
            "commonsense/cm_train.csv"
        )
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        # Count rows for confirmation
        lines = resp.text.count("\n")
        log.info("[download] ETHICS saved ~%d rows to %s", lines, dest)
    except Exception as exc:
        log.warning("[download] Failed to pull ETHICS: %s", exc)
    return dest


def pull_all_datasets() -> None:
    _pull_jailbreakbench()
    _pull_liar()
    _pull_hf_dataset(
        "trivia_qa", "rc", "train[:3000]",
        _DATA_DIR / "trivia_qa.csv",
    )
    _pull_ethics()

# ---------------------------------------------------------------------------
# Dataset loaders → (text, label) lists
# ---------------------------------------------------------------------------

def _load_manipulation_data() -> List[Tuple[str, str]]:
    """
    Returns (text, manip_label) pairs from:
      - JailbreakBench/artifacts  → JAILBREAK_ATTEMPT
      - LIAR politifact lies → COERCIVE
      - TriviaQA questions → NOT_MANIPULATIVE
      - dataset_logger JSONL (live labelled, if present)
    """
    samples: List[Tuple[str, str]] = []

    # --- JailbreakBench/artifacts: walk all .csv files recursively
    jbb_root = _DATA_DIR / "jailbreakbench"
    jbb_count = 0
    if jbb_root.exists():
        csv_files = list(jbb_root.rglob("*.csv"))
        if not csv_files:
            log.warning(
                "[manip] No CSV files found under %s. "
                "The artifacts repo may have changed structure.",
                jbb_root,
            )
        for csv_path in csv_files:
            try:
                with open(csv_path, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        goal = (
                            row.get("Goal")
                            or row.get("goal")
                            or row.get("prompt")
                            or row.get("jailbreak_prompt")
                            or ""
                        )
                        if goal.strip():
                            samples.append((goal.strip(), "JAILBREAK_ATTEMPT"))
                            jbb_count += 1
            except Exception:
                pass
    else:
        log.warning("[manip] JailbreakBench dir not found: %s", jbb_root)
    log.info("[manip] jailbreakbench: %d samples", jbb_count)

    # --- LIAR: map label to manipulation category
    liar_path = _DATA_DIR / "liar_train.csv"
    liar_count = 0
    if liar_path.exists():
        try:
            with open(liar_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    stmt = row.get("statement", "").strip()
                    lbl  = row.get("label", "").strip().lower()
                    if not stmt:
                        continue
                    if lbl in {"pants-fire", "false"}:
                        samples.append((stmt, "COERCIVE"))
                        liar_count += 1
                    elif lbl in {"half-true", "mostly-true", "true"}:
                        samples.append((stmt, "NOT_MANIPULATIVE"))
                        liar_count += 1
        except Exception as exc:
            log.warning("[manip] LIAR load error: %s", exc)
    else:
        log.warning("[manip] liar_train.csv not found — re-run without --skip-download")
    log.info("[manip] liar: %d samples", liar_count)

    # --- TriviaQA: factual questions → NOT_MANIPULATIVE
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

    # --- dataset_logger live JSONL (manipulation component)
    _append_from_logger(samples, "manipulation")

    log.info("[manip] total samples: %d", len(samples))
    return samples


def _load_objectivity_data() -> List[Tuple[str, str]]:
    """
    Returns (text, obj_label) pairs from:
      - TriviaQA factual questions → OBJECTIVE
      - Ethics QA → VALUE_LADEN
      - LIAR subjective statements → SUBJECTIVE
    """
    samples: List[Tuple[str, str]] = []

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
                        tqa_count += 1
        except Exception as exc:
            log.warning("[obj] TriviaQA load error: %s", exc)
    else:
        log.warning("[obj] trivia_qa.csv not found — re-run without --skip-download")
    log.info("[obj] trivia_qa: %d samples", tqa_count)

    ethics_path = _DATA_DIR / "ethics_qa.csv"
    ethics_count = 0
    if ethics_path.exists():
        try:
            with open(ethics_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # GitHub raw CSV has columns: label, input
                    inp = row.get("input", "").strip()
                    if inp:
                        samples.append((inp, "VALUE_LADEN"))
                        ethics_count += 1
        except Exception as exc:
            log.warning("[obj] Ethics load error: %s", exc)
    else:
        log.warning("[obj] ethics_qa.csv not found — re-run without --skip-download")
    log.info("[obj] ethics_qa: %d samples", ethics_count)

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
                        samples.append((stmt, "SUBJECTIVE"))
                        liar_count += 1
        except Exception as exc:
            log.warning("[obj] LIAR load error: %s", exc)
    else:
        log.warning("[obj] liar_train.csv not found — re-run without --skip-download")
    log.info("[obj] liar: %d samples", liar_count)

    _append_from_logger(samples, "objectivity")
    log.info("[obj] total samples: %d", len(samples))
    return samples


def _load_assumption_data() -> List[Tuple[str, List[int]]]:
    """
    Returns (text, multi_hot_vector) pairs.
    Multi-hot vector has len == len(ASSUMPTION_TYPES).
    Sources:
      - TriviaQA factual → mostly empty vector (no assumptions)
      - Ethics QA → VALUE_FRAME + NORMATIVE_UNIVERSAL flags
      - dataset_logger JSONL (assumption component)
    Auto-labels using rule signals from NLPPreprocessor.
    """
    from mycelium.pipeline.layer0.nlp_preprocessor import get_preprocessor
    pre = get_preprocessor()
    samples: List[Tuple[str, List[int]]] = []

    def _auto_label(text: str) -> List[int]:
        """Derive a multi-hot assumption vector from rule signals."""
        try:
            a = pre.analyse(text)
        except Exception:
            return [0] * len(ASSUMPTION_TYPES)
        vec = [0] * len(ASSUMPTION_TYPES)
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
                    inp = row.get("input", "").strip()
                    if inp:
                        samples.append((inp, _auto_label(inp)))
        except Exception as exc:
            log.warning("[assumption] Ethics load error: %s", exc)

    # dataset_logger: load assumption entries
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
    """Append live-labelled entries from dataset_log.jsonl if present."""
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
    Dimensions (~30):
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

    v[0]  = 1.0 if "jailbreak_structural"   in sigs  else 0.0
    v[1]  = 1.0 if "forced_agreement_phrase" in sigs  else 0.0
    v[2]  = 1.0 if any("modal_imperative"   in s for s in sigs)   else 0.0
    v[3]  = 1.0 if any("absolutist_quantifier" in s for s in sigs) else 0.0
    v[4]  = 1.0 if any("imperative_urgency" in s for s in sigs)   else 0.0
    v[5]  = 1.0 if "passive_agency_hiding"  in sigs  else 0.0
    v[6]  = 1.0 if any("factive_verb"        in p for p in preps) else 0.0
    v[7]  = 1.0 if any("change_of_state"     in p for p in preps) else 0.0
    v[8]  = 1.0 if "definite_superlative"   in preps else 0.0
    v[9]  = 1.0 if "cleft_construction"     in preps else 0.0
    v[10] = 1.0 if "additive_particle:also" in preps else 0.0
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
    """Return (N, 384+17) feature matrix."""
    from sentence_transformers import SentenceTransformer  # type: ignore
    log.info(
        "[features] encoding %d texts with %s on device=%s ...",
        len(texts),
        embed_model_name,
        _DEVICE,
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
# Model trainers
# ---------------------------------------------------------------------------

def train_manipulation_classifier(skip_if_exists: bool = False) -> Path:
    out_path = _MODELS_DIR / "manipulation_classifier.joblib"
    if skip_if_exists and out_path.exists():
        log.info("[train] manipulation_classifier already trained, skipping.")
        return out_path

    from sklearn.svm import LinearSVC          # type: ignore
    from sklearn.preprocessing import LabelEncoder  # type: ignore
    from sklearn.calibration import CalibratedClassifierCV  # type: ignore

    samples = _load_manipulation_data()
    if not samples:
        log.error("[train] No manipulation training data found. Run with dataset pull first.")
        return out_path

    n = len(samples)
    if n < _MIN_TRAIN_SAMPLES:
        log.error(
            "[train] ManipulationClassifier: only %d samples loaded — need at least %d.\n"
            "        Check the WARNING lines above to see which source CSVs are missing.\n"
            "        Delete any incomplete CSVs in training_data/ and re-run without "
            "--skip-download.",
            n, _MIN_TRAIN_SAMPLES,
        )
        return out_path

    texts  = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    le = LabelEncoder()
    le.fit(MANIP_LABELS)
    y = le.transform(labels)

    # Guard: need at least 2 distinct classes in the actual loaded data.
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        loaded_label_names = [MANIP_LABELS[c] for c in unique_classes.tolist()]
        log.error(
            "[train] ManipulationClassifier: only 1 class present in loaded data: %s.\n"
            "        COERCIVE needs liar_train.csv and JAILBREAK_ATTEMPT needs "
            "jailbreakbench CSVs.\n"
            "        Delete stale CSVs and re-run without --skip-download.",
            loaded_label_names,
        )
        return out_path

    X = _build_feature_matrix(texts)

    cv_folds = min(5, n)
    log.info("[train] ManipulationClassifier: X=%s  classes=%s  cv=%d", X.shape, le.classes_, cv_folds)
    clf = CalibratedClassifierCV(
        LinearSVC(C=1.0, max_iter=2000, class_weight="balanced"),
        cv=cv_folds,
    )
    clf.fit(X, y)

    artifact = {"model": clf, "label_encoder": le, "version": "1.0"}
    joblib.dump(artifact, out_path)
    log.info("[train] Saved → %s", out_path)
    return out_path


def train_objectivity_classifier(skip_if_exists: bool = False) -> Path:
    out_path = _MODELS_DIR / "objectivity_classifier.joblib"
    if skip_if_exists and out_path.exists():
        log.info("[train] objectivity_classifier already trained, skipping.")
        return out_path

    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.preprocessing import LabelEncoder       # type: ignore

    samples = _load_objectivity_data()
    if not samples:
        log.error("[train] No objectivity training data found.")
        return out_path

    n = len(samples)
    if n < _MIN_TRAIN_SAMPLES:
        log.error(
            "[train] ObjectivityClassifier: only %d samples loaded — need at least %d.\n"
            "        Check the WARNING lines above to see which source CSVs are missing.\n"
            "        Delete any incomplete CSVs in training_data/ and re-run without "
            "--skip-download.",
            n, _MIN_TRAIN_SAMPLES,
        )
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
            "Need ethics_qa.csv and liar_train.csv.",
            [OBJ_LABELS[c] for c in unique_classes.tolist()],
        )
        return out_path

    X = _build_feature_matrix(texts)

    log.info("[train] ObjectivityClassifier: X=%s  classes=%s", X.shape, le.classes_)
    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                              multi_class="multinomial", solver="lbfgs")
    clf.fit(X, y)

    artifact = {"model": clf, "label_encoder": le, "version": "1.0"}
    joblib.dump(artifact, out_path)
    log.info("[train] Saved → %s", out_path)
    return out_path


def train_assumption_typer(skip_if_exists: bool = False) -> Path:
    out_path = _MODELS_DIR / "assumption_typer.joblib"
    if skip_if_exists and out_path.exists():
        log.info("[train] assumption_typer already trained, skipping.")
        return out_path

    from sklearn.linear_model import LogisticRegression   # type: ignore
    from sklearn.multioutput import MultiOutputClassifier  # type: ignore

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
    base_clf = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
    clf = MultiOutputClassifier(base_clf, n_jobs=-1)
    clf.fit(X, Y_active)

    artifact = {
        "model": clf,
        "active_types": active_types,
        "all_types": ASSUMPTION_TYPES,
        "version": "1.0",
    }
    joblib.dump(artifact, out_path)
    log.info("[train] Saved → %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Layer 0 classifiers for Mycelium."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["manipulation", "objectivity", "assumption", "all"],
        default=["all"],
        help="Which classifiers to train (default: all).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip dataset download step.",
    )
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="Skip training if model file already present.",
    )
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
