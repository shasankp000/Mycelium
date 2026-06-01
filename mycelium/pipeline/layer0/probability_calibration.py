"""
probability_calibration.py
===========================
Post-hoc sklearn classifier probability calibration for Layer 0 models.

This is the module that ``calibration_api.py`` was misleadingly named
after.  It actually does calibrate sklearn classifiers — specifically
the three Layer 0 joblib artifacts stored in ``models/layer0/``.

Background
----------
The Layer 0 classifiers (ManipulationClassifier, ObjectivityClassifier,
AssumptionTyper) output ``predict_proba`` scores that feed directly into
DST (Dempster-Shafer Theory) mass function construction in the fusion
engine.  If those probability scores are poorly calibrated the mass
functions become unreliable regardless of how good the DST fusion logic
is:

  * Over-confident scores (e.g. SVM decision margins mapped to 0.95+
    for ambiguous inputs) push the DST belief mass to near-certainty,
    causing downstream routes to behave as if the classifier is far
    more certain than it really is.
  * Under-confident scores flatten the mass functions, making the fusion
    engine indifferent between competing hypotheses.

Calibration method
------------------
Isotonic regression is used (``CalibratedClassifierCV(method='isotonic')
``) rather than Platt scaling (``method='sigmoid'``) because:

  1. LinearSVC decision margins are not monotone in posterior probability,
     so sigmoid consistently under-estimates extreme class probabilities.
  2. The minority classes (JAILBREAK_ATTEMPT, AMBIGUOUS) have irregular
     calibration curves that isotonic handles better than sigmoid's
     monotone restriction.
  3. Dataset sizes are large enough (>500 samples per class after the
     LIAR + JailbreakBench + TriviaQA combination) that isotonic
     regression doesn't overfit.

Note: as of train_layer0_models.py v1.1, all three classifiers are
already trained with CalibratedClassifierCV(method='isotonic', cv=5)
inside the training loop.  This module exists for two additional
use-cases:

  1. **Re-calibration on new data** without full retraining — useful
     when the held-out calibration set grows but the base model is
     frozen (e.g. it took 4 hours to train).
  2. **Calibration evaluation** — ECE and Brier score computation plus
     reliability diagram data so calibration quality can be measured,
     logged, and compared across model versions.

Public API
----------
  calibrate_artifact(name, calibration_texts, calibration_labels)
      Load a .joblib artifact, wrap its inner estimator with
      CalibratedClassifierCV, fit on the supplied calibration set, and
      save the updated artifact back in-place.

  evaluate_calibration(name, texts, labels) -> CalibrationReport
      Load a .joblib artifact, run predict_proba, compute per-class
      Brier score and ECE, return a CalibrationReport with reliability
      diagram data.

CLI
---
  python -m mycelium.pipeline.layer0.probability_calibration --model manipulation
  python -m mycelium.pipeline.layer0.probability_calibration --model objectivity --eval-only
  python -m mycelium.pipeline.layer0.probability_calibration --model all
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

_HERE  = Path(__file__).resolve().parent          # layer0/
_ROOT  = _HERE.parent.parent.parent               # project root
_MODELS_DIR = _ROOT / "models" / "layer0"

_ARTIFACT_NAMES = {
    "manipulation": "manipulation_classifier.joblib",
    "objectivity":  "objectivity_classifier.joblib",
    "assumption":   "assumption_typer.joblib",
}


# ---------------------------------------------------------------------------
# Calibration report
# ---------------------------------------------------------------------------

@dataclass
class ClassCalibrationStats:
    """Per-class calibration statistics."""
    label: str
    brier_score: float
    # Reliability diagram: list of (mean_predicted_prob, fraction_positive)
    # bucketed into n_bins equal-width bins.
    reliability_diagram: List[Tuple[float, float]] = field(default_factory=list)
    ece: float = 0.0   # Expected Calibration Error for this class


@dataclass
class CalibrationReport:
    """Calibration evaluation results for one model artifact."""
    model_name: str
    n_samples: int
    mean_brier_score: float
    mean_ece: float
    per_class: List[ClassCalibrationStats] = field(default_factory=list)

    def log_summary(self) -> None:
        log.info(
            "[calibration] %s — n=%d  mean_brier=%.4f  mean_ece=%.4f",
            self.model_name, self.n_samples,
            self.mean_brier_score, self.mean_ece,
        )
        for cs in self.per_class:
            log.info(
                "  class=%-30s  brier=%.4f  ece=%.4f",
                cs.label, cs.brier_score, cs.ece,
            )


# ---------------------------------------------------------------------------
# Feature helpers (thin wrappers around train_layer0_models helpers)
# ---------------------------------------------------------------------------

def _build_X(texts: List[str]) -> np.ndarray:
    """Build the (N, 401) feature matrix used by all three classifiers."""
    from mycelium.pipeline.layer0.train_layer0_models import (
        _build_feature_matrix,
    )
    return _build_feature_matrix(texts)


# ---------------------------------------------------------------------------
# Reliability diagram + ECE helpers
# ---------------------------------------------------------------------------

def _reliability_diagram(
    y_true_bin: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[List[Tuple[float, float]], float]:
    """
    Compute reliability diagram data and ECE for a single binary class.

    Parameters
    ----------
    y_true_bin : (N,) array of {0, 1}
    y_prob     : (N,) array of predicted probabilities for the positive class
    n_bins     : number of equal-width probability bins

    Returns
    -------
    diagram : list of (mean_predicted_prob, fraction_positive) per bin
    ece     : Expected Calibration Error
    """
    bins   = np.linspace(0.0, 1.0, n_bins + 1)
    diagram: List[Tuple[float, float]] = []
    ece    = 0.0
    n      = len(y_prob)

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if not mask.any():
            continue
        mean_pred = float(y_prob[mask].mean())
        frac_pos  = float(y_true_bin[mask].mean())
        diagram.append((mean_pred, frac_pos))
        ece += (mask.sum() / n) * abs(mean_pred - frac_pos)

    return diagram, float(ece)


# ---------------------------------------------------------------------------
# Core public functions
# ---------------------------------------------------------------------------

def evaluate_calibration(
    model_name: str,
    texts: List[str],
    labels: List[str],
    n_bins: int = 10,
) -> CalibrationReport:
    """
    Evaluate calibration quality for a trained Layer 0 classifier.

    Parameters
    ----------
    model_name : one of "manipulation", "objectivity", "assumption"
    texts      : list of raw query strings
    labels     : list of ground-truth label strings (must match the
                 artifact's LabelEncoder classes)
    n_bins     : number of bins for ECE / reliability diagram

    Returns
    -------
    CalibrationReport with per-class Brier scores, ECE, and reliability
    diagram data.
    """
    from sklearn.metrics import brier_score_loss
    from sklearn.preprocessing import label_binarize

    artifact_path = _MODELS_DIR / _ARTIFACT_NAMES[model_name]
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Artifact not found: {artifact_path}. "
            "Run train_layer0_models.py first."
        )

    artifact = joblib.load(artifact_path)
    clf = artifact["model"]
    le  = artifact["label_encoder"]

    X = _build_X(texts)
    y = le.transform(labels)
    classes = list(range(len(le.classes_)))

    proba  = clf.predict_proba(X)
    Y_bin  = label_binarize(y, classes=classes)
    if Y_bin.shape[1] == 1:       # binary edge case
        Y_bin = np.hstack([1 - Y_bin, Y_bin])

    per_class: List[ClassCalibrationStats] = []
    brier_scores: List[float] = []
    ece_scores:   List[float] = []

    for i, class_label in enumerate(le.classes_):
        bs = float(brier_score_loss(Y_bin[:, i], proba[:, i]))
        diagram, ece = _reliability_diagram(Y_bin[:, i], proba[:, i], n_bins)
        per_class.append(ClassCalibrationStats(
            label=class_label,
            brier_score=bs,
            reliability_diagram=diagram,
            ece=ece,
        ))
        brier_scores.append(bs)
        ece_scores.append(ece)

    report = CalibrationReport(
        model_name=model_name,
        n_samples=len(texts),
        mean_brier_score=float(np.mean(brier_scores)),
        mean_ece=float(np.mean(ece_scores)),
        per_class=per_class,
    )
    report.log_summary()
    return report


def calibrate_artifact(
    model_name: str,
    calibration_texts: List[str],
    calibration_labels: List[str],
    method: str = "isotonic",
    cv: int = 5,
) -> Path:
    """
    Re-calibrate a trained Layer 0 classifier on a new calibration set
    without full retraining.

    This is useful when:
    - The base model took hours to train and you don't want to redo it.
    - New labelled data has arrived that is specifically useful for
      calibration (e.g. human-reviewed ambiguous cases).

    The inner estimator is extracted from the existing
    CalibratedClassifierCV wrapper (or used directly if unwrapped),
    re-wrapped with a fresh CalibratedClassifierCV fitted on the supplied
    calibration set, and saved back to the same .joblib path.

    Parameters
    ----------
    model_name          : one of "manipulation", "objectivity", "assumption"
    calibration_texts   : list of raw query strings for calibration
    calibration_labels  : ground-truth label strings
    method              : "isotonic" (default) or "sigmoid"
    cv                  : number of CV folds; use "prefit" to treat the
                          base estimator as already fitted and calibrate
                          on the supplied set directly (faster, but
                          requires calibration_texts to be held-out data
                          the base model has never seen)

    Returns
    -------
    Path to the updated .joblib artifact.
    """
    from sklearn.calibration import CalibratedClassifierCV

    artifact_path = _MODELS_DIR / _ARTIFACT_NAMES[model_name]
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Artifact not found: {artifact_path}. "
            "Run train_layer0_models.py first."
        )

    artifact = joblib.load(artifact_path)
    clf = artifact["model"]
    le  = artifact["label_encoder"]

    # Extract the base estimator if already wrapped.
    base_estimator = clf
    if hasattr(clf, "estimator"):
        # CalibratedClassifierCV stores the base as .estimator
        base_estimator = clf.estimator
    elif hasattr(clf, "base_estimator"):
        base_estimator = clf.base_estimator

    log.info(
        "[calibration] Re-calibrating %s on %d samples "
        "(method=%s, cv=%s) ...",
        model_name, len(calibration_texts), method, cv,
    )

    X = _build_X(calibration_texts)
    y = le.transform(calibration_labels)

    min_class_count = int(np.bincount(y).min())
    safe_cv: int | str
    if cv == "prefit" or cv == 0:
        safe_cv = "prefit"
    else:
        safe_cv = max(2, min(int(cv), min_class_count))

    new_clf = CalibratedClassifierCV(
        base_estimator,
        method=method,
        cv=safe_cv,
    )
    new_clf.fit(X, y)

    artifact["model"] = new_clf
    artifact["version"] = artifact.get("version", "1.0") + "+recal"
    joblib.dump(artifact, artifact_path)
    log.info("[calibration] Saved recalibrated artifact → %s", artifact_path)
    return artifact_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _auto_load_eval_data(
    model_name: str,
) -> Optional[Tuple[List[str], List[str]]]:
    """
    Try to load evaluation data from the training CSVs already on disk.
    Returns (texts, labels) or None if insufficient data is available.
    This is a best-effort helper for the CLI; it is NOT a substitute for
    a proper held-out evaluation set.
    """
    import csv

    data_dir = _ROOT / "training_data"

    if model_name == "manipulation":
        path = data_dir / "trivia_qa.csv"
        if not path.exists():
            return None
        texts, labels = [], []
        with open(path, newline="", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                if i >= 300:
                    break
                q = row.get("question", "").strip()
                if q:
                    texts.append(q)
                    labels.append("NOT_MANIPULATIVE")
        return (texts, labels) if len(texts) >= 10 else None

    if model_name == "objectivity":
        path = data_dir / "trivia_qa.csv"
        if not path.exists():
            return None
        texts, labels = [], []
        with open(path, newline="", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                if i >= 300:
                    break
                q = row.get("question", "").strip()
                if q:
                    texts.append(q)
                    labels.append("OBJECTIVE")
        return (texts, labels) if len(texts) >= 10 else None

    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate or re-calibrate Layer 0 sklearn classifiers."
    )
    parser.add_argument(
        "--model",
        choices=["manipulation", "objectivity", "assumption", "all"],
        default="all",
        help="Which classifier to process (default: all).",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only evaluate calibration; do not re-calibrate.",
    )
    parser.add_argument(
        "--method",
        choices=["isotonic", "sigmoid"],
        default="isotonic",
        help="Calibration method (default: isotonic).",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="CV folds for calibration (default: 5; use 0 for prefit mode).",
    )
    args = parser.parse_args()

    targets = (
        ["manipulation", "objectivity"]
        if args.model == "all"
        else [args.model]
    )
    # assumption_typer evaluation requires multi-label handling not yet
    # implemented in this CLI; skip it gracefully.
    targets = [t for t in targets if t != "assumption"]
    if not targets:
        log.warning("[calibration] No applicable targets (assumption_typer not supported by CLI yet).")
        return

    for model_name in targets:
        artifact_path = _MODELS_DIR / _ARTIFACT_NAMES[model_name]
        if not artifact_path.exists():
            log.warning(
                "[calibration] %s: artifact not found at %s — skipping.",
                model_name, artifact_path,
            )
            continue

        data = _auto_load_eval_data(model_name)
        if data is None:
            log.warning(
                "[calibration] %s: no eval data available — "
                "run train_layer0_models.py --skip-download first or supply "
                "your own texts/labels via the Python API.",
                model_name,
            )
            continue

        texts, labels = data

        if args.eval_only:
            evaluate_calibration(model_name, texts, labels)
        else:
            calibrate_artifact(
                model_name, texts, labels,
                method=args.method,
                cv=args.cv,
            )
            evaluate_calibration(model_name, texts, labels)


if __name__ == "__main__":
    main()
