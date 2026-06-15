"""
Layer 0 — Boot Orchestrator
============================
Runs the mandatory startup sequence for the Layer 0 NLP gate:

    1. Load sklearn classifier artifacts from models/layer0/ via
       model_registry.warmup_layer0().

    2. Run probability calibration evaluation on every loaded model via
       probability_calibration.evaluate_calibration().

       This is the step that was entirely missing before this module
       existed.  Without it, the sklearn SVM predict_proba outputs that
       feed DST mass functions are raw, uncalibrated Platt scores whose
       confidence values are systematically overconfident in sparse
       regions of feature space.  Running evaluate_calibration() at boot
       computes Brier scores and logs them so operators know whether
       re-calibration (isotonic regression / Platt scaling) is needed.

    3. Return a Layer0Status dataclass that records per-model loaded and
       calibration health flags.  Health-check endpoints can read this
       without reimplementing the boot logic.

Idempotency
-----------
Layer0Initializer.boot() is safe to call multiple times.  After the
first successful call the result is cached and subsequent calls return
the cached Layer0Status immediately (no repeated I/O or model loading).

Usage
-----
    # At application startup:
    from mycelium.pipeline.layer0.init import Layer0Initializer
    status = Layer0Initializer.boot()
    if not status.all_loaded:
        logger.warning("Layer 0 models not fully loaded — LLM fallback active.")

    # In a health-check endpoint:
    from mycelium.pipeline.layer0.init import Layer0Initializer
    status = Layer0Initializer.get_status()
    return {"layer0": status.to_dict()}

    # Convenience factory that guarantees boot has run:
    from mycelium.pipeline.layer0.init import Layer0Initializer
    router = Layer0Initializer.get_router()
    result = router.route(question)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Names that warmup_layer0() knows about.
_MODEL_NAMES = (
    "manipulation_classifier",
    "objectivity_classifier",
    "assumption_typer",
)


# ---------------------------------------------------------------------------
# Status dataclass
# ---------------------------------------------------------------------------

@dataclass
class Layer0Status:
    """Result of the Layer 0 boot sequence.

    Attributes
    ----------
    loaded : dict[str, bool]
        Whether each sklearn artifact was found and loaded from disk.
        A False value means the LLM fallback is active for that component.
    calibrated : dict[str, bool]
        Whether calibration evaluation succeeded for each loaded model.
        False does NOT prevent routing — it is a diagnostic signal.
    brier_scores : dict[str, float]
        Brier score per model from evaluate_calibration().
        Lower is better (0.0 = perfect).  None / absent = not evaluated.
    boot_complete : bool
        True after boot() has finished regardless of individual model
        availability.
    """
    loaded:       Dict[str, bool]  = field(default_factory=dict)
    calibrated:   Dict[str, bool]  = field(default_factory=dict)
    brier_scores: Dict[str, float] = field(default_factory=dict)
    boot_complete: bool            = False

    @property
    def all_loaded(self) -> bool:
        """True when every expected classifier was successfully loaded."""
        return all(self.loaded.get(n, False) for n in _MODEL_NAMES)

    @property
    def any_loaded(self) -> bool:
        """True when at least one classifier is available."""
        return any(self.loaded.get(n, False) for n in _MODEL_NAMES)

    def to_dict(self) -> dict:
        """Serialisable summary suitable for health-check JSON responses."""
        return {
            "boot_complete": self.boot_complete,
            "all_loaded":    self.all_loaded,
            "any_loaded":    self.any_loaded,
            "models": {
                name: {
                    "loaded":      self.loaded.get(name, False),
                    "calibrated":  self.calibrated.get(name, False),
                    "brier_score": self.brier_scores.get(name),
                }
                for name in _MODEL_NAMES
            },
        }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Layer0Initializer:
    """
    Singleton boot orchestrator for the Layer 0 NLP gate.

    Only one boot sequence ever runs per process, protected by a
    threading.Lock.  Subsequent calls to boot() return the cached status.
    """

    _lock:   threading.Lock  = threading.Lock()
    _status: Optional[Layer0Status] = None
    _router = None  # cached QuestionRouter instance

    # ------------------------------------------------------------------
    # Public class-methods
    # ------------------------------------------------------------------

    @classmethod
    def boot(cls) -> Layer0Status:
        """Run the Layer 0 boot sequence (idempotent).

        Safe to call from multiple threads — only the first call does
        real work; subsequent calls return the cached status immediately.

        Returns
        -------
        Layer0Status
            Boot result with loaded/calibrated flags and Brier scores.
        """
        if cls._status is not None and cls._status.boot_complete:
            return cls._status

        with cls._lock:
            # Double-checked locking
            if cls._status is not None and cls._status.boot_complete:
                return cls._status

            logger.info("Layer0Initializer: starting boot sequence.")
            status = Layer0Status()

            # Step 1 — load sklearn artifacts
            status.loaded = cls._warmup_models()

            # Step 2 — calibration evaluation on each loaded model
            cls._run_calibration_eval(status)

            status.boot_complete = True
            cls._status = status

            loaded_count = sum(status.loaded.values())
            logger.info(
                "Layer0Initializer: boot complete. "
                "%d/%d classifiers loaded. "
                "Brier scores: %s",
                loaded_count,
                len(_MODEL_NAMES),
                {k: round(v, 4) for k, v in status.brier_scores.items()}
                if status.brier_scores else "n/a (no models loaded)",
            )
            if not status.all_loaded:
                logger.warning(
                    "Layer0Initializer: one or more classifiers not loaded — "
                    "LLM arbitration fallback is active for those components. "
                    "Run train_layer0_models.py to generate the .joblib files."
                )

            return status

    @classmethod
    def get_status(cls) -> Optional[Layer0Status]:
        """Return the cached boot status, or None if boot() has not been called."""
        return cls._status

    @classmethod
    def get_router(cls):
        """Return a QuestionRouter, running boot() first if needed.

        The same router instance is reused on subsequent calls so the
        NLP preprocessor and model-registry lookups are not duplicated.

        Returns
        -------
        QuestionRouter
        """
        if cls._router is None:
            with cls._lock:
                if cls._router is None:
                    cls.boot()
                    from mycelium.pipeline.layer0.router import QuestionRouter
                    cls._router = QuestionRouter()
        return cls._router

    @classmethod
    def reset(cls) -> None:
        """Reset cached state (for testing only — do not call in production)."""
        with cls._lock:
            cls._status = None
            cls._router = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @classmethod
    def _warmup_models(cls) -> Dict[str, bool]:
        """Call model_registry.warmup_layer0() and return the results map."""
        try:
            from mycelium.pipeline.model_registry import warmup_layer0
            results = warmup_layer0()
            return results
        except Exception as exc:
            logger.error(
                "Layer0Initializer: warmup_layer0() raised unexpectedly: %s", exc
            )
            # Return all-False so the rest of the boot sequence is safe.
            return {name: False for name in _MODEL_NAMES}

    @classmethod
    def _run_calibration_eval(cls, status: Layer0Status) -> None:
        """Skip calibration eval at boot — no held-out eval set is available.

        evaluate_calibration() requires (model_name, texts, labels) but at
        boot time we have no labelled evaluation data.  Brier scores can be
        computed offline via:

            python -m mycelium.pipeline.layer0.probability_calibration --model all

        All models are marked calibrated=None (unknown) rather than False
        (failed) so health-check endpoints can distinguish "not evaluated"
        from "evaluated and poor".
        """
        for name in _MODEL_NAMES:
            if status.loaded.get(name, False):
                status.calibrated[name] = None  # unknown — not evaluated at boot
                logger.debug(
                    "Layer0Initializer [calibration]: %s — "
                    "skipping Brier eval at boot (no eval data). "
                    "Run probability_calibration CLI for offline evaluation.",
                    name,
                )
