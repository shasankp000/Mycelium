"""
mycelium.pipeline.layer0
========================
Public API surface for the Layer 0 NLP gate.

Import the high-level entry-points from here rather than from the
individual sub-modules — this keeps downstream code stable if internal
module organisation changes.

Typical usage
-------------
    # One-shot boot (loads sklearn models + runs calibration eval):
    from mycelium.pipeline.layer0 import Layer0Initializer
    status = Layer0Initializer.boot()

    # Route a question:
    from mycelium.pipeline.layer0 import QuestionRouter
    router = QuestionRouter()
    result = router.route("Is it possible to build a thermoelectric device?")

    # Direct DST fusion (advanced / testing):
    from mycelium.pipeline.layer0 import fuse_layer0_classifiers
    belief = fuse_layer0_classifiers(
        manip_proba=proba_vec,
        manip_labels=label_list,
        obj_proba=obj_vec,
        obj_labels=obj_labels,
        n_assumptions=2,
    )
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Router  (primary public entry-point)
# ---------------------------------------------------------------------------
from mycelium.pipeline.layer0.router import (
    QuestionRouter,
    REFUSE_CONFIDENCE_THRESHOLD,
    _HARD_REFUSE_LABELS as HARD_REFUSE_LABELS,
)

# ---------------------------------------------------------------------------
# Individual classifiers (exposed for unit-testing and direct use)
# ---------------------------------------------------------------------------
from mycelium.pipeline.layer0.manipulation_detector import (
    ManipulationDetector,
    ManipulationDetectionResult,
)
from mycelium.pipeline.layer0.objectivity_classifier import (
    ObjectivityClassifier,
    ObjectivityResult,
)
from mycelium.pipeline.layer0.value_assumption_extractor import (
    ValueAssumptionExtractor,
    ValueAssumptionResult,
    Assumption,
)

# ---------------------------------------------------------------------------
# DST fusion
# ---------------------------------------------------------------------------
from mycelium.pipeline.layer0.dst_fusion import (
    fuse_layer0_classifiers,
    fuse_from_results,
    Layer0BeliefState,
    MassFunction,
    REFUSE,
    MULTI_PERSPECTIVE,
    CLARIFICATION,
    REASONING_PIPELINE,
    ALL_ROUTES,
    CONFLICT_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Boot orchestrator
# ---------------------------------------------------------------------------
from mycelium.pipeline.layer0.init import (
    Layer0Initializer,
    Layer0Status,
)

__all__ = [
    # Router
    "QuestionRouter",
    "REFUSE_CONFIDENCE_THRESHOLD",
    "HARD_REFUSE_LABELS",
    # Classifiers
    "ManipulationDetector",
    "ManipulationDetectionResult",
    "ObjectivityClassifier",
    "ObjectivityResult",
    "ValueAssumptionExtractor",
    "ValueAssumptionResult",
    "Assumption",
    # DST
    "fuse_layer0_classifiers",
    "fuse_from_results",
    "Layer0BeliefState",
    "MassFunction",
    "REFUSE",
    "MULTI_PERSPECTIVE",
    "CLARIFICATION",
    "REASONING_PIPELINE",
    "ALL_ROUTES",
    "CONFLICT_THRESHOLD",
    # Boot
    "Layer0Initializer",
    "Layer0Status",
]
