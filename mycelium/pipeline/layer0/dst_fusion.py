"""
Layer 0 — Classifier-Output DST Fusion
=======================================

Fuses the calibrated ``predict_proba`` vectors from the three Layer 0
sklearn classifiers (manipulation detector, objectivity classifier, and
assumption typer) using Dempster-Shafer Theory (DST) over the **routing**
frame of discernment.

Why this is different from ``mycelium/fusion/dst_fusion.py``
------------------------------------------------------------
``mycelium/fusion/dst_fusion.py`` fuses *truth-claim* ConfidenceState
objects over the frame {TRUE, FALSE, UNKNOWN} — it answers "how confident
is the system that a factual claim is true?".

This module fuses *classifier outputs* over the **routing** frame
Ω = {REFUSE, MULTI_PERSPECTIVE, CLARIFICATION, REASONING_PIPELINE}.
Its inputs are three ``predict_proba`` probability distributions; its
output is a principled routing decision backed by a full belief mass
function rather than a naked ``argmax``.

Frame of Discernment
--------------------
Ω = {REFUSE, MULTI_PERSPECTIVE, CLARIFICATION, REASONING_PIPELINE}

Focal elements are *subsets* of Ω.  The power set 2^Ω has 16 elements but
in practice only four singletons and the full ignorance set Ω are used,
keeping the representation sparse.

Mass Function Representation
-----------------------------
``MassFunction`` is a ``dict[frozenset[Route], float]`` where:
  - keys   = focal elements (non-empty subsets of Ω)
  - values = belief mass assigned to that focal element
  - invariant: sum(values) == 1.0  (within float tolerance)

The full ignorance set ``frozenset(ALL_ROUTES)`` (= Ω) is always a valid
focal element and absorbs all classifier uncertainty.

Label → Route Mapping
---------------------
Manipulation classifier labels map to routes as follows:

  NOT_MANIPULATIVE        → REASONING_PIPELINE  (clean question)
  LOADED_QUESTION         → MULTI_PERSPECTIVE    (needs balanced answer)
  PRESUPPOSITION_INJECTION→ MULTI_PERSPECTIVE    (hidden assumptions)
  COERCIVE                → REFUSE               (hard block label)
  JAILBREAK_ATTEMPT       → REFUSE               (hard block label)

Objectivity classifier labels:

  OBJECTIVE               → REASONING_PIPELINE
  SUBJECTIVE              → MULTI_PERSPECTIVE
  VALUE_LADEN             → MULTI_PERSPECTIVE
  AMBIGUOUS               → CLARIFICATION

Assumption typer contributes only a soft MULTI_PERSPECTIVE signal:
  any detected assumption  → partial mass on MULTI_PERSPECTIVE
  no assumptions           → mass goes to Ω (ignorance)

Dempster's Rule of Combination
-------------------------------
For two mass functions m1, m2::

    K = Σ_{A ∩ B = ∅} m1(A) * m2(B)       # conflict mass

    m12(C) = Σ_{A ∩ B = C} m1(A) * m2(B) / (1 − K)   for C ≠ ∅

When K >= CONFLICT_THRESHOLD (0.90), a hard raise is avoided: instead
the conflicted mass is redistributed proportionally to Ω (ignorance) so
the router always gets a result.  ``conflict_k`` in the output signals
that this occurred.

Pignistic Transformation
------------------------
The Pignistic probability BetP maps each route ``r`` to::

    BetP(r) = Σ_{A ∋ r} m(A) / |A|

This distributes ignorance mass equally among routes it covers, providing
a scalar probability for each route that is compatible with expected-utility
decision making.  ``dominant_route`` is argmax(BetP).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# Routing frame of discernment
# ---------------------------------------------------------------------------

Route = str  # one of the four route strings below

REFUSE               = "REFUSE"
MULTI_PERSPECTIVE    = "MULTI_PERSPECTIVE"
CLARIFICATION        = "CLARIFICATION"
REASONING_PIPELINE   = "REASONING_PIPELINE"

ALL_ROUTES: FrozenSet[Route] = frozenset([
    REFUSE, MULTI_PERSPECTIVE, CLARIFICATION, REASONING_PIPELINE
])

# The ignorance focal element — mass here means "I don't know which route".
_OMEGA: FrozenSet[Route] = ALL_ROUTES

# ---------------------------------------------------------------------------
# Label → focal-element maps
# ---------------------------------------------------------------------------

# Each label maps to a *singleton* focal element.  Labels that are
# legitimately uncertain get mapped to _OMEGA by the mass-builder instead.

_MANIP_LABEL_TO_ROUTE: Dict[str, Route] = {
    "NOT_MANIPULATIVE":         REASONING_PIPELINE,
    "LOADED_QUESTION":          MULTI_PERSPECTIVE,
    "PRESUPPOSITION_INJECTION": MULTI_PERSPECTIVE,
    "COERCIVE":                 REFUSE,
    "JAILBREAK_ATTEMPT":        REFUSE,
}

_OBJ_LABEL_TO_ROUTE: Dict[str, Route] = {
    "OBJECTIVE":   REASONING_PIPELINE,
    "SUBJECTIVE":  MULTI_PERSPECTIVE,
    "VALUE_LADEN": MULTI_PERSPECTIVE,
    "AMBIGUOUS":   CLARIFICATION,
}

# ---------------------------------------------------------------------------
# Conflict threshold
# ---------------------------------------------------------------------------

# K >= this triggers graceful conflict handling (mass → Ω) instead of raising.
CONFLICT_THRESHOLD: float = 0.90

# Minimum ignorance mass always retained so the engine never over-commits.
_MIN_IGNORANCE: float = 0.02

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

MassFunction = Dict[FrozenSet[Route], float]


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class Layer0BeliefState:
    """DST fusion result for the Layer 0 routing decision.

    Attributes
    ----------
    dominant_route : str
        The route with the highest Pignistic probability — the routing
        decision the router should use.
    pignistic_probs : dict[str, float]
        BetP(r) for each route r ∈ Ω.  Sums to 1.0.
    mass_function : MassFunction
        The raw combined Dempster-Shafer mass function over subsets of Ω.
        Useful for logging and downstream calibration diagnostics.
    conflict_k : float
        Total conflict mass K observed across all pairwise combinations.
        Values near 1.0 indicate the classifiers strongly disagree.
    is_uncertain : bool
        True when the ignorance mass m(Ω) > 0.30, meaning the classifiers
        collectively lack confidence in any single route.
    sources : list[str]
        Human-readable labels for the evidence sources that were fused.
    """

    dominant_route:  Route
    pignistic_probs: Dict[Route, float]
    mass_function:   MassFunction
    conflict_k:      float
    is_uncertain:    bool
    sources:         List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core DST operations
# ---------------------------------------------------------------------------

def _normalise(m: MassFunction) -> MassFunction:
    """Normalise a mass function so its values sum to exactly 1.0."""
    total = sum(m.values())
    if total < 1e-12:
        # Degenerate: return pure ignorance
        return {_OMEGA: 1.0}
    return {k: v / total for k, v in m.items() if v > 0.0}


def _dempster_combine(m1: MassFunction, m2: MassFunction) -> Tuple[MassFunction, float]:
    """Dempster's rule of combination for two mass functions.

    Parameters
    ----------
    m1, m2 : MassFunction
        Two independent evidence sources.

    Returns
    -------
    combined : MassFunction
        The combined mass function, normalised to sum to 1.0.
    k : float
        The conflict mass K = Σ_{A∩B=∅} m1(A)·m2(B).
        When K >= CONFLICT_THRESHOLD, conflicted mass is redirected to Ω
        rather than raising, so the router always gets a result.
    """
    unnorm: MassFunction = {}
    k = 0.0

    for focal_a, mass_a in m1.items():
        if mass_a < 1e-12:
            continue
        for focal_b, mass_b in m2.items():
            if mass_b < 1e-12:
                continue
            intersection = focal_a & focal_b
            product = mass_a * mass_b
            if not intersection:
                # Conflict mass
                k += product
            else:
                unnorm[intersection] = unnorm.get(intersection, 0.0) + product

    if k >= CONFLICT_THRESHOLD:
        # Graceful degradation: redistribute all conflicted mass to Ω
        # and log the event.  This keeps the router functional even when
        # classifiers strongly disagree (e.g. first-boot before calibration).
        logger.warning(
            "DST conflict K=%.4f >= %.2f — redistributing conflict mass to Ω. "
            "Consider re-calibrating the Layer 0 classifiers.",
            k, CONFLICT_THRESHOLD,
        )
        unnorm[_OMEGA] = unnorm.get(_OMEGA, 0.0) + k
        k_reported = k
        k = 0.0  # no longer need to normalise by (1-k) since we absorbed it
    else:
        k_reported = k

    normaliser = 1.0 - k
    if normaliser < 1e-12:
        # Shouldn't happen after the graceful-degradation branch above,
        # but guard anyway.
        return {_OMEGA: 1.0}, k_reported

    combined: MassFunction = {
        focal: mass / normaliser
        for focal, mass in unnorm.items()
        if mass > 0.0
    }
    return _normalise(combined), k_reported


def _pignistic(m: MassFunction) -> Dict[Route, float]:
    """Pignistic probability transformation BetP.

    BetP(r) = Σ_{A ∋ r} m(A) / |A|

    Distributes the mass of each focal element equally among its members.
    Produces a classical probability distribution over the routes for
    decision-making purposes.
    """
    bet_p: Dict[Route, float] = {r: 0.0 for r in ALL_ROUTES}
    for focal, mass in m.items():
        size = len(focal)
        share = mass / size
        for route in focal:
            bet_p[route] = bet_p.get(route, 0.0) + share
    # Normalise for floating-point safety
    total = sum(bet_p.values())
    if total > 1e-12:
        bet_p = {r: v / total for r, v in bet_p.items()}
    return bet_p


# ---------------------------------------------------------------------------
# Mass-function builders
# ---------------------------------------------------------------------------

def _build_manipulation_mass(
    proba: np.ndarray,
    label_order: List[str],
    refuse_confidence_threshold: float = 0.65,
) -> Tuple[MassFunction, str]:
    """Convert a manipulation-classifier predict_proba vector to a MassFunction.

    Strategy
    --------
    Each label l has probability p[l].  The mass it contributes to its
    singleton focal element is scaled by p[l].  However, REFUSE is only
    backed by *confident* coercive/jailbreak signals — if the combined
    probability on REFUSE-route labels is below ``refuse_confidence_threshold``,
    that mass is re-routed to MULTI_PERSPECTIVE (the safe soft-handling path)
    rather than REFUSE.

    The residual uncertainty mass (1 − Σ committed mass) goes to Ω.

    Parameters
    ----------
    proba : np.ndarray shape (n_labels,)
        Calibrated predict_proba output.
    label_order : list[str]
        Label names corresponding to each position in ``proba``, in the
        same order as the LabelEncoder used during training.
    refuse_confidence_threshold : float
        Minimum combined probability on hard-refuse labels (COERCIVE,
        JAILBREAK_ATTEMPT) needed to commit mass to {REFUSE}.
        Below this threshold the mass goes to {MULTI_PERSPECTIVE} instead.

    Returns
    -------
    MassFunction, source_label
    """
    route_mass: Dict[Route, float] = {r: 0.0 for r in ALL_ROUTES}
    refuse_candidate_mass = 0.0

    for label, p in zip(label_order, proba):
        route = _MANIP_LABEL_TO_ROUTE.get(label)
        if route is None:
            continue
        if route == REFUSE:
            refuse_candidate_mass += float(p)
        else:
            route_mass[route] += float(p)

    # Apply confidence gate on REFUSE
    if refuse_candidate_mass >= refuse_confidence_threshold:
        route_mass[REFUSE] += refuse_candidate_mass
    else:
        # Not confident enough to hard-refuse: treat as soft-handling
        route_mass[MULTI_PERSPECTIVE] += refuse_candidate_mass

    # Committed mass is everything except the ignorance residual
    committed = sum(route_mass.values())
    # Always retain a small ignorance floor so the mass function never
    # fully forecloses the possibility of being wrong.
    ignorance = max(_MIN_IGNORANCE, 1.0 - committed)

    m: MassFunction = {}
    for route, mass in route_mass.items():
        if mass > 1e-12:
            m[frozenset([route])] = mass
    m[_OMEGA] = ignorance

    return _normalise(m), "manipulation_classifier"


def _build_objectivity_mass(
    proba: np.ndarray,
    label_order: List[str],
) -> Tuple[MassFunction, str]:
    """Convert an objectivity-classifier predict_proba vector to a MassFunction.

    Each label maps directly to a singleton route focal element.
    The AMBIGUOUS label maps to {CLARIFICATION}.
    The residual uncertainty goes to Ω.
    """
    route_mass: Dict[Route, float] = {r: 0.0 for r in ALL_ROUTES}

    for label, p in zip(label_order, proba):
        route = _OBJ_LABEL_TO_ROUTE.get(label)
        if route is not None:
            route_mass[route] += float(p)

    committed = sum(route_mass.values())
    ignorance = max(_MIN_IGNORANCE, 1.0 - committed)

    m: MassFunction = {}
    for route, mass in route_mass.items():
        if mass > 1e-12:
            m[frozenset([route])] = mass
    m[_OMEGA] = ignorance

    return _normalise(m), "objectivity_classifier"


def _build_assumption_mass(
    n_assumptions: int,
    max_assumptions: int = 5,
) -> Tuple[MassFunction, str]:
    """Build a soft MassFunction from the assumption-typer output.

    The assumption typer outputs a multi-hot label vector (multiple
    assumption types per query), not a probability distribution, so it
    cannot contribute per-class probabilities to the routing frame in the
    same way.  Instead we use *assumption count* as a soft signal:

        - 0 assumptions → pure ignorance (mass entirely on Ω)
        - 1+ assumptions → increasing mass on {MULTI_PERSPECTIVE},
          saturating at ``max_assumptions``

    This is intentionally conservative: the assumption typer should nudge
    toward MULTI_PERSPECTIVE but never dominate the fusion result.
    """
    if n_assumptions <= 0:
        return {_OMEGA: 1.0}, "assumption_typer(0)"

    # Linear ramp from 0.10 (1 assumption) to 0.40 (max_assumptions)
    strength = min(1.0, n_assumptions / max(max_assumptions, 1))
    multi_mass = 0.10 + 0.30 * strength
    ignorance  = max(_MIN_IGNORANCE, 1.0 - multi_mass)

    m: MassFunction = {
        frozenset([MULTI_PERSPECTIVE]): multi_mass,
        _OMEGA: ignorance,
    }
    return _normalise(m), f"assumption_typer({n_assumptions})"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fuse_layer0_classifiers(
    manip_proba:    Optional[np.ndarray],
    manip_labels:   Optional[List[str]],
    obj_proba:      Optional[np.ndarray],
    obj_labels:     Optional[List[str]],
    n_assumptions:  int = 0,
    refuse_threshold: float = 0.65,
) -> Layer0BeliefState:
    """Fuse all available Layer 0 classifier outputs into a routing decision.

    This is the primary public entry-point.  The router calls this after
    collecting raw ``predict_proba`` vectors from the three classifiers.

    Parameters
    ----------
    manip_proba : np.ndarray or None
        Calibrated predict_proba output from the manipulation classifier
        (shape: ``(n_manip_labels,)``).  Pass ``None`` if the model is not
        loaded (cold-start); its evidence is omitted from fusion.
    manip_labels : list[str] or None
        Label names in the same order as ``manip_proba`` (from LabelEncoder).
    obj_proba : np.ndarray or None
        Calibrated predict_proba output from the objectivity classifier.
    obj_labels : list[str] or None
        Label names in the same order as ``obj_proba``.
    n_assumptions : int
        Number of assumptions extracted by the assumption typer.  Set to 0
        if the typer was not run.
    refuse_threshold : float
        Minimum combined probability on COERCIVE/JAILBREAK_ATTEMPT labels
        required to commit mass to {REFUSE}.  Mirrors the router's
        ``REFUSE_CONFIDENCE_THRESHOLD``.  Default: 0.65.

    Returns
    -------
    Layer0BeliefState
        Fusion result with dominant_route, pignistic_probs, raw mass function,
        conflict_k, and is_uncertain flag.

    Notes
    -----
    If *all* classifiers are unavailable (cold-start), the function returns
    a pure-ignorance belief state that routes to REASONING_PIPELINE with
    low certainty (is_uncertain=True), allowing the existing rule-based
    router logic to take over without crashing.
    """
    frames: List[Tuple[MassFunction, str]] = []

    # --- Manipulation classifier ---
    if manip_proba is not None and manip_labels:
        try:
            m, label = _build_manipulation_mass(
                np.asarray(manip_proba, dtype=float),
                manip_labels,
                refuse_confidence_threshold=refuse_threshold,
            )
            frames.append((m, label))
        except Exception as exc:
            logger.debug("DST: failed to build manipulation mass: %s", exc)

    # --- Objectivity classifier ---
    if obj_proba is not None and obj_labels:
        try:
            m, label = _build_objectivity_mass(
                np.asarray(obj_proba, dtype=float),
                obj_labels,
            )
            frames.append((m, label))
        except Exception as exc:
            logger.debug("DST: failed to build objectivity mass: %s", exc)

    # --- Assumption typer (soft signal) ---
    try:
        m, label = _build_assumption_mass(n_assumptions)
        frames.append((m, label))
    except Exception as exc:
        logger.debug("DST: failed to build assumption mass: %s", exc)

    # --- Cold-start guard ---
    if not frames:
        logger.warning(
            "DST fusion: no classifier evidence available — "
            "returning vacuous belief state (all routes equally probable)."
        )
        vacuous: MassFunction = {_OMEGA: 1.0}
        pignistic = _pignistic(vacuous)
        dominant = max(pignistic, key=lambda r: pignistic[r])
        return Layer0BeliefState(
            dominant_route=dominant,
            pignistic_probs=pignistic,
            mass_function=vacuous,
            conflict_k=0.0,
            is_uncertain=True,
            sources=[],
        )

    # --- Sequential Dempster combination ---
    combined_m, total_k = frames[0][0], 0.0
    sources = [frames[0][1]]

    for m_next, src in frames[1:]:
        combined_m, k = _dempster_combine(combined_m, m_next)
        total_k += k
        sources.append(src)

    # --- Pignistic transformation → routing decision ---
    pignistic = _pignistic(combined_m)
    dominant  = max(pignistic, key=lambda r: pignistic[r])

    # Uncertainty flag: ignorance mass > 30%
    ignorance_mass = combined_m.get(_OMEGA, 0.0)
    is_uncertain   = ignorance_mass > 0.30

    return Layer0BeliefState(
        dominant_route=dominant,
        pignistic_probs={r: round(p, 6) for r, p in pignistic.items()},
        mass_function=combined_m,
        conflict_k=round(total_k, 6),
        is_uncertain=is_uncertain,
        sources=sources,
    )


# ---------------------------------------------------------------------------
# Convenience: build from already-computed ManipulationDetectionResult
#              and ObjectivityResult (no raw proba needed from caller)
# ---------------------------------------------------------------------------

def fuse_from_results(
    manip_label:       str,
    manip_confidence:  float,
    obj_label:         str,
    obj_confidence:    float,
    n_assumptions:     int = 0,
    refuse_threshold:  float = 0.65,
) -> Layer0BeliefState:
    """Build a belief state from high-level classifier results (no raw proba).

    This convenience wrapper is useful when the router has already run
    the classifiers but the raw ``predict_proba`` vectors were not
    forwarded.  It synthesises minimal two-mass mass functions from the
    label + confidence scalars:

        m({dominant_route}) = confidence
        m(Ω)               = 1 − confidence  (+ _MIN_IGNORANCE floor)

    This is a degenerate but valid DST mass function that faithfully
    represents a single deterministic classifier output with uncertainty.

    Parameters
    ----------
    manip_label, manip_confidence
        Label and confidence from ManipulationDetectionResult.
    obj_label, obj_confidence
        label and confidence from ObjectivityResult.
    n_assumptions : int
        Number of assumptions from ValueAssumptionExtractor.
    refuse_threshold : float
        Confidence gate for REFUSE routing.
    """
    def _scalar_mass(label: str, confidence: float, label_map: Dict[str, Route]) -> MassFunction:
        route = label_map.get(label)
        if route is None:
            return {_OMEGA: 1.0}
        # Apply REFUSE confidence gate
        if route == REFUSE and confidence < refuse_threshold:
            route = MULTI_PERSPECTIVE
        committed  = max(0.0, min(1.0 - _MIN_IGNORANCE, float(confidence)))
        ignorance  = max(_MIN_IGNORANCE, 1.0 - committed)
        return _normalise({
            frozenset([route]): committed,
            _OMEGA: ignorance,
        })

    frames: List[Tuple[MassFunction, str]] = []

    frames.append((_scalar_mass(manip_label, manip_confidence, _MANIP_LABEL_TO_ROUTE),
                   f"manipulation({manip_label})"))
    frames.append((_scalar_mass(obj_label, obj_confidence, _OBJ_LABEL_TO_ROUTE),
                   f"objectivity({obj_label})"))

    assumption_m, assumption_src = _build_assumption_mass(n_assumptions)
    frames.append((assumption_m, assumption_src))

    combined_m, total_k = frames[0][0], 0.0
    sources = [frames[0][1]]
    for m_next, src in frames[1:]:
        combined_m, k = _dempster_combine(combined_m, m_next)
        total_k += k
        sources.append(src)

    pignistic    = _pignistic(combined_m)
    dominant     = max(pignistic, key=lambda r: pignistic[r])
    ignorance_mass = combined_m.get(_OMEGA, 0.0)

    return Layer0BeliefState(
        dominant_route=dominant,
        pignistic_probs={r: round(p, 6) for r, p in pignistic.items()},
        mass_function=combined_m,
        conflict_k=round(total_k, 6),
        is_uncertain=ignorance_mass > 0.30,
        sources=sources,
    )
