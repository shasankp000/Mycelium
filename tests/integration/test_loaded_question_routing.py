"""
Integration tests — LOADED_QUESTION routes to MULTI_PERSPECTIVE (Problem 1b).

A loaded question must never hard-block; it deserves a balanced answer.
These tests verify that LOADED_QUESTION is treated as a non-refusing label
by the router regardless of confidence.
"""
import pytest
from unittest.mock import patch

from mycelium.pipeline.layer0.router import QuestionRouter
from mycelium.pipeline.layer0.manipulation_detector import ManipulationResult


DETECTOR_PATH = (
    "mycelium.pipeline.layer0.router.ManipulationDetector.detect"
)


def _loaded(confidence: float) -> ManipulationResult:
    return ManipulationResult(
        label="LOADED_QUESTION",
        confidence=confidence,
        is_manipulative=True,
        explanation="stub",
    )


# ---------------------------------------------------------------------------
# 1. LOADED_QUESTION → MULTI_PERSPECTIVE at any confidence level
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("confidence", [0.20, 0.50, 0.65, 0.80, 0.99])
def test_loaded_question_routes_to_multi_perspective(confidence):
    """
    LOADED_QUESTION must always route to MULTI_PERSPECTIVE, never REFUSE,
    regardless of how confident the detector is.
    """
    router = QuestionRouter()
    with patch(DETECTOR_PATH, return_value=_loaded(confidence)):
        result = router.route("Have you stopped using manipulative AI yet?")

    assert result.route == "MULTI_PERSPECTIVE", (
        f"LOADED_QUESTION at confidence={confidence} routed to {result.route!r} "
        "instead of MULTI_PERSPECTIVE"
    )


# ---------------------------------------------------------------------------
# 2. LOADED_QUESTION is never REFUSE at any confidence level
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("confidence", [0.65, 0.75, 0.95, 1.00])
def test_loaded_question_never_refuses(confidence):
    """
    Even above the REFUSE threshold, LOADED_QUESTION must not produce REFUSE.
    This is the exact failure mode described in Problem 1b of the bug report.
    """
    router = QuestionRouter()
    with patch(DETECTOR_PATH, return_value=_loaded(confidence)):
        result = router.route("Isn't it obvious that AI will destroy humanity?")

    assert result.route != "REFUSE", (
        f"LOADED_QUESTION at confidence={confidence} was hard-refused — "
        "routing table not updated"
    )


# ---------------------------------------------------------------------------
# 3. Real-model smoke test on the canonical motivating example
# ---------------------------------------------------------------------------

def test_is_it_possible_routes_to_multi_perspective_or_reasoning():
    """
    Live router call (no mock): 'Is it possible to create a device that
    converts heat to electricity?' must not result in REFUSE.
    Acceptable routes: MULTI_PERSPECTIVE or REASONING_PIPELINE.
    """
    router = QuestionRouter()
    result = router.route(
        "Is it possible to create a device that converts heat to electricity?"
    )
    assert result.route in {"MULTI_PERSPECTIVE", "REASONING_PIPELINE", "CLARIFICATION"}, (
        f"Unexpected route {result.route!r} for canonical loaded-question example"
    )
