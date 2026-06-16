"""
tests/cold_storage/test_cold_storage.py

Covers mycelium/trm/v2/cold/  —  the TRM v2 cold-storage subsystem.

  Tier 1 – Structural  : import hygiene, class presence, instantiation,
                         attribute contracts.  No ML dependencies.
  Tier 2 – Semantic    : @pytest.mark.semantic
                         Behavioural correctness of DriftDetector,
                         ReplayBuffer / ReplayBufferManager, and
                         ColdStorageManager using the real module APIs.

All tests are pure-Python; torch is not needed here.
"""
from __future__ import annotations

import math
from typing import List

import pytest

# ---------------------------------------------------------------------------
# Real imports — the conftest.py torch-stub makes these safe
# ---------------------------------------------------------------------------
from mycelium.trm.v2.cold.drift   import DriftDetector, DriftConfig, DriftSignal
from mycelium.trm.v2.cold.replay  import ReplayBuffer, ReplayEntry, ReplayBufferManager
from mycelium.trm.v2.cold.manager import ColdStorageManager
from mycelium.domain_graph.registry import DomainGraphRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 32


def _unit(value: float = 1.0, dim: int = DIM) -> List[float]:
    """Return a normalised vector where every component equals *value*."""
    raw  = [value] * dim
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


def _orthogonal(dim: int = DIM) -> List[float]:
    """Return a vector orthogonal to _unit(1.0): first half +, second half -."""
    half  = dim // 2
    raw   = [1.0] * half + [-1.0] * (dim - half)
    norm  = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


def _make_registry(*domain_ids: str) -> DomainGraphRegistry:
    """Create a registry pre-populated with HOT domains (no centroid)."""
    reg = DomainGraphRegistry()
    for did in domain_ids:
        reg.register(did, display_name=did.replace("_", " ").title())
    return reg


def _make_entry(domain_id: str, dim: int = DIM, label: int = 1) -> ReplayEntry:
    return ReplayEntry(
        embedding=_unit(1.0, dim),
        query_text="test query",
        domain_id=domain_id,
        label=label,
    )


# ===========================================================================
# TIER 1 — STRUCTURAL
# ===========================================================================


class TestDriftDetectorStructure:
    """DriftDetector can be instantiated and exposes its documented interface."""

    def test_instantiation(self):
        dd = DriftDetector("cardiology", _unit(), DriftConfig())
        assert dd.domain_id == "cardiology"

    def test_sample_count_starts_at_zero(self):
        dd = DriftDetector("onco", _unit())
        assert dd.sample_count == 0

    def test_current_centroid_length(self):
        dd = DriftDetector("neuro", _unit())
        assert len(dd.current_centroid) == DIM

    def test_update_increments_sample_count(self):
        dd = DriftDetector("cardio", _unit())
        dd.update(_unit())
        assert dd.sample_count == 1

    def test_evaluate_returns_drift_signal(self):
        dd = DriftDetector("cardio", _unit())
        sig = dd.evaluate()
        assert isinstance(sig, DriftSignal)

    def test_drift_signal_fields_present(self):
        dd  = DriftDetector("cardio", _unit())
        sig = dd.evaluate()
        for attr in ("domain_id", "drift_score", "centroid_delta",
                     "variance_ratio", "sample_count", "triggered", "reason"):
            assert hasattr(sig, attr), f"DriftSignal missing field: {attr}"

    def test_reset_baseline_does_not_raise(self):
        dd = DriftDetector("onco", _unit())
        dd.reset_baseline()  # should be a no-op here

    def test_dim_mismatch_is_silently_skipped(self):
        dd = DriftDetector("onco", _unit(dim=DIM))
        dd.update([0.5, 0.5])   # wrong dim — should log warning, not raise
        assert dd.sample_count == 0


class TestDriftConfigStructure:
    def test_default_thresholds(self):
        cfg = DriftConfig()
        assert cfg.centroid_delta_threshold > 0
        assert cfg.variance_ratio_threshold > 0
        assert cfg.min_samples_to_evaluate > 0

    def test_custom_thresholds(self):
        cfg = DriftConfig(centroid_delta_threshold=0.05, variance_ratio_threshold=1.5)
        assert cfg.centroid_delta_threshold == pytest.approx(0.05)
        assert cfg.variance_ratio_threshold == pytest.approx(1.5)


class TestReplayBufferStructure:
    def test_instantiation(self):
        buf = ReplayBuffer("cardiology", max_size=100)
        assert buf.domain_id == "cardiology"

    def test_empty_buffer_len_zero(self):
        assert len(ReplayBuffer("onco")) == 0

    def test_push_increments_length(self):
        buf = ReplayBuffer("onco")
        buf.push(_make_entry("onco"))
        assert len(buf) == 1

    def test_wrong_domain_is_rejected(self):
        buf = ReplayBuffer("onco")
        buf.push(_make_entry("cardio"))   # domain_id mismatch
        assert len(buf) == 0

    def test_is_full_false_when_empty(self):
        assert not ReplayBuffer("onco", max_size=5).is_full

    def test_drain_empties_buffer(self):
        buf = ReplayBuffer("onco")
        buf.push(_make_entry("onco"))
        entries = buf.drain()
        assert len(entries) == 1
        assert len(buf) == 0

    def test_label_counts_returns_tuple(self):
        buf = ReplayBuffer("onco")
        buf.push(_make_entry("onco", label=1))
        buf.push(_make_entry("onco", label=0))
        pos, neg = buf.label_counts()
        assert pos == 1 and neg == 1


class TestReplayBufferManagerStructure:
    def test_get_or_create_returns_buffer(self):
        mgr = ReplayBufferManager()
        buf = mgr.get_or_create("cardio")
        assert isinstance(buf, ReplayBuffer)

    def test_same_domain_returns_same_buffer(self):
        mgr = ReplayBufferManager()
        assert mgr.get_or_create("onco") is mgr.get_or_create("onco")

    def test_sizes_returns_dict(self):
        mgr = ReplayBufferManager()
        mgr.push(_make_entry("onco"))
        sizes = mgr.sizes()
        assert "onco" in sizes and sizes["onco"] == 1

    def test_all_domain_ids(self):
        mgr = ReplayBufferManager()
        mgr.get_or_create("a")
        mgr.get_or_create("b")
        assert set(mgr.all_domain_ids()) == {"a", "b"}


class TestColdStorageManagerStructure:
    def test_instantiation(self):
        reg = _make_registry()
        mgr = ColdStorageManager(registry=reg)
        assert mgr is not None

    def test_reactivation_queue_size_starts_zero(self):
        mgr = ColdStorageManager(registry=_make_registry())
        assert mgr.reactivation_queue_size() == 0

    def test_replay_sizes_is_dict(self):
        mgr = ColdStorageManager(registry=_make_registry())
        assert isinstance(mgr.replay_sizes(), dict)

    def test_drift_detector_none_for_unknown_domain(self):
        mgr = ColdStorageManager(registry=_make_registry())
        assert mgr.drift_detector("nonexistent") is None

    def test_ensure_detector_creates_entry(self):
        reg = _make_registry("cardio")
        # Give the domain a centroid so ensure_detector can build the detector
        node = reg.get("cardio")
        node.meta["centroid"] = _unit()
        mgr = ColdStorageManager(registry=reg)
        mgr.ensure_detector("cardio")
        assert mgr.drift_detector("cardio") is not None


# ===========================================================================
# TIER 2 — SEMANTIC
# ===========================================================================


@pytest.mark.semantic
class TestDriftDetectorSemantic:
    """Verify the drift detector fires at the right threshold."""

    # min_samples_to_evaluate=1 so tests don't need to push 20+ samples
    _CFG = DriftConfig(
        centroid_delta_threshold=0.10,
        variance_ratio_threshold=100.0,  # effectively disabled — test centroid only
        min_samples_to_evaluate=1,
    )

    def test_stable_stream_does_not_trigger(self):
        dd  = DriftDetector("cardio", _unit(), self._CFG)
        vec = _unit()
        for _ in range(30):
            dd.update(vec)
        sig = dd.evaluate()
        assert not sig.triggered, (
            f"Stable stream triggered drift unexpectedly: delta={sig.centroid_delta:.4f}"
        )

    def test_large_centroid_shift_triggers(self):
        """Feed the opposite direction vector — centroid delta should exceed 0.10."""
        dd       = DriftDetector("cardio", _unit(1.0), self._CFG)
        opposite = _unit(-1.0)
        # Push a flood of opposite-direction embeddings to shift the EMA centroid
        for _ in range(200):
            dd.update(opposite)
        sig = dd.evaluate()
        assert sig.triggered, (
            f"Large centroid shift did not trigger: delta={sig.centroid_delta:.4f}"
        )

    def test_drift_score_in_range(self):
        dd = DriftDetector("onco", _unit(), self._CFG)
        for _ in range(5):
            dd.update(_unit())
        sig = dd.evaluate()
        assert 0.0 <= sig.drift_score <= 1.0

    def test_reset_baseline_clears_delta(self):
        dd = DriftDetector("onco", _unit(1.0), self._CFG)
        for _ in range(200):
            dd.update(_unit(-1.0))
        assert dd.evaluate().triggered  # confirm drift first
        dd.reset_baseline()
        # After reset the baseline is the current centroid — delta resets to ~0
        sig = dd.evaluate()
        assert sig.centroid_delta < self._CFG.centroid_delta_threshold, (
            f"Baseline reset did not clear centroid delta: {sig.centroid_delta:.4f}"
        )

    def test_insufficient_samples_returns_not_triggered(self):
        cfg = DriftConfig(min_samples_to_evaluate=50)
        dd  = DriftDetector("onco", _unit(), cfg)
        dd.update(_unit(-1.0))
        sig = dd.evaluate()
        assert not sig.triggered
        assert "Insufficient" in sig.reason


@pytest.mark.semantic
class TestReplayBufferSemantic:
    """Verify capacity enforcement, FIFO eviction, sample behaviour."""

    def test_capacity_enforced(self):
        buf = ReplayBuffer("onco", max_size=5)
        for _ in range(10):
            buf.push(_make_entry("onco"))
        assert len(buf) <= 5

    def test_is_full_after_capacity_reached(self):
        buf = ReplayBuffer("onco", max_size=3)
        for _ in range(3):
            buf.push(_make_entry("onco"))
        assert buf.is_full

    def test_sample_returns_correct_count(self):
        buf = ReplayBuffer("onco", max_size=20)
        for _ in range(10):
            buf.push(_make_entry("onco"))
        samples = buf.sample(3)
        assert len(samples) == 3

    def test_sample_k_larger_than_size_is_clamped(self):
        buf = ReplayBuffer("onco", max_size=10)
        buf.push(_make_entry("onco"))
        samples = buf.sample(50)
        assert len(samples) <= 1

    def test_drain_returns_all_and_clears(self):
        buf = ReplayBuffer("onco", max_size=10)
        for _ in range(5):
            buf.push(_make_entry("onco"))
        entries = buf.drain()
        assert len(entries) == 5
        assert len(buf) == 0

    def test_positive_negative_label_counts(self):
        buf = ReplayBuffer("onco", max_size=20)
        for _ in range(4):
            buf.push(_make_entry("onco", label=1))
        for _ in range(2):
            buf.push(_make_entry("onco", label=0))
        pos, neg = buf.label_counts()
        assert pos == 4 and neg == 2


@pytest.mark.semantic
class TestColdStorageManagerSemantic:
    """Behavioural tests for ColdStorageManager using real registry + modules."""

    _DRIFT_CFG = DriftConfig(
        centroid_delta_threshold=0.10,
        variance_ratio_threshold=100.0,
        min_samples_to_evaluate=1,
    )

    def _setup(self, *domain_ids: str, centroid: List[float] | None = None):
        reg = _make_registry(*domain_ids)
        c   = centroid or _unit()
        for did in domain_ids:
            reg.get(did).meta["centroid"] = c
        mgr = ColdStorageManager(
            registry=reg,
            drift_config=self._DRIFT_CFG,
            reactivation_demand_threshold=3,
        )
        return reg, mgr

    def test_on_query_routed_returns_none_when_stable(self):
        reg, mgr = self._setup("cardio")
        result = mgr.on_query_routed("cardio", _unit(), "stable query")
        assert result is None

    def test_on_query_routed_returns_signal_on_drift(self):
        reg, mgr = self._setup("cardio", centroid=_unit(1.0))
        opposite = _unit(-1.0)
        signal   = None
        for _ in range(200):
            signal = mgr.on_query_routed("cardio", opposite, "shifted query")
        # At least the last call should have returned a DriftSignal
        assert signal is not None, "Expected DriftSignal after large centroid shift"
        assert isinstance(signal, DriftSignal)
        assert signal.triggered

    def test_replay_buffer_populated_after_routing(self):
        reg, mgr = self._setup("onco")
        for _ in range(5):
            mgr.on_query_routed("onco", _unit(), "q")
        sizes = mgr.replay_sizes()
        assert sizes.get("onco", 0) == 5

    def test_cold_store_transitions_domain_state(self):
        from mycelium.domain_graph.state import DomainState
        reg, mgr = self._setup("neuro")
        mgr.cold_store("neuro", reason="inactivity")
        node = reg.get("neuro")
        assert node.state == DomainState.COLD

    def test_cold_domain_hit_queues_reactivation_after_threshold(self):
        reg, mgr = self._setup("neuro")
        mgr.cold_store("neuro", reason="inactivity")
        # Threshold is 3 — hit 3 times
        results = [mgr.on_cold_domain_hit("neuro", 0.8, "q") for _ in range(3)]
        assert results[-1] is True, "Third hit should queue reactivation"
        assert mgr.reactivation_queue_size() >= 1

    def test_pop_reactivation_returns_ticket(self):
        reg, mgr = self._setup("neuro")
        mgr.cold_store("neuro", reason="inactivity")
        for _ in range(3):
            mgr.on_cold_domain_hit("neuro", 0.8, "q")
        ticket = mgr.pop_reactivation()
        assert ticket is not None
        assert ticket.domain_id == "neuro"

    def test_pop_empty_queue_returns_none(self):
        _, mgr = self._setup("cardio")
        assert mgr.pop_reactivation() is None
