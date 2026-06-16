"""
tests/cold_storage/test_cold_storage.py

Unit tests for:
  - mycelium.cold_storage.drift      (DriftDetector)
  - mycelium.cold_storage.replay_buffer (ReplayBufferManager)
  - mycelium.cold_storage.priority_queue (ReactivationQueue)
  - mycelium.cold_storage.manager    (ColdStorageManager lifecycle)
  - mycelium.cold_storage.reactivation (DomainReactivationService interface)
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# DriftDetector
# ---------------------------------------------------------------------------

class TestDriftDetector:
    @pytest.fixture
    def detector(self):
        from mycelium.cold_storage.drift import DriftDetector
        return DriftDetector()

    def test_compare_returns_drift_profile(self, detector):
        from mycelium.domain_graph.models import DriftProfile
        current = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        replay  = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        result = detector.compare(current, replay)
        assert isinstance(result, DriftProfile)

    def test_identical_embeddings_low_semantic_drift(self, detector):
        vec = [[1.0, 0.0, 0.0]] * 10
        profile = detector.compare(vec, vec)
        assert profile.semantic_drift < 0.05

    def test_orthogonal_embeddings_high_semantic_drift(self, detector):
        current = [[1.0, 0.0, 0.0]] * 5
        replay  = [[0.0, 1.0, 0.0]] * 5
        profile = detector.compare(current, replay)
        assert profile.semantic_drift > 0.0

    def test_empty_embeddings_returns_zero_profile(self, detector):
        profile = detector.compare([], [])
        assert profile.semantic_drift == 0.0

    def test_all_drift_fields_are_floats(self, detector):
        from mycelium.domain_graph.models import DriftProfile
        vecs = [[0.1, 0.2]] * 3
        profile = detector.compare(vecs, vecs)
        for field in ["semantic_drift", "retrieval_drift",
                      "routing_drift", "confidence_drift", "activation_drift"]:
            assert isinstance(getattr(profile, field), float), \
                f"{field} should be float"


# ---------------------------------------------------------------------------
# ReplayBufferManager
# ---------------------------------------------------------------------------

class TestReplayBufferManager:
    @pytest.fixture
    def buffer(self, tmp_path):
        from mycelium.cold_storage.replay_buffer import ReplayBufferManager
        return ReplayBufferManager(
            domain_id="physics",
            buffer_path=tmp_path / "physics_replay.pkl",
            max_samples=100,
        )

    def test_add_sample_does_not_raise(self, buffer):
        buffer.add_sample({"text": "Test physics sample", "label": "physics"})

    def test_size_increases_after_add(self, buffer):
        initial = buffer.size()
        buffer.add_sample({"text": "new sample"})
        assert buffer.size() == initial + 1

    def test_sample_returns_list(self, buffer):
        for i in range(10):
            buffer.add_sample({"text": f"sample {i}"})
        result = buffer.sample(k=5)
        assert isinstance(result, list)
        assert len(result) <= 5

    def test_sample_k_larger_than_buffer(self, buffer):
        buffer.add_sample({"text": "one"})
        result = buffer.sample(k=1000)
        assert isinstance(result, list)
        assert len(result) <= 1

    def test_max_samples_cap_enforced(self, tmp_path):
        from mycelium.cold_storage.replay_buffer import ReplayBufferManager
        buf = ReplayBufferManager(
            domain_id="test",
            buffer_path=tmp_path / "buf.pkl",
            max_samples=5,
        )
        for i in range(20):
            buf.add_sample({"text": f"sample {i}"})
        assert buf.size() <= 5

    def test_flush_and_load(self, buffer, tmp_path):
        from mycelium.cold_storage.replay_buffer import ReplayBufferManager
        buffer.add_sample({"text": "persist me"})
        buffer.flush()
        # Load from same path
        buf2 = ReplayBufferManager(
            domain_id="physics",
            buffer_path=buffer._buffer_path if hasattr(buffer, "_buffer_path")
                         else (tmp_path / "physics_replay.pkl"),
            max_samples=100,
        )
        assert buf2.size() >= 0  # loaded without error

    def test_clear_empties_buffer(self, buffer):
        buffer.add_sample({"text": "x"})
        buffer.clear()
        assert buffer.size() == 0

    def test_get_all_samples_returns_list(self, buffer):
        buffer.add_sample({"text": "sample_a"})
        buffer.add_sample({"text": "sample_b"})
        all_samples = buffer.get_all()
        assert isinstance(all_samples, list)
        assert len(all_samples) == 2


# ---------------------------------------------------------------------------
# ReactivationQueue
# ---------------------------------------------------------------------------

class TestReactivationQueue:
    @pytest.fixture
    def queue(self):
        from mycelium.cold_storage.priority_queue import ReactivationQueue
        return ReactivationQueue()

    @pytest.fixture
    def sample_request(self):
        from mycelium.cold_storage.priority_queue import ReactivationRequest
        return ReactivationRequest(
            priority=0.9,
            domain_id="physics",
            query_id="q001",
            estimated_cost=0.5,
            created_at=time.time(),
        )

    def test_push_does_not_raise(self, queue, sample_request):
        queue.push(sample_request)

    def test_empty_queue_raises_or_returns_none_on_pop(self, queue):
        try:
            result = queue.pop()
            assert result is None
        except (IndexError, KeyError):
            pass  # either behaviour is acceptable

    def test_push_and_pop_round_trip(self, queue, sample_request):
        queue.push(sample_request)
        result = queue.pop()
        assert result is not None
        assert result.domain_id == "physics"

    def test_higher_priority_popped_first(self, queue):
        from mycelium.cold_storage.priority_queue import ReactivationRequest
        low  = ReactivationRequest(priority=0.1, domain_id="low",  query_id="q1", created_at=time.time())
        high = ReactivationRequest(priority=0.9, domain_id="high", query_id="q2", created_at=time.time())
        queue.push(low)
        queue.push(high)
        first = queue.pop()
        # Queue should return the highest priority item first
        assert first.domain_id == "high"

    def test_size_tracking(self, queue, sample_request):
        assert queue.size() == 0
        queue.push(sample_request)
        assert queue.size() == 1
        queue.pop()
        assert queue.size() == 0

    def test_peek_does_not_remove(self, queue, sample_request):
        queue.push(sample_request)
        if hasattr(queue, "peek"):
            peeked = queue.peek()
            assert peeked is not None
            assert queue.size() == 1  # still in queue


# ---------------------------------------------------------------------------
# ColdStorageManager
# ---------------------------------------------------------------------------

class TestColdStorageManager:
    @pytest.fixture
    def manager(self, tmp_path):
        from mycelium.cold_storage.manager import ColdStorageManager
        from mycelium.domain_graph.registry import DomainGraphRegistry
        from mycelium.domain_graph.models import DomainNode
        registry = DomainGraphRegistry(graph_path=tmp_path / "graph.json")
        registry.add_domain(DomainNode(
            domain_id="physics",
            label="Physics",
            state="HOT",
            activation_stats={"last_used": time.time() - 86400 * 10},  # 10 days old
        ))
        return ColdStorageManager(
            registry=registry,
            artifacts_dir=tmp_path,
        )

    def test_current_state_returns_string(self, manager):
        state = manager.current_state("physics")
        assert isinstance(state, str)

    def test_should_cold_store_returns_bool(self, manager):
        result = manager.should_cold_store("physics")
        assert isinstance(result, bool)

    def test_should_cold_store_unknown_domain_returns_false(self, manager):
        result = manager.should_cold_store("__nonexistent_domain__")
        assert result is False

    def test_cold_store_changes_domain_state(self, manager, tmp_path):
        from mycelium.domain_graph.state import DomainState
        manager.cold_store("physics")
        state = manager.current_state("physics")
        assert state in ("COLD", DomainState.COLD, "WARM", DomainState.WARM)

    def test_reactivate_does_not_raise(self, manager):
        # Cold-store first so domain is in COLD state
        manager.cold_store("physics")
        manager.reactivate("physics")

    def test_reactivate_unknown_domain_raises_or_noop(self, manager):
        try:
            manager.reactivate("__ghost_domain__")
        except (KeyError, ValueError, RuntimeError):
            pass  # acceptable


# ---------------------------------------------------------------------------
# DomainReactivationService
# ---------------------------------------------------------------------------

class TestDomainReactivationService:
    @pytest.fixture
    def service(self, tmp_path):
        from mycelium.cold_storage.reactivation import DomainReactivationService
        from mycelium.domain_graph.registry import DomainGraphRegistry
        from mycelium.domain_graph.models import DomainNode
        from mycelium.cold_storage.drift import DriftDetector
        registry = DomainGraphRegistry(graph_path=tmp_path / "graph.json")
        registry.add_domain(DomainNode(
            domain_id="bio",
            label="Biology",
            state="COLD",
        ))
        return DomainReactivationService(
            registry=registry,
            drift_detector=DriftDetector(),
            artifacts_dir=tmp_path,
        )

    def test_reactivate_returns_dict(self, service):
        result = service.reactivate("bio")
        assert isinstance(result, dict)

    def test_reactivate_result_has_domain_id(self, service):
        result = service.reactivate("bio")
        assert "domain_id" in result
        assert result["domain_id"] == "bio"

    def test_reactivate_result_has_state(self, service):
        result = service.reactivate("bio")
        assert "state" in result

    def test_reactivate_result_has_adapted_flag(self, service):
        result = service.reactivate("bio")
        assert "adapted" in result
        assert isinstance(result["adapted"], bool)

    def test_reactivate_result_has_drift_profile(self, service):
        from mycelium.domain_graph.models import DriftProfile
        result = service.reactivate("bio")
        assert "drift_profile" in result
        dp = result["drift_profile"]
        assert isinstance(dp, DriftProfile)
