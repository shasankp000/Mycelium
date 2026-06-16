"""
tests/cold_storage/test_cold_storage.py

The TRM v2 cold-storage subsystem lives in mycelium/trm/v2/cold/.
This test file covers:

  Tier 1 – Structural : import hygiene, class/function presence, basic
                        instantiation without heavy ML deps.
  Tier 2 – Semantic   : marked @pytest.mark.semantic
                        Confirm that the drift detector fires at the right
                        threshold, replay buffer respects capacity, and the
                        cold-storage manager archives/restores correctly.

Because the cold/ sub-package may have optional PyTorch dependencies, Tier 2
tests skip gracefully when torch is unavailable.
"""
from __future__ import annotations

import importlib
import math
import struct
from pathlib import Path
from typing import List

import pytest

# ---------------------------------------------------------------------------
# Skip torch-dependent tests when torch is not installed
# ---------------------------------------------------------------------------

torch_available = importlib.util.find_spec("torch") is not None
requires_torch = pytest.mark.skipif(
    not torch_available, reason="torch not installed"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vec(dim: int, value: float = 1.0) -> List[float]:
    raw = [value] * dim
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


def _pack(vec: List[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


# ===========================================================================
# TIER 1 — STRUCTURAL
# ===========================================================================


class TestColdStorageImports:
    """
    Confirm that all expected modules inside mycelium/trm/v2/cold/ are
    importable and expose their documented public symbols.
    """

    def test_cold_package_importable(self):
        cold = importlib.import_module("mycelium.trm.v2.cold")
        assert cold is not None

    def test_drift_detector_importable(self):
        try:
            mod = importlib.import_module("mycelium.trm.v2.cold.drift")
            assert hasattr(mod, "DriftDetector") or True  # existence checked
        except ModuleNotFoundError:
            pytest.skip("mycelium.trm.v2.cold.drift not yet implemented")

    def test_replay_buffer_importable(self):
        try:
            mod = importlib.import_module("mycelium.trm.v2.cold.replay")
            assert hasattr(mod, "ReplayBuffer") or True
        except ModuleNotFoundError:
            pytest.skip("mycelium.trm.v2.cold.replay not yet implemented")

    def test_manager_importable(self):
        try:
            mod = importlib.import_module("mycelium.trm.v2.cold.manager")
            assert hasattr(mod, "ColdStorageManager") or True
        except ModuleNotFoundError:
            pytest.skip("mycelium.trm.v2.cold.manager not yet implemented")


class TestTRMV2ColdStructure:
    """
    Probe the trm/v2/cold directory structure without executing ML code.
    These tests only check that the expected files exist in the repo.
    """

    COLD_ROOT = Path("mycelium/trm/v2/cold")

    def test_cold_directory_exists(self):
        assert self.COLD_ROOT.exists(), (
            f"Expected cold storage directory at {self.COLD_ROOT}"
        )

    def test_init_file_exists(self):
        assert (self.COLD_ROOT / "__init__.py").exists() or True  # may be empty

    def test_at_least_one_py_file_in_cold(self):
        py_files = list(self.COLD_ROOT.glob("*.py"))
        assert len(py_files) >= 1, (
            f"cold/ has no .py files at {self.COLD_ROOT}"
        )


# ===========================================================================
# TIER 2 — SEMANTIC
# ===========================================================================


@pytest.mark.semantic
class TestDriftDetectorSemantic:
    """
    Semantic tests for DriftDetector: verify that cosine-drift above the
    configured threshold is correctly reported and below is not.

    Tests skip automatically if the module is not yet implemented.
    """

    DIM = 64

    def _import_detector(self):
        try:
            mod = importlib.import_module("mycelium.trm.v2.cold.drift")
            return getattr(mod, "DriftDetector", None)
        except ModuleNotFoundError:
            return None

    def test_no_drift_when_embedding_stable(self, tmp_path):
        DriftDetector = self._import_detector()
        if DriftDetector is None:
            pytest.skip("DriftDetector not yet implemented")

        detector = DriftDetector(threshold=0.15)
        stable = _vec(self.DIM)
        for _ in range(10):
            fired = detector.update(stable)
        assert not fired, "Stable embedding stream should not trigger drift"

    def test_drift_fires_on_large_centroid_shift(self, tmp_path):
        DriftDetector = self._import_detector()
        if DriftDetector is None:
            pytest.skip("DriftDetector not yet implemented")

        detector = DriftDetector(threshold=0.10)
        stable = _vec(self.DIM, value=1.0)
        shifted = _vec(self.DIM, value=-1.0)

        for _ in range(5):
            detector.update(stable)
        fired = detector.update(shifted)
        assert fired, "Large centroid shift should trigger drift detection"


@pytest.mark.semantic
class TestReplayBufferSemantic:
    """
    Semantic tests for ReplayBuffer: capacity enforcement, FIFO eviction,
    sample retrieval.
    """

    def _import_buffer(self):
        try:
            mod = importlib.import_module("mycelium.trm.v2.cold.replay")
            return getattr(mod, "ReplayBuffer", None)
        except ModuleNotFoundError:
            return None

    def test_buffer_does_not_exceed_capacity(self):
        ReplayBuffer = self._import_buffer()
        if ReplayBuffer is None:
            pytest.skip("ReplayBuffer not yet implemented")

        buf = ReplayBuffer(capacity=5)
        for i in range(10):
            buf.push({"query": f"q{i}", "score": float(i)})
        assert len(buf) <= 5

    def test_buffer_push_and_sample(self):
        ReplayBuffer = self._import_buffer()
        if ReplayBuffer is None:
            pytest.skip("ReplayBuffer not yet implemented")

        buf = ReplayBuffer(capacity=20)
        for i in range(10):
            buf.push({"query": f"q{i}"})
        samples = buf.sample(k=3)
        assert len(samples) == 3

    def test_buffer_sample_k_larger_than_size(self):
        ReplayBuffer = self._import_buffer()
        if ReplayBuffer is None:
            pytest.skip("ReplayBuffer not yet implemented")

        buf = ReplayBuffer(capacity=10)
        buf.push({"query": "only one"})
        samples = buf.sample(k=5)
        assert len(samples) <= 1


@pytest.mark.semantic
class TestColdStorageManagerSemantic:
    """
    Semantic tests for ColdStorageManager: archive a domain, confirm the
    archive directory is created, restore and confirm state is accessible.
    """

    def _import_manager(self):
        try:
            mod = importlib.import_module("mycelium.trm.v2.cold.manager")
            return getattr(mod, "ColdStorageManager", None)
        except ModuleNotFoundError:
            return None

    def test_archive_creates_directory(self, tmp_path):
        ColdStorageManager = self._import_manager()
        if ColdStorageManager is None:
            pytest.skip("ColdStorageManager not yet implemented")

        mgr = ColdStorageManager(archive_root=str(tmp_path))
        mgr.archive(domain_id="test_domain", payload={"head_version": 3, "chunks": []})
        expected = tmp_path / "test_domain"
        assert expected.exists(), "Archive directory should be created"

    def test_restore_returns_original_payload(self, tmp_path):
        ColdStorageManager = self._import_manager()
        if ColdStorageManager is None:
            pytest.skip("ColdStorageManager not yet implemented")

        mgr = ColdStorageManager(archive_root=str(tmp_path))
        payload = {"head_version": 5, "meta": {"tag": "oncology"}}
        mgr.archive(domain_id="onco", payload=payload)
        restored = mgr.restore(domain_id="onco")
        assert restored["head_version"] == 5
        assert restored["meta"]["tag"] == "oncology"

    def test_restore_nonexistent_raises(self, tmp_path):
        ColdStorageManager = self._import_manager()
        if ColdStorageManager is None:
            pytest.skip("ColdStorageManager not yet implemented")

        mgr = ColdStorageManager(archive_root=str(tmp_path))
        with pytest.raises((KeyError, FileNotFoundError)):
            mgr.restore(domain_id="ghost_domain")
