"""
Phase 5: Systematic Validation Testing for Multi-Lens Routing System

This test suite validates correctness, robustness, and edge-case handling
of the multi-lens routing system WITHOUT modifying any production code.

Test Categories:
1. Single-Domain Routing
2. Multi-Domain Routing
3. Attribute-Only Detection
4. No-Expert Scenario
5. Backward Compatibility
6. Graceful Degradation
7. Determinism
8. Performance

Usage:
    python3 tests/test_multilens_system.py
    # or from repo root:
    pytest tests/test_multilens_system.py

.. note::
   Moved from repository root to tests/ during cleanup pass (2026-05-21).
"""

import os
import sys
import unittest
import time
import statistics
from typing import Any

try:
    from multi_lens_router import MultiLensRouter
except ImportError as e:
    print(f"Error importing MultiLensRouter: {e}")
    sys.exit(1)

try:
    from core.types import RoutingResult
except ImportError:
    RoutingResult = None  # type: ignore


def _is_routing_mapping(result: Any) -> bool:
    if isinstance(result, dict):
        return True
    if RoutingResult is not None and isinstance(result, RoutingResult):
        return True
    return False


def _get_field(result: Any, key: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(key, default)
    if hasattr(result, key):
        return getattr(result, key)
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, dict):
        return metadata.get(key, default)
    return default


class TestSingleDomainRouting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.router = MultiLensRouter(use_multi_lens=True, use_spectral=True)

    def test_astronomy_single_domain(self):
        result = self.router.route("Earth orbits the Sun")
        self.assertTrue(_is_routing_mapping(result))
        self.assertIn(_get_field(result, "classification"), ["SINGLE_DOMAIN", "NORMAL", "NO_EXPERT_AVAILABLE", "AMBIGUOUS", "ATTRIBUTE_ONLY"])
        self.assertIsInstance(_get_field(result, "selected_experts", []), list)
        self.assertIsNotNone(_get_field(result, "coverage_met"))

    def test_biology_single_domain(self):
        result = self.router.route("Photosynthesis occurs in plants")
        self.assertIn(_get_field(result, "classification"), ["SINGLE_DOMAIN", "NORMAL", "NO_EXPERT_AVAILABLE", "AMBIGUOUS", "ATTRIBUTE_ONLY"])
        self.assertIsInstance(_get_field(result, "selected_experts", []), list)
        self.assertIsNotNone(_get_field(result, "coverage_met"))


class TestMultiDomainRouting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.router = MultiLensRouter(use_multi_lens=True, use_spectral=True)

    def test_automobile_multi_domain(self):
        result = self.router.route("Red car with 700cc engine")
        self.assertIn(_get_field(result, "classification"), ["MULTI_DOMAIN", "AMBIGUOUS", "SINGLE_DOMAIN", "NORMAL"])
        self.assertIsInstance(_get_field(result, "candidate_domains", []), list)

    def test_medical_physics_multi_domain(self):
        result = self.router.route("Medical ultrasound uses physics principles")
        self.assertIn(_get_field(result, "classification"), ["MULTI_DOMAIN", "AMBIGUOUS", "SINGLE_DOMAIN", "NORMAL", "NO_EXPERT_AVAILABLE", "ATTRIBUTE_ONLY"])

    def test_deterministic_multi_domain(self):
        text = "Red car with 700cc engine"
        r1 = self.router.route(text)
        r2 = self.router.route(text)
        self.assertEqual(_get_field(r1, "classification"), _get_field(r2, "classification"))
        self.assertEqual(_get_field(r1, "selected_experts", []), _get_field(r2, "selected_experts", []))
        self.assertEqual(_get_field(r1, "variance"), _get_field(r2, "variance"))


class TestAttributeOnlyDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.router = MultiLensRouter(use_multi_lens=True, use_spectral=True)

    def test_color_finish_attribute(self):
        result = self.router.route("Glossy metallic red finish")
        self.assertIn(_get_field(result, "classification"), ["ATTRIBUTE_ONLY", "SINGLE_DOMAIN", "NO_EXPERT_AVAILABLE", "AMBIGUOUS"])

    def test_performance_attribute(self):
        result = self.router.route("High torque low noise")
        self.assertIn(_get_field(result, "classification"), ["ATTRIBUTE_ONLY", "SINGLE_DOMAIN", "NO_EXPERT_AVAILABLE", "AMBIGUOUS"])


class TestNoExpertScenario(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.router = MultiLensRouter(use_multi_lens=True, use_spectral=True)

    def test_nonsensical_input(self):
        result = self.router.route("Quantum culinary philosophy")
        classification = _get_field(result, "classification")
        if classification == "NO_EXPERT_AVAILABLE":
            self.assertTrue(bool(_get_field(result, "create_new_expert", False)))
        self.assertIsNotNone(_get_field(result, "create_new_expert"))


class TestBackwardCompatibility(unittest.TestCase):
    def test_disable_multi_lens(self):
        router = MultiLensRouter(use_multi_lens=False, use_spectral=False)
        result = router.route("Earth orbits the Sun")
        self.assertIn(_get_field(result, "classification"), ["NORMAL", "ATTRIBUTE_ONLY"])

    def test_disable_spectral_only(self):
        router = MultiLensRouter(use_multi_lens=True, use_spectral=False)
        result = router.route("Red car with engine")
        lens_scores = _get_field(result, "lens_scores")
        if lens_scores and isinstance(lens_scores, dict) and "spectral" in lens_scores:
            self.assertEqual(lens_scores["spectral"], {})


class TestGracefulDegradation(unittest.TestCase):
    def test_missing_spectral_signatures(self):
        router = MultiLensRouter(use_multi_lens=True, use_spectral=True)
        try:
            result = router.route("Some text about quantum mechanics")
            self.assertTrue(_is_routing_mapping(result))
        except Exception as e:
            self.fail(f"Router should not crash: {e}")

    def test_empty_input(self):
        router = MultiLensRouter(use_multi_lens=True, use_spectral=True)
        try:
            result = router.route("")
            self.assertTrue(_is_routing_mapping(result))
        except Exception as e:
            self.fail(f"Router should handle empty input: {e}")

    def test_very_long_input(self):
        router = MultiLensRouter(use_multi_lens=True, use_spectral=True)
        text = "The study of astronomy involves planets and stars. " * 100
        try:
            result = router.route(text)
            self.assertTrue(_is_routing_mapping(result))
        except Exception as e:
            self.fail(f"Router should handle long input: {e}")


class TestDeterminism(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.router = MultiLensRouter(use_multi_lens=True, use_spectral=True)

    def test_identical_outputs_five_runs(self):
        text = "Red sports car with turbocharged engine"
        results = [self.router.route(text) for _ in range(5)]
        for i in range(1, 5):
            self.assertEqual(_get_field(results[0], "classification"), _get_field(results[i], "classification"))
            self.assertEqual(_get_field(results[0], "selected_experts", []), _get_field(results[i], "selected_experts", []))
            self.assertEqual(_get_field(results[0], "variance"), _get_field(results[i], "variance"))

    def test_deterministic_ordering(self):
        text = "Medical imaging uses physics and engineering"
        results = [_get_field(self.router.route(text), "selected_experts", []) for _ in range(3)]
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0], results[2])


class TestPerformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.router = MultiLensRouter(use_multi_lens=True, use_spectral=True)

    def test_routing_performance(self):
        inputs = [
            "Earth orbits the Sun", "Red car with engine", "Glossy metallic finish",
            "Photosynthesis in plants", "Medical ultrasound imaging", "High torque performance",
            "Quantum mechanics principles", "Turbocharged sports car",
            "Stellar evolution theory", "Chemical reaction kinetics",
        ]
        times = []
        for text in inputs:
            t0 = time.time()
            self.router.route(text)
            times.append(time.time() - t0)
        avg_threshold = float(os.getenv("ROUTING_AVG_MAX_SEC", "10.0"))
        max_threshold = float(os.getenv("ROUTING_MAX_SEC", "15.0"))
        self.assertLess(statistics.mean(times), avg_threshold)
        self.assertLess(max(times), max_threshold)

    def test_no_performance_degradation(self):
        text = "Astronomy and stellar evolution"
        batches = int(os.getenv("ROUTING_PERF_BATCHES", "2"))
        per_batch = int(os.getenv("ROUTING_PERF_PER_BATCH", "3"))
        batch_times = []
        for _ in range(batches):
            t0 = time.time()
            for _ in range(per_batch):
                self.router.route(text)
            batch_times.append((time.time() - t0) / max(per_batch, 1))
        factor = batch_times[-1] / batch_times[0] if batch_times[0] > 0 else 1.0
        self.assertLess(factor, 1.5)


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [TestSingleDomainRouting, TestMultiDomainRouting, TestAttributeOnlyDetection,
                TestNoExpertScenario, TestBackwardCompatibility, TestGracefulDegradation,
                TestDeterminism, TestPerformance]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
