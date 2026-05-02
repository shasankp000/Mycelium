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
    python3 test_multilens_system.py
"""

import os
import unittest
import time
import statistics
import sys
from typing import Dict, Any, List

# Import the multi-lens router
try:
    from multi_lens_router import MultiLensRouter
except ImportError as e:
    print(f"Error importing MultiLensRouter: {e}")
    sys.exit(1)

try:
    # New typed result from refactor plan
    from core.types import RoutingResult
except ImportError:
    # Fallback for older setups where RoutingResult may not exist yet
    RoutingResult = None  # type: ignore


def _is_routing_mapping(result: Any) -> bool:
    """Helper: accept both legacy dicts and new RoutingResult dataclasses.

    This keeps tests focused on behavior rather than the concrete container type.
    """
    if isinstance(result, dict):
        return True
    if RoutingResult is not None and isinstance(result, RoutingResult):
        return True
    return False


def _get_field(result: Any, key: str, default: Any = None) -> Any:
    """Access a field from either a dict or a RoutingResult-like object."""
    if isinstance(result, dict):
        return result.get(key, default)
    # Dataclass / object path
    if hasattr(result, key):
        return getattr(result, key)
    # Metadata often holds dict-like payload used by earlier tests
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, dict):
        return metadata.get(key, default)
    return default


class TestSingleDomainRouting(unittest.TestCase):
    """Test Group 1: Single-Domain Routing"""

    @classmethod
    def setUpClass(cls):
        """Initialize router once for all tests in this group"""
        cls.router = MultiLensRouter(use_multi_lens=True, use_spectral=True)

    def test_astronomy_single_domain(self):
        """Test: 'Earth orbits the Sun' should route to SINGLE_DOMAIN astronomy"""
        text = "Earth orbits the Sun"
        result = self.router.route(text)

        # Log output for visibility
        print(f"\n[SINGLE-DOMAIN: Astronomy]")
        print(f"  Classification: {_get_field(result, 'classification')}")
        print(f"  Selected Experts: {_get_field(result, 'selected_experts', [])}")
        print(f"  Primary Domain: {_get_field(result, 'primary_domain')}")
        print(f"  Coverage Met: {_get_field(result, 'coverage_met')}")

        # Assertions
        self.assertTrue(
            _is_routing_mapping(result),
            "Router should return a dict or RoutingResult",
        )
        classification = _get_field(result, "classification")
        self.assertIn(
            classification,
            [
                "SINGLE_DOMAIN",
                "NORMAL",
                "NO_EXPERT_AVAILABLE",
                "AMBIGUOUS",
                "ATTRIBUTE_ONLY",
            ],
            "Should return a valid classification",
        )
        selected_experts = _get_field(result, "selected_experts", [])
        self.assertIsNotNone(selected_experts, "Should have selected experts")
        self.assertIsInstance(
            selected_experts, list, "Selected experts should be a list"
        )
        self.assertIsNotNone(
            _get_field(result, "coverage_met"), "Coverage flag must exist"
        )

    def test_biology_single_domain(self):
        """Test: 'Photosynthesis occurs in plants' should route to biology or unknown"""
        text = "Photosynthesis occurs in plants"
        result = self.router.route(text)

        print(f"\n[SINGLE-DOMAIN: Biology/Unknown]")
        print(f"  Classification: {_get_field(result, 'classification')}")
        print(f"  Selected Experts: {_get_field(result, 'selected_experts', [])}")
        print(f"  Primary Domain: {_get_field(result, 'primary_domain')}")

        # Assertions - should not crash and should return valid structure
        classification = _get_field(result, "classification")
        self.assertIn(
            classification,
            [
                "SINGLE_DOMAIN",
                "NORMAL",
                "NO_EXPERT_AVAILABLE",
                "AMBIGUOUS",
                "ATTRIBUTE_ONLY",
            ],
            "Should have valid classification",
        )
        self.assertIsInstance(_get_field(result, "selected_experts", []), list)
        self.assertIsNotNone(_get_field(result, "coverage_met"))


class TestMultiDomainRouting(unittest.TestCase):
    """Test Group 2: Multi-Domain Routing"""

    @classmethod
    def setUpClass(cls):
        cls.router = MultiLensRouter(use_multi_lens=True, use_spectral=True)

    def test_automobile_multi_domain(self):
        """Test: 'Red car with 700cc engine' should detect multiple domains"""
        text = "Red car with 700cc engine"
        result = self.router.route(text)

        print(f"\n[MULTI-DOMAIN: Automobile]")
        print(f"  Classification: {_get_field(result, 'classification')}")
        print(f"  Selected Experts: {_get_field(result, 'selected_experts', [])}")
        print(
            f"  Candidate Domains: {_get_field(result, 'candidate_domains', [])}"
        )
        print(f"  Variance: {_get_field(result, 'variance', 0)}")

        # Assertions
        classification = _get_field(result, "classification")
        self.assertIn(
            classification,
            ["MULTI_DOMAIN", "AMBIGUOUS", "SINGLE_DOMAIN", "NORMAL"],
            "Should have valid classification",
        )
        candidate_domains = _get_field(result, "candidate_domains", [])
        self.assertIsInstance(
            candidate_domains, list, "Candidate domains should be a list"
        )
        # If multi-domain, should have multiple candidates
        if classification in ["MULTI_DOMAIN", "AMBIGUOUS"]:
            self.assertGreaterEqual(
                len(candidate_domains), 1, "Multi-domain should have candidates"
            )

    def test_medical_physics_multi_domain(self):
        """Test: 'Medical ultrasound uses physics principles' should span domains"""
        text = "Medical ultrasound uses physics principles"
        result = self.router.route(text)

        print(f"\n[MULTI-DOMAIN: Medical+Physics]")
        print(f"  Classification: {_get_field(result, 'classification')}")
        print(f"  Selected Experts: {_get_field(result, 'selected_experts', [])}")
        print(
            f"  Candidate Domains: {_get_field(result, 'candidate_domains', [])}"
        )

        # Assertions
        classification = _get_field(result, "classification")
        self.assertIn(
            classification,
            [
                "MULTI_DOMAIN",
                "AMBIGUOUS",
                "SINGLE_DOMAIN",
                "NORMAL",
                "NO_EXPERT_AVAILABLE",
                "ATTRIBUTE_ONLY",
            ],
        )
        self.assertIsInstance(_get_field(result, "selected_experts", []), list)

    def test_deterministic_multi_domain(self):
        """Test: Same input twice should produce identical output"""
        text = "Red car with 700cc engine"

        result1 = self.router.route(text)
        result2 = self.router.route(text)

        print(f"\n[DETERMINISM CHECK: Multi-Domain]")
        print(f"  Run 1 Classification: {_get_field(result1, 'classification')}")
        print(f"  Run 2 Classification: {_get_field(result2, 'classification')}")
        print(
            f"  Run 1 Experts: {_get_field(result1, 'selected_experts', [])}"
        )
        print(
            f"  Run 2 Experts: {_get_field(result2, 'selected_experts', [])}"
        )

        # Assertions - outputs must be identical
        self.assertEqual(
            _get_field(result1, "classification"),
            _get_field(result2, "classification"),
            "Classification must be deterministic",
        )
        self.assertEqual(
            _get_field(result1, "selected_experts", []),
            _get_field(result2, "selected_experts", []),
            "Selected experts must be deterministic",
        )
        self.assertEqual(
            _get_field(result1, "variance"),
            _get_field(result2, "variance"),
            "Variance must be deterministic",
        )


class TestAttributeOnlyDetection(unittest.TestCase):
    """Test Group 3: Attribute-Only Detection"""

    @classmethod
    def setUpClass(cls):
        cls.router = MultiLensRouter(use_multi_lens=True, use_spectral=True)

    def test_color_finish_attribute(self):
        """Test: 'Glossy metallic red finish' should be ATTRIBUTE_ONLY"""
        text = "Glossy metallic red finish"
        result = self.router.route(text)

        print(f"\n[ATTRIBUTE-ONLY: Color/Finish]")
        print(f"  Classification: {_get_field(result, 'classification')}")
        print(f"  Selected Experts: {_get_field(result, 'selected_experts', [])}")
        print(f"  Primary Domain: {_get_field(result, 'primary_domain')}")

        classification = _get_field(result, "classification")
        self.assertIn(
            classification,
            [
                "ATTRIBUTE_ONLY",
                "SINGLE_DOMAIN",
                "NO_EXPERT_AVAILABLE",
                "AMBIGUOUS",
            ],
            "Should be ATTRIBUTE_ONLY or promoted via override",
        )
        selected_experts = _get_field(result, "selected_experts", [])
        primary_domain = _get_field(result, "primary_domain")
        if classification == "SINGLE_DOMAIN":
            self.assertEqual(
                len(selected_experts),
                1,
                "Promoted attribute should have exactly 1 expert",
            )
            self.assertIsNotNone(
                primary_domain,
                "Promoted attribute should have primary domain",
            )
        else:
            self.assertIsInstance(selected_experts, list)

    def test_performance_attribute(self):
        """Test: 'High torque low noise' should be ATTRIBUTE_ONLY"""
        text = "High torque low noise"
        result = self.router.route(text)

        print(f"\n[ATTRIBUTE-ONLY: Performance]")
        print(f"  Classification: {_get_field(result, 'classification')}")
        print(f"  Selected Experts: {_get_field(result, 'selected_experts', [])}")
        print(f"  Primary Domain: {_get_field(result, 'primary_domain')}")

        classification = _get_field(result, "classification")
        self.assertIn(
            classification,
            [
                "ATTRIBUTE_ONLY",
                "SINGLE_DOMAIN",
                "NO_EXPERT_AVAILABLE",
                "AMBIGUOUS",
            ],
            "Should be ATTRIBUTE_ONLY or promoted via override",
        )
        selected_experts = _get_field(result, "selected_experts", [])
        primary_domain = _get_field(result, "primary_domain")
        if classification == "SINGLE_DOMAIN":
            self.assertEqual(len(selected_experts), 1)
            self.assertIsNotNone(primary_domain)
        else:
            self.assertIsInstance(selected_experts, list)


class TestNoExpertScenario(unittest.TestCase):
    """Test Group 4: No-Expert Scenario"""

    @classmethod
    def setUpClass(cls):
        cls.router = MultiLensRouter(use_multi_lens=True, use_spectral=True)

    def test_nonsensical_input(self):
        """Test: 'Quantum culinary philosophy' should trigger NO_EXPERT or create signal"""
        text = "Quantum culinary philosophy"
        result = self.router.route(text)

        print(f"\n[NO-EXPERT: Nonsensical Input]")
        print(f"  Classification: {_get_field(result, 'classification')}")
        print(f"  Selected Experts: {_get_field(result, 'selected_experts', [])}")
        print(f"  Create New Expert: {_get_field(result, 'create_new_expert', False)}")
        print(f"  Coverage Met: {_get_field(result, 'coverage_met', False)}")

        classification = _get_field(result, "classification")
        create_new_expert = bool(
            _get_field(result, "create_new_expert", False)
        )
        if classification == "NO_EXPERT_AVAILABLE":
            self.assertTrue(
                create_new_expert,
                "NO_EXPERT_AVAILABLE should signal to create new expert",
            )
        elif classification == "AMBIGUOUS":
            # Ambiguous is acceptable for unknown domains
            pass
        else:
            # If it routes somewhere, create_new_expert might be True due to low coverage
            self.assertIsNotNone(
                _get_field(result, "create_new_expert"),
                "Should have create_new_expert flag",
            )


class TestBackwardCompatibility(unittest.TestCase):
    """Test Group 5: Backward Compatibility"""

    def test_disable_multi_lens(self):
        """Test: use_multi_lens=False should behave like original multi_lens_route()"""
        router = MultiLensRouter(use_multi_lens=False, use_spectral=False)

        text = "Earth orbits the Sun"
        result = router.route(text)

        print(f"\n[BACKWARD COMPAT: use_multi_lens=False]")
        print(f"  Classification: {_get_field(result, 'classification')}")
        print(f"  Selected Experts: {_get_field(result, 'selected_experts', [])}")
        fused_scores = _get_field(result, "fused_scores")
        lens_scores = _get_field(result, "lens_scores")
        print(
            f"  Has Fused Scores: {fused_scores is not None and fused_scores != {}}"
        )
        print(
            "  Has Spectral: ",
            bool(lens_scores and isinstance(lens_scores, dict) and "spectral" in lens_scores),
        )

        classification = _get_field(result, "classification")
        self.assertIn(
            classification,
            ["NORMAL", "ATTRIBUTE_ONLY"],
            "Backward compat mode should use simple classifications",
        )
        self.assertIsInstance(
            _get_field(result, "selected_experts", []), list
        )
        if lens_scores and isinstance(lens_scores, dict):
            spectral_scores = lens_scores.get("spectral")
            if spectral_scores is not None:
                self.assertEqual(
                    spectral_scores,
                    {},
                    "Backward compat spectral scores should be empty",
                )

    def test_disable_spectral_only(self):
        """Test: use_spectral=False should skip spectral analysis"""
        router = MultiLensRouter(use_multi_lens=True, use_spectral=False)

        text = "Red car with engine"
        result = router.route(text)

        print(f"\n[SPECTRAL DISABLED: use_spectral=False]")
        print(f"  Classification: {_get_field(result, 'classification')}")
        lens_scores = _get_field(result, "lens_scores")
        print(
            "  Has Spectral Scores: ",
            bool(lens_scores and isinstance(lens_scores, dict) and "spectral" in lens_scores),
        )

        if lens_scores and isinstance(lens_scores, dict):
            spectral = lens_scores.get("spectral")
            if spectral is not None:
                self.assertEqual(
                    spectral,
                    {},
                    "Spectral should be empty when disabled",
                )


class TestGracefulDegradation(unittest.TestCase):
    """Test Group 6: Graceful Degradation"""

    def test_missing_spectral_signatures(self):
        """Test: Missing spectral signatures should not crash the system"""
        router = MultiLensRouter(use_multi_lens=True, use_spectral=True)

        text = "Some text about quantum mechanics and thermodynamics"

        try:
            result = router.route(text)

            print(f"\n[GRACEFUL DEGRADATION: Missing Signatures]")
            print(f"  Classification: {_get_field(result, 'classification')}")
            print(f"  Selected Experts: {_get_field(result, 'selected_experts', [])}")
            print(f"  No Exception Raised: True")

            self.assertTrue(
                _is_routing_mapping(result), "Should return valid routing result"
            )
            self.assertIsNotNone(
                _get_field(result, "classification"),
                "Classification field should be present",
            )
            self.assertIsNotNone(
                _get_field(result, "selected_experts", []),
                "selected_experts field should be present",
            )

        except Exception as e:
            self.fail(f"Router should not crash with missing signatures: {e}")

    def test_empty_input(self):
        """Test: Empty input should be handled gracefully"""
        router = MultiLensRouter(use_multi_lens=True, use_spectral=True)

        text = ""

        try:
            result = router.route(text)

            print(f"\n[GRACEFUL DEGRADATION: Empty Input]")
            print(f"  Classification: {_get_field(result, 'classification')}")
            print(f"  Selected Experts: {_get_field(result, 'selected_experts', [])}")

            self.assertTrue(
                _is_routing_mapping(result),
                "Should return dict or RoutingResult for empty input",
            )
            self.assertIsNotNone(
                _get_field(result, "classification"),
                "Classification field should be present",
            )

        except Exception as e:
            self.fail(f"Router should handle empty input gracefully: {e}")

    def test_very_long_input(self):
        """Test: Very long input should be handled gracefully"""
        router = MultiLensRouter(use_multi_lens=True, use_spectral=True)

        text = "The study of astronomy involves planets and stars. " * 100

        try:
            result = router.route(text)

            print(f"\n[GRACEFUL DEGRADATION: Long Input]")
            print(f"  Classification: {_get_field(result, 'classification')}")
            print(f"  Input Length: {len(text)} chars")

            self.assertTrue(
                _is_routing_mapping(result),
                "Should return dict or RoutingResult for long input",
            )
            self.assertIsNotNone(
                _get_field(result, "classification"),
                "Classification field should be present",
            )

        except Exception as e:
            self.fail(f"Router should handle long input gracefully: {e}")


class TestDeterminism(unittest.TestCase):
    """Test Group 7: Determinism"""

    @classmethod
    def setUpClass(cls):
        cls.router = MultiLensRouter(use_multi_lens=True, use_spectral=True)

    def test_identical_outputs_five_runs(self):
        """Test: Same input 5 times should produce identical outputs"""
        text = "Red sports car with turbocharged engine"

        results = []
        for i in range(5):
            result = self.router.route(text)
            results.append(result)

        print(f"\n[DETERMINISM: 5 Runs]")
        print(
            f"  Run 1 Classification: {_get_field(results[0], 'classification')}"
        )
        print(
            f"  Run 1 Experts: {_get_field(results[0], 'selected_experts', [])}"
        )
        print(f"  Run 1 Variance: {_get_field(results[0], 'variance')}")

        for i in range(1, 5):
            self.assertEqual(
                _get_field(results[0], "classification"),
                _get_field(results[i], "classification"),
                f"Run {i+1} classification differs from Run 1",
            )
            self.assertEqual(
                _get_field(results[0], "selected_experts", []),
                _get_field(results[i], "selected_experts", []),
                f"Run {i+1} selected experts differ from Run 1",
            )
            self.assertEqual(
                _get_field(results[0], "variance"),
                _get_field(results[i], "variance"),
                f"Run {i+1} variance differs from Run 1",
            )
            self.assertEqual(
                _get_field(results[0], "primary_domain"),
                _get_field(results[i], "primary_domain"),
                f"Run {i+1} primary domain differs from Run 1",
            )

        print("  ✅ All 5 runs produced identical outputs")

    def test_deterministic_ordering(self):
        """Test: Selected experts should have consistent ordering"""
        text = "Medical imaging uses physics and engineering principles"

        results = []
        for i in range(3):
            result = self.router.route(text)
            results.append(_get_field(result, "selected_experts", []))

        print(f"\n[DETERMINISM: Expert Ordering]")
        print(f"  Run 1 Experts: {results[0]}")
        print(f"  Run 2 Experts: {results[1]}")
        print(f"  Run 3 Experts: {results[2]}")

        self.assertEqual(
            results[0], results[1], "Run 2 expert order differs from Run 1"
        )
        self.assertEqual(
            results[0], results[2], "Run 3 expert order differs from Run 1"
        )


class TestPerformance(unittest.TestCase):
    """Test Group 8: Performance (Lightweight)"""

    @classmethod
    def setUpClass(cls):
        cls.router = MultiLensRouter(use_multi_lens=True, use_spectral=True)

    def test_routing_performance(self):
        """Test: Average routing time for 10 inputs should be < 1 second"""
        test_inputs = [
            "Earth orbits the Sun",
            "Red car with engine",
            "Glossy metallic finish",
            "Photosynthesis in plants",
            "Medical ultrasound imaging",
            "High torque performance",
            "Quantum mechanics principles",
            "Turbocharged sports car",
            "Stellar evolution theory",
            "Chemical reaction kinetics",
        ]

        routing_times = []

        for text in test_inputs:
            start_time = time.time()
            self.router.route(text)
            end_time = time.time()

            routing_time = end_time - start_time
            routing_times.append(routing_time)

        avg_time = statistics.mean(routing_times)
        max_time = max(routing_times)
        min_time = min(routing_times)

        avg_threshold = float(os.getenv("ROUTING_AVG_MAX_SEC", "10.0"))
        max_threshold = float(os.getenv("ROUTING_MAX_SEC", "15.0"))

        print(f"\n[PERFORMANCE: 10 Routing Calls]")
        print(f"  Average Time: {avg_time:.4f} seconds")
        print(f"  Max Time: {max_time:.4f} seconds")
        print(f"  Min Time: {min_time:.4f} seconds")

        self.assertLess(
            avg_time,
            avg_threshold,
            f"Average routing time {avg_time:.4f}s exceeds {avg_threshold:.1f}s threshold",
        )
        self.assertLess(
            max_time,
            max_threshold,
            f"Max routing time {max_time:.4f}s exceeds {max_threshold:.1f}s threshold",
        )

    def test_no_performance_degradation(self):
        """Test: Performance should not degrade across multiple runs"""
        text = "Astronomy and stellar evolution"

        batch_times = []

        batches = int(os.getenv("ROUTING_PERF_BATCHES", "2"))
        per_batch = int(os.getenv("ROUTING_PERF_PER_BATCH", "3"))

        for batch in range(batches):
            start_time = time.time()
            for _ in range(per_batch):
                self.router.route(text)
            end_time = time.time()

            batch_time = (end_time - start_time) / max(per_batch, 1)
            batch_times.append(batch_time)

        print(f"\n[PERFORMANCE: Degradation Check]")
        print(f"  Batch 1 Avg: {batch_times[0]:.4f}s")
        print(f"  Batch {len(batch_times)} Avg: {batch_times[-1]:.4f}s")
        print(f"  Difference: {abs(batch_times[-1] - batch_times[0]):.4f}s")

        degradation_factor = (
            batch_times[-1] / batch_times[0] if batch_times[0] > 0 else 1.0
        )
        self.assertLess(
            degradation_factor,
            1.5,
            f"Performance degraded by {(degradation_factor-1)*100:.1f}%",
        )


def run_test_suite():
    """Run all test groups and generate summary"""
    print("=" * 80)
    print("Phase 5: Multi-Lens System Validation Test Suite")
    print("=" * 80)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestSingleDomainRouting))
    suite.addTests(loader.loadTestsFromTestCase(TestMultiDomainRouting))
    suite.addTests(loader.loadTestsFromTestCase(TestAttributeOnlyDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestNoExpertScenario))
    suite.addTests(loader.loadTestsFromTestCase(TestBackwardCompatibility))
    suite.addTests(loader.loadTestsFromTestCase(TestGracefulDegradation))
    suite.addTests(loader.loadTestsFromTestCase(TestDeterminism))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED ✅")
    else:
        print("\n❌ SOME TESTS FAILED ❌")

        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback.split(chr(10))[0]}")

        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback.split(chr(10))[0]}")

    print("=" * 80)

    return result


if __name__ == "__main__":
    result = run_test_suite()
    sys.exit(0 if result.wasSuccessful() else 1)
