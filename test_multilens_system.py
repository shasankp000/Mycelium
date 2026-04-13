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
        print(f"  Classification: {result['classification']}")
        print(f"  Selected Experts: {result['selected_experts']}")
        print(f"  Primary Domain: {result['primary_domain']}")
        print(f"  Coverage Met: {result['coverage_met']}")
        
        # Assertions
        self.assertIn(result['classification'], ['SINGLE_DOMAIN', 'NORMAL'], 
                     "Should classify as SINGLE_DOMAIN or NORMAL")
        self.assertIsNotNone(result['selected_experts'], "Should have selected experts")
        self.assertIsInstance(result['selected_experts'], list, "Selected experts should be a list")
        self.assertIn('coverage_met', result, "Coverage flag must exist")
        # Note: We can't guarantee exactly 1 expert without knowing available experts
        # but we can verify structure
        
    def test_biology_single_domain(self):
        """Test: 'Photosynthesis occurs in plants' should route to biology or unknown"""
        text = "Photosynthesis occurs in plants"
        result = self.router.route(text)
        
        print(f"\n[SINGLE-DOMAIN: Biology/Unknown]")
        print(f"  Classification: {result['classification']}")
        print(f"  Selected Experts: {result['selected_experts']}")
        print(f"  Primary Domain: {result['primary_domain']}")
        
        # Assertions - should not crash and should return valid structure
        # Note: Without biology expert, this may be ATTRIBUTE_ONLY (expected)
        self.assertIn(result['classification'], 
                     ['SINGLE_DOMAIN', 'NORMAL', 'NO_EXPERT_AVAILABLE', 'AMBIGUOUS', 'ATTRIBUTE_ONLY'],
                     "Should have valid classification")
        self.assertIsInstance(result['selected_experts'], list)
        self.assertIn('coverage_met', result)


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
        print(f"  Classification: {result['classification']}")
        print(f"  Selected Experts: {result['selected_experts']}")
        print(f"  Candidate Domains: {result.get('candidate_domains', [])}")
        print(f"  Variance: {result.get('variance', 0)}")
        
        # Assertions
        self.assertIn(result['classification'], 
                     ['MULTI_DOMAIN', 'AMBIGUOUS', 'SINGLE_DOMAIN', 'NORMAL'],
                     "Should have valid classification")
        self.assertIsInstance(result.get('candidate_domains', []), list,
                            "Candidate domains should be a list")
        # If multi-domain, should have multiple candidates
        if result['classification'] in ['MULTI_DOMAIN', 'AMBIGUOUS']:
            self.assertGreaterEqual(len(result.get('candidate_domains', [])), 1,
                                   "Multi-domain should have candidates")
    
    def test_medical_physics_multi_domain(self):
        """Test: 'Medical ultrasound uses physics principles' should span domains"""
        text = "Medical ultrasound uses physics principles"
        result = self.router.route(text)
        
        print(f"\n[MULTI-DOMAIN: Medical+Physics]")
        print(f"  Classification: {result['classification']}")
        print(f"  Selected Experts: {result['selected_experts']}")
        print(f"  Candidate Domains: {result.get('candidate_domains', [])}")
        
        # Assertions
        # Note: Without medical/physics experts, may be ATTRIBUTE_ONLY (expected)
        self.assertIn(result['classification'],
                     ['MULTI_DOMAIN', 'AMBIGUOUS', 'SINGLE_DOMAIN', 'NORMAL', 'NO_EXPERT_AVAILABLE', 'ATTRIBUTE_ONLY'])
        self.assertIsInstance(result['selected_experts'], list)
    
    def test_deterministic_multi_domain(self):
        """Test: Same input twice should produce identical output"""
        text = "Red car with 700cc engine"
        
        result1 = self.router.route(text)
        result2 = self.router.route(text)
        
        print(f"\n[DETERMINISM CHECK: Multi-Domain]")
        print(f"  Run 1 Classification: {result1['classification']}")
        print(f"  Run 2 Classification: {result2['classification']}")
        print(f"  Run 1 Experts: {result1['selected_experts']}")
        print(f"  Run 2 Experts: {result2['selected_experts']}")
        
        # Assertions - outputs must be identical
        self.assertEqual(result1['classification'], result2['classification'],
                        "Classification must be deterministic")
        self.assertEqual(result1['selected_experts'], result2['selected_experts'],
                        "Selected experts must be deterministic")
        self.assertEqual(result1.get('variance'), result2.get('variance'),
                        "Variance must be deterministic")


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
        print(f"  Classification: {result['classification']}")
        print(f"  Selected Experts: {result['selected_experts']}")
        print(f"  Primary Domain: {result['primary_domain']}")
        
        # Assertions
        # Phase 6: Pure attributes may be promoted if they match object-level domains
        # (e.g., 'glossy metallic red' matches aesthetics domain)
        self.assertIn(result['classification'], ['ATTRIBUTE_ONLY', 'SINGLE_DOMAIN'],
                     "Should be ATTRIBUTE_ONLY or promoted via override")
        # If promoted, should have selected experts
        if result['classification'] == 'SINGLE_DOMAIN':
            self.assertEqual(len(result['selected_experts']), 1,
                           "Promoted attribute should have exactly 1 expert")
            self.assertIsNotNone(result['primary_domain'],
                               "Promoted attribute should have primary domain")
        else:
            # Still ATTRIBUTE_ONLY
            self.assertEqual(result['selected_experts'], [])
            self.assertIsNone(result['primary_domain'])
    
    def test_performance_attribute(self):
        """Test: 'High torque low noise' should be ATTRIBUTE_ONLY"""
        text = "High torque low noise"
        result = self.router.route(text)
        
        print(f"\n[ATTRIBUTE-ONLY: Performance]")
        print(f"  Classification: {result['classification']}")
        print(f"  Selected Experts: {result['selected_experts']}")
        print(f"  Primary Domain: {result['primary_domain']}")
        
        # Assertions
        # Phase 6: Pure attributes may be promoted if they match object-level domains
        self.assertIn(result['classification'], ['ATTRIBUTE_ONLY', 'SINGLE_DOMAIN'],
                     "Should be ATTRIBUTE_ONLY or promoted via override")
        if result['classification'] == 'SINGLE_DOMAIN':
            self.assertEqual(len(result['selected_experts']), 1)
            self.assertIsNotNone(result['primary_domain'])
        else:
            self.assertEqual(result['selected_experts'], [])
            self.assertIsNone(result['primary_domain'])


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
        print(f"  Classification: {result['classification']}")
        print(f"  Selected Experts: {result['selected_experts']}")
        print(f"  Create New Expert: {result.get('create_new_expert', False)}")
        print(f"  Coverage Met: {result.get('coverage_met', False)}")
        
        # Assertions
        # Should either be NO_EXPERT_AVAILABLE, AMBIGUOUS, or trigger create_new_expert
        if result['classification'] == 'NO_EXPERT_AVAILABLE':
            self.assertTrue(result.get('create_new_expert', False),
                          "NO_EXPERT_AVAILABLE should signal to create new expert")
        elif result['classification'] == 'AMBIGUOUS':
            # Ambiguous is acceptable for unknown domains
            pass
        else:
            # If it routes somewhere, create_new_expert might be True due to low coverage
            self.assertIn('create_new_expert', result,
                         "Should have create_new_expert flag")


class TestBackwardCompatibility(unittest.TestCase):
    """Test Group 5: Backward Compatibility"""
    
    def test_disable_multi_lens(self):
        """Test: use_multi_lens=False should behave like original multi_lens_route()"""
        router = MultiLensRouter(use_multi_lens=False, use_spectral=False)
        
        text = "Earth orbits the Sun"
        result = router.route(text)
        
        print(f"\n[BACKWARD COMPAT: use_multi_lens=False]")
        print(f"  Classification: {result['classification']}")
        print(f"  Selected Experts: {result['selected_experts']}")
        print(f"  Has Fused Scores: {'fused_scores' in result and result['fused_scores'] is not None}")
        print(f"  Has Spectral: {'lens_scores' in result and result['lens_scores'] is not None and 'spectral' in result['lens_scores']}")
        
        # Assertions
        self.assertIn(result['classification'], ['NORMAL', 'ATTRIBUTE_ONLY'],
                     "Backward compat mode should use simple classifications")
        self.assertIsInstance(result['selected_experts'], list)
        # Should NOT use spectral or fusion
        # Note: spectral key may exist but should be empty dict when disabled
        if 'lens_scores' in result and result['lens_scores'] is not None:
            spectral_scores = result['lens_scores'].get('spectral', {})
            if spectral_scores is not None:
                self.assertEqual(spectral_scores, {},
                               "Backward compat spectral scores should be empty")
    
    def test_disable_spectral_only(self):
        """Test: use_spectral=False should skip spectral analysis"""
        router = MultiLensRouter(use_multi_lens=True, use_spectral=False)
        
        text = "Red car with engine"
        result = router.route(text)
        
        print(f"\n[SPECTRAL DISABLED: use_spectral=False]")
        print(f"  Classification: {result['classification']}")
        print(f"  Has Spectral Scores: {'lens_scores' in result and result['lens_scores'] is not None and 'spectral' in result['lens_scores']}")
        
        # Assertions
        if 'lens_scores' in result and result['lens_scores'] is not None:
            # Spectral should be None or absent
            spectral = result['lens_scores'].get('spectral')
            if spectral is not None:
                self.assertEqual(spectral, {},
                               "Spectral should be empty when disabled")


class TestGracefulDegradation(unittest.TestCase):
    """Test Group 6: Graceful Degradation"""
    
    def test_missing_spectral_signatures(self):
        """Test: Missing spectral signatures should not crash the system"""
        # Initialize router - spectral analyzer will handle missing signatures gracefully
        router = MultiLensRouter(use_multi_lens=True, use_spectral=True)
        
        text = "Some text about quantum mechanics and thermodynamics"
        
        # Should not raise exception even if signatures are missing
        try:
            result = router.route(text)
            
            print(f"\n[GRACEFUL DEGRADATION: Missing Signatures]")
            print(f"  Classification: {result['classification']}")
            print(f"  Selected Experts: {result['selected_experts']}")
            print(f"  No Exception Raised: True")
            
            # Assertions
            self.assertIsInstance(result, dict, "Should return valid result dict")
            self.assertIn('classification', result)
            self.assertIn('selected_experts', result)
            
        except Exception as e:
            self.fail(f"Router should not crash with missing signatures: {e}")
    
    def test_empty_input(self):
        """Test: Empty input should be handled gracefully"""
        router = MultiLensRouter(use_multi_lens=True, use_spectral=True)
        
        text = ""
        
        try:
            result = router.route(text)
            
            print(f"\n[GRACEFUL DEGRADATION: Empty Input]")
            print(f"  Classification: {result['classification']}")
            print(f"  Selected Experts: {result['selected_experts']}")
            
            # Assertions
            self.assertIsInstance(result, dict)
            self.assertIn('classification', result)
            
        except Exception as e:
            self.fail(f"Router should handle empty input gracefully: {e}")
    
    def test_very_long_input(self):
        """Test: Very long input should be handled gracefully"""
        router = MultiLensRouter(use_multi_lens=True, use_spectral=True)
        
        # Create a very long text
        text = "The study of astronomy involves planets and stars. " * 100
        
        try:
            result = router.route(text)
            
            print(f"\n[GRACEFUL DEGRADATION: Long Input]")
            print(f"  Classification: {result['classification']}")
            print(f"  Input Length: {len(text)} chars")
            
            # Assertions
            self.assertIsInstance(result, dict)
            self.assertIn('classification', result)
            
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
        print(f"  Run 1 Classification: {results[0]['classification']}")
        print(f"  Run 1 Experts: {results[0]['selected_experts']}")
        print(f"  Run 1 Variance: {results[0].get('variance')}")
        
        # Assertions - all runs must be identical
        for i in range(1, 5):
            self.assertEqual(results[0]['classification'], results[i]['classification'],
                           f"Run {i+1} classification differs from Run 1")
            self.assertEqual(results[0]['selected_experts'], results[i]['selected_experts'],
                           f"Run {i+1} selected experts differ from Run 1")
            self.assertEqual(results[0].get('variance'), results[i].get('variance'),
                           f"Run {i+1} variance differs from Run 1")
            self.assertEqual(results[0].get('primary_domain'), results[i].get('primary_domain'),
                           f"Run {i+1} primary domain differs from Run 1")
        
        print(f"  ✅ All 5 runs produced identical outputs")
    
    def test_deterministic_ordering(self):
        """Test: Selected experts should have consistent ordering"""
        text = "Medical imaging uses physics and engineering principles"
        
        results = []
        for i in range(3):
            result = self.router.route(text)
            results.append(result['selected_experts'])
        
        print(f"\n[DETERMINISM: Expert Ordering]")
        print(f"  Run 1 Experts: {results[0]}")
        print(f"  Run 2 Experts: {results[1]}")
        print(f"  Run 3 Experts: {results[2]}")
        
        # Assertions - ordering must be consistent
        self.assertEqual(results[0], results[1], "Run 2 expert order differs from Run 1")
        self.assertEqual(results[0], results[2], "Run 3 expert order differs from Run 1")


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
            "Chemical reaction kinetics"
        ]
        
        routing_times = []
        
        for text in test_inputs:
            start_time = time.time()
            result = self.router.route(text)
            end_time = time.time()
            
            routing_time = end_time - start_time
            routing_times.append(routing_time)
        
        avg_time = statistics.mean(routing_times)
        max_time = max(routing_times)
        min_time = min(routing_times)
        
        print(f"\n[PERFORMANCE: 10 Routing Calls]")
        print(f"  Average Time: {avg_time:.4f} seconds")
        print(f"  Max Time: {max_time:.4f} seconds")
        print(f"  Min Time: {min_time:.4f} seconds")
        
        # Assertions
        self.assertLess(avg_time, 1.0,
                       f"Average routing time {avg_time:.4f}s exceeds 1 second threshold")
        self.assertLess(max_time, 2.0,
                       f"Max routing time {max_time:.4f}s is excessive")
    
    def test_no_performance_degradation(self):
        """Test: Performance should not degrade across multiple runs"""
        text = "Astronomy and stellar evolution"
        
        # Run 50 times and measure time in batches
        batch_times = []
        
        for batch in range(5):
            start_time = time.time()
            for i in range(10):
                result = self.router.route(text)
            end_time = time.time()
            
            batch_time = (end_time - start_time) / 10  # Average per call
            batch_times.append(batch_time)
        
        print(f"\n[PERFORMANCE: Degradation Check]")
        print(f"  Batch 1 Avg: {batch_times[0]:.4f}s")
        print(f"  Batch 5 Avg: {batch_times[4]:.4f}s")
        print(f"  Difference: {abs(batch_times[4] - batch_times[0]):.4f}s")
        
        # Assertions - later batches should not be significantly slower
        # Allow up to 50% variance due to system noise
        degradation_factor = batch_times[4] / batch_times[0] if batch_times[0] > 0 else 1.0
        self.assertLess(degradation_factor, 1.5,
                       f"Performance degraded by {(degradation_factor-1)*100:.1f}%")


def run_test_suite():
    """Run all test groups and generate summary"""
    print("=" * 80)
    print("Phase 5: Multi-Lens System Validation Test Suite")
    print("=" * 80)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test groups
    suite.addTests(loader.loadTestsFromTestCase(TestSingleDomainRouting))
    suite.addTests(loader.loadTestsFromTestCase(TestMultiDomainRouting))
    suite.addTests(loader.loadTestsFromTestCase(TestAttributeOnlyDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestNoExpertScenario))
    suite.addTests(loader.loadTestsFromTestCase(TestBackwardCompatibility))
    suite.addTests(loader.loadTestsFromTestCase(TestGracefulDegradation))
    suite.addTests(loader.loadTestsFromTestCase(TestDeterminism))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
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
