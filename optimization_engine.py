"""
Phase 3: Optimization Engine (Minimal Expert Selection)

Standalone module that:
- Calculates coverage value per expert (fused_score × confidence)
- Greedily selects the minimal set of experts
- Decides when to create new expert
- Operates on scores only (no text)

Constraints:
- Optional, deterministic, non-breaking
- Works with fused scores and confidence values
- Missing confidence defaults to 1.0

Phase 6 Updates:
- Reads coverage_threshold and max_experts from tuning_config
- Adds logging for coverage progression
- Implements soft-stop at 90% of coverage threshold
"""

from typing import Dict, List, Tuple
import sys

# Import tuning config (graceful fallback if not available)
try:
    from tuning_config import (
        COVERAGE_THRESHOLD,
        MAX_EXPERTS,
        SOFT_STOP_PERCENTAGE,
        ENABLE_LOGGING
    )
except ImportError:
    # Fallback defaults if config not available
    COVERAGE_THRESHOLD = 0.8
    MAX_EXPERTS = 3
    SOFT_STOP_PERCENTAGE = 0.9
    ENABLE_LOGGING = False


class CoverageCalculator:
    """
    Computes coverage value for an expert.

    Rule:
        value = fused_score × confidence

    Result clipped to [0, 1].
    Deterministic, no side effects.
    """

    def compute_value(self, fused_score: float, confidence: float) -> float:
        """
        Compute coverage value.

        Args:
            fused_score: [0, 1]
            confidence: [0, 1]

        Returns:
            value clipped to [0, 1]
        """
        try:
            score = float(fused_score) if fused_score is not None else 0.0
            conf = float(confidence) if confidence is not None else 1.0
        except (TypeError, ValueError):
            return 0.0

        # Clip inputs to [0, 1]
        score = max(0.0, min(1.0, score))
        conf = max(0.0, min(1.0, conf))

        # Compute product
        value = score * conf

        # Clip result to [0, 1]
        return max(0.0, min(1.0, value))


class GreedyExpertSelector:
    """
    Greedily selects minimal expert set to meet coverage threshold.

    Algorithm:
    1. If fused_scores is empty -> create_new_expert = True
    2. For each domain: compute value = fused_score × confidence
       (assume cost = 1.0 for all)
    3. Sort by: value/cost (desc), confidence (desc), fused_score (desc), alphabetical
    4. Greedy loop: add best expert, accumulate coverage, stop when:
       - coverage >= threshold -> success
       - max_experts reached -> check if sufficient
    5. If coverage >= threshold: return selected
    6. Else: return create_new_expert = True

    Deterministic: sorted iteration, explicit tie-breakers.
    """

    def __init__(
        self,
        coverage_threshold: float = None,  # Phase 6: Use config default
        max_experts: int = None,  # Phase 6: Use config default
    ) -> None:
        # Phase 6: Use config values if not explicitly provided
        self.coverage_threshold = float(coverage_threshold) if coverage_threshold is not None else COVERAGE_THRESHOLD
        self.max_experts = int(max_experts) if max_experts is not None else MAX_EXPERTS
        self.calculator = CoverageCalculator()
        self.enable_logging = ENABLE_LOGGING

    def select_experts(
        self,
        fused_scores: Dict[str, float],
        confidence_scores: Dict[str, float],
    ) -> Dict:
        """
        Greedily select experts to meet coverage threshold.

        Args:
            fused_scores: domain -> [0,1]
            confidence_scores: domain -> [0,1], optional

        Returns:
            {
              "selected_experts": [...],
              "total_coverage": float,
              "coverage_threshold": float,
              "coverage_met": bool,
              "create_new_expert": bool
            }
        """
        try:
            # Handle empty input
            if not fused_scores:
                return {
                    "selected_experts": [],
                    "total_coverage": 0.0,
                    "coverage_threshold": self.coverage_threshold,
                    "coverage_met": False,
                    "create_new_expert": True,
                }

            # Build candidate list: (domain, value, confidence, fused_score)
            candidates: List[Tuple[str, float, float, float]] = []
            confidence_scores = confidence_scores or {}

            for domain in fused_scores.keys():
                fused = float(fused_scores[domain]) if fused_scores[domain] is not None else 0.0
                conf = float(confidence_scores.get(domain, 1.0)) if confidence_scores.get(domain) is not None else 1.0
                # Clip
                fused = max(0.0, min(1.0, fused))
                conf = max(0.0, min(1.0, conf))
                # Compute value
                value = self.calculator.compute_value(fused, conf)
                candidates.append((domain, value, conf, fused))

            # Sort by: value/cost (desc) [cost=1.0], confidence (desc), fused_score (desc), alphabetical
            # For determinism, explicit tuple-based sort
            candidates_sorted = sorted(
                candidates,
                key=lambda x: (-x[1], -x[2], -x[3], x[0]),
            )

            # Greedy loop
            selected: List[str] = []
            total_coverage = 0.0
            
            # Phase 6: Calculate soft-stop threshold
            soft_stop_threshold = self.coverage_threshold * SOFT_STOP_PERCENTAGE
            
            # Phase 6: Log initial state
            if self.enable_logging:
                print(f"  [OPTIMIZATION] Starting greedy selection: threshold={self.coverage_threshold:.4f}, soft_stop={soft_stop_threshold:.4f}, max_experts={self.max_experts}", file=sys.stderr)

            for domain, value, conf, fused in candidates_sorted:
                # Phase 6: Check soft-stop condition BEFORE adding
                if total_coverage >= soft_stop_threshold and len(selected) > 0:
                    if self.enable_logging:
                        print(f"  [OPTIMIZATION] Soft-stop triggered at coverage={total_coverage:.4f} (≥{soft_stop_threshold:.4f})", file=sys.stderr)
                    break
                
                if len(selected) >= self.max_experts:
                    if self.enable_logging:
                        print(f"  [OPTIMIZATION] Max experts ({self.max_experts}) reached", file=sys.stderr)
                    break
                    
                selected.append(domain)
                total_coverage += value
                
                # Phase 6: Log each expert addition
                if self.enable_logging:
                    print(f"  [OPTIMIZATION] Added {domain}: value={value:.4f}, cumulative_coverage={total_coverage:.4f}", file=sys.stderr)

            # Sort selected alphabetically for deterministic output
            selected_sorted = sorted(selected)

            # Check if coverage met
            coverage_met = total_coverage >= self.coverage_threshold

            return {
                "selected_experts": selected_sorted,
                "total_coverage": round(total_coverage, 6),
                "coverage_threshold": self.coverage_threshold,
                "coverage_met": coverage_met,
                "create_new_expert": not coverage_met,
            }

        except Exception:
            # Graceful degradation: never raise
            return {
                "selected_experts": [],
                "total_coverage": 0.0,
                "coverage_threshold": self.coverage_threshold,
                "coverage_met": False,
                "create_new_expert": True,
            }


def _print_case(title: str, result: Dict) -> None:
    print(f"\n=== {title} ===")
    print(f"Selected Experts: {result.get('selected_experts')}")
    print(f"Total Coverage: {result.get('total_coverage'):.3f} / {result.get('coverage_threshold'):.3f}")
    print(f"Coverage Met: {result.get('coverage_met')}")
    print(f"Create New Expert: {result.get('create_new_expert')}")


if __name__ == "__main__":
    # Minimal tests
    selector = GreedyExpertSelector(coverage_threshold=0.8, max_experts=3)

    # 1) Single expert sufficient
    fused1 = {"astronomy": 0.9}
    conf1 = {"astronomy": 0.95}
    result1 = selector.select_experts(fused1, conf1)
    _print_case("Single Expert Sufficient", result1)

    # 2) Multiple experts required
    fused2 = {"astronomy": 0.6, "physics": 0.5, "chemistry": 0.4}
    conf2 = {"astronomy": 0.9, "physics": 0.85, "chemistry": 0.8}
    result2 = selector.select_experts(fused2, conf2)
    _print_case("Multiple Experts Required", result2)

    # 3) Insufficient coverage -> create_new_expert
    fused3 = {"astronomy": 0.3, "automobile": 0.2}
    conf3 = {"astronomy": 0.5, "automobile": 0.5}
    result3 = selector.select_experts(fused3, conf3)
    _print_case("Insufficient Coverage", result3)

    # 4) Empty input
    fused4 = {}
    conf4 = {}
    result4 = selector.select_experts(fused4, conf4)
    _print_case("Empty Input", result4)

    print("\n✅ Phase 3 Optimization Engine minimal tests completed.")
