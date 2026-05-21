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

try:
    from mycelium.pipeline.tuning_config import (
        COVERAGE_THRESHOLD,
        MAX_EXPERTS,
        SOFT_STOP_PERCENTAGE,
        ENABLE_LOGGING
    )
except ImportError:
    COVERAGE_THRESHOLD = 0.8
    MAX_EXPERTS = 3
    SOFT_STOP_PERCENTAGE = 0.9
    ENABLE_LOGGING = False


class CoverageCalculator:
    def compute_value(self, fused_score: float, confidence: float) -> float:
        try:
            score = float(fused_score) if fused_score is not None else 0.0
            conf = float(confidence) if confidence is not None else 1.0
        except (TypeError, ValueError):
            return 0.0
        score = max(0.0, min(1.0, score))
        conf = max(0.0, min(1.0, conf))
        value = score * conf
        return max(0.0, min(1.0, value))


class GreedyExpertSelector:
    def __init__(self, coverage_threshold: float = None, max_experts: int = None) -> None:
        self.coverage_threshold = float(coverage_threshold) if coverage_threshold is not None else COVERAGE_THRESHOLD
        self.max_experts = int(max_experts) if max_experts is not None else MAX_EXPERTS
        self.calculator = CoverageCalculator()
        self.enable_logging = ENABLE_LOGGING

    def select_experts(
        self,
        fused_scores: Dict[str, float],
        confidence_scores: Dict[str, float],
    ) -> Dict:
        try:
            if not fused_scores:
                return {
                    "selected_experts": [],
                    "total_coverage": 0.0,
                    "coverage_threshold": self.coverage_threshold,
                    "coverage_met": False,
                    "create_new_expert": True,
                }

            candidates: List[Tuple[str, float, float, float]] = []
            confidence_scores = confidence_scores or {}

            for domain in fused_scores.keys():
                fused = float(fused_scores[domain]) if fused_scores[domain] is not None else 0.0
                conf = float(confidence_scores.get(domain, 1.0)) if confidence_scores.get(domain) is not None else 1.0
                fused = max(0.0, min(1.0, fused))
                conf = max(0.0, min(1.0, conf))
                value = self.calculator.compute_value(fused, conf)
                candidates.append((domain, value, conf, fused))

            candidates_sorted = sorted(
                candidates,
                key=lambda x: (-x[1], -x[2], -x[3], x[0]),
            )

            selected: List[str] = []
            total_coverage = 0.0
            soft_stop_threshold = self.coverage_threshold * SOFT_STOP_PERCENTAGE

            if self.enable_logging:
                print(f"  [OPTIMIZATION] Starting greedy selection: threshold={self.coverage_threshold:.4f}, soft_stop={soft_stop_threshold:.4f}, max_experts={self.max_experts}", file=sys.stderr)

            for domain, value, conf, fused in candidates_sorted:
                if total_coverage >= soft_stop_threshold and len(selected) > 0:
                    if self.enable_logging:
                        print(f"  [OPTIMIZATION] Soft-stop triggered at coverage={total_coverage:.4f}", file=sys.stderr)
                    break
                if len(selected) >= self.max_experts:
                    break
                selected.append(domain)
                total_coverage += value
                if self.enable_logging:
                    print(f"  [OPTIMIZATION] Added {domain}: value={value:.4f}, cumulative_coverage={total_coverage:.4f}", file=sys.stderr)

            selected_sorted = sorted(selected)
            coverage_met = total_coverage >= self.coverage_threshold

            return {
                "selected_experts": selected_sorted,
                "total_coverage": round(total_coverage, 6),
                "coverage_threshold": self.coverage_threshold,
                "coverage_met": coverage_met,
                "create_new_expert": not coverage_met,
            }

        except Exception:
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
