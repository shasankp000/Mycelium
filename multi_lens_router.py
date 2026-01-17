"""
Phase 4: Integration (Orchestration Layer)

Connects Phases 1–3 into Layer 1 routing without breaking backward compatibility.

Pipeline:
  INPUT TEXT
    ↓
  layer_1_prototype.multi_lens_route()  [existing, always]
    ↓
  spectral_analyzer.RuntimeSpectralAnalyzer  [optional]
    ↓
  fusion_engine.FusionEngine  [optional]
    ↓
  optimization_engine.GreedyExpertSelector  [optional]
    ↓
  STRUCTURED ROUTING RESULT

Constraints:
- Pure orchestration (no new math, heuristics, or ML)
- Graceful degradation at every step
- Backward compatibility (ATTRIBUTE_ONLY returns immediately)
- Feature flags for optional components
- No modifications to existing logic

Phase 6 Updates:
- Reads hyperparameters from tuning_config
- Implements ATTRIBUTE_ONLY override rule
- Adds lightweight metrics and observability
"""

import logging
from typing import Dict, List, Optional, Any
import sys

# Import tuning config (graceful fallback if not available)
try:
    from tuning_config import (
        COVERAGE_THRESHOLD,
        MAX_EXPERTS,
        DOMAIN_SCORE_THRESHOLD,
        ENABLE_ATTRIBUTE_OVERRIDE,
        ENABLE_LOGGING,
        LOG_SAMPLE_RATE
    )
except ImportError:
    COVERAGE_THRESHOLD = 0.8
    MAX_EXPERTS = 3
    DOMAIN_SCORE_THRESHOLD = 0.45
    ENABLE_ATTRIBUTE_OVERRIDE = True
    ENABLE_LOGGING = False
    LOG_SAMPLE_RATE = 1

try:
    from layer_1_prototype import multi_lens_route
except ImportError:
    multi_lens_route = None

try:
    from spectral_analyzer import RuntimeSpectralAnalyzer
except ImportError:
    RuntimeSpectralAnalyzer = None

try:
    from fusion_engine import FusionEngine
except ImportError:
    FusionEngine = None

try:
    from optimization_engine import GreedyExpertSelector
except ImportError:
    GreedyExpertSelector = None


# Configure logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Phase 6: Global metrics tracking (lightweight aggregates)
_METRICS = {
    "total_requests": 0,
    "attribute_only": 0,
    "single_domain": 0,
    "multi_domain": 0,
    "ambiguous": 0,
    "no_expert": 0,
    "create_new_expert_true": 0,
    "attribute_override_applied": 0,
}


class MultiLensRouter:
    """
    Orchestrates multi-lens routing pipeline: Layer 1 + Spectral + Fusion + Optimization.

    Constraints:
    - Optional and non-breaking
    - Graceful degradation at every step
    - Deterministic output
    - Feature flags for each phase
    """

    def __init__(
        self,
        use_multi_lens: bool = True,
        use_spectral: bool = True,
        spectral_dir: str = "signatures",
        coverage_threshold: float = None,  # Phase 6: Use config default
    ) -> None:
        """
        Initialize router with feature flags.

        Args:
            use_multi_lens: if False, return base layer_1_prototype result only
            use_spectral: if True, add spectral scoring (Phase 1)
            spectral_dir: directory where spectral signatures are stored
            coverage_threshold: for optimization engine (Phase 3), defaults to config value
        """
        self.use_multi_lens = bool(use_multi_lens)
        self.use_spectral = bool(use_spectral)
        self.spectral_dir = str(spectral_dir)
        # Phase 6: Use config value if not provided
        self.coverage_threshold = float(coverage_threshold) if coverage_threshold is not None else COVERAGE_THRESHOLD
        self.enable_logging = ENABLE_LOGGING
        self.request_count = 0  # Phase 6: Track request count for sampling

        # Initialize optional components
        self._spectral_analyzer: Optional[RuntimeSpectralAnalyzer] = None
        self._fusion_engine: Optional[FusionEngine] = None
        self._optimization_selector: Optional[GreedyExpertSelector] = None

        if self.use_spectral and RuntimeSpectralAnalyzer is not None:
            try:
                self._spectral_analyzer = RuntimeSpectralAnalyzer(
                    signature_dir=self.spectral_dir,
                    model_name="all-MiniLM-L6-v2",
                )
            except Exception as e:
                logger.warning(f"Failed to initialize spectral analyzer: {e}")

        if FusionEngine is not None:
            try:
                # Phase 6: FusionEngine now uses config defaults
                self._fusion_engine = FusionEngine()
            except Exception as e:
                logger.warning(f"Failed to initialize fusion engine: {e}")

        if GreedyExpertSelector is not None:
            try:
                # Phase 6: GreedyExpertSelector now uses config defaults
                self._optimization_selector = GreedyExpertSelector(
                    coverage_threshold=self.coverage_threshold,
                )
            except Exception as e:
                logger.warning(f"Failed to initialize optimization selector: {e}")

    def route(self, text: str) -> Dict[str, Any]:
        """
        Full routing pipeline: Layer 1 → Spectral → Fusion → Optimization.

        Args:
            text: input query

        Returns:
            structured routing decision with all fields
        """
        try:
            # Phase 6: Increment request counter and update metrics
            self.request_count += 1
            _METRICS["total_requests"] += 1
            should_log = self.enable_logging and (self.request_count % LOG_SAMPLE_RATE == 0)
            
            if should_log:
                print(f"\n[ROUTER] Request #{self.request_count}: {text[:50]}...", file=sys.stderr)
            
            # ============= STEP 1: BASELINE ROUTING (ALWAYS) =============
            if multi_lens_route is None:
                return self._fallback_result(
                    "layer_1_prototype not available",
                    classification="NO_EXPERT_AVAILABLE",
                )

            if not self.use_multi_lens:
                # Feature flag: disable multi-lens, return base result
                base_result = multi_lens_route(text)
                return self._enrich_base_result(base_result)

            base_result = multi_lens_route(text)
            base_classification = base_result.get("classification", "NORMAL")

            # Phase 6: Check for ATTRIBUTE_ONLY override opportunity
            # Store this for later application after fusion
            attribute_only_candidate = (base_classification == "ATTRIBUTE_ONLY")
            object_level_domains = self._extract_object_level_domains(base_result)
            
            if should_log and attribute_only_candidate:
                print(f"  [ROUTER] ATTRIBUTE_ONLY detected, object_level_domains: {object_level_domains}", file=sys.stderr)

            # Early return: ATTRIBUTE_ONLY if override not applicable
            # (will be checked again after fusion if override enabled)
            if attribute_only_candidate and not object_level_domains:
                _METRICS["attribute_only"] += 1
                if should_log:
                    print(f"  [ROUTER] ATTRIBUTE_ONLY confirmed (no object-level domains)", file=sys.stderr)
                return self._enrich_base_result(base_result)

            # ============= STEP 2: SPECTRAL ANALYSIS (OPTIONAL) =============
            spectral_scores: Dict[str, float] = {}
            if self.use_spectral and self._spectral_analyzer is not None:
                try:
                    if self._spectral_analyzer.is_ready():
                        spectral_scores = self._spectral_analyzer.analyze_text(text) or {}
                except Exception as e:
                    logger.warning(f"Spectral analysis failed: {e}")

            # ============= STEP 3: FUSION ENGINE =============
            # Extract semantic scores from Layer 1
            semantic_scores = self._extract_semantic_scores(base_result)

            # Default confidence: 0.5 for all domains
            confidence_scores = {d: 0.5 for d in semantic_scores.keys()}

            fused_result: Dict[str, Any] = {}
            if self._fusion_engine is not None:
                try:
                    fused_result = self._fusion_engine.fuse(
                        semantic_scores=semantic_scores,
                        spectral_scores=spectral_scores,
                        confidence_scores=confidence_scores,
                    ) or {}
                except Exception as e:
                    logger.warning(f"Fusion failed: {e}")

            # Extract fused scores for optimization
            fused_scores = self._extract_fused_scores(fused_result)

            # Compute variance for output
            variance = self._compute_variance(fused_scores)

            # ============= STEP 4: OPTIMIZATION ENGINE =============
            selected_experts: List[str] = []
            coverage_met = False
            create_new_expert = False

            if self._optimization_selector is not None and fused_scores:
                try:
                    opt_result = self._optimization_selector.select_experts(
                        fused_scores=fused_scores,
                        confidence_scores=confidence_scores,
                    ) or {}
                    selected_experts = opt_result.get("selected_experts", [])
                    coverage_met = opt_result.get("coverage_met", False)
                    create_new_expert = opt_result.get("create_new_expert", False)
                except Exception as e:
                    logger.warning(f"Optimization failed: {e}")
                    create_new_expert = True

            # If no selected experts, try to use base result's primary domain
            if not selected_experts and base_result.get("primary_domain"):
                selected_experts = [base_result.get("primary_domain")]
            
            # Phase 6: ATTRIBUTE_ONLY Override Rule
            # If we had ATTRIBUTE_ONLY but found object-level domains with strong scores, override
            override_applied = False
            if attribute_only_candidate and ENABLE_ATTRIBUTE_OVERRIDE and object_level_domains:
                # Check if any object-level domain has fused score >= threshold
                for domain in object_level_domains:
                    domain_score = fused_scores.get(domain, 0.0)
                    if domain_score >= DOMAIN_SCORE_THRESHOLD:
                        # Override ATTRIBUTE_ONLY - add this domain to selected experts
                        if domain not in selected_experts:
                            selected_experts.append(domain)
                        override_applied = True
                        _METRICS["attribute_override_applied"] += 1
                        if should_log:
                            print(f"  [ROUTER] ATTRIBUTE_ONLY override applied for {domain} (score={domain_score:.4f} >= {DOMAIN_SCORE_THRESHOLD})", file=sys.stderr)
                        break  # Only need one strong domain to override

            # ============= STEP 5: CLASSIFY SUPERPOSITION =============
            # Determine final classification based on selected experts
            if create_new_expert and not selected_experts:
                final_classification = "NO_EXPERT_AVAILABLE"
                candidate_domains = []
                _METRICS["no_expert"] += 1
            elif len(selected_experts) == 1:
                final_classification = "SINGLE_DOMAIN"
                candidate_domains = selected_experts
                _METRICS["single_domain"] += 1
            elif len(selected_experts) > 1:
                # Determine if MULTI_DOMAIN or AMBIGUOUS based on variance
                final_classification = (
                    "MULTI_DOMAIN" if variance >= 0.1 else "AMBIGUOUS"
                )
                candidate_domains = selected_experts
                if final_classification == "MULTI_DOMAIN":
                    _METRICS["multi_domain"] += 1
                else:
                    _METRICS["ambiguous"] += 1
            else:
                final_classification = "NO_EXPERT_AVAILABLE"
                candidate_domains = []
                _METRICS["no_expert"] += 1
            
            # Phase 6: Track create_new_expert metric
            if create_new_expert:
                _METRICS["create_new_expert_true"] += 1
            
            # Phase 6: Log final decision
            if should_log:
                print(f"  [ROUTER] Final: classification={final_classification}, experts={selected_experts}, coverage_met={coverage_met}, create_new={create_new_expert}", file=sys.stderr)

            # ============= STEP 6: BUILD FINAL OUTPUT =============
            primary_domain = selected_experts[0] if selected_experts else None
            explanation = self._build_explanation(
                base_result,
                final_classification,
                selected_experts,
                coverage_met,
                create_new_expert,
            )

            return {
                "primary_domain": primary_domain,
                "selected_experts": selected_experts,
                "candidate_domains": candidate_domains,
                "classification": final_classification,
                "coverage_met": coverage_met,
                "create_new_expert": create_new_expert,
                "lens_scores": {
                    "semantic": semantic_scores,
                    "spectral": spectral_scores,
                    "confidence": confidence_scores,
                },
                "fused_scores": fused_scores,
                "variance": round(variance, 6),
                "explanation": explanation,
            }

        except Exception as e:
            logger.error(f"Critical routing error: {e}")
            return self._fallback_result(
                f"Routing failed: {e}",
                classification="NO_EXPERT_AVAILABLE",
            )

    # ============= HELPER METHODS =============

    def _enrich_base_result(self, base_result: Dict) -> Dict[str, Any]:
        """Enrich base layer_1_prototype result with Phase 4 fields."""
        primary = base_result.get("primary_domain")
        selected = [primary] if primary else []
        return {
            "primary_domain": primary,
            "selected_experts": selected,
            "candidate_domains": selected,
            "classification": base_result.get("classification", "NORMAL"),
            "coverage_met": True,  # Base result assumed to be valid
            "create_new_expert": False,
            "lens_scores": {
                "semantic": self._extract_semantic_scores(base_result),
                "spectral": {},
                "confidence": {d: 0.5 for d in selected},
            },
            "fused_scores": {d: 0.5 for d in selected},
            "variance": 0.0,
            "explanation": base_result.get("explanation", "Returned from Layer 1 baseline."),
        }

    def _fallback_result(
        self,
        explanation: str,
        classification: str = "NO_EXPERT_AVAILABLE",
    ) -> Dict[str, Any]:
        """Return a safe fallback result."""
        return {
            "primary_domain": None,
            "selected_experts": [],
            "candidate_domains": [],
            "classification": classification,
            "coverage_met": False,
            "create_new_expert": True,
            "lens_scores": {"semantic": {}, "spectral": {}, "confidence": {}},
            "fused_scores": {},
            "variance": 0.0,
            "explanation": explanation,
        }

    @staticmethod
    def _extract_semantic_scores(base_result: Dict) -> Dict[str, float]:
        """Extract semantic scores from Layer 1 result."""
        semantic_scores: Dict[str, float] = {}
        if base_result.get("lens1_candidates"):
            for candidate in base_result["lens1_candidates"]:
                if isinstance(candidate, (list, tuple)) and len(candidate) >= 2:
                    domain, score = candidate[0], float(candidate[1])
                    semantic_scores[domain] = min(1.0, max(0.0, score))
        return semantic_scores

    @staticmethod
    def _extract_fused_scores(fused_result: Dict) -> Dict[str, float]:
        """Extract fused_score from fusion engine output."""
        fused_scores: Dict[str, float] = {}
        for domain, info in fused_result.items():
            if isinstance(info, dict):
                score = info.get("fused_score", 0.0)
                fused_scores[domain] = min(1.0, max(0.0, float(score)))
        return fused_scores

    @staticmethod
    def _compute_variance(scores: Dict[str, float]) -> float:
        """Compute population variance of scores."""
        if not scores:
            return 0.0
        values = list(scores.values())
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return variance

    @staticmethod
    def _build_explanation(
        base_result: Dict,
        classification: str,
        selected_experts: List[str],
        coverage_met: bool,
        create_new_expert: bool,
    ) -> str:
        """Build human-readable explanation."""
        parts = []

        # Base explanation
        base_explanation = base_result.get("explanation", "")
        if base_explanation:
            parts.append(f"Base routing: {base_explanation}")

        # Classification explanation
        if classification == "ATTRIBUTE_ONLY":
            parts.append("Detected ATTRIBUTE_ONLY input; no domain experts selected.")
        elif classification == "SINGLE_DOMAIN":
            parts.append(f"Single domain detected: {selected_experts[0] if selected_experts else 'unknown'}.")
        elif classification == "MULTI_DOMAIN":
            parts.append(f"Multiple domains detected: {', '.join(selected_experts)}.")
        elif classification == "AMBIGUOUS":
            parts.append(
                f"Ambiguous input; multiple domains with similar relevance: {', '.join(selected_experts)}."
            )
        elif classification == "NO_EXPERT_AVAILABLE":
            parts.append("No suitable expert available.")

        # Coverage explanation
        if coverage_met:
            parts.append("Coverage threshold met.")
        elif create_new_expert:
            parts.append("Coverage threshold not met; new expert may be required.")

        return " ".join(parts)
    
    @staticmethod
    def _extract_object_level_domains(base_result: Dict) -> List[str]:
        """
        Phase 6: Extract object-level domains from Layer 1 (Lens 2) result.
        
        Object-level domains are those matched at the core/object level
        (not just modifiers/attributes).
        """
        object_domains = []
        try:
            lens3_signature = base_result.get("lens3_signature", {})
            if lens3_signature:
                levels = lens3_signature.get("levels", {})
                core_domains = levels.get("core", {})
                # Domains with core features are considered object-level
                for domain, features in core_domains.items():
                    if features:  # Has at least one core feature
                        object_domains.append(domain)
            
            # Also check lens2_explanations for object-level matches
            lens2_explanations = base_result.get("lens2_explanations", [])
            for explanation in lens2_explanations:
                if explanation.get("level") == "object" and explanation.get("matched_core"):
                    concept = explanation.get("concept")
                    if concept and concept not in object_domains:
                        object_domains.append(concept)
        except Exception:
            # Graceful degradation
            pass
        
        return object_domains
    
    @staticmethod
    def get_metrics() -> Dict[str, Any]:
        """
        Phase 6: Return aggregate metrics.
        
        Returns:
            Dictionary with request counts and percentages
        """
        total = _METRICS["total_requests"]
        if total == 0:
            return {**_METRICS, "percentages": {}}
        
        percentages = {
            "attribute_only_pct": (_METRICS["attribute_only"] / total) * 100,
            "single_domain_pct": (_METRICS["single_domain"] / total) * 100,
            "multi_domain_pct": (_METRICS["multi_domain"] / total) * 100,
            "ambiguous_pct": (_METRICS["ambiguous"] / total) * 100,
            "no_expert_pct": (_METRICS["no_expert"] / total) * 100,
            "create_new_expert_pct": (_METRICS["create_new_expert_true"] / total) * 100,
            "attribute_override_pct": (_METRICS["attribute_override_applied"] / total) * 100,
        }
        
        return {**_METRICS, "percentages": percentages}


def _print_case(title: str, result: Dict) -> None:
    """Print routing result for a test case."""
    print(f"\n=== {title} ===")
    print(f"Classification: {result.get('classification')}")
    print(f"Selected Experts: {result.get('selected_experts')}")
    print(f"Primary Domain: {result.get('primary_domain')}")
    print(f"Coverage Met: {result.get('coverage_met')}")
    print(f"Create New Expert: {result.get('create_new_expert')}")
    print(f"Fused Scores: {result.get('fused_scores')}")
    print(f"Variance: {result.get('variance'):.6f}")
    print(f"Explanation: {result.get('explanation')}")


if __name__ == "__main__":
    router = MultiLensRouter(
        use_multi_lens=True,
        use_spectral=True,
        spectral_dir="signatures",
        coverage_threshold=0.8,
    )

    # Test A: Single Domain
    text_a = "Earth orbits the Sun in an elliptical orbit while planets move according to gravitational forces."
    result_a = router.route(text_a)
    _print_case("Test A: Single Domain (Astronomy)", result_a)

    # Test B: Multi Domain
    text_b = "Red metallic car with 700cc engine and sleek design"
    result_b = router.route(text_b)
    _print_case("Test B: Multi Domain", result_b)

    # Test C: Attribute Only
    text_c = "Glossy metallic red finish with smooth texture"
    result_c = router.route(text_c)
    _print_case("Test C: Attribute Only", result_c)

    # Test D: Backward Compatibility (use_multi_lens=False)
    router_base = MultiLensRouter(use_multi_lens=False)
    result_d = router_base.route(text_a)
    _print_case("Test D: Backward Compatibility (use_multi_lens=False)", result_d)

    print("\n✅ Phase 4 Integration minimal tests completed.")
