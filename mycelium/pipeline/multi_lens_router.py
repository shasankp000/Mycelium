"""
Multi-Lens Router — Complete Rewrite
=====================================
See original docstring for full design notes.
"""
import logging
import sys
from typing import Any, Dict, List, Optional, Tuple

from core.types import RoutingResult

try:
    from mycelium.pipeline.tuning_config import (
        COVERAGE_THRESHOLD,
        MAX_EXPERTS,
        DOMAIN_SCORE_THRESHOLD,
        ENABLE_ATTRIBUTE_OVERRIDE,
        ENABLE_LOGGING,
        LOG_SAMPLE_RATE,
    )
except ImportError:
    COVERAGE_THRESHOLD = 0.8
    MAX_EXPERTS = 3
    DOMAIN_SCORE_THRESHOLD = 0.45
    ENABLE_ATTRIBUTE_OVERRIDE = True
    ENABLE_LOGGING = False
    LOG_SAMPLE_RATE = 1

_ABSOLUTE_MAX_EXPERTS: int = max(1, MAX_EXPERTS)

try:
    from mycelium.pipeline.layer_1_prototype import multi_lens_route
except ImportError:
    multi_lens_route = None  # type: ignore[assignment]

try:
    from mycelium.pipeline.spectral_analyzer import RuntimeSpectralAnalyzer
except ImportError:
    RuntimeSpectralAnalyzer = None  # type: ignore[assignment]

try:
    from mycelium.pipeline.fusion_engine import FusionEngine
except ImportError:
    FusionEngine = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_METRICS: Dict[str, int] = {
    "total_requests": 0,
    "attribute_only": 0,
    "single_domain": 0,
    "multi_domain": 0,
    "ambiguous": 0,
    "no_expert": 0,
    "create_new_expert_true": 0,
    "attribute_override_applied": 0,
    "budget_cap_applied": 0,
}


def _budget_select(
    fused_scores: Dict[str, float],
    confidence_scores: Dict[str, float],
    min_score: float = DOMAIN_SCORE_THRESHOLD,
    max_experts: int = _ABSOLUTE_MAX_EXPERTS,
) -> Tuple[List[str], bool]:
    if not fused_scores:
        return [], False
    candidates: List[Tuple[float, float, str]] = []
    for domain, score in fused_scores.items():
        if score >= min_score:
            conf = confidence_scores.get(domain, 0.5)
            candidates.append((score, conf, domain))
    if not candidates:
        return [], False
    candidates.sort(key=lambda t: (-t[0], -t[1], t[2]))
    budget_cap_applied = len(candidates) > max_experts
    selected = [domain for _, _, domain in candidates[:max_experts]]
    return selected, budget_cap_applied


class MultiLensRouter:
    def __init__(
        self,
        use_multi_lens: bool = True,
        use_spectral: bool = True,
        spectral_dir: str = "signatures",
        coverage_threshold: float = COVERAGE_THRESHOLD,
        max_experts: Optional[int] = None,
        spectral_analyzer: Optional[Any] = None,
    ) -> None:
        self.use_multi_lens = bool(use_multi_lens)
        self.use_spectral = bool(use_spectral)
        self.spectral_dir = str(spectral_dir)
        self.coverage_threshold = float(coverage_threshold)
        self.max_experts: int = (
            min(int(max_experts), _ABSOLUTE_MAX_EXPERTS)
            if max_experts is not None
            else _ABSOLUTE_MAX_EXPERTS
        )
        self.enable_logging = ENABLE_LOGGING
        self.request_count = 0
        self._spectral_analyzer: Optional[Any] = None
        self._fusion_engine: Optional[Any] = None
        if self.use_spectral:
            if spectral_analyzer is not None:
                self._spectral_analyzer = spectral_analyzer
            elif RuntimeSpectralAnalyzer is not None:
                try:
                    self._spectral_analyzer = RuntimeSpectralAnalyzer(
                        signature_dir=self.spectral_dir, model_name="all-MiniLM-L6-v2"
                    )
                except Exception as exc:
                    logger.warning("Failed to initialise spectral analyzer: %s", exc)
        if FusionEngine is not None:
            try:
                self._fusion_engine = FusionEngine()
            except Exception as exc:
                logger.warning("Failed to initialise fusion engine: %s", exc)

    def route(self, text: str) -> RoutingResult:
        try:
            self.request_count += 1
            _METRICS["total_requests"] += 1
            should_log = self.enable_logging and (self.request_count % LOG_SAMPLE_RATE == 0)
            if multi_lens_route is None:
                return self._fallback_result("layer_1_prototype not available")
            if not self.use_multi_lens:
                base = multi_lens_route(text)
                return self._wrap_base_result(base)
            base_result = multi_lens_route(text)
            base_classification = base_result.get("classification", "NORMAL")
            is_attribute_only = base_classification == "ATTRIBUTE_ONLY"
            object_level_domains = self._extract_object_level_domains(base_result)
            semantic_scores = self._extract_semantic_scores(base_result)
            if should_log:
                logger.debug(
                    "[ROUTER] req#%d  text=%r  attribute_only=%s  semantic_domains=%s",
                    self.request_count, text[:50], is_attribute_only, list(semantic_scores.keys()),
                )
            spectral_scores: Dict[str, float] = {}
            if self.use_spectral and self._spectral_analyzer is not None:
                try:
                    if self._spectral_analyzer.is_ready():
                        spectral_scores = self._spectral_analyzer.analyze_text(text) or {}
                except Exception as exc:
                    logger.warning("Spectral analysis failed: %s", exc)
            all_candidate_domains = set(semantic_scores) | set(spectral_scores)
            confidence_scores: Dict[str, float] = {d: 0.5 for d in all_candidate_domains}
            fused_scores: Dict[str, float] = {}
            if self._fusion_engine is not None:
                try:
                    raw_fusion = self._fusion_engine.fuse(
                        semantic_scores=semantic_scores,
                        spectral_scores=spectral_scores,
                        confidence_scores=confidence_scores,
                    ) or {}
                    fused_scores = self._extract_fused_scores(raw_fusion)
                except Exception as exc:
                    logger.warning("Fusion failed: %s", exc)
            if not fused_scores and semantic_scores:
                fused_scores = dict(semantic_scores)
            selected_experts, cap_applied = _budget_select(
                fused_scores=fused_scores,
                confidence_scores=confidence_scores,
                min_score=DOMAIN_SCORE_THRESHOLD,
                max_experts=self.max_experts,
            )
            if cap_applied:
                _METRICS["budget_cap_applied"] += 1
                if should_log:
                    logger.debug("[ROUTER] Budget cap applied: trimmed to %d expert(s)", self.max_experts)
            create_new_expert = False
            override_applied = False
            if is_attribute_only and ENABLE_ATTRIBUTE_OVERRIDE and object_level_domains:
                override_candidates: List[Tuple[float, str]] = [
                    (fused_scores.get(d, 0.0), d)
                    for d in object_level_domains
                    if fused_scores.get(d, 0.0) >= DOMAIN_SCORE_THRESHOLD
                ]
                if override_candidates:
                    override_candidates.sort(key=lambda t: (-t[0], t[1]))
                    existing_set = set(selected_experts)
                    for score, domain in override_candidates:
                        if len(selected_experts) >= self.max_experts:
                            cap_applied = True
                            _METRICS["budget_cap_applied"] += 1
                            break
                        if domain not in existing_set:
                            selected_experts.append(domain)
                            existing_set.add(domain)
                    override_applied = True
                    _METRICS["attribute_override_applied"] += 1
            if not selected_experts and not create_new_expert:
                primary_fallback = base_result.get("primary_domain")
                if primary_fallback:
                    selected_experts = [primary_fallback]
            variance = self._compute_variance(fused_scores)
            if is_attribute_only and not override_applied:
                final_cls = "ATTRIBUTE_ONLY"
                _METRICS["attribute_only"] += 1
            elif create_new_expert and not selected_experts:
                final_cls = "NO_EXPERT_AVAILABLE"
                _METRICS["no_expert"] += 1
            elif len(selected_experts) == 1:
                final_cls = "SINGLE_DOMAIN"
                _METRICS["single_domain"] += 1
            elif len(selected_experts) > 1:
                final_cls = "MULTI_DOMAIN" if variance >= 0.1 else "AMBIGUOUS"
                if final_cls == "MULTI_DOMAIN":
                    _METRICS["multi_domain"] += 1
                else:
                    _METRICS["ambiguous"] += 1
            else:
                final_cls = "NO_EXPERT_AVAILABLE"
                _METRICS["no_expert"] += 1
            if create_new_expert:
                _METRICS["create_new_expert_true"] += 1
            coverage_met = bool(selected_experts)
            primary_domain = selected_experts[0] if selected_experts else None
            if should_log:
                logger.debug(
                    "[ROUTER] result  cls=%s  experts=%s  cap=%s  variance=%.4f",
                    final_cls, selected_experts, cap_applied, variance,
                )
            return RoutingResult(
                classification=final_cls,
                selected_domains=selected_experts,
                primary_domain=primary_domain,
                fusion_scores=fused_scores,
                coverage=1.0 if coverage_met else 0.0,
                create_new_expert=create_new_expert,
                metadata={
                    "selected_experts": selected_experts,
                    "candidate_domains": selected_experts,
                    "coverage_met": coverage_met,
                    "budget_cap_applied": cap_applied,
                    "lens_scores": {
                        "semantic": semantic_scores,
                        "spectral": spectral_scores,
                        "confidence": confidence_scores,
                    },
                    "fused_scores": fused_scores,
                    "variance": round(variance, 6),
                    "explanation": self._build_explanation(
                        base_result, final_cls, selected_experts,
                        coverage_met, create_new_expert, cap_applied,
                    ),
                },
            )
        except Exception as exc:
            logger.error("Critical routing error: %s", exc, exc_info=True)
            return self._fallback_result(f"Routing failed: {exc}")

    @staticmethod
    def _extract_semantic_scores(base_result: Dict) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for item in base_result.get("lens1_candidates") or []:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                domain, score = item[0], float(item[1])
                scores[domain] = min(1.0, max(0.0, score))
        return scores

    @staticmethod
    def _extract_fused_scores(fused_result: Dict) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for domain, info in fused_result.items():
            if isinstance(info, dict):
                scores[domain] = min(1.0, max(0.0, float(info.get("fused_score", 0.0))))
        return scores

    @staticmethod
    def _compute_variance(scores: Dict[str, float]) -> float:
        values = list(scores.values())
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)

    @staticmethod
    def _extract_object_level_domains(base_result: Dict) -> List[str]:
        domains: List[str] = []
        try:
            lens3 = base_result.get("lens3_signature") or {}
            for domain, features in (lens3.get("levels") or {}).get("core", {}).items():
                if features:
                    domains.append(domain)
            for exp in base_result.get("lens2_explanations") or []:
                if exp.get("level") == "object" and exp.get("matched_core"):
                    concept = exp.get("concept")
                    if concept and concept not in domains:
                        domains.append(concept)
        except Exception:
            pass
        return domains

    def _wrap_base_result(self, base_result: Dict) -> RoutingResult:
        primary = base_result.get("primary_domain")
        selected = [primary] if primary else []
        fused: Dict[str, float] = {d: 0.5 for d in selected}
        return RoutingResult(
            classification=base_result.get("classification", "NORMAL"),
            selected_domains=selected,
            primary_domain=primary,
            fusion_scores=fused,
            coverage=1.0 if selected else 0.0,
            create_new_expert=not bool(selected),
            metadata={
                "selected_experts": selected,
                "candidate_domains": selected,
                "coverage_met": bool(selected),
                "budget_cap_applied": False,
                "lens_scores": {
                    "semantic": self._extract_semantic_scores(base_result),
                    "spectral": {},
                    "confidence": {d: 0.5 for d in selected},
                },
                "fused_scores": fused,
                "variance": 0.0,
                "explanation": base_result.get("explanation", "Layer 1 baseline result."),
            },
        )

    @staticmethod
    def _fallback_result(explanation: str) -> RoutingResult:
        return RoutingResult(
            classification="NO_EXPERT_AVAILABLE",
            selected_domains=[],
            primary_domain=None,
            fusion_scores={},
            coverage=0.0,
            create_new_expert=True,
            metadata={
                "selected_experts": [],
                "candidate_domains": [],
                "coverage_met": False,
                "budget_cap_applied": False,
                "lens_scores": {"semantic": {}, "spectral": {}, "confidence": {}},
                "fused_scores": {},
                "variance": 0.0,
                "explanation": explanation,
            },
        )

    @staticmethod
    def _build_explanation(
        base_result: Dict,
        classification: str,
        selected_experts: List[str],
        coverage_met: bool,
        create_new_expert: bool,
        budget_cap_applied: bool,
    ) -> str:
        parts: List[str] = []
        base_exp = base_result.get("explanation", "")
        if base_exp:
            parts.append(f"Base: {base_exp}")
        cls_msg = {
            "ATTRIBUTE_ONLY": (
                "Pure reasoning query — no structural domain match. "
                "All available experts will be evaluated by the reasoning pipeline."
            ),
            "SINGLE_DOMAIN": f"Single domain: {selected_experts[0] if selected_experts else '?'}.",
            "MULTI_DOMAIN": f"Multiple domains: {', '.join(selected_experts)}.",
            "AMBIGUOUS": f"Ambiguous — similar relevance across: {', '.join(selected_experts)}.",
            "NO_EXPERT_AVAILABLE": "No suitable expert found.",
        }.get(classification, "")
        if cls_msg:
            parts.append(cls_msg)
        if budget_cap_applied:
            parts.append(
                f"Budget cap applied: output limited to {_ABSOLUTE_MAX_EXPERTS} expert(s) "
                "to prevent fan-out explosion."
            )
        if coverage_met:
            parts.append("Coverage satisfied.")
        elif create_new_expert:
            parts.append("Coverage not met — new expert may be required.")
        return " ".join(parts)

    @staticmethod
    def get_metrics() -> Dict[str, Any]:
        total = _METRICS["total_requests"]
        if total == 0:
            return {**_METRICS, "percentages": {}}
        pcts = {
            f"{k}_pct": round(v / total * 100, 2)
            for k, v in _METRICS.items()
            if k != "total_requests"
        }
        return {**_METRICS, "percentages": pcts}
