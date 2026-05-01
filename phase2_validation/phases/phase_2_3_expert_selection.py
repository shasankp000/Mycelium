"""
Phase 2.3 — Expert Pool Selection & Ranking.

This module implements the third phase of the six-phase reasoning
pipeline.  It ranks, filters, groups, and selects experts based
on the semantic understanding produced by Phase 2.2.

Components:
    * ExpertRankingEngine — multi-criteria expert scoring
    * ExpertPoolFilter — threshold-based pool filtering
    * ExpertGrouping — domain grouping and diverse selection
    * ExpertSelectionPipeline — orchestrator

Example:
    >>> from phase2_validation.phases.phase_2_3_expert_selection import (
    ...     ExpertSelectionPipeline,
    ... )
    >>> pipeline = ExpertSelectionPipeline()
    >>> result = pipeline.select_experts(
    ...     semantic_result, experts
    ... )
    >>> print(result.total_selected)
"""

import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

from phase2_validation.config.phase2_config import Phase2Config
from phase2_validation.utils.semantic_types import (
    ExpertSelectionResult,
    RankedExpert,
    SemanticResult,
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Custom exceptions
# ------------------------------------------------------------------


class ExpertSelectionError(Exception):
    """Base exception for expert selection failures."""


class NoSuitableExpertError(ExpertSelectionError):
    """Raised when no expert meets the minimum threshold."""


class InvalidStrategyError(ExpertSelectionError):
    """Raised when an unsupported selection strategy is used."""


# ------------------------------------------------------------------
# ExpertRankingEngine
# ------------------------------------------------------------------


class ExpertRankingEngine:
    """Rank experts using multi-criteria scoring.

    Weights: domain (50%) + concepts (30%) + graph (10%) +
    confidence (10%).
    """

    def rank_experts(
        self,
        semantic_result: SemanticResult,
        available_experts: List[Dict[str, object]],
    ) -> List[RankedExpert]:
        """Rank all available experts against the semantic result.

        Args:
            semantic_result: Output of Phase 2.2.
            available_experts: List of expert config dicts.
                Each must have ``name`` and ``domain`` keys.

        Returns:
            List of ``RankedExpert`` sorted by match score
            (descending).
        """
        ranked: List[RankedExpert] = []
        for expert in available_experts:
            score = self.compute_expert_match_score(
                semantic_result, expert
            )
            factors = self._breakdown_factors(
                semantic_result, expert
            )
            low, high = self.compute_confidence_interval(
                score, factors
            )
            ranked.append(
                RankedExpert(
                    name=str(expert.get("name", "")),
                    domain=str(expert.get("domain", "")),
                    match_score=round(score, 6),
                    confidence_low=round(low, 6),
                    confidence_high=round(high, 6),
                    ranking_factors=factors,
                )
            )
        ranked.sort(
            key=lambda r: r.match_score, reverse=True
        )
        return ranked

    def compute_expert_match_score(
        self,
        semantic_result: SemanticResult,
        expert_config: Dict[str, object],
    ) -> float:
        """Compute the match score for a single expert.

        Args:
            semantic_result: Semantic understanding output.
            expert_config: Expert configuration dict with
                ``name`` and ``domain``.

        Returns:
            Score in ``[0, 1]``.
        """
        factors = self._breakdown_factors(
            semantic_result, expert_config
        )
        score = (
            0.5 * factors.get("domain", 0.0)
            + 0.3 * factors.get("concepts", 0.0)
            + 0.1 * factors.get("graph", 0.0)
            + 0.1 * factors.get("confidence", 0.0)
        )
        return min(max(score, 0.0), 1.0)

    def compute_confidence_interval(
        self,
        score: float,
        factors: Dict[str, float],
    ) -> Tuple[float, float]:
        """Compute a confidence interval around the score.

        Uses the variance of factor values to estimate
        uncertainty.

        Args:
            score: Point estimate of match quality.
            factors: Breakdown of scoring components.

        Returns:
            Tuple of ``(lower, upper)`` bounds.
        """
        values = list(factors.values()) if factors else [0.0]
        if len(values) < 2:
            margin = 0.1
        else:
            mean = sum(values) / len(values)
            variance = sum(
                (v - mean) ** 2 for v in values
            ) / len(values)
            margin = min(math.sqrt(variance), 0.3)

        low = max(score - margin, 0.0)
        high = min(score + margin, 1.0)
        return (low, high)

    # ---- internals -------------------------------------------------

    def _breakdown_factors(
        self,
        semantic_result: SemanticResult,
        expert_config: Dict[str, object],
    ) -> Dict[str, float]:
        """Compute individual ranking factors."""
        domain = str(
            expert_config.get("domain", "")
        ).lower()

        # Domain factor
        domain_score = semantic_result.domain_relevance_scores.get(
            domain, 0.0
        )

        # Concept factor
        concept_score = 0.0
        if semantic_result.concept_scores:
            concept_score = max(
                semantic_result.concept_scores.values()
            )

        # Graph factor
        graph_score = min(
            semantic_result.graph_density * 2.0, 1.0
        )

        # Confidence factor
        confidence = semantic_result.confidence_score

        return {
            "domain": round(domain_score, 6),
            "concepts": round(concept_score, 6),
            "graph": round(graph_score, 6),
            "confidence": round(confidence, 6),
        }


# ------------------------------------------------------------------
# ExpertPoolFilter
# ------------------------------------------------------------------


class ExpertPoolFilter:
    """Filter the expert pool based on thresholds."""

    def filter_pool(
        self,
        experts: List[RankedExpert],
        semantic_result: SemanticResult,
    ) -> Dict[str, List[RankedExpert]]:
        """Split experts into selected and filtered-out groups.

        Experts with ``match_score`` below 0.1 are filtered out.

        Args:
            experts: Ranked expert list.
            semantic_result: Semantic understanding output.

        Returns:
            Dictionary with ``selected`` and ``filtered_out``
            lists.
        """
        selected: List[RankedExpert] = []
        filtered_out: List[RankedExpert] = []

        for expert in experts:
            if expert.match_score >= 0.1:
                selected.append(expert)
            else:
                filtered_out.append(expert)

        return {
            "selected": selected,
            "filtered_out": filtered_out,
        }

    def apply_thresholds(
        self,
        ranked_experts: List[RankedExpert],
        min_score: float = 0.3,
        max_pool_size: int = 4,
    ) -> List[RankedExpert]:
        """Apply score and size thresholds.

        Args:
            ranked_experts: Experts sorted by score.
            min_score: Minimum match score to include.
            max_pool_size: Maximum number of experts.

        Returns:
            Filtered list.
        """
        above = [
            e for e in ranked_experts
            if e.match_score >= min_score
        ]
        return above[:max_pool_size]

    def detect_no_suitable_expert(
        self,
        ranked_experts: List[RankedExpert],
    ) -> bool:
        """Determine if no expert is suitable.

        Returns ``True`` when the best expert score is below
        0.2 or the list is empty.

        Args:
            ranked_experts: Experts sorted by score.

        Returns:
            ``True`` if no suitable expert exists.
        """
        if not ranked_experts:
            return True
        return ranked_experts[0].match_score < 0.2


# ------------------------------------------------------------------
# ExpertGrouping
# ------------------------------------------------------------------


class ExpertGrouping:
    """Group and diversify expert pools."""

    def group_by_domain(
        self,
        experts: List[RankedExpert],
    ) -> Dict[str, List[RankedExpert]]:
        """Group experts by their domain.

        Args:
            experts: List of ranked experts.

        Returns:
            Dictionary mapping domain to list of experts.
        """
        groups: Dict[str, List[RankedExpert]] = {}
        for expert in experts:
            groups.setdefault(expert.domain, []).append(expert)
        return groups

    def prioritize_experts(
        self,
        experts: List[RankedExpert],
        strategy: str = "greedy",
    ) -> List[RankedExpert]:
        """Re-order experts according to a strategy.

        Supported strategies:
            * ``greedy`` — sort by descending match score.
            * ``diversity`` — interleave domains.
            * ``uncertainty`` — prefer low-confidence experts
              for exploration.

        Args:
            experts: List of ranked experts.
            strategy: Selection strategy name.

        Returns:
            Re-ordered list.

        Raises:
            InvalidStrategyError: If strategy is unsupported.
        """
        if strategy == "greedy":
            return sorted(
                experts,
                key=lambda e: e.match_score,
                reverse=True,
            )

        if strategy == "diversity":
            return self._diversity_order(experts)

        if strategy == "uncertainty":
            return sorted(
                experts,
                key=lambda e: (
                    e.confidence_high - e.confidence_low
                ),
                reverse=True,
            )

        raise InvalidStrategyError(
            f"Unknown strategy: {strategy!r}"
        )

    def select_diverse_pool(
        self,
        experts: List[RankedExpert],
        pool_size: int = 3,
    ) -> List[RankedExpert]:
        """Select a diverse pool using dissimilarity.

        Greedily picks experts that maximise domain coverage.

        Args:
            experts: Ranked experts.
            pool_size: Desired pool size.

        Returns:
            List of up to *pool_size* diverse experts.
        """
        if len(experts) <= pool_size:
            return list(experts)

        selected: List[RankedExpert] = []
        seen_domains: set = set()

        # First pass: one expert per domain
        for expert in experts:
            if expert.domain not in seen_domains:
                selected.append(expert)
                seen_domains.add(expert.domain)
            if len(selected) >= pool_size:
                break

        # Second pass: fill remaining slots by score
        if len(selected) < pool_size:
            remaining = [
                e for e in experts if e not in selected
            ]
            remaining.sort(
                key=lambda e: e.match_score, reverse=True
            )
            for expert in remaining:
                if len(selected) >= pool_size:
                    break
                selected.append(expert)

        return selected

    # ---- internals -------------------------------------------------

    @staticmethod
    def _diversity_order(
        experts: List[RankedExpert],
    ) -> List[RankedExpert]:
        """Interleave experts from different domains."""
        from collections import defaultdict
        domain_queues: Dict[str, List[RankedExpert]] = (
            defaultdict(list)
        )
        for expert in sorted(
            experts,
            key=lambda e: e.match_score,
            reverse=True,
        ):
            domain_queues[expert.domain].append(expert)

        result: List[RankedExpert] = []
        keys = list(domain_queues.keys())
        idx = 0
        while any(domain_queues.values()):
            domain = keys[idx % len(keys)]
            if domain_queues[domain]:
                result.append(domain_queues[domain].pop(0))
            idx += 1
            # Safety: break if stuck
            if idx > len(experts) * 2:
                break
        return result


# ------------------------------------------------------------------
# ExpertSelectionPipeline
# ------------------------------------------------------------------


class ExpertSelectionPipeline:
    """Orchestrate all Phase 2.3 components.

    Usage:
        >>> pipeline = ExpertSelectionPipeline()
        >>> result = pipeline.select_experts(
        ...     semantic_result, experts
        ... )
    """

    def __init__(
        self,
        config: Optional[Phase2Config] = None,
    ) -> None:
        self._config = config or Phase2Config()
        self._ranking_engine = ExpertRankingEngine()
        self._pool_filter = ExpertPoolFilter()
        self._grouping = ExpertGrouping()

    def select_experts(
        self,
        semantic_result: SemanticResult,
        available_experts: List[Dict[str, object]],
        max_experts: int = 4,
        min_score: float = 0.3,
        strategy: str = "greedy",
            routing_context: Optional[Dict[str, Any]] = None,
    ) -> ExpertSelectionResult:
        """Run the full expert selection pipeline.

        Args:
            semantic_result: Output of Phase 2.2.
            available_experts: List of expert config dicts.
            max_experts: Maximum experts to select.
            min_score: Minimum match score threshold.
            strategy: Selection strategy
                (``greedy``, ``diversity``, ``uncertainty``).

        Returns:
            An ``ExpertSelectionResult`` with all fields.
        """
        start = time.perf_counter()
        warnings: List[str] = []
        cold_start = False
        ranked: List[RankedExpert] = []
        ranked_all: List[RankedExpert] = []
        ranked_for_selection: List[RankedExpert] = []

        try:
            # Step 1: rank
            ranked_all = self._ranking_engine.rank_experts(
                semantic_result, available_experts
            )
            ranked = list(ranked_all)
            ranked_for_selection = list(ranked_all)

            routing_class, preferred_domains = (
                self._extract_routing_preferences(routing_context)
            )

            if routing_class == "NO_EXPERT_AVAILABLE":
                cold_start = True
                warnings.append(
                    "Layer 0 reported no expert available; "
                    "skipping expert selection"
                )
                selected_pool = []
                filtered_out = ranked_all
            else:
                if preferred_domains:
                    prioritized = [
                        e for e in ranked_all
                        if e.domain in preferred_domains
                    ]
                    if prioritized:
                        ranked_for_selection = prioritized
                        warnings.append(
                            "Layer 0 routing prior applied to "
                            "expert selection"
                        )
                    else:
                        warnings.append(
                            "Layer 0 routing prior did not match "
                            "any experts; using semantic ranking"
                        )

                # Step 2: filter
                pool = self._pool_filter.filter_pool(
                    ranked_for_selection, semantic_result
                )
                selected_pool = pool["selected"]
                filtered_out = pool["filtered_out"]

                # Step 3: apply thresholds
                selected_pool = self._pool_filter.apply_thresholds(
                    selected_pool, min_score, max_experts
                )

                # Step 4: detect cold start
                if self._pool_filter.detect_no_suitable_expert(
                    ranked_for_selection
                ):
                    cold_start = True
                    warnings.append(
                        "No expert meets threshold; "
                        "using cold-start fallback"
                    )
                    # Fallback: return top expert regardless
                    if ranked_for_selection:
                        selected_pool = [ranked_for_selection[0]]
                        filtered_out = ranked_for_selection[1:]

                # Step 5: prioritize / diversify
                if selected_pool:
                    if strategy == "diversity":
                        selected_pool = (
                            self._grouping.select_diverse_pool(
                                selected_pool, max_experts
                            )
                        )
                    else:
                        selected_pool = (
                            self._grouping.prioritize_experts(
                                selected_pool, strategy
                            )
                        )[:max_experts]
                elif ranked_for_selection:
                    selected_pool = [ranked_for_selection[0]]
                    filtered_out = ranked_for_selection[1:]

                # Build filtered_out from ranked minus selected
                selected_names = {e.name for e in selected_pool}
                filtered_out = [
                    e for e in ranked_all
                    if e.name not in selected_names
                ]

        except InvalidStrategyError:
            raise
        except Exception as exc:
            logger.error("Expert selection error: %s", exc)
            warnings.append(f"Selection error: {exc}")
            selected_pool = []
            filtered_out = []
            ranked = []

        elapsed = (time.perf_counter() - start) * 1000

        expert_scores = {
            e.name: e.match_score for e in ranked
        }

        confidence = self._compute_selection_confidence(
            selected_pool
        )

        return ExpertSelectionResult(
            selected_experts=selected_pool,
            filtered_out_experts=filtered_out,
            selection_strategy=strategy,
            total_candidates=len(available_experts),
            total_selected=len(selected_pool),
            expert_scores=expert_scores,
            confidence_in_selection=round(confidence, 6),
            processing_time_ms=round(elapsed, 2),
            cold_start_fallback_used=cold_start,
            warnings=warnings,
        )

    @staticmethod
    def build_default_experts(
        semantic_result: SemanticResult,
    ) -> List[Dict[str, object]]:
        experts: List[Dict[str, object]] = []
        domains = list(semantic_result.domain_relevance_scores.keys())
        if not domains:
            domains = list(semantic_result.ranked_domains)

        for domain in domains:
            if not domain:
                continue
            experts.append({
                "name": f"expert_{domain}",
                "domain": domain,
            })

        return experts

    def get_selection_justification(
        self,
        result: ExpertSelectionResult,
    ) -> str:
        """Generate a human-readable justification.

        Args:
            result: Expert selection result.

        Returns:
            Justification string.
        """
        lines: List[str] = [
            f"Strategy: {result.selection_strategy}",
            f"Candidates: {result.total_candidates}",
            f"Selected: {result.total_selected}",
            f"Confidence: {result.confidence_in_selection:.2f}",
        ]
        if result.cold_start_fallback_used:
            lines.append("Note: cold-start fallback was used")

        for expert in result.selected_experts:
            lines.append(
                f"  - {expert.name} ({expert.domain}): "
                f"{expert.match_score:.4f} "
                f"[{expert.confidence_low:.2f}, "
                f"{expert.confidence_high:.2f}]"
            )

        if result.warnings:
            lines.append("Warnings:")
            for w in result.warnings:
                lines.append(f"  ! {w}")

        return "\n".join(lines)

    # ---- internals -------------------------------------------------

    @staticmethod
    def _compute_selection_confidence(
        selected: List[RankedExpert],
    ) -> float:
        """Derive overall confidence from selected experts."""
        if not selected:
            return 0.0
        avg = sum(e.match_score for e in selected) / len(
            selected
        )
        return min(avg, 1.0)

    @staticmethod
    def _extract_routing_preferences(
        routing_context: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[str], List[str]]:
        if not routing_context:
            return None, []

        classification = routing_context.get("classification")
        preferred: List[str] = []

        for key in ("selected_experts", "candidate_domains"):
            values = routing_context.get(key) or []
            if isinstance(values, list):
                preferred.extend(
                    [str(v) for v in values if v is not None]
                )

        primary = routing_context.get("primary_domain")
        if primary:
            preferred.append(str(primary))

        deduped: List[str] = []
        seen: set = set()
        for item in preferred:
            key = item.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(item)

        return classification, deduped
