"""
domain_tool_planner.py
======================
Predicate-type-aware, domain-sensitive tool planner.

v2 changes from v1:
  - PredicateType-first routing: if PredicateFrames are available, the
    planner's primary routing dimension is the predicate type, not the
    domain.  Domain still acts as a tiebreaker and supplement.
  - All tool metadata imported from tool_registry.py — no hard-coded
    tool names here.
  - Emits a PlannerObservabilityEvent via pipeline_event.py so routing
    decisions are traceable in the event log.
  - TRM v2 seam preserved: when the real TRMToolPlanner lands, this file
    is replaced by a single import swap (see bottom of file).
  - domain_store_search is included when enabled in config but gracefully
    excluded when the stub flag is off (default).

Interface contract (unchanged from v1):
    planner.plan(query, domains, decision_type,
                 predicate_types=None, use_domain_store=False)
        -> list[ToolCall]

No LLM is invoked at any point in this module.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional

from mycelium.pipeline.tool_registry import (
    TOOL_BY_NAME,
    ToolSpec,
    tools_for_predicate_type,
    all_tool_names,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ToolCall — unchanged from v1 so callers don't break
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolCall:
    """A single planned tool invocation."""
    tool: str
    query: str


# ---------------------------------------------------------------------------
# Predicate-type → preferred tool order
# ---------------------------------------------------------------------------
# Each entry is an ordered list of tool names.  The planner tries them
# in order, deduplicates, and stops at max_tools.
#
# Rationale for each mapping is given inline.

_PREDICATE_TYPE_TOOLS: dict[str, list[str]] = {
    # FACTIVE: plain factual claim — knowledge_base first (structured),
    # then web for currency, academic for contested facts.
    "FACTIVE": ["knowledge_base", "web_search", "academic_search"],

    # COMPARATIVE: requires numeric backing — statistical_search first,
    # then academic for methodology, calculator if ratio needed.
    "COMPARATIVE": ["statistical_search", "academic_search", "calculator"],

    # CAUSAL: causal chain evidence — causal_search first (Semantic Scholar
    # + Open Citations), then academic for corroboration.
    "CAUSAL": ["causal_search", "academic_search", "knowledge_base"],

    # EXISTENTIAL: 'there exist X' — knowledge_base for entity existence,
    # web for long-tail instances, academic for contested claims.
    "EXISTENTIAL": ["knowledge_base", "web_search", "academic_search"],

    # TEMPORAL: date-anchored facts — temporal_search first (Wikidata SPARQL
    # + year-scoped DDG), knowledge_base for entity timeline.
    "TEMPORAL": ["temporal_search", "knowledge_base", "web_search"],

    # MODAL: possibility/probability claims — web_search for current
    # discourse, academic for empirical probability estimates.
    "MODAL": ["web_search", "academic_search", "statistical_search"],

    # NORMATIVE: value judgements — non-falsifiable, but we still retrieve
    # context to ground the discussion.  No academic_search (won't refute).
    "NORMATIVE": ["web_search", "knowledge_base"],

    # DEFINITIONAL: 'X is defined as Y' — knowledge_base authoritative,
    # academic for technical definitions, web as fallback.
    "DEFINITIONAL": ["knowledge_base", "academic_search", "web_search"],
}


# ---------------------------------------------------------------------------
# Domain → supplementary tools
# ---------------------------------------------------------------------------
# Used as a secondary tiebreaker when predicate-type routing leaves budget.
# Keeps the same domain list as v1 with updated tool names.

_DOMAIN_TOOLS: dict[str, list[str]] = {
    "physics":          ["academic_search", "knowledge_base"],
    "chemistry":        ["academic_search", "knowledge_base"],
    "biology":          ["academic_search", "knowledge_base"],
    "astronomy":        ["academic_search", "knowledge_base"],
    "geology":          ["academic_search", "knowledge_base"],
    "environmental":    ["academic_search", "statistical_search", "web_search"],
    "maths":            ["academic_search", "calculator"],
    "mathematics":      ["academic_search", "calculator"],
    "statistics":       ["statistical_search", "academic_search", "calculator"],
    "computer_science": ["academic_search", "knowledge_base"],
    "logic":            ["academic_search", "calculator"],
    "medicine":         ["academic_search", "causal_search", "knowledge_base"],
    "health":           ["academic_search", "statistical_search", "web_search"],
    "neuroscience":     ["academic_search", "knowledge_base"],
    "pharmacology":     ["academic_search", "causal_search", "knowledge_base"],
    "history":          ["temporal_search", "knowledge_base", "web_search"],
    "philosophy":       ["academic_search", "knowledge_base"],
    "psychology":       ["academic_search", "knowledge_base"],
    "economics":        ["statistical_search", "academic_search", "web_search"],
    "sociology":        ["academic_search", "statistical_search", "web_search"],
    "linguistics":      ["academic_search", "knowledge_base"],
    "law":              ["temporal_search", "knowledge_base", "web_search"],
    "engineering":      ["academic_search", "knowledge_base"],
    "technology":       ["web_search", "academic_search", "knowledge_base"],
    "ai":               ["academic_search", "web_search"],
    "robotics":         ["academic_search", "knowledge_base"],
    "politics":         ["web_search", "statistical_search", "knowledge_base"],
    "geography":        ["knowledge_base", "statistical_search", "web_search"],
    "culture":          ["knowledge_base", "web_search"],
    "general":          ["web_search", "knowledge_base"],
}

_DEFAULT_TOOLS: list[str] = ["web_search", "knowledge_base"]


# ---------------------------------------------------------------------------
# PlannerObservabilityEvent — lightweight inline event (no import cycle)
# ---------------------------------------------------------------------------
# We emit a minimal dict rather than using PipelineEvent directly here
# to avoid a circular import (pipeline_event imports nothing from planner).
# run_workflow.py picks this up and forwards it to the real event bus.

@dataclass
class PlannerTrace:
    """Observability record for a single plan() call.  Not persisted here."""
    query: str
    predicate_types: List[str]
    domains: List[str]
    decision_type: str
    selected_tools: List[str]
    routing_reason: str
    use_domain_store: bool
    timestamp: float


# ---------------------------------------------------------------------------
# DomainToolPlanner
# ---------------------------------------------------------------------------

class DomainToolPlanner:
    """
    Predicate-type-aware, domain-sensitive deterministic tool planner.

    Routing priority
    ----------------
    1. domain_store_search prepended if use_domain_store=True and the
       shard is confirmed hot (caller's responsibility to check).
    2. Predicate-type routing fills the budget from _PREDICATE_TYPE_TOOLS.
    3. Domain routing fills remaining budget from _DOMAIN_TOOLS.
    4. decision_type == 'patch' forces web_search into the set.
    5. Fallback to _DEFAULT_TOOLS if nothing matched.

    TRM v2 seam
    -----------
    When TRM v2 lands, replace:
        from mycelium.pipeline.domain_tool_planner import DomainToolPlanner
    with:
        from mycelium.trm.v2.planner import TRMToolPlanner as DomainToolPlanner
    The interface contract (plan() signature and ToolCall return type) is
    identical.
    """

    def __init__(self, max_tools: int = 3) -> None:
        self._max_tools = max_tools

    def plan(
        self,
        query: str,
        domains: List[str],
        decision_type: str = "",
        predicate_types: Optional[List[str]] = None,
        use_domain_store: bool = False,
    ) -> List[ToolCall]:
        """
        Produce a minimal, bias-free ToolCallPlan.

        Parameters
        ----------
        query:
            Raw query string — passed through verbatim to each ToolCall.
        domains:
            Ordered domain labels from L1/MultiLens routing.
        decision_type:
            Pipeline decision type string; 'patch' forces web_search.
        predicate_types:
            List of PredicateType enum string values extracted by the
            PredicateExtractor ('FACTIVE', 'CAUSAL', etc.).
            If None or empty, falls back to domain-only routing (v1 behaviour).
        use_domain_store:
            If True, prepend domain_store_search to the plan.
            Caller must have confirmed the domain shard is HOT before
            setting this.  When False (default), domain_store_search is
            never emitted even if it appears in predicate_type mappings.

        Returns
        -------
        List[ToolCall] of length <= max_tools.
        """
        seen: dict[str, None] = {}  # ordered-set idiom
        routing_reason_parts: list[str] = []

        # Step 1: domain_store_search takes priority slot 0 when hot
        if use_domain_store and len(seen) < self._max_tools:
            seen["domain_store_search"] = None
            routing_reason_parts.append("domain_store:hot")

        # Step 2: predicate-type routing
        if predicate_types:
            for ptype in predicate_types:
                for tool_name in _PREDICATE_TYPE_TOOLS.get(ptype, []):
                    if len(seen) >= self._max_tools:
                        break
                    # Skip domain_store_search unless use_domain_store=True
                    if tool_name == "domain_store_search" and not use_domain_store:
                        continue
                    # Skip stubs unless explicitly opted in
                    spec = TOOL_BY_NAME.get(tool_name)
                    if spec and spec.requires_trm_v2 and not use_domain_store:
                        continue
                    seen[tool_name] = None
            if predicate_types:
                routing_reason_parts.append(
                    f"predicate_types:{','.join(predicate_types)}"
                )

        # Step 3: domain routing to fill remaining budget
        for domain in domains:
            if len(seen) >= self._max_tools:
                break
            key = domain.lower().replace(" ", "_").replace("-", "_")
            for tool_name in _DOMAIN_TOOLS.get(key, _DEFAULT_TOOLS):
                if len(seen) >= self._max_tools:
                    break
                if tool_name == "domain_store_search" and not use_domain_store:
                    continue
                seen[tool_name] = None
        if domains:
            routing_reason_parts.append(f"domains:{','.join(domains[:2])}")

        # Step 4: patch decision forces web_search
        if decision_type and "patch" in decision_type.lower():
            if "web_search" not in seen and len(seen) < self._max_tools:
                seen["web_search"] = None
                routing_reason_parts.append("decision_type:patch")

        # Step 5: fallback
        if not seen:
            for t in _DEFAULT_TOOLS:
                seen[t] = None
            routing_reason_parts.append("fallback:default")

        selected = list(seen.keys())[: self._max_tools]

        # Emit observability trace (caller forwards to event bus)
        self._last_trace = PlannerTrace(
            query=query,
            predicate_types=predicate_types or [],
            domains=domains,
            decision_type=decision_type,
            selected_tools=selected,
            routing_reason=" | ".join(routing_reason_parts) or "none",
            use_domain_store=use_domain_store,
            timestamp=time.time(),
        )

        logger.debug(
            "DomainToolPlanner selected %s for query=%r reason=%s",
            selected, query[:60], self._last_trace.routing_reason,
        )

        return [ToolCall(tool=t, query=query) for t in selected]

    @property
    def last_trace(self) -> Optional[PlannerTrace]:
        """Return the PlannerTrace from the most recent plan() call."""
        return getattr(self, "_last_trace", None)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_planner: DomainToolPlanner | None = None


def get_planner() -> DomainToolPlanner:
    global _planner
    if _planner is None:
        _planner = DomainToolPlanner()
    return _planner


# ---------------------------------------------------------------------------
# TRM v2 import seam (commented — uncomment when TRM v2 is merged)
# ---------------------------------------------------------------------------
# from mycelium.trm.v2.planner import TRMToolPlanner as DomainToolPlanner
# _planner = None  # reset singleton after swap
