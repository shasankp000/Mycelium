"""
domain_tool_planner.py
======================
Deterministic domain-to-tool planner.

This module is the **TRM seam** in the sandbox pipeline.  Today it uses a
static domain→tool map to produce a minimal, bias-free ToolCallPlan from
the pipeline's structured outputs (domains, decision_type, query).

When the full Temporal Reasoning Module (TRM) architecture lands, this
file is replaced by a single import swap:

    # Before (Option A stub):
    from domain_tool_planner import DomainToolPlanner

    # After (real TRM):
    from trm.planner import TRMToolPlanner as DomainToolPlanner

The interface contract DomainToolPlanner exposes is:

    planner.plan(query, domains, decision_type) -> list[ToolCall]

where ToolCall is a plain dataclass:
    .tool   str   — must match a tool name in mcp_tools_server
    .query  str   — the argument passed to that tool

No LLM is invoked at any point in this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ToolCall:
    """A single planned tool invocation."""
    tool: str    # matches mcp_tools_server tool name
    query: str   # argument forwarded to the tool


# ---------------------------------------------------------------------------
# Domain → tool map
#
# Rules:
#   - academic_search  : empirical / theoretical scientific domains
#   - knowledge_base   : entity-heavy or taxonomy-rich domains
#   - web_search       : current-events, social, fast-changing facts
#   - calculator       : quantitative / numerical domains
#
# When the TRM lands this map becomes the *fallback* for domains the TRM
# has not yet learned a signature for.
# ---------------------------------------------------------------------------

_DOMAIN_TOOLS: dict[str, list[str]] = {
    # Natural sciences
    "physics":          ["academic_search", "knowledge_base"],
    "chemistry":        ["academic_search", "knowledge_base"],
    "biology":          ["academic_search", "knowledge_base"],
    "astronomy":        ["academic_search", "knowledge_base"],
    "geology":          ["academic_search", "knowledge_base"],
    "environmental":    ["academic_search", "web_search"],

    # Formal sciences
    "maths":            ["academic_search", "calculator"],
    "mathematics":      ["academic_search", "calculator"],
    "statistics":       ["academic_search", "calculator"],
    "computer_science": ["academic_search", "knowledge_base"],
    "logic":            ["academic_search", "calculator"],

    # Medicine / health
    "medicine":         ["academic_search", "knowledge_base"],
    "health":           ["academic_search", "web_search"],
    "neuroscience":     ["academic_search", "knowledge_base"],
    "pharmacology":     ["academic_search", "knowledge_base"],

    # Social sciences / humanities
    "history":          ["knowledge_base", "web_search"],
    "philosophy":       ["academic_search", "knowledge_base"],
    "psychology":       ["academic_search", "knowledge_base"],
    "economics":        ["academic_search", "web_search"],
    "sociology":        ["academic_search", "web_search"],
    "linguistics":      ["academic_search", "knowledge_base"],
    "law":              ["knowledge_base", "web_search"],

    # Technology / engineering
    "engineering":      ["academic_search", "knowledge_base"],
    "technology":       ["web_search", "knowledge_base"],
    "ai":               ["academic_search", "web_search"],
    "robotics":         ["academic_search", "knowledge_base"],

    # Current events / general
    "politics":         ["web_search", "knowledge_base"],
    "geography":        ["knowledge_base", "web_search"],
    "culture":          ["knowledge_base", "web_search"],
    "general":          ["web_search", "knowledge_base"],
}

# Fallback when a domain is not in the map above
_DEFAULT_TOOLS: list[str] = ["web_search", "knowledge_base"]


class DomainToolPlanner:
    """
    Deterministic planner: maps pipeline outputs → minimal ToolCallPlan.

    This is the component that will be replaced by TRM.plan_tools() once
    the full TRM architecture is implemented.  The interface is kept thin
    on purpose so the swap is a single line change.

    Parameters
    ----------
    max_tools : int
        Hard cap on distinct tools per plan.  Keeps sandbox time bounded.
    """

    def __init__(self, max_tools: int = 3) -> None:
        self._max_tools = max_tools

    def plan(
        self,
        query: str,
        domains: List[str],
        decision_type: str = "",
    ) -> List[ToolCall]:
        """
        Produce a minimal, ordered list of ToolCall objects.

        Logic
        -----
        1. Collect the tool set for each domain from the static map.
        2. Preserve insertion order so higher-priority tools appear first.
        3. Always add web_search for CREATE_NEW_PATCH queries (no trained
           expert → broader net needed).
        4. Cap at self._max_tools to bound sandbox wall time.
        5. Map each selected tool to the user query as its argument.

        Parameters
        ----------
        query        : str   — original user query string
        domains      : list  — domain tags from RoutingSummary.selected_domains
        decision_type: str   — decision type string from ExpertDecisionSummary
        """
        seen: dict[str, None] = {}  # ordered set via dict

        for domain in domains:
            key = domain.lower().replace(" ", "_").replace("-", "_")
            for tool in _DOMAIN_TOOLS.get(key, _DEFAULT_TOOLS):
                seen[tool] = None

        # Patch queries need a wider net — no trained expert, so fallback
        # to web_search even if the domain map didn't include it.
        if decision_type and "patch" in decision_type.lower():
            seen["web_search"] = None

        # If nothing resolved (empty domains list), use defaults
        if not seen:
            for t in _DEFAULT_TOOLS:
                seen[t] = None

        selected = list(seen.keys())[: self._max_tools]
        return [ToolCall(tool=t, query=query) for t in selected]


# Process-level singleton — callers should use this rather than
# constructing their own instance.
_planner: DomainToolPlanner | None = None


def get_planner() -> DomainToolPlanner:
    global _planner
    if _planner is None:
        _planner = DomainToolPlanner()
    return _planner
