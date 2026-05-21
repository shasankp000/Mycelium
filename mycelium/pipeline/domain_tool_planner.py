"""
domain_tool_planner.py
======================
Deterministic domain-to-tool planner.

This module is the **TRM seam** in the sandbox pipeline.  Today it uses a
static domain\u2192tool map to produce a minimal, bias-free ToolCallPlan from
the pipeline's structured outputs (domains, decision_type, query).

When the full Temporal Reasoning Module (TRM) architecture lands, this
file is replaced by a single import swap:

    # Before (Option A stub):
    from mycelium.pipeline.domain_tool_planner import DomainToolPlanner

    # After (real TRM):
    from trm.planner import TRMToolPlanner as DomainToolPlanner

The interface contract DomainToolPlanner exposes is:

    planner.plan(query, domains, decision_type) -> list[ToolCall]

where ToolCall is a plain dataclass:
    .tool   str   \u2014 must match a tool name in mcp_tools_server
    .query  str   \u2014 the argument passed to that tool

No LLM is invoked at any point in this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ToolCall:
    """A single planned tool invocation."""
    tool: str
    query: str


_DOMAIN_TOOLS: dict[str, list[str]] = {
    "physics":          ["academic_search", "knowledge_base"],
    "chemistry":        ["academic_search", "knowledge_base"],
    "biology":          ["academic_search", "knowledge_base"],
    "astronomy":        ["academic_search", "knowledge_base"],
    "geology":          ["academic_search", "knowledge_base"],
    "environmental":    ["academic_search", "web_search"],
    "maths":            ["academic_search", "calculator"],
    "mathematics":      ["academic_search", "calculator"],
    "statistics":       ["academic_search", "calculator"],
    "computer_science": ["academic_search", "knowledge_base"],
    "logic":            ["academic_search", "calculator"],
    "medicine":         ["academic_search", "knowledge_base"],
    "health":           ["academic_search", "web_search"],
    "neuroscience":     ["academic_search", "knowledge_base"],
    "pharmacology":     ["academic_search", "knowledge_base"],
    "history":          ["knowledge_base", "web_search"],
    "philosophy":       ["academic_search", "knowledge_base"],
    "psychology":       ["academic_search", "knowledge_base"],
    "economics":        ["academic_search", "web_search"],
    "sociology":        ["academic_search", "web_search"],
    "linguistics":      ["academic_search", "knowledge_base"],
    "law":              ["knowledge_base", "web_search"],
    "engineering":      ["academic_search", "knowledge_base"],
    "technology":       ["web_search", "knowledge_base"],
    "ai":               ["academic_search", "web_search"],
    "robotics":         ["academic_search", "knowledge_base"],
    "politics":         ["web_search", "knowledge_base"],
    "geography":        ["knowledge_base", "web_search"],
    "culture":          ["knowledge_base", "web_search"],
    "general":          ["web_search", "knowledge_base"],
}

_DEFAULT_TOOLS: list[str] = ["web_search", "knowledge_base"]


class DomainToolPlanner:
    """
    Deterministic planner: maps pipeline outputs \u2192 minimal ToolCallPlan.
    """

    def __init__(self, max_tools: int = 3) -> None:
        self._max_tools = max_tools

    def plan(
        self,
        query: str,
        domains: List[str],
        decision_type: str = "",
    ) -> List[ToolCall]:
        seen: dict[str, None] = {}

        for domain in domains:
            key = domain.lower().replace(" ", "_").replace("-", "_")
            for tool in _DOMAIN_TOOLS.get(key, _DEFAULT_TOOLS):
                seen[tool] = None

        if decision_type and "patch" in decision_type.lower():
            seen["web_search"] = None

        if not seen:
            for t in _DEFAULT_TOOLS:
                seen[t] = None

        selected = list(seen.keys())[: self._max_tools]
        return [ToolCall(tool=t, query=query) for t in selected]


_planner: DomainToolPlanner | None = None


def get_planner() -> DomainToolPlanner:
    global _planner
    if _planner is None:
        _planner = DomainToolPlanner()
    return _planner
