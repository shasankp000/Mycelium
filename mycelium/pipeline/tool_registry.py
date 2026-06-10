"""
tool_registry.py
================
Single source-of-truth for every tool Mycelium can invoke.

All other modules (mcp_tools_server, domain_tool_planner, and the
future TRM v2 TRMToolPlanner) import from here rather than
hard-coding tool names or descriptions.

Adding a new tool means:
  1. Add a ToolSpec entry to TOOL_REGISTRY below.
  2. Implement the function in mcp_tools_server.py and register it
     in _TOOL_FN.
  3. Optionally update PREDICATE_TYPE_TOOLS and DOMAIN_TOOLS in
     domain_tool_planner.py.

No LLM is invoked anywhere in this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# ToolSpec — metadata record for a single tool
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolSpec:
    name: str
    """Canonical tool name.  Must match the key in mcp_tools_server._TOOL_FN."""

    description: str
    """One-line description surfaced in list_tools() and planner reasoning."""

    input_arg: str
    """Name of the primary input argument ('query' or 'expression')."""

    predicate_types: List[str] = field(default_factory=list)
    """
    PredicateType values this tool is best suited for.
    Empty list means the tool is domain-general.
    Values must match PredicateType enum strings in predicate_types.py:
      FACTIVE | COMPARATIVE | CAUSAL | EXISTENTIAL |
      TEMPORAL | MODAL | NORMATIVE | DEFINITIONAL
    """

    requires_trm_v2: bool = False
    """
    If True, this tool is a stub until TRM v2 / SQLite shards land.
    The planner will only emit it when USE_DOMAIN_STORE_STUB=True in config.
    """

    external: bool = True
    """False for tools that are resolved entirely inside the process."""

    tags: List[str] = field(default_factory=list)
    """Free-form tags used by TRM v2 gate to rank tools."""


# ---------------------------------------------------------------------------
# Registry — ordered; earlier entries are preferred when max_tools is tight
# ---------------------------------------------------------------------------

TOOL_REGISTRY: List[ToolSpec] = [
    ToolSpec(
        name="web_search",
        description=(
            "General web search via DuckDuckGo. "
            "Use for recent news, current facts, or broad factual claims."
        ),
        input_arg="query",
        predicate_types=["FACTIVE", "EXISTENTIAL", "MODAL"],
        tags=["general", "current"],
    ),
    ToolSpec(
        name="academic_search",
        description=(
            "Peer-reviewed paper search via Semantic Scholar. "
            "Use for scientific, medical, or technical claims."
        ),
        input_arg="query",
        predicate_types=["FACTIVE", "CAUSAL", "COMPARATIVE", "EXISTENTIAL"],
        tags=["scientific", "peer-reviewed"],
    ),
    ToolSpec(
        name="knowledge_base",
        description=(
            "Structured entity facts via Wikidata. "
            "Use for entity definitions, taxonomy, and factual attributes."
        ),
        input_arg="query",
        predicate_types=["FACTIVE", "DEFINITIONAL", "EXISTENTIAL"],
        tags=["structured", "entities"],
    ),
    ToolSpec(
        name="calculator",
        description=(
            "Safe arithmetic evaluator. "
            "Use for numeric expressions, unit conversions, and mathematical computations only."
        ),
        input_arg="expression",
        predicate_types=["COMPARATIVE"],
        external=False,
        tags=["math", "numeric"],
    ),
    ToolSpec(
        name="temporal_search",
        description=(
            "Date-scoped fact retrieval combining Wikidata date properties and "
            "a DuckDuckGo web search narrowed by year range. "
            "Use for TEMPORAL predicates: claims anchored to a specific time, "
            "era, or sequence of events."
        ),
        input_arg="query",
        predicate_types=["TEMPORAL"],
        tags=["temporal", "date-scoped"],
    ),
    ToolSpec(
        name="causal_search",
        description=(
            "Causal-chain evidence retrieval. Queries Semantic Scholar with "
            "causal-framing keywords, then fetches an Open Citations provenance "
            "chain for the top paper. "
            "Use for CAUSAL predicates: 'X causes Y', 'X leads to Y'."
        ),
        input_arg="query",
        predicate_types=["CAUSAL"],
        tags=["causal", "provenance"],
    ),
    ToolSpec(
        name="statistical_search",
        description=(
            "Structured statistics and data retrieval from World Bank Open Data "
            "and OECD Stats APIs. Falls back to a DuckDuckGo snippet search "
            "for series not available in those catalogues. "
            "Use for COMPARATIVE predicates with numeric backing: per-capita rates, "
            "risk ratios, percentages."
        ),
        input_arg="query",
        predicate_types=["COMPARATIVE"],
        tags=["statistics", "numeric", "comparative"],
    ),
    ToolSpec(
        name="domain_store_search",
        description=(
            "Internal domain-specific retrieval from the local SQLite expert shards. "
            "Returns pre-indexed, domain-curated chunks ranked by FTS5 + cosine similarity. "
            "Preferred over web_search when the domain shard is available and HOT. "
            "STUB until TRM v2 SQLite shards are built."
        ),
        input_arg="query",
        predicate_types=["FACTIVE", "DEFINITIONAL", "CAUSAL", "TEMPORAL",
                         "COMPARATIVE", "EXISTENTIAL"],
        requires_trm_v2=True,
        external=False,
        tags=["internal", "domain-shard", "trm-v2"],
    ),
]

# Fast lookup by name
TOOL_BY_NAME: Dict[str, ToolSpec] = {t.name: t for t in TOOL_REGISTRY}

# All live (non-stub) tool names in registry order
LIVE_TOOL_NAMES: List[str] = [
    t.name for t in TOOL_REGISTRY if not t.requires_trm_v2
]

# All stub tool names (needs TRM v2)
STUB_TOOL_NAMES: List[str] = [
    t.name for t in TOOL_REGISTRY if t.requires_trm_v2
]


def tools_for_predicate_type(predicate_type: str) -> List[ToolSpec]:
    """Return all live ToolSpecs that cover the given predicate type."""
    return [
        t for t in TOOL_REGISTRY
        if predicate_type in t.predicate_types and not t.requires_trm_v2
    ]


def all_tool_names(include_stubs: bool = False) -> List[str]:
    """Return ordered list of tool names."""
    return [
        t.name for t in TOOL_REGISTRY
        if include_stubs or not t.requires_trm_v2
    ]
