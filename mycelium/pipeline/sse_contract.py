"""
sse_contract.py — Canonical SSE phase_name definitions.

This is the **single source of truth** for every ``phase_name`` string
emitted by ``run_workflow.py`` (and consumed by the web-UI).  Import
``PhaseNames`` here instead of writing bare string literals so that a
rename is a hard error at import time rather than a silent graph-rendering
bug on the frontend.

Keep this file in sync with  ``web-ui/types/sseContract.ts``.
When adding a new phase:
  1. Add a constant to ``PhaseNames``.
  2. Add an entry to ``PHASE_REGISTRY``.
  3. Mirror both additions in the TypeScript file.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Dict

# ---------------------------------------------------------------------------
# Zone / Kind type aliases (mirrors web-ui/types/graph.ts)
# ---------------------------------------------------------------------------

SubgraphZone = Literal["pipeline", "reasoning", "evidence"]
NodeKind     = Literal[
    "pipeline_stage",
    "reasoning_step",
    "expert",
    "decision_point",
    "synthesis_node",
    "tool_call",
    "evidence_node",
    "contradiction_node",
    "cluster_node",
]


@dataclass(frozen=True)
class _PhaseRegistry:
    zone:        SubgraphZone
    kind:        NodeKind
    description: str


# ---------------------------------------------------------------------------
# PhaseNames — import and use these constants everywhere in run_workflow.py
# ---------------------------------------------------------------------------

class PhaseNames:
    # ── startup / environment ─────────────────────────────────────────
    SETTING_UP          = "setting_up"
    ENVIRONMENT_READY   = "environment_ready"

    # ── expert system initialisation ──────────────────────────────────
    GRAPH_EXPERT_INIT   = "graph_expert_init"
    GRAPH_SPECTRAL_SYNC = "graph_spectral_sync"
    GRAPH_ROUTER_READY  = "graph_router_ready"

    # ── Layer 0 classification ─────────────────────────────────────────
    GRAPH_LAYER0        = "graph_layer0"

    # ── routing ────────────────────────────────────────────────────────
    ROUTING             = "routing"
    GRAPH_ROUTING       = "graph_routing"

    # ── TRM / DFS ──────────────────────────────────────────────────────
    GRAPH_DFS_STEP      = "graph_dfs_step"
    GRAPH_TRM_DECISION  = "graph_trm_decision"

    # ── shadow domain ─────────────────────────────────────────────────
    PROMOTE_SHADOW_DOMAIN = "promote_shadow_domain"

    # ── phase 2 / 3 tools ─────────────────────────────────────────────
    GRAPH_TOOL_START    = "graph_tool_start"
    GRAPH_TOOL_DONE     = "graph_tool_done"

    # ── synthesis / expert decision ────────────────────────────────────
    GRAPH_SYNTHESIS_START = "graph_synthesis_start"
    EXPERT_DECISION       = "expert_decision"

    # ── terminal ──────────────────────────────────────────────────────
    DONE  = "done"
    ERROR = "error"


# ---------------------------------------------------------------------------
# PHASE_REGISTRY — metadata for each phase (used for validation / docs)
# ---------------------------------------------------------------------------

PHASE_REGISTRY: Dict[str, _PhaseRegistry] = {
    PhaseNames.SETTING_UP: _PhaseRegistry(
        zone="pipeline",
        kind="pipeline_stage",
        description="Pre-flight: model registry warm-up",
    ),
    PhaseNames.ENVIRONMENT_READY: _PhaseRegistry(
        zone="pipeline",
        kind="pipeline_stage",
        description="Model registry warm — N model(s) resident",
    ),
    PhaseNames.GRAPH_EXPERT_INIT: _PhaseRegistry(
        zone="reasoning",
        kind="expert",
        description="Unified expert system initialisation (K-Medoids + Calibration + OOD)",
    ),
    PhaseNames.GRAPH_SPECTRAL_SYNC: _PhaseRegistry(
        zone="reasoning",
        kind="expert",
        description="Spectral signatures synced with registered expert domains",
    ),
    PhaseNames.GRAPH_ROUTER_READY: _PhaseRegistry(
        zone="reasoning",
        kind="expert",
        description="MultiLens router + expert filter + Layer0 router initialised",
    ),
    PhaseNames.GRAPH_LAYER0: _PhaseRegistry(
        zone="pipeline",
        kind="pipeline_stage",
        description="Layer 0 classification — route or pass to reasoning pipeline",
    ),
    PhaseNames.ROUTING: _PhaseRegistry(
        zone="pipeline",
        kind="pipeline_stage",
        description="Multi-lens routing + TRM lens refinement",
    ),
    PhaseNames.GRAPH_ROUTING: _PhaseRegistry(
        zone="reasoning",
        kind="expert",
        description="Routing complete — selected domains + classification",
    ),
    PhaseNames.GRAPH_DFS_STEP: _PhaseRegistry(
        zone="reasoning",
        kind="expert",
        description="DFS domain exploration step",
    ),
    PhaseNames.GRAPH_TRM_DECISION: _PhaseRegistry(
        zone="reasoning",
        kind="reasoning_step",
        description="TRM routing decision — halt_confidence vs threshold",
    ),
    PhaseNames.PROMOTE_SHADOW_DOMAIN: _PhaseRegistry(
        zone="reasoning",
        kind="reasoning_step",
        description="Shadow domain promoted to full expert",
    ),
    PhaseNames.GRAPH_TOOL_START: _PhaseRegistry(
        zone="evidence",
        kind="tool_call",
        description="Phase 2/3 tool invocation started",
    ),
    PhaseNames.GRAPH_TOOL_DONE: _PhaseRegistry(
        zone="evidence",
        kind="tool_call",
        description="Phase 2/3 tool invocation completed",
    ),
    PhaseNames.GRAPH_SYNTHESIS_START: _PhaseRegistry(
        zone="reasoning",
        kind="synthesis_node",
        description="Expert synthesis arbitration starting",
    ),
    PhaseNames.EXPERT_DECISION: _PhaseRegistry(
        zone="reasoning",
        kind="decision_point",
        description="Final expert decision emitted",
    ),
    PhaseNames.DONE: _PhaseRegistry(
        zone="pipeline",
        kind="pipeline_stage",
        description="Workflow completed successfully",
    ),
    PhaseNames.ERROR: _PhaseRegistry(
        zone="pipeline",
        kind="pipeline_stage",
        description="Workflow terminated with error",
    ),
}

# Convenience set of all valid phase name strings — useful for validation
ALL_PHASE_NAMES: frozenset = frozenset(PHASE_REGISTRY.keys())
