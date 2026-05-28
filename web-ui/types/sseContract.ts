/**
 * sseContract.ts — Canonical SSE phase_name definitions (TypeScript mirror).
 *
 * This file is the frontend counterpart of
 * ``mycelium/pipeline/sse_contract.py``.
 *
 * Import PHASE_NAMES here instead of writing bare string literals so that
 * a rename on the backend produces a TypeScript type error rather than a
 * silent graph-rendering bug.
 *
 * When adding a new phase:
 *   1. Add a key+value to PHASE_NAMES below.
 *   2. Add the matching entry to SSE_PHASE_REGISTRY.
 *   3. Mirror both additions in the Python file.
 *
 * Keep in sync with  mycelium/pipeline/sse_contract.py
 */

import type { SubgraphZone, NodeKind } from './graph';

// ---------------------------------------------------------------------------
// PHASE_NAMES — use these everywhere instead of raw string literals
// ---------------------------------------------------------------------------

export const PHASE_NAMES = {
  // startup / environment
  SETTING_UP:            'setting_up',
  ENVIRONMENT_READY:     'environment_ready',

  // expert system initialisation
  GRAPH_EXPERT_INIT:     'graph_expert_init',
  GRAPH_SPECTRAL_SYNC:   'graph_spectral_sync',
  GRAPH_ROUTER_READY:    'graph_router_ready',

  // Layer 0 classification
  GRAPH_LAYER0:          'graph_layer0',

  // routing
  ROUTING:               'routing',
  GRAPH_ROUTING:         'graph_routing',

  // TRM / DFS
  GRAPH_DFS_STEP:        'graph_dfs_step',
  GRAPH_TRM_DECISION:    'graph_trm_decision',

  // shadow domain
  PROMOTE_SHADOW_DOMAIN: 'promote_shadow_domain',

  // phase 2 / 3 tools
  GRAPH_TOOL_START:      'graph_tool_start',
  GRAPH_TOOL_DONE:       'graph_tool_done',

  // Phase D — predicate extraction + evidence DST + contradiction
  PREDICATE_EXTRACTION:      'predicate_extraction',
  EVIDENCE_DST_FUSION:       'evidence_dst_fusion',
  CONTRADICTION_INTEGRATION: 'contradiction_integration',
  EVIDENCE_DST_DONE:         'evidence_dst_done',

  // synthesis / expert decision
  GRAPH_SYNTHESIS_START: 'graph_synthesis_start',
  EXPERT_DECISION:       'expert_decision',

  // terminal
  DONE:                  'done',
  ERROR:                 'error',
} as const;

/** Union of all valid phase name strings */
export type PhaseName = (typeof PHASE_NAMES)[keyof typeof PHASE_NAMES];

// ---------------------------------------------------------------------------
// SSE_PHASE_REGISTRY — zone + kind metadata for every phase
// ---------------------------------------------------------------------------

export interface PhaseRegistryEntry {
  zone:        SubgraphZone;
  kind:        NodeKind;
  description: string;
}

export const SSE_PHASE_REGISTRY: Record<PhaseName, PhaseRegistryEntry> = {
  [PHASE_NAMES.SETTING_UP]: {
    zone: 'pipeline',
    kind: 'pipeline_stage',
    description: 'Pre-flight: model registry warm-up',
  },
  [PHASE_NAMES.ENVIRONMENT_READY]: {
    zone: 'pipeline',
    kind: 'pipeline_stage',
    description: 'Model registry warm — N model(s) resident',
  },
  [PHASE_NAMES.GRAPH_EXPERT_INIT]: {
    zone: 'reasoning',
    kind: 'expert',
    description: 'Unified expert system initialisation (K-Medoids + Calibration + OOD)',
  },
  [PHASE_NAMES.GRAPH_SPECTRAL_SYNC]: {
    zone: 'reasoning',
    kind: 'expert',
    description: 'Spectral signatures synced with registered expert domains',
  },
  [PHASE_NAMES.GRAPH_ROUTER_READY]: {
    zone: 'reasoning',
    kind: 'expert',
    description: 'MultiLens router + expert filter + Layer0 router initialised',
  },
  [PHASE_NAMES.GRAPH_LAYER0]: {
    zone: 'pipeline',
    kind: 'pipeline_stage',
    description: 'Layer 0 classification — route or pass to reasoning pipeline',
  },
  [PHASE_NAMES.ROUTING]: {
    zone: 'pipeline',
    kind: 'pipeline_stage',
    description: 'Multi-lens routing + TRM lens refinement',
  },
  [PHASE_NAMES.GRAPH_ROUTING]: {
    zone: 'reasoning',
    kind: 'expert',
    description: 'Routing complete — selected domains + classification',
  },
  [PHASE_NAMES.GRAPH_DFS_STEP]: {
    zone: 'reasoning',
    kind: 'expert',
    description: 'DFS domain exploration step',
  },
  [PHASE_NAMES.GRAPH_TRM_DECISION]: {
    zone: 'reasoning',
    kind: 'reasoning_step',
    description: 'TRM routing decision — halt_confidence vs threshold',
  },
  [PHASE_NAMES.PROMOTE_SHADOW_DOMAIN]: {
    zone: 'reasoning',
    kind: 'reasoning_step',
    description: 'Shadow domain promoted to full expert',
  },
  [PHASE_NAMES.GRAPH_TOOL_START]: {
    zone: 'evidence',
    kind: 'tool_call',
    description: 'Phase 2/3 tool invocation started',
  },
  [PHASE_NAMES.GRAPH_TOOL_DONE]: {
    zone: 'evidence',
    kind: 'tool_call',
    description: 'Phase 2/3 tool invocation completed',
  },
  // ── Phase D ──────────────────────────────────────────────────────────
  [PHASE_NAMES.PREDICATE_EXTRACTION]: {
    zone: 'evidence',
    kind: 'evidence_node',
    description: 'Phase D — predicate extraction from sentence',
  },
  [PHASE_NAMES.EVIDENCE_DST_FUSION]: {
    zone: 'evidence',
    kind: 'evidence_node',
    description: 'Phase D — DSTFusion over structured ScoredBundles',
  },
  [PHASE_NAMES.CONTRADICTION_INTEGRATION]: {
    zone: 'evidence',
    kind: 'contradiction_node',
    description: 'Phase D — contradiction classification over predicate pairs',
  },
  [PHASE_NAMES.EVIDENCE_DST_DONE]: {
    zone: 'evidence',
    kind: 'evidence_node',
    description: 'Phase D — EvidenceDSTResult complete',
  },
  // ── synthesis / decision ─────────────────────────────────────────────
  [PHASE_NAMES.GRAPH_SYNTHESIS_START]: {
    zone: 'reasoning',
    kind: 'synthesis_node',
    description: 'Expert synthesis arbitration starting',
  },
  [PHASE_NAMES.EXPERT_DECISION]: {
    zone: 'reasoning',
    kind: 'decision_point',
    description: 'Final expert decision emitted',
  },
  [PHASE_NAMES.DONE]: {
    zone: 'pipeline',
    kind: 'pipeline_stage',
    description: 'Workflow completed successfully',
  },
  [PHASE_NAMES.ERROR]: {
    zone: 'pipeline',
    kind: 'pipeline_stage',
    description: 'Workflow terminated with error',
  },
};

// ---------------------------------------------------------------------------
// Helpers consumed by graphBuilder.ts
// ---------------------------------------------------------------------------

/** Returns the SubgraphZone for a given phase name string. Falls back to
 *  'reasoning' for unknown phases (same behaviour as the old PHASE_ZONE map)
 *  but now at least logs a warning in development. */
export function zoneForPhaseName(phase: string): SubgraphZone {
  const entry = SSE_PHASE_REGISTRY[phase as PhaseName];
  if (!entry) {
    if (process.env.NODE_ENV !== 'production') {
      console.warn(`[sseContract] Unknown phase_name: "${phase}" — defaulting zone to 'reasoning'`);
    }
    if (phase.startsWith('sandbox_tool/')) return 'evidence';
    return 'reasoning';
  }
  return entry.zone;
}

/** Returns the NodeKind for a given phase name string. Falls back to
 *  'reasoning_step' for unknown phases. */
export function kindForPhaseName(phase: string): NodeKind {
  const entry = SSE_PHASE_REGISTRY[phase as PhaseName];
  if (!entry) {
    if (phase.startsWith('sandbox_tool/')) return 'tool_call';
    return 'reasoning_step';
  }
  return entry.kind;
}
