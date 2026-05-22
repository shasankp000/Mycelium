// ---------------------------------------------------------------------------
// types/graph.ts — Phase 3 extension
// Ground-truth type definitions for the Reasoning Graph system.
// Plan refs: §2.3.1, §2.3.2, §2.3.7, §2.3.9, §4.5, §9.1
// ---------------------------------------------------------------------------

import type { ReasoningMode } from './pipeline';

// ---------------------------------------------------------------------------
// Node and Edge kinds
// ---------------------------------------------------------------------------

export type NodeKind =
  | 'pipeline_stage'
  | 'expert'
  | 'tool_call'
  | 'reasoning_step'
  | 'evidence_node'
  | 'decision_point'
  | 'cluster_node'
  | 'contradiction_node'
  | 'synthesis_node';

export type NodeState = 'running' | 'done' | 'error' | 'skipped';

export type SubgraphZone = 'pipeline' | 'reasoning' | 'evidence';

// ---------------------------------------------------------------------------
// GraphNode
// ---------------------------------------------------------------------------

export interface GraphNode {
  id: string;
  label: string;
  kind: NodeKind;
  state: NodeState;
  layerDepth: number;
  zone: SubgraphZone;
  metadata: Record<string, unknown>;

  // Semantic weighting (§2.3.8)
  confidence?: number;    // 0–1
  leverage?: number;      // relative influence score
  centrality?: number;    // graph centrality measure

  // Visual overrides
  color?: string;
  size?: number;
  opacity?: number;

  // Clustering (§2.3.7)
  clusterId?: string;
  isCluster?: boolean;
  clusterNodeCount?: number;
  clusterAvgConfidence?: number;
  clusterDominantDomain?: string;
  clusterMaxDepth?: number;
  expanded?: boolean;     // whether cluster is expanded

  // Ordering (§4.3 deterministic insertion)
  sequenceNumber: number;
  eventId?: string;
  timestamp?: number;     // ms epoch

  // Position hints for layout zones (§2.3.3)
  fx?: number | null;     // fixed x (set after physics freeze)
  fy?: number | null;     // fixed y
}

// ---------------------------------------------------------------------------
// GraphEdge
// ---------------------------------------------------------------------------

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  label?: string;

  // Semantic weighting (§9.3 — opacity preferred over thickness)
  weight?: number;        // 0–1 confidence/strength
  opacity?: number;       // derived from weight; thickness remains subtle
  thickness?: number;     // capped; never extreme

  // Ordering
  sequenceNumber: number;
}

// ---------------------------------------------------------------------------
// Node colour map (§2.3.2)
// ---------------------------------------------------------------------------

export const NODE_COLORS: Record<NodeKind, string> = {
  pipeline_stage:     '#6366f1',   // indigo
  expert:             '#10b981',   // emerald
  tool_call:          '#f59e0b',   // amber
  reasoning_step:     '#64748b',   // slate
  evidence_node:      '#3b82f6',   // blue
  decision_point:     '#ef4444',   // red
  cluster_node:       '#8b5cf6',   // violet
  contradiction_node: '#dc2626',   // bright red
  synthesis_node:     '#22c55e',   // green
};

// ---------------------------------------------------------------------------
// Clustering thresholds (§2.3.7)
// ---------------------------------------------------------------------------

export const CLUSTER_THRESHOLDS = {
  /** Node count above which reasoning nodes begin collapsing */
  reasoningCollapse:  20,
  /** Node count above which evidence trees summarise */
  evidenceSummarise:  15,
  /** Low-confidence threshold — branches below this aggregate */
  lowConfidence:      0.35,
  /** Max nodes before mobile graph simplification kicks in */
  mobileMaxNodes:     60,
} as const;

// ---------------------------------------------------------------------------
// Graph lifecycle / layout enums (§2.3.6)
// ---------------------------------------------------------------------------

export type GraphLifecycle =
  | 'idle'         // no query in progress
  | 'streaming'    // receiving SSE events; physics running
  | 'stabilising'  // done/error received; 2-3 s cooldown
  | 'frozen'       // physics paused; exploration mode
  | 'resumed';     // user clicked Resume simulation

export type LayoutMode = 'force' | 'hierarchy' | 'radial';

// ---------------------------------------------------------------------------
// GraphSnapshot (§2.3.9, §4.5, §9.1)
// ---------------------------------------------------------------------------

/** Semantic schema version. Increment when node/edge shape changes. */
export const GRAPH_SCHEMA_VERSION = 1;

export interface GraphSnapshot {
  // Identity
  snapshotId: string;
  traceId?: string;
  queryText?: string;
  reasoningMode: ReasoningMode;

  // Schema version for replay compatibility (§4.5)
  graphSchemaVersion: number;

  // Graph state
  nodes: GraphNode[];
  edges: GraphEdge[];

  // Lifecycle state at time of snapshot
  lifecycle: GraphLifecycle;
  layoutMode: LayoutMode;

  // Timestamps
  createdAt: number;       // ms epoch
  stabilisedAt?: number;   // when physics froze

  // Metadata
  nodeCount: number;
  edgeCount: number;
  totalElapsedMs?: number;

  // §9.1: frontend vs backend persistence intent
  // 'local'   — session-local acceleration layer only
  // 'remote'  — canonical historical archive
  // 'both'    — persisted in both layers
  persistenceScope: 'local' | 'remote' | 'both';
}

// ---------------------------------------------------------------------------
// Snapshot storage helpers
// ---------------------------------------------------------------------------

const SNAPSHOT_STORAGE_PREFIX = 'mycelium_graph_snapshot_';
const MAX_LOCAL_SNAPSHOTS = 20;

export function saveSnapshotLocal(snapshot: GraphSnapshot): void {
  if (typeof window === 'undefined') return;
  try {
    const key = `${SNAPSHOT_STORAGE_PREFIX}${snapshot.snapshotId}`;
    window.localStorage.setItem(key, JSON.stringify(snapshot));
    // Prune oldest if over limit
    const allKeys = Object.keys(window.localStorage)
      .filter((k) => k.startsWith(SNAPSHOT_STORAGE_PREFIX))
      .sort();
    if (allKeys.length > MAX_LOCAL_SNAPSHOTS) {
      allKeys.slice(0, allKeys.length - MAX_LOCAL_SNAPSHOTS).forEach((k) => {
        window.localStorage.removeItem(k);
      });
    }
  } catch { /* quota exceeded or SSR — safe to ignore */ }
}

export function loadSnapshotLocal(snapshotId: string): GraphSnapshot | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = window.localStorage.getItem(`${SNAPSHOT_STORAGE_PREFIX}${snapshotId}`);
    if (!raw) return null;
    return JSON.parse(raw) as GraphSnapshot;
  } catch { return null; }
}

export function listLocalSnapshots(): GraphSnapshot[] {
  if (typeof window === 'undefined') return [];
  try {
    return Object.keys(window.localStorage)
      .filter((k) => k.startsWith(SNAPSHOT_STORAGE_PREFIX))
      .map((k) => {
        try { return JSON.parse(window.localStorage.getItem(k) ?? '') as GraphSnapshot; }
        catch { return null; }
      })
      .filter(Boolean) as GraphSnapshot[];
  } catch { return []; }
}
