// ---------------------------------------------------------------------------
// types/graph.ts — Phase 5 extension
// Adds: GraphDiff type and diffSnapshots() pure function (§9.5).
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

  confidence?: number;
  leverage?: number;
  centrality?: number;

  color?: string;
  size?: number;
  opacity?: number;

  clusterId?: string;
  isCluster?: boolean;
  clusterNodeCount?: number;
  clusterAvgConfidence?: number;
  clusterDominantDomain?: string;
  clusterMaxDepth?: number;
  expanded?: boolean;

  sequenceNumber: number;
  eventId?: string;
  timestamp?: number;

  fx?: number | null;
  fy?: number | null;
}

// ---------------------------------------------------------------------------
// GraphEdge
// ---------------------------------------------------------------------------

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  label?: string;

  weight?: number;
  opacity?: number;
  thickness?: number;

  sequenceNumber: number;
}

// ---------------------------------------------------------------------------
// Node colour map
// ---------------------------------------------------------------------------

export const NODE_COLORS: Record<NodeKind, string> = {
  pipeline_stage:     '#6366f1',
  expert:             '#10b981',
  tool_call:          '#f59e0b',
  reasoning_step:     '#64748b',
  evidence_node:      '#3b82f6',
  decision_point:     '#ef4444',
  cluster_node:       '#8b5cf6',
  contradiction_node: '#dc2626',
  synthesis_node:     '#22c55e',
};

// ---------------------------------------------------------------------------
// Clustering thresholds
// ---------------------------------------------------------------------------

export const CLUSTER_THRESHOLDS = {
  reasoningCollapse:  20,
  evidenceSummarise:  15,
  lowConfidence:      0.35,
  mobileMaxNodes:     60,
} as const;

// ---------------------------------------------------------------------------
// Graph lifecycle / layout enums
// ---------------------------------------------------------------------------

export type GraphLifecycle =
  | 'idle'
  | 'streaming'
  | 'stabilising'
  | 'frozen'
  | 'resumed';

export type LayoutMode = 'force' | 'hierarchy' | 'radial';

// ---------------------------------------------------------------------------
// GraphSnapshot
// ---------------------------------------------------------------------------

export const GRAPH_SCHEMA_VERSION = 1;

export interface GraphSnapshot {
  snapshotId: string;
  traceId?: string;
  queryText?: string;
  reasoningMode: ReasoningMode;
  graphSchemaVersion: number;
  nodes: GraphNode[];
  edges: GraphEdge[];
  lifecycle: GraphLifecycle;
  layoutMode: LayoutMode;
  createdAt: number;
  stabilisedAt?: number;
  nodeCount: number;
  edgeCount: number;
  totalElapsedMs?: number;
  persistenceScope: 'local' | 'remote' | 'both';
}

// ---------------------------------------------------------------------------
// §9.5 GraphDiff — snapshot comparison result
// ---------------------------------------------------------------------------

/** Change type for a node or edge in a diff */
export type DiffChangeType = 'added' | 'removed' | 'changed' | 'unchanged';

export interface NodeDiffEntry {
  changeType: DiffChangeType;
  /** Present for 'added', 'unchanged', 'changed' — the node in snapshot B */
  nodeB?: GraphNode;
  /** Present for 'removed', 'unchanged', 'changed' — the node in snapshot A */
  nodeA?: GraphNode;
  /** Human-readable summary of what changed (for 'changed' entries) */
  changeSummary?: string;
}

export interface EdgeDiffEntry {
  changeType: DiffChangeType;
  edgeB?: GraphEdge;
  edgeA?: GraphEdge;
}

export interface GraphDiff {
  /** Snapshot A = "before" / "base" */
  snapshotIdA: string;
  /** Snapshot B = "after" / "new" */
  snapshotIdB: string;

  nodeDiffs: NodeDiffEntry[];
  edgeDiffs: EdgeDiffEntry[];

  /** Summary counts */
  nodesAdded:    number;
  nodesRemoved:  number;
  nodesChanged:  number;
  nodesUnchanged: number;

  edgesAdded:    number;
  edgesRemoved:  number;
  edgesChanged:  number;

  /** Dominant zone that changed most */
  dominantChangeZone?: SubgraphZone;
}

// ---------------------------------------------------------------------------
// diffSnapshots() — pure function, no side effects
// ---------------------------------------------------------------------------

function nodeChangeSummary(a: GraphNode, b: GraphNode): string {
  const parts: string[] = [];
  if (a.state !== b.state)       parts.push(`state ${a.state}→${b.state}`);
  if (a.kind  !== b.kind)        parts.push(`kind ${a.kind}→${b.kind}`);
  if (a.zone  !== b.zone)        parts.push(`zone ${a.zone}→${b.zone}`);
  if (
    a.confidence !== undefined &&
    b.confidence !== undefined &&
    Math.abs(a.confidence - b.confidence) > 0.05
  ) {
    parts.push(`conf ${a.confidence.toFixed(2)}→${b.confidence.toFixed(2)}`);
  }
  return parts.join(', ');
}

export function diffSnapshots(a: GraphSnapshot, b: GraphSnapshot): GraphDiff {
  const nodeMapA = new Map(a.nodes.map((n) => [n.id, n]));
  const nodeMapB = new Map(b.nodes.map((n) => [n.id, n]));
  const edgeMapA = new Map(a.edges.map((e) => [e.id, e]));
  const edgeMapB = new Map(b.edges.map((e) => [e.id, e]));

  const nodeDiffs: NodeDiffEntry[] = [];
  const edgeDiffs: EdgeDiffEntry[] = [];

  // Nodes in A
  for (const [id, nodeA] of nodeMapA) {
    const nodeB = nodeMapB.get(id);
    if (!nodeB) {
      nodeDiffs.push({ changeType: 'removed', nodeA });
    } else {
      const summary = nodeChangeSummary(nodeA, nodeB);
      if (summary) {
        nodeDiffs.push({ changeType: 'changed', nodeA, nodeB, changeSummary: summary });
      } else {
        nodeDiffs.push({ changeType: 'unchanged', nodeA, nodeB });
      }
    }
  }

  // Nodes only in B
  for (const [id, nodeB] of nodeMapB) {
    if (!nodeMapA.has(id)) {
      nodeDiffs.push({ changeType: 'added', nodeB });
    }
  }

  // Edges in A
  for (const [id, edgeA] of edgeMapA) {
    const edgeB = edgeMapB.get(id);
    if (!edgeB) {
      edgeDiffs.push({ changeType: 'removed', edgeA });
    } else {
      const weightChanged =
        edgeA.weight !== undefined &&
        edgeB.weight !== undefined &&
        Math.abs(edgeA.weight - edgeB.weight) > 0.05;
      edgeDiffs.push({ changeType: weightChanged ? 'changed' : 'unchanged', edgeA, edgeB });
    }
  }

  // Edges only in B
  for (const [id, edgeB] of edgeMapB) {
    if (!edgeMapA.has(id)) {
      edgeDiffs.push({ changeType: 'added', edgeB });
    }
  }

  // Summary counts
  const nodesAdded    = nodeDiffs.filter((d) => d.changeType === 'added').length;
  const nodesRemoved  = nodeDiffs.filter((d) => d.changeType === 'removed').length;
  const nodesChanged  = nodeDiffs.filter((d) => d.changeType === 'changed').length;
  const nodesUnchanged = nodeDiffs.filter((d) => d.changeType === 'unchanged').length;
  const edgesAdded    = edgeDiffs.filter((d) => d.changeType === 'added').length;
  const edgesRemoved  = edgeDiffs.filter((d) => d.changeType === 'removed').length;
  const edgesChanged  = edgeDiffs.filter((d) => d.changeType === 'changed').length;

  // Dominant change zone: which zone had the most non-unchanged nodes?
  const zoneCounts = new Map<SubgraphZone, number>();
  nodeDiffs
    .filter((d) => d.changeType !== 'unchanged')
    .forEach((d) => {
      const zone = (d.nodeB ?? d.nodeA)?.zone;
      if (zone) zoneCounts.set(zone, (zoneCounts.get(zone) ?? 0) + 1);
    });
  let dominantChangeZone: SubgraphZone | undefined;
  let maxZoneCount = 0;
  for (const [zone, count] of zoneCounts) {
    if (count > maxZoneCount) { maxZoneCount = count; dominantChangeZone = zone; }
  }

  return {
    snapshotIdA: a.snapshotId,
    snapshotIdB: b.snapshotId,
    nodeDiffs,
    edgeDiffs,
    nodesAdded,
    nodesRemoved,
    nodesChanged,
    nodesUnchanged,
    edgesAdded,
    edgesRemoved,
    edgesChanged,
    dominantChangeZone,
  };
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
    const allKeys = Object.keys(window.localStorage)
      .filter((k) => k.startsWith(SNAPSHOT_STORAGE_PREFIX))
      .sort();
    if (allKeys.length > MAX_LOCAL_SNAPSHOTS) {
      allKeys.slice(0, allKeys.length - MAX_LOCAL_SNAPSHOTS).forEach((k) => {
        window.localStorage.removeItem(k);
      });
    }
  } catch { /* quota exceeded or SSR */ }
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
