// ---------------------------------------------------------------------------
// Graph schema types — defined in Phase 1, consumed in Phase 3+
// Revision: Hardened scalability + runtime observability
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Node & edge primitives
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

export interface GraphNode {
  id: string;
  label: string;
  kind: NodeKind;
  state: 'running' | 'done' | 'error' | 'skipped';
  layerDepth: number;
  metadata: Record<string, unknown>;

  // Semantic weighting
  confidence?: number;
  leverage?: number;
  centrality?: number;

  // Visual
  color?: string;
  size?: number;
  opacity?: number;

  // Clustering
  clusterId?: string;

  // Internal: fixed position after stabilization
  fx?: number;
  fy?: number;

  // Sequence for deterministic insertion ordering (from backend)
  sequenceNumber?: number;
}

export interface GraphEdge {
  source: string;
  target: string;
  label?: string;
  weight?: number;
  opacity?: number;
}

// ---------------------------------------------------------------------------
// Node colour mapping (matches plan Section 2.3.2)
// ---------------------------------------------------------------------------

export const NODE_COLORS: Record<NodeKind, string> = {
  pipeline_stage:      '#6366f1', // indigo
  expert:              '#10b981', // emerald
  tool_call:           '#f59e0b', // amber
  reasoning_step:      '#64748b', // slate
  evidence_node:       '#3b82f6', // blue
  decision_point:      '#ef4444', // red
  cluster_node:        '#8b5cf6', // violet
  contradiction_node:  '#dc2626', // dark red
  synthesis_node:      '#22c55e', // green
};

// ---------------------------------------------------------------------------
// Subgraph zone identifiers
// ---------------------------------------------------------------------------

export type SubgraphZone = 'pipeline' | 'reasoning' | 'evidence';

// ---------------------------------------------------------------------------
// Cluster metadata
// ---------------------------------------------------------------------------

export interface ClusterMeta {
  clusterId: string;
  nodeCount: number;
  averageConfidence: number;
  dominantDomain: string;
  maxDepth: number;
  expanded: boolean;
}

// ---------------------------------------------------------------------------
// Graph snapshot — persisted after each query completion (plan Section 2.3.9)
// ---------------------------------------------------------------------------

export interface GraphSnapshot {
  snapshotId: string;
  graphSchemaVersion: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  clusters: ClusterMeta[];
  reasoningMode: 'fast' | 'smart' | 'researcher';
  stabilized: boolean;
  timestamps: {
    queryStart: number;
    queryEnd: number;
    stabilizedAt?: number;
  };
  metadata: {
    nodeCount: number;
    edgeCount: number;
    maxDepth: number;
  };
}

// Current schema version — increment on breaking changes
export const GRAPH_SCHEMA_VERSION = '1.0.0';

// ---------------------------------------------------------------------------
// Thresholds for clustering (plan Section 2.3.7)
// ---------------------------------------------------------------------------

export const CLUSTER_THRESHOLDS = {
  /** Collapse reasoning nodes into clusters above this count */
  reasoningNodes: 20,
  /** Collapse DFS branches above this depth */
  dfsDepth: 4,
  /** Confidence below which nodes are aggregated */
  lowConfidence: 0.3,
} as const;

// ---------------------------------------------------------------------------
// Graph lifecycle state (plan Section 2.3.6)
// ---------------------------------------------------------------------------

export type GraphLifecycleState =
  | 'idle'       // no query in progress
  | 'streaming'  // live SSE updates arriving, physics running
  | 'stabilizing'// physics winding down
  | 'frozen'     // physics paused, exploration mode
  | 'replaying'; // deterministic replay from snapshot

// ---------------------------------------------------------------------------
// Layout mode (plan Section 2.3.4)
// ---------------------------------------------------------------------------

export type LayoutMode = 'force' | 'hierarchy' | 'radial';
