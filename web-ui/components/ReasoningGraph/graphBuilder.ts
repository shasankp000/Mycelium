// ---------------------------------------------------------------------------
// graphBuilder.ts — Pure, side-effect-free graph construction functions.
// All functions are deterministic given the same input.
// Plan refs: §2.3.1–2.3.3, §2.3.7–2.3.8, §4.3
// ---------------------------------------------------------------------------

import type { SseEvent, ReasoningMode } from '../../types/pipeline';
import {
  type GraphNode,
  type GraphEdge,
  type SubgraphZone,
  type NodeKind,
  NODE_COLORS,
  CLUSTER_THRESHOLDS,
  GRAPH_SCHEMA_VERSION,
  type GraphSnapshot,
  type GraphLifecycle,
  type LayoutMode,
} from '../../types/graph';

// ---------------------------------------------------------------------------
// SSE phase → graph zone mapping
// ---------------------------------------------------------------------------

const PHASE_ZONE: Record<string, SubgraphZone> = {
  setting_up:            'pipeline',
  environment_ready:     'pipeline',
  routing:               'pipeline',
  graph_routing:         'reasoning',
  graph_expert_init:     'reasoning',
  graph_coverage_report: 'reasoning',
  heartbeat:             'reasoning',
  expert_decision:       'reasoning',
  sandbox_plan:          'evidence',
  sandbox_summary:       'evidence',
  conversation:          'pipeline',
  done:                  'pipeline',
  error:                 'pipeline',
  graph_dfs_step:        'reasoning',
  graph_tool_start:      'evidence',
  graph_tool_done:       'evidence',
  graph_synthesis_start: 'reasoning',
  graph_cluster_expand:  'reasoning',
  graph_snapshot_saved:  'pipeline',
};

function zoneForPhase(phase: string): SubgraphZone {
  if (PHASE_ZONE[phase]) return PHASE_ZONE[phase];
  if (phase.startsWith('sandbox_tool/')) return 'evidence';
  return 'reasoning';
}

// ---------------------------------------------------------------------------
// SSE phase → NodeKind mapping
// ---------------------------------------------------------------------------

function kindForPhase(phase: string): NodeKind {
  if (phase === 'expert_decision')        return 'decision_point';
  if (phase === 'graph_synthesis_start')  return 'synthesis_node';
  if (phase.startsWith('sandbox_tool/') ||
      phase === 'sandbox_plan' ||
      phase === 'graph_tool_start' ||
      phase === 'graph_tool_done')        return 'tool_call';
  if (phase === 'graph_routing' ||
      phase === 'graph_expert_init' ||
      phase === 'graph_coverage_report' ||
      phase === 'graph_dfs_step')         return 'expert';
  if (phase === 'heartbeat')              return 'reasoning_step';
  if (phase === 'routing' ||
      phase === 'setting_up' ||
      phase === 'environment_ready' ||
      phase === 'conversation' ||
      phase === 'done' ||
      phase === 'error')                  return 'pipeline_stage';
  if (phase === 'sandbox_summary' ||
      phase === 'graph_snapshot_saved')   return 'evidence_node';
  return 'reasoning_step';
}

// ---------------------------------------------------------------------------
// Node size derivation from semantic weight (§2.3.8)
// ---------------------------------------------------------------------------

const BASE_NODE_SIZE = 6;

function nodeSizeFromWeight(kind: NodeKind, confidence?: number, leverage?: number): number {
  let size = BASE_NODE_SIZE;
  if (kind === 'contradiction_node') size = BASE_NODE_SIZE * 2.2;
  else if (kind === 'synthesis_node') size = BASE_NODE_SIZE * 1.7;
  else if (kind === 'decision_point') size = BASE_NODE_SIZE * 1.5;
  else if (kind === 'cluster_node')   size = BASE_NODE_SIZE * 1.8;
  if (leverage !== undefined) size *= 0.8 + leverage * 0.4;
  return Math.round(size * 10) / 10;
}

// ---------------------------------------------------------------------------
// Node opacity from confidence (§2.3.8)
// ---------------------------------------------------------------------------

function nodeOpacity(confidence?: number): number {
  if (confidence === undefined) return 0.85;
  // Low-confidence branches fade (floor at 0.3)
  return Math.max(0.3, 0.4 + confidence * 0.6);
}

// ---------------------------------------------------------------------------
// Edge opacity from weight (§9.3 — opacity preferred over thickness)
// ---------------------------------------------------------------------------

export function edgeOpacity(weight?: number): number {
  if (weight === undefined) return 0.5;
  return Math.max(0.15, 0.2 + weight * 0.75);
}

export function edgeThickness(weight?: number, highDetail = false): number {
  // Subtle thickness; never extreme (§9.3)
  if (!highDetail) return 1;
  if (weight === undefined) return 1;
  return Math.min(1 + weight * 1.5, 2.8);
}

// ---------------------------------------------------------------------------
// buildNodeFromSseEvent
// The main mapping function: SSE event → GraphNode.
// Uses sequenceNumber for deterministic ordering (§4.3).
// ---------------------------------------------------------------------------

export function buildNodeFromSseEvent(
  event: SseEvent,
  sequenceNumber: number,
): GraphNode {
  const phase     = (event.phase_name ?? event.phase ?? 'unknown') as string;
  const kind      = kindForPhase(phase);
  const zone      = zoneForPhase(phase);
  const state     = event.state === 'done' ? 'done'
                  : event.state === 'error' ? 'error'
                  : event.phase === 'done' ? 'done'
                  : event.phase === 'error' ? 'error'
                  : 'running';

  const label     = event.message ?? event.detail ?? phase;
  const metadata  = event.metadata ?? {};

  const confidence = typeof metadata.confidence === 'number'
    ? (metadata.confidence as number) : undefined;
  const leverage   = typeof metadata.leverage === 'number'
    ? (metadata.leverage as number) : undefined;
  const depth      = typeof metadata.depth === 'number'
    ? (metadata.depth as number) : 0;

  const id = event.event_id
    ?? `${phase}_${sequenceNumber}`;

  return {
    id,
    label:         label.slice(0, 80),  // cap label length
    kind,
    state,
    layerDepth:    depth,
    zone,
    metadata,
    confidence,
    leverage,
    color:         NODE_COLORS[kind],
    size:          nodeSizeFromWeight(kind, confidence, leverage),
    opacity:       nodeOpacity(confidence),
    sequenceNumber,
    eventId:       event.event_id,
    timestamp:     typeof event.timestamp === 'number' ? event.timestamp : Date.now(),
  };
}

// ---------------------------------------------------------------------------
// buildEdgeFromNodes
// Creates a directed edge between two adjacent nodes by sequence.
// ---------------------------------------------------------------------------

export function buildEdgeFromNodes(
  source: GraphNode,
  target: GraphNode,
  sequenceNumber: number,
  weight?: number,
  highDetail = false,
): GraphEdge {
  const w = weight ?? source.confidence;
  return {
    id:             `e_${source.id}_${target.id}`,
    source:         source.id,
    target:         target.id,
    weight:         w,
    opacity:        edgeOpacity(w),
    thickness:      edgeThickness(w, highDetail),
    sequenceNumber,
  };
}

// ---------------------------------------------------------------------------
// insertNodeDeterministic
// Inserts a node into a sorted-by-sequenceNumber list.
// Prevents duplicates (same id). If an existing node has the same id
// but a newer state, it updates in place.
// ---------------------------------------------------------------------------

export function insertNodeDeterministic(
  nodes: GraphNode[],
  incoming: GraphNode,
): GraphNode[] {
  const existingIdx = nodes.findIndex((n) => n.id === incoming.id);
  if (existingIdx !== -1) {
    // Update in-place — preserve position if already frozen
    const existing = nodes[existingIdx];
    const updated: GraphNode = {
      ...existing,
      ...incoming,
      fx: existing.fx,
      fy: existing.fy,
    };
    const result = [...nodes];
    result[existingIdx] = updated;
    return result;
  }
  // Insert in sequence order
  const insertAt = nodes.findIndex((n) => n.sequenceNumber > incoming.sequenceNumber);
  if (insertAt === -1) return [...nodes, incoming];
  const result = [...nodes];
  result.splice(insertAt, 0, incoming);
  return result;
}

// ---------------------------------------------------------------------------
// Clustering (§2.3.7)
// Collapses reasoning nodes into cluster nodes when thresholds are exceeded.
// Smart mode: summarize by default, expandable on click.
// ---------------------------------------------------------------------------

export interface ClusterGroup {
  clusterId: string;
  zone: SubgraphZone;
  nodes: GraphNode[];
}

function clusterIdForGroup(zone: SubgraphZone, domain: string): string {
  return `cluster_${zone}_${domain.replace(/\s+/g, '_').toLowerCase()}`;
}

/**
 * Builds a synthetic cluster node from a group of real nodes.
 * The cluster is expandable (expanded=false by default).
 */
export function buildClusterNode(group: ClusterGroup, sequenceNumber: number): GraphNode {
  const { clusterId, zone, nodes } = group;
  const confidences = nodes.map((n) => n.confidence).filter((c): c is number => c !== undefined);
  const avgConf = confidences.length > 0
    ? confidences.reduce((a, b) => a + b, 0) / confidences.length
    : undefined;
  const domains = nodes
    .map((n) => n.metadata?.primary_domain as string | undefined)
    .filter(Boolean) as string[];
  const dominantDomain = domains.length > 0
    ? domains.sort((a, b) =>
        domains.filter((d) => d === b).length - domains.filter((d) => d === a).length
      )[0]
    : undefined;
  const maxDepth = Math.max(...nodes.map((n) => n.layerDepth));
  const minSeq   = Math.min(...nodes.map((n) => n.sequenceNumber));

  return {
    id:                     clusterId,
    label:                  `${zone.charAt(0).toUpperCase() + zone.slice(1)} cluster (${nodes.length} nodes)`,
    kind:                   'cluster_node',
    state:                  'done',
    layerDepth:             maxDepth,
    zone,
    metadata:               { node_count: nodes.length, dominant_domain: dominantDomain },
    confidence:             avgConf,
    color:                  NODE_COLORS.cluster_node,
    size:                   nodeSizeFromWeight('cluster_node', avgConf),
    opacity:                nodeOpacity(avgConf),
    isCluster:              true,
    clusterNodeCount:       nodes.length,
    clusterAvgConfidence:   avgConf,
    clusterDominantDomain:  dominantDomain,
    clusterMaxDepth:        maxDepth,
    expanded:               false,
    sequenceNumber:         minSeq,
  };
}

/**
 * applySmartModeClustering
 * For Smart mode: collapse reasoning nodes into clusters.
 * Researcher mode: returns nodes unchanged (full DFS visible).
 * Fast mode: collapses both reasoning + evidence.
 */
export function applySmartModeClustering(
  nodes: GraphNode[],
  mode: ReasoningMode,
  isMobile = false,
): GraphNode[] {
  const threshold = isMobile ? CLUSTER_THRESHOLDS.mobileMaxNodes : Infinity;

  // Researcher: no clustering by default
  if (mode === 'researcher' && !isMobile) return nodes;

  const reasoningNodes = nodes.filter(
    (n) => n.zone === 'reasoning' &&
           !n.isCluster &&
           n.kind !== 'decision_point' &&
           n.kind !== 'synthesis_node',
  );
  const evidenceNodes  = nodes.filter(
    (n) => n.zone === 'evidence' && !n.isCluster,
  );
  const otherNodes     = nodes.filter(
    (n) => n.zone === 'pipeline' ||
           n.kind === 'decision_point' ||
           n.kind === 'synthesis_node' ||
           n.isCluster,
  );

  const shouldCollapseReasoning =
    mode === 'smart' && reasoningNodes.length >= CLUSTER_THRESHOLDS.reasoningCollapse ||
    mode === 'fast'  && reasoningNodes.length > 0 ||
    isMobile && reasoningNodes.length >= CLUSTER_THRESHOLDS.mobileMaxNodes;

  const shouldCollapseEvidence =
    mode === 'fast'  && evidenceNodes.length > 0 ||
    isMobile && evidenceNodes.length >= CLUSTER_THRESHOLDS.evidenceSummarise ||
    mode === 'smart' && evidenceNodes.length >= CLUSTER_THRESHOLDS.evidenceSummarise;

  // Group reasoning nodes by domain for cleaner cluster labels
  function groupByDomain(nodeList: GraphNode[], zone: SubgraphZone): ClusterGroup[] {
    const byDomain = new Map<string, GraphNode[]>();
    nodeList.forEach((n) => {
      const domain = (n.metadata?.primary_domain as string | undefined) ?? 'general';
      const existing = byDomain.get(domain) ?? [];
      byDomain.set(domain, [...existing, n]);
    });
    return Array.from(byDomain.entries()).map(([domain, domNodes]) => ({
      clusterId: clusterIdForGroup(zone, domain),
      zone,
      nodes: domNodes,
    }));
  }

  const collapsedReasoning: GraphNode[] = shouldCollapseReasoning
    ? groupByDomain(reasoningNodes, 'reasoning').map((g, i) =>
        buildClusterNode(g, i)
      )
    : reasoningNodes;

  const collapsedEvidence: GraphNode[] = shouldCollapseEvidence
    ? [buildClusterNode({ clusterId: 'cluster_evidence', zone: 'evidence', nodes: evidenceNodes }, evidenceNodes[0]?.sequenceNumber ?? 999)]
    : evidenceNodes;

  return [...otherNodes, ...collapsedReasoning, ...collapsedEvidence];
}

/**
 * expandCluster
 * Replaces a cluster node with its constituent nodes.
 * Called on click (§9.2).
 */
export function expandCluster(
  nodes: GraphNode[],
  clusterId: string,
  originalNodes: GraphNode[],  // the full unfiltered node list
): GraphNode[] {
  const clusterNode = nodes.find((n) => n.id === clusterId);
  if (!clusterNode) return nodes;
  // Find originals that belonged to this cluster
  const members = originalNodes.filter(
    (n) => !n.isCluster &&
           n.zone === clusterNode.zone &&
           n.kind !== 'decision_point' &&
           n.kind !== 'synthesis_node',
  );
  const withoutCluster = nodes.filter((n) => n.id !== clusterId);
  return [...withoutCluster, ...members].sort((a, b) => a.sequenceNumber - b.sequenceNumber);
}

// ---------------------------------------------------------------------------
// buildGraphSnapshot
// Creates a snapshot of the current graph state for persistence (§2.3.9).
// ---------------------------------------------------------------------------

export function buildGraphSnapshot(
  params: {
    snapshotId: string;
    traceId?: string;
    queryText?: string;
    reasoningMode: ReasoningMode;
    nodes: GraphNode[];
    edges: GraphEdge[];
    lifecycle: GraphLifecycle;
    layoutMode: LayoutMode;
    totalElapsedMs?: number;
    persistenceScope?: GraphSnapshot['persistenceScope'];
  }
): GraphSnapshot {
  return {
    snapshotId:         params.snapshotId,
    traceId:            params.traceId,
    queryText:          params.queryText,
    reasoningMode:      params.reasoningMode,
    graphSchemaVersion: GRAPH_SCHEMA_VERSION,
    nodes:              params.nodes,
    edges:              params.edges,
    lifecycle:          params.lifecycle,
    layoutMode:         params.layoutMode,
    createdAt:          Date.now(),
    nodeCount:          params.nodes.length,
    edgeCount:          params.edges.length,
    totalElapsedMs:     params.totalElapsedMs,
    persistenceScope:   params.persistenceScope ?? 'local',
  };
}
