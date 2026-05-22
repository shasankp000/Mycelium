// ---------------------------------------------------------------------------
// __tests__/graphBuilder.test.ts — Phase 5 unit tests
// Covers core pure functions in components/ReasoningGraph/graphBuilder.ts
// and the new diffSnapshots() in types/graph.ts.
//
// Run with: cd web-ui && npx jest --testPathPattern=graphBuilder
// ---------------------------------------------------------------------------

import {
  buildNodeFromSseEvent,
  buildEdgeFromNodes,
  insertNodeDeterministic,
  applySmartModeClustering,
  expandCluster,
  buildGraphSnapshot,
} from '../components/ReasoningGraph/graphBuilder';

import {
  diffSnapshots,
  GRAPH_SCHEMA_VERSION,
  type GraphNode,
  type GraphSnapshot,
} from '../types/graph';

import type { SseEvent } from '../types/pipeline';

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

function makeSseEvent(overrides: Partial<SseEvent> = {}): SseEvent {
  return {
    phase:      'routing',
    phase_name: 'routing',
    detail:     'test detail',
    elapsed_ms: 100,
    state:      'running',
    ...overrides,
  } as SseEvent;
}

function makeNode(overrides: Partial<GraphNode> = {}): GraphNode {
  return {
    id:             overrides.id ?? 'node_1',
    label:          overrides.label ?? 'Test node',
    kind:           overrides.kind ?? 'pipeline_stage',
    state:          overrides.state ?? 'done',
    layerDepth:     overrides.layerDepth ?? 0,
    zone:           overrides.zone ?? 'pipeline',
    metadata:       overrides.metadata ?? {},
    sequenceNumber: overrides.sequenceNumber ?? 1,
    ...overrides,
  };
}

function makeSnapshot(
  id: string,
  nodes: GraphNode[],
  overrides: Partial<GraphSnapshot> = {},
): GraphSnapshot {
  return {
    snapshotId:         id,
    reasoningMode:      'smart',
    graphSchemaVersion: GRAPH_SCHEMA_VERSION,
    nodes,
    edges:              [],
    lifecycle:          'frozen',
    layoutMode:         'force',
    createdAt:          Date.now(),
    nodeCount:          nodes.length,
    edgeCount:          0,
    persistenceScope:   'local',
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// buildNodeFromSseEvent
// ---------------------------------------------------------------------------

describe('buildNodeFromSseEvent', () => {
  it('maps routing phase to pipeline_stage kind', () => {
    const node = buildNodeFromSseEvent(makeSseEvent({ phase: 'routing' }), 1);
    expect(node.kind).toBe('pipeline_stage');
    expect(node.zone).toBe('pipeline');
  });

  it('maps expert_decision to decision_point kind', () => {
    const node = buildNodeFromSseEvent(makeSseEvent({ phase: 'expert_decision' }), 2);
    expect(node.kind).toBe('decision_point');
    expect(node.zone).toBe('reasoning');
  });

  it('maps sandbox_plan to tool_call in evidence zone', () => {
    const node = buildNodeFromSseEvent(makeSseEvent({ phase: 'sandbox_plan' }), 3);
    expect(node.kind).toBe('tool_call');
    expect(node.zone).toBe('evidence');
  });

  it('maps sandbox_tool/* to tool_call in evidence zone', () => {
    const node = buildNodeFromSseEvent(makeSseEvent({ phase: 'sandbox_tool/1_search' }), 4);
    expect(node.kind).toBe('tool_call');
    expect(node.zone).toBe('evidence');
  });

  it('maps graph_routing to expert kind in reasoning zone', () => {
    const node = buildNodeFromSseEvent(makeSseEvent({ phase: 'graph_routing' }), 5);
    expect(node.kind).toBe('expert');
    expect(node.zone).toBe('reasoning');
  });

  it('uses event_id as node id when present', () => {
    const node = buildNodeFromSseEvent(makeSseEvent({ event_id: 'evt_abc123' }), 6);
    expect(node.id).toBe('evt_abc123');
  });

  it('falls back to phase_seqNum id when event_id absent', () => {
    const node = buildNodeFromSseEvent(makeSseEvent({ phase: 'heartbeat', event_id: undefined }), 7);
    expect(node.id).toBe('heartbeat_7');
  });

  it('caps label at 80 characters', () => {
    const longLabel = 'x'.repeat(100);
    const node = buildNodeFromSseEvent(makeSseEvent({ detail: longLabel }), 8);
    expect(node.label.length).toBeLessThanOrEqual(80);
  });

  it('reads confidence from metadata', () => {
    const node = buildNodeFromSseEvent(
      makeSseEvent({ metadata: { confidence: 0.72 } }),
      9,
    );
    expect(node.confidence).toBeCloseTo(0.72);
  });

  it('assigns state=error when phase=error', () => {
    const node = buildNodeFromSseEvent(makeSseEvent({ phase: 'error', state: undefined }), 10);
    expect(node.state).toBe('error');
  });

  it('assigns state=done when phase=done', () => {
    const node = buildNodeFromSseEvent(makeSseEvent({ phase: 'done', state: undefined }), 11);
    expect(node.state).toBe('done');
  });
});

// ---------------------------------------------------------------------------
// buildEdgeFromNodes
// ---------------------------------------------------------------------------

describe('buildEdgeFromNodes', () => {
  it('creates a directed edge with correct source/target', () => {
    const a = makeNode({ id: 'a', sequenceNumber: 1 });
    const b = makeNode({ id: 'b', sequenceNumber: 2 });
    const edge = buildEdgeFromNodes(a, b, 2, 0.8, false);
    expect(edge.source).toBe('a');
    expect(edge.target).toBe('b');
  });

  it('sets opacity proportional to weight', () => {
    const a = makeNode({ id: 'a', sequenceNumber: 1 });
    const b = makeNode({ id: 'b', sequenceNumber: 2 });
    const highConf = buildEdgeFromNodes(a, b, 2, 1.0, false);
    const lowConf  = buildEdgeFromNodes(a, b, 2, 0.0, false);
    expect(highConf.opacity).toBeGreaterThan(lowConf.opacity!);
  });

  it('generates a stable deterministic edge id', () => {
    const a = makeNode({ id: 'a', sequenceNumber: 1 });
    const b = makeNode({ id: 'b', sequenceNumber: 2 });
    const edge1 = buildEdgeFromNodes(a, b, 2);
    const edge2 = buildEdgeFromNodes(a, b, 5); // different seq, same nodes
    expect(edge1.id).toBe(edge2.id);
  });
});

// ---------------------------------------------------------------------------
// insertNodeDeterministic
// ---------------------------------------------------------------------------

describe('insertNodeDeterministic', () => {
  it('inserts in sequence order', () => {
    const n1 = makeNode({ id: 'n1', sequenceNumber: 1 });
    const n3 = makeNode({ id: 'n3', sequenceNumber: 3 });
    const n2 = makeNode({ id: 'n2', sequenceNumber: 2 });
    const result = insertNodeDeterministic(
      insertNodeDeterministic([n1], n3),
      n2,
    );
    expect(result.map((n) => n.sequenceNumber)).toEqual([1, 2, 3]);
  });

  it('updates in-place on duplicate id', () => {
    const original = makeNode({ id: 'n1', state: 'running', sequenceNumber: 1 });
    const updated  = makeNode({ id: 'n1', state: 'done',    sequenceNumber: 1 });
    const result = insertNodeDeterministic([original], updated);
    expect(result).toHaveLength(1);
    expect(result[0].state).toBe('done');
  });

  it('preserves frozen position on update', () => {
    const original = makeNode({ id: 'n1', sequenceNumber: 1, fx: 42, fy: 99 });
    const updated  = makeNode({ id: 'n1', sequenceNumber: 1, fx: undefined, fy: undefined });
    const result = insertNodeDeterministic([original], updated);
    expect(result[0].fx).toBe(42);
    expect(result[0].fy).toBe(99);
  });

  it('appends when seq is largest', () => {
    const n1 = makeNode({ id: 'n1', sequenceNumber: 1 });
    const n5 = makeNode({ id: 'n5', sequenceNumber: 5 });
    const result = insertNodeDeterministic([n1], n5);
    expect(result[result.length - 1].id).toBe('n5');
  });
});

// ---------------------------------------------------------------------------
// applySmartModeClustering
// ---------------------------------------------------------------------------

describe('applySmartModeClustering', () => {
  function buildReasoningNodes(count: number): GraphNode[] {
    return Array.from({ length: count }, (_, i) =>
      makeNode({
        id: `r${i}`,
        kind: 'reasoning_step',
        zone: 'reasoning',
        sequenceNumber: i,
      }),
    );
  }

  it('researcher mode: no clustering by default', () => {
    const nodes = buildReasoningNodes(30);
    const result = applySmartModeClustering(nodes, 'researcher', false);
    expect(result).toHaveLength(30);
  });

  it('smart mode: collapses reasoning above threshold (20)', () => {
    const nodes = buildReasoningNodes(21);
    const result = applySmartModeClustering(nodes, 'smart', false);
    const clusterNodes = result.filter((n) => n.isCluster);
    expect(clusterNodes.length).toBeGreaterThanOrEqual(1);
    expect(result.length).toBeLessThan(21);
  });

  it('fast mode: collapses even 1 reasoning node', () => {
    const nodes = buildReasoningNodes(1);
    const result = applySmartModeClustering(nodes, 'fast', false);
    const clusterNodes = result.filter((n) => n.isCluster);
    expect(clusterNodes.length).toBe(1);
  });

  it('decision_point and synthesis_node are never clustered', () => {
    const decisionNode = makeNode({
      id: 'dp1', kind: 'decision_point', zone: 'reasoning', sequenceNumber: 0,
    });
    const synthNode = makeNode({
      id: 'syn1', kind: 'synthesis_node', zone: 'reasoning', sequenceNumber: 1,
    });
    const nodes = [
      ...buildReasoningNodes(25),
      decisionNode,
      synthNode,
    ];
    const result = applySmartModeClustering(nodes, 'smart', false);
    expect(result.find((n) => n.id === 'dp1')).toBeTruthy();
    expect(result.find((n) => n.id === 'syn1')).toBeTruthy();
  });
});

// ---------------------------------------------------------------------------
// expandCluster
// ---------------------------------------------------------------------------

describe('expandCluster', () => {
  it('replaces cluster node with its member nodes', () => {
    const rawNodes: GraphNode[] = [
      makeNode({ id: 'r1', kind: 'reasoning_step', zone: 'reasoning', sequenceNumber: 1 }),
      makeNode({ id: 'r2', kind: 'reasoning_step', zone: 'reasoning', sequenceNumber: 2 }),
    ];
    const clusterNode = makeNode({
      id: 'cluster_reasoning_general',
      kind: 'cluster_node',
      zone: 'reasoning',
      isCluster: true,
      sequenceNumber: 0,
    });
    const displayNodes = [clusterNode];
    const result = expandCluster(displayNodes, 'cluster_reasoning_general', rawNodes);
    expect(result.find((n) => n.id === 'cluster_reasoning_general')).toBeUndefined();
    expect(result.find((n) => n.id === 'r1')).toBeTruthy();
    expect(result.find((n) => n.id === 'r2')).toBeTruthy();
  });

  it('returns unchanged list when clusterId not found', () => {
    const nodes = [makeNode({ id: 'n1', sequenceNumber: 1 })];
    const result = expandCluster(nodes, 'nonexistent_cluster', nodes);
    expect(result).toEqual(nodes);
  });
});

// ---------------------------------------------------------------------------
// diffSnapshots
// ---------------------------------------------------------------------------

describe('diffSnapshots', () => {
  it('detects added nodes', () => {
    const a = makeSnapshot('snap_a', [makeNode({ id: 'n1' })]);
    const b = makeSnapshot('snap_b', [
      makeNode({ id: 'n1' }),
      makeNode({ id: 'n2' }),
    ]);
    const diff = diffSnapshots(a, b);
    expect(diff.nodesAdded).toBe(1);
    expect(diff.nodesRemoved).toBe(0);
  });

  it('detects removed nodes', () => {
    const a = makeSnapshot('snap_a', [
      makeNode({ id: 'n1' }),
      makeNode({ id: 'n2' }),
    ]);
    const b = makeSnapshot('snap_b', [makeNode({ id: 'n1' })]);
    const diff = diffSnapshots(a, b);
    expect(diff.nodesRemoved).toBe(1);
    expect(diff.nodesAdded).toBe(0);
  });

  it('detects state change as changed', () => {
    const a = makeSnapshot('snap_a', [makeNode({ id: 'n1', state: 'running' })]);
    const b = makeSnapshot('snap_b', [makeNode({ id: 'n1', state: 'done' })]);
    const diff = diffSnapshots(a, b);
    expect(diff.nodesChanged).toBe(1);
    expect(diff.nodeDiffs[0].changeSummary).toContain('running→done');
  });

  it('detects identical snapshots', () => {
    const node = makeNode({ id: 'n1', state: 'done' });
    const a = makeSnapshot('snap_a', [node]);
    const b = makeSnapshot('snap_b', [node]);
    const diff = diffSnapshots(a, b);
    expect(diff.nodesAdded).toBe(0);
    expect(diff.nodesRemoved).toBe(0);
    expect(diff.nodesChanged).toBe(0);
    expect(diff.nodesUnchanged).toBe(1);
  });

  it('detects dominant change zone', () => {
    const aNodes = [
      makeNode({ id: 'r1', zone: 'reasoning', state: 'running' }),
      makeNode({ id: 'r2', zone: 'reasoning', state: 'running' }),
      makeNode({ id: 'p1', zone: 'pipeline',  state: 'running' }),
    ];
    const bNodes = [
      makeNode({ id: 'r1', zone: 'reasoning', state: 'done' }),
      makeNode({ id: 'r2', zone: 'reasoning', state: 'done' }),
      makeNode({ id: 'p1', zone: 'pipeline',  state: 'done' }),
    ];
    const diff = diffSnapshots(
      makeSnapshot('snap_a', aNodes),
      makeSnapshot('snap_b', bNodes),
    );
    expect(diff.dominantChangeZone).toBe('reasoning');
  });

  it('populates snapshotIdA and snapshotIdB correctly', () => {
    const diff = diffSnapshots(
      makeSnapshot('alpha', []),
      makeSnapshot('beta',  []),
    );
    expect(diff.snapshotIdA).toBe('alpha');
    expect(diff.snapshotIdB).toBe('beta');
  });
});

// ---------------------------------------------------------------------------
// buildGraphSnapshot
// ---------------------------------------------------------------------------

describe('buildGraphSnapshot', () => {
  it('sets graphSchemaVersion correctly', () => {
    const snap = buildGraphSnapshot({
      snapshotId: 'test_1',
      reasoningMode: 'smart',
      nodes: [],
      edges: [],
      lifecycle: 'frozen',
      layoutMode: 'force',
    });
    expect(snap.graphSchemaVersion).toBe(GRAPH_SCHEMA_VERSION);
  });

  it('counts nodes and edges', () => {
    const nodes = [makeNode(), makeNode({ id: 'n2' })];
    const snap = buildGraphSnapshot({
      snapshotId: 'test_2',
      reasoningMode: 'smart',
      nodes,
      edges: [],
      lifecycle: 'frozen',
      layoutMode: 'force',
    });
    expect(snap.nodeCount).toBe(2);
    expect(snap.edgeCount).toBe(0);
  });

  it('defaults persistenceScope to local', () => {
    const snap = buildGraphSnapshot({
      snapshotId: 'test_3',
      reasoningMode: 'fast',
      nodes: [],
      edges: [],
      lifecycle: 'idle',
      layoutMode: 'force',
    });
    expect(snap.persistenceScope).toBe('local');
  });
});
