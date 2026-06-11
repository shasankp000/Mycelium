// ---------------------------------------------------------------------------
// useGraphBuilder — stateful hook that consumes SSE events and builds
// the live graph state deterministically.
// Plan refs: §2.3.1, §3, §4.3
// Phase 5: swap `mode` closure dep → modeRef (Gap 2 fix).
// ---------------------------------------------------------------------------

import { useState, useCallback, useRef, useEffect } from 'react';
import type { SseEvent, ReasoningMode } from '../types/pipeline';
import type { GraphNode, GraphEdge, GraphLifecycle, LayoutMode } from '../types/graph';
import {
  buildNodeFromSseEvent,
  buildEdgeFromNodes,
  insertNodeDeterministic,
  applySmartModeClustering,
  expandCluster,
  buildGraphSnapshot,
} from '../components/ReasoningGraph/graphBuilder';
import {
  GRAPH_SCHEMA_VERSION,
  saveSnapshotLocal,
  type GraphSnapshot,
} from '../types/graph';

// ---------------------------------------------------------------------------
// State shape
// ---------------------------------------------------------------------------

export interface GraphBuilderState {
  /** All nodes in deterministic insertion order (pre-clustering) */
  rawNodes: GraphNode[];
  /** All edges in deterministic insertion order */
  edges: GraphEdge[];
  /** Nodes after clustering applied (used for rendering) */
  displayNodes: GraphNode[];
  /** Current graph lifecycle */
  lifecycle: GraphLifecycle;
  /** Active layout mode */
  layoutMode: LayoutMode;
  /** Latest snapshot (set after done/error) */
  snapshot: GraphSnapshot | null;
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useGraphBuilder(
  mode: ReasoningMode,
  isMobile = false,
) {
  const seqRef      = useRef<number>(0);
  const lastNodeRef = useRef<GraphNode | null>(null);

  // Gap 2 fix: keep mode in a ref so ingestSseEvent / finalise callbacks
  // always read the current value without needing to be re-created on every
  // mode change.  This eliminates the stale-closure risk while avoiding
  // unnecessary renders.
  const modeRef = useRef<ReasoningMode>(mode);
  useEffect(() => { modeRef.current = mode; }, [mode]);

  const [state, setState] = useState<GraphBuilderState>({
    rawNodes:     [],
    edges:        [],
    displayNodes: [],
    lifecycle:    'idle',
    layoutMode:   'force',
    snapshot:     null,
  });

  // ── ingestSseEvent ────────────────────────────────────────────────────
  // Main entry point. Called from useSseStream callbacks in index.tsx.
  // `mode` dep removed — reads modeRef.current instead (Gap 2).
  const ingestSseEvent = useCallback(
    (event: SseEvent) => {
      const phase = (event.phase_name ?? event.phase ?? '') as string;
      if (!phase || phase === 'done' || phase === 'error') return;

      // Use backend sequence_number if present (§4.3), else increment local
      const seq = typeof event.sequence_number === 'number'
        ? event.sequence_number
        : ++seqRef.current;

      const newNode = buildNodeFromSseEvent(event, seq);

      setState((prev) => {
        const currentMode = modeRef.current;
        const updatedRaw  = insertNodeDeterministic(prev.rawNodes, newNode);

        // Build edge to previous node in same zone (simple sequential DAG)
        let updatedEdges = prev.edges;
        if (lastNodeRef.current) { // cross-zone edges allowed — zone gate removed (Bug #9)
          const newEdge = buildEdgeFromNodes(
            lastNodeRef.current,
            newNode,
            seq,
            newNode.confidence,
            currentMode === 'researcher',
          );
          // Avoid duplicate edges
          const edgeExists = prev.edges.some((e) => e.id === newEdge.id);
          if (!edgeExists) updatedEdges = [...prev.edges, newEdge];
        }
        lastNodeRef.current = newNode;

        const displayNodes = applySmartModeClustering(updatedRaw, currentMode, isMobile);

        return {
          ...prev,
          rawNodes:     updatedRaw,
          edges:        updatedEdges,
          displayNodes,
          lifecycle:    'streaming',
        };
      });
    },
    // isMobile is a stable prop that changes at most on viewport resize;
    // modeRef is a ref so it never changes identity — no mode dep needed.
    [isMobile],
  );

  // ── finalise ──────────────────────────────────────────────────────
  // Called when done/error is received. Transitions to stabilising,
  // then builds + saves the snapshot.
  // `mode` dep removed — reads modeRef.current instead (Gap 2).
  const finalise = useCallback(
    (params: {
      traceId?: string;
      queryText?: string;
      totalElapsedMs?: number;
    }) => {
      setState((prev) => {
        const snapshotId = `snap_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
        const snapshot = buildGraphSnapshot({
          snapshotId,
          traceId:          params.traceId,
          queryText:        params.queryText,
          reasoningMode:    modeRef.current,
          nodes:            prev.rawNodes,
          edges:            prev.edges,
          lifecycle:        'stabilising',
          layoutMode:       prev.layoutMode,
          totalElapsedMs:   params.totalElapsedMs,
          persistenceScope: 'local',
        });
        // Persist to localStorage session-layer (§9.1)
        saveSnapshotLocal(snapshot);
        return {
          ...prev,
          lifecycle: 'stabilising',
          snapshot,
        };
      });
    },
    // No mode dep — modeRef.current is read at call time (Gap 2).
    [],
  );

  // ── freeze ────────────────────────────────────────────────────────
  // Called by useGraphStabilization after the cooldown fires.
  // Fixes all node positions to pause physics.
  const freeze = useCallback((frozenPositions: Map<string, { x: number; y: number }>) => {
    setState((prev) => ({
      ...prev,
      lifecycle:    'frozen',
      rawNodes:     prev.rawNodes.map((n) => {
        const pos = frozenPositions.get(n.id);
        return pos ? { ...n, fx: pos.x, fy: pos.y } : n;
      }),
      displayNodes: prev.displayNodes.map((n) => {
        const pos = frozenPositions.get(n.id);
        return pos ? { ...n, fx: pos.x, fy: pos.y } : n;
      }),
    }));
  }, []);

  // ── resumePhysics ────────────────────────────────────────────────────
  // Unfreezes positions so physics simulation can resume.
  const resumePhysics = useCallback(() => {
    setState((prev) => ({
      ...prev,
      lifecycle:    'resumed',
      rawNodes:     prev.rawNodes.map((n) => ({ ...n, fx: null, fy: null })),
      displayNodes: prev.displayNodes.map((n) => ({ ...n, fx: null, fy: null })),
    }));
  }, []);

  // ── setLayoutMode ───────────────────────────────────────────────────
  const setLayoutMode = useCallback((lm: LayoutMode) => {
    setState((prev) => ({ ...prev, layoutMode: lm }));
  }, []);

  // ── expandClusterById ─────────────────────────────────────────────────
  // Expands a cluster node to reveal constituent nodes (§9.2).
  const expandClusterById = useCallback((clusterId: string) => {
    setState((prev) => ({
      ...prev,
      displayNodes: expandCluster(prev.displayNodes, clusterId, prev.rawNodes),
    }));
  }, []);

  // ── reset ──────────────────────────────────────────────────────────
  const reset = useCallback(() => {
    seqRef.current      = 0;
    lastNodeRef.current = null;
    setState({
      rawNodes:     [],
      edges:        [],
      displayNodes: [],
      lifecycle:    'idle',
      layoutMode:   'force',
      snapshot:     null,
    });
  }, []);

  return {
    ...state,
    ingestSseEvent,
    finalise,
    freeze,
    resumePhysics,
    setLayoutMode,
    expandClusterById,
    reset,
  };
}
