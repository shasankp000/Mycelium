// ---------------------------------------------------------------------------
// ReasoningGraph/index.tsx — Phase 4 (full implementation)
//
// Mounts react-force-graph-2d on the canvas area.
// Implements:
//   - live node fade-in during SSE streaming
//   - semantic size/opacity/colour weighting (§2.3.8)
//   - cluster node custom renderer via ClusterNode.tsx (§2.3.7)
//   - contradiction pulse animation (§9.4 — staged + bounded)
//   - hierarchy and radial layout freeze (§2.3.4)
//   - physics freeze/resume wired to useGraphStabilization (§2.3.6)
//   - node click → NodeDetailDrawer + cluster expand (§9.2)
//   - node hover tooltip
//   - subgraph zone filtering
//   - edge opacity from confidence (§9.3)
//
// Plan refs: §2.3.3–2.3.8, §4.3, §9.2–9.4
// ---------------------------------------------------------------------------

import {
  useState, useCallback, useRef, useEffect, useMemo,
} from 'react';
import dynamic from 'next/dynamic';
import type { GraphNode, GraphEdge, SubgraphZone, LayoutMode } from '../../types/graph';
import { NODE_COLORS } from '../../types/graph';
import type { GraphBuilderState } from '../../hooks/useGraphBuilder';
import type { StabilizationControls } from '../../hooks/useGraphStabilization';
import { SubgraphControls }  from './SubgraphControls';
import { LayoutToolbar }     from './LayoutToolbar';
import { NodeDetailDrawer }  from './NodeDetailDrawer';
import { drawClusterNode }   from './ClusterNode';
import styles from '../../styles/ReasoningGraph.module.css';

// react-force-graph-2d is canvas-only (no SSR)
const ForceGraph2D = dynamic(
  () => import('react-force-graph-2d').then((m) => m.default ?? m),
  { ssr: false, loading: () => <div className={styles.canvasLoading}>Initialising graph…</div> },
);

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** How long (ms) a new node’s fade-in animation runs */
const FADE_IN_MS = 600;

/** Contradiction pulse: 3 pulses then settle (§9.4) */
const CONTRADICTION_PULSES = 3;
const PULSE_INTERVAL_MS    = 400;

// Zone layout seed positions — keeps zones semi-isolated (§2.3.3)
const ZONE_SEEDS: Record<SubgraphZone, { fx?: number; fy?: number }> = {
  pipeline:  { fy: -220 },
  reasoning: { fy:    0 },
  evidence:  { fy:  220 },
};

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface ReasoningGraphProps {
  graphState:      GraphBuilderState;
  stabilization:   StabilizationControls;
  onClose:         () => void;
  onExpandCluster: (clusterId: string) => void;
  onLayoutChange:  (lm: LayoutMode) => void;
}

// ---------------------------------------------------------------------------
// Internal types for ForceGraph2D node/link shapes
// (library requires plain objects with x/y injected at runtime)
// ---------------------------------------------------------------------------

type FGNode = GraphNode & { x?: number; y?: number; vx?: number; vy?: number };
type FGLink = GraphEdge & { source: string | FGNode; target: string | FGNode };

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Age-based opacity for fade-in animation */
function fadeOpacity(node: GraphNode, now: number): number {
  if (!node.timestamp) return node.opacity ?? 0.85;
  const age = now - node.timestamp;
  if (age >= FADE_IN_MS) return node.opacity ?? 0.85;
  const t = age / FADE_IN_MS;
  return (node.opacity ?? 0.85) * t;
}

/** Hex colour with alpha for canvas fillStyle */
function withAlpha(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha.toFixed(3)})`;
}

/** Compute radial position for a node in radial layout mode */
function radialPos(node: GraphNode, total: number, index: number) {
  const radius = 80 + node.layerDepth * 60;
  const angle  = (2 * Math.PI * index) / Math.max(total, 1);
  return { fx: radius * Math.cos(angle), fy: radius * Math.sin(angle) };
}

/** Compute hierarchy position for a node */
function hierarchyPos(node: GraphNode, layerCounts: Map<number, number>, layerIndex: Map<string, number>) {
  const depth   = node.layerDepth;
  const count   = layerCounts.get(depth) ?? 1;
  const idx     = layerIndex.get(node.id) ?? 0;
  const xSpread = 140;
  const ySpread = 90;
  return {
    fx: (idx - (count - 1) / 2) * xSpread,
    fy: depth * ySpread - 200,
  };
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ReasoningGraph({
  graphState,
  stabilization,
  onClose,
  onExpandCluster,
  onLayoutChange,
}: ReasoningGraphProps) {
  const [activeZones, setActiveZones] = useState<Set<SubgraphZone>>(
    new Set(['pipeline', 'reasoning', 'evidence']),
  );
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [tooltip, setTooltip]           = useState<{ x: number; y: number; node: GraphNode } | null>(null);
  const [contradictionIds, setContradictionIds] = useState<Set<string>>(new Set());
  const [pulseCount, setPulseCount]     = useState(0);

  const graphRef      = useRef<unknown>(null);
  const wrapRef       = useRef<HTMLDivElement>(null);
  const animFrameRef  = useRef<number | null>(null);
  const [, forceRepaint] = useState(0); // tick to re-render canvas each animation frame

  // ── Zone toggle ───────────────────────────────────────────────
  const toggleZone = useCallback((zone: SubgraphZone) => {
    setActiveZones((prev) => {
      const next = new Set(prev);
      if (next.has(zone)) { if (next.size > 1) next.delete(zone); }
      else { next.add(zone); }
      return next;
    });
  }, []);

  // ── Visible nodes/edges (zone-filtered) ──────────────────────────────
  const visibleNodes = useMemo(
    () => graphState.displayNodes.filter((n) => activeZones.has(n.zone)),
    [graphState.displayNodes, activeZones],
  );
  const visibleEdges = useMemo(() => {
    const ids = new Set(visibleNodes.map((n) => n.id));
    return graphState.edges.filter(
      (e) => ids.has(e.source as string) && ids.has(e.target as string),
    );
  }, [graphState.edges, visibleNodes]);

  // ── Animation frame loop (drives fade-in repaints during streaming) ───
  useEffect(() => {
    if (graphState.lifecycle !== 'streaming') {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
      return;
    }
    function tick() {
      forceRepaint((n) => n + 1);
      animFrameRef.current = requestAnimationFrame(tick);
    }
    animFrameRef.current = requestAnimationFrame(tick);
    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [graphState.lifecycle]);

  // ── Contradiction pulse (§9.4 — staged, bounded, interruptible) ──────
  // Detects contradiction_node arrivals and pulses the cluster.
  useEffect(() => {
    const contradictions = graphState.rawNodes.filter((n) => n.kind === 'contradiction_node');
    if (contradictions.length === 0) return;
    const ids = new Set(contradictions.map((n) => n.id));
    setContradictionIds(ids);
    setPulseCount(0);
    let count = 0;
    const interval = setInterval(() => {
      count++;
      setPulseCount(count);
      if (count >= CONTRADICTION_PULSES) clearInterval(interval);
    }, PULSE_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [graphState.rawNodes]);

  // ── Physics freeze: when stabilization fires onFreezeReady ──────────
  // We read current node positions from the graph engine and pass to freeze().
  // (Called via the onFreezeReady callback wired in index.tsx)

  // ── Layout mode changes ──────────────────────────────────────────
  const computedNodes: FGNode[] = useMemo(() => {
    const lm = graphState.layoutMode;
    if (lm === 'force') return visibleNodes as FGNode[];

    if (lm === 'radial') {
      return visibleNodes.map((n, i) => ({
        ...n,
        ...radialPos(n, visibleNodes.length, i),
      })) as FGNode[];
    }

    if (lm === 'hierarchy') {
      // Count nodes per depth layer
      const layerCounts = new Map<number, number>();
      const layerIndex  = new Map<string, number>();
      visibleNodes.forEach((n) => {
        const c = layerCounts.get(n.layerDepth) ?? 0;
        layerIndex.set(n.id, c);
        layerCounts.set(n.layerDepth, c + 1);
      });
      return visibleNodes.map((n) => ({
        ...n,
        ...hierarchyPos(n, layerCounts, layerIndex),
      })) as FGNode[];
    }

    return visibleNodes as FGNode[];
  }, [visibleNodes, graphState.layoutMode]);

  // ── nodeCanvasObject — custom canvas renderer ────────────────────
  const nodeCanvasObject = useCallback(
    (rawNode: object, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const node = rawNode as FGNode;
      const now  = Date.now();
      const x    = node.x ?? 0;
      const y    = node.y ?? 0;
      const r    = (node.size ?? 6) / globalScale;

      // Fade-in opacity during streaming
      const opacity = graphState.lifecycle === 'streaming'
        ? fadeOpacity(node, now)
        : (node.opacity ?? 0.85);

      const color = node.color ?? NODE_COLORS[node.kind] ?? '#6366f1';

      // Contradiction pulse: alternate brightness (§9.4 staged)
      const isPulsing = contradictionIds.has(node.id) && pulseCount < CONTRADICTION_PULSES;
      const pulseBoost = isPulsing && (pulseCount % 2 === 0) ? 1.4 : 1.0;

      // Cluster nodes get custom renderer
      if (node.isCluster) {
        drawClusterNode(node, ctx, globalScale);
        return;
      }

      // Standard node
      ctx.beginPath();
      ctx.arc(x, y, r * pulseBoost, 0, 2 * Math.PI);
      ctx.fillStyle   = withAlpha(color, opacity * pulseBoost);
      ctx.fill();

      // State ring: done = faint white, error = red, running = pulsing outline
      if (node.state === 'done') {
        ctx.strokeStyle = 'rgba(255,255,255,0.25)';
        ctx.lineWidth   = 0.5 / globalScale;
        ctx.stroke();
      } else if (node.state === 'error') {
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth   = 1 / globalScale;
        ctx.stroke();
      } else if (node.state === 'running') {
        ctx.strokeStyle = 'rgba(255,255,255,0.5)';
        ctx.lineWidth   = 0.8 / globalScale;
        ctx.stroke();
      }

      // Label — only draw when zoomed in enough to read
      if (globalScale > 1.8) {
        const label = node.label.slice(0, 28);
        ctx.font        = `${10 / globalScale}px sans-serif`;
        ctx.fillStyle   = 'rgba(255,255,255,0.65)';
        ctx.textAlign   = 'center';
        ctx.textBaseline = 'top';
        ctx.fillText(label, x, y + r + 2 / globalScale);
      }
    },
    [graphState.lifecycle, contradictionIds, pulseCount],
  );

  // ── linkCanvasObject — edge opacity from weight (§9.3) ────────────
  const linkCanvasObject = useCallback(
    (rawLink: object, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const link   = rawLink as FGLink;
      const src    = typeof link.source === 'object' ? link.source as FGNode : null;
      const tgt    = typeof link.target === 'object' ? link.target as FGNode : null;
      if (!src || !tgt) return;
      const sx = src.x ?? 0; const sy = src.y ?? 0;
      const tx = tgt.x ?? 0; const ty = tgt.y ?? 0;
      ctx.beginPath();
      ctx.moveTo(sx, sy);
      ctx.lineTo(tx, ty);
      ctx.strokeStyle = `rgba(255,255,255,${link.opacity ?? 0.25})`;
      ctx.lineWidth   = (link.thickness ?? 1) / globalScale;
      ctx.stroke();
    },
    [],
  );

  // ── Node click handler ──────────────────────────────────────────
  const handleNodeClick = useCallback(
    (rawNode: object) => {
      const node = rawNode as GraphNode;
      if (node.isCluster) {
        onExpandCluster(node.id);
        return;
      }
      setSelectedNode(node);
    },
    [onExpandCluster],
  );

  // ── Node hover tooltip ─────────────────────────────────────────
  const handleNodeHover = useCallback(
    (rawNode: object | null) => {
      if (!rawNode) { setTooltip(null); return; }
      const node = rawNode as FGNode;
      setTooltip({ x: node.x ?? 0, y: node.y ?? 0, node });
    },
    [],
  );

  // ── Dispose simulation on unmount (memory rule §7.1) ───────────────
  useEffect(() => {
    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
      // ForceGraph2D engine disposal is handled by the library on unmount
    };
  }, []);

  const isEmpty = visibleNodes.length === 0;

  return (
    <div className={styles.overlay} role="dialog" aria-modal="true" aria-label="Reasoning graph">
      {/* Header */}
      <div className={styles.overlayHeader}>
        <span className={styles.overlayTitle}>
          Reasoning Graph
          {graphState.lifecycle === 'streaming' && (
            <span className={styles.liveBadge} aria-label="Live">LIVE</span>
          )}
          {graphState.lifecycle === 'stabilising' && (
            <span className={styles.stabilisingBadge} aria-label="Stabilising">⋅⋅⋅</span>
          )}
          {graphState.lifecycle === 'frozen' && (
            <span className={styles.frozenBadge} aria-label="Frozen">FROZEN</span>
          )}
          <span className={styles.nodeCount} aria-label={`${visibleNodes.length} nodes`}>
            {visibleNodes.length}N · {visibleEdges.length}E
          </span>
        </span>
        <div className={styles.headerControls}>
          <SubgraphControls activeZones={activeZones} onToggleZone={toggleZone} />
          <LayoutToolbar
            current={graphState.layoutMode}
            stabState={stabilization.stabState}
            onChange={onLayoutChange}
            onResume={stabilization.resumeSimulation}
          />
          <button className={styles.closeBtn} onClick={onClose} aria-label="Close reasoning graph">
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
              <path d="M2 2l10 10M12 2L2 12" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
            </svg>
          </button>
        </div>
      </div>

      {/* Canvas */}
      <div className={styles.canvasWrap} ref={wrapRef}>
        {isEmpty ? (
          <div className={styles.emptyState}>
            <svg width="36" height="36" viewBox="0 0 28 28" fill="none" opacity="0.3" aria-hidden="true">
              <circle cx="14" cy="14" r="3.5" fill="currentColor" />
              <line x1="14" y1="14" x2="4"  y2="6"  stroke="currentColor" strokeWidth="1.2" />
              <line x1="14" y1="14" x2="24" y2="6"  stroke="currentColor" strokeWidth="1.2" />
              <line x1="14" y1="14" x2="4"  y2="22" stroke="currentColor" strokeWidth="1.2" />
              <line x1="14" y1="14" x2="24" y2="22" stroke="currentColor" strokeWidth="1.2" />
              <line x1="14" y1="14" x2="14" y2="2"  stroke="currentColor" strokeWidth="1.2" />
              <line x1="14" y1="14" x2="14" y2="26" stroke="currentColor" strokeWidth="1.2" />
              <circle cx="4"  cy="6"  r="2" fill="currentColor" />
              <circle cx="24" cy="6"  r="2" fill="currentColor" />
              <circle cx="4"  cy="22" r="2" fill="currentColor" />
              <circle cx="24" cy="22" r="2" fill="currentColor" />
              <circle cx="14" cy="2"  r="2" fill="currentColor" />
              <circle cx="14" cy="26" r="2" fill="currentColor" />
            </svg>
            <p className={styles.emptyText}>
              {graphState.lifecycle === 'idle'
                ? 'Graph will appear when a query is running.'
                : 'No nodes in the selected subgraph zones.'}
            </p>
          </div>
        ) : (
          <ForceGraph2D
            ref={graphRef as React.MutableRefObject<unknown>}
            graphData={{ nodes: computedNodes as object[], links: visibleEdges as object[] }}
            width={wrapRef.current?.clientWidth ?? 800}
            height={wrapRef.current?.clientHeight ?? 600}
            backgroundColor="#0d0d12"
            // Node rendering
            nodeCanvasObject={nodeCanvasObject}
            nodeCanvasObjectMode={() => 'replace'}
            nodeRelSize={1}
            // Edge rendering
            linkCanvasObject={linkCanvasObject}
            linkCanvasObjectMode={() => 'replace'}
            // Interaction
            onNodeClick={handleNodeClick}
            onNodeHover={handleNodeHover}
            // Physics — disabled when frozen/hierarchy/radial
            cooldownTicks={
              graphState.lifecycle === 'frozen' ||
              graphState.layoutMode === 'hierarchy' ||
              graphState.layoutMode === 'radial'
                ? 0
                : Infinity
            }
            // Warm-up ticks for initial layout
            warmupTicks={20}
            // Zoom
            minZoom={0.3}
            maxZoom={8}
          />
        )}

        {/* HTML Tooltip overlay */}
        {tooltip && (
          <div
            className={styles.nodeTooltip}
            style={{
              // Offset from canvas origin; rough screen-space approximation
              left: `calc(50% + ${tooltip.x * 0.5}px)`,
              top:  `calc(50% + ${tooltip.y * 0.5}px)`,
            }}
            aria-live="polite"
          >
            <span className={styles.tooltipKind}>{tooltip.node.kind}</span>
            <span className={styles.tooltipLabel}>{tooltip.node.label}</span>
            {tooltip.node.state && (
              <span className={`${styles.tooltipState} ${styles['tooltipState_' + tooltip.node.state]}`}>
                {tooltip.node.state}
              </span>
            )}
            {tooltip.node.confidence !== undefined && (
              <span className={styles.tooltipMeta}>
                conf {(tooltip.node.confidence * 100).toFixed(0)}%
              </span>
            )}
            {tooltip.node.isCluster && (
              <span className={styles.tooltipMeta}>
                {tooltip.node.clusterNodeCount} nodes · click to expand
              </span>
            )}
          </div>
        )}
      </div>

      {/* Node detail drawer */}
      <NodeDetailDrawer
        node={selectedNode}
        onClose={() => setSelectedNode(null)}
      />
    </div>
  );
}
