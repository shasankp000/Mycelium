// ---------------------------------------------------------------------------
// ReasoningGraph/index.tsx — Phase 5 polish
// Adds: keyboard shortcuts, mobile constraints, simulation disposal on unmount.
// All Phase 4 + Phase 5 diff/snapshot features retained.
// ---------------------------------------------------------------------------

'use client';

import React, {
  useState, useCallback, useRef, useEffect,
} from 'react';
import dynamic from 'next/dynamic';
import type { ForceGraphMethods } from 'react-force-graph-2d';
import styles from '../../styles/ReasoningGraph.module.css';
import type { GraphBuilderState } from '../../hooks/useGraphBuilder';
import type { StabilizationControls } from '../../hooks/useGraphStabilization';
import type {
  GraphNode, GraphEdge, LayoutMode, GraphSnapshot, GraphDiff,
} from '../../types/graph';
import { NODE_COLORS, CLUSTER_THRESHOLDS } from '../../types/graph';
import { SubgraphControls }     from './SubgraphControls';
import { LayoutToolbar }        from './LayoutToolbar';
import { NodeDetailDrawer }     from './NodeDetailDrawer';
import { GraphSnapshotLoader }  from './GraphSnapshotLoader';
import { GraphDiffView }        from './GraphDiffView';

// ---------------------------------------------------------------------------
// Dynamic import — ForceGraph2D is browser-only
// ---------------------------------------------------------------------------

const ForceGraph2D = dynamic(
  () => import('react-force-graph-2d').then((m) => m.default),
  { ssr: false, loading: () => <div className={styles.canvasLoading}>Initialising graph renderer…</div> },
);

// ---------------------------------------------------------------------------
// Mobile detection helper (runs once per render, safe for SSR)
// ---------------------------------------------------------------------------

function useIsMobile(): boolean {
  const [mobile, setMobile] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia('(max-width: 768px)');
    setMobile(mq.matches);
    const handler = (e: MediaQueryListEvent) => setMobile(e.matches);
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, []);
  return mobile;
}

// ---------------------------------------------------------------------------
// Canvas drawing helpers
// ---------------------------------------------------------------------------

const FADE_IN_DURATION_MS = 400;

function easeIn(t: number): number {
  return t * t;
}

function nodeAge(node: GraphNode): number {
  const ts = node.timestamp ?? 0;
  if (!ts) return 1;
  return Math.min(1, (Date.now() - ts) / FADE_IN_DURATION_MS);
}

const CONTRADICTION_COLORS = ['#dc2626', '#ff6b6b'];
let _contraFrame = 0;
function contraColor(): string {
  _contraFrame = (_contraFrame + 1) % (CONTRADICTION_COLORS.length * 8);
  return CONTRADICTION_COLORS[Math.floor(_contraFrame / 8)];
}

function drawNode(
  node: GraphNode & { x?: number; y?: number },
  ctx: CanvasRenderingContext2D,
  globalScale: number,
  selectedId: string | null,
  frozenPositions: Map<string, { x: number; y: number }> | null,
  isMobile: boolean,
) {
  const x = node.x ?? 0;
  const y = node.y ?? 0;
  const r = (node.size ?? 6) / 2;
  const age = easeIn(nodeAge(node));

  const isSelected = selectedId === node.id;
  const baseOpacity = node.opacity ?? 0.85;
  const opacity = baseOpacity * age;

  ctx.save();
  ctx.globalAlpha = opacity;

  // Contradiction pulse ring
  if (node.kind === 'contradiction_node') {
    ctx.beginPath();
    ctx.arc(x, y, r + 3, 0, 2 * Math.PI);
    ctx.strokeStyle = contraColor();
    ctx.lineWidth = 1.5;
    ctx.globalAlpha = 0.4 * age;
    ctx.stroke();
    ctx.globalAlpha = opacity;
  }

  // Main node fill
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.fillStyle = node.color ?? NODE_COLORS[node.kind] ?? '#888';
  ctx.fill();

  // State ring
  if (node.state === 'running') {
    ctx.strokeStyle = 'rgba(251,191,36,0.8)';
    ctx.lineWidth   = 1.2;
    ctx.stroke();
  } else if (node.state === 'error') {
    ctx.strokeStyle = 'rgba(239,68,68,0.9)';
    ctx.lineWidth   = 1.5;
    ctx.stroke();
  } else if (node.state === 'skipped') {
    ctx.globalAlpha = 0.3 * age;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.fillStyle = '#888';
    ctx.fill();
    ctx.globalAlpha = opacity;
  }

  // Selection ring
  if (isSelected) {
    ctx.beginPath();
    ctx.arc(x, y + 2.5, r + 2.5, 0, 2 * Math.PI);
    ctx.strokeStyle = 'rgba(165,180,252,0.9)';
    ctx.lineWidth   = 2;
    ctx.stroke();
  }

  // Frozen pin dot
  if (frozenPositions?.has(node.id)) {
    ctx.beginPath();
    ctx.arc(x, y - r - 3, 1.5, 0, 2 * Math.PI);
    ctx.fillStyle = 'rgba(255,255,255,0.25)';
    ctx.fill();
  }

  // Labels: suppressed entirely on mobile; visible at zoom ≥1.2 on desktop
  if (!isMobile && globalScale >= 1.2) {
    ctx.font         = `${Math.min(4.5, 4 / globalScale * 5)}px sans-serif`;
    ctx.fillStyle    = 'rgba(255,255,255,0.75)';
    ctx.textAlign    = 'center';
    ctx.textBaseline = 'middle';
    ctx.globalAlpha  = Math.min(1, (globalScale - 1.2) / 0.5) * age;
    ctx.fillText(node.label.slice(0, 18), x, y + r + 6);
  }

  ctx.restore();
}

function drawEdge(
  link: { source: GraphNode & { x?: number; y?: number }; target: GraphNode & { x?: number; y?: number } } & GraphEdge,
  ctx: CanvasRenderingContext2D,
) {
  const sx = link.source.x ?? 0;
  const sy = link.source.y ?? 0;
  const tx = link.target.x ?? 0;
  const ty = link.target.y ?? 0;
  ctx.save();
  ctx.globalAlpha = link.opacity ?? 0.4;
  ctx.strokeStyle = 'rgba(255,255,255,0.6)';
  ctx.lineWidth   = link.thickness ?? 1;
  ctx.beginPath();
  ctx.moveTo(sx, sy);
  ctx.lineTo(tx, ty);
  ctx.stroke();
  ctx.restore();
}

// ---------------------------------------------------------------------------
// Subgraph zone cycle order for keyboard nav
// ---------------------------------------------------------------------------

type ZoneFilter = 'all' | 'pipeline' | 'reasoning' | 'evidence';
const ZONE_CYCLE: ZoneFilter[] = ['all', 'pipeline', 'reasoning', 'evidence'];

// ---------------------------------------------------------------------------
// ReasoningGraph props
// ---------------------------------------------------------------------------

interface ReasoningGraphProps {
  graphState:           GraphBuilderState;
  stabilization:        StabilizationControls;
  onClose:              () => void;
  onExpandCluster:      (clusterId: string) => void;
  onLayoutChange:       (mode: LayoutMode) => void;
}

// ---------------------------------------------------------------------------
// ReasoningGraph component
// ---------------------------------------------------------------------------

export function ReasoningGraph({
  graphState,
  stabilization,
  onClose,
  onExpandCluster,
  onLayoutChange,
}: ReasoningGraphProps) {
  const { displayNodes, edges, lifecycle, layoutMode, snapshot } = graphState;

  const fgRef = useRef<ForceGraphMethods<GraphNode, GraphEdge>>(null);
  const isMobile = useIsMobile();

  const [activeTab, setActiveTab]               = useState<'graph' | 'snapshots'>('graph');
  const [selectedNode, setSelectedNode]         = useState<GraphNode | null>(null);
  const [frozenPositions, setFrozenPositions]   = useState<Map<string, { x: number; y: number }> | null>(null);
  const [tooltip, setTooltip]                   = useState<{ node: GraphNode; x: number; y: number } | null>(null);
  const [zoneFilter, setZoneFilter]             = useState<ZoneFilter>('all');

  // Phase 5: diff state
  const [activeDiff, setActiveDiff]   = useState<GraphDiff | null>(null);
  const [diffSnapA, setDiffSnapA]     = useState<GraphSnapshot | null>(null);
  const [diffSnapB, setDiffSnapB]     = useState<GraphSnapshot | null>(null);

  const isLive        = lifecycle === 'streaming';
  const isStabilising = lifecycle === 'stabilising';
  const isFrozen      = lifecycle === 'frozen' || lifecycle === 'resumed';
  const isEmpty       = displayNodes.length === 0;

  // Mobile node cap — slice to mobileMaxNodes, preferring high-confidence nodes
  const visibleNodes = React.useMemo(() => {
    let nodes = zoneFilter === 'all'
      ? displayNodes
      : displayNodes.filter((n) => n.zone === zoneFilter);
    if (isMobile && nodes.length > CLUSTER_THRESHOLDS.mobileMaxNodes) {
      nodes = [...nodes]
        .sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0))
        .slice(0, CLUSTER_THRESHOLDS.mobileMaxNodes);
    }
    return nodes;
  }, [displayNodes, zoneFilter, isMobile]);

  // Edges that connect visible nodes only
  const visibleNodeIds = React.useMemo(
    () => new Set(visibleNodes.map((n) => n.id)),
    [visibleNodes],
  );
  const visibleEdges = React.useMemo(
    () => edges.filter(
      (e) => visibleNodeIds.has(String(e.source)) && visibleNodeIds.has(String(e.target)),
    ),
    [edges, visibleNodeIds],
  );

  // Freeze: collect positions from engine
  useEffect(() => {
    if (isFrozen && fgRef.current) {
      const positions = new Map<string, { x: number; y: number }>();
      displayNodes.forEach((n) => {
        const d = n as GraphNode & { x?: number; y?: number };
        if (d.x !== undefined && d.y !== undefined) {
          positions.set(n.id, { x: d.x, y: d.y });
        }
      });
      setFrozenPositions(positions);
    } else if (!isFrozen) {
      setFrozenPositions(null);
    }
  }, [isFrozen, displayNodes]);

  // Hierarchy / radial layout pin
  useEffect(() => {
    if (layoutMode === 'hierarchy' && fgRef.current) {
      const tierMap = new Map<number, GraphNode[]>();
      displayNodes.forEach((n) => {
        const tier = tierMap.get(n.layerDepth) ?? [];
        tier.push(n);
        tierMap.set(n.layerDepth, tier);
      });
      const W = 700;
      const yStep = 80;
      tierMap.forEach((nodes, depth) => {
        nodes.forEach((n, i) => {
          const nodeObj = n as GraphNode & { fx?: number | null; fy?: number | null };
          nodeObj.fx = (i - (nodes.length - 1) / 2) * (W / Math.max(nodes.length, 1));
          nodeObj.fy = depth * yStep - ((Math.max(...[...tierMap.keys()]) * yStep) / 2);
        });
      });
    } else if (layoutMode === 'radial' && fgRef.current) {
      const N = displayNodes.length;
      displayNodes.forEach((n, i) => {
        const nodeObj = n as GraphNode & { fx?: number | null; fy?: number | null };
        const angle = (2 * Math.PI * i) / Math.max(N, 1);
        const radius = 200 + n.layerDepth * 40;
        nodeObj.fx = Math.cos(angle) * radius;
        nodeObj.fy = Math.sin(angle) * radius;
      });
    } else if (layoutMode === 'force') {
      displayNodes.forEach((n) => {
        const nodeObj = n as GraphNode & { fx?: number | null; fy?: number | null };
        if (!frozenPositions?.has(n.id)) {
          nodeObj.fx = null;
          nodeObj.fy = null;
        }
      });
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layoutMode]);

  // ── Phase 5: Force simulation disposal on unmount ──────────────────────
  useEffect(() => {
    return () => {
      if (fgRef.current) {
        try { fgRef.current.pauseAnimation(); } catch { /* ignore */ }
      }
    };
  }, []);

  // ── Phase 5: Keyboard shortcuts ────────────────────────────────────────
  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      // Don't fire when user is typing in an input / textarea
      const tag = (e.target as HTMLElement).tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

      switch (e.key) {
        case 'Escape':
          onClose();
          break;
        case 'h':
        case 'H':
          onLayoutChange('hierarchy');
          break;
        case 'r':
        case 'R':
          onLayoutChange('radial');
          break;
        case 'f':
        case 'F':
          onLayoutChange('force');
          break;
        case ' ':
          e.preventDefault(); // stop page scroll
          if (isFrozen) stabilization.resumeSimulation();
          break;
        case 'ArrowLeft': {
          e.preventDefault();
          setZoneFilter((prev) => {
            const idx = ZONE_CYCLE.indexOf(prev);
            return ZONE_CYCLE[(idx - 1 + ZONE_CYCLE.length) % ZONE_CYCLE.length];
          });
          break;
        }
        case 'ArrowRight': {
          e.preventDefault();
          setZoneFilter((prev) => {
            const idx = ZONE_CYCLE.indexOf(prev);
            return ZONE_CYCLE[(idx + 1) % ZONE_CYCLE.length];
          });
          break;
        }
      }
    }

    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [onClose, onLayoutChange, isFrozen, stabilization]);

  const handleNodeClick = useCallback((node: GraphNode) => {
    if (node.isCluster) {
      onExpandCluster(node.id);
      return;
    }
    setSelectedNode((prev) => (prev?.id === node.id ? null : node));
  }, [onExpandCluster]);

  const handleNodeHover = useCallback((node: GraphNode | null) => {
    if (!node) { setTooltip(null); return; }
    const d = node as GraphNode & { x?: number; y?: number };
    if (d.x !== undefined && d.y !== undefined) {
      setTooltip({ node, x: d.x, y: d.y });
    }
  }, []);

  const handleLoadSnapshot = useCallback((snap: GraphSnapshot) => {
    console.info('[ReasoningGraph] snapshot loaded:', snap.snapshotId, snap.nodeCount, 'nodes');
  }, []);

  const handleCompare = useCallback((
    diff: GraphDiff,
    snapA: GraphSnapshot,
    snapB: GraphSnapshot,
  ) => {
    setActiveDiff(diff);
    setDiffSnapA(snapA);
    setDiffSnapB(snapB);
    setActiveTab('graph');
  }, []);

  const handleCloseDiff = useCallback(() => {
    setActiveDiff(null);
    setDiffSnapA(null);
    setDiffSnapB(null);
  }, []);

  const graphData = React.useMemo(() => ({
    nodes: visibleNodes as (GraphNode & object)[],
    links: visibleEdges as (GraphEdge & object)[],
  }), [visibleNodes, visibleEdges]);

  // Mobile: halve cooldown ticks so physics settles sooner
  const cooldownTicks = isFrozen ? 0 : (isMobile ? 75 : 150);

  // Canvas dimensions
  const canvasWidth  = typeof window !== 'undefined' ? window.innerWidth  : 800;
  const canvasHeight = typeof window !== 'undefined'
    ? window.innerHeight - (isMobile ? 160 : 100)
    : 600;

  return (
    <div
      className={styles.overlay}
      role="dialog"
      aria-modal="true"
      aria-label="Reasoning graph overlay"
    >
      {/* Header */}
      <div className={styles.overlayHeader}>
        <div className={styles.overlayTitle}>
          Reasoning Graph
          {isLive        && <span className={styles.liveBadge}       aria-label="Streaming live">LIVE</span>}
          {isStabilising && <span className={styles.stabilisingBadge} aria-label="Stabilising">⋅⋅⋅</span>}
          {isFrozen      && <span className={styles.frozenBadge}     aria-label="Physics frozen">▣ frozen</span>}
          <span className={styles.nodeCount}>
            {visibleNodes.length}n / {visibleEdges.length}e
            {isMobile && displayNodes.length > CLUSTER_THRESHOLDS.mobileMaxNodes && (
              <span className={styles.mobileCap} title="Mobile node cap active">
                {' '}(of {displayNodes.length})
              </span>
            )}
          </span>
        </div>

        <div className={styles.headerControls}>
          <SubgraphControls />
          <LayoutToolbar
            current={layoutMode}
            onChange={onLayoutChange}
          />
          {isFrozen && (
            <button
              className={styles.resumeBtn}
              onClick={stabilization.resumeSimulation}
              aria-label="Resume physics simulation (Space)"
              title="Resume simulation [Space]"
            >
              ⟳ Resume
            </button>
          )}
          <button
            className={styles.closeBtn}
            onClick={onClose}
            aria-label="Close graph overlay (Escape)"
            title="Close [Esc]"
          >
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
              <path d="M2 2l8 8M10 2l-8 8" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
            </svg>
          </button>
        </div>
      </div>

      {/* Keyboard hint bar — desktop only, shown once until dismissed */}
      {!isMobile && (
        <div className={styles.keyHintBar} aria-hidden="true">
          <kbd>H</kbd> hierarchy &nbsp;·&nbsp;
          <kbd>R</kbd> radial &nbsp;·&nbsp;
          <kbd>F</kbd> force &nbsp;·&nbsp;
          <kbd>Space</kbd> resume &nbsp;·&nbsp;
          <kbd>←</kbd><kbd>→</kbd> zone &nbsp;·&nbsp;
          <kbd>Esc</kbd> close
        </div>
      )}

      {/* Tab strip */}
      <div className={styles.tabStrip}>
        {(['graph', 'snapshots'] as const).map((tab) => (
          <button
            key={tab}
            className={`${styles.tabBtn} ${activeTab === tab ? styles.tabBtnActive : ''}`}
            onClick={() => setActiveTab(tab)}
          >
            {tab === 'graph' ? 'Graph' : 'Snapshots'}
            {tab === 'snapshots' && activeDiff && (
              <span className={styles.tabDiffDot} aria-label="Diff active" />
            )}
          </button>
        ))}
      </div>

      {/* Main content */}
      <div className={styles.canvasWrap}>
        {/* Snapshots tab */}
        {activeTab === 'snapshots' && (
          <div className={styles.snapshotPanel}>
            <GraphSnapshotLoader
              onLoad={handleLoadSnapshot}
              onCompare={handleCompare}
            />
          </div>
        )}

        {/* Graph tab */}
        {activeTab === 'graph' && (
          <>
            {isEmpty ? (
              <div className={styles.emptyState}>
                <svg width="32" height="32" viewBox="0 0 28 28" fill="none" aria-hidden="true" opacity="0.3">
                  <circle cx="14" cy="14" r="3.5" fill="white" />
                  <line x1="14" y1="14" x2="4"  y2="6"  stroke="white" strokeWidth="1" />
                  <line x1="14" y1="14" x2="24" y2="6"  stroke="white" strokeWidth="1" />
                  <line x1="14" y1="14" x2="4"  y2="22" stroke="white" strokeWidth="1" />
                  <line x1="14" y1="14" x2="24" y2="22" stroke="white" strokeWidth="1" />
                </svg>
                <p className={styles.emptyText}>
                  No graph data yet. Send a query and the pipeline&apos;s reasoning graph will appear here in real-time.
                </p>
              </div>
            ) : (
              <ForceGraph2D
                ref={fgRef}
                graphData={graphData}
                nodeCanvasObject={(node, ctx, globalScale) =>
                  drawNode(
                    node as GraphNode & { x?: number; y?: number },
                    ctx,
                    globalScale,
                    selectedNode?.id ?? null,
                    frozenPositions,
                    isMobile,
                  )
                }
                linkCanvasObject={(link, ctx) =>
                  drawEdge(
                    link as { source: GraphNode & { x?: number; y?: number }; target: GraphNode & { x?: number; y?: number } } & GraphEdge,
                    ctx,
                  )
                }
                onNodeClick={(node) => handleNodeClick(node as GraphNode)}
                onNodeHover={(node) => handleNodeHover(node as GraphNode | null)}
                cooldownTicks={cooldownTicks}
                nodeId="id"
                linkSource="source"
                linkTarget="target"
                backgroundColor="#0d0d12"
                width={canvasWidth}
                height={canvasHeight}
              />
            )}

            {/* Tooltip */}
            {tooltip && (
              <div
                className={styles.nodeTooltip}
                style={{
                  left: `calc(${tooltip.x}px + 50%)`,
                  top:  `calc(${tooltip.y}px)`,
                }}
              >
                <span className={styles.tooltipKind}>{tooltip.node.kind}</span>
                <span className={styles.tooltipLabel}>{tooltip.node.label}</span>
                <span className={`${styles.tooltipState} ${styles[`tooltipState_${tooltip.node.state}`]}`}>
                  {tooltip.node.state}
                </span>
                {tooltip.node.confidence !== undefined && (
                  <span className={styles.tooltipMeta}>
                    conf {Math.round(tooltip.node.confidence * 100)}%
                  </span>
                )}
              </div>
            )}

            {/* Phase 5: GraphDiffView panel */}
            {activeDiff && (
              <GraphDiffView
                diff={activeDiff}
                queryA={diffSnapA?.queryText}
                queryB={diffSnapB?.queryText}
                onClose={handleCloseDiff}
              />
            )}

            {/* Node detail drawer */}
            {selectedNode && (
              <NodeDetailDrawer
                node={selectedNode}
                snapshot={snapshot}
                onClose={() => setSelectedNode(null)}
              />
            )}
          </>
        )}
      </div>
    </div>
  );
}
