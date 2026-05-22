// ---------------------------------------------------------------------------
// ReasoningGraph/index.tsx — Phase 3 stub
//
// Shell of the full-panel overlay. The canvas mount and
// react-force-graph-2d rendering are wired in Phase 4.
//
// What this file does NOW (Phase 3):
//   - Accepts all props it will need in Phase 4
//   - Renders the chrome: subgraph toggle bar + layout toolbar
//   - Renders a placeholder canvas area
//   - Renders NodeDetailDrawer (empty until a node is clicked)
//
// Plan refs: §2.3.4, §2.3.6, §2.3.7
// ---------------------------------------------------------------------------

import { useState, useCallback, useRef } from 'react';
import type { GraphNode, GraphEdge, SubgraphZone, LayoutMode } from '../../types/graph';
import type { GraphBuilderState } from '../../hooks/useGraphBuilder';
import type { StabilizationControls } from '../../hooks/useGraphStabilization';
import { SubgraphControls }  from './SubgraphControls';
import { LayoutToolbar }     from './LayoutToolbar';
import { NodeDetailDrawer }  from './NodeDetailDrawer';
import styles from '../../styles/ReasoningGraph.module.css';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface ReasoningGraphProps {
  graphState:     GraphBuilderState;
  stabilization:  StabilizationControls;
  onClose:        () => void;
  onExpandCluster:(clusterId: string) => void;
  onLayoutChange: (lm: LayoutMode) => void;
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
  const [activeZones, setActiveZones]   = useState<Set<SubgraphZone>>(new Set(['pipeline', 'reasoning', 'evidence']));
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const canvasRef                       = useRef<HTMLDivElement>(null);

  const toggleZone = useCallback((zone: SubgraphZone) => {
    setActiveZones((prev) => {
      const next = new Set(prev);
      if (next.has(zone)) {
        // Always keep at least one zone active
        if (next.size > 1) next.delete(zone);
      } else {
        next.add(zone);
      }
      return next;
    });
  }, []);

  // Filter display nodes by active zones
  const visibleNodes = graphState.displayNodes.filter((n) => activeZones.has(n.zone));
  const visibleEdges = graphState.edges.filter((e) => {
    const src = graphState.rawNodes.find((n) => n.id === e.source);
    const tgt = graphState.rawNodes.find((n) => n.id === e.target);
    return src && tgt && activeZones.has(src.zone) && activeZones.has(tgt.zone);
  });

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
          {graphState.lifecycle === 'frozen' && (
            <span className={styles.frozenBadge} aria-label="Frozen">FROZEN</span>
          )}
        </span>
        <div className={styles.headerControls}>
          <SubgraphControls
            activeZones={activeZones}
            onToggleZone={toggleZone}
          />
          <LayoutToolbar
            current={graphState.layoutMode}
            stabState={stabilization.stabState}
            onChange={onLayoutChange}
            onResume={stabilization.resumeSimulation}
          />
          <button
            className={styles.closeBtn}
            onClick={onClose}
            aria-label="Close reasoning graph"
          >
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
              <path d="M2 2l10 10M12 2L2 12" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
            </svg>
          </button>
        </div>
      </div>

      {/* Canvas area — Phase 4 mounts ForceGraph2D here */}
      <div className={styles.canvasWrap} ref={canvasRef}>
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
                : 'No nodes in selected subgraph zones.'}
            </p>
          </div>
        ) : (
          <div className={styles.canvasPlaceholder} aria-label={`${visibleNodes.length} nodes, ${visibleEdges.length} edges. Graph rendering mounts here in Phase 4.`}>
            {/* Phase 4: <ForceGraph2D nodes={visibleNodes} links={visibleEdges} ... /> */}
            <p className={styles.placeholderText}>
              {visibleNodes.length} nodes · {visibleEdges.length} edges
              <br />
              <span className={styles.placeholderSub}>Graph canvas mounts in Phase 4</span>
            </p>
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
