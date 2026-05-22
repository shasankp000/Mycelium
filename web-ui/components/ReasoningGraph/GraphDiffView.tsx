// ---------------------------------------------------------------------------
// GraphDiffView (Phase 5 — §9.5)
// Renders a structured, readable diff between two graph snapshots.
// Shows added / removed / changed nodes grouped by zone,
// plus a summary bar of counts.
// ---------------------------------------------------------------------------

import type { GraphDiff, NodeDiffEntry, DiffChangeType } from '../../types/graph';
import { NODE_COLORS } from '../../types/graph';
import styles from '../../styles/ReasoningGraph.module.css';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const CHANGE_LABEL: Record<DiffChangeType, string> = {
  added:     '+ added',
  removed:   '− removed',
  changed:   'Δ changed',
  unchanged: '= same',
};

const CHANGE_CLASS: Record<DiffChangeType, string> = {
  added:     styles.diffAdded,
  removed:   styles.diffRemoved,
  changed:   styles.diffChanged,
  unchanged: styles.diffUnchanged,
};

// ---------------------------------------------------------------------------
// NodeDiffRow
// ---------------------------------------------------------------------------

function NodeDiffRow({ entry }: { entry: NodeDiffEntry }) {
  const node    = entry.nodeB ?? entry.nodeA;
  const color   = node ? NODE_COLORS[node.kind] : '#888';
  const label   = node?.label ?? node?.id ?? '—';
  const kind    = node?.kind ?? '—';
  const zone    = node?.zone ?? '—';
  const conf    = node?.confidence !== undefined
    ? `${Math.round(node.confidence * 100)}%`
    : null;

  return (
    <div className={`${styles.diffRow} ${CHANGE_CLASS[entry.changeType]}`}>
      <span className={styles.diffRowDot} style={{ background: color }} />
      <div className={styles.diffRowBody}>
        <span className={styles.diffRowLabel}>{label}</span>
        <span className={styles.diffRowMeta}>
          {kind} · {zone}
          {conf && ` · conf ${conf}`}
          {entry.changeSummary && (
            <span className={styles.diffRowChange}> — {entry.changeSummary}</span>
          )}
        </span>
      </div>
      <span className={styles.diffRowTag}>{CHANGE_LABEL[entry.changeType]}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// GraphDiffView
// ---------------------------------------------------------------------------

interface GraphDiffViewProps {
  diff: GraphDiff;
  queryA?: string;
  queryB?: string;
  onClose: () => void;
}

export function GraphDiffView({ diff, queryA, queryB, onClose }: GraphDiffViewProps) {
  const hasChanges =
    diff.nodesAdded + diff.nodesRemoved + diff.nodesChanged > 0;

  // Group non-unchanged diffs by zone
  const interestingDiffs = diff.nodeDiffs.filter(
    (d) => d.changeType !== 'unchanged',
  );
  const byZone: Record<string, NodeDiffEntry[]> = {};
  interestingDiffs.forEach((d) => {
    const z = (d.nodeB ?? d.nodeA)?.zone ?? 'unknown';
    byZone[z] = [...(byZone[z] ?? []), d];
  });

  return (
    <div className={styles.diffPanel} role="dialog" aria-label="Graph snapshot diff">
      {/* Header */}
      <div className={styles.diffHeader}>
        <div className={styles.diffTitles}>
          <span className={styles.diffTitleA} title={queryA}>
            A: {queryA ? queryA.slice(0, 40) : diff.snapshotIdA}
          </span>
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
            <path d="M2 6h8M7 3l3 3-3 3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <span className={styles.diffTitleB} title={queryB}>
            B: {queryB ? queryB.slice(0, 40) : diff.snapshotIdB}
          </span>
        </div>
        <button className={styles.diffCloseBtn} onClick={onClose} aria-label="Close diff view">
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
            <path d="M2 2l8 8M10 2l-8 8" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
          </svg>
        </button>
      </div>

      {/* Summary bar */}
      <div className={styles.diffSummaryBar}>
        <span className={styles.diffSumAdded}>+{diff.nodesAdded} nodes</span>
        <span className={styles.diffSumRemoved}>−{diff.nodesRemoved} nodes</span>
        <span className={styles.diffSumChanged}>Δ{diff.nodesChanged} changed</span>
        <span className={styles.diffSumUnchanged}>{diff.nodesUnchanged} same</span>
        {diff.dominantChangeZone && (
          <span className={styles.diffSumZone}>▶ {diff.dominantChangeZone}</span>
        )}
        <span className={styles.diffSumEdges}>
          edges: +{diff.edgesAdded} −{diff.edgesRemoved}
        </span>
      </div>

      {!hasChanges && (
        <p className={styles.diffNoChanges}>
          ✔ Snapshots are structurally identical ({diff.nodesUnchanged} nodes unchanged).
        </p>
      )}

      {/* Diff rows grouped by zone */}
      <div className={styles.diffBody}>
        {(['pipeline', 'reasoning', 'evidence'] as const).map((zone) => {
          const rows = byZone[zone];
          if (!rows || rows.length === 0) return null;
          return (
            <div key={zone} className={styles.diffZoneGroup}>
              <p className={styles.diffZoneLabel}>{zone}</p>
              {rows.map((entry, i) => (
                <NodeDiffRow key={i} entry={entry} />
              ))}
            </div>
          );
        })}
      </div>
    </div>
  );
}
