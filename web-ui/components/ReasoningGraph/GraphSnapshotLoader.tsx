// ---------------------------------------------------------------------------
// GraphSnapshotLoader — Phase 5 update
// Adds: select-two-for-compare mode (§9.5).
// User can click one snapshot to load it, or select two and hit “Compare”.
// ---------------------------------------------------------------------------

import { useState, useEffect } from 'react';
import { listLocalSnapshots, type GraphSnapshot, diffSnapshots } from '../../types/graph';
import type { GraphDiff } from '../../types/graph';
import styles from '../../styles/ReasoningGraph.module.css';

export function GraphSnapshotLoader({
  onLoad,
  onCompare,
}: {
  onLoad: (snapshot: GraphSnapshot) => void;
  onCompare: (diff: GraphDiff, snapshotA: GraphSnapshot, snapshotB: GraphSnapshot) => void;
}) {
  const [snapshots, setSnapshots]   = useState<GraphSnapshot[]>([]);
  const [selected, setSelected]     = useState<Set<string>>(new Set());
  const [compareMode, setCompareMode] = useState(false);

  useEffect(() => {
    setSnapshots(listLocalSnapshots().sort((a, b) => b.createdAt - a.createdAt));
  }, []);

  function toggleSelect(id: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else if (next.size < 2) {
        next.add(id);
      }
      return next;
    });
  }

  function handleCompare() {
    const ids = [...selected];
    if (ids.length !== 2) return;
    const a = snapshots.find((s) => s.snapshotId === ids[0]);
    const b = snapshots.find((s) => s.snapshotId === ids[1]);
    if (!a || !b) return;
    // Always diff oldest→newest
    const [older, newer] = a.createdAt <= b.createdAt ? [a, b] : [b, a];
    onCompare(diffSnapshots(older, newer), older, newer);
  }

  if (snapshots.length === 0) {
    return (
      <p className={styles.snapshotEmpty}>
        No local snapshots yet. Snapshots are saved after each completed query.
      </p>
    );
  }

  return (
    <div className={styles.snapshotList}>
      <div className={styles.snapshotListHeader}>
        <button
          className={`${styles.snapshotModeBtn} ${compareMode ? styles.snapshotModeBtnActive : ''}`}
          onClick={() => { setCompareMode((v) => !v); setSelected(new Set()); }}
        >
          {compareMode ? 'Cancel compare' : 'Compare two…'}
        </button>
        {compareMode && selected.size === 2 && (
          <button className={styles.snapshotCompareBtn} onClick={handleCompare}>
            Compare
          </button>
        )}
        {compareMode && (
          <span className={styles.snapshotCompareHint}>
            {selected.size}/2 selected
          </span>
        )}
      </div>

      {snapshots.map((snap) => {
        const isSelected = selected.has(snap.snapshotId);
        return (
          <button
            key={snap.snapshotId}
            className={`${styles.snapshotItem} ${isSelected ? styles.snapshotItemSelected : ''}`}
            onClick={() => {
              if (compareMode) {
                toggleSelect(snap.snapshotId);
              } else {
                onLoad(snap);
              }
            }}
          >
            {compareMode && (
              <span
                className={`${styles.snapshotCheckbox} ${isSelected ? styles.snapshotCheckboxChecked : ''}`}
                aria-hidden="true"
              />
            )}
            <span className={styles.snapshotQuery}>
              {snap.queryText ? snap.queryText.slice(0, 60) : snap.snapshotId}
            </span>
            <span className={styles.snapshotMeta}>
              {new Date(snap.createdAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              {' · '}{snap.nodeCount} nodes
              {' · '}{snap.reasoningMode}
            </span>
          </button>
        );
      })}
    </div>
  );
}
