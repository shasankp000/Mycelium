// ---------------------------------------------------------------------------
// GraphSnapshotLoader — loads a local snapshot for replay/comparison.
// Integrates with §9.1 (local session-layer persistence) and
// §9.5 (side-by-side comparison, stubbed for Phase 5).
// ---------------------------------------------------------------------------

import { useState, useEffect } from 'react';
import { listLocalSnapshots, type GraphSnapshot } from '../../types/graph';
import styles from '../../styles/ReasoningGraph.module.css';

export function GraphSnapshotLoader({
  onLoad,
}: {
  onLoad: (snapshot: GraphSnapshot) => void;
}) {
  const [snapshots, setSnapshots] = useState<GraphSnapshot[]>([]);

  useEffect(() => {
    setSnapshots(listLocalSnapshots().sort((a, b) => b.createdAt - a.createdAt));
  }, []);

  if (snapshots.length === 0) {
    return (
      <p className={styles.snapshotEmpty}>No local snapshots yet. Snapshots are saved after each completed query.</p>
    );
  }

  return (
    <div className={styles.snapshotList}>
      {snapshots.map((snap) => (
        <button
          key={snap.snapshotId}
          className={styles.snapshotItem}
          onClick={() => onLoad(snap)}
        >
          <span className={styles.snapshotQuery}>
            {snap.queryText ? snap.queryText.slice(0, 60) : snap.snapshotId}
          </span>
          <span className={styles.snapshotMeta}>
            {new Date(snap.createdAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            {' · '}{snap.nodeCount} nodes
            {' · '}{snap.reasoningMode}
          </span>
        </button>
      ))}
    </div>
  );
}
