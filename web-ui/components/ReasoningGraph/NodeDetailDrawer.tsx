// ---------------------------------------------------------------------------
// NodeDetailDrawer — right-side drawer showing full metadata for a clicked node.
// Plan ref: §2.3.4
// ---------------------------------------------------------------------------

import type { GraphNode } from '../../types/graph';
import { NODE_COLORS } from '../../types/graph';
import styles from '../../styles/ReasoningGraph.module.css';

function fmtConfidence(c?: number): string {
  if (c === undefined) return '—';
  return `${(c * 100).toFixed(0)}%`;
}

function fmtMs(ms?: number): string {
  if (ms === undefined) return '—';
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
}

export function NodeDetailDrawer({
  node,
  onClose,
}: {
  node:    GraphNode | null;
  onClose: () => void;
}) {
  if (!node) return null;

  const metaEntries = Object.entries(node.metadata ?? {}).filter(
    ([, v]) => v !== null && v !== undefined && v !== '',
  );

  return (
    <aside
      className={styles.nodeDrawer}
      aria-label={`Node detail: ${node.label}`}
    >
      <div className={styles.nodeDrawerHeader}>
        <span
          className={styles.nodeDrawerKindDot}
          style={{ background: NODE_COLORS[node.kind] }}
          aria-hidden="true"
        />
        <span className={styles.nodeDrawerTitle}>{node.label}</span>
        <button
          className={styles.nodeDrawerClose}
          onClick={onClose}
          aria-label="Close node detail"
        >
          ×
        </button>
      </div>

      <div className={styles.nodeDrawerBody}>
        <table className={styles.nodeDrawerTable}>
          <tbody>
            <tr><th>Kind</th>       <td><code>{node.kind}</code></td></tr>
            <tr><th>State</th>      <td><code>{node.state}</code></td></tr>
            <tr><th>Zone</th>       <td>{node.zone}</td></tr>
            <tr><th>Layer depth</th><td>{node.layerDepth}</td></tr>
            <tr><th>Confidence</th> <td>{fmtConfidence(node.confidence)}</td></tr>
            <tr><th>Leverage</th>   <td>{node.leverage !== undefined ? node.leverage.toFixed(3) : '—'}</td></tr>
            <tr><th>Centrality</th> <td>{node.centrality !== undefined ? node.centrality.toFixed(3) : '—'}</td></tr>
            <tr><th>Seq #</th>      <td>{node.sequenceNumber}</td></tr>
            {node.eventId && <tr><th>Event ID</th><td><code>{node.eventId}</code></td></tr>}
            {node.timestamp && (
              <tr><th>Timestamp</th><td>{new Date(node.timestamp).toISOString()}</td></tr>
            )}
            {node.isCluster && (
              <>
                <tr><th>Cluster nodes</th> <td>{node.clusterNodeCount}</td></tr>
                <tr><th>Avg confidence</th><td>{fmtConfidence(node.clusterAvgConfidence)}</td></tr>
                <tr><th>Domain</th>        <td>{node.clusterDominantDomain ?? '—'}</td></tr>
                <tr><th>Max depth</th>     <td>{node.clusterMaxDepth ?? '—'}</td></tr>
              </>
            )}
          </tbody>
        </table>

        {metaEntries.length > 0 && (
          <>
            <p className={styles.nodeDrawerSection}>Raw metadata</p>
            <table className={styles.nodeDrawerTable}>
              <tbody>
                {metaEntries.map(([k, v]) => (
                  <tr key={k}>
                    <th>{k}</th>
                    <td><code>{typeof v === 'object' ? JSON.stringify(v) : String(v)}</code></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
      </div>
    </aside>
  );
}
