// ---------------------------------------------------------------------------
// TracePanel — extracted from pages/index.tsx (Phase 1 refactor)
// Renders the key-value grid of pipeline trace metadata below an assistant
// bubble when the user expands it.
// ---------------------------------------------------------------------------

import styles from '../styles/Home.module.css';
import type { PipelineTrace } from '../types/pipeline';

function fmtDuration(rawMs: number) {
  // Guard: backend may emit float-seconds instead of ms on older runs
  const ms = rawMs > 0 && rawMs < 2 ? rawMs * 1000 : rawMs;
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`;
}

export function TracePanel({ trace }: { trace: PipelineTrace }) {
  const latencies = trace.phase_latencies ?? {};
  return (
    <div className={styles.tracePanel}>
      <div className={styles.traceGrid}>
        {trace.layer0_route && (
          <div className={styles.traceCell}>
            <span className={styles.traceLabel}>Layer 0 route</span>
            <span className={styles.traceValue}>{trace.layer0_route}</span>
          </div>
        )}
        {trace.routing_classification && (
          <div className={styles.traceCell}>
            <span className={styles.traceLabel}>Classification</span>
            <span className={styles.traceValue}>{trace.routing_classification}</span>
          </div>
        )}
        {(trace.routing_domains ?? []).length > 0 && (
          <div className={styles.traceCell}>
            <span className={styles.traceLabel}>Domains</span>
            <span className={styles.traceValue}>{(trace.routing_domains ?? []).join(', ')}</span>
          </div>
        )}
        {trace.expert_decision_type && (
          <div className={styles.traceCell}>
            <span className={styles.traceLabel}>Expert decision</span>
            <span
              className={`${styles.traceValue} ${
                trace.expert_decision_type === 'CREATE_NEW_PATCH' ? styles.traceValuePatch : ''
              }`}
            >
              {trace.expert_decision_type}
            </span>
          </div>
        )}
        {(trace.selected_experts ?? []).length > 0 && (
          <div className={styles.traceCell}>
            <span className={styles.traceLabel}>Experts used</span>
            <span className={styles.traceValue}>{(trace.selected_experts ?? []).join(', ')}</span>
          </div>
        )}
        {trace.expert_confidence != null && (
          <div className={styles.traceCell}>
            <span className={styles.traceLabel}>Confidence</span>
            <span className={styles.traceValue}>
              {(Number(trace.expert_confidence) * 100).toFixed(1)}%
            </span>
          </div>
        )}
        {trace.validation_result && (
          <div className={styles.traceCell}>
            <span className={styles.traceLabel}>Validation</span>
            <span className={styles.traceValue}>{trace.validation_result}</span>
          </div>
        )}
        {Object.keys(latencies).length > 0 && (
          <div className={`${styles.traceCell} ${styles.traceCellFull}`}>
            <span className={styles.traceLabel}>Phase latencies</span>
            <span className={styles.traceValue}>
              {Object.entries(latencies)
                .map(([k, v]) => `${k}: ${fmtDuration(v)}`)
                .join(' · ')}
            </span>
          </div>
        )}
        {trace.trace_id && (
          <div className={`${styles.traceCell} ${styles.traceCellFull}`}>
            <span className={styles.traceLabel}>Trace ID</span>
            <span className={`${styles.traceValue} ${styles.traceId}`}>{trace.trace_id}</span>
          </div>
        )}
      </div>
    </div>
  );
}
