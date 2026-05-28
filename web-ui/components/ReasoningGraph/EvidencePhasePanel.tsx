/**
 * EvidencePhasePanel.tsx
 *
 * Renders Phase D evidence nodes (predicate_extraction, evidence_dst_fusion,
 * contradiction_integration, evidence_dst_done) that arrive via the SSE stream
 * and are stored in the evidence subgraph.
 *
 * Mount this panel inside ReasoningGraphCanvas alongside the existing
 * SandboxPanel.  It only renders when at least one Phase D node is present,
 * so it is a no-op for sessions that predate Phase D.
 *
 * Props
 * -----
 * nodes  — full list of GraphNodes currently in the graph store; the panel
 *           filters to the Phase D subset internally.
 */

import React from 'react';
import type { GraphNode } from '../../types/graph';
import { PHASE_NAMES } from '../../types/sseContract';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface EvidencePhasePanelProps {
  nodes: GraphNode[];
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const PHASE_D_PHASES = new Set<string>([
  PHASE_NAMES.PREDICATE_EXTRACTION,
  PHASE_NAMES.EVIDENCE_DST_FUSION,
  PHASE_NAMES.CONTRADICTION_INTEGRATION,
  PHASE_NAMES.EVIDENCE_DST_DONE,
]);

// Human-readable labels for the phase names
const PHASE_LABELS: Record<string, string> = {
  [PHASE_NAMES.PREDICATE_EXTRACTION]:      'Predicate Extraction',
  [PHASE_NAMES.EVIDENCE_DST_FUSION]:       'DST Fusion',
  [PHASE_NAMES.CONTRADICTION_INTEGRATION]: 'Contradiction Integration',
  [PHASE_NAMES.EVIDENCE_DST_DONE]:         'Evidence DST Done',
};

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function DSTBar({ label, value }: { label: string; value: number | undefined }) {
  if (value === undefined || value === null) return null;
  const pct = Math.round(Math.min(1, Math.max(0, value)) * 100);
  const colour =
    label === 'm_false'   ? '#e05252'
    : label === 'm_true'  ? '#4caf8a'
    : '#7a7a9a';

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 3 }}>
      <span style={{ fontSize: 10, color: '#999', width: 68, flexShrink: 0 }}>
        {label}
      </span>
      <div
        style={{
          height: 6,
          width: `${pct}%`,
          maxWidth: 100,
          minWidth: pct > 0 ? 2 : 0,
          background: colour,
          borderRadius: 3,
          transition: 'width 0.3s ease',
        }}
      />
      <span style={{ fontSize: 10, color: '#bbb' }}>{pct}%</span>
    </div>
  );
}

function PhaseDNodeCard({ node }: { node: GraphNode }) {
  const m = (node.metadata ?? {}) as Record<string, unknown>;
  const phaseName = (m.phase_name as string | undefined) ?? '';
  const label = PHASE_LABELS[phaseName] ?? node.label ?? phaseName;

  const mTrue    = m.m_true    as number | undefined;
  const mFalse   = m.m_false   as number | undefined;
  const mUnknown = m.m_unknown as number | undefined;

  const contradictions   = m.contradictions   as number | undefined;
  const severityMax      = m.severity_max     as number | undefined;
  const modalCeilings    = m.modal_ceilings   as string[] | undefined;
  const netConf          = m.net_confidence   as number | undefined;
  const isUncertain      = m.is_genuinely_uncertain as boolean | undefined;

  const hasBeliefBars = mTrue !== undefined || mFalse !== undefined || mUnknown !== undefined;

  return (
    <div
      style={{
        marginBottom: 10,
        paddingBottom: 10,
        borderBottom: '1px solid #2e2e3a',
      }}
    >
      {/* Phase label */}
      <div
        style={{
          fontSize: 11,
          fontWeight: 600,
          color:
            phaseName === PHASE_NAMES.CONTRADICTION_INTEGRATION
              ? '#e08080'
              : '#9ab4f0',
          marginBottom: 5,
          letterSpacing: '0.01em',
        }}
      >
        {label}
      </div>

      {/* Belief bars */}
      {hasBeliefBars && (
        <div style={{ marginBottom: 4 }}>
          <DSTBar label="m_true"    value={mTrue} />
          <DSTBar label="m_false"   value={mFalse} />
          <DSTBar label="m_unknown" value={mUnknown} />
        </div>
      )}

      {/* net_confidence */}
      {netConf !== undefined && (
        <div style={{ fontSize: 10, color: '#aaa', marginBottom: 3 }}>
          net conf:{' '}
          <span style={{ color: netConf > 0.6 ? '#4caf8a' : netConf > 0.35 ? '#e0c060' : '#e08080' }}>
            {(netConf * 100).toFixed(1)}%
          </span>
          {isUncertain === true && (
            <span style={{ color: '#aaa', marginLeft: 5 }}>⚠ uncertain</span>
          )}
        </div>
      )}

      {/* Contradiction summary */}
      {contradictions !== undefined && (
        <div
          style={{
            fontSize: 10,
            color: contradictions > 0 ? '#e08080' : '#6db86d',
            marginBottom: 3,
          }}
        >
          {contradictions} contradiction{contradictions !== 1 ? 's' : ''}
          {severityMax !== undefined && contradictions > 0
            ? ` · max sev ${severityMax.toFixed(2)}`
            : ''}
        </div>
      )}

      {/* Modal ceiling badge */}
      {Array.isArray(modalCeilings) && modalCeilings.length > 0 && (
        <div
          style={{
            fontSize: 10,
            color: '#c0a060',
            background: 'rgba(192,160,96,0.12)',
            borderRadius: 4,
            padding: '2px 5px',
            display: 'inline-block',
            marginTop: 2,
          }}
        >
          ⚠️ modal ceiling applied ({modalCeilings.length})
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// EvidencePhasePanel
// ---------------------------------------------------------------------------

export function EvidencePhasePanel({ nodes }: EvidencePhasePanelProps) {
  const phaseDNodes = nodes.filter(
    (n) =>
      n.zone === 'evidence' &&
      PHASE_D_PHASES.has((n.metadata?.phase_name as string | undefined) ?? ''),
  );

  if (phaseDNodes.length === 0) return null;

  return (
    <div
      style={{
        position: 'absolute',
        bottom: 12,
        right: 12,
        background: 'rgba(16,16,24,0.93)',
        border: '1px solid #2e2e3a',
        borderRadius: 8,
        padding: '10px 14px',
        minWidth: 230,
        maxWidth: 300,
        fontSize: 12,
        color: '#ccc',
        zIndex: 20,
        boxShadow: '0 4px 24px rgba(0,0,0,0.4)',
        backdropFilter: 'blur(6px)',
      }}
    >
      {/* Panel header */}
      <div
        style={{
          fontWeight: 700,
          fontSize: 11,
          color: '#e0e0e0',
          letterSpacing: '0.06em',
          textTransform: 'uppercase',
          marginBottom: 10,
          paddingBottom: 6,
          borderBottom: '1px solid #2e2e3a',
        }}
      >
        Phase D — Evidence
      </div>

      {/* Node cards */}
      {phaseDNodes.map((node) => (
        <PhaseDNodeCard key={node.id} node={node} />
      ))}
    </div>
  );
}

export default EvidencePhasePanel;
