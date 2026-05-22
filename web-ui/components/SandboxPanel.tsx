// ---------------------------------------------------------------------------
// SandboxPanel + LiveToolFeed — extracted from pages/index.tsx (Phase 1)
// ---------------------------------------------------------------------------

import { useState } from 'react';
import styles from '../styles/Home.module.css';
import type { SandboxResult, LiveToolEvent } from '../types/pipeline';

// ---------------------------------------------------------------------------
// Helpers (local to this module)
// ---------------------------------------------------------------------------

function fmtDuration(ms: number) {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms.toFixed(0)}ms`;
}

function shortId(id: string) {
  return id ? id.slice(0, 8) + '…' : '';
}

function outputPreview(output: Record<string, unknown>): string {
  if (!output) return '';
  if (typeof output.error === 'string') return `⚠ ${output.error}`;
  if (output.status === 'empty') return '(no results returned)';
  if (Array.isArray(output.results)) {
    const r = output.results as Array<Record<string, string>>;
    return r.slice(0, 2).map((x) => x.snippet ?? x.abstract ?? x.title ?? '').filter(Boolean).join(' · ').slice(0, 200);
  }
  if (Array.isArray(output.papers)) {
    const p = output.papers as Array<Record<string, unknown>>;
    return p.slice(0, 2).map((x) => `${x.title ?? ''} (${x.year ?? '?'})`).join('; ').slice(0, 200);
  }
  if (Array.isArray(output.entities)) {
    const e = output.entities as Array<Record<string, string>>;
    return e.slice(0, 2).map((x) => `${x.label}: ${x.description ?? ''}`).join('; ').slice(0, 200);
  }
  if (output.result !== undefined) return String(output.result);
  return JSON.stringify(output).slice(0, 200);
}

function toolDisplayName(tool: string): string {
  const names: Record<string, string> = {
    web_search:      'Web Search',
    academic_search: 'Academic',
    knowledge_base:  'Knowledge',
    calculator:      'Calculator',
  };
  return names[tool] ?? tool;
}

// ---------------------------------------------------------------------------
// LiveToolFeed
// ---------------------------------------------------------------------------

export function LiveToolFeed({
  events,
  planDetail,
}: {
  events: LiveToolEvent[];
  planDetail: string;
}) {
  return (
    <div className={styles.liveToolFeed} aria-label="Live tool activity" aria-live="polite">
      {planDetail && (
        <div className={`${styles.liveToolRow} ${styles.liveToolPlan}`}>
          <span className={styles.liveToolIcon} aria-hidden="true">📋</span>
          <span className={styles.liveToolDetail}>Plan: {planDetail}</span>
        </div>
      )}
      {events.map((ev) => {
        const isRunning  = ev.phase === 'running';
        const isOk       = ev.phase === 'ok';
        const isErr      = ev.phase === 'err';
        const rowClass   = isOk ? styles.liveToolOk : isErr ? styles.liveToolErr : styles.liveToolRunning;
        const icon       = isOk ? '✓' : isErr ? '✗' : '⟳';
        const statusLabel = isOk ? 'completed' : isErr ? 'failed' : 'running';
        return (
          <div
            key={ev.toolIndex}
            className={`${styles.liveToolRow} ${rowClass}`}
            aria-label={`${toolDisplayName(ev.toolName)}: ${ev.query} — ${statusLabel}`}
          >
            <span
              className={`${styles.liveToolIcon} ${isRunning ? styles.liveToolSpinning : ''}`}
              aria-hidden="true"
            >
              {icon}
            </span>
            <div className={styles.liveToolBody}>
              <div className={styles.liveToolTop}>
                <span className={styles.liveToolBadge}>{toolDisplayName(ev.toolName)}</span>
                <span className={styles.liveToolQuery}>{ev.query}</span>
                {!isRunning && ev.durationMs !== undefined && (
                  <span className={styles.liveToolElapsed}>{fmtDuration(ev.durationMs)}</span>
                )}
              </div>
              {!isRunning && ev.preview && (
                <p className={styles.liveToolPreview}>{ev.preview}</p>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// SandboxPanel
// ---------------------------------------------------------------------------

export function SandboxPanel({
  sandbox,
  liveEvents,
  planDetail,
  isLoading,
}: {
  sandbox: SandboxResult | null;
  liveEvents: LiveToolEvent[];
  planDetail: string;
  isLoading: boolean;
}) {
  const [openSteps, setOpenSteps] = useState<Set<number>>(new Set());

  function toggleStep(i: number) {
    setOpenSteps((prev) => {
      const next = new Set(prev);
      next.has(i) ? next.delete(i) : next.add(i);
      return next;
    });
  }

  if (isLoading) {
    const hasEvents = liveEvents.length > 0 || planDetail;
    return (
      <div className={styles.sandboxPanel}>
        {!hasEvents ? (
          <div className={styles.sandboxEmpty}>
            <div className={styles.sandboxWaiting}>
              <span className={styles.sandboxDot} />
              <span className={styles.sandboxDot} />
              <span className={styles.sandboxDot} />
            </div>
            <p>Waiting for tool plan…</p>
          </div>
        ) : (
          <LiveToolFeed events={liveEvents} planDetail={planDetail} />
        )}
      </div>
    );
  }

  if (!sandbox) {
    return (
      <div className={styles.sandboxEmpty}>
        <p>Send a message to see sandbox evidence here.</p>
      </div>
    );
  }

  const okSteps  = sandbox.steps.filter((s) => s.status === 'ok');
  const errSteps = sandbox.steps.filter((s) => s.status !== 'ok');

  return (
    <div className={styles.sandboxPanel}>
      <div className={styles.sandboxStats}>
        <span className={styles.sandboxStatOk}>{okSteps.length} ok</span>
        {errSteps.length > 0 && (
          <span className={styles.sandboxStatErr}>{errSteps.length} failed</span>
        )}
        <span className={styles.sandboxStatId}>{shortId(sandbox.trace_id)}</span>
      </div>
      {sandbox.summary && <p className={styles.sandboxSummary}>{sandbox.summary}</p>}
      <div className={styles.sandboxSteps}>
        {sandbox.steps.map((step, i) => (
          <div
            key={i}
            className={`${styles.sandboxStep} ${
              step.status === 'ok' ? styles.sandboxStepOk : styles.sandboxStepErr
            }`}
          >
            <button
              className={styles.sandboxStepHeader}
              onClick={() => toggleStep(i)}
              aria-expanded={openSteps.has(i)}
              aria-controls={`sandbox-step-body-${i}`}
            >
              <span className={styles.sandboxToolBadge}>{toolDisplayName(step.tool)}</span>
              <span className={step.status === 'ok' ? styles.sandboxStatusOk : styles.sandboxStatusErr}>
                {step.status}
              </span>
              <span className={styles.sandboxStepInput}>{step.input?.query ?? ''}</span>
              <span className={styles.sandboxStepDuration}>{fmtDuration(step.duration_ms)}</span>
              <svg
                width="10" height="10" viewBox="0 0 10 10" fill="none" aria-hidden="true"
                style={{
                  transform:  openSteps.has(i) ? 'rotate(90deg)' : 'rotate(0)',
                  transition: 'transform 150ms ease',
                  flexShrink: 0,
                }}
              >
                <path d="M3 2l4 3-4 3" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>
            {openSteps.has(i) && (
              <div id={`sandbox-step-body-${i}`} className={styles.sandboxStepBody}>
                {step.commentary && <p className={styles.sandboxCommentary}>{step.commentary}</p>}
                <p className={styles.sandboxOutput}>{outputPreview(step.output)}</p>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
