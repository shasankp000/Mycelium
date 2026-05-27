// ---------------------------------------------------------------------------
// ChatBubble — two-zone assistant bubble per plan Section 2.2
//
// Zone 1: answer text rendered via react-markdown + remark-gfm
// Zone 2: footer chips — Reasoning graph | Evidence | Elapsed time
//          + "Need a second opinion?" chip shown only on fast/smart answers
//
// The "Reasoning graph" chip is wired up in Phase 4 when ReasoningGraph is
// built. For now it calls onGraphOpen() which index.tsx will eventually
// connect to the graph overlay state.
// ---------------------------------------------------------------------------

import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { PipelineTrace, SandboxResult, ReasoningMode } from '../types/pipeline';
import styles from '../styles/ChatBubble.module.css';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function totalElapsedMs(trace?: PipelineTrace): number | null {
  if (!trace?.phase_latencies) return null;
  const vals = Object.values(trace.phase_latencies);
  if (vals.length === 0) return null;
  return vals.reduce((a, b) => a + b, 0);
}

function fmtElapsed(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms.toFixed(0)}ms`;
}

// ---------------------------------------------------------------------------
// ChatBubble component
// ---------------------------------------------------------------------------

interface ChatBubbleProps {
  content: string;
  trace?: PipelineTrace;
  sandbox?: SandboxResult;
  /** Called when user clicks the Reasoning Graph chip — Phase 4 connects this */
  onGraphOpen?: () => void;
  /** Called when user clicks the Evidence chip */
  onEvidenceOpen?: () => void;
  /** Whether the graph chip is in "active" (panel open) state */
  graphActive?: boolean;
  /**
   * The mode that produced this answer.
   * When 'fast' or 'smart', a "Need a second opinion?" chip is rendered
   * that escalates to the full researcher pipeline in a new session.
   */
  reasoningMode?: ReasoningMode;
  /**
   * Called when the user clicks "Need a second opinion?"
   * The parent opens a fresh researcher-mode session with the same query.
   * Only rendered when reasoningMode is 'fast' or 'smart'.
   */
  onSecondOpinion?: () => void;
}

export function ChatBubble({
  content,
  trace,
  sandbox,
  onGraphOpen,
  onEvidenceOpen,
  graphActive = false,
  reasoningMode,
  onSecondOpinion,
}: ChatBubbleProps) {
  const elapsedMs = totalElapsedMs(trace);

  // Show the escalation chip only for fast / smart answers that have a handler
  const showSecondOpinion =
    (reasoningMode === 'fast' || reasoningMode === 'smart') && !!onSecondOpinion;

  return (
    <div className={styles.bubble}>
      {/* Zone 1 — answer text */}
      <div className={styles.answerZone}>
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
          {content}
        </ReactMarkdown>
      </div>

      {/* Divider */}
      <div className={styles.divider} aria-hidden="true" />

      {/* Zone 2 — footer chips */}
      <div className={styles.footerRow}>
        {/* Reasoning graph chip — always rendered; disabled until Phase 4 connects it */}
        <button
          className={[
            styles.chip,
            graphActive ? styles.chipActive : '',
          ].join(' ').trim()}
          onClick={onGraphOpen}
          aria-label="Open reasoning graph"
          aria-pressed={graphActive}
          disabled={!onGraphOpen}
        >
          <span className={styles.chipIcon} aria-hidden="true">🔍</span>
          Reasoning graph
        </button>

        {/* Evidence chip — only shown when sandbox data exists */}
        {sandbox && (
          <button
            className={styles.chip}
            onClick={onEvidenceOpen}
            aria-label="Open evidence panel"
            disabled={!onEvidenceOpen}
          >
            <span className={styles.chipIcon} aria-hidden="true">📋</span>
            Evidence
          </button>
        )}

        {/* Second opinion chip — only on fast/smart answers */}
        {showSecondOpinion && (
          <button
            className={`${styles.chip} ${styles.chipSecondOpinion}`}
            onClick={onSecondOpinion}
            aria-label="Not satisfied? Re-run with the full Researcher pipeline in a new session"
          >
            <span className={styles.chipIcon} aria-hidden="true">🔄</span>
            Need a second opinion?
          </button>
        )}

        {/* Elapsed time badge — non-interactive */}
        {elapsedMs !== null && (
          <span
            className={`${styles.chip} ${styles.chipElapsed}`}
            aria-label={`Total pipeline time: ${fmtElapsed(elapsedMs)}`}
          >
            <span className={styles.chipIcon} aria-hidden="true">⏱</span>
            {fmtElapsed(elapsedMs)}
          </span>
        )}
      </div>
    </div>
  );
}
