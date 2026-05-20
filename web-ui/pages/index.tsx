import Head from 'next/head';
import { useState, useRef, useEffect, useCallback } from 'react';
import axios, { AxiosError } from 'axios';
import styles from '../styles/Home.module.css';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? 'http://localhost:8000';
const MAX_INPUT_CHARS = 2000;
const SSE_RETRY_ATTEMPTS = 3;
const SSE_RETRY_DELAY_MS = 1500;
const CALIBRATION_POLL_INTERVAL_MS = 4000;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface PipelineTrace {
  layer0_route?: string;
  routing_classification?: string;
  routing_domains?: string[];
  expert_decision_type?: string;
  selected_experts?: string[];
  expert_confidence?: number | null;
  validation_result?: string;
  phase_latencies?: Record<string, number>;
  trace_id?: string;
}

interface SandboxStep {
  tool: string;
  input: Record<string, string>;
  output: Record<string, unknown>;
  commentary: string;
  status: 'ok' | 'error' | string;
  duration_ms: number;
}

interface SandboxResult {
  trace_id: string;
  summary: string;
  steps: SandboxStep[];
  started_at: string;
  finished_at: string;
}

interface Message {
  role: 'user' | 'assistant' | 'error';
  content: string;
  trace?: PipelineTrace;
  traceOpen?: boolean;
  sandbox?: SandboxResult;
  retryQuery?: string;
}

interface MyceliumRunSummary {
  trace_id?: string;
  timestamp?: string;
  sentence?: string;
  layer0?: { route?: string };
  routing?: { classification?: string; selected_domains?: string[] };
  expert_decision?: {
    decision_type?: string;
    selected_experts?: string[];
    expert_confidence?: number;
  };
  phase3?: {
    validation_decision?: { result_class?: string };
    phase_latencies_ms?: Record<string, number>;
  };
}

interface ChatApiResponse {
  answer?: string;
  trace?: MyceliumRunSummary;
  sandbox?: SandboxResult;
}

interface HistoryTrace {
  trace_id: string;
  timestamp: string;
  user_query: string;
  run_summary: MyceliumRunSummary;
  sandbox_result?: Record<string, unknown>;
}

export interface LiveToolEvent {
  toolIndex: number;
  toolName: string;
  query: string;
  phase: 'running' | 'ok' | 'err' | 'plan';
  durationMs?: number;
  preview?: string;
  elapsedMs: number;
}

// ---------------------------------------------------------------------------
// SseEvent — extended to carry structured PipelineEvent fields (Stage 6)
// ---------------------------------------------------------------------------

interface SseEvent {
  // Legacy fields (always present)
  phase: string;
  detail: string;
  elapsed_ms: number;
  payload?: ChatApiResponse;
  // Structured PipelineEvent fields (present on routing/graph/heartbeat events)
  phase_name?: string;
  substep?: string;
  state?: 'running' | 'done' | 'error';
  visibility?: 'public' | 'internal';
  message?: string;
  metadata?: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// ThinkingEvent — internal state model for the ThinkingPanel
// ---------------------------------------------------------------------------

type ThinkingEventKind =
  | 'routing'
  | 'heartbeat'
  | 'graph_routing'
  | 'graph_expert_init'
  | 'graph_coverage_report';

interface ThinkingEvent {
  id: number;
  kind: ThinkingEventKind;
  state: 'running' | 'done' | 'error';
  message: string;
  detail: string;
  metadata: Record<string, unknown>;
  ts: number; // Date.now() at arrival
}

// Set of phase_name values that belong to the ThinkingPanel
const THINKING_PHASES = new Set<string>([
  'routing',
  'heartbeat',
  'graph_routing',
  'graph_expert_init',
  'graph_coverage_report',
]);

// ---------------------------------------------------------------------------
// Typewriter texts
// ---------------------------------------------------------------------------

const TYPEWRITER_TEXTS = [
  "Truth isn't assumed. It's earned.",
  'An AI that admits when it doesn\'t know is more powerful than one that pretends it does.',
  'Facts and values are different things. We treat them that way.',
  'Not built to impress. Built to be honest.',
  'Bias enters when we pretend values are facts. We don\'t pretend.',
  'Modular by design. Honest by principle.',
  'The no-bullshit promise: find truth where it exists, admit when it doesn\'t.',
  'Intelligence distributed like mycelium — resilient, adaptive, no single point of failure.',
];

const PROMPT_SUGGESTIONS = [
  'Does coffee cause cancer?',
  'Is string theory scientifically proven?',
  'What are the effects of universal basic income?',
  'How does CRISPR gene editing work?',
];

// ---------------------------------------------------------------------------
// useTypewriter hook
// ---------------------------------------------------------------------------

function useTypewriter(texts: string[], typingSpeed = 68, deletingSpeed = 32, pauseMs = 2400) {
  const [displayed, setDisplayed] = useState('');
  const [textIdx, setTextIdx] = useState(0);
  const [charIdx, setCharIdx] = useState(0);
  const [deleting, setDeleting] = useState(false);
  const [paused, setPaused] = useState(false);

  useEffect(() => {
    if (paused) {
      const t = setTimeout(() => { setPaused(false); setDeleting(true); }, pauseMs);
      return () => clearTimeout(t);
    }
    const current = texts[textIdx];
    if (!deleting) {
      if (charIdx < current.length) {
        const t = setTimeout(() => { setDisplayed(current.slice(0, charIdx + 1)); setCharIdx((c) => c + 1); }, typingSpeed);
        return () => clearTimeout(t);
      } else { setPaused(true); }
    } else {
      if (charIdx > 0) {
        const t = setTimeout(() => { setDisplayed(current.slice(0, charIdx - 1)); setCharIdx((c) => c - 1); }, deletingSpeed);
        return () => clearTimeout(t);
      } else { setDeleting(false); setTextIdx((i) => (i + 1) % texts.length); }
    }
  }, [charIdx, deleting, paused, textIdx, texts, typingSpeed, deletingSpeed, pauseMs]);

  return displayed;
}

// ---------------------------------------------------------------------------
// Phase display config — extended with new semantic-architecture event types
// ---------------------------------------------------------------------------

const PHASE_META: Record<string, { label: string; progress: number }> = {
  setting_up:              { label: 'Setting up environment…',              progress: 5  },
  environment_ready:       { label: 'Environment ready',                    progress: 12 },
  routing:                 { label: 'Running reasoning pipeline…',          progress: 20 },
  graph_routing:           { label: 'Semantic graph routing complete',       progress: 30 },
  graph_expert_init:       { label: 'Initialising expert graph…',           progress: 22 },
  graph_coverage_report:   { label: 'Expert coverage assessed',             progress: 33 },
  heartbeat:               { label: 'Thinking…',                            progress: 25 },
  expert_decision:         { label: 'Expert decision resolved',             progress: 35 },
  sandbox_plan:            { label: 'Planning sandbox tools…',              progress: 45 },
  sandbox_summary:         { label: 'Sandbox complete',                     progress: 80 },
  conversation:            { label: 'Generating answer…',                   progress: 90 },
  done:                    { label: 'Done',                                 progress: 100 },
  error:                   { label: 'Error',                                progress: 100 },
};

function phaseLabel(phase: string): string {
  if (PHASE_META[phase]) return PHASE_META[phase].label;
  if (phase.startsWith('sandbox_tool/')) {
    const rest = phase.replace('sandbox_tool/', '');
    if (rest.endsWith('_ok'))  return `Tool ${rest.replace('_ok', '')} completed ✓`;
    if (rest.endsWith('_err')) return `Tool ${rest.replace('_err', '')} failed ✗`;
    return `Calling tool ${rest}…`;
  }
  return phase;
}

function phaseProgress(phase: string): number {
  if (PHASE_META[phase]) return PHASE_META[phase].progress;
  if (phase.startsWith('sandbox_tool/')) {
    const n = parseInt(phase.replace(/\D/g, '') || '1', 10);
    return Math.min(45 + n * 8, 78);
  }
  return 50;
}

function isSandboxPhase(phase: string): boolean {
  return phase === 'sandbox_plan' || phase.startsWith('sandbox_tool/');
}

// ---------------------------------------------------------------------------
// Parse pipe-separated backend detail strings
// ---------------------------------------------------------------------------

function parseSandboxDetail(phase: string, detail: string): Partial<LiveToolEvent> {
  const parts = detail.split(' | ');
  if (phase === 'sandbox_plan') return { query: parts.slice(1).join(', ') };
  const indexMatch = phase.match(/sandbox_tool\/(\d+)/);
  const toolIndex = indexMatch ? parseInt(indexMatch[1], 10) : 0;
  const toolName = parts[0] ?? '';
  if (phase.endsWith('_ok') || phase.endsWith('_err')) {
    const durationMatch = (parts[1] ?? '').match(/(\d+)ms/);
    const durationMs = durationMatch ? parseInt(durationMatch[1], 10) : undefined;
    const preview = parts[2] ?? undefined;
    return { toolIndex, toolName, durationMs, preview };
  }
  const query = parts[1] ?? '';
  return { toolIndex, toolName, query };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function extractTrace(data: ChatApiResponse): PipelineTrace {
  const t = data?.trace ?? {};
  return {
    layer0_route: t.layer0?.route,
    routing_classification: t.routing?.classification,
    routing_domains: t.routing?.selected_domains ?? [],
    expert_decision_type: t.expert_decision?.decision_type,
    selected_experts: t.expert_decision?.selected_experts ?? [],
    expert_confidence: t.expert_decision?.expert_confidence ?? null,
    validation_result: t.phase3?.validation_decision?.result_class,
    phase_latencies: t.phase3?.phase_latencies_ms ?? {},
    trace_id: t.trace_id,
  };
}

function shortId(id: string) { return id ? id.slice(0, 8) + '…' : ''; }
function fmtDuration(ms: number) { return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms.toFixed(0)}ms`; }

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
    web_search: 'Web Search',
    academic_search: 'Academic',
    knowledge_base: 'Knowledge',
    calculator: 'Calculator',
  };
  return names[tool] ?? tool;
}

// ---------------------------------------------------------------------------
// ThinkingPanel — renders semantic-architecture pipeline events (Stage 6)
//
// Displays routing, heartbeat, graph_routing, graph_expert_init, and
// graph_coverage_report events as a live "thinking" feed inside the loading
// bubble. Each event kind has a distinct icon and colour treatment. Events
// whose state === 'running' show a subtle pulse animation; 'done' events show
// a static checkmark. The panel is only rendered when there is at least one
// event to show, so it never produces an empty box mid-query.
// ---------------------------------------------------------------------------

const THINKING_ICONS: Record<ThinkingEventKind, string> = {
  routing:               '⟁',   // triangle → decision point
  heartbeat:             '◌',   // hollow circle → continuity ping
  graph_routing:         '✦',   // sparkle → resolved decision
  graph_expert_init:     '⬡',   // hexagon → expert node coming online
  graph_coverage_report: '▤',   // grid → coverage matrix
};

const THINKING_KIND_LABEL: Record<ThinkingEventKind, string> = {
  routing:               'Routing',
  heartbeat:             'Thinking',
  graph_routing:         'Graph routing',
  graph_expert_init:     'Expert init',
  graph_coverage_report: 'Coverage',
};

function ThinkingPanel({ events }: { events: ThinkingEvent[] }) {
  if (events.length === 0) return null;

  return (
    <div className={styles.thinkingPanel} aria-label="Reasoning activity" aria-live="polite">
      {events.map((ev) => {
        const isRunning = ev.state === 'running';
        const icon = THINKING_ICONS[ev.kind] ?? '·';
        const kindLabel = THINKING_KIND_LABEL[ev.kind] ?? ev.kind;

        // For heartbeat events show the rotating message, otherwise show the
        // structured message from the backend.
        const displayMsg = ev.message || ev.detail;

        // Pull useful metadata for supplementary display
        const meta = ev.metadata ?? {};
        const primaryDomain = meta.primary_domain as string | undefined;
        const classification = meta.classification as string | undefined;
        const candidateCount = meta.candidate_count as number | undefined;
        const elapsedMs = meta.elapsed_ms as number | undefined;
        const expertCount = meta.expert_count as number | undefined;
        const coveredDomains = meta.covered_domains as string[] | undefined;

        return (
          <div
            key={ev.id}
            className={`${styles.thinkingRow} ${isRunning ? styles.thinkingRowRunning : styles.thinkingRowDone}`}
            aria-label={`${kindLabel}: ${displayMsg}`}
          >
            {/* Icon column */}
            <span
              className={`${styles.thinkingIcon} ${isRunning ? styles.thinkingIconPulse : ''}`}
              aria-hidden="true"
            >
              {icon}
            </span>

            {/* Content column */}
            <div className={styles.thinkingBody}>
              <div className={styles.thinkingTop}>
                <span className={styles.thinkingBadge}>{kindLabel}</span>
                <span className={styles.thinkingMsg}>{displayMsg}</span>
                {!isRunning && elapsedMs !== undefined && (
                  <span className={styles.thinkingElapsed}>{fmtDuration(elapsedMs)}</span>
                )}
              </div>

              {/* Supplementary metadata line */}
              {(primaryDomain || classification || candidateCount !== undefined || expertCount !== undefined) && (
                <p className={styles.thinkingMeta}>
                  {primaryDomain && <span>domain: <strong>{primaryDomain}</strong></span>}
                  {classification && <span> · {classification}</span>}
                  {candidateCount !== undefined && <span> · {candidateCount} candidate{candidateCount !== 1 ? 's' : ''}</span>}
                  {expertCount !== undefined && <span> · {expertCount} expert{expertCount !== 1 ? 's' : ''}</span>}
                  {coveredDomains && coveredDomains.length > 0 && (
                    <span> · covers: {coveredDomains.join(', ')}</span>
                  )}
                </p>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// CalibrationGate
// ---------------------------------------------------------------------------

type CalibrationStatus = 'checking' | 'pending' | 'running' | 'complete' | 'error';

function CalibrationGate({ onReady }: { onReady: () => void }) {
  const [status, setStatus]         = useState<CalibrationStatus>('checking');
  const [progress, setProgress]     = useState(0);
  const [statusText, setStatusText] = useState('Checking expert system…');
  const [errorMsg, setErrorMsg]     = useState<string | null>(null);
  const pollRef  = useRef<ReturnType<typeof setInterval> | null>(null);
  const jobIdRef = useRef<string | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
  }, []);

  const startPolling = useCallback((jobId: string) => {
    stopPolling();
    pollRef.current = setInterval(async () => {
      try {
        const res = await axios.get<{
          job_id: string; status: string; progress: number; error: string | null;
        }>(`${API_BASE}/api/calibrate/status/${jobId}`, { timeout: 8000 });
        const data = res.data;
        setProgress(data.progress ?? 0);
        if (data.status === 'complete') {
          stopPolling();
          setStatus('complete');
          setStatusText('Expert system ready ✓');
          setProgress(100);
          setTimeout(onReady, 600);
        } else if (data.status === 'error') {
          stopPolling();
          setStatus('error');
          setErrorMsg(data.error ?? 'Unknown calibration error');
          setStatusText('Calibration failed');
        } else if (data.status === 'running') {
          setStatus('running');
          setStatusText(`Calibrating expert system… (${data.progress ?? 0}%)`);
        } else {
          setStatusText('Expert calibration queued…');
        }
      } catch {
        setStatusText('Waiting for backend…');
      }
    }, CALIBRATION_POLL_INTERVAL_MS);
  }, [onReady, stopPolling]);

  const kickOffCalibration = useCallback(async () => {
    setStatus('pending');
    setStatusText('Starting expert calibration…');
    setErrorMsg(null);
    try {
      const res = await axios.post<{ job_id: string; status: string }>(
        `${API_BASE}/api/calibrate/start`, {}, { timeout: 10000 },
      );
      const jobId = res.data.job_id;
      jobIdRef.current = jobId;
      setStatusText('Expert calibration started — initialising models…');
      startPolling(jobId);
    } catch (err) {
      const axiosErr = err as AxiosError<{ detail?: string }>;
      setStatus('error');
      setErrorMsg(axiosErr?.response?.data?.detail ?? axiosErr?.message ?? 'Could not reach backend');
      setStatusText('Failed to start calibration');
    }
  }, [startPolling]);

  useEffect(() => {
    let cancelled = false;
    async function init() {
      try {
        const res = await axios.get<{ ready: boolean }>(
          `${API_BASE}/api/calibrate/ready`, { timeout: 5000 },
        );
        if (cancelled) return;
        if (res.data.ready) { onReady(); return; }
      } catch { /* proceed to full calibration */ }
      if (!cancelled) kickOffCalibration();
    }
    init();
    return () => { cancelled = true; stopPolling(); };
  }, [kickOffCalibration, onReady, stopPolling]);

  const isComplete = status === 'complete';
  const isError    = status === 'error';

  return (
    <div className={styles.calibrationGate} role="status" aria-live="polite">
      <div className={styles.calibrationCard}>
        <div className={styles.calibrationLogo} aria-hidden="true">
          <svg width="48" height="48" viewBox="0 0 28 28" fill="none">
            <circle cx="14" cy="14" r="3.5" fill="currentColor" opacity="0.9" />
            <line x1="14" y1="14" x2="4"  y2="6"  stroke="currentColor" strokeWidth="1.2" opacity="0.45" />
            <line x1="14" y1="14" x2="24" y2="6"  stroke="currentColor" strokeWidth="1.2" opacity="0.45" />
            <line x1="14" y1="14" x2="4"  y2="22" stroke="currentColor" strokeWidth="1.2" opacity="0.45" />
            <line x1="14" y1="14" x2="24" y2="22" stroke="currentColor" strokeWidth="1.2" opacity="0.45" />
            <line x1="14" y1="14" x2="14" y2="2"  stroke="currentColor" strokeWidth="1.2" opacity="0.45" />
            <line x1="14" y1="14" x2="14" y2="26" stroke="currentColor" strokeWidth="1.2" opacity="0.45" />
            <circle cx="4"  cy="6"  r="2" fill="currentColor" opacity="0.3" />
            <circle cx="24" cy="6"  r="2" fill="currentColor" opacity="0.3" />
            <circle cx="4"  cy="22" r="2" fill="currentColor" opacity="0.3" />
            <circle cx="24" cy="22" r="2" fill="currentColor" opacity="0.3" />
            <circle cx="14" cy="2"  r="2" fill="currentColor" opacity="0.3" />
            <circle cx="14" cy="26" r="2" fill="currentColor" opacity="0.3" />
          </svg>
        </div>
        <h1 className={styles.calibrationTitle}>Mycelium</h1>
        <p className={styles.calibrationSubtitle}>Setting up expert system</p>
        <div
          className={styles.calibrationBarTrack}
          role="progressbar"
          aria-valuenow={progress}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label="Calibration progress"
        >
          <div
            className={`${styles.calibrationBarFill} ${
              isComplete ? styles.calibrationBarComplete :
              isError    ? styles.calibrationBarError    : ''
            }`}
            style={{ width: `${progress}%` }}
          />
        </div>
        <p className={`${styles.calibrationStatus} ${
          isComplete ? styles.calibrationStatusOk  :
          isError    ? styles.calibrationStatusErr : ''
        }`}>
          {statusText}
        </p>
        {isError && errorMsg && (
          <div className={styles.calibrationError}>
            <p className={styles.calibrationErrorDetail}>{errorMsg}</p>
            <button className={styles.calibrationRetry} onClick={kickOffCalibration} aria-label="Retry calibration">
              Retry
            </button>
          </div>
        )}
        {!isComplete && !isError && status !== 'checking' && (
          <p className={styles.calibrationHint}>
            This only runs once on startup — K-Medoids clustering, OOD detection,
            and probability calibration across all expert domains.
          </p>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// HealthBanner
// ---------------------------------------------------------------------------

function HealthBanner({ onDismiss }: { onDismiss: () => void }) {
  return (
    <div className={styles.healthBanner} role="alert" aria-live="assertive">
      <span className={styles.healthBannerIcon} aria-hidden="true">⚠</span>
      <span className={styles.healthBannerText}>
        Cannot reach backend at <code>{API_BASE}</code>. Is the FastAPI server running?
      </span>
      <button className={styles.healthBannerDismiss} onClick={onDismiss} aria-label="Dismiss backend warning">
        <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
          <path d="M2 2l8 8M10 2l-8 8" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
        </svg>
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// ErrorBubble
// ---------------------------------------------------------------------------

function ErrorBubble({ detail, onRetry }: { detail: string; onRetry?: () => void }) {
  return (
    <div className={styles.errorBubble} role="alert">
      <div className={styles.errorBubbleIcon} aria-hidden="true">
        <svg width="15" height="15" viewBox="0 0 16 16" fill="none">
          <circle cx="8" cy="8" r="7" stroke="currentColor" strokeWidth="1.4" />
          <line x1="8" y1="4.5" x2="8" y2="9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
          <circle cx="8" cy="11.5" r="0.9" fill="currentColor" />
        </svg>
      </div>
      <div className={styles.errorBubbleBody}>
        <p className={styles.errorBubbleDetail}>{detail}</p>
        {onRetry && (
          <button className={styles.errorBubbleRetry} onClick={onRetry}>Retry</button>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// ChatEmptyState
// ---------------------------------------------------------------------------

function ChatEmptyState({ onSuggestion }: { onSuggestion: (s: string) => void }) {
  return (
    <div className={styles.chatEmpty}>
      <div className={styles.chatEmptyLogo} aria-hidden="true">
        <svg width="40" height="40" viewBox="0 0 28 28" fill="none">
          <circle cx="14" cy="14" r="3.5" fill="currentColor" opacity="0.7" />
          <line x1="14" y1="14" x2="4"  y2="6"  stroke="currentColor" strokeWidth="1.2" opacity="0.35" />
          <line x1="14" y1="14" x2="24" y2="6"  stroke="currentColor" strokeWidth="1.2" opacity="0.35" />
          <line x1="14" y1="14" x2="4"  y2="22" stroke="currentColor" strokeWidth="1.2" opacity="0.35" />
          <line x1="14" y1="14" x2="24" y2="22" stroke="currentColor" strokeWidth="1.2" opacity="0.35" />
          <line x1="14" y1="14" x2="14" y2="2"  stroke="currentColor" strokeWidth="1.2" opacity="0.35" />
          <line x1="14" y1="14" x2="14" y2="26" stroke="currentColor" strokeWidth="1.2" opacity="0.35" />
          <circle cx="4"  cy="6"  r="2" fill="currentColor" opacity="0.25" />
          <circle cx="24" cy="6"  r="2" fill="currentColor" opacity="0.25" />
          <circle cx="4"  cy="22" r="2" fill="currentColor" opacity="0.25" />
          <circle cx="24" cy="22" r="2" fill="currentColor" opacity="0.25" />
          <circle cx="14" cy="2"  r="2" fill="currentColor" opacity="0.25" />
          <circle cx="14" cy="26" r="2" fill="currentColor" opacity="0.25" />
        </svg>
      </div>
      <p className={styles.chatEmptyHeading}>Ask Mycelium anything</p>
      <p className={styles.chatEmptyHint}>The reasoning pipeline routes your query through Layer 0, expert selection, evidence grounding, and synthesis.</p>
      <div className={styles.chatEmptySuggestions}>
        {PROMPT_SUGGESTIONS.map((s) => (
          <button key={s} className={styles.chatEmptySuggestion} onClick={() => onSuggestion(s)}>{s}</button>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// PhaseIndicator
// ---------------------------------------------------------------------------

function PhaseIndicator({ phase, detail, elapsedMs }: { phase: string; detail: string; elapsedMs: number }) {
  const label    = phaseLabel(phase);
  const progress = phaseProgress(phase);
  const secs     = (elapsedMs / 1000).toFixed(1);
  return (
    <div className={styles.phaseIndicator} aria-live="polite" aria-atomic="true" aria-label={`Pipeline phase: ${label}`}>
      <div className={styles.phaseHeader}>
        <span className={styles.phaseLabel}>{label}</span>
        <span className={styles.phaseElapsed} aria-hidden="true">{secs}s</span>
      </div>
      {detail && <p className={styles.phaseDetail}>{detail}</p>}
      <div className={styles.phaseBarTrack} role="progressbar" aria-valuenow={progress} aria-valuemin={0} aria-valuemax={100} aria-label="Pipeline progress">
        <div className={styles.phaseBarFill} style={{ width: `${progress}%` }} />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// TracePanel
// ---------------------------------------------------------------------------

function TracePanel({ trace }: { trace: PipelineTrace }) {
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
            <span className={`${styles.traceValue} ${trace.expert_decision_type === 'CREATE_NEW_PATCH' ? styles.traceValuePatch : ''}`}>
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
            <span className={styles.traceValue}>{(Number(trace.expert_confidence) * 100).toFixed(1)}%</span>
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
            <span className={styles.traceLabel}>Phase latencies (ms)</span>
            <span className={styles.traceValue}>
              {Object.entries(latencies).map(([k, v]) => `${k}: ${v}`).join(' · ')}
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

// ---------------------------------------------------------------------------
// LiveToolFeed
// ---------------------------------------------------------------------------

function LiveToolFeed({ events, planDetail }: { events: LiveToolEvent[]; planDetail: string }) {
  return (
    <div className={styles.liveToolFeed} aria-label="Live tool activity" aria-live="polite">
      {planDetail && (
        <div className={`${styles.liveToolRow} ${styles.liveToolPlan}`}>
          <span className={styles.liveToolIcon} aria-hidden="true">📋</span>
          <span className={styles.liveToolDetail}>Plan: {planDetail}</span>
        </div>
      )}
      {events.map((ev) => {
        const isRunning = ev.phase === 'running';
        const isOk      = ev.phase === 'ok';
        const isErr     = ev.phase === 'err';
        const rowClass  = isOk ? styles.liveToolOk : isErr ? styles.liveToolErr : styles.liveToolRunning;
        const icon      = isOk ? '✓' : isErr ? '✗' : '⟳';
        const statusLabel = isOk ? 'completed' : isErr ? 'failed' : 'running';
        return (
          <div
            key={ev.toolIndex}
            className={`${styles.liveToolRow} ${rowClass}`}
            aria-label={`${toolDisplayName(ev.toolName)}: ${ev.query} — ${statusLabel}`}
          >
            <span className={`${styles.liveToolIcon} ${isRunning ? styles.liveToolSpinning : ''}`} aria-hidden="true">
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

function SandboxPanel({
  sandbox, liveEvents, planDetail, isLoading,
}: {
  sandbox: SandboxResult | null;
  liveEvents: LiveToolEvent[];
  planDetail: string;
  isLoading: boolean;
}) {
  const [openSteps, setOpenSteps] = useState<Set<number>>(new Set());
  function toggleStep(i: number) {
    setOpenSteps((prev) => { const next = new Set(prev); next.has(i) ? next.delete(i) : next.add(i); return next; });
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
    return <div className={styles.sandboxEmpty}><p>Send a message to see sandbox evidence here.</p></div>;
  }

  const okSteps  = sandbox.steps.filter((s) => s.status === 'ok');
  const errSteps = sandbox.steps.filter((s) => s.status !== 'ok');

  return (
    <div className={styles.sandboxPanel}>
      <div className={styles.sandboxStats}>
        <span className={styles.sandboxStatOk}>{okSteps.length} ok</span>
        {errSteps.length > 0 && <span className={styles.sandboxStatErr}>{errSteps.length} failed</span>}
        <span className={styles.sandboxStatId}>{shortId(sandbox.trace_id)}</span>
      </div>
      {sandbox.summary && <p className={styles.sandboxSummary}>{sandbox.summary}</p>}
      <div className={styles.sandboxSteps}>
        {sandbox.steps.map((step, i) => (
          <div key={i} className={`${styles.sandboxStep} ${step.status === 'ok' ? styles.sandboxStepOk : styles.sandboxStepErr}`}>
            <button
              className={styles.sandboxStepHeader}
              onClick={() => toggleStep(i)}
              aria-expanded={openSteps.has(i)}
              aria-controls={`sandbox-step-body-${i}`}
            >
              <span className={styles.sandboxToolBadge}>{toolDisplayName(step.tool)}</span>
              <span className={step.status === 'ok' ? styles.sandboxStatusOk : styles.sandboxStatusErr}>{step.status}</span>
              <span className={styles.sandboxStepInput}>{step.input?.query ?? ''}</span>
              <span className={styles.sandboxStepDuration}>{fmtDuration(step.duration_ms)}</span>
              <svg width="10" height="10" viewBox="0 0 10 10" fill="none" aria-hidden="true"
                style={{ transform: openSteps.has(i) ? 'rotate(90deg)' : 'rotate(0)', transition: 'transform 150ms ease', flexShrink: 0 }}>
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

// ---------------------------------------------------------------------------
// HistoryItem
// ---------------------------------------------------------------------------

function HistoryItem({ item, active, onClick }: { item: HistoryTrace; active: boolean; onClick: () => void }) {
  const ts = item.timestamp ? new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '';
  return (
    <button className={`${styles.historyItem} ${active ? styles.historyItemActive : ''}`} onClick={onClick}>
      <span className={styles.historyQuery}>{item.user_query}</span>
      <span className={styles.historyMeta}>
        {item.run_summary?.expert_decision?.decision_type && (
          <span className={`${styles.historyBadge} ${item.run_summary.expert_decision.decision_type === 'CREATE_NEW_PATCH' ? styles.historyBadgePatch : styles.historyBadgeNormal}`}>
            {item.run_summary.expert_decision.decision_type === 'CREATE_NEW_PATCH' ? 'PATCH' : 'OK'}
          </span>
        )}
        <span className={styles.historyTs}>{ts}</span>
      </span>
    </button>
  );
}

// ---------------------------------------------------------------------------
// HomeScreen
// ---------------------------------------------------------------------------

function HomeScreen({
  input, onInputChange, onSend, onKeyDown, loading,
}: {
  input: string;
  onInputChange: (v: string) => void;
  onSend: () => void;
  onKeyDown: (e: React.KeyboardEvent<HTMLInputElement>) => void;
  loading: boolean;
}) {
  const typed   = useTypewriter(TYPEWRITER_TEXTS);
  const atLimit = input.length >= MAX_INPUT_CHARS;

  return (
    <div className={styles.homeScreen}>
      <div className={styles.homeContent}>
        <div className={styles.homeLogo} aria-hidden="true">
          <svg width="52" height="52" viewBox="0 0 28 28" fill="none">
            <circle cx="14" cy="14" r="3.5" fill="currentColor" opacity="0.9" />
            <line x1="14" y1="14" x2="4"  y2="6"  stroke="currentColor" strokeWidth="1.2" opacity="0.55" />
            <line x1="14" y1="14" x2="24" y2="6"  stroke="currentColor" strokeWidth="1.2" opacity="0.55" />
            <line x1="14" y1="14" x2="4"  y2="22" stroke="currentColor" strokeWidth="1.2" opacity="0.55" />
            <line x1="14" y1="14" x2="24" y2="22" stroke="currentColor" strokeWidth="1.2" opacity="0.55" />
            <line x1="14" y1="14" x2="14" y2="2"  stroke="currentColor" strokeWidth="1.2" opacity="0.55" />
            <line x1="14" y1="14" x2="14" y2="26" stroke="currentColor" strokeWidth="1.2" opacity="0.55" />
            <circle cx="4"  cy="6"  r="2" fill="currentColor" opacity="0.4" />
            <circle cx="24" cy="6"  r="2" fill="currentColor" opacity="0.4" />
            <circle cx="4"  cy="22" r="2" fill="currentColor" opacity="0.4" />
            <circle cx="24" cy="22" r="2" fill="currentColor" opacity="0.4" />
            <circle cx="14" cy="2"  r="2" fill="currentColor" opacity="0.4" />
            <circle cx="14" cy="26" r="2" fill="currentColor" opacity="0.4" />
          </svg>
        </div>
        <h1 className={styles.homeTitle}>Mycelium</h1>
        <div className={styles.homeSubtitle}>
          <span className={styles.homeTyped}>{typed}</span>
          <span className={styles.homeCursor} aria-hidden="true" />
        </div>
        <div className={styles.homeInputWrap}>
          <input
            className={styles.homeInput}
            placeholder="Ask anything…"
            value={input}
            onChange={(e) => onInputChange(e.target.value.slice(0, MAX_INPUT_CHARS))}
            onKeyDown={onKeyDown}
            disabled={loading}
            autoFocus
            maxLength={MAX_INPUT_CHARS}
            aria-label="Ask Mycelium a question"
          />
          <button
            className={styles.homeSendButton}
            onClick={onSend}
            disabled={loading || !input.trim()}
            aria-label="Send message"
          >
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
              <path d="M2 9h14M10 3l6 6-6 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        </div>
        {input.length > MAX_INPUT_CHARS * 0.8 && (
          <p className={`${styles.charCounter} ${atLimit ? styles.charCounterLimit : ''}`} aria-live="polite">
            {input.length} / {MAX_INPUT_CHARS}
          </p>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function Home() {
  const [calibrationDone, setCalibrationDone] = useState(false);
  const [hasStarted, setHasStarted]           = useState(false);
  const [messages, setMessages]               = useState<Message[]>([]);
  const [input, setInput]                     = useState('');
  const [loading, setLoading]                 = useState(false);
  const [activeSandbox, setActiveSandbox]     = useState<SandboxResult | null>(null);

  const [liveToolEvents, setLiveToolEvents] = useState<LiveToolEvent[]>([]);
  const [livePlanDetail, setLivePlanDetail] = useState<string>('');

  // ThinkingPanel state (Stage 6)
  const [thinkingEvents, setThinkingEvents]   = useState<ThinkingEvent[]>([]);
  const thinkingCounterRef                    = useRef<number>(0);

  const [history, setHistory]                       = useState<HistoryTrace[]>([]);
  const [activeHistoryId, setActiveHistoryId]       = useState<string | null>(null);
  const [historySidebarOpen, setHistorySidebarOpen] = useState(true);
  const [mobileSidebarOpen, setMobileSidebarOpen]   = useState(false);

  const [currentPhase, setCurrentPhase]   = useState<string>('routing');
  const [currentDetail, setCurrentDetail] = useState<string>('');
  const [elapsedMs, setElapsedMs]         = useState<number>(0);

  const [backendDown, setBackendDown]                 = useState(false);
  const [healthBannerDismissed, setHealthBannerDismissed] = useState(false);

  const bottomRef      = useRef<HTMLDivElement>(null);
  const esRef          = useRef<EventSource | null>(null);
  const tickRef        = useRef<ReturnType<typeof setInterval> | null>(null);
  const startTimeRef   = useRef<number>(0);
  const lastEventAtRef = useRef<number>(0);
  const sseTimeoutRef  = useRef<ReturnType<typeof setTimeout> | null>(null);
  const sseRetryCount  = useRef<number>(0);
  const pendingRetryText = useRef<string>('');

  // ── Health check ───────────────────────────────────────────────────────────
  useEffect(() => {
    async function checkHealth() {
      try {
        await axios.get(`${API_BASE}/api/v1/health`, { timeout: 4000 });
        setBackendDown(false);
      } catch {
        setBackendDown(true);
      }
    }
    checkHealth();
    const interval = setInterval(checkHealth, 30_000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  useEffect(() => {
    return () => {
      esRef.current?.close();
      if (tickRef.current)       clearInterval(tickRef.current);
      if (sseTimeoutRef.current) clearTimeout(sseTimeoutRef.current);
    };
  }, []);

  const loadHistory = useCallback(async () => {
    try {
      const res = await axios.get<HistoryTrace[]>(`${API_BASE}/api/v1/traces/recent?limit=20`);
      setHistory(res.data ?? []);
    } catch { /* optional */ }
  }, []);

  useEffect(() => { loadHistory(); }, [loadHistory]);

  function toggleTrace(idx: number) {
    setMessages((prev) => prev.map((m, i) => (i === idx ? { ...m, traceOpen: !m.traceOpen } : m)));
  }

  function startElapsedTick() {
    startTimeRef.current = Date.now();
    setElapsedMs(0);
    if (tickRef.current) clearInterval(tickRef.current);
    tickRef.current = setInterval(() => { setElapsedMs(Date.now() - startTimeRef.current); }, 250);
  }

  function stopElapsedTick() {
    if (tickRef.current) { clearInterval(tickRef.current); tickRef.current = null; }
  }

  function clearSseTimeout() {
    if (sseTimeoutRef.current) { clearTimeout(sseTimeoutRef.current); sseTimeoutRef.current = null; }
  }

  function finishWithResponse(data: ChatApiResponse) {
    const answer  = data.answer ?? 'Mycelium returned no answer text. Check the pipeline trace below.';
    const trace   = extractTrace(data);
    const sandbox = data.sandbox ?? null;
    setMessages((prev) => [...prev, { role: 'assistant', content: answer, trace, traceOpen: false, sandbox }]);
    if (sandbox) setActiveSandbox(sandbox);
    setLiveToolEvents([]);
    setLivePlanDetail('');
    setThinkingEvents([]);          // clear thinking feed on completion
    setLoading(false);
    stopElapsedTick();
    clearSseTimeout();
    sseRetryCount.current = 0;
    loadHistory();
  }

  function finishWithError(detail: string, retryQuery?: string) {
    setMessages((prev) => [...prev, { role: 'error', content: detail, retryQuery }]);
    setLiveToolEvents([]);
    setLivePlanDetail('');
    setThinkingEvents([]);          // clear thinking feed on error
    setLoading(false);
    stopElapsedTick();
    clearSseTimeout();
    sseRetryCount.current = 0;
  }

  function handleSandboxEvent(phase: string, detail: string, elapsedMsVal: number) {
    if (phase === 'sandbox_plan') {
      const parts = detail.split(' | ');
      setLivePlanDetail(parts.slice(1).join(', ') || detail);
      return;
    }
    const parsed = parseSandboxDetail(phase, detail);
    const { toolIndex, toolName, query, durationMs, preview } = parsed as Required<typeof parsed>;
    if (!toolIndex) return;
    if (phase.endsWith('_ok') || phase.endsWith('_err')) {
      setLiveToolEvents((prev) =>
        prev.map((ev) =>
          ev.toolIndex === toolIndex
            ? { ...ev, phase: phase.endsWith('_ok') ? 'ok' : 'err', durationMs, preview, elapsedMs: elapsedMsVal }
            : ev
        )
      );
    } else {
      const newRow: LiveToolEvent = { toolIndex, toolName: toolName || '', query: query || '', phase: 'running', elapsedMs: elapsedMsVal };
      setLiveToolEvents((prev) => {
        const exists = prev.find((e) => e.toolIndex === toolIndex);
        if (exists) return prev.map((e) => e.toolIndex === toolIndex ? newRow : e);
        return [...prev, newRow];
      });
    }
  }

  // ---------------------------------------------------------------------------
  // handleThinkingEvent — routes structured PipelineEvents into ThinkingPanel
  //
  // Called when an SSE frame carries a phase_name that belongs to the
  // THINKING_PHASES set. The function either appends a new ThinkingEvent or
  // updates an existing one in-place (matched by kind + substep) so that a
  // running → done transition animates smoothly rather than producing a
  // duplicate row. Heartbeat events are always appended (never deduplicated)
  // because they represent distinct continuity pings.
  // ---------------------------------------------------------------------------

  function handleThinkingEvent(event: SseEvent) {
    const kind = (event.phase_name ?? event.phase) as ThinkingEventKind;
    const state = event.state ?? (event.phase === 'done' ? 'done' : 'running');
    const message = event.message ?? event.detail ?? '';
    const detail  = event.detail ?? '';
    const metadata = event.metadata ?? {};

    if (kind === 'heartbeat') {
      // Always append heartbeats — they are distinct continuity pings
      thinkingCounterRef.current += 1;
      const newEv: ThinkingEvent = {
        id: thinkingCounterRef.current,
        kind,
        state,
        message,
        detail,
        metadata,
        ts: Date.now(),
      };
      setThinkingEvents((prev) => [...prev, newEv]);
      return;
    }

    // For all other kinds: upsert by (kind, substep) so running → done
    // transitions update the existing row instead of duplicating it.
    const substep = event.substep ?? '';
    setThinkingEvents((prev) => {
      const existingIdx = prev.findIndex(
        (e) => e.kind === kind && (e.metadata?.substep ?? '') === substep
      );
      if (existingIdx !== -1) {
        // Update existing row in-place
        const updated = [...prev];
        updated[existingIdx] = {
          ...updated[existingIdx],
          state,
          message,
          detail,
          metadata: { ...metadata, substep },
        };
        return updated;
      }
      // New event — append
      thinkingCounterRef.current += 1;
      return [
        ...prev,
        {
          id: thinkingCounterRef.current,
          kind,
          state,
          message,
          detail,
          metadata: { ...metadata, substep },
          ts: Date.now(),
        },
      ];
    });
  }

  function openSseStream(text: string) {
    const sseUrl = `${API_BASE}/api/v1/chat/stream?text=${encodeURIComponent(text)}`;
    const es = new EventSource(sseUrl);
    esRef.current = es;
    let gotDone = false;

    const SSE_TIMEOUT_MS = 3 * 60 * 1000;
    clearSseTimeout();
    sseTimeoutRef.current = setTimeout(() => {
      if (!gotDone) { es.close(); esRef.current = null; finishWithError('Request timed out after 3 minutes.', text); }
    }, SSE_TIMEOUT_MS);

    es.onmessage = (ev) => {
      lastEventAtRef.current = Date.now();
      sseRetryCount.current = 0;
      try {
        const event: SseEvent = JSON.parse(ev.data);
        setCurrentPhase(event.phase);
        setCurrentDetail(event.detail);
        setElapsedMs(event.elapsed_ms);

        // Route to ThinkingPanel if this is a structured pipeline event
        const phaseName = event.phase_name ?? event.phase;
        if (THINKING_PHASES.has(phaseName) && event.visibility !== 'internal') {
          handleThinkingEvent(event);
        }

        if (isSandboxPhase(event.phase)) handleSandboxEvent(event.phase, event.detail, event.elapsed_ms);

        if (event.phase === 'done' && event.payload) {
          gotDone = true; es.close(); esRef.current = null; finishWithResponse(event.payload);
        } else if (event.phase === 'error') {
          gotDone = true; es.close(); esRef.current = null;
          finishWithError(event.detail || 'Unknown SSE error', text);
        }
      } catch { /* malformed SSE frame */ }
    };

    es.onerror = () => {
      if (gotDone) return;
      const silentForMs = Date.now() - lastEventAtRef.current;
      const neverReceived = lastEventAtRef.current === 0;
      if (!neverReceived && silentForMs < 20_000) return;
      es.close(); esRef.current = null;
      if (sseRetryCount.current < SSE_RETRY_ATTEMPTS) {
        sseRetryCount.current += 1;
        setCurrentDetail(`SSE disconnected — retrying (${sseRetryCount.current}/${SSE_RETRY_ATTEMPTS})…`);
        setTimeout(() => openSseStream(text), SSE_RETRY_DELAY_MS * sseRetryCount.current);
      } else {
        clearSseTimeout();
        setCurrentDetail('SSE unavailable — using fallback POST…');
        fallbackPost(text);
      }
    };
  }

  async function handleSend(overrideText?: string) {
    const text = (typeof overrideText === 'string' ? overrideText : input).trim();
    if (!text || loading) return;
    if (!hasStarted) setHasStarted(true);
    setMessages((prev) => [...prev, { role: 'user', content: text }]);
    if (!overrideText) setInput('');
    setLoading(true);
    setLiveToolEvents([]);
    setLivePlanDetail('');
    setThinkingEvents([]);          // reset thinking feed for new query
    setActiveSandbox(null);
    setCurrentPhase('routing');
    setCurrentDetail('Connecting to Mycelium…');
    startElapsedTick();
    lastEventAtRef.current = 0;
    sseRetryCount.current = 0;
    pendingRetryText.current = text;
    openSseStream(text);
  }

  async function fallbackPost(text: string) {
    setCurrentPhase('routing');
    setCurrentDetail('SSE unavailable — using fallback POST…');
    try {
      const res = await axios.post<ChatApiResponse>(`${API_BASE}/api/v1/chat`, { text });
      finishWithResponse(res.data);
    } catch (err) {
      const axiosErr = err as AxiosError<{ detail?: string }>;
      finishWithError(axiosErr?.response?.data?.detail ?? axiosErr?.message ?? 'Unknown error', text);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); }
  }

  function handleHistoryClick(item: HistoryTrace) {
    setActiveHistoryId(item.trace_id);
    const sr = item.sandbox_result as SandboxResult | undefined;
    if (sr) setActiveSandbox(sr);
  }

  const hasLiveActivity   = liveToolEvents.length > 0 || !!livePlanDetail;
  const sandboxPaneTitle  = loading && hasLiveActivity ? 'Running tools…' : loading ? 'Sandbox' : 'Sandbox evidence';
  const atCharLimit       = input.length >= MAX_INPUT_CHARS;
  const charCounterVisible = input.length > MAX_INPUT_CHARS * 0.8;

  // ── Render: calibration gate ───────────────────────────────────────────────
  if (!calibrationDone) {
    return (
      <>
        <Head>
          <title>Mycelium — Starting up</title>
          <meta name="viewport" content="width=device-width, initial-scale=1" />
        </Head>
        <CalibrationGate onReady={() => setCalibrationDone(true)} />
      </>
    );
  }

  // ── Render: home screen ────────────────────────────────────────────────────
  if (!hasStarted) {
    return (
      <>
        <Head>
          <title>Mycelium</title>
          <meta name="viewport" content="width=device-width, initial-scale=1" />
        </Head>
        <HomeScreen
          input={input}
          onInputChange={setInput}
          onSend={handleSend}
          onKeyDown={handleKeyDown}
          loading={loading}
        />
      </>
    );
  }

  // ── Render: chat UI ────────────────────────────────────────────────────────
  return (
    <div className={`${styles.shell} ${styles.shellVisible}`}>
      <Head>
        <title>Mycelium Console</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      {backendDown && !healthBannerDismissed && (
        <HealthBanner onDismiss={() => setHealthBannerDismissed(true)} />
      )}

      {mobileSidebarOpen && (
        <div
          className={styles.mobileSidebarBackdrop}
          onClick={() => setMobileSidebarOpen(false)}
          aria-hidden="true"
        />
      )}

      <header className={styles.header}>
        <div className={styles.headerLeft}>
          <button
            className={styles.sidebarToggle}
            onClick={() => {
              if (window.innerWidth <= 768) {
                setMobileSidebarOpen((v) => !v);
              } else {
                setHistorySidebarOpen((v) => !v);
              }
            }}
            aria-label="Toggle history panel"
            aria-expanded={historySidebarOpen || mobileSidebarOpen}
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
              <rect x="1" y="3"    width="14" height="1.5" rx="0.75" fill="currentColor" />
              <rect x="1" y="7.25" width="10" height="1.5" rx="0.75" fill="currentColor" />
              <rect x="1" y="11.5" width="12" height="1.5" rx="0.75" fill="currentColor" />
            </svg>
          </button>
          <div className={styles.headerLogo} aria-label="Mycelium">
            <svg width="26" height="26" viewBox="0 0 28 28" fill="none" aria-hidden="true">
              <circle cx="14" cy="14" r="3.5" fill="currentColor" opacity="0.9" />
              <line x1="14" y1="14" x2="4"  y2="6"  stroke="currentColor" strokeWidth="1.2" opacity="0.5" />
              <line x1="14" y1="14" x2="24" y2="6"  stroke="currentColor" strokeWidth="1.2" opacity="0.5" />
              <line x1="14" y1="14" x2="4"  y2="22" stroke="currentColor" strokeWidth="1.2" opacity="0.5" />
              <line x1="14" y1="14" x2="24" y2="22" stroke="currentColor" strokeWidth="1.2" opacity="0.5" />
              <line x1="14" y1="14" x2="14" y2="2"  stroke="currentColor" strokeWidth="1.2" opacity="0.5" />
              <line x1="14" y1="14" x2="14" y2="26" stroke="currentColor" strokeWidth="1.2" opacity="0.5" />
              <circle cx="4"  cy="6"  r="2" fill="currentColor" opacity="0.4" />
              <circle cx="24" cy="6"  r="2" fill="currentColor" opacity="0.4" />
              <circle cx="4"  cy="22" r="2" fill="currentColor" opacity="0.4" />
              <circle cx="24" cy="22" r="2" fill="currentColor" opacity="0.4" />
              <circle cx="14" cy="2"  r="2" fill="currentColor" opacity="0.4" />
              <circle cx="14" cy="26" r="2" fill="currentColor" opacity="0.4" />
            </svg>
            <span className={styles.headerTitle}>Mycelium</span>
          </div>
          <span className={styles.headerSubtitle} aria-hidden="true">
            Layer 0 · Routing · Experts · Validation · Sandbox · Synthesis
          </span>
        </div>
      </header>

      <div className={styles.body}>
        {(historySidebarOpen || mobileSidebarOpen) && (
          <aside
            className={`${styles.historySidebar} ${mobileSidebarOpen ? styles.historySidebarMobile : ''}`}
            aria-label="Query history"
          >
            <div className={styles.sidebarHeader}>
              <span className={styles.sidebarTitle}>Recent traces</span>
              <button className={styles.sidebarRefresh} onClick={loadHistory} aria-label="Refresh history">
                <svg width="13" height="13" viewBox="0 0 14 14" fill="none" aria-hidden="true">
                  <path d="M12 7A5 5 0 1 1 7 2v0l-1.5-1.5M7 2l1.5-1.5L7 2z" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </button>
            </div>
            <div className={styles.historyList}>
              {history.length === 0 && <p className={styles.historyEmpty}>No traces yet.</p>}
              {history.map((item) => (
                <HistoryItem
                  key={item.trace_id}
                  item={item}
                  active={activeHistoryId === item.trace_id}
                  onClick={() => { handleHistoryClick(item); setMobileSidebarOpen(false); }}
                />
              ))}
            </div>
          </aside>
        )}

        <main className={styles.chatPane} id="main-content">
          <div className={styles.chatWindow} role="log" aria-live="polite" aria-label="Conversation">
            {messages.length === 0 && !loading && (
              <ChatEmptyState onSuggestion={(s) => { setInput(s); handleSend(s); }} />
            )}

            {messages.map((m, idx) => {
              if (m.role === 'error') {
                return (
                  <div key={idx} className={styles.assistantBubbleWrap}>
                    <ErrorBubble
                      detail={m.content}
                      onRetry={m.retryQuery ? () => handleSend(m.retryQuery) : undefined}
                    />
                  </div>
                );
              }
              return (
                <div key={idx} className={m.role === 'user' ? styles.userBubbleWrap : styles.assistantBubbleWrap}>
                  <div className={m.role === 'user' ? styles.userBubble : styles.assistantBubble}>
                    <pre className={styles.bubbleText}>{m.content}</pre>
                  </div>
                  {m.role === 'assistant' && m.trace && (
                    <>
                      <div className={styles.traceToggleRow}>
                        <button
                          className={styles.traceToggleBtn}
                          onClick={() => toggleTrace(idx)}
                          aria-expanded={m.traceOpen}
                          aria-controls={`trace-panel-${idx}`}
                        >
                          <svg width="11" height="11" viewBox="0 0 12 12" fill="none" aria-hidden="true"
                            style={{ transform: m.traceOpen ? 'rotate(90deg)' : 'rotate(0deg)', transition: 'transform 180ms ease' }}>
                            <path d="M4 2l4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                          </svg>
                          Pipeline trace
                        </button>
                        {m.sandbox && (
                          <button
                            className={styles.traceToggleBtn}
                            onClick={() => setActiveSandbox(m.sandbox ?? null)}
                            aria-label="Show sandbox evidence in right panel"
                          >
                            <svg width="11" height="11" viewBox="0 0 12 12" fill="none" aria-hidden="true">
                              <circle cx="6" cy="6" r="4.5" stroke="currentColor" strokeWidth="1.4" />
                              <line x1="6" y1="3" x2="6" y2="6.5" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
                              <circle cx="6" cy="8.5" r="0.75" fill="currentColor" />
                            </svg>
                            Evidence
                          </button>
                        )}
                      </div>
                      {m.traceOpen && <div id={`trace-panel-${idx}`}><TracePanel trace={m.trace} /></div>}
                    </>
                  )}
                </div>
              );
            })}

            {loading && (
              <div className={styles.assistantBubbleWrap}>
                <div className={styles.assistantBubble}>
                  {/* ThinkingPanel sits above PhaseIndicator when events exist */}
                  <ThinkingPanel events={thinkingEvents} />
                  <PhaseIndicator phase={currentPhase} detail={currentDetail} elapsedMs={elapsedMs} />
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          <div className={styles.inputRow}>
            <div className={styles.inputWrap}>
              <input
                className={styles.input}
                placeholder="Ask Mycelium something…"
                value={input}
                onChange={(e) => setInput(e.target.value.slice(0, MAX_INPUT_CHARS))}
                onKeyDown={handleKeyDown}
                disabled={loading}
                autoFocus
                maxLength={MAX_INPUT_CHARS}
                aria-label="Type your message"
              />
              {charCounterVisible && (
                <span className={`${styles.charCounter} ${atCharLimit ? styles.charCounterLimit : ''}`} aria-live="polite">
                  {input.length}/{MAX_INPUT_CHARS}
                </span>
              )}
            </div>
            <button
              className={styles.sendButton}
              onClick={() => handleSend()}
              disabled={loading || !input.trim()}
              aria-label="Send message"
            >
              <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
                <path d="M2 9h14M10 3l6 6-6 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>
          </div>
        </main>

        <aside className={styles.sandboxPane} aria-label="Sandbox evidence">
          <div className={styles.sidebarHeader}>
            <span className={styles.sidebarTitle}>{sandboxPaneTitle}</span>
            {loading && hasLiveActivity && <span className={styles.sandboxLiveBadge} aria-label="Live tool activity">LIVE</span>}
          </div>
          <SandboxPanel
            sandbox={activeSandbox}
            liveEvents={liveToolEvents}
            planDetail={livePlanDetail}
            isLoading={loading}
          />
        </aside>
      </div>
    </div>
  );
}
