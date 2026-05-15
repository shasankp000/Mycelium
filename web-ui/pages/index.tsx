import Head from 'next/head';
import { useState, useRef, useEffect, useCallback } from 'react';
import axios, { AxiosError } from 'axios';
import styles from '../styles/Home.module.css';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? 'http://localhost:8000';

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
  role: 'user' | 'assistant';
  content: string;
  trace?: PipelineTrace;
  traceOpen?: boolean;
  sandbox?: SandboxResult;
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

// A live sandbox event streamed before the final SandboxResult is ready
export interface LiveToolEvent {
  phase: string;   // e.g. 'sandbox_tool/1', 'sandbox_tool/1_ok'
  detail: string;
  elapsed_ms: number;
}

// SSE event from /api/v1/chat/stream
interface SseEvent {
  phase: string;
  detail: string;
  elapsed_ms: number;
  payload?: ChatApiResponse;
}

// ---------------------------------------------------------------------------
// Phase display config
// ---------------------------------------------------------------------------

const PHASE_META: Record<string, { label: string; progress: number }> = {
  setting_up:        { label: 'Setting up environment…',         progress: 5  },
  environment_ready: { label: 'Environment ready',               progress: 12 },
  routing:           { label: 'Running reasoning pipeline…',     progress: 20 },
  expert_decision:   { label: 'Expert decision resolved',        progress: 35 },
  sandbox_plan:      { label: 'Planning sandbox tools…',         progress: 45 },
  sandbox_summary:   { label: 'Sandbox complete',                progress: 80 },
  conversation:      { label: 'Generating answer…',              progress: 90 },
  done:              { label: 'Done',                            progress: 100 },
  error:             { label: 'Error',                           progress: 100 },
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

// Does this phase belong to the sandbox tool-call section?
function isSandboxToolPhase(phase: string): boolean {
  return (
    phase === 'sandbox_plan' ||
    phase.startsWith('sandbox_tool/')
  );
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

function shortId(id: string) {
  return id ? id.slice(0, 8) + '…' : '';
}

function fmtDuration(ms: number) {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms.toFixed(0)}ms`;
}

function outputPreview(output: Record<string, unknown>): string {
  if (!output) return '';
  if (typeof output.error === 'string') return `⚠ ${output.error}`;
  if (output.status === 'empty') return '(no results returned)';
  if (Array.isArray(output.results)) {
    const r = output.results as Array<Record<string, string>>;
    return r
      .slice(0, 2)
      .map((x) => x.snippet ?? x.abstract ?? x.title ?? '')
      .filter(Boolean)
      .join(' · ')
      .slice(0, 200);
  }
  if (Array.isArray(output.papers)) {
    const p = output.papers as Array<Record<string, unknown>>;
    return p
      .slice(0, 2)
      .map((x) => `${x.title ?? ''} (${x.year ?? '?'})`)
      .join('; ')
      .slice(0, 200);
  }
  if (Array.isArray(output.entities)) {
    const e = output.entities as Array<Record<string, string>>;
    return e
      .slice(0, 2)
      .map((x) => `${x.label}: ${x.description ?? ''}`)
      .join('; ')
      .slice(0, 200);
  }
  if (output.result !== undefined) return String(output.result);
  return JSON.stringify(output).slice(0, 200);
}

// Derive a human tool name from the tool string
function toolDisplayName(tool: string): string {
  const names: Record<string, string> = {
    web_search: 'Web Search',
    academic_search: 'Academic Search',
    knowledge_base: 'Knowledge Base',
    calculator: 'Calculator',
  };
  return names[tool] ?? tool;
}

// ---------------------------------------------------------------------------
// PhaseIndicator
// ---------------------------------------------------------------------------

function PhaseIndicator({
  phase,
  detail,
  elapsedMs,
}: {
  phase: string;
  detail: string;
  elapsedMs: number;
}) {
  const label = phaseLabel(phase);
  const progress = phaseProgress(phase);
  const secs = (elapsedMs / 1000).toFixed(1);

  return (
    <div className={styles.phaseIndicator}>
      <div className={styles.phaseHeader}>
        <span className={styles.phaseLabel}>{label}</span>
        <span className={styles.phaseElapsed}>{secs}s</span>
      </div>
      {detail && <p className={styles.phaseDetail}>{detail}</p>}
      <div className={styles.phaseBarTrack}>
        <div
          className={styles.phaseBarFill}
          style={{ width: `${progress}%` }}
        />
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
            <span className={styles.traceValue}>
              {(trace.routing_domains ?? []).join(', ')}
            </span>
          </div>
        )}
        {trace.expert_decision_type && (
          <div className={styles.traceCell}>
            <span className={styles.traceLabel}>Expert decision</span>
            <span
              className={`${styles.traceValue} ${
                trace.expert_decision_type === 'CREATE_NEW_PATCH'
                  ? styles.traceValuePatch
                  : ''
              }`}
            >
              {trace.expert_decision_type}
            </span>
          </div>
        )}
        {(trace.selected_experts ?? []).length > 0 && (
          <div className={styles.traceCell}>
            <span className={styles.traceLabel}>Experts used</span>
            <span className={styles.traceValue}>
              {(trace.selected_experts ?? []).join(', ')}
            </span>
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
            <span className={styles.traceLabel}>Phase latencies (ms)</span>
            <span className={styles.traceValue}>
              {Object.entries(latencies)
                .map(([k, v]) => `${k}: ${v}`)
                .join(' · ')}
            </span>
          </div>
        )}
        {trace.trace_id && (
          <div className={`${styles.traceCell} ${styles.traceCellFull}`}>
            <span className={styles.traceLabel}>Trace ID</span>
            <span className={`${styles.traceValue} ${styles.traceId}`}>
              {trace.trace_id}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// LiveToolFeed — shown in the sandbox pane while tools are running
// ---------------------------------------------------------------------------

function LiveToolFeed({ events }: { events: LiveToolEvent[] }) {
  if (events.length === 0) return null;
  return (
    <div className={styles.liveToolFeed}>
      {events.map((ev, i) => {
        const isOk  = ev.phase.endsWith('_ok');
        const isErr = ev.phase.endsWith('_err');
        const isPlan = ev.phase === 'sandbox_plan';
        const isStart = !isOk && !isErr && !isPlan;
        return (
          <div
            key={i}
            className={`${styles.liveToolRow} ${
              isOk  ? styles.liveToolOk  :
              isErr ? styles.liveToolErr :
              isPlan ? styles.liveToolPlan :
              styles.liveToolRunning
            }`}
          >
            <span className={styles.liveToolIcon}>
              {isOk ? '✓' : isErr ? '✗' : isPlan ? '📋' : '⟳'}
            </span>
            <span className={styles.liveToolDetail}>{ev.detail}</span>
            <span className={styles.liveToolElapsed}>
              {(ev.elapsed_ms / 1000).toFixed(1)}s
            </span>
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
  sandbox,
  liveEvents,
  isLoading,
}: {
  sandbox: SandboxResult | null;
  liveEvents: LiveToolEvent[];
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

  // While loading, show the live feed
  if (isLoading) {
    return (
      <div className={styles.sandboxPanel}>
        {liveEvents.length === 0 ? (
          <div className={styles.sandboxEmpty}>
            <div className={styles.sandboxWaiting}>
              <span className={styles.sandboxDot} />
              <span className={styles.sandboxDot} />
              <span className={styles.sandboxDot} />
            </div>
            <p>Waiting for tool plan…</p>
          </div>
        ) : (
          <LiveToolFeed events={liveEvents} />
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

  const okSteps = sandbox.steps.filter((s) => s.status === 'ok');
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

      {sandbox.summary && (
        <p className={styles.sandboxSummary}>{sandbox.summary}</p>
      )}

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
            >
              <span className={styles.sandboxToolBadge}>
                {toolDisplayName(step.tool)}
              </span>
              <span
                className={
                  step.status === 'ok'
                    ? styles.sandboxStatusOk
                    : styles.sandboxStatusErr
                }
              >
                {step.status}
              </span>
              <span className={styles.sandboxStepInput}>
                {step.input?.query ?? ''}
              </span>
              <span className={styles.sandboxStepDuration}>
                {fmtDuration(step.duration_ms)}
              </span>
              <svg
                width="10"
                height="10"
                viewBox="0 0 10 10"
                fill="none"
                style={{
                  transform: openSteps.has(i) ? 'rotate(90deg)' : 'rotate(0)',
                  transition: 'transform 150ms ease',
                  flexShrink: 0,
                }}
              >
                <path
                  d="M3 2l4 3-4 3"
                  stroke="currentColor"
                  strokeWidth="1.4"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </button>

            {openSteps.has(i) && (
              <div className={styles.sandboxStepBody}>
                {step.commentary && (
                  <p className={styles.sandboxCommentary}>{step.commentary}</p>
                )}
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

function HistoryItem({
  item,
  active,
  onClick,
}: {
  item: HistoryTrace;
  active: boolean;
  onClick: () => void;
}) {
  const ts = item.timestamp
    ? new Date(item.timestamp).toLocaleTimeString([], {
        hour: '2-digit',
        minute: '2-digit',
      })
    : '';
  return (
    <button
      className={`${styles.historyItem} ${active ? styles.historyItemActive : ''}`}
      onClick={onClick}
    >
      <span className={styles.historyQuery}>{item.user_query}</span>
      <span className={styles.historyMeta}>
        {item.run_summary?.expert_decision?.decision_type && (
          <span
            className={`${styles.historyBadge} ${
              item.run_summary.expert_decision.decision_type === 'CREATE_NEW_PATCH'
                ? styles.historyBadgePatch
                : styles.historyBadgeNormal
            }`}
          >
            {item.run_summary.expert_decision.decision_type === 'CREATE_NEW_PATCH'
              ? 'PATCH'
              : 'OK'}
          </span>
        )}
        <span className={styles.historyTs}>{ts}</span>
      </span>
    </button>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [activeSandbox, setActiveSandbox] = useState<SandboxResult | null>(null);
  const [liveToolEvents, setLiveToolEvents] = useState<LiveToolEvent[]>([]);
  const [history, setHistory] = useState<HistoryTrace[]>([]);
  const [activeHistoryId, setActiveHistoryId] = useState<string | null>(null);
  const [historySidebarOpen, setHistorySidebarOpen] = useState(true);

  // Live phase state for the PhaseIndicator in the chat pane
  const [currentPhase, setCurrentPhase] = useState<string>('routing');
  const [currentDetail, setCurrentDetail] = useState<string>('');
  const [elapsedMs, setElapsedMs] = useState<number>(0);

  const bottomRef = useRef<HTMLDivElement>(null);
  const esRef = useRef<EventSource | null>(null);
  const tickRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startTimeRef = useRef<number>(0);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  useEffect(() => {
    return () => {
      esRef.current?.close();
      if (tickRef.current) clearInterval(tickRef.current);
    };
  }, []);

  const loadHistory = useCallback(async () => {
    try {
      const res = await axios.get<HistoryTrace[]>(
        `${API_BASE}/api/v1/traces/recent?limit=20`,
      );
      setHistory(res.data ?? []);
    } catch {
      // History is optional
    }
  }, []);

  useEffect(() => { loadHistory(); }, [loadHistory]);

  function toggleTrace(idx: number) {
    setMessages((prev) =>
      prev.map((m, i) => (i === idx ? { ...m, traceOpen: !m.traceOpen } : m)),
    );
  }

  function startElapsedTick() {
    startTimeRef.current = Date.now();
    setElapsedMs(0);
    if (tickRef.current) clearInterval(tickRef.current);
    tickRef.current = setInterval(() => {
      setElapsedMs(Date.now() - startTimeRef.current);
    }, 250);
  }

  function stopElapsedTick() {
    if (tickRef.current) {
      clearInterval(tickRef.current);
      tickRef.current = null;
    }
  }

  function finishWithResponse(data: ChatApiResponse) {
    const answer: string =
      data.answer ??
      'Mycelium returned no answer text. Check the pipeline trace below.';
    const trace = extractTrace(data);
    const sandbox = data.sandbox ?? null;

    setMessages((prev) => [
      ...prev,
      { role: 'assistant', content: answer, trace, traceOpen: false, sandbox },
    ]);
    if (sandbox) setActiveSandbox(sandbox);
    setLiveToolEvents([]);  // clear live feed once we have the real result
    setLoading(false);
    stopElapsedTick();
    loadHistory();
  }

  function finishWithError(detail: string) {
    setMessages((prev) => [
      ...prev,
      {
        role: 'assistant',
        content: `⚠ Error contacting Mycelium backend: ${detail}\n\nIs the FastAPI server running at ${API_BASE}?`,
      },
    ]);
    setLiveToolEvents([]);
    setLoading(false);
    stopElapsedTick();
  }

  async function handleSend() {
    if (!input.trim() || loading) return;
    const text = input.trim();
    setMessages((prev) => [...prev, { role: 'user', content: text }]);
    setInput('');
    setLoading(true);
    setLiveToolEvents([]);
    setActiveSandbox(null);
    setCurrentPhase('routing');
    setCurrentDetail('Connecting to Mycelium…');
    startElapsedTick();

    const sseUrl = `${API_BASE}/api/v1/chat/stream?text=${encodeURIComponent(text)}`;

    try {
      const es = new EventSource(sseUrl);
      esRef.current = es;
      let gotDone = false;

      es.onmessage = (ev) => {
        try {
          const event: SseEvent = JSON.parse(ev.data);

          // Always update the phase indicator in the chat pane
          setCurrentPhase(event.phase);
          setCurrentDetail(event.detail);
          setElapsedMs(event.elapsed_ms);

          // If it's a sandbox tool/plan event, also push to the live feed
          if (isSandboxToolPhase(event.phase)) {
            setLiveToolEvents((prev) => [
              ...prev,
              { phase: event.phase, detail: event.detail, elapsed_ms: event.elapsed_ms },
            ]);
          }

          if (event.phase === 'done' && event.payload) {
            gotDone = true;
            es.close();
            esRef.current = null;
            finishWithResponse(event.payload);
          } else if (event.phase === 'error') {
            gotDone = true;
            es.close();
            esRef.current = null;
            finishWithError(event.detail || 'Unknown SSE error');
          }
        } catch {
          // malformed SSE line — ignore
        }
      };

      es.onerror = () => {
        if (gotDone) return;
        es.close();
        esRef.current = null;
        fallbackPost(text);
      };
    } catch {
      fallbackPost(text);
    }
  }

  async function fallbackPost(text: string) {
    setCurrentPhase('routing');
    setCurrentDetail('SSE unavailable — using fallback POST…');
    try {
      const res = await axios.post<ChatApiResponse>(`${API_BASE}/api/v1/chat`, { text });
      finishWithResponse(res.data);
    } catch (err) {
      const axiosErr = err as AxiosError<{ detail?: string }>;
      const detail =
        axiosErr?.response?.data?.detail ??
        axiosErr?.message ??
        'Unknown error';
      finishWithError(detail);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  function handleHistoryClick(item: HistoryTrace) {
    setActiveHistoryId(item.trace_id);
    const sr = item.sandbox_result as SandboxResult | undefined;
    if (sr) setActiveSandbox(sr);
  }

  // Dynamic sandbox pane title
  const sandboxPaneTitle = loading && liveToolEvents.length > 0
    ? 'Running tools…'
    : loading
    ? 'Sandbox'
    : 'Sandbox evidence';

  return (
    <div className={styles.shell}>
      <Head>
        <title>Mycelium Console</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      {/* ── Header ───────────────────────────────────────── */}
      <header className={styles.header}>
        <div className={styles.headerLeft}>
          <button
            className={styles.sidebarToggle}
            onClick={() => setHistorySidebarOpen((v) => !v)}
            aria-label="Toggle history"
            title="Toggle history panel"
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <rect x="1" y="3" width="14" height="1.5" rx="0.75" fill="currentColor" />
              <rect x="1" y="7.25" width="10" height="1.5" rx="0.75" fill="currentColor" />
              <rect x="1" y="11.5" width="12" height="1.5" rx="0.75" fill="currentColor" />
            </svg>
          </button>
          <div className={styles.headerLogo}>
            <svg width="26" height="26" viewBox="0 0 28 28" fill="none" aria-label="Mycelium">
              <circle cx="14" cy="14" r="3.5" fill="currentColor" opacity="0.9" />
              <line x1="14" y1="14" x2="4" y2="6" stroke="currentColor" strokeWidth="1.2" opacity="0.5" />
              <line x1="14" y1="14" x2="24" y2="6" stroke="currentColor" strokeWidth="1.2" opacity="0.5" />
              <line x1="14" y1="14" x2="4" y2="22" stroke="currentColor" strokeWidth="1.2" opacity="0.5" />
              <line x1="14" y1="14" x2="24" y2="22" stroke="currentColor" strokeWidth="1.2" opacity="0.5" />
              <line x1="14" y1="14" x2="14" y2="2" stroke="currentColor" strokeWidth="1.2" opacity="0.5" />
              <line x1="14" y1="14" x2="14" y2="26" stroke="currentColor" strokeWidth="1.2" opacity="0.5" />
              <circle cx="4" cy="6" r="2" fill="currentColor" opacity="0.4" />
              <circle cx="24" cy="6" r="2" fill="currentColor" opacity="0.4" />
              <circle cx="4" cy="22" r="2" fill="currentColor" opacity="0.4" />
              <circle cx="24" cy="22" r="2" fill="currentColor" opacity="0.4" />
              <circle cx="14" cy="2" r="2" fill="currentColor" opacity="0.4" />
              <circle cx="14" cy="26" r="2" fill="currentColor" opacity="0.4" />
            </svg>
            <span className={styles.headerTitle}>Mycelium</span>
          </div>
          <span className={styles.headerSubtitle}>
            Layer 0 · Routing · Experts · Validation · Sandbox · Synthesis
          </span>
        </div>
      </header>

      {/* ── Body ─────────────────────────────────────────── */}
      <div className={styles.body}>

        {/* History sidebar */}
        {historySidebarOpen && (
          <aside className={styles.historySidebar}>
            <div className={styles.sidebarHeader}>
              <span className={styles.sidebarTitle}>Recent traces</span>
              <button
                className={styles.sidebarRefresh}
                onClick={loadHistory}
                title="Refresh"
                aria-label="Refresh history"
              >
                <svg width="13" height="13" viewBox="0 0 14 14" fill="none">
                  <path
                    d="M12 7A5 5 0 1 1 7 2v0l-1.5-1.5M7 2l1.5-1.5L7 2z"
                    stroke="currentColor"
                    strokeWidth="1.4"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </button>
            </div>
            <div className={styles.historyList}>
              {history.length === 0 && (
                <p className={styles.historyEmpty}>No traces yet.</p>
              )}
              {history.map((item) => (
                <HistoryItem
                  key={item.trace_id}
                  item={item}
                  active={activeHistoryId === item.trace_id}
                  onClick={() => handleHistoryClick(item)}
                />
              ))}
            </div>
          </aside>
        )}

        {/* Chat pane */}
        <main className={styles.chatPane}>
          <div className={styles.chatWindow}>
            {messages.length === 0 && !loading && (
              <div className={styles.emptyState}>
                <svg width="36" height="36" viewBox="0 0 28 28" fill="none" opacity="0.2">
                  <circle cx="14" cy="14" r="3.5" fill="currentColor" />
                  <line x1="14" y1="14" x2="4" y2="6" stroke="currentColor" strokeWidth="1.2" />
                  <line x1="14" y1="14" x2="24" y2="6" stroke="currentColor" strokeWidth="1.2" />
                  <line x1="14" y1="14" x2="4" y2="22" stroke="currentColor" strokeWidth="1.2" />
                  <line x1="14" y1="14" x2="24" y2="22" stroke="currentColor" strokeWidth="1.2" />
                  <line x1="14" y1="14" x2="14" y2="2" stroke="currentColor" strokeWidth="1.2" />
                  <line x1="14" y1="14" x2="14" y2="26" stroke="currentColor" strokeWidth="1.2" />
                </svg>
                <p>Ask anything. Mycelium routes it through the full reasoning pipeline, gathers real evidence via sandbox tools, and explains what it found.</p>
              </div>
            )}

            {messages.map((m, idx) => (
              <div
                key={idx}
                className={
                  m.role === 'user'
                    ? styles.userBubbleWrap
                    : styles.assistantBubbleWrap
                }
              >
                <div
                  className={
                    m.role === 'user' ? styles.userBubble : styles.assistantBubble
                  }
                >
                  <pre className={styles.bubbleText}>{m.content}</pre>
                </div>

                {m.role === 'assistant' && m.trace && (
                  <>
                    <div className={styles.traceToggleRow}>
                      <button
                        className={styles.traceToggleBtn}
                        onClick={() => toggleTrace(idx)}
                        aria-expanded={m.traceOpen}
                      >
                        <svg
                          width="11" height="11" viewBox="0 0 12 12" fill="none"
                          style={{
                            transform: m.traceOpen ? 'rotate(90deg)' : 'rotate(0deg)',
                            transition: 'transform 180ms ease',
                          }}
                        >
                          <path d="M4 2l4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                        </svg>
                        Pipeline trace
                      </button>
                      {m.sandbox && (
                        <button
                          className={styles.traceToggleBtn}
                          onClick={() => setActiveSandbox(m.sandbox ?? null)}
                          title="Show sandbox evidence in right panel"
                        >
                          <svg width="11" height="11" viewBox="0 0 12 12" fill="none">
                            <circle cx="6" cy="6" r="4.5" stroke="currentColor" strokeWidth="1.4" />
                            <line x1="6" y1="3" x2="6" y2="6.5" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
                            <circle cx="6" cy="8.5" r="0.75" fill="currentColor" />
                          </svg>
                          Evidence
                        </button>
                      )}
                    </div>
                    {m.traceOpen && <TracePanel trace={m.trace} />}
                  </>
                )}
              </div>
            ))}

            {loading && (
              <div className={styles.assistantBubbleWrap}>
                <div className={styles.assistantBubble}>
                  <PhaseIndicator
                    phase={currentPhase}
                    detail={currentDetail}
                    elapsedMs={elapsedMs}
                  />
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          <div className={styles.inputRow}>
            <input
              className={styles.input}
              placeholder="Ask Mycelium something…"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={loading}
              autoFocus
            />
            <button
              className={styles.sendButton}
              onClick={handleSend}
              disabled={loading || !input.trim()}
              aria-label="Send"
            >
              <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
                <path
                  d="M2 9h14M10 3l6 6-6 6"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </button>
          </div>
        </main>

        {/* Sandbox evidence panel */}
        <aside className={styles.sandboxPane}>
          <div className={styles.sidebarHeader}>
            <span className={styles.sidebarTitle}>{sandboxPaneTitle}</span>
            {loading && liveToolEvents.length > 0 && (
              <span className={styles.sandboxLiveBadge}>LIVE</span>
            )}
          </div>
          <SandboxPanel
            sandbox={activeSandbox}
            liveEvents={liveToolEvents}
            isLoading={loading}
          />
        </aside>
      </div>
    </div>
  );
}
