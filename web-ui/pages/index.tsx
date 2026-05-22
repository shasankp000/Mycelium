// ---------------------------------------------------------------------------
// pages/index.tsx — Phase 1 refactor
// Orchestrator only. All components, hooks and types now live in their own
// files. This file owns top-level state and wires everything together.
// ---------------------------------------------------------------------------

import Head from 'next/head';
import { useState, useRef, useEffect, useCallback } from 'react';
import axios, { AxiosError } from 'axios';
import styles from '../styles/Home.module.css';

// Extracted types
import type {
  PipelineTrace,
  SandboxResult,
  Message,
  MyceliumRunSummary,
  ChatApiResponse,
  HistoryTrace,
  LiveToolEvent,
  SseEvent,
  ThinkingEvent,
  ThinkingEventKind,
  ReasoningMode,
} from '../types/pipeline';
import { THINKING_PHASES } from '../types/pipeline';

// Extracted hooks
import { useElapsedTick } from '../hooks/useElapsedTick';
import { useSseStream }   from '../hooks/useSseStream';

// Extracted components
import { CalibrationGate }               from '../components/CalibrationGate';
import { HomeScreen, PROMPT_SUGGESTIONS } from '../components/HomeScreen';
import { TracePanel }                    from '../components/TracePanel';
import { SandboxPanel }                  from '../components/SandboxPanel';

const API_BASE       = process.env.NEXT_PUBLIC_API_BASE ?? 'http://localhost:8000';
const MAX_INPUT_CHARS = 2000;

// ---------------------------------------------------------------------------
// Phase display config
// ---------------------------------------------------------------------------

const PHASE_META: Record<string, { label: string; progress: number }> = {
  setting_up:            { label: 'Setting up environment…',            progress: 5  },
  environment_ready:     { label: 'Environment ready',                  progress: 12 },
  routing:               { label: 'Running reasoning pipeline…',       progress: 20 },
  graph_routing:         { label: 'Semantic graph routing complete',    progress: 30 },
  graph_expert_init:     { label: 'Initialising expert graph…',        progress: 22 },
  graph_coverage_report: { label: 'Expert coverage assessed',           progress: 33 },
  heartbeat:             { label: 'Thinking…',                         progress: 25 },
  expert_decision:       { label: 'Expert decision resolved',           progress: 35 },
  sandbox_plan:          { label: 'Planning sandbox tools…',           progress: 45 },
  sandbox_summary:       { label: 'Sandbox complete',                   progress: 80 },
  conversation:          { label: 'Generating answer…',               progress: 90 },
  done:                  { label: 'Done',                               progress: 100 },
  error:                 { label: 'Error',                              progress: 100 },
};

function phaseLabel(phase: string): string {
  if (!phase) return 'Processing…';
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
  if (!phase) return 50;
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

function parseSandboxDetail(phase: string, detail: string): Partial<LiveToolEvent> {
  const parts = detail.split(' | ');
  if (phase === 'sandbox_plan') return { query: parts.slice(1).join(', ') };
  const indexMatch = phase.match(/sandbox_tool\/(\d+)/);
  const toolIndex  = indexMatch ? parseInt(indexMatch[1], 10) : 0;
  const toolName   = parts[0] ?? '';
  if (phase.endsWith('_ok') || phase.endsWith('_err')) {
    const durationMatch = (parts[1] ?? '').match(/(\d+)ms/);
    const durationMs    = durationMatch ? parseInt(durationMatch[1], 10) : undefined;
    const preview       = parts[2] ?? undefined;
    return { toolIndex, toolName, durationMs, preview };
  }
  const query = parts[1] ?? '';
  return { toolIndex, toolName, query };
}

function extractTrace(data: ChatApiResponse): PipelineTrace {
  const t = data?.trace ?? {};
  return {
    layer0_route:           t.layer0?.route,
    routing_classification: t.routing?.classification,
    routing_domains:        t.routing?.selected_domains ?? [],
    expert_decision_type:   t.expert_decision?.decision_type,
    selected_experts:       t.expert_decision?.selected_experts ?? [],
    expert_confidence:      t.expert_decision?.expert_confidence ?? null,
    validation_result:      t.phase3?.validation_decision?.result_class,
    phase_latencies:        t.phase3?.phase_latencies_ms ?? {},
    trace_id:               t.trace_id,
  };
}

function fmtDuration(ms: number) {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms.toFixed(0)}ms`;
}

// ---------------------------------------------------------------------------
// ThinkingPanel (inline — small enough to stay here for now)
// ---------------------------------------------------------------------------

const THINKING_ICONS: Record<ThinkingEventKind, string> = {
  routing:               '⟁',
  heartbeat:             '◌',
  graph_routing:         '✦',
  graph_expert_init:     '⬡',
  graph_coverage_report: '▤',
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
        const isRunning  = ev.state === 'running';
        const icon       = THINKING_ICONS[ev.kind] ?? '·';
        const kindLabel  = THINKING_KIND_LABEL[ev.kind] ?? ev.kind;
        const displayMsg = ev.message || ev.detail;
        const meta       = ev.metadata ?? {};
        const primaryDomain    = meta.primary_domain as string | undefined;
        const classification   = meta.classification as string | undefined;
        const candidateCount   = meta.candidate_count as number | undefined;
        const elapsedMsMeta    = meta.elapsed_ms as number | undefined;
        const expertCount      = meta.expert_count as number | undefined;
        const coveredDomains   = meta.covered_domains as string[] | undefined;
        return (
          <div
            key={ev.id}
            className={`${styles.thinkingRow} ${isRunning ? styles.thinkingRowRunning : styles.thinkingRowDone}`}
            aria-label={`${kindLabel}: ${displayMsg}`}
          >
            <span className={`${styles.thinkingIcon} ${isRunning ? styles.thinkingIconPulse : ''}`} aria-hidden="true">
              {icon}
            </span>
            <div className={styles.thinkingBody}>
              <div className={styles.thinkingTop}>
                <span className={styles.thinkingBadge}>{kindLabel}</span>
                <span className={styles.thinkingMsg}>{displayMsg}</span>
                {!isRunning && elapsedMsMeta !== undefined && (
                  <span className={styles.thinkingElapsed}>{fmtDuration(elapsedMsMeta)}</span>
                )}
              </div>
              {(primaryDomain || classification || candidateCount !== undefined || expertCount !== undefined) && (
                <p className={styles.thinkingMeta}>
                  {primaryDomain   && <span>domain: <strong>{primaryDomain}</strong></span>}
                  {classification  && <span> · {classification}</span>}
                  {candidateCount !== undefined && <span> · {candidateCount} candidate{candidateCount !== 1 ? 's' : ''}</span>}
                  {expertCount    !== undefined && <span> · {expertCount} expert{expertCount !== 1 ? 's' : ''}</span>}
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
// PhaseIndicator (inline — tightly coupled to phase state)
// ---------------------------------------------------------------------------

function PhaseIndicator({
  phase, detail, elapsedMs,
}: { phase: string; detail: string; elapsedMs: number }) {
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
// Misc small components (inline)
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

function HistoryItem({
  item, active, onClick,
}: { item: HistoryTrace; active: boolean; onClick: () => void }) {
  const ts = item.timestamp
    ? new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : '';
  return (
    <button className={`${styles.historyItem} ${active ? styles.historyItemActive : ''}`} onClick={onClick}>
      <span className={styles.historyQuery}>{item.user_query}</span>
      <span className={styles.historyMeta}>
        {item.run_summary?.expert_decision?.decision_type && (
          <span className={`${styles.historyBadge} ${
            item.run_summary.expert_decision.decision_type === 'CREATE_NEW_PATCH'
              ? styles.historyBadgePatch
              : styles.historyBadgeNormal
          }`}>
            {item.run_summary.expert_decision.decision_type === 'CREATE_NEW_PATCH' ? 'PATCH' : 'OK'}
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
  const [calibrationDone, setCalibrationDone] = useState(false);
  const [hasStarted, setHasStarted]           = useState(false);
  const [messages, setMessages]               = useState<Message[]>([]);
  const [input, setInput]                     = useState('');
  const [loading, setLoading]                 = useState(false);
  const [activeSandbox, setActiveSandbox]     = useState<SandboxResult | null>(null);
  // Phase 2: mode selector (default smart)
  const [mode, setMode]                       = useState<ReasoningMode>('smart');

  const [liveToolEvents, setLiveToolEvents] = useState<LiveToolEvent[]>([]);
  const [livePlanDetail, setLivePlanDetail] = useState<string>('');

  const [thinkingEvents, setThinkingEvents] = useState<ThinkingEvent[]>([]);
  const thinkingCounterRef                  = useRef<number>(0);

  const [history, setHistory]                     = useState<HistoryTrace[]>([]);
  const [activeHistoryId, setActiveHistoryId]     = useState<string | null>(null);
  const [historySidebarOpen, setHistorySidebarOpen] = useState(true);
  const [mobileSidebarOpen, setMobileSidebarOpen]   = useState(false);

  const [currentPhase, setCurrentPhase]   = useState<string>('routing');
  const [currentDetail, setCurrentDetail] = useState<string>('');

  const [backendDown, setBackendDown]                     = useState(false);
  const [healthBannerDismissed, setHealthBannerDismissed] = useState(false);

  const bottomRef = useRef<HTMLDivElement>(null);

  const { elapsedMs, start: startTick, stop: stopTick, dispose: disposeTick } = useElapsedTick();

  // ── Health check ────────────────────────────────────────────────────────
  useEffect(() => {
    async function checkHealth() {
      try {
        await axios.get(`${API_BASE}/api/v1/health`, { timeout: 4000 });
        setBackendDown(false);
      } catch { setBackendDown(true); }
    }
    checkHealth();
    const interval = setInterval(checkHealth, 30_000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  useEffect(() => () => disposeTick(), [disposeTick]);

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

  // ── Thinking event handler ───────────────────────────────────────────────
  function handleThinkingEvent(event: SseEvent) {
    const kind  = (event.phase_name ?? event.phase) as ThinkingEventKind;
    const state = event.state ?? (event.phase === 'done' ? 'done' : 'running');
    const message  = event.message ?? event.detail ?? '';
    const detail   = event.detail ?? '';
    const metadata = event.metadata ?? {};

    if (kind === 'heartbeat') {
      thinkingCounterRef.current += 1;
      setThinkingEvents((prev) => [
        ...prev,
        { id: thinkingCounterRef.current, kind, state, message, detail, metadata, ts: Date.now() },
      ]);
      return;
    }

    const substep = event.substep ?? '';
    setThinkingEvents((prev) => {
      const existingIdx = prev.findIndex(
        (e) => e.kind === kind && (e.metadata?.substep ?? '') === substep,
      );
      if (existingIdx !== -1) {
        const updated = [...prev];
        updated[existingIdx] = { ...updated[existingIdx], state, message, detail, metadata: { ...metadata, substep } };
        return updated;
      }
      thinkingCounterRef.current += 1;
      return [
        ...prev,
        { id: thinkingCounterRef.current, kind, state, message, detail, metadata: { ...metadata, substep }, ts: Date.now() },
      ];
    });
  }

  // ── Sandbox event handler ────────────────────────────────────────────────
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
            : ev,
        ),
      );
    } else {
      const newRow: LiveToolEvent = {
        toolIndex, toolName: toolName || '', query: query || '', phase: 'running', elapsedMs: elapsedMsVal,
      };
      setLiveToolEvents((prev) => {
        const exists = prev.find((e) => e.toolIndex === toolIndex);
        if (exists) return prev.map((e) => e.toolIndex === toolIndex ? newRow : e);
        return [...prev, newRow];
      });
    }
  }

  // ── Finish helpers ───────────────────────────────────────────────────────
  function finishWithResponse(data: ChatApiResponse) {
    const answer  = data.answer ?? 'Mycelium returned no answer text. Check the pipeline trace below.';
    const trace   = extractTrace(data);
    const sandbox = data.sandbox ?? null;
    setMessages((prev) => [...prev, { role: 'assistant', content: answer, trace, traceOpen: false, sandbox }]);
    if (sandbox) setActiveSandbox(sandbox);
    setLiveToolEvents([]);
    setLivePlanDetail('');
    setThinkingEvents([]);
    setLoading(false);
    stopTick();
    sseStream.close();
    loadHistory();
  }

  function finishWithError(detail: string, retryQuery?: string) {
    setMessages((prev) => [...prev, { role: 'error', content: detail, retryQuery }]);
    setLiveToolEvents([]);
    setLivePlanDetail('');
    setThinkingEvents([]);
    setLoading(false);
    stopTick();
    sseStream.close();
  }

  // ── SSE stream hook ──────────────────────────────────────────────────────
  const sseStream = useSseStream({
    mode,
    onThinkingEvent: handleThinkingEvent,
    onSandboxEvent:  handleSandboxEvent,
    onPhaseUpdate:   (phase, detail) => {
      setCurrentPhase(phase);
      setCurrentDetail(detail);
    },
    onDone:  finishWithResponse,
    onError: finishWithError,
  });

  // ── Send handler ─────────────────────────────────────────────────────────
  async function handleSend(overrideText?: string) {
    const text = (typeof overrideText === 'string' ? overrideText : input).trim();
    if (!text || loading) return;
    if (!hasStarted) setHasStarted(true);
    setMessages((prev) => [...prev, { role: 'user', content: text }]);
    if (!overrideText) setInput('');
    setLoading(true);
    setLiveToolEvents([]);
    setLivePlanDetail('');
    setThinkingEvents([]);
    setActiveSandbox(null);
    setCurrentPhase('routing');
    setCurrentDetail('Connecting to Mycelium…');
    startTick();
    sseStream.open(text);
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

  // ── Render: calibration gate ─────────────────────────────────────────────
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

  // ── Render: home screen ──────────────────────────────────────────────────
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

  // ── Render: chat UI ──────────────────────────────────────────────────────
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
                    {/* Phase 1: still using <pre>; Phase 2 will swap for react-markdown ChatBubble */}
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
                      {m.traceOpen && (
                        <div id={`trace-panel-${idx}`}>
                          <TracePanel trace={m.trace} />
                        </div>
                      )}
                    </>
                  )}
                </div>
              );
            })}

            {loading && (
              <div className={styles.assistantBubbleWrap}>
                <div className={styles.assistantBubble}>
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
                <span
                  className={`${styles.charCounter} ${atCharLimit ? styles.charCounterLimit : ''}`}
                  aria-live="polite"
                >
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
            {loading && hasLiveActivity && (
              <span className={styles.sandboxLiveBadge} aria-label="Live tool activity">LIVE</span>
            )}
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
