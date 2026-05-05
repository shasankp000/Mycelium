import Head from 'next/head';
import { useState, useRef, useEffect } from 'react';
import axios, { AxiosError } from 'axios';
import styles from '../styles/Home.module.css';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? 'http://localhost:8000';

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

interface Message {
  role: 'user' | 'assistant';
  content: string;
  trace?: PipelineTrace;
  traceOpen?: boolean;
}

interface ApiResponseData {
  answer?: string;
  content?: string;
  trace?: {
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
    trace_id?: string;
  };
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
  trace_id?: string;
}

function extractTrace(data: ApiResponseData): PipelineTrace {
  return {
    layer0_route: data?.trace?.layer0?.route ?? data?.layer0?.route,
    routing_classification:
      data?.trace?.routing?.classification ?? data?.routing?.classification,
    routing_domains:
      data?.trace?.routing?.selected_domains ?? data?.routing?.selected_domains ?? [],
    expert_decision_type:
      data?.trace?.expert_decision?.decision_type ??
      data?.expert_decision?.decision_type,
    selected_experts:
      data?.trace?.expert_decision?.selected_experts ??
      data?.expert_decision?.selected_experts ?? [],
    expert_confidence:
      data?.trace?.expert_decision?.expert_confidence ??
      data?.expert_decision?.expert_confidence ?? null,
    validation_result:
      data?.trace?.phase3?.validation_decision?.result_class ??
      data?.phase3?.validation_decision?.result_class,
    phase_latencies:
      data?.trace?.phase3?.phase_latencies_ms ??
      data?.phase3?.phase_latencies_ms ?? {},
    trace_id: data?.trace?.trace_id ?? data?.trace_id,
  };
}

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
            <span className={styles.traceValue}>{trace.expert_decision_type}</span>
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

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  function toggleTrace(idx: number) {
    setMessages((prev) =>
      prev.map((m, i) =>
        i === idx ? { ...m, traceOpen: !m.traceOpen } : m,
      ),
    );
  }

  async function handleSend() {
    if (!input.trim() || loading) return;
    const text = input.trim();
    setMessages((prev) => [...prev, { role: 'user', content: text }]);
    setInput('');
    setLoading(true);

    try {
      const res = await axios.post<ApiResponseData>(`${API_BASE}/api/v1/chat`, { text });
      const data = res.data;
      const answer: string =
        data.answer ||
        data.content ||
        'Mycelium returned no answer text. Check the pipeline trace below.';
      const trace = extractTrace(data);
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: answer, trace, traceOpen: false },
      ]);
    } catch (err) {
      const axiosErr = err as AxiosError<{ detail?: string }>;
      const detail =
        axiosErr?.response?.data?.detail ??
        axiosErr?.message ??
        'Unknown error';
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `⚠ Error contacting Mycelium backend: ${detail}\n\nIs the FastAPI server running at ${API_BASE}?`,
        },
      ]);
      console.error('Mycelium request failed', err);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  return (
    <div className={styles.container}>
      <Head>
        <title>Mycelium Console</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <header className={styles.header}>
        <div className={styles.headerLogo}>
          <svg width="28" height="28" viewBox="0 0 28 28" fill="none" aria-label="Mycelium">
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
        <span className={styles.headerSubtitle}>Layer 0 · Routing · Experts · Validation</span>
      </header>

      <main className={styles.main}>
        <div className={styles.chatWindow}>
          {messages.length === 0 && !loading && (
            <div className={styles.emptyState}>
              <p>Ask anything. Mycelium will route it through the full reasoning pipeline and explain what it found.</p>
            </div>
          )}

          {messages.map((m, idx) => (
            <div
              key={idx}
              className={m.role === 'user' ? styles.userBubbleWrap : styles.assistantBubbleWrap}
            >
              <div
                className={m.role === 'user' ? styles.userBubble : styles.assistantBubble}
              >
                <pre className={styles.bubbleText}>{m.content}</pre>
              </div>

              {m.role === 'assistant' && m.trace && (
                <div className={styles.traceToggleRow}>
                  <button
                    className={styles.traceToggleBtn}
                    onClick={() => toggleTrace(idx)}
                    aria-expanded={m.traceOpen}
                  >
                    <svg
                      width="12"
                      height="12"
                      viewBox="0 0 12 12"
                      fill="none"
                      style={{
                        transform: m.traceOpen ? 'rotate(90deg)' : 'rotate(0deg)',
                        transition: 'transform 180ms ease',
                      }}
                    >
                      <path d="M4 2l4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                    Pipeline trace
                  </button>
                </div>
              )}

              {m.role === 'assistant' && m.trace && m.traceOpen && (
                <TracePanel trace={m.trace} />
              )}
            </div>
          ))}

          {loading && (
            <div className={styles.assistantBubbleWrap}>
              <div className={`${styles.assistantBubble} ${styles.typingBubble}`}>
                <span className={styles.dot} />
                <span className={styles.dot} />
                <span className={styles.dot} />
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
              <path d="M2 9h14M10 3l6 6-6 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        </div>
      </main>
    </div>
  );
}
