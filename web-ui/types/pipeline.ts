// ---------------------------------------------------------------------------
// Pipeline types — extracted from pages/index.tsx (Phase 1 refactor)
// Phase 4: SseEvent extended with deterministic ordering fields (§4.3)
// ---------------------------------------------------------------------------

export interface PipelineTrace {
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

export interface SandboxStep {
  tool: string;
  input: Record<string, string>;
  output: Record<string, unknown>;
  commentary: string;
  status: 'ok' | 'error' | string;
  duration_ms: number;
}

export interface SandboxResult {
  trace_id: string;
  summary: string;
  steps: SandboxStep[];
  started_at: string;
  finished_at: string;
}

export interface Message {
  role: 'user' | 'assistant' | 'error';
  content: string;
  trace?: PipelineTrace;
  traceOpen?: boolean;
  sandbox?: SandboxResult;
  retryQuery?: string;
  /** The ReasoningMode that produced this assistant message. */
  reasoningMode?: ReasoningMode;
}

export interface MyceliumRunSummary {
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

export interface ChatApiResponse {
  answer?: string;
  trace?: MyceliumRunSummary;
  sandbox?: SandboxResult;
}

export interface HistoryTrace {
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
// SseEvent — extended to carry structured PipelineEvent fields
// Phase 4 additions: sequence_number, event_id, timestamp (§4.3)
// ---------------------------------------------------------------------------

export interface SseEvent {
  // Legacy fields (always present)
  phase: string;
  detail: string;
  elapsed_ms: number;
  payload?: ChatApiResponse;
  // Structured PipelineEvent fields
  phase_name?: string;
  substep?: string;
  state?: 'running' | 'done' | 'error';
  visibility?: 'public' | 'internal';
  message?: string;
  metadata?: Record<string, unknown>;
  // Deterministic graph ordering fields (§4.3) — emitted by backend in Phase 4+
  sequence_number?: number;
  event_id?: string;
  timestamp?: number;   // ms epoch
}

// ---------------------------------------------------------------------------
// ThinkingEvent — internal state model for the ThinkingPanel
// ---------------------------------------------------------------------------

export type ThinkingEventKind =
  | 'routing'
  | 'heartbeat'
  | 'graph_routing'
  | 'graph_expert_init'
  | 'graph_coverage_report';

export interface ThinkingEvent {
  id: number;
  kind: ThinkingEventKind;
  state: 'running' | 'done' | 'error';
  message: string;
  detail: string;
  metadata: Record<string, unknown>;
  ts: number;
}

// Set of phase_name values that belong to the ThinkingPanel
export const THINKING_PHASES = new Set<string>([
  'routing',
  'heartbeat',
  'graph_routing',
  'graph_expert_init',
  'graph_coverage_report',
]);

// ---------------------------------------------------------------------------
// Reasoning mode — introduced in Phase 2, defined here for type-safety
// ---------------------------------------------------------------------------

export type ReasoningMode = 'fast' | 'smart' | 'researcher';
