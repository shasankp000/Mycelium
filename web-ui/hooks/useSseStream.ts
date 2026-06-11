// ---------------------------------------------------------------------------
// useSseStream — owns the EventSource lifecycle, retry logic, and event
// routing. Returns a single `open` function that the page calls on submit.
//
// Phase 4: adds onGraphEvent callback for all non-done/error phases,
//          enabling useGraphBuilder.ingestSseEvent to consume every SSE event.
// ---------------------------------------------------------------------------

import { useRef, useCallback } from 'react';
import axios, { AxiosError } from 'axios';
import type { SseEvent, ChatApiResponse } from '../types/pipeline';
import { THINKING_PHASES } from '../types/pipeline';
import type { ReasoningMode } from '../types/pipeline';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? 'http://localhost:8000';
const SSE_RETRY_ATTEMPTS = 3;
const SSE_RETRY_DELAY_MS = 1500;
const SSE_TIMEOUT_MS     = 3 * 60 * 1000;

function isSandboxPhase(phase: string): boolean {
  return phase === 'sandbox_plan' || phase.startsWith('sandbox_tool/');
}

interface UseSseStreamOptions {
  mode: ReasoningMode;
  onThinkingEvent: (event: SseEvent) => void;
  onSandboxEvent:  (phase: string, detail: string, elapsedMs: number) => void;
  onPhaseUpdate:   (phase: string, detail: string, elapsedMs: number) => void;
  onDone:          (data: ChatApiResponse) => void;
  onError:         (detail: string, retryText?: string) => void;
  /** Phase 4: called for every non-terminal SSE event so the graph can ingest it */
  onGraphEvent?:   (event: SseEvent) => void;
}

export function useSseStream({
  mode,
  onThinkingEvent,
  onSandboxEvent,
  onPhaseUpdate,
  onDone,
  onError,
  onGraphEvent,
}: UseSseStreamOptions) {
  const esRef           = useRef<EventSource | null>(null);
  const sseTimeoutRef   = useRef<ReturnType<typeof setTimeout> | null>(null);
  const sseRetryCount   = useRef<number>(0);
  const lastEventAtRef  = useRef<number>(0);

  const clearSseTimeout = useCallback(() => {
    if (sseTimeoutRef.current) {
      clearTimeout(sseTimeoutRef.current);
      sseTimeoutRef.current = null;
    }
  }, []);

  const fallbackPost = useCallback(async (text: string) => {
    onPhaseUpdate('routing', 'SSE unavailable — using fallback POST…', 0);
    try {
      const res = await axios.post<ChatApiResponse>(`${API_BASE}/api/v1/chat`, { text });
      onDone(res.data);
    } catch (err) {
      const axiosErr = err as AxiosError<{ detail?: string }>;
      onError(
        axiosErr?.response?.data?.detail ?? axiosErr?.message ?? 'Unknown error',
        text,
      );
    }
  }, [onDone, onError, onPhaseUpdate]);

  const openSseStream = useCallback((text: string) => {
    const sseUrl = `${API_BASE}/api/v1/chat/stream?text=${encodeURIComponent(text)}&mode=${mode}`;
    const es = new EventSource(sseUrl);
    esRef.current = es;
    let gotDone = false;

    clearSseTimeout();
    sseTimeoutRef.current = setTimeout(() => {
      if (!gotDone) {
        es.close();
        esRef.current = null;
        onError('Request timed out after 3 minutes.', text);
      }
    }, SSE_TIMEOUT_MS);

    es.onmessage = (ev) => {
      lastEventAtRef.current = Date.now();
      sseRetryCount.current = 0;
      try {
        const event: SseEvent = JSON.parse(ev.data);
        const phaseName = event.phase_name ?? event.phase;

        onPhaseUpdate(phaseName, event.detail, event.elapsed_ms);

        if (THINKING_PHASES.has(phaseName) && event.visibility !== 'internal') {
          onThinkingEvent(event);
        }

        if (isSandboxPhase(event.phase)) {
          onSandboxEvent(event.phase, event.detail, event.elapsed_ms);
        }

        if (event.phase === 'done' && event.payload) {
          gotDone = true;
          es.close();
          esRef.current = null;
          clearSseTimeout();
          sseRetryCount.current = 0;
          onDone(event.payload);
        } else if (event.phase === 'error') {
          gotDone = true;
          es.close();
          esRef.current = null;
          clearSseTimeout();
          sseRetryCount.current = 0;
          onError(event.detail || 'Unknown SSE error', text);
        } else if (!isSandboxPhase(event.phase)) {
          // Phase 4: route every non-terminal, non-sandbox event to the graph builder
          onGraphEvent?.(event);
        }
      } catch { /* malformed SSE frame — ignore */ }
    };

    es.onerror = () => {
      if (gotDone) return;
      const silentForMs   = Date.now() - lastEventAtRef.current;
      const neverReceived = lastEventAtRef.current === 0;
      if (!neverReceived && silentForMs < 20_000) return;

      es.close();
      esRef.current = null;

      if (sseRetryCount.current < SSE_RETRY_ATTEMPTS) {
        sseRetryCount.current += 1;
        onPhaseUpdate(
          'routing',
          `SSE disconnected — retrying (${sseRetryCount.current}/${SSE_RETRY_ATTEMPTS})…`,
          0,
        );
        setTimeout(() => openSseStream(text), SSE_RETRY_DELAY_MS * sseRetryCount.current);
      } else {
        clearSseTimeout();
        fallbackPost(text);
      }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, onThinkingEvent, onSandboxEvent, onPhaseUpdate, onDone, onError, onGraphEvent, clearSseTimeout, fallbackPost]);

  const open = useCallback((text: string) => {
    lastEventAtRef.current = 0;
    sseRetryCount.current  = 0;
    esRef.current?.close();
    openSseStream(text);
  }, [openSseStream]);

  const close = useCallback(() => {
    esRef.current?.close();
    esRef.current = null;
    clearSseTimeout();
  }, [clearSseTimeout]);

  return { open, close };
}
