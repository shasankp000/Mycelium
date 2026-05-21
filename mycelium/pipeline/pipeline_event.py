"""
pipeline_event.py
-----------------
Structured lifecycle event system for the Mycelium transparency emitter layer.

Replaces the raw (phase: str, detail: str) callback signature with a typed,
replay-safe, flood-controlled event pipeline:

    Reasoning Workers
            ↓
    Internal Event Bus  (EventEmitter)
            ↓
    Aggregation / Coalescing Layer  (CoalescingBuffer)
            ↓
    Visibility Filter  (strips internal events before SSE emission)
            ↓
    Replay Journal  (ReplayJournal — per-request short-lived log)
            ↓
    SSE Emitter  (broadcast_api.py consumes PipelineEvent dicts)
            ↓
    Frontend

Design constraints
------------------
- All events carry a strictly monotonic sequence_number so the frontend
  can reject stale / duplicate / replayed events.
- phase_id is a monotonic lifecycle stage index; frontend rejects regression
  unless state == "rollback".
- visibility="internal" events are filtered before SSE emission; they are
  only written to logs/internal_events/.
- The CoalescingBuffer enforces a maximum frontend emission rate of 10 Hz to
  prevent SSE saturation and frontend rerender storms.
- The ReplayJournal buffers the last N events per request to support
  reconnect recovery (frontend sends Last-Event-ID, backend replays missed
  events).
- _emit() is always wrapped in try/except so emitter errors never abort the
  reasoning pipeline.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phase registry — strictly monotonic lifecycle stage IDs
# ---------------------------------------------------------------------------

#: Maps phase_name → (phase_id, default_visibility)
#: phase_id must increase strictly along the happy-path lifecycle.
PHASE_REGISTRY: Dict[str, tuple[int, str]] = {
    # ── Bootstrap ───────────────────────────────────────────────────────────
    "setting_up":            (0,  "public"),
    "environment_ready":     (1,  "public"),
    # ── Graph init (new granular phases) ────────────────────────────────────
    "graph_warmup":          (2,  "public"),
    "graph_expert_init":     (3,  "public"),
    "graph_spectral_sync":   (4,  "public"),
    "graph_router_ready":    (5,  "public"),
    # ── Routing ─────────────────────────────────────────────────────────────
    "routing":               (6,  "public"),
    "graph_layer0":          (7,  "public"),
    "graph_routing":         (8,  "public"),
    # ── Reasoning ───────────────────────────────────────────────────────────
    "graph_phase2":          (9,  "public"),
    "graph_reasoning_chain": (10, "internal"),   # per-step; coalesced before emit
    # ── Decision ────────────────────────────────────────────────────────────
    "expert_decision":       (11, "public"),
    "graph_unified_decision":(11, "public"),
    # ── Validation ──────────────────────────────────────────────────────────
    "graph_validation_check":(12, "internal"),
    "graph_phase3":          (13, "public"),
    # ── Post-processing ─────────────────────────────────────────────────────
    "graph_clustering":      (14, "public"),
    # ── Sandbox ─────────────────────────────────────────────────────────────
    "sandbox_plan":          (15, "public"),
    # sandbox_tool/<n>, sandbox_tool/<n>_ok, sandbox_tool/<n>_err are dynamic
    "sandbox_summary":       (20, "public"),
    # ── Answer generation ───────────────────────────────────────────────────
    "conversation":          (21, "public"),
    # ── Terminal ────────────────────────────────────────────────────────────
    "done":                  (22, "public"),
    "error":                 (22, "public"),
}

# Heartbeat and semantic heartbeat messages (public, no phase advancement)
_HEARTBEAT_MESSAGES: List[str] = [
    "Reconciling conflicting evidence…",
    "Stabilising reasoning graph…",
    "Reviewing semantic dependencies…",
    "Cross-checking source consistency…",
    "Processing retrieved knowledge…",
]


def _phase_id_for(phase_name: str) -> int:
    """Return the lifecycle phase_id for a phase name.

    Dynamic sandbox_tool phases get phase_id 16–19 based on tool index.
    Unknown phases fall back to 50 (after done) so they never regress.
    """
    if phase_name in PHASE_REGISTRY:
        return PHASE_REGISTRY[phase_name][0]
    if phase_name.startswith("sandbox_tool/"):
        return 16
    return 50


def _visibility_for(phase_name: str) -> str:
    if phase_name in PHASE_REGISTRY:
        return PHASE_REGISTRY[phase_name][1]
    return "public"


# ---------------------------------------------------------------------------
# PipelineEvent dataclass
# ---------------------------------------------------------------------------

@dataclass
class PipelineEvent:
    """Structured lifecycle event emitted by the Mycelium reasoning pipeline.

    All fields are required on construction; defaults are provided for
    optional metadata to make call-sites ergonomic.
    """

    # Identity
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str = ""
    sequence_number: int = 0          # set by EventEmitter, not caller

    # Lifecycle position
    timestamp: float = field(default_factory=time.time)
    phase_id: int = 0                 # monotonic lifecycle stage
    phase_name: str = ""
    substep: str = ""

    # State
    state: str = "running"            # running | done | error | rollback
    visibility: str = "public"        # public | internal

    # Human-readable content
    message: str = ""
    detail: str = ""

    # Timing
    elapsed_ms: float = 0.0

    # Structured payload (optional)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_sse_dict(self) -> Dict[str, Any]:
        """Subset safe for SSE emission (public fields only)."""
        return {
            "event_id":       self.event_id,
            "request_id":     self.request_id,
            "sequence_number": self.sequence_number,
            "timestamp":      self.timestamp,
            "phase_id":       self.phase_id,
            "phase_name":     self.phase_name,
            "substep":        self.substep,
            "state":          self.state,
            "visibility":     self.visibility,
            "message":        self.message,
            "detail":         self.detail,
            "elapsed_ms":     self.elapsed_ms,
            "metadata":       self.metadata,
        }


# ---------------------------------------------------------------------------
# ReplayJournal — per-request short-lived event log for reconnect recovery
# ---------------------------------------------------------------------------

class ReplayJournal:
    """Buffers the last *maxlen* events for a single request.

    On reconnect the frontend sends ``Last-Event-ID: <sequence_number>``;
    the caller asks ``journal.replay_from(last_seen)`` to get missed events.
    """

    def __init__(self, maxlen: int = 200) -> None:
        self._buf: Deque[PipelineEvent] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def record(self, ev: PipelineEvent) -> None:
        with self._lock:
            self._buf.append(ev)

    def replay_from(self, last_seen_sequence: int) -> List[PipelineEvent]:
        """Return all events with sequence_number > last_seen_sequence."""
        with self._lock:
            return [e for e in self._buf if e.sequence_number > last_seen_sequence]

    def all_events(self) -> List[PipelineEvent]:
        with self._lock:
            return list(self._buf)


# ---------------------------------------------------------------------------
# CoalescingBuffer — throttle to max_hz before forwarding to SSE
# ---------------------------------------------------------------------------

class CoalescingBuffer:
    """Aggregates high-frequency internal events and re-emits at ≤ max_hz.

    For phases like graph_reasoning_chain that fire per-step, the buffer
    keeps only the latest event for each phase_name and flushes on a timer,
    preventing SSE saturation.

    Thread-safe.
    """

    def __init__(
        self,
        max_hz: float = 10.0,
        downstream: Optional[Callable[[PipelineEvent], None]] = None,
    ) -> None:
        self._interval = 1.0 / max(max_hz, 0.1)
        self._downstream = downstream
        self._pending: Dict[str, PipelineEvent] = {}
        self._lock = threading.Lock()
        self._last_flush = time.monotonic()

    def push(self, ev: PipelineEvent) -> None:
        """Accept an event; flush if interval has elapsed."""
        with self._lock:
            # Keep only the latest event per phase_name (coalescing)
            self._pending[ev.phase_name] = ev
            now = time.monotonic()
            if now - self._last_flush >= self._interval:
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._pending or self._downstream is None:
            self._last_flush = time.monotonic()
            return
        # Emit in phase_id order for deterministic rendering
        events = sorted(self._pending.values(), key=lambda e: (e.phase_id, e.sequence_number))
        self._pending.clear()
        self._last_flush = time.monotonic()
        for ev in events:
            try:
                self._downstream(ev)
            except Exception:
                logger.exception("CoalescingBuffer downstream error for phase=%s", ev.phase_name)


# ---------------------------------------------------------------------------
# EventEmitter — main entry point used by run_workflow.py
# ---------------------------------------------------------------------------

class EventEmitter:
    """Thread-safe event emitter for a single pipeline run.

    Usage (in run_workflow.py)::

        emitter = EventEmitter(request_id=trace_id, wall_start=time.monotonic())
        # wire emitter.on_public_event into the SSE queue
        ...
        emitter.emit(phase_name="graph_warmup", message="Loading model weights…",
                     detail=f"{n} models resident", metadata={"model_count": n})

    The emitter:
    1. Builds a full PipelineEvent (fills event_id, sequence_number, phase_id,
       visibility, elapsed_ms automatically).
    2. Records the event in its ReplayJournal.
    3. Logs internal events to the internal event logger.
    4. Routes public events through the CoalescingBuffer to the SSE callback.
    """

    def __init__(
        self,
        request_id: str = "",
        wall_start: Optional[float] = None,
        max_hz: float = 10.0,
        on_public_event: Optional[Callable[[PipelineEvent], None]] = None,
    ) -> None:
        self._request_id = request_id
        self._wall_start = wall_start or time.monotonic()
        self._seq = 0
        self._seq_lock = threading.Lock()
        self.journal = ReplayJournal()
        self._coalescer = CoalescingBuffer(
            max_hz=max_hz,
            downstream=self._dispatch_public,
        )
        self._on_public_event = on_public_event
        self._internal_logger = logging.getLogger("mycelium.internal_events")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def emit(
        self,
        phase_name: str,
        message: str = "",
        detail: str = "",
        substep: str = "",
        state: str = "running",
        metadata: Optional[Dict[str, Any]] = None,
        visibility: Optional[str] = None,
    ) -> None:
        """Build and dispatch a PipelineEvent. Never raises."""
        try:
            with self._seq_lock:
                self._seq += 1
                seq = self._seq

            ev = PipelineEvent(
                request_id=self._request_id,
                sequence_number=seq,
                timestamp=time.time(),
                phase_id=_phase_id_for(phase_name),
                phase_name=phase_name,
                substep=substep,
                state=state,
                visibility=visibility or _visibility_for(phase_name),
                message=message,
                detail=detail,
                elapsed_ms=round((time.monotonic() - self._wall_start) * 1000, 1),
                metadata=metadata or {},
            )

            # Always record in journal (supports replay)
            self.journal.record(ev)

            # Route by visibility
            if ev.visibility == "internal":
                self._internal_logger.debug(
                    "[internal] request=%s seq=%d phase=%s detail=%s",
                    self._request_id, seq, phase_name, detail,
                )
            else:
                # Public events go through the coalescer → SSE
                self._coalescer.push(ev)

        except Exception:
            logger.exception("EventEmitter.emit error for phase=%s", phase_name)

    def emit_heartbeat(self, idx: int = 0) -> None:
        """Emit a semantic heartbeat (public, no phase advancement)."""
        msg = _HEARTBEAT_MESSAGES[idx % len(_HEARTBEAT_MESSAGES)]
        self.emit(
            phase_name="heartbeat",
            message=msg,
            detail="",
            state="running",
            visibility="public",
        )

    def flush(self) -> None:
        """Force the coalescing buffer to flush any pending events."""
        self._coalescer.flush()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _dispatch_public(self, ev: PipelineEvent) -> None:
        if self._on_public_event is not None:
            try:
                self._on_public_event(ev)
            except Exception:
                logger.exception("EventEmitter on_public_event callback error")


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_emitter(
    request_id: str,
    wall_start: float,
    on_public_event: Callable[[PipelineEvent], None],
    max_hz: float = 10.0,
) -> EventEmitter:
    """Create a ready-to-use EventEmitter wired to an SSE callback."""
    return EventEmitter(
        request_id=request_id,
        wall_start=wall_start,
        max_hz=max_hz,
        on_public_event=on_public_event,
    )
