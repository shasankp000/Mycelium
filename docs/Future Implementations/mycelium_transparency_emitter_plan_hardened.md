# Mycelium -- Transparency Emitter Modification Plan (Hardened Revision)

## Overview

This document updates the original transparency emitter plan with architectural hardening fixes required for:

- async reasoning stability,
- replay-safe SSE streaming,
- frontend/backend synchronization correctness,
- semantic safety,
- event flood control,
- and cancellation isolation.

The transparency layer is now treated as:

> a structured lifecycle event system

rather than:

> simple progress-message streaming.

All changes remain additive.

---

# 1. Core Architectural Changes

## 1.1 Replace Raw Phase Messages with Structured Events

The original plan emits:

```python
status_cb(phase, detail)
```

This is insufficient for:

- replay safety,
- async ordering,
- deduplication,
- reconnect recovery,
- deterministic rendering.

---

## REQUIRED REPLACEMENT

All status events MUST emit the following structure:

```python
{
    "event_id": str,
    "request_id": str,
    "sequence_number": int,
    "timestamp": float,

    "phase_id": int,
    "phase_name": str,
    "substep": str,

    "state": str,
    "visibility": "public" | "internal",

    "message": str,
    "detail": str,

    "elapsed_ms": float,

    "metadata": dict,
}
```

---

## Required Semantics

### sequence_number

Strictly monotonic.

Frontend MUST reject:

- stale events,
- duplicate events,
- replayed events.

---

### phase_id

Strictly monotonic lifecycle stage.

Frontend MUST reject:

- lower phase IDs,
- backwards progress.

Unless:

```python
state == "rollback"
```

---

### visibility

Prevents semantic leakage.

Used to separate:

- developer/internal reasoning events,
- user-safe UI events.

---

# 2. `run_workflow.py` Changes

## 2.1 Replace `status_cb(phase, detail)` Signature

### OLD

```python
status_cb(phase, detail)
```

---

### NEW

```python
status_cb(event: Dict[str, Any])
```

---

## 2.2 Internal `_emit()` Helper

```python
sequence_counter += 1

_emit(
    phase_name="graph_warmup",
    phase_id=1,
    message="Loading model weights...",
    detail="4 models resident",
    visibility="public",
)
```

---

## 2.3 Add Event Metadata

Every emitted event should contain:

```python
metadata={
    "worker_count": ...,
    "reasoning_depth": ...,
    "active_domains": ...,
}
```

where appropriate.

---

# 3. Prevent Event Flooding

## Problem

The original plan allows:

```text
per reasoning step events
```

which risks:

- frontend overload,
- SSE saturation,
- rerender storms,
- reconnect instability.

---

# REQUIRED FIX

Introduce:

# Event Coalescing Layer

Reasoning-worker updates MUST pass through:

```text
worker events
→ aggregation buffer
→ coalesced emitter
→ SSE stream
```

---

## Emission Limit

Maximum frontend emission frequency:

```text
5–10 Hz
```

---

## Example

### BAD

```text
worker_step_51
worker_step_52
worker_step_53
```

---

### GOOD

```python
{
    "phase_name": "graph_reasoning_chain",
    "message": "Reasoning workers active",
    "detail": "4 workers · latest step 53",
}
```

---

# 4. Semantic Leakage Protection

## Problem

The original spec risks exposing:

- graph IDs,
- ontology internals,
- contradiction mechanics,
- stabilization details,
- reasoning topology.

This creates:

- attack surfaces,
- prompt manipulation vectors,
- confusing UX.

---

# REQUIRED FIX

Separate:

## Internal Events

Used for:

- debugging,
- developer tooling,
- replay analysis,
- profiling.

---

## Public Events

Human-readable abstraction layer.

---

## Example

### INTERNAL

```text
TRM contradiction arbitration on graph G124-v5
```

### PUBLIC

```text
Verifying consistency of retrieved knowledge...
```

---

# 5. Cancellation Isolation

## Problem

The original spec allows frontend disconnect handling but does not formally separate:

- request lifecycle,
- semantic mutation lifecycle.

This risks:

- partial stabilization corruption,
- orphaned graph revisions,
- interrupted leverage propagation.

---

# REQUIRED FIX

Frontend disconnects MUST NOT cancel:

- graph persistence,
- stabilization,
- ontology updates,
- contradiction propagation,
- leverage decay,
- revision commits.

---

## Only These May Cancel

- SSE streaming,
- temporary visualization workers,
- UI-only summarizers,
- ephemeral frontend tasks.

---

# 6. Replay-Safe SSE Journaling

## Problem

The original spec lacks replay protection.

Reconnects may duplicate:

- progress messages,
- tool rows,
- phase transitions.

---

# REQUIRED FIX

Frontend stores:

```python
last_seen_sequence
```

and rejects:

- stale sequence numbers,
- duplicate event IDs,
- replayed events.

---

## Backend Addition

Maintain:

```text
short-lived replay journal
```

per request.

Supports:

- reconnect recovery,
- missed-event replay,
- deterministic state reconstruction.

---

# 7. Monotonic Phase State Machine

## Problem

Parallel TRM + reasoning workers may emit out-of-order phase updates.

Example:

```text
Reasoning complete
```

appearing before:

```text
Validation started
```

---

# REQUIRED FIX

All phases MUST belong to a backend-authoritative lifecycle graph.

Example:

```text
Initialization
→ Retrieval
→ Routing
→ Reasoning
→ Validation
→ Stabilization
→ Finalization
```

Frontend NEVER infers state.

Frontend ONLY renders backend-declared lifecycle state.

---

# 8. Replace Percentage Completion with Phase Progress

## Problem

Semantic reasoning is:

- recursive,
- contradiction-sensitive,
- dynamically branching.

Hard percentages become misleading.

---

# REQUIRED FIX

Replace:

```text
92% complete
```

with:

```text
[Retrieval]
[Reasoning]
[Verification]
[Stabilization]
[Finalization]
```

Optional:

```text
soft estimates
```

may be shown.

---

# 9. Semantic Heartbeat System

## Problem

Long stabilization phases appear frozen.

---

# REQUIRED FIX

Emit semantic heartbeat events every:

```text
2–5 seconds
```

Example:

```text
Reconciling conflicting evidence...
Stabilizing reasoning graph...
Reviewing semantic dependencies...
```

These are:

- continuity signals,
- NOT exact internal dumps.

---

# 10. Deterministic Tool Feed Allocation

## Problem

Async tool completion may reorder tool rows.

---

# REQUIRED FIX

Allocate tool rows:

```text
at dispatch time
```

NOT:

```text
at completion time
```

---

## Example

```text
[1] web_search        (running)
[2] knowledge_base    (running)
[3] calculator        (complete)
```

Stable ordering prevents UI flicker.

---

# 11. Recommended Internal Event Pipeline

```text
Reasoning Workers
        ↓
Internal Event Bus
        ↓
Aggregation Layer
        ↓
Visibility Filter
        ↓
Replay Journal
        ↓
SSE Emitter
        ↓
Frontend
```

---

# 12. Frontend Changes

## 12.1 `PhaseEntry` Additions

### OLD

```typescript
interface PhaseEntry {
  phase: string;
  detail: string;
  elapsedMs: number;
  status: 'done' | 'active' | 'pending';
  wallMs: number;
}
```

---

### NEW

```typescript
interface PhaseEntry {
  eventId: string;
  sequenceNumber: number;

  phase: string;
  phaseId: number;

  message: string;
  detail: string;

  elapsedMs: number;

  visibility: 'public' | 'internal';

  status: 'done' | 'active' | 'pending';

  wallMs: number;
}
```

---

## 12.2 Frontend Deduplication

Frontend MUST:

- track highest sequence number,
- ignore stale events,
- ignore duplicate event IDs,
- reject phase regression.

---

## 12.3 Public Event Rendering Only

Frontend MUST render:

```typescript
visibility === 'public'
```

only.

Internal events remain developer-only.

---

# 13. Graph Timing Collection

## REQUIRED CHANGE

Timing collection MUST use:

```text
phase transitions
```

NOT:

```text
individual worker timestamps
```

otherwise parallel worker noise corrupts timings.

---

## Recommended Structure

```python
graph_timings = {
    "routing": ...,
    "reasoning": ...,
    "validation": ...,
    "stabilization": ...,
}
```

Aggregate phase timings only.

---

# 14. Logging Separation

Persist:

```text
logs/internal_events/
logs/public_events/
```

separately.

---

## Benefits

Supports:

- debugging,
- semantic replay,
- stabilization analysis,
- contradiction tracing,
- performance profiling.

---

# 15. Stress-Test Requirements

The emitter system MUST be stress-tested under:

- forced reconnects,
- delayed packets,
- dropped SSE streams,
- contradiction storms,
- long stabilization phases,
- parallel reasoning workers,
- ontology update cascades,
- frontend disconnects.

---

# 16. Final Architectural Principle

The transparency emitter layer is now formally treated as:

> a distributed lifecycle orchestration subsystem

NOT:

> a console logger with SSE attached.

This distinction is critical for:

- future scalability,
- deterministic reasoning observability,
- semantic consistency,
- and distributed reasoning stability.

