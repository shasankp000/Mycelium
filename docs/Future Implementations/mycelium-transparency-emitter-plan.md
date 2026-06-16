# Mycelium — Transparency Emitter Modification Plan

## Overview

This document details the full plan for adding fine-grained status emitters to the Mycelium reasoning pipeline, covering both backend (`run_workflow.py`, `broadcast_api.py`) and frontend (`web-ui/pages/index.tsx`) modifications. The goal is to give end users maximum transparency into every stage of the graph reasoning process via the existing SSE (Server-Sent Events) stream.

---

## Current State Analysis

### Backend SSE Stream (sparse — 9 phases)

The backend currently emits the following coarse phases via `broadcast_api.py`:

| Phase token | Progress % | Description |
|---|---|---|
| `setting_up` | 5% | Environment initialisation started |
| `environment_ready` | 12% | Environment ready |
| `routing` | 20% | Reasoning pipeline running |
| `expert_decision` | 35% | Expert decision resolved |
| `sandbox_plan` | 45% | Planning sandbox tools |
| `sandbox_tool/<n>` | 45–78% | Individual tool calls |
| `sandbox_summary` | 80% | Sandbox complete |
| `conversation` | 90% | Generating answer |
| `done` | 100% | Complete |
| `error` | 100% | Error state |

### The Core Transparency Gap

The `routing` phase in practice spans **7 sequential sub-steps** inside `run_mycelium_workflow()`:

1. Model registry warmup
2. `UnifiedExpertSystem` initialisation
3. Spectral signature sync via `DynamicSignatureManager`
4. `MultiLensRouter` + `QuestionRouter` init
5. Layer 0 classification (`QuestionRouter.route()`)
6. Multi-lens routing (`MultiLensRouter.route()`)
7. Phase 2 → Unified decision → Phase 3 pipeline

**Zero mid-phase signals are emitted during this window.** In practice, the UI progress bar freezes for 60–150 seconds then jumps suddenly to `expert_decision`. This is the primary UX problem to solve.

---

## Backend Changes

### 1. `run_workflow.py` — Add `status_cb` callback parameter

Add an optional `status_cb: Optional[Callable[[str, str], None]]` parameter to `run_mycelium_workflow()`. At each meaningful internal checkpoint, call `status_cb(phase, detail)`. This is purely additive — all existing behaviour is unchanged when `status_cb=None`.

#### Injection points

| Location in code | SSE `phase` emitted | `detail` string example |
|---|---|---|
| After `warmup(STARTUP_SPECS)` | `graph_warmup` | `"4 models resident"` |
| After `UnifiedExpertSystem()` init | `graph_expert_init` | `"12 expert domains loaded"` |
| After `sig_manager.sync_signatures(...)` | `graph_spectral_sync` | `"8 spectral signatures synced"` |
| After `MultiLensRouter(...)` + `QuestionRouter()` | `graph_router_ready` | `"MultiLens router initialised"` |
| After `question_router.route(text)` | `graph_layer0` | `"Route: REASONING_PIPELINE"` |
| After `router.route(text)` | `graph_routing` | `"Classification: EMPIRICAL_CLAIM · 3 domains"` |
| After `phase2_pipeline.run(...)` | `graph_phase2` | `"Phase 2 complete"` |
| After `unified_decision_analysis(...)` + `combine_routing_and_expert_decisions(...)` | `graph_unified_decision` | `"USE_EXISTING_EXPERT · biology (0.87)"` |
| After `phase3_pipeline.run_complete_pipeline(...)` | `graph_phase3` | `"Validation: SUPPORTED"` |
| After `embed_tags_transformer(...)` + `cluster_tags_transformer(...)` | `graph_clustering` | `"14 tags → 3 clusters"` |

#### Signature change

```python
def run_mycelium_workflow(
    sentences: Sequence[str],
    trace_id: Optional[str] = None,
    status_cb: Optional[Callable[[str, str], None]] = None,  # NEW
) -> Tuple[List[Dict[str, Any]], WorkflowMetrics]:
    ...
    def _emit(phase: str, detail: str = "") -> None:
        if status_cb:
            try:
                status_cb(phase, detail)
            except Exception:
                pass  # never let emitter errors abort the pipeline
```

All 10 injection points call `_emit(phase, detail)`.

---

### 2. `broadcast_api.py` — Wire `on_graph_event` into `_full_pipeline_generator`

The `_build_run_summary()` helper calls `run_mycelium_workflow()` inside a worker thread. Modify it to:

1. Accept a `status_cb` argument.
2. Create a thread-safe `on_graph_event` closure that posts to the SSE queue via `loop.call_soon_threadsafe(sse_queue.put_nowait, ev)`.
3. Pass `on_graph_event` into `run_mycelium_workflow(status_cb=on_graph_event)`.

```python
# Inside _full_pipeline_generator, before spawning the worker thread:
def on_graph_event(phase: str, detail: str) -> None:
    ev = _sse_event(phase, detail, elapsed())
    try:
        loop.call_soon_threadsafe(sse_queue.put_nowait, ev)
    except RuntimeError:
        pass  # loop closed (client disconnected)
```

Additionally, **shorten the SSE heartbeat interval from 15 s to 5 s** — graph events fire frequently, and the shorter heartbeat makes the stream feel more alive and prevents proxy timeouts.

---

### 3. Optional — Phase-level hooks in `Phase2Pipeline` and `Phase3To5Pipeline`

For maximum granularity, add two optional sub-phase emitters:

| Phase token | Emitter location | When fired |
|---|---|---|
| `graph_reasoning_chain` | `Phase2Pipeline.run()` — after each reasoning step | Per reasoning chain step (emit count + label) |
| `graph_validation_check` | `Phase3To5Pipeline.run_complete_pipeline()` — after validation | With `result_class` and `confidence` |

These are lower-priority and can be added after the core 10 injection points are live.

---

### 4. Extend the `done` payload with `graph_timings`

Capture wall-clock milliseconds for each `graph_*` phase in `_full_pipeline_generator` (by timestamping each `on_graph_event` call) and include a `graph_timings: Dict[str, float]` field in the `done` SSE event's payload alongside the existing `ChatApiResponse`. This enables the frontend to render a per-phase timing breakdown chart.

---

## Frontend Changes (`web-ui/pages/index.tsx`)

### 1. Expand `PHASE_META` to cover all new graph phases

Add the following entries to the existing `PHASE_META` record:

```typescript
graph_warmup:           { label: 'Loading model weights…',         progress: 8  },
graph_expert_init:      { label: 'Initialising expert domains…',   progress: 14 },
graph_spectral_sync:    { label: 'Syncing spectral signatures…',   progress: 18 },
graph_router_ready:     { label: 'Router ready',                   progress: 22 },
graph_layer0:           { label: 'Layer 0 classification…',        progress: 25 },
graph_routing:          { label: 'Multi-lens routing…',            progress: 30 },
graph_phase2:           { label: 'Phase 2: expert predictions…',   progress: 40 },
graph_reasoning_chain:  { label: 'Building reasoning chain…',      progress: 43 },
graph_unified_decision: { label: 'Unified expert decision…',       progress: 50 },
graph_validation_check: { label: 'Validation check…',              progress: 55 },
graph_phase3:           { label: 'Phase 3: validation…',           progress: 60 },
graph_clustering:       { label: 'Tag clustering complete',        progress: 65 },
```

No existing phases are modified or removed — this is purely additive.

---

### 2. New `PhaseEntry` type and `phaseLog` state

Replace the single `currentPhase` string with a full log of received phase events:

```typescript
interface PhaseEntry {
  phase: string;
  detail: string;
  elapsedMs: number;
  status: 'done' | 'active' | 'pending';
  wallMs: number;  // Date.now() at receipt
}

// In the main component:
const [phaseLog, setPhaseLog] = useState<PhaseEntry[]>([]);

// On each SSE message:
setPhaseLog(prev => {
  const updated = prev.map(e =>
    e.status === 'active' ? { ...e, status: 'done' } : e
  );
  return [...updated, {
    phase: ev.phase,
    detail: ev.detail,
    elapsedMs: ev.elapsed_ms,
    status: 'active',
    wallMs: Date.now(),
  }];
});
```

---

### 3. Replace `PhaseIndicator` with `ReasoningTimeline`

The current `PhaseIndicator` renders a single label + progress bar. Replace it with a vertical stepper that renders the full `phaseLog` so the user can see the complete reasoning path:

```
✓  Loading model weights         4 models resident              0.8s
✓  Initialising expert domains   12 expert domains loaded       2.1s
✓  Syncing spectral signatures   8 signatures synced            0.4s
✓  Layer 0 classification        REASONING_PIPELINE             0.3s
✓  Multi-lens routing            EMPIRICAL_CLAIM · biology      1.2s
⟳  Phase 2: expert predictions…  [animated pulse]               4.1s
○  Unified expert decision        —                              —
○  Phase 3: validation            —                              —
```

**Visual conventions:**
- `✓` — completed step, teal (`--color-primary`)
- `⟳` — active step, amber with CSS spin animation
- `○` — pending step, muted (`--color-text-faint`)
- `detail` string shown inline in a muted secondary column
- Elapsed time shown right-aligned

The overall top progress bar is retained but driven by `Math.max(...phaseLog.map(e => phaseProgress(e.phase)))`.

---

### 4. Add `GraphInsightBar` — inline decision callout

When a `graph_unified_decision` or `graph_layer0` SSE event arrives, parse the structured `detail` string and render a highlighted callout directly above the assistant message bubble:

```
┌────────────────────────────────────────────────────────────────┐
│  🧠  Expert Decision    USE_EXISTING_EXPERT                     │
│      Domain: biology  ·  Confidence: 87.4%                      │
└────────────────────────────────────────────────────────────────┘
```

For `graph_layer0`:

```
┌────────────────────────────────────────────────────────────────┐
│  🔀  Layer 0 Route    REASONING_PIPELINE                        │
└────────────────────────────────────────────────────────────────┘
```

The callout uses `--color-primary-highlight` as background, `--color-primary` for the icon and label, and collapses to nothing for non-pipeline routes.

---

### 5. Restructure `TracePanel` into collapsible sections

The current trace panel renders all fields as a flat key/value grid. Replace with grouped, collapsible sections:

| Section | Fields |
|---|---|
| **Layer 0** | `layer0_route` |
| **Routing** | `routing_classification`, `routing_domains` |
| **Expert Decision** | `expert_decision_type`, `selected_experts`, `expert_confidence` |
| **Validation** | `validation_result` |
| **Phase Timings** | `phase_latencies` — rendered as a mini horizontal bar chart using inline SVG |
| **Graph Timings** *(new)* | `graph_timings` from the `done` payload — per-phase wall-clock breakdown |

Each section has a `▶ / ▼` toggle. Sections with non-null values default to expanded; empty sections default to collapsed.

---

## Complete New SSE Event Taxonomy

| Phase token | Emitter file | Trigger point |
|---|---|---|
| `graph_warmup` | `run_workflow.py` | After `warmup(STARTUP_SPECS)` |
| `graph_expert_init` | `run_workflow.py` | After `UnifiedExpertSystem()` |
| `graph_spectral_sync` | `run_workflow.py` | After `sig_manager.sync_signatures()` |
| `graph_router_ready` | `run_workflow.py` | After `MultiLensRouter` + `QuestionRouter` init |
| `graph_layer0` | `run_workflow.py` | After `question_router.route(text)` |
| `graph_routing` | `run_workflow.py` | After `router.route(text)` |
| `graph_phase2` | `run_workflow.py` | After `phase2_pipeline.run()` |
| `graph_reasoning_chain` | `Phase2Pipeline` *(optional hook)* | Per reasoning step |
| `graph_unified_decision` | `run_workflow.py` | After `unified_decision_analysis()` + `combine_routing_and_expert_decisions()` |
| `graph_validation_check` | `Phase3To5Pipeline` *(optional hook)* | After validation result |
| `graph_phase3` | `run_workflow.py` | After `phase3_pipeline.run_complete_pipeline()` |
| `graph_clustering` | `run_workflow.py` | After `embed_tags_transformer()` + `cluster_tags_transformer()` |

---

## Implementation Staging

All stages are additive — no existing behaviour is broken at any point.

| Stage | Files modified | Scope |
|---|---|---|
| **1** | `run_workflow.py` | Add `status_cb` param + `_emit()` helper + all 10 injection sites |
| **2** | `broadcast_api.py` | Wire `on_graph_event` closure into `_build_run_summary`; shorten heartbeat to 5 s; pass `status_cb` into `run_mycelium_workflow` |
| **3** | `web-ui/pages/index.tsx` | Expand `PHASE_META`; add `PhaseEntry` type + `phaseLog` state; wire SSE handler to append entries |
| **4** | `web-ui/pages/index.tsx` | Build `ReasoningTimeline` component; replace `PhaseIndicator` |
| **5** | `web-ui/pages/index.tsx` | Add `GraphInsightBar`; restructure `TracePanel` into collapsible sections |
| **6** | `broadcast_api.py` + `api_models.py` | Add `graph_timings` dict to `done` payload; render in `TracePanel` as SVG bar chart |
| **7** *(optional)* | `phase2_validation/pipeline.py`, `phase3_validation/pipeline.py` | Add `graph_reasoning_chain` and `graph_validation_check` hook callbacks |

