# Mycelium Proof‑of‑Concept Implementation Plan

> Branch: `web-ui-prototype`

This document describes the implementation plan for a full Mycelium proof‑of‑concept (PoC) that combines the existing reasoning pipeline with a conversational LLM front‑end and a dedicated "sandbox" research layer.

The goal is to move from a working routing/decision demo to a minimal but complete architecture that demonstrates:

- How queries flow through Layer 0, routing, experts, validation, and improvement.
- How a conversational LLM can *explain* and *surface* those internals without doing its own opaque reasoning.
- How a sandbox layer can run concrete experiments, calculations, and literature lookups and archive reasoning traces.

**LLM vendor preference:** Unless otherwise noted, both the conversational agent and the sandbox agent will use **Ollama** as the primary runtime, with **Hugging Face‑hosted models** as the secondary fallback/provider.

---

## 1. Current Baseline

### 1.1 Backend

- `run_workflow.py` orchestrates the end‑to‑end Mycelium pipeline (Layer 0 → routing → Phase 2 → Phase 3–5) via `run_mycelium_workflow`.
- `phase3_validation/pipeline.py` implements `Phase3To5Pipeline`, which runs validation, action execution, feedback collection, performance analysis, integration, and continuous improvement, returning a `SystemExecutionResult` with per‑phase latencies and metadata.
- `broadcast_api.py` (FastAPI) exposes `/api/query`, wraps `run_mycelium_workflow([text])`, and returns a compact JSON snapshot containing routing, Phase 2/3 outputs, and aggregated `WorkflowMetrics`.

### 1.2 Frontend

- The `web-ui/` Next.js app provides a chat‑style console that:
  - Sends user text to `http://localhost:8000/api/query`.
  - Displays Layer 0 route, routing classification, expert decision, and validation summary.
  - Logs the full JSON response to the browser console for inspection.

### 1.3 Sandbox status

- A repository‑wide search for `sandbox`, `playground`, or `reasoning trace` in the Mycelium GitHub repo returns no dedicated sandbox or reasoning‑trace archival module yet.
- The current system logs metrics and some evaluation artifacts (e.g. JSON exports in `evaluation_data/`), but there is no structured, queryable trace store and no explicit research sandbox.

### 1.4 Known Pipeline Gap: Unhandled `CREATE_NEW_PATCH` Decision

**Observed behaviour (trace `21414ddc-6cab-4c30-a7be-ce667a8ae2d0`):**

When a query arrives for which no domain expert exists (e.g. "string theory"), the expert router produces:

| Field | Value |
|---|---|
| Layer 0 Route | `REASONING_PIPELINE` |
| Classification | `ATTRIBUTE_ONLY` |
| Expert Decision | `CREATE_NEW_PATCH` |
| Confidence | `0.0 %` |
| Slowest Phase | `phase_3_validation` (6 733 ms) |

The pipeline currently has **no handler** for `CREATE_NEW_PATCH`. The query falls through to validation with zero expert grounding, producing a high-latency, low-quality result.

**Fix (detailed in Milestone 1.5):** When the expert decision is `CREATE_NEW_PATCH`, the system must:

1. **Batch and log** the input query + generated response into a dedicated dataset so the next patch model can be trained on real traffic.
2. **Immediately continue** processing the query through the full 6-phase reasoning pipeline — the sandbox (evidence grounding, consequence generation) compensates for the missing expert.

---

## 2. Target Architecture Overview

At PoC completion, the architecture will consist of four cooperating layers:

1. **Core Reasoning Pipeline (Mycelium)**
   - Existing multi‑layer routing + Phase 2 + Phase 3–5 implementation.
   - Responsible for *structured* reasoning, validation, and improvement cycles.
   - Extended with a `CREATE_NEW_PATCH` handler (see §4 Milestone 1.5).

2. **Sandbox Research Layer**
   - A controlled environment where tools can be executed: numerical computations, simple simulations, web / literature search, and sub‑agent dialogues.
   - **Placement in the 6-phase pipeline:** The sandbox is *not* a post-processing step. It is invoked **inside Phase 4 (Consequence Generation) and Phase 5 (Evidence Grounding)** of the reasoning pipeline. Specifically:
     - **Phase 4 – Consequence Generation:** The sandbox runs simulations, calculations, and sub‑agent calls to explore hypothetical outcomes of proposed actions.
     - **Phase 5 – Evidence Grounding:** The sandbox performs web/paper searches to anchor claims in external evidence before the response is finalized.
   - Driven by a tool‑calling LLM (Ollama first, Hugging Face second) that receives structured outputs from Mycelium and a narrow instruction: *"Design and run experiments to test or extend these hypotheses; log every step and its result."*
   - Writes detailed reasoning traces to an archive store.

3. **Conversational LLM Agent**
   - A separate LLM (Ollama‑backed by default, with Hugging Face as fallback) used *only* for natural‑language interaction with the user (summaries, explanations, follow‑up questions).
   - Its prompts are seeded with Mycelium outputs and sandbox traces; it does **not** perform independent knowledge inference beyond those sources.

4. **Web UI Layer**
   - Extends the current console into a multi‑pane interface:
     - Chat panel (user ↔ conversational LLM).
     - Pipeline panel (Mycelium internal state for the latest query).
     - Sandbox panel (experiments, tools used, intermediate results).
     - Trace/History panel (archived reasoning episodes searchable by query, tags, and time).

---

## 3. Data Flow (High Level)

1. **User Query → Mycelium**
   - User submits text in the web UI.
   - Backend forwards to `run_mycelium_workflow` and obtains:
     - Layer 0 decision, routing context, expert decision.
     - Phase 2 and Phase 3–5 outputs.
     - Per‑run metrics and any improvement‑cycle metadata.

2. **Expert Decision Branching (NEW)**
   - After Phase 2 routing, the pipeline checks the expert decision type:
     - `USE_EXISTING_EXPERT` → normal flow, domain expert is loaded and runs.
     - `CREATE_NEW_PATCH` → **Patch Batching Path** (see §4 Milestone 1.5): log input + response to patch dataset, then continue via the sandbox-augmented reasoning pipeline (no expert loaded).
     - Other unknown decision types → treated as `CREATE_NEW_PATCH` for safety.

3. **Mycelium → Sandbox (Phases 4 & 5)**
   - A `SandboxTask` is produced from the current pipeline state, containing:
     - Key hypotheses, uncertainties, and metrics from Phase 3 validation.
     - Relevant tags/domains and expert suggestions.
   - The sandbox orchestration layer passes `SandboxTask` plus available tools to the tool‑calling LLM.
   - **Phase 4 (Consequence Generation):** Sandbox explores hypothetical outcomes.
   - **Phase 5 (Evidence Grounding):** Sandbox searches papers/web and appends evidence to the pipeline context.

4. **Sandbox → Trace Archive**
   - Every research step (tool call, result, and explanation) is appended to a persistent `ReasoningTrace` object, linked to the original user query and Mycelium run.
   - The archive is stored in a queryable format (JSONL or a small DB) and exposed via backend APIs.

5. **Mycelium + Sandbox Traces → Conversational LLM**
   - When generating a user‑visible answer, the conversational LLM receives:
     - Original question.
     - Mycelium pipeline summary (structured, not free‑text only).
     - Selected reasoning‑trace snippets from the sandbox.
   - It is instructed to *explain and synthesize* these artifacts, not to invent new facts outside them.

6. **Backend → Web UI**
   - Backend aggregates:
     - User‑safe summary for the chat.
     - Structured pipeline state.
     - Structured sandbox trace.
   - Web UI renders them into the appropriate panels.

---

## 4. PoC Milestones and Steps

### Milestone 0 – Stabilize Current Demo (DONE)

- [x] Expose `run_mycelium_workflow` via `broadcast_api.py`.
- [x] Next.js chat UI calling `/api/query` and displaying basic routing / expert / validation summaries.

### Milestone 1 – Formalize Response Schema

**Goal:** Make the backend response explicit and stable so later layers can consume it reliably.

1. Define `MyceliumRunSummary` Pydantic model that captures:
   - Layer 0 route and metadata.
   - Routing classification and selected domains.
   - Expert decision (type, selected experts, confidence).
   - Phase 2 result summary.
   - Phase 3–5 `SystemExecutionResult` subset (validation result, action status, feedback ids, latencies).
   - Per‑run `WorkflowMetrics` snapshot.
2. Update `/api/query` to return a `MyceliumRunSummary` instance instead of an untyped dict.
3. Version the API (e.g. `/api/v1/query`) so future changes are explicit.

### Milestone 1.5 – `CREATE_NEW_PATCH` Handler & Patch Batching System (NEW)

**Goal:** Eliminate the unhandled expert-decision gap and begin collecting data for future expert patches.

#### Background

When the router cannot find a matching domain expert it emits `CREATE_NEW_PATCH`. Without a handler, the query reaches Phase 3 validation with `confidence = 0.0`, causing slow, low-quality results. Two things must happen simultaneously:

1. The system must still answer the user's query as well as it can right now.
2. The system must record enough information to train a patch expert later.

#### Changes to the Reasoning Pipeline

```
Phase 2 (Routing) output
        │
        ▼
 ┌─────────────────────────────────┐
 │  Expert Decision Type?          │
 │                                 │
 │  USE_EXISTING_EXPERT ──────────►│ Load expert → normal Phase 3-6
 │                                 │
 │  CREATE_NEW_PATCH  ────────────►│ Patch Batching Path (below)
 │                                 │
 │  UNKNOWN / other ──────────────►│ Treat as CREATE_NEW_PATCH
 └─────────────────────────────────┘

CREATE_NEW_PATCH Path
        │
        ▼
 [1] Log input (query, routing metadata, timestamp, trace_id)
     to patch_dataset/<domain_tag>/<date>.jsonl
        │
        ▼
 [2] Continue Phase 3-6 WITHOUT a domain expert:
     - Phase 3: Validation runs with confidence = 0.0 flagged explicitly
       (no silent failure; validation log notes "no expert available")
     - Phase 4: Sandbox triggered for Consequence Generation
       (tool-calling LLM explores query space)
     - Phase 5: Sandbox triggered for Evidence Grounding
       (web/paper search to anchor the response)
     - Phase 6: Response assembled from sandbox evidence + pipeline state
        │
        ▼
 [3] Log final response + sandbox trace alongside the input entry
     in the same patch_dataset record (for dataset completeness)
        │
        ▼
 [4] Return response to user (normal API response path)
```

#### Dataset Format

Each JSONL record in `patch_dataset/<domain_tag>/<YYYY-MM-DD>.jsonl`:

```json
{
  "trace_id": "21414ddc-6cab-4c30-a7be-ce667a8ae2d0",
  "timestamp": "2026-05-05T18:07:00Z",
  "query": "Explain string theory",
  "routing": {
    "layer0_route": "REASONING_PIPELINE",
    "classification": "ATTRIBUTE_ONLY",
    "domains": [],
    "expert_decision": "CREATE_NEW_PATCH",
    "confidence": 0.0
  },
  "sandbox_evidence": [
    { "tool": "paper_search", "query": "...", "result_summary": "..." }
  ],
  "response": "...",
  "phase_latencies_ms": { "phase_3_validation": 6733.95 }
}
```

The `domain_tag` directory is derived from the query's detected topic cluster (or `"unclassified"` if none). This groups records by potential patch domain for easier future training runs.

#### Implementation Steps

1. Add `ExpertDecisionRouter` middleware in `run_workflow.py` (or a new `expert_decision_router.py`) that branches on the expert decision type immediately after Phase 2.
2. Implement `PatchBatchLogger`:
   - `log_input(trace_id, query, routing_meta) -> None` — writes the input portion of the record.
   - `log_response(trace_id, response, sandbox_evidence, latencies) -> None` — appends the response portion to the same record (matched by `trace_id`).
   - Writes to `patch_dataset/<domain_tag>/<YYYY-MM-DD>.jsonl` (append-only).
3. Modify Phase 3 validation to **explicitly flag** `confidence = 0.0` cases in its log output instead of silently proceeding.
4. Ensure Phase 4 and Phase 5 trigger the Sandbox for `CREATE_NEW_PATCH` queries (see Milestone 3).
5. Add `/api/patch-dataset/stats` endpoint (optional, for monitoring) that returns record counts per domain tag and date.

### Milestone 2 – Reasoning Trace Archive (Core, No Sandbox Yet)

**Goal:** Capture Mycelium's own reasoning state in a structured, queryable form.

1. Design a `ReasoningTrace` schema:
   - `trace_id`, `timestamp`, `user_query`.
   - Serialized `MyceliumRunSummary`.
   - Optional links to sandbox experiments (initially empty).
   - `patch_dataset_record_path` (optional) — if this trace was a `CREATE_NEW_PATCH` run, link to the corresponding patch record.
2. Implement a lightweight trace store:
   - Start with JSONL files (e.g. `traces/2026-05-*.jsonl`) or a tiny SQLite DB.
   - Provide append‑only API: `append_trace(trace: ReasoningTrace)`.
   - Provide read API: list recent traces, get by `trace_id`.
3. Hook `/api/query` to:
   - Create a `ReasoningTrace` for each request.
   - Persist it after the Mycelium run completes.
4. Expose `/api/traces/recent` to fetch recent runs for the UI.

### Milestone 3 – Sandbox Orchestration Layer

**Goal:** Introduce a pluggable sandbox that can run experiments and log detailed reasoning, without yet wiring in many heavy tools. The sandbox is integrated into **Phase 4 (Consequence Generation) and Phase 5 (Evidence Grounding)** of the 6-phase pipeline.

#### Sandbox Placement in the 6-Phase Pipeline

```
Phase 1 – Layer 0 Routing
Phase 2 – Expert Routing & Decision
Phase 3 – Validation
Phase 4 – Consequence Generation   ◄── Sandbox: simulations, sub-agent calls
Phase 5 – Evidence Grounding        ◄── Sandbox: web/paper search, fact anchoring
Phase 6 – Integration & Response
```

The sandbox is **not** called in Phase 1, 2, 3, or 6. Phases 4 and 5 each receive a `SandboxTask` tailored to their purpose:

- **Phase 4 SandboxTask type:** `"consequence"` — explore what happens if proposed actions or hypotheses are true.
- **Phase 5 SandboxTask type:** `"evidence"` — find papers, articles, or data that corroborate or refute claims from Phase 4.

#### Steps

1. Define `SandboxTask` and `SandboxResult` models:
   - `SandboxTask`: inputs from Mycelium (hypotheses, uncertainty flags, domains, tags, metrics, task_type: `"consequence" | "evidence"`).
   - `SandboxResult`: list of `SandboxStep` objects (tool name, input, output, commentary, status).
2. Implement a `SandboxManager`:
   - `run_consequence_phase(task: SandboxTask) -> SandboxResult` — called by Phase 4.
   - `run_evidence_phase(task: SandboxTask) -> SandboxResult` — called by Phase 5.
   - Stub tool interfaces for:
     - Python code execution (calculations / small simulations).
     - Web / paper search.
     - Simple sub‑agent discussion (e.g. multiple small LLM calls).
   - For the PoC, tools can initially be mocked or narrowed to a few safe operations.
3. Integrate a tool‑calling LLM:
   - Wrap Ollama + Hugging Face in a single interface (e.g. configurable provider priority in `ToolCallingLLM.run(task, tools)`).
   - Default resolution order: try Ollama; if unavailable or unsuitable for a given tool, fall back to Hugging Face.
   - Implement prompt templates that:
     - Emphasize experimentation and evidence gathering.
     - Require explicit logging of each step.
4. Connect Sandbox to trace archive:
   - When a sandbox run completes, attach its `SandboxResult` to the corresponding `ReasoningTrace`.

### Milestone 4 – Conversational LLM Agent

**Goal:** Add a conversational layer that explains Mycelium + sandbox results without replacing them.

1. Define `ConversationTurn` and `ConversationContext` models.
2. Implement a `ConversationAgent` that:
   - Accepts user query + `ReasoningTrace` (+ optional `SandboxResult`).
   - Builds a prompt that:
     - Describes Mycelium's decisions and any sandbox experiments.
     - Asks the LLM to produce a clear, faithful explanation and answer.
     - Forbids introducing unsupported claims.
   - Resolves its backing model via a shared LLM config (Ollama first, Hugging Face fallback) so both conversational and sandbox agents share vendor preferences.
3. Expose `/api/chat` endpoint:
   - On first turn, runs Mycelium + (optional) sandbox.
   - On follow‑up turns, fetches the existing `ReasoningTrace` by `trace_id` and sends updated instructions to the conversational agent.
4. Log conversational outputs back into the trace for full audibility.

### Milestone 5 – Web UI Evolution

**Goal:** Turn the current console into a multi‑pane reasoning explorer.

1. **Chat Panel**
   - Keep the existing chat as the main interaction surface.
   - Display conversational LLM messages (from `/api/chat`) instead of raw summary strings once the agent is ready.

2. **Pipeline Panel**
   - Add a collapsible sidebar or bottom drawer to render `MyceliumRunSummary` in a structured way:
     - Layer 0 route.
     - Routing classification and domains.
     - Expert decision type and experts.
     - Validation result and any blocked‑action reasons.
     - Phase latencies and metrics.
     - **`CREATE_NEW_PATCH` badge** if the query triggered the patch batching path.

3. **Sandbox Panel**
   - Render `SandboxResult` as a step‑by‑step timeline:
     - Tool used.
     - Input parameters.
     - Output summary.
     - Commentary from the tool‑calling LLM.
   - Distinguish Phase 4 (consequence) steps from Phase 5 (evidence) steps with a label.

4. **Trace / History Panel**
   - List recent `ReasoningTrace` entries via `/api/traces/recent`.
   - Allow clicking a past trace to:
     - Reload its pipeline state.
     - Replay or inspect its sandbox steps.
     - Ask the conversational agent follow‑up questions against that trace.

### Milestone 6 – Hardening and Constraints

**Goal:** Ensure the PoC behaves safely and predictably.

1. **Access Control / Keys**
   - Centralize all external API keys (LLM providers, search, etc.) via `config.toml` (see `config_loader.py`).
2. **Resource Limits**
   - Enforce per‑query caps on:
     - Number of sandbox tool calls.
     - Maximum wall‑clock time for sandbox runs.
     - Maximum stored trace size.
     - Maximum patch dataset record size per day per domain tag.
3. **Telemetry & Debugging**
   - Add structured logging for:
     - Each Mycelium run (input, route, expert decision, validation result).
     - Each sandbox step (tool, duration, success/failure).
     - Each `CREATE_NEW_PATCH` event (domain tag, dataset path written).
   - Ensure logs do not leak sensitive user data outside the sandbox.

---

## 5. Implementation Order (Checklist View)

1. **Schema Stabilization**
   - [ ] Define and adopt `MyceliumRunSummary` on the backend.
   - [ ] Update `/api/query` to return it.

2. **`CREATE_NEW_PATCH` Handler & Patch Batching** *(unblocks all queries, no expert needed)*
   - [ ] Add `ExpertDecisionRouter` branching after Phase 2.
   - [ ] Implement `PatchBatchLogger` (`log_input`, `log_response`).
   - [ ] Modify Phase 3 to explicitly flag zero-confidence runs.
   - [ ] Wire Phase 4 & 5 to trigger sandbox for `CREATE_NEW_PATCH` queries.
   - [ ] Add `/api/patch-dataset/stats` (optional monitoring endpoint).

3. **Trace Archive**
   - [ ] Implement `ReasoningTrace` and JSONL/SQLite store.
   - [ ] Add `patch_dataset_record_path` field to `ReasoningTrace`.
   - [ ] Wire traces into `/api/query` and add `/api/traces/recent`.

4. **Sandbox Layer**
   - [ ] Define `SandboxTask` (with `task_type`), `SandboxResult`, and `SandboxStep`.
   - [ ] Implement `SandboxManager.run_consequence_phase` (Phase 4) and `run_evidence_phase` (Phase 5).
   - [ ] Integrate tool‑calling LLM (Ollama primary, Hugging Face secondary) and attach results to traces.

5. **Conversational Agent**
   - [ ] Implement `ConversationAgent` and `/api/chat`.
   - [ ] Ensure prompts use Mycelium + sandbox outputs as the primary context.
   - [ ] Route conversational calls through the same LLM provider config (Ollama primary, Hugging Face secondary).

6. **UI Integration**
   - [ ] Add pipeline, sandbox, and trace panels to the web UI.
   - [ ] Add `CREATE_NEW_PATCH` badge to the pipeline panel.
   - [ ] Switch chat responses to conversational LLM once stable.

7. **Hardening**
   - [ ] Centralize configuration and keys via `config.toml`.
   - [ ] Add resource guards and structured logging including patch dataset events.

---

This plan should be refined as implementation progresses, but it provides a concrete roadmap from the current console demo to a full PoC with a conversational agent, sandbox research layer (integrated into Phases 4 & 5), patch batching for unknown domains, and reasoning trace archive, all wired into the Mycelium architecture.
