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

- `run_workflow.py` orchestrates the end‑to‑end Mycelium pipeline (Layer 0 → routing → Phase 2 → Phase 3–5) via `run_mycelium_workflow`.[cite:79]
- `phase3_validation/pipeline.py` implements `Phase3To5Pipeline`, which runs validation, action execution, feedback collection, performance analysis, integration, and continuous improvement, returning a `SystemExecutionResult` with per‑phase latencies and metadata.[cite:78]
- `broadcast_api.py` (FastAPI) exposes `/api/query`, wraps `run_mycelium_workflow([text])`, and returns a compact JSON snapshot containing routing, Phase 2/3 outputs, and aggregated `WorkflowMetrics`.[cite:89]

### 1.2 Frontend

- The `web-ui/` Next.js app provides a chat‑style console that:
  - Sends user text to `http://localhost:8000/api/query`.
  - Displays Layer 0 route, routing classification, expert decision, and validation summary.
  - Logs the full JSON response to the browser console for inspection.[cite:99]

### 1.3 Sandbox status

- A repository‑wide search for `sandbox`, `playground`, or `reasoning trace` in the Mycelium GitHub repo returns no dedicated sandbox or reasoning‑trace archival module yet.[cite:103]
- The current system logs metrics and some evaluation artifacts (e.g. JSON exports in `evaluation_data/`), but there is no structured, queryable trace store and no explicit research sandbox.

---

## 2. Target Architecture Overview

At PoC completion, the architecture will consist of four cooperating layers:

1. **Core Reasoning Pipeline (Mycelium)**
   - Existing multi‑layer routing + Phase 2 + Phase 3–5 implementation.
   - Responsible for *structured* reasoning, validation, and improvement cycles.

2. **Sandbox Research Layer**
   - A controlled environment where tools can be executed: numerical computations, simple simulations, web / literature search, and sub‑agent dialogues.
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

2. **Mycelium → Sandbox (optional per query)**
   - For queries that require deeper investigation (as flagged by Mycelium or by user intent), a structured `SandboxTask` is produced, containing:
     - Key hypotheses, uncertainties, and metrics from the pipeline.
     - Relevant tags/domains and expert suggestions.
   - The sandbox orchestration layer passes `SandboxTask` plus available tools to the tool‑calling LLM, which plans and executes research steps.

3. **Sandbox → Trace Archive**
   - Every research step (tool call, result, and explanation) is appended to a persistent `ReasoningTrace` object, linked to the original user query and Mycelium run.
   - The archive is stored in a queryable format (JSONL or a small DB) and exposed via backend APIs.

4. **Mycelium + Sandbox Traces → Conversational LLM**
   - When generating a user‑visible answer, the conversational LLM receives:
     - Original question.
     - Mycelium pipeline summary (structured, not free‑text only).
     - Selected reasoning‑trace snippets from the sandbox.
   - It is instructed to *explain and synthesize* these artifacts, not to invent new facts outside them.

5. **Backend → Web UI**
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

### Milestone 2 – Reasoning Trace Archive (Core, No Sandbox Yet)

**Goal:** Capture Mycelium’s own reasoning state in a structured, queryable form.

1. Design a `ReasoningTrace` schema:
   - `trace_id`, `timestamp`, `user_query`.
   - Serialized `MyceliumRunSummary`.
   - Optional links to sandbox experiments (initially empty).
2. Implement a lightweight trace store:
   - Start with JSONL files (e.g. `traces/2026-05-*.jsonl`) or a tiny SQLite DB.
   - Provide append‑only API: `append_trace(trace: ReasoningTrace)`.
   - Provide read API: list recent traces, get by `trace_id`.
3. Hook `/api/query` to:
   - Create a `ReasoningTrace` for each request.
   - Persist it after the Mycelium run completes.
4. Expose `/api/traces/recent` to fetch recent runs for the UI.

### Milestone 3 – Sandbox Orchestration Layer

**Goal:** Introduce a pluggable sandbox that can run experiments and log detailed reasoning, without yet wiring in many heavy tools.

1. Define `SandboxTask` and `SandboxResult` models:
   - `SandboxTask`: inputs from Mycelium (hypotheses, uncertainty flags, domains, tags, metrics).
   - `SandboxResult`: list of `SandboxStep` objects (tool name, input, output, commentary, status).
2. Implement a `SandboxManager`:
   - Functions to create tasks from `MyceliumRunSummary` given simple heuristics (e.g. low confidence, ambiguous routing, new domain).
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
     - Describes Mycelium’s decisions and any sandbox experiments.
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

3. **Sandbox Panel**
   - Render `SandboxResult` as a step‑by‑step timeline:
     - Tool used.
     - Input parameters.
     - Output summary.
     - Commentary from the tool‑calling LLM.

4. **Trace / History Panel**
   - List recent `ReasoningTrace` entries via `/api/traces/recent`.
   - Allow clicking a past trace to:
     - Reload its pipeline state.
     - Replay or inspect its sandbox steps.
     - Ask the conversational agent follow‑up questions against that trace.

### Milestone 6 – Hardening and Constraints

**Goal:** Ensure the PoC behaves safely and predictably.

1. **Access Control / Keys**
   - Centralize all external API keys (LLM providers, search, etc.) via environment variables and a small config layer.
2. **Resource Limits**
   - Enforce per‑query caps on:
     - Number of sandbox tool calls.
     - Maximum wall‑clock time for sandbox runs.
     - Maximum stored trace size.
3. **Telemetry & Debugging**
   - Add structured logging for:
     - Each Mycelium run (input, route, expert decision, validation result).
     - Each sandbox step (tool, duration, success/failure).
   - Ensure logs do not leak sensitive user data outside the sandbox.

---

## 5. Implementation Order (Checklist View)

1. **Schema Stabilization**
   - [ ] Define and adopt `MyceliumRunSummary` on the backend.
   - [ ] Update `/api/query` to return it.

2. **Trace Archive**
   - [ ] Implement `ReasoningTrace` and JSONL/SQLite store.
   - [ ] Wire traces into `/api/query` and add `/api/traces/recent`.

3. **Sandbox Layer**
   - [ ] Define `SandboxTask`, `SandboxResult`, and `SandboxStep`.
   - [ ] Implement `SandboxManager` with minimal safe tools.
   - [ ] Integrate tool‑calling LLM (Ollama primary, Hugging Face secondary) and attach results to traces.

4. **Conversational Agent**
   - [ ] Implement `ConversationAgent` and `/api/chat`.
   - [ ] Ensure prompts use Mycelium + sandbox outputs as the primary context.
   - [ ] Route conversational calls through the same LLM provider config (Ollama primary, Hugging Face secondary).

5. **UI Integration**
   - [ ] Add pipeline, sandbox, and trace panels to the web UI.
   - [ ] Switch chat responses to conversational LLM once stable.

6. **Hardening**
   - [ ] Centralize configuration and keys.
   - [ ] Add resource guards and structured logging.

This plan should be refined as implementation progresses, but it provides a concrete roadmap from the current console demo to a full PoC with a conversational agent, sandbox research layer, and reasoning trace archive, all wired into the Mycelium architecture.
