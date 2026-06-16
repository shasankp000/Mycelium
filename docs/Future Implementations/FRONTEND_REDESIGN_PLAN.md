# Mycelium Frontend Redesign Plan
## Chat UI Overhaul + Reasoning Mode Selector + Force-Directed Graph View

> **Branch:** `web-ui-prototype`  
> **Scope:** `web-ui/` only — zero changes to Python pipeline unless noted under "Backend additions"  
> **Stack:** Next.js (Pages Router), TypeScript, existing SSE pipeline, new deps: `react-force-graph-2d`, `framer-motion`

---

## 1. Current State Audit

### What exists (`web-ui/pages/index.tsx`, 64 kB monolith)

| Component | Status | Problem |
|---|---|---|
| `CalibrationGate` | ✅ solid | — |
| `HomeScreen` | ✅ clean | — |
| `ThinkingPanel` | ⚠️ cluttered | Linear list of raw SSE events in the loading bubble; no hierarchy |
| `PhaseIndicator` | ⚠️ redundant | Progress bar duplicates ThinkingPanel info |
| `TracePanel` | ⚠️ data-dense | Key-value grid; not visual; no graph |
| `SandboxPanel` | ⚠️ hidden | Right sidebar nobody opens |
| `LiveToolFeed` | ⚠️ buried | Inside loading bubble |
| Chat bubbles | ❌ cluttered | `<pre>` for assistant text; no markdown; trace toggle crammed inline |
| Reasoning mode | ❌ absent | No fast/smart/researcher selector |
| Graph view | ❌ absent | No force-directed reasoning graph |

### SSE event types already emitted by the pipeline

```
routing | heartbeat | graph_routing | graph_expert_init
graph_coverage_report | sandbox_plan | sandbox_tool/<n>
sandbox_tool/<n>_ok | sandbox_tool/<n>_err | expert_decision
done | error
```

`pipeline_event.py` already has a structured `PipelineEvent` dataclass — this is the ground truth for graph nodes.

---

## 2. Feature Scope

### 2.1 Reasoning Mode Selector

Three modes control **DAG+DFS depth** sent to the backend as a query param `?mode=fast|smart|researcher`.

| Mode | Icon | DAG layers explored | DFS max depth | Use case |
|---|---|---|---|---|
| **Fast** | ⚡ | Layer 0 + Layer 1 only | 1 | Simple factual, low latency |
| **Smart** | 🧠 | Layer 0 → Layer 1 → Phase 2 expert | 3 | Default; most queries |
| **Researcher** | 🔬 | All layers + patch DAG + multi-lens | 5 | Complex, multi-domain, evidence-heavy |

The selector lives in the input row as a pill toggle — **not** a settings menu. It persists in `localStorage`.

Backend addition required: `run_workflow.py` reads the `mode` param and passes it to `layer1_router` and `patch_dag` to gate depth.

---

### 2.2 Chat Bubble Redesign

Replace the current cluttered bubble with a clean two-zone layout:

```
┌──────────────────────────────────────────────────┐
│  Assistant bubble                                 │
│  ┌────────────────────────────────────────────┐  │
│  │  [Markdown-rendered answer text]           │  │
│  └────────────────────────────────────────────┘  │
│  ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──   │
│  [🔍 Reasoning graph]  [📋 Evidence]  [⏱ 4.2s]  │
└──────────────────────────────────────────────────┘
```

- Answer text renders with `react-markdown` + `remark-gfm`
- The three footer chips are always visible (not hidden behind a toggle)
- "Reasoning graph" chip opens the graph panel (Section 2.3)
- "Evidence" chip opens the sandbox panel (already exists)
- Elapsed time badge pulled from `phase_latencies`

---

### 2.3 Reasoning Graph View (force-directed)

The centrepiece of this redesign. A full-panel overlay (not a sidebar) that renders after each response using `react-force-graph-2d`.

#### 2.3.1 Graph Data Model

Each SSE event maps to a graph node:

```typescript
interface GraphNode {
  id: string;            // unique, e.g. "layer0", "routing", "expert:physics"
  label: string;
  kind: NodeKind;        // see below
  state: 'running' | 'done' | 'error' | 'skipped';
  layerDepth: number;    // 0–5 matching reasoning mode depth
  metadata: Record<string, unknown>;
  // visual
  color?: string;
  size?: number;
}

interface GraphEdge {
  source: string;
  target: string;
  label?: string;
  weight?: number;       // confidence / similarity score
}

type NodeKind =
  | 'pipeline_stage'    // layer0, routing, expert_init, validation, synthesis
  | 'expert'            // physics, chemistry, biology, …
  | 'tool_call'         // web_search, academic_search, calculator, …
  | 'reasoning_step'    // heartbeat / intermediate DFS step
  | 'evidence_node'     // each sandbox step result
  | 'decision_point';   // expert_decision, validation_decision
```

#### 2.3.2 Node → Colour Mapping

| Kind | Colour |
|---|---|
| `pipeline_stage` | `#6366f1` (indigo) |
| `expert` | `#10b981` (emerald) |
| `tool_call` | `#f59e0b` (amber) |
| `reasoning_step` | `#64748b` (slate) |
| `evidence_node` | `#3b82f6` (blue) |
| `decision_point` | `#ef4444` (red) |

#### 2.3.3 Subgraph Structure

The full graph is composed of three dedicated subgraphs connected by edges:

```
[Pipeline graph]  ──connect──►  [Reasoning subgraph]
                  ──connect──►  [Tool/Evidence subgraph]
```

**Pipeline subgraph** (always present):
```
Query → Layer0 → Routing → ExpertInit → ExpertDecision
      → Phase2:Expert(s) → Phase3:Validation → Synthesis → Answer
```

**Reasoning subgraph** (one per heartbeat/DFS step):
- Each `heartbeat` SSE event appends a `reasoning_step` node connected to the current pipeline stage
- In **Researcher** mode the DFS tree is fully visible as a directed acyclic subgraph with edge weights (similarity scores from `metadata.score`)

**Tool/Evidence subgraph**:
- Each `sandbox_tool/<n>` event creates a `tool_call` node
- Each completed tool result creates an `evidence_node` child
- Evidence nodes connect back to the `Synthesis` pipeline node with a dashed edge

#### 2.3.4 Interaction Model (Obsidian-style)

- **Pan/zoom**: mouse drag + scroll wheel (built-in to `react-force-graph-2d`)
- **Node click**: opens a detail drawer on the right with full metadata for that node
- **Node hover**: tooltip showing `label`, `state`, elapsed ms
- **Subgraph toggle**: three buttons top-left: `Pipeline | Reasoning | Evidence` — hide/show each subgraph layer
- **Force layout controls**: a mini toolbar bottom-right: `[⟳ Reset layout]  [→ Hierarchy]  [○ Radial]`
  - *Hierarchy* switches to a top-down DAG layout (dagre-d3 computed positions, then frozen)
  - *Radial* keeps the Query node at centre, layers radiate outward by `layerDepth`
- **Live animation**: while the query is in flight, nodes arrive one by one with a fade-in + spring physics settle; edges draw as animated dashes

#### 2.3.5 Reasoning Mode → Graph Depth

| Mode | Nodes visible | Subgraphs |
|---|---|---|
| Fast ⚡ | Layer 0 + Routing only | Pipeline only |
| Smart 🧠 | Full pipeline + top expert | Pipeline + Reasoning |
| Researcher 🔬 | Full pipeline + all DFS steps + full evidence tree | All three subgraphs |

---

## 3. File Structure After Redesign

```
web-ui/
├── pages/
│   └── index.tsx              (slimmed to ~500 lines — orchestration only)
├── components/
│   ├── CalibrationGate.tsx    (extracted, unchanged)
│   ├── HomeScreen.tsx         (extracted, add mode selector)
│   ├── ChatBubble.tsx         (NEW — markdown + footer chips)
│   ├── ReasoningGraph/
│   │   ├── index.tsx          (full-panel overlay, mounts react-force-graph-2d)
│   │   ├── graphBuilder.ts    (SSE event → GraphNode/GraphEdge converters)
│   │   ├── NodeDetailDrawer.tsx
│   │   ├── SubgraphControls.tsx
│   │   └── LayoutToolbar.tsx
│   ├── ModeSelector.tsx       (NEW — Fast/Smart/Researcher pill toggle)
│   ├── SandboxPanel.tsx       (extracted, unchanged)
│   └── TracePanel.tsx         (extracted, becomes secondary — graph is primary)
├── hooks/
│   ├── useSseStream.ts        (extracted from index.tsx)
│   ├── useGraphBuilder.ts     (NEW — accumulates GraphNodes from SSE stream)
│   └── useElapsedTick.ts      (extracted timer logic)
├── styles/
│   ├── Home.module.css        (existing)
│   ├── ChatBubble.module.css  (NEW)
│   ├── ReasoningGraph.module.css (NEW)
│   └── ModeSelector.module.css (NEW)
└── types/
    ├── graph.ts               (GraphNode, GraphEdge, NodeKind types)
    └── pipeline.ts            (PipelineTrace, SseEvent, etc. — extracted)
```

---

## 4. Backend Additions Required

### 4.1 `mode` query param in `/api/v1/chat/stream`

In `broadcast_api.py`:
```python
@router.get("/chat/stream")
async def chat_stream(text: str, mode: str = "smart"):
    ...
    async for event in run_mycelium_workflow([text], mode=mode, ...):
        yield sse_frame(event)
```

In `run_workflow.py`:
```python
def run_mycelium_workflow(sentences, mode="smart", ...):
    depth_map = {"fast": 1, "smart": 3, "researcher": 5}
    max_depth = depth_map.get(mode, 3)
    # pass max_depth to layer1_router and patch_dag
```

### 4.2 Additional graph-relevant SSE events

New events to emit (all use existing `PipelineEvent` schema, just new `phase_name` values):

| phase_name | When emitted | Key metadata fields |
|---|---|---|
| `graph_dfs_step` | Each DFS iteration in patch_dag | `depth`, `domain`, `score`, `parent_id` |
| `graph_tool_start` | Start of each sandbox tool | `tool_name`, `tool_index`, `query` |
| `graph_tool_done` | End of each sandbox tool | `tool_name`, `duration_ms`, `result_preview` |
| `graph_synthesis_start` | Start of conversation_agent synthesis | `expert_count`, `evidence_count` |

These replace the raw `sandbox_tool/<n>` phase strings with structured events that carry enough data for proper graph node construction.

---

## 5. Implementation Phases

### Phase 1 — Structural cleanup (no new features)
- Extract all components from `index.tsx` into `components/`
- Extract all hooks into `hooks/`
- Extract types into `types/`
- Install `react-markdown`, `remark-gfm`
- Render answer text as markdown in `ChatBubble`
- Result: `index.tsx` drops from 64 kB to ~15 kB

### Phase 2 — Mode selector
- Build `ModeSelector.tsx` pill toggle
- Wire `?mode=` param into `useSseStream`
- Backend: add `mode` param to `run_workflow.py` (fast → skip patch_dag, researcher → full depth)
- Persist choice in `localStorage`

### Phase 3 — Graph data pipeline
- Define `types/graph.ts`
- Write `hooks/useGraphBuilder.ts` — consumes SSE stream, builds nodes/edges incrementally
- Write `components/ReasoningGraph/graphBuilder.ts` — pure SSE → graph converters
- Unit-test `graphBuilder.ts` with fixture SSE payloads

### Phase 4 — Graph rendering
- Install `react-force-graph-2d`
- Build `ReasoningGraph/index.tsx` with live node arrival animation
- Build `NodeDetailDrawer.tsx`
- Build `SubgraphControls.tsx` + `LayoutToolbar.tsx`
- Wire to `ChatBubble` "Reasoning graph" chip as a slide-up panel

### Phase 5 — Polish + responsive layout
- Dark mode graph colours (already matches existing dark theme)
- Mobile: graph view replaces full screen; pinch-zoom
- Keyboard: `Esc` closes graph panel, `G` toggles it
- Accessibility: all nodes have `aria-label`, focus ring on click
- Performance: memoize `graphBuilder` output; only rerender on node-count change

---

## 6. Dependencies to Add

```json
{
  "react-force-graph-2d": "^1.25.x",
  "react-markdown": "^9.x",
  "remark-gfm": "^4.x",
  "framer-motion": "^11.x"
}
```

`react-force-graph-2d` is a thin wrapper over `d3-force` + Canvas — no WebGL required, works on all devices.

---

## 7. Design Principles

1. **The graph is the reasoning** — it is not an optional debug panel. It is the primary way users understand what Mycelium did.
2. **Nodes arrive live** — the graph builds in real-time as SSE events arrive, not post-hoc. Users watch the reasoning unfold.
3. **Click everything** — every node, every edge is interactive. Nothing is decorative.
4. **Mode selector is ambient** — it sits next to the input, always visible, a one-click choice. Not buried in settings.
5. **Subgraphs are composable** — the pipeline graph, reasoning graph, and evidence graph are independent layers that can be toggled. In Fast mode you see a clean three-node pipeline. In Researcher mode you see the full sprawling DAG.
6. **No clutter in the bubble** — the chat bubble contains only the answer and three chips. All diagnostic detail lives in the graph.

---

## 8. Open Questions

- [ ] Should the reasoning graph persist across multiple messages as a single growing graph, or reset per message? (Recommendation: per-message, with a "history" button that loads a previous message's graph)
- [ ] Should `graph_dfs_step` events be emitted in **Smart** mode too, just at shallower depth? (Recommendation: yes, max 2 levels)
- [ ] Should edge weights (confidence/similarity scores) be visible as edge thickness by default, or opt-in? (Recommendation: opt-in via toolbar)
