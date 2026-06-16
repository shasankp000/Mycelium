# Mycelium Frontend Redesign Plan (Hardened Revision)
## Chat UI Overhaul + Reasoning Mode Selector + Force-Directed Graph View

> **Branch:** `web-ui-prototype`
> **Scope:** `web-ui/` only — zero changes to Python pipeline unless noted under "Backend additions"
> **Stack:** Next.js (Pages Router), TypeScript, existing SSE pipeline, new deps: `react-force-graph-2d`, `framer-motion`
> **Revision:** Hardened scalability + runtime observability revision

---

# 1. Current State Audit

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

```text
routing | heartbeat | graph_routing | graph_expert_init
graph_coverage_report | sandbox_plan | sandbox_tool/<n>
sandbox_tool/<n>_ok | sandbox_tool/<n>_err | expert_decision
done | error
```

`pipeline_event.py` already has a structured `PipelineEvent` dataclass — this is the ground truth for graph nodes.

---

# 2. Feature Scope

## 2.1 Reasoning Mode Selector

Three modes control **DAG+DFS depth** sent to the backend as a query param `?mode=fast|smart|researcher`.

| Mode | Icon | DAG layers explored | DFS max depth | Use case |
|---|---|---|---|---|
| **Fast** | ⚡ | Layer 0 + Layer 1 only | 1 | Simple factual, low latency |
| **Smart** | 🧠 | Layer 0 → Layer 1 → Phase 2 expert | 3 | Default; most queries |
| **Researcher** | 🔬 | All layers + patch DAG + multi-lens | 5 | Complex, multi-domain, evidence-heavy |

The selector lives in the input row as a pill toggle — **not** a settings menu. It persists in `localStorage`.

Backend addition required: `run_workflow.py` reads the `mode` param and passes it to `layer1_router` and `patch_dag` to gate depth.

---

## 2.2 Chat Bubble Redesign

Replace the current cluttered bubble with a clean two-zone layout:

```text
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

## 2.3 Reasoning Graph View (force-directed)

The centrepiece of this redesign. A full-panel overlay (not a sidebar) that renders after each response using `react-force-graph-2d`.

The graph is treated as:

- reasoning visualization,
- runtime introspection,
- semantic observability,
- replay/debugging infrastructure.

NOT merely a decorative debug panel.

---

### 2.3.1 Graph Data Model

Each SSE event maps to a graph node:

```typescript
interface GraphNode {
  id: string;
  label: string;
  kind: NodeKind;
  state: 'running' | 'done' | 'error' | 'skipped';
  layerDepth: number;
  metadata: Record<string, unknown>;

  // semantic weighting
  confidence?: number;
  leverage?: number;
  centrality?: number;

  // visual
  color?: string;
  size?: number;
  opacity?: number;

  // clustering
  clusterId?: string;
}

interface GraphEdge {
  source: string;
  target: string;
  label?: string;
  weight?: number;
  opacity?: number;
}

type NodeKind =
  | 'pipeline_stage'
  | 'expert'
  | 'tool_call'
  | 'reasoning_step'
  | 'evidence_node'
  | 'decision_point'
  | 'cluster_node'
  | 'contradiction_node'
  | 'synthesis_node';
```

---

### 2.3.2 Node → Colour Mapping

| Kind | Colour |
|---|---|
| `pipeline_stage` | `#6366f1` (indigo) |
| `expert` | `#10b981` (emerald) |
| `tool_call` | `#f59e0b` (amber) |
| `reasoning_step` | `#64748b` (slate) |
| `evidence_node` | `#3b82f6` (blue) |
| `decision_point` | `#ef4444` (red) |
| `cluster_node` | `#8b5cf6` (violet) |
| `contradiction_node` | `#dc2626` |
| `synthesis_node` | `#22c55e` |

---

### 2.3.3 Subgraph Layout Zones

Subgraphs are visually connected but physically semi-isolated.

Layout zones:

```text
Top:
- pipeline DAG

Center:
- reasoning subgraph

Bottom/right:
- tool + evidence subgraph
```

Each zone has:

- independent force-group centering
- bounded positional constraints
- local clustering behavior
- independent stabilization

Edges may connect zones visually,
but nodes should not freely drift between zones.

---

### 2.3.4 Interaction Model (Obsidian-style)

- **Pan/zoom**: mouse drag + scroll wheel
- **Node click**: opens a detail drawer on the right with full metadata for that node
- **Node hover**: tooltip showing `label`, `state`, elapsed ms
- **Subgraph toggle**: three buttons top-left: `Pipeline | Reasoning | Evidence`
- **Force layout controls**: `[⟳ Reset layout] [→ Hierarchy] [○ Radial]`

Hierarchy mode:
- computes DAG positions once
- freezes layout afterwards

Radial mode:
- Query node fixed at centre
- layers radiate outward by `layerDepth`

Live animation:
- nodes fade in during active reasoning
- edges animate during insertion
- graph settles progressively during execution

---

### 2.3.5 Reasoning Mode → Graph Depth

| Mode | Nodes visible | Subgraphs |
|---|---|---|
| Fast ⚡ | Layer 0 + Routing only | Pipeline only |
| Smart 🧠 | Full pipeline + summarized reasoning | Pipeline + summarized reasoning |
| Researcher 🔬 | Full pipeline + full DFS + evidence DAG | All subgraphs |

---

### 2.3.6 Graph Stabilization

The force simulation MUST NOT run indefinitely.

Graph lifecycle:

```text
live SSE updates
→ animated force simulation
→ stabilization timeout
→ physics freeze
→ exploration mode
```

Freeze triggers:
- 2–3 seconds without node arrival
- OR receipt of `done` / `error`

After freeze:
- node positions become fixed
- force simulation pauses
- graph enters low-CPU navigation mode

Toolbar action:

```text
[⟳ Resume simulation]
```

may temporarily re-enable physics.

---

### 2.3.7 Semantic Clustering

Large graphs MUST support clustering.

When node count exceeds configurable thresholds:

- reasoning nodes collapse into clusters
- repetitive DFS branches compress
- low-confidence paths aggregate
- evidence trees summarize automatically

Example:

```text
Reasoning Cluster (27 nodes)
```

Clusters are expandable on click.

Cluster metadata:
- node count
- average confidence
- dominant domain
- max depth

---

### 2.3.8 Semantic Priority Rendering

Visual rendering reflects semantic importance.

Examples:
- contradiction nodes → larger + brighter
- synthesis nodes → emphasized
- low-confidence branches → faded
- highly influential evidence → stronger edge opacity
- inactive branches → reduced saturation

Priority derives from:
- confidence
- leverage score
- graph centrality
- contradiction significance
- synthesis relevance

---

### 2.3.9 Graph Snapshot Persistence

After query completion:

persist:
- nodes
- edges
- metadata
- timestamps
- reasoning mode
- stabilization state
- graph schema version

as a `GraphSnapshot` object.

Snapshots support:
- replay
- debugging
- semantic archaeology
- contradiction analysis
- future analytics

---

# 3. File Structure After Redesign

```text
web-ui/
├── pages/
│   └── index.tsx
├── components/
│   ├── CalibrationGate.tsx
│   ├── HomeScreen.tsx
│   ├── ChatBubble.tsx
│   ├── ReasoningGraph/
│   │   ├── index.tsx
│   │   ├── graphBuilder.ts
│   │   ├── NodeDetailDrawer.tsx
│   │   ├── SubgraphControls.tsx
│   │   ├── LayoutToolbar.tsx
│   │   ├── GraphSnapshotLoader.tsx
│   │   └── ClusterNode.tsx
│   ├── ModeSelector.tsx
│   ├── SandboxPanel.tsx
│   └── TracePanel.tsx
├── hooks/
│   ├── useSseStream.ts
│   ├── useGraphBuilder.ts
│   ├── useElapsedTick.ts
│   └── useGraphStabilization.ts
├── styles/
│   ├── Home.module.css
│   ├── ChatBubble.module.css
│   ├── ReasoningGraph.module.css
│   └── ModeSelector.module.css
└── types/
    ├── graph.ts
    └── pipeline.ts
```

---

# 4. Backend Additions Required

## 4.1 `mode` query param in `/api/v1/chat/stream`

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
```

---

## 4.2 Additional graph-relevant SSE events

| phase_name | When emitted | Key metadata fields |
|---|---|---|
| `graph_dfs_step` | Each DFS iteration | `depth`, `domain`, `score`, `parent_id` |
| `graph_tool_start` | Tool start | `tool_name`, `tool_index`, `query` |
| `graph_tool_done` | Tool end | `tool_name`, `duration_ms`, `result_preview` |
| `graph_synthesis_start` | Synthesis start | `expert_count`, `evidence_count` |
| `graph_cluster_expand` | Cluster expansion | `cluster_id`, `node_count` |
| `graph_snapshot_saved` | Snapshot persisted | `snapshot_id`, `node_count` |

---

## 4.3 Deterministic Graph Event Ordering

All SSE graph events MUST contain:

```text
sequence_number
event_id
timestamp
```

Graph insertion order MUST use:

```text
sequence_number
```

NOT raw arrival order.

This guarantees:
- replay determinism
- stable reconstruction
- consistent animations
- reproducible snapshots

---

## 4.4 Future Runtime Compatibility

The frontend graph/event system MUST remain transport-agnostic.

Do NOT tightly couple graph logic to:
- Python-specific orchestration
- temporary SSE layouts
- backend implementation details

Future runtime layers may migrate to:
- Rust-native graph engines
- distributed event buses
- websocket multiplexing
- replay services

Graph/event interfaces should remain:
- schema-based
- deterministic
- versionable
- transport-independent

---

## 4.5 Graph Schema Versioning

Each GraphSnapshot and live stream MUST contain:

```text
graph_schema_version
```

This protects:
- replay compatibility
- snapshot migration
- frontend evolution
- runtime upgrades

---

# 5. Implementation Phases

## Phase 1 — Structural cleanup

- Extract all components from `index.tsx`
- Extract hooks into `hooks/`
- Extract types into `types/`
- Install markdown dependencies
- Render assistant output via markdown
- Reduce `index.tsx` complexity

---

## Phase 2 — Mode selector

- Build `ModeSelector.tsx`
- Wire `?mode=` param into `useSseStream`
- Backend: mode-aware DAG depth gating
- Persist mode in `localStorage`

---

## Phase 3 — Graph data pipeline

- Define graph schema
- Build `useGraphBuilder.ts`
- Build deterministic graph insertion pipeline
- Add graph clustering support
- Unit-test graph reconstruction using fixture SSE payloads

---

## Phase 4 — Graph rendering

- Install `react-force-graph-2d`
- Build graph overlay
- Add graph stabilization/freeze logic
- Add semantic weighting renderer
- Add clustering + expansion controls
- Add snapshot persistence

---

## Phase 5 — Polish + responsive layout

- Dark-mode optimization
- Mobile graph simplification
- Keyboard shortcuts
- Accessibility improvements
- Graph memoization
- Force simulation disposal
- Large-graph stress testing

---

# 6. Dependencies to Add

```json
{
  "react-force-graph-2d": "^1.25.x",
  "react-markdown": "^9.x",
  "remark-gfm": "^4.x",
  "framer-motion": "^11.x"
}
```

`react-force-graph-2d` remains canvas-based for broad compatibility.

---

# 7. Memory + Performance Rules

## 7.1 Graph Memory Management

Frontend graph state MUST support pruning.

Rules:
- inactive graphs unload when hidden
- old snapshots lazily reload
- force simulations dispose on unmount
- avoid full-array graph reconstruction
- node metadata shallow-copy only

Target:

```text
Researcher mode remains responsive beyond 1000+ nodes
```

---

## 7.2 Mobile Constraints

On mobile devices:
- reasoning graphs default to summarized mode
- evidence graph collapsed by default
- node labels selectively rendered
- force simulation freezes sooner
- max live node count reduced

Full expansion remains optional.

---

# 8. Design Principles

1. **The graph is the reasoning** — not an optional debug panel.
2. **Nodes arrive live** — users watch reasoning unfold in real time.
3. **Click everything** — all nodes and edges are interactive.
4. **Mode selector is ambient** — always visible near input.
5. **Subgraphs are composable** — pipeline, reasoning, and evidence layers remain independently controllable.
6. **No clutter in the bubble** — detailed introspection lives in the graph.
7. **The frontend is a semantic runtime observability interface** — not merely a chatbot shell.
8. **Physics is temporary** — stable exploration matters more than perpetual animation.
9. **Graph state is replayable** — all visual state should support deterministic reconstruction.
10. **Reasoning complexity must scale progressively** — casual users see summarized cognition; researchers may inspect the full DAG.

---

# 9. Finalized Architectural Decisions

## 9.1 Graph Snapshot Persistence

Graph snapshots will persist BOTH:

- frontend-side for fast local replay/navigation
- backend-side for canonical historical persistence

This supports:
- replay
- debugging
- semantic archaeology
- contradiction tracing
- cross-session inspection
- future distributed observability

Frontend persistence is treated as:

```text
session-local acceleration layer
```

Backend persistence is treated as:

```text
canonical historical reasoning archive
```

---

## 9.2 Smart Mode Expandable DFS Clusters

Smart mode exposes:

```text
summarized reasoning clusters
```

with optional expansion.

This enables:
- progressive semantic disclosure
- cleaner UX for normal users
- optional deeper introspection without requiring full Researcher mode

Default Smart-mode behavior:

```text
summarized cognition
```

Expandable behavior:

```text
click → reveal underlying DFS branch structure
```

---

## 9.3 Edge Confidence Visualization

Edge confidence weighting is enabled by default.

However:

visual clutter prevention takes priority.

Implementation rules:
- opacity variance preferred over extreme thickness variance
- edge thickness scaling remains subtle
- dense graphs automatically reduce weight exaggeration
- optional "high-detail weighting" toggle may exist in Researcher mode

Goal:

```text
semantic emphasis without graph overload
```

---

## 9.4 Contradiction Propagation Animation

Researcher mode supports contradiction propagation visualization.

However:

full real-time propagation traversal animation is forbidden.

Reason:
- GPU cost
- visual chaos
- frontend instability
- cognitive overload

Instead:

contradiction animation is staged.

Recommended lifecycle:

```text
contradiction detected
→ affected cluster pulse
→ downstream nodes recolor/fade progressively
→ stabilization outcome rendered
```

Animation must remain:
- lightweight
- bounded
- interruptible
- scalable to large graphs

---

## 9.5 Historical Snapshot Comparison

Historical graph snapshots support:

```text
side-by-side comparison mode
```

This enables:
- ontology drift analysis
- reasoning evolution inspection
- contradiction-resolution comparison
- semantic stabilization analysis
- regression debugging
- future research tooling

Comparison mode should support:
- synchronized navigation
- node-diff highlighting
- edge-diff highlighting
- reasoning path divergence overlays

---

# 10. Final Architectural Principle

The frontend is NOT merely a chatbot shell.

It is:

```text
a semantic runtime observability interface
```

The graph is treated as:
- reasoning visualization
- runtime introspection
- semantic navigation
- replay/debugging infrastructure
- and future semantic analysis substrate.

Reasoning itself becomes:

```text
navigable structure
```

rather than:

```text
hidden implementation detail
```

