# Mycelium — Architecture Reference

**Branch:** `web-ui-prototype`  
**Last updated:** June 2026  
**Scope:** Full backend pipeline + frontend client  
**Status:** Research Preview

---

## 0. Repository Layout

```
Mycelium/
├── mycelium/
│   └── pipeline/               ← Core reasoning engine (Python)
│       ├── layer0/             ← Gatekeeper: NLP pre-processing, classification, routing
│       ├── phase2/             ← Evidence subsystem: finding, scoring, contradiction
│       ├── phase3/             ← Deep synthesis pipeline
│       ├── predicates/         ← Semantic IR layer (PredicateFrame engine)
│       ├── expert_post_check/  ← Post-processing validators on expert outputs
│       ├── run_workflow.py     ← Master orchestrator (69 KB — largest file)
│       ├── layer1_router.py    ← Post-L0 domain routing (49 KB)
│       ├── broadcast_api.py    ← SSE/streaming event broadcast (27 KB)
│       ├── mcp_tools_server.py ← MCP tool integration server (26 KB)
│       ├── multi_lens_router.py← Multi-perspective value-laden routing (22 KB)
│       ├── config_loader.py    ← YAML/env config system (21 KB)
│       ├── pipeline_event.py   ← Typed event bus (18 KB)
│       ├── patch_dag.py        ← Knowledge patch DAG (15 KB)
│       ├── auto_semantic_clusterer.py ← Auto-clustering of semantic domains (14 KB)
│       ├── expert_filter.py    ← Domain expert selection/filtering (18 KB)
│       ├── dynamic_signature_manager.py ← Runtime DSPy signature mgmt (14 KB)
│       ├── fusion_engine.py    ← DST-based belief fusion (10 KB)
│       ├── conversation_agent.py ← Conversational context agent (12 KB)
│       ├── domain_tool_planner.py ← Domain-specific tool planning (13 KB)
│       ├── llm_providers.py    ← Multi-provider LLM abstraction (8 KB)
│       ├── model_registry.py   ← Model lifecycle registry (10 KB)
│       ├── lexis_bridge.py     ← Lexis knowledge graph bridge (6 KB)
│       ├── lexis_condenser.py  ← Knowledge condensation pass (7 KB)
│       ├── optimization_engine.py ← DSPy program optimization (5 KB)
│       ├── mcp_client.py       ← MCP client protocol (3 KB)
│       ├── api_models.py       ← Pydantic API schemas (15 KB)
│       └── ...
├── web-ui/                     ← React + Vite frontend
│   └── src/
│       ├── components/
│       ├── pages/
│       └── themes/
├── docs/
│   └── frontend-upgrade/       ← UI specs (this document lives here)
└── README.md
```

---

## 1. System Overview

Mycelium is a **multi-phase semantic reasoning runtime**. It is not an LLM wrapper. Every user query passes through a structured pipeline that classifies the query's epistemic type, decomposes it into structured semantic predicates, routes it to domain-specific expert models, retrieves and scores evidence with a refutation-first strategy, fuses belief states using Dempster-Shafer Theory, and synthesises a final response with full provenance.

The pipeline has **three major phases**, each with internal sub-stages:

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────┐
│  PHASE 0 — L0 GATEKEEPER (layer0/)              │
│  NLP Pre-processing → Manipulation Detection    │
│  → Objectivity Classification → Value Assumption│
│    Extraction → Query Routing                   │
└──────────────┬──────────────────────────────────┘
               │
       ┌───────┴────────┐
       │                │
       ▼                ▼
  Objective         Value-laden
  Query Path        Query Path
       │                │
       ▼                ▼
┌──────────────┐  ┌────────────────────────────────┐
│  PHASE 1     │  │  MULTI-LENS ROUTER             │
│  layer1_router│  │  multi_lens_router.py          │
│  + Expert    │  │  Multi-perspective evidence    │
│  System      │  │  engine (no synthetic opinion) │
└──────┬───────┘  └────────────────┬───────────────┘
       │                           │
       ▼                           │
┌──────────────────────────────────▼───────────────┐
│  PHASE 2 — EVIDENCE SUBSYSTEM (phase2/)           │
│  Evidence Finder → Evidence Scorer               │
│  → Contradiction Integrator                      │
└──────────────────────────────┬───────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────┐
│  PREDICATE ENGINE (predicates/)                  │
│  Text → PredicateFrames → Negation/Scope         │
│  → Predicate Store (persistent semantic graph)  │
└──────────────────────────────┬───────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────┐
│  PHASE 3 — SYNTHESIS PIPELINE (phase3/)          │
│  DST Fusion → Hypothesis Evaluation             │
│  → Reasoning Synthesis → Response Generation   │
└──────────────────────────────┬───────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────┐
│  BROADCAST + STREAMING (broadcast_api.py)        │
│  SSE event stream → Web UI                      │
└──────────────────────────────────────────────────┘
```

---

## 2. Phase 0 — L0 Gatekeeper (`pipeline/layer0/`)

L0 is the entry gate. It determines the epistemic type of the incoming query before any reasoning begins. It has no opinion. It only classifies.

### 2.1 Sub-components

| File | Role |
|------|------|
| `nlp_preprocessor.py` | Tokenisation, dependency parsing, entity extraction, spaCy/NLTK pipeline (19 KB) |
| `manipulation_detector.py` | Detects instruction override, conclusion-forcing, cherry-pick requests, framing traps (15 KB) |
| `objectivity_classifier.py` | Classifies query as OBJECTIVE / VALUE_LADEN / MIXED / UNANSWERABLE (11 KB) |
| `value_assumption_extractor.py` | Extracts hidden value premises embedded in apparently factual questions (14 KB) |
| `router.py` | Routes query to the correct downstream path based on L0 output (11 KB) |
| `dst_fusion.py` | Dempster-Shafer Theory fusion engine — also used upstream in Phase 3 (19 KB) |
| `probability_calibration.py` | Platt scaling / isotonic regression calibration for classifier outputs (16 KB) |
| `dataset_logger.py` | Logs L0 decisions to training datasets for future fine-tuning (4 KB) |
| `train_layer0_models.py` | Training harness for the local L0 classifiers (43 KB) |

### 2.2 L0 Decision Tree

```
Incoming query
    │
    ├─ Manipulation signal detected?
    │       YES → ManipulationRefusal + explanation + honest reframe suggestion
    │       NO  → continue
    │
    ├─ Classify: OBJECTIVE / VALUE_LADEN / MIXED / UNANSWERABLE
    │
    ├─ OBJECTIVE     → extract hidden value assumptions → Phase 1 (full pipeline)
    ├─ VALUE_LADEN   → extract value premises → Multi-Lens Router
    ├─ MIXED         → split into objective sub-claims + value sub-claims, route each
    └─ UNANSWERABLE  → explain why + suggest what a better question would look like
```

### 2.3 Manipulation Detection Categories

The `manipulation_detector.py` identifies:
- **Instruction override** — attempts to override system prompt or safety context
- **Conclusion forcing** — "prove that X is true" / "why is X the best"
- **Cherry-pick requests** — "give me only evidence for Y"
- **Framing traps** — presupposition injection ("since X is clearly wrong...")
- **False dichotomies** — "is it A or B?" when C, D, and E all exist

---

## 3. Phase 1 — Domain Routing & Expert System (`layer1_router.py`, `layer2_expert_loader.py`)

### 3.1 Layer 1 Router (`layer1_router.py`, 49 KB)

After L0 classification, the L1 router:
1. Embeds the query semantically
2. Scores against the registered domain graph (via `model_registry.py`)
3. Selects relevant domain expert heads
4. Constructs a per-query reasoning plan via `domain_tool_planner.py`
5. Optionally invokes MCP tools via `mcp_tools_server.py`

The router is the largest single component after `run_workflow.py` because it handles the full domain-matching, tool-calling, and plan-construction logic before any evidence retrieval starts.

### 3.2 Expert System

| File | Role |
|------|------|
| `layer2_expert_loader.py` | Loads, validates, and caches domain expert LoRA heads (5 KB) |
| `expert_filter.py` | Filters and ranks candidate experts by query-domain match score (18 KB) |
| `expert_post_check/` | Post-processing validators that run after each expert produces output |
| `expert_system_init_api.py` | API for initialising and registering new expert domains at runtime (3 KB) |
| `dynamic_signature_manager.py` | Runtime management of DSPy signatures for each expert (14 KB) |

### 3.3 Model Registry & Lifecycle (`model_registry.py`)

The model registry implements the **TRM domain lifecycle** as actual code:

```
CREATING → HOT → WARM → COLD → REMEMBERING → DEPRECATED → ARCHIVED
```

- **HOT**: Actively loaded in GPU memory, receives live traffic
- **WARM**: On disk, can be loaded within seconds
- **COLD**: Compressed, requires decompress + load
- **REMEMBERING**: Being retrained/updated with new patch data
- **ARCHIVED**: Frozen, never deleted, retrievable for audit

Domain transitions are event-driven via `pipeline_event.py`. No hard deletion ever occurs.

### 3.4 Auto-Semantic Clusterer (`auto_semantic_clusterer.py`)

Monitors live query traffic and automatically proposes new domain splits when:
- A domain's query distribution drifts beyond a threshold
- Sub-cluster coherence exceeds a split-readiness score
- Manual override by admin is triggered

New domains are created non-destructively — the parent domain remains intact.

---

## 4. Multi-Lens Router (`multi_lens_router.py`, 22 KB)

Value-laden queries take a completely separate path. The Multi-Lens Router:

1. Decomposes the value question into its constituent ethical/empirical dimensions
2. For each dimension, retrieves evidence from multiple ideological/methodological lenses
3. Scores each lens's evidence independently (no cross-contamination)
4. Returns a structured multi-perspective response — no synthetic opinion is generated
5. Labels each perspective with its underlying value assumptions (surfaced by L0's `value_assumption_extractor.py`)

This is the architectural operationalisation of the Is/Ought distinction. The system cannot produce a "recommendation" on a value-laden question. It can only produce an evidence map.

---

## 5. Phase 2 — Evidence Subsystem (`pipeline/phase2/`)

### 5.1 Sub-components

| File | Role |
|------|------|
| `evidence_finder.py` | Retrieval with refutation-first ordering: counterevidence first, confirming evidence second (12 KB) |
| `evidence_scorer.py` | Multi-dimensional evidence scoring: source credibility, recency, corroboration count, methodological quality (13 KB) |
| `contradiction_integrator.py` | Detects, classifies, and integrates contradictions between evidence items (13 KB) |
| `evidence_types.py` | Typed evidence schema: EMPIRICAL, STATISTICAL, TESTIMONIAL, THEORETICAL, EXPERT_CONSENSUS (4 KB) |
| `pipeline.py` | Phase 2 pipeline orchestrator (11 KB) |
| `phases/` | Sub-phases of the evidence pipeline (decomposed retrieval stages) |
| `utils/` | Shared utilities: deduplication, normalisation, provenance tagging |
| `config/` | Evidence budget configuration per reasoning mode (Quick / Smart / Researcher) |

### 5.2 Refutation-First Retrieval

Standard RAG systems retrieve supporting evidence and then optionally check for contradictions. Mycelium inverts this:

```
Claim received
    │
    ▼
1. Generate logical negation of the claim
2. Retrieve evidence FOR the negation (i.e., against the claim)
3. Evaluate strength of refutation evidence
4. If strong refutation found → halt confirming retrieval, report refutation
5. If weak/absent refutation → retrieve confirming evidence
6. Score both sets independently
7. Return evidence map with confidence delta
```

For universal claims ("all X are Y"), a **single strong counterexample** halts retrieval immediately. This prevents the system from burying a decisive refutation under a pile of confirming evidence.

### 5.3 Evidence Scoring Dimensions

`evidence_scorer.py` scores each evidence item on:
- **Source credibility** — peer-reviewed > preprint > expert commentary > general publication > unverified
- **Recency weight** — configurable decay function per domain (physics: slow decay, politics: fast decay)
- **Corroboration count** — how many independent sources reach the same conclusion
- **Methodological quality** — RCT > observational > correlational > anecdotal (domain-dependent)
- **Chain of reasoning** — how many inferential steps separate the evidence from the claim

---

## 6. Predicate Engine (`pipeline/predicates/`)

The Predicate Engine is Mycelium's semantic intermediate representation layer. Instead of reasoning over raw text, the system converts all claims into structured `PredicateFrame` objects.

### 6.1 Sub-components

| File | Role |
|------|------|
| `predicate_types.py` | PredicateFrame dataclass + predicate type taxonomy (15 KB) |
| `predicate_extractor.py` | Text → PredicateFrame conversion using NLP + LLM (19 KB) |
| `predicate_negator.py` | Generates structurally correct logical negations of predicates (20 KB) |
| `predicate_store.py` | Persistent predicate graph with provenance lineage (9 KB) |
| `predicate_relations.py` | Relation type taxonomy: IMPLIES, CONTRADICTS, SUPPORTS, SUBSUMES, etc. (2 KB) |
| `predicate_pipeline.py` | Orchestrates the full predicate processing pass (7 KB) |

### 6.2 PredicateFrame Schema

```python
PredicateFrame(
    subject:           str,           # "vaccines"
    relation:          RelationType,  # CAUSES
    object:            str,           # "autism"
    negated:           bool,          # True → "vaccines do NOT cause autism"
    quantifier:        Quantifier,    # ALL / SOME / NONE / MOST
    predicate_type:    PredicateType, # CAUSAL / CORRELATIONAL / DEFINITIONAL / NORMATIVE
    falsifiable:       bool,          # NORMATIVE predicates → always False
    refutation_burden: float,         # Prior probability of refutation (0.0–1.0)
    confidence:        float,         # Current belief mass
    provenance:        List[Source],  # Full source chain
    timestamp:         datetime,
    lineage:           List[PredicateID]  # Parent predicates this was derived from
)
```

### 6.3 Key Invariant

**NORMATIVE predicates (value judgments) have `falsifiable = False` and are structurally excluded from contradiction trees.** This is the code-level enforcement of the Is/Ought boundary. A predicate like "universal healthcare is good" cannot be placed in a contradiction relationship with an empirical claim — the type system prevents it.

### 6.4 Predicate Negator (`predicate_negator.py`, 20 KB)

The largest file in the predicate subsystem. Generates linguistically and logically correct negations:
- "All X are Y" → "Some X are not Y" (quantifier-aware)
- "X causes Y" → "X does not cause Y" AND "Something other than X causes Y" (causal decomposition)
- "X is better than Y" → rejected (NORMATIVE, non-negatable in contradiction tree)

---

## 7. Patch System — Non-Destructive Knowledge Updates

### 7.1 Patch DAG (`patch_dag.py`, 15 KB)

New knowledge enters the system as **patches** — structured updates with full provenance. The Patch DAG tracks:
- Which domains were modified
- What was added vs. revised
- The full dependency graph between patches (a later patch may supersede an earlier one)
- Conflict detection between patches targeting the same predicate cluster

### 7.2 Patch Logging

| File | Role |
|------|------|
| `patch_batch_logger.py` | Batches patch events for async persistence (7 KB) |
| `patch_dataset_logger.py` | Logs accepted patches to fine-tuning datasets (6 KB) |

Patches are never hard-applied to a live domain. They flow through:

```
Patch proposed
    → Patch DAG conflict check
    → Staging validation
    → Expert post-check
    → Commit to WARM domain
    → Promote to HOT
```

---

## 8. DST Fusion Engine (`fusion_engine.py`, `layer0/dst_fusion.py`)

Dempster-Shafer Theory (DST) is used in two places:

1. **L0 classifier fusion** (`layer0/dst_fusion.py`): Multiple weak classifiers vote on query type; DST fuses their belief masses into a single classification with explicit uncertainty quantification.

2. **Phase 3 hypothesis fusion** (`fusion_engine.py`): After evidence retrieval, multiple hypothesis scores are fused into a final belief distribution. The output is not a point estimate — it is a belief interval `[support, plausibility]` for each hypothesis.

DST handles **ignorance** differently from probability theory. Where a Bayesian system assigns probability 0.5 to an unknown, DST assigns it to the `Θ` (full frame of discernment) — explicitly representing "we don't know" rather than "it's fifty-fifty."

---

## 9. Phase 3 — Synthesis Pipeline (`pipeline/phase3/`)

Phase 3 takes the fused evidence + predicate graph and produces the final structured response.

| File | Role |
|------|------|
| `pipeline.py` | Phase 3 orchestrator: hypothesis eval → synthesis → response formatting (18 KB) |
| `phases/` | Sub-phases: hypothesis generation, ranking, synthesis |
| `utils/` | Shared: citation formatter, confidence interval renderer, reasoning graph serialiser |
| `config/` | Synthesis parameters per mode (verbosity, citation depth, graph inclusion) |

The synthesis phase produces:
- A primary answer with confidence interval
- A full reasoning graph (nodes = predicates, edges = inference steps)
- A ranked list of supporting and refuting evidence with scores
- Explicit uncertainty callouts where evidence is weak or contradictory

---

## 10. Supporting Infrastructure

### 10.1 Run Workflow (`run_workflow.py`, 69 KB)

The master orchestrator. Coordinates all phases, manages async execution, handles cancellation, streams events to `broadcast_api.py`, and implements the three reasoning mode profiles:

| Mode | L0 depth | Evidence budget | Phase 3 depth | DST iterations |
|------|----------|-----------------|---------------|----------------|
| Quick | Lightweight | 5–10 sources | Single-pass | 1 |
| Smart | Full | 15–30 sources | Multi-pass | 3 |
| Researcher | Full + extended | 50–100 sources | Exhaustive | 5+ |

### 10.2 Broadcast API (`broadcast_api.py`, 27 KB)

All pipeline progress is streamed to the frontend via **Server-Sent Events (SSE)**. The web UI receives typed events as each pipeline stage completes:
- `layer0_complete` — classification result
- `evidence_found` — each evidence item as it's retrieved
- `predicate_extracted` — each PredicateFrame as it's created
- `contradiction_detected` — when a contradiction is found between evidence items
- `synthesis_complete` — final response with full reasoning graph

This is what powers the live reasoning graph display in the UI.

### 10.3 MCP Integration (`mcp_tools_server.py`, 26 KB; `mcp_client.py`, 3 KB)

Mycelium implements the **Model Context Protocol (MCP)** as both server and client. Domain experts can invoke external tools (web search, code execution, database queries, API calls) through the MCP server. All tool calls are tracked as first-class pipeline events with full provenance.

### 10.4 Lexis Knowledge Bridge (`lexis_bridge.py`, `lexis_condenser.py`)

The Lexis layer provides a structured knowledge graph that pre-populates certain domains with curated knowledge. `lexis_condenser.py` compresses verbose knowledge representations into compact predicate-compatible form before they enter the main pipeline.

### 10.5 Conversation Agent (`conversation_agent.py`, 12 KB)

Maintains conversational context across a session:
- Prior queries and their resolved predicate graphs
- Domain context established in earlier turns
- Open hypotheses from prior reasoning that are still unresolved
- User-expressed constraints ("assume X for this session")

### 10.6 LLM Provider Abstraction (`llm_providers.py`, 8 KB)

Multi-provider abstraction over OpenAI (GPT-4o), Anthropic (Claude 3.x), Google (Gemini), and local providers (Ollama, vLLM). Provider selection is per-component — L0 classifiers can use a fast local model while Phase 3 synthesis uses a larger remote model. All provider calls are tracked as pipeline events.

### 10.7 Optimisation Engine (`optimization_engine.py`, 5 KB)

Wraps DSPy's `BootstrapFewShot` and `MIPROv2` optimisers. Periodically re-optimises prompt signatures using logged query/response pairs from `patch_dataset_logger.py` and `layer0/dataset_logger.py`. Optimisation is applied per-domain and per-component — not globally — which is how the system improves without catastrophic forgetting.

---

## 11. Web UI Architecture (`web-ui/`)

The frontend is a **React + Vite** SPA that connects to the backend via the SSE broadcast stream and a REST API.

### Key Capabilities
- **Live reasoning graph** — renders the predicate DAG as the pipeline runs, node by node
- **Four visual themes** — AMOLED, Anthropic, Apple, Arc (CSS custom properties + `data-theme`)
- **Three reasoning modes** — Quick, Smart, Researcher (passed to backend via API)
- **Evidence panel** — shows each evidence item, its score, type, and source
- **Contradiction viewer** — highlights contradictions between evidence items
- **Confidence display** — DST belief intervals, not point probabilities

---

## 12. End-to-End Data Flow

```
User types query
        │
        ▼
[Web UI] → POST /query {query, mode}
        │
        ▼
[run_workflow.py] — master orchestrator
        │
        ├──▶ [layer0/nlp_preprocessor.py]         parse + tokenise
        ├──▶ [layer0/manipulation_detector.py]     detect adversarial patterns
        ├──▶ [layer0/objectivity_classifier.py]    classify epistemic type
        ├──▶ [layer0/value_assumption_extractor.py] extract hidden premises
        ├──▶ [layer0/router.py]                    route to objective or value path
        │
        ├──▶ [predicates/predicate_extractor.py]   query → PredicateFrames
        ├──▶ [predicates/predicate_negator.py]     generate logical negations
        │
        ├──▶ [layer1_router.py]                    match domains, build tool plan
        ├──▶ [expert_filter.py]                    select domain experts
        ├──▶ [layer2_expert_loader.py]             load expert LoRA heads
        │
        ├──▶ [phase2/evidence_finder.py]           refutation-first retrieval
        ├──▶ [phase2/evidence_scorer.py]           score all evidence items
        ├──▶ [phase2/contradiction_integrator.py]  detect + classify contradictions
        │
        ├──▶ [fusion_engine.py / layer0/dst_fusion.py]  DST belief fusion
        │
        ├──▶ [phase3/pipeline.py]                  hypothesis evaluation + synthesis
        │
        ├──▶ [predicates/predicate_store.py]       persist predicate graph
        ├──▶ [patch_dag.py]                        if new knowledge: queue patch
        │
        └──▶ [broadcast_api.py]                    stream all events via SSE

[Web UI] receives SSE stream → renders reasoning graph live → displays final answer
```

---

## 13. Key Design Invariants

These are not conventions — they are hard constraints enforced by the type system and architecture:

1. **NORMATIVE predicates cannot enter contradiction trees.** `predicate_types.py` enforces `falsifiable = False` for all value judgments.

2. **Refutation evidence is retrieved before confirming evidence**, always. The order in `evidence_finder.py` is non-configurable.

3. **No domain is ever hard-deleted.** The model registry lifecycle ends at ARCHIVED, not deleted.

4. **All pipeline events are typed and logged.** `pipeline_event.py` defines the full event schema. No untyped side effects.

5. **Value-laden queries never receive a synthetic opinion.** The multi-lens router produces an evidence map; it has no synthesis phase for normative conclusions.

6. **All LLM calls have provenance.** Every generated text fragment is tagged with the model, provider, prompt version, and timestamp.

7. **DST uncertainty is preserved through to the final response.** Confidence is displayed as a belief interval, not a single percentage.
