# Mycelium — AI Context Handoff Document

> **Purpose:** This document exists so any AI assistant (or the author) can be dropped into the Mycelium project mid-work without requiring a long verbal re-briefing. It captures the architectural understanding of the system, the current state of the codebase, and the exact work order in progress.

---

## 1. What Mycelium Is

Mycelium is **not** a chatbot, an LLM wrapper, or a retrieval-augmented generation (RAG) system. It is a **semantic cognition substrate with epistemic stabilization** — a system designed to reason about the *truth value* of claims, not merely to retrieve or generate text.

The core design thesis is:

> Most AI systems generate plausible-sounding output. Mycelium is designed to work out whether output is *true*, by decomposing claims, grounding them in evidence, propagating contradictions, and stabilizing knowledge over time — without catastrophically forgetting old domains as new ones are added.

It is architecturally inspired by neuroscience (mycelium-like distributed knowledge), continual-learning research, Dempster-Shafer belief theory, BFS+DFS graph reasoning, and Samsung's TRM (Transformer Routing Memory) architecture.

---

## 2. The Full Pipeline (As Designed)

### Stage 1 — Input Ingestion & Cleaning
- Raw text input is cleaned (garbage characters, encoding normalisation).
- An LLM assigns a **tag list** to the query.

### Stage 2 — MultiLensRouter (Domain Routing)
Tags are evaluated against known domains using **three simultaneous lenses**:
1. **Semantic similarity** — sentence-transformer embeddings vs. domain centroids.
2. **Spectral structural analysis** — vocabulary + syntactic fingerprinting.
3. **Ontological similarity** — object-level abstraction hierarchy (e.g. `String Theory → Physics → Science`).

Lens outputs are fused via a confidence prior to produce a domain routing decision. The MultiLensRouter is evolving into a full **semantic arbitration engine** — handling ontology reconciliation, relation-family inference, and predicate disambiguation beyond simple routing.

### Stage 3 — Layer 0 NLP Classification + DST Fusion
A set of NLP models trained at Layer 0 classifies the query into one of four poles:

| Class | Nature |
|---|---|
| Objective / Factual | Verifiable claim |
| Value-laden | Normative statement |
| Subjective | Opinion / personal framing |
| Manipulative | Jailbreak, adversarial, propaganda |

This is cross-checked with a **Dempster-Shafer Theory (DST) fusion check**, which returns a belief mass distribution across all four poles without collapsing uncertain cases to a single class. DST recurs downstream in reasoning worker aggregation.

### Stage 4 — Expert / SQLite-Shard Routing (Factual Path)
For factual queries, the system routes to domain experts. The **target architecture** (in progress) replaces BERT-based expert models with **SQLite-sharded expert stores** — structured domain knowledge that can be queried directly rather than inferred through a transformer. This eliminates inference overhead and hallucination risk for structured domains.

Each shard contains:
- `chunks(id, text, embedding, source, created_at, metadata_json)`
- `fts_chunks` — FTS5 lexical index for prefiltering
- `entities(entity_id, label, type, metadata_json)`
- `relations(src_id, rel_type, dst_id, weight)`
- `artifacts(key, value)`

Retrieval policy: FTS5 prefilter → vector similarity → graph/metadata reranking → top-k handoff to reasoning pipeline.

### Stage 5 — TRM Expert & Tool-Grounded Verification
The query reaches the **TRM expert** — a routing memory model that trains on routing decisions. Its role is distinct from the MultiLensRouter:

- MultiLensRouter answers: *"What does this probably mean?"*
- TRM answers: *"How has this meaning historically stabilised?"*

The TRM calls external **fact-fetching tools** to ground claims in verifiable sources. If confidence is sufficient, a response is returned. If not, the query falls to the deep reasoning pipeline.

### Stage 6 — L0→L6 Reasoning Pipeline
The deep reasoning pipeline operates on the **assumption of falsehood** — it starts by treating the claim as false and works backwards (adversarial proof-by-contradiction, inspired by discrete mathematics).

Uses a **BFS+DFS hybrid graph decomposition** constrained by a DAG (directed acyclic graph) to prevent cycles:

```
Depth 0: Smoking causes cancer
Depth 1: Smoking damages tissue | correlates with cancer | precedes cancer
Depth 2: DNA mutation | Oxidative stress | Carcinogens
```

Three reasoning modes control depth and iteration:
- **Quick** — fast, shallow BFS
- **Smart** — balanced BFS+DFS
- **Researcher** — deep iterative DFS with full evidence grounding

### Stage 7 — Epistemic IR Layer (Already Implemented)
All reasoning operates over a **Formal Intermediate Representation (IR)**. Every node in the reasoning graph carries:

- `SemanticSignature` — semantic hash, embedding, spectral signature, canonical form
- `ConfidenceState` — multi-dimensional (semantic, structural, epistemic, temporal, evidence confidence)
- `TemporalState` — typed temporal encoding (EXACT, INTERVAL, RELATIVE, HISTORICAL_ESTIMATE, UNKNOWN)
- `ProvenanceChain` — full lineage from source → decomposition path → reasoning chain

The reasoning memory system behaves like a **distributed semantic version-control system** — graphs are immutable, competing hypotheses coexist as branches, and TRM performs semantic merge arbitration analogous to Git merges but for *meaning* rather than text.

---

## 3. TRM v2 — The Target Architecture

The current TRM is a single model. The target architecture (TRM v2) replaces it with a **modular, non-destructively expandable system**:

| Component | Description |
|---|---|
| Frozen shared encoder | `z = E(x)` — frozen after stabilisation; provides shared latent space |
| Domain-isolated heads | `h_d = H_d(z)` — one lightweight head per domain; new domains get new heads, existing heads never modified |
| Gating network | Sparse top-k routing: `y = Σ w_d · H_d(z)` over active heads |
| DomainGraph registry | Graph `(V, E)` where each node = domain with centroid, spectral sig, head ref, SQLite shard ref, cold-storage ref, drift profile |
| Cold storage lifecycle | Every domain is Hot / Warm / Cold; cold domains are dequantised and replayed on reactivation |

**Three gate regimes:**
1. Bootstrap — gate fully trainable while foundational domains are being learned
2. Expansion — gate updates restricted to calibration for new domains only
3. Stability — gate frozen; only new heads or local adapters are trained

**Domain addition policies:**
- **Case A (distant domain):** New graph node + new isolated head + new SQLite shard
- **Case B (near-neighbour):** Child node initialised from nearest parent head, or patch/LoRA adapter on parent
- **Case C (ambiguous):** Provisional probationary node; promoted to full domain after persistent separability is confirmed

---

## 4. Key Architectural Principles (Summary)

- **No destructive retraining.** New domains allocate new parameters; existing heads are frozen by default.
- **Retrieval ≠ Reasoning.** Domain knowledge lives in SQLite shards. Lightweight reasoning logic lives in domain heads. These are separate concerns.
- **DST over scalar confidence.** Uncertainty is a belief mass distribution, not a float.
- **Immutable graphs.** Reasoning graphs are append-only, versioned like Git commits.
- **Contradiction is typed.** DIRECT, CONTEXTUAL, TEMPORAL, PROBABILISTIC_DISAGREEMENT — not binary.
- **Evidence lineage is mandatory.** Every conclusion traces back to its decomposition path and source.
- **Ontology is emergent.** Ontologies are derived from reasoning, not hardcoded.

---

## 5. Repository Structure & Branch Context

- **Repo:** `https://github.com/shasankp000/Mycelium`
- **Primary docs branch:** `web-ui-prototype` — contains all future implementation specs and architecture docs under `docs/Future Implementations/`
- **Active work branch:** To be forked fresh from `web-ui-prototype` (the previous `feature/trm-v02-clean-rebuild` was nuked)
- **Key docs:**
  - `docs/Future Implementations/Mycelium -- Semantic Architecture Consolidation Notes.md` — full IR, contradiction ontology, graph lifecycle, DST aggregation
  - `docs/Future Implementations/mycelium_trm_v2_theoretical_spec.md` — TRM v2 frozen encoder + isolated heads + DomainGraph + SQLite shard architecture
  - `docs/Future Implementations/Mycelium Reasoning Architecture -- Implementation Roadmap.md` — L0→L6 reasoning pipeline detail
  - `docs/REBUILD_NOTES.md` (on `web-ui-prototype`) — hardcoded domain bugs and known issues to fix

---

## 6. Current Work Order

The following four phases are to be executed **in sequence**, on a new branch forked from `web-ui-prototype`.

### Phase 1 — Replace BERT Experts with SQLite Shards

**What:** Remove BERT-based domain expert models and the expert pre-check inference gate entirely. Replace with per-domain SQLite shards using the layered retrieval policy (FTS5 → vector similarity → graph reranking → top-k handoff).

**Why:** Asking a transformer model "do you know this?" wastes compute, introduces latency, and risks hallucination. Querying a structured dataset directly is faster, deterministic, and auditable.

**Scope:**
- Remove `unified_bert_expert`, `expert_filter`, and related inference plumbing
- Implement `DomainShard` class: SQLite backend with `chunks`, `fts_chunks`, `entities`, `relations`, `artifacts` tables
- Implement `QueryStore` retrieval: FTS5 prefilter → embedding similarity → rerank → top-k return
- Wire `DomainShard` into the routing pipeline where BERT experts previously sat

### Phase 2 — TRM v2 Graph Architecture

**What:** Replace the current monolithic TRM model expert with the modular frozen-encoder + isolated-heads + gating-network + DomainGraph architecture described in the TRM v2 spec.

**Why:** The current single TRM model is a bottleneck and a catastrophic forgetting risk. Every new domain currently requires retraining the whole model. TRM v2 makes non-destructive growth a structural property.

**Scope:**
- Implement `SharedEncoder` (frozen after stabilisation)
- Implement `DomainHead` (lightweight per-domain module)
- Implement `GatingNetwork` (top-k sparse routing, three gate regimes)
- Implement `DomainGraph` registry (graph of DomainNodes with centroid, head ref, shard ref, cold-storage ref, drift profile, neighbour links)
- Implement cold-storage lifecycle (Hot / Warm / Cold states + reactivation queue)
- Wire all components into `TRMV2Pipeline`

### Phase 3 — Hardcoded Domain Purge + Bug Sweep

**What:** Audit the entire codebase for hardcoded domain references and any bugs flagged in `REBUILD_NOTES.md`. Make the domain system fully dynamic — domains must live in the DomainGraph, never baked into source.

**Why:** Hardcoded domains make the system brittle, prevent proper cold-storage management, and are architecturally inconsistent with the emergent-ontology design philosophy.

**Scope:**
- Full grep audit for hardcoded domain strings / enum values
- Replace all static domain references with DomainGraph lookups
- Fix all bugs listed in `REBUILD_NOTES.md`

### Phase 4 — `run_workflow.py` CLI Rewrite

**What:** Replace the 700-line monolith `run_workflow.py` with a clean, ergonomic CLI entrypoint.

**Why:** The current file is unmaintainable, mixes concerns, and makes it difficult to invoke specific pipeline modes cleanly.

**Scope:**
- Clean argument parser with `--mode` (quick / smart / researcher), `--input-file`, `--output`, `--backend-only`, `--enable-trm-v2`, `--legacy`
- Separate `run_trm_v2()` and `run_legacy()` dispatch paths
- Legacy path delegates cleanly to archived v1 pipeline via thin shim
- No business logic in the CLI layer itself

---

## 7. What Is Already Done

The following components were implemented on the (now-nuked) `feature/trm-v02-clean-rebuild` branch. Before starting Phase 1, verify which of these are **already present on `web-ui-prototype`**. Only implement what is genuinely missing:

- `SemanticSignature` + canonical hash generation
- `ConfidenceState` multi-dimensional struct (replacing scalar floats throughout)
- `TemporalState` typed encoding
- `ProvenanceChain` lineage tracking
- `ContradictionEdge` classification system
- DST fusion layer (Layer 0 + reasoning worker aggregation)
- `DomainGraph` data layer (basic)
- `DomainStore` persistence layer
- `SharedEncoder` + `DomainHead` + `TRMV2InferenceEngine` (neural layer)
- `DriftDetector`, `ReplayBuffer`, `ColdStorageManager`
- Per-domain SQLite shards: `DomainShard`, `QueryStore`
- `TRMV2Pipeline` wiring
- Parity test suite (15 tests passed, Tier 3 skipped pending legacy bridge)

---

## 8. Conventions & Hard Rules

- **No hardcoded domains anywhere. Ever.**
- **DST, not scalar confidence**, for any multi-class uncertainty.
- **SQLite shards are the knowledge layer.** Domain heads are the reasoning layer. Never merge these concerns.
- **Immutable graphs.** Never mutate a stored reasoning graph; create a new revision.
- **The gate has three regimes.** Do not allow unrestricted gate training after the bootstrap phase.
- **Legacy path must remain callable** via `--legacy` flag, delegating to the archived v1 pipeline shim at `archive/legacy_run_workflow_v1.py`.
- **All tests live under** `tests/trm_v2/`, `tests/domain_graph/`, `tests/domain_store/`, `tests/cold_storage/`.
- **New branch is forked from `web-ui-prototype`**, not from any previous rebuild branch.
