# Mycelium TRM v2 Theoretical Specification (v0.2)

## Overview

This document defines a theoretical architecture for **TRM v2** in Project Mycelium. The goal is to recover the original Mycelium design objective of **incremental, non-destructive learning** while preserving compatibility with the current reasoning-oriented system design described in the repository README.

TRM v2 is designed around four constraints:

1. New domains must be addable without destructive retraining of existing domain behavior.
2. The system must remain modular, so that routing, retrieval, domain adaptation, and long-term retention are separable concerns.
3. Dormant domains should be recoverable through a cold-storage lifecycle rather than remaining permanently active in memory.
4. Lightweight retrieval backends should be able to replace heavyweight always-loaded domain expert models when retrieval, not dense inference, is the dominant runtime cost.

The architecture in this document extends the README's original meta-controller, patch-network, and cold-storage ideas into a formal continual-learning system using a frozen shared backbone, domain-isolated heads, a graph-backed domain registry, and SQLite-sharded expert stores.

## Design Premise

The README explicitly frames Mycelium as a response to catastrophic forgetting in monolithic systems, proposing domain experts, isolated patch networks, and cold storage for long-term retention. Over time, the implementation evolved toward routing and reasoning, but the original anti-forgetting goal remains compatible with the present stack if the system is decomposed into stable shared components and mutable per-domain components.

The core theoretical shift in TRM v2 is this:

- **Do not fine-tune the whole routing model when a new domain arrives.**
- **Freeze shared representations once they are good enough.**
- **Attach new learnable capacity only where new knowledge should live.**
- **Treat retrieval memory and reasoning memory as distinct layers.**

This aligns with theoretical and empirical findings from continual-learning literature. Mixture-of-Experts (MoE) architectures can reduce forgetting by distributing tasks across specialized experts selected by a router, while parameter-isolation methods provide stronger guarantees against interference by structurally separating learnable parameters across tasks.

## Architectural Principles

### Shared backbone, isolated domain heads

TRM v2 uses a **frozen shared encoder** to transform a query or task context into a common latent representation. This encoder is trained during an initial base phase and then frozen for normal domain expansion.

On top of the frozen encoder sit **domain-isolated heads**, one per domain or domain cluster. Each head is a lightweight module such as a small MLP, projection head, classifier head, halting calibrator, or adapter bundle. When a new domain arrives, a new head is instantiated without modifying existing heads.

This makes the anti-forgetting rule structural rather than heuristic:

- Old domain heads are immutable by default.
- New domain learning happens in newly allocated capacity.
- Shared encoder drift is prevented because the encoder is frozen in ordinary expansion mode.

### Routing by gated specialization

A **gating network** maps the shared latent representation to one or more domain-head activation weights. The gate selects the most relevant domain heads for downstream scoring, halting, and evidence routing.

Research on MoE in continual learning suggests an important practical rule: the gate should not remain freely trainable forever. After the system stabilizes across current domains, gate updates should be restricted, decayed, or frozen to preserve convergence and avoid destabilizing previous expert allocation.

Therefore, TRM v2 distinguishes three gate regimes:

1. **Bootstrap regime** — gate is trainable while foundational domains are being learned.
2. **Expansion regime** — gate updates are limited to calibration for newly added domains.
3. **Stability regime** — gate is frozen, and only newly introduced heads or local adapters are trained.

### Domain graph as registry and memory topology

The domain registry is modeled as a **graph**, not only as a list of experts. This graph generalizes the README's "lightweight, dynamically updatable mapping structure" into a topology that can represent semantic proximity, lineage, cold-storage state, and patch ancestry.

Each domain node contains:

- Domain identifier and metadata.
- Spectral signature or centroid embedding.
- Pointers to active head checkpoints.
- Pointers to cold-storage artifacts.
- Links to neighboring or parent domains.
- Statistics on recency, usage, confidence, and drift.

Edges represent one or more of the following:

- Semantic similarity.
- Routing confusion or co-activation frequency.
- Patch lineage.
- Retrieval overlap.
- Cold-to-hot reactivation ancestry.

This graph enables expansion decisions that a flat registry cannot express.

### Retrieval and reasoning are separate subsystems

TRM v2 enforces a strict four-way responsibility separation:

```
SQLite Shards     =  Knowledge Memory
Predicate Engine  =  Semantic Representation
TRM               =  Reasoning
Domain Heads      =  Calibration + Routing
```

Domain heads are responsible for routing calibration, confidence calibration, retrieval policy hints, and local adaptation parameters. They are NOT responsible for knowledge storage or primary reasoning.

The original architecture treated domain experts as submodels that could be patched or replaced. TRM v2 refines this by replacing heavyweight BERT-style domain experts with **SQLite-sharded expert stores** that hold domain corpora, embeddings, lexical indexes, and metadata, while the TRM head retains only routing and local calibration logic.

## Formal Components

### 1. Frozen shared encoder

The shared encoder is a stable representation function:

\[
z = E(x)
\]

where `x` is the query or task state and `z` is a shared latent representation.

The encoder may be initialized from the current TRM stack or retrained for TRM v2 bootstrap. After stabilization, `E` is frozen in normal operation.

### 2. Domain head family

For each domain `d`, TRM v2 defines a lightweight head:

\[
h_d = H_d(z)
\]

A head may output:

- Domain relevance score.
- Confidence score.
- Halting contribution.
- Retrieval policy hints.
- Patch recommendation signal.
- Cross-domain handoff signal.

A new domain creates a new `H_d` rather than modifying existing `H_i` for `i ≠ d`.

### 3. Gating network

The gate produces sparse or top-k routing weights:

\[
g(z) = \{w_d\}_{d \in D}
\]

with only a limited number of high-probability weights used downstream:

\[
y = \sum_{d \in \text{TopK}(g(z))} w_d H_d(z)
\]

### 4. Halt controller

The halt controller in TRM v2 should be separated conceptually from domain expansion. If the halt head is stable, it should be frozen after base training. Only if new evidence shows systematic failure in new domain classes should a local calibration layer or domain-specific halting adapter be introduced.

### 5. DomainGraph

The theoretical `DomainGraph` is defined as:

```text
DomainGraph = (V, E)
```

where each node `v ∈ V` is a `DomainNode` and each edge `e ∈ E` captures typed relations between domains.

A theoretical `DomainNode` contains:

- `domain_id`
- `domain_version`
- `schema_version`
- `head_version`
- `retrieval_version`
- `semantic_centroid`
- `spectral_signature`
- `head_ref`
- `cold_storage_ref`
- `sqlite_shard_ref`
- `neighbors`
- `activation_stats`
- `drift_profile`
- `lineage`
- `parent_domains`
- `derived_from`
- `creation_reason`
- `created_by`
- `creation_timestamp`

The `DomainGraph` itself also carries:

- `graph_version`
- `schema_version`
- `creation_timestamp`
- `last_stabilized`

Every snapshot must carry `graph_version` to support migration, replay, graph comparison, and topology archaeology.

## Domain Lifecycle State Machine

The original cold-storage states are extended into a full formal state machine to support deprecation, archival, failed creation handling, and replay compatibility.

### DomainState

```python
class DomainState(str, Enum):
    CREATING     = "CREATING"
    HOT          = "HOT"
    WARM         = "WARM"
    COLD         = "COLD"
    REMEMBERING  = "REMEMBERING"
    DEPRECATED   = "DEPRECATED"
    ARCHIVED     = "ARCHIVED"
    FAILED       = "FAILED"
```

### Lifecycle transitions

Primary path:

```text
CREATING → HOT → WARM → COLD → REMEMBERING → HOT
```

Retirement path:

```text
HOT → DEPRECATED → ARCHIVED
```

Failure path:

```text
CREATING → FAILED
```

### Hard-delete prohibition

Domains MUST NEVER be hard-deleted. Domain retirement occurs through `DEPRECATED → ARCHIVED` only. This preserves replay, provenance, semantic archaeology, and patch lineage.

## Domain Versioning

A domain is not a static entity. Over time, retrieval evolves, calibration evolves, topology evolves, and storage evolves. Replay and debugging require explicit versioning.

The following version fields are added to `DomainNode`:

```python
domain_version:    str   # overall domain version
schema_version:    str   # schema compatibility version
head_version:      str   # checkpoint version for the domain head
retrieval_version: str   # shard version
```

**Rule:** All persisted artifacts MUST carry `domain_version`, including snapshots, cold-storage bundles, replay archives, and graph exports.

## Domain Provenance

The Predicate Engine supports lineage tracking. The DomainGraph must support equivalent ancestry tracking.

The following provenance fields are added to `DomainNode`:

```python
parent_domains:      List[str]  # immediate parent domain IDs
derived_from:        List[str]  # ancestry chain
creation_reason:     str        # why this domain was created
created_by:          str        # subsystem or user that triggered creation
creation_timestamp:  float
```

**Design rule:** Domain creation MUST be explainable. Every domain must answer "Why was I created?" without external metadata.

## Structured Drift Profile

The `drift_profile` field referenced in the original spec is now formally defined.

```python
@dataclass
class DriftProfile:
    semantic_drift:    float  # change in domain meaning
    retrieval_drift:   float  # change in retrieval quality
    routing_drift:     float  # change in gate behavior
    confidence_drift:  float  # calibration degradation
    activation_drift:  float  # usage-pattern shift
```

**Rule:** Drift responses must be localized. For example, `retrieval_drift high` with `routing_drift low` MUST NOT trigger gate recalibration. Each drift dimension is handled independently.

## Retrieval-Only Domains

Not every domain requires a dedicated head. Some domains are primarily retrieval problems.

### DomainMode

```python
class DomainMode(str, Enum):
    RETRIEVAL_ONLY = "RETRIEVAL_ONLY"
    FULL_DOMAIN    = "FULL_DOMAIN"
```

A `RETRIEVAL_ONLY` domain contains a SQLite shard, metadata, and a domain graph node, but no specialized head.

### Promotion policy

```text
RETRIEVAL_ONLY
      ↓ persistent usage, reasoning failures, calibration need
FULL_DOMAIN
```

### Demotion policy

```text
FULL_DOMAIN
      ↓ long inactivity, head archived
RETRIEVAL_ONLY
```

This prevents domain-head explosion while preserving knowledge accessibility.

## Gate Recovery Protocol

Frozen gates prevent forgetting, but frozen-forever gates prevent adaptation. A recovery mechanism is required.

### GateState

```python
class GateState(str, Enum):
    BOOTSTRAP      = "BOOTSTRAP"
    EXPANSION      = "EXPANSION"
    STABLE         = "STABLE"
    THAWING        = "THAWING"
    RECALIBRATING  = "RECALIBRATING"
```

### Recovery lifecycle

```text
STABLE
   ↓ routing instability detected
THAWING
   ↓
RECALIBRATING
   ↓
STABLE
```

### Trigger conditions

Recalibration may be triggered by:

- Routing confusion spike.
- Repeated misrouting.
- Domain overlap increase.
- Confidence collapse.

**Rule:** Only local recalibration is allowed. Global retraining remains forbidden.

## Domain Addition Policy

TRM v2 decides how to absorb a new domain by comparing it to existing domains in the shared semantic space and in the spectral-signature space.

### Case A: Clearly distant domain

If the new domain is sufficiently far from existing domain centroids or signatures, the system should:

1. Create a new graph node.
2. Create a new isolated domain head.
3. Create a new SQLite shard for retrieval memory.
4. Train only the new head and associated local adapters.
5. Optionally recalibrate the gate with restricted updates.

### Case B: Near-neighbor or sub-domain

If the new domain is close to an existing domain, the system may choose:

- **Sub-domain attachment** — add a child node with its own head, initialized from the nearest existing head.
- **Patch adaptation** — keep the parent node but attach a local LoRA-style adapter to the parent head.

### Case C: Ambiguous novelty

Stage the domain in a provisional node with a probationary head and temporary shard. Promote to full domain only after persistent separability is demonstrated.

### Case D: Retrieval-only entry

If the domain does not immediately require reasoning calibration, enter it as `RETRIEVAL_ONLY`. Promote only when warranted.

## Catastrophic Forgetting Strategy

TRM v2 addresses catastrophic forgetting through structural isolation first, selective adaptation second, and replay-based refresh last.

### Structural isolation

Parameter isolation provides the strongest baseline guarantee because new learning is assigned to new parameters rather than overlapping parameters. In TRM v2:

- Existing domain heads are frozen by default.
- The shared encoder is frozen after stabilization.
- Gate updates are minimized after convergence.

### Selective adaptation

When near-neighbor transfer is desirable, use local adaptation methods:

- LoRA adapters on domain heads.
- Orthogonal gradient projection for local updates.
- Neighbor-initialized expert expansion.

### Replay-based refresh

Mini-buffers and synthetic-sample rules in cold storage serve as replay buffers for safe reactivation or calibration. Replay is used only when necessary, and only for the affected domain.

## Cold Storage in TRM v2

### Hot, warm, cold states

Every domain exists in one of the lifecycle states defined in the state machine. The primary operational states are:

- **Hot** — head and retrieval shard are active in fast memory.
- **Warm** — retrieval shard is available, but the head may be quantized or lazily loaded.
- **Cold** — head is quantized and offloaded, shard is persisted on disk, replay buffer is archived.

### Cold-storage artifact bundle

A cold-stored domain preserves a versioned bundle containing:

- Quantized domain head checkpoint (tagged with `domain_version` and `head_version`).
- Domain metadata and provenance.
- SQLite retrieval shard snapshot (tagged with `retrieval_version`).
- Replay mini-buffer.
- Synthetic-data rules or augmentation recipe.
- Drift profile at time of cold-store.
- Parent/neighbor references from the graph.

### Remembering state

During reactivation:

1. Set state to `REMEMBERING`.
2. Reload or dequantize head checkpoint.
3. Open SQLite shard.
4. Warm a query embedding cache if present.
5. Run drift detection against replay buffer using structured `DriftProfile`.
6. Optionally perform localized low-risk adaptation if `semantic_drift` or `confidence_drift` exceeds threshold.
7. Promote to `WARM` or `HOT`.

### Reactivation queue

If several cold domains are simultaneously needed, order recalls by a priority score based on relevance, urgency, recency, and estimated VRAM cost. This prevents memory thrash during multi-domain bursts.

## Domain Observability Events

TRM v2 must expose domain lifecycle events to support frontend graph visualization, replay, semantic archaeology, and domain lifecycle debugging.

### Required events

```text
graph_domain_create
graph_domain_activate
graph_domain_coldstore
graph_domain_remember
graph_domain_promote
graph_domain_deprecate
graph_gate_recalibration
graph_drift_detected
```

### Required metadata per event

Every event MUST contain:

```text
sequence_number
event_id
timestamp
graph_schema_version
domain_version
```

## SQLite-Sharded Expert Theory

### Why replace BERT experts

In many real workloads, a BERT expert serves two distinct roles: acting as a semantic retriever over domain knowledge, and providing local domain reasoning. TRM v2 splits those roles.

- Local domain reasoning remains in the lightweight domain head.
- Domain knowledge retrieval moves into a **SQLite shard**.

### Structure of a SQLite shard

A theoretical SQLite-sharded expert contains:

```text
chunks(id, text, embedding, source, created_at, metadata_json)
fts_chunks(text)                                  ← FTS5 lexical index
entities(entity_id, label, type, metadata_json)
relations(src_id, rel_type, dst_id, weight)
artifacts(key, value)
replay_samples(sample_id, text, label_json, embedding, metadata_json)
```

### Retrieval flow

1. Lexical prefilter via FTS5.
2. Vector similarity over candidate rows.
3. Graph or metadata reranking when structured relations exist.
4. Top-k handoff to the reasoning pipeline.

### Relationship to cold storage

The SQLite shard is the natural persistence unit for cold storage. When a domain is offloaded, the retrieval memory does not disappear — it becomes the durable knowledge substrate that can be reopened on demand.

- The **head** is the mutable reasoning shell.
- The **SQLite shard** is the durable knowledge substrate.

## Compatibility With Predicate Engine v0.2

TRM v2 SHALL treat `PredicateFrame` objects as the canonical reasoning substrate.

Domain heads MUST NOT operate directly on raw text when `PredicateFrame` objects are available.

### Preferred flow

```text
Query
 ↓
Predicate Engine
 ↓
PredicateFrames
 ↓
TRM
 ↓
Domain Selection
 ↓
Retrieval
 ↓
DSTFusion
 ↓
Stabilization
```

This guarantees compatibility with contradiction propagation, semantic replay, provenance tracking, graph reasoning, ontology governance, and leverage propagation, aligning TRM v2 with the broader Mycelium semantic runtime architecture.

## Relation to Existing Mycelium Ideas

TRM v2 is not a rejection of the original README architecture. It is a formalization and modernization of it.

| README idea | TRM v2 interpretation |
|---|---|
| Meta-controller | Gate + DomainGraph registry |
| Domain experts | Frozen encoder + isolated domain heads + SQLite retrieval shard |
| Patch networks | Local adapters, child heads, or patch modules |
| Cold storage | Versioned head + shard + replay bundle lifecycle |
| Non-destructive learning | Structural parameter isolation + replay-assisted reactivation |

## Open Theoretical Questions

### How many domains should share one gate?

A single gate across too many domains may become unstable. A hierarchical or clustered gate may be better once the graph grows large.

### When should a close domain become a child domain instead of a patch?

Depends on persistence, semantic distance, retrieval overlap, and routing confusion. Both should be supported; policy thresholds need empirical tuning.

### Can some domains become retrieval-only permanently?

Yes — the `RETRIEVAL_ONLY` mode formalizes this. Some domains may never need a trainable head.

### How much replay is enough?

The README suggests 500 to 1000 historical samples. That is a sensible starting range, but optimal replay size likely depends on domain entropy, drift rate, and head capacity.

### How should gate thawing be bounded?

Local recalibration is required, but the boundary between local and global must be operationalized. Candidate bound: only gates connected to newly confused or newly added domain nodes are unfrozen.

## Conclusion

TRM v2 restores Mycelium's original anti-forgetting mission by turning non-destructive growth into a structural property of the architecture. Its key commitments are a frozen shared encoder, isolated domain heads, a graph-backed domain registry with formal lifecycle states, cold-storage lifecycle management, SQLite-sharded experts that separate durable domain knowledge from lightweight reasoning logic, structured drift profiles, domain provenance, gate recovery protocols, and full compatibility with the Predicate Engine v0.2 reasoning substrate.

This architecture remains faithful to the original Mycelium ideas of patching, modularity, and long-term retention, while updating them with a modern continual-learning interpretation suitable for the present reasoning stack.
