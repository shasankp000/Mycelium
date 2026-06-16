# Mycelium TRM v2 Theoretical Specification

## Overview

This document defines a theoretical architecture for **TRM v2** in Project Mycelium. The goal is to recover the original Mycelium design objective of **incremental, non-destructive learning** while preserving compatibility with the current reasoning-oriented system design described in the repository README.[cite:371]

TRM v2 is designed around four constraints:

1. New domains must be addable without destructive retraining of existing domain behavior.[cite:371][cite:376]
2. The system must remain modular, so that routing, retrieval, domain adaptation, and long-term retention are separable concerns.[cite:371]
3. Dormant domains should be recoverable through a cold-storage lifecycle rather than remaining permanently active in memory.[cite:371]
4. Lightweight retrieval backends should be able to replace heavyweight always-loaded domain expert models when retrieval, not dense inference, is the dominant runtime cost.[cite:377]

The architecture in this document extends the README's original meta-controller, patch-network, and cold-storage ideas into a formal continual-learning system using a frozen shared backbone, domain-isolated heads, a graph-backed domain registry, and SQLite-sharded expert stores.[cite:371][cite:375][cite:376]

## Design Premise

The README explicitly frames Mycelium as a response to catastrophic forgetting in monolithic systems, proposing domain experts, isolated patch networks, and cold storage for long-term retention.[cite:371] Over time, the implementation evolved toward routing and reasoning, but the original anti-forgetting goal remains compatible with the present stack if the system is decomposed into stable shared components and mutable per-domain components.[cite:371]

The core theoretical shift in TRM v2 is this:

- **Do not fine-tune the whole routing model when a new domain arrives.**
- **Freeze shared representations once they are good enough.**
- **Attach new learnable capacity only where new knowledge should live.**
- **Treat retrieval memory and reasoning memory as distinct layers.**

This aligns with theoretical and empirical findings from continual-learning literature. Mixture-of-Experts (MoE) architectures can reduce forgetting by distributing tasks across specialized experts selected by a router, while parameter-isolation methods provide stronger guarantees against interference by structurally separating learnable parameters across tasks.[cite:375][cite:376]

## Architectural Principles

### Shared backbone, isolated domain heads

TRM v2 uses a **frozen shared encoder** to transform a query or task context into a common latent representation. This encoder is trained during an initial base phase and then frozen for normal domain expansion.[cite:375]

On top of the frozen encoder sit **domain-isolated heads**, one per domain or domain cluster. Each head is a lightweight module such as a small MLP, projection head, classifier head, halting calibrator, or adapter bundle. When a new domain arrives, a new head is instantiated without modifying existing heads.[cite:376]

This makes the anti-forgetting rule structural rather than heuristic:

- Old domain heads are immutable by default.
- New domain learning happens in newly allocated capacity.
- Shared encoder drift is prevented because the encoder is frozen in ordinary expansion mode.

### Routing by gated specialization

A **gating network** maps the shared latent representation to one or more domain-head activation weights. The gate selects the most relevant domain heads for downstream scoring, halting, and evidence routing.[cite:375]

Research on MoE in continual learning suggests an important practical rule: the gate should not remain freely trainable forever. After the system stabilizes across current domains, gate updates should be restricted, decayed, or frozen to preserve convergence and avoid destabilizing previous expert allocation.[cite:375]

Therefore, TRM v2 distinguishes three gate regimes:

1. **Bootstrap regime** — gate is trainable while foundational domains are being learned.
2. **Expansion regime** — gate updates are limited to calibration for newly added domains.
3. **Stability regime** — gate is frozen, and only newly introduced heads or local adapters are trained.

### Domain graph as registry and memory topology

The domain registry is modeled as a **graph**, not only as a list of experts. This graph generalizes the README's "lightweight, dynamically updatable mapping structure" into a topology that can represent semantic proximity, lineage, cold-storage state, and patch ancestry.[cite:371]

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

The original architecture treated domain experts as submodels that could be patched or replaced.[cite:371] TRM v2 refines this further by distinguishing:

- **Reasoning memory** — lightweight learned parameters in the domain head.
- **Knowledge memory** — retrieval content in a domain data store.

This separation allows heavyweight BERT-style domain experts to be replaced, in many cases, with **SQLite-sharded expert stores** that hold domain corpora, embeddings, lexical indexes, and metadata, while the TRM head retains only routing and local calibration logic.[cite:377]

## Formal Components

### 1. Frozen shared encoder

The shared encoder is a stable representation function:

\[
z = E(x)
\]

where `x` is the query or task state and `z` is a shared latent representation.

The encoder may be initialized from the current TRM stack or retrained for TRM v2 bootstrap. After stabilization, `E` is frozen in normal operation.

Its purpose is not to memorize domain-specific knowledge, but to provide a reusable semantic space in which routing, similarity, and retrieval coordination can operate.

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

A new domain creates a new `H_d` rather than modifying existing `H_i` for `i \neq d`.[cite:376]

### 3. Gating network

The gate produces sparse or top-k routing weights:

\[
g(z) = \{w_d\}_{d \in D}
\]

with only a limited number of non-zero or high-probability weights used downstream. Final domain selection can be interpreted as:

\[
y = \sum_{d \in \text{TopK}(g(z))} w_d H_d(z)
\]

This retains the MoE property that different experts specialize while the gate learns conditional routing.[cite:375]

### 4. Halt controller

The halt controller in TRM v2 should be separated conceptually from domain expansion. If the halt head is good and stable, it should be frozen after base training. Only if new evidence shows systematic failure in new domain classes should a local calibration layer or domain-specific halting adapter be introduced.

This avoids dragging the global halt behavior every time a new domain appears.

### 5. DomainGraph

The theoretical `DomainGraph` is defined as:

```text
DomainGraph = (V, E)
```

where each node `v ∈ V` is a `DomainNode` and each edge `e ∈ E` captures typed relations between domains.

A theoretical `DomainNode` contains:

- `domain_id`
- `semantic_centroid`
- `spectral_signature`
- `head_ref`
- `cold_storage_ref`
- `sqlite_shard_ref`
- `neighbors`
- `activation_stats`
- `drift_profile`
- `lineage`

This graph is both a routing scaffold and a memory-management scaffold.

## Domain Addition Policy

TRM v2 decides how to absorb a new domain by comparing it to existing domains in the shared semantic space and in the spectral-signature space already envisioned in the project.[cite:371]

### Case A: Clearly distant domain

If the new domain is sufficiently far from existing domain centroids or signatures, then the system should:

1. Create a new graph node.
2. Create a new isolated domain head.
3. Create a new SQLite shard for retrieval memory.
4. Train only the new head and associated local adapters.
5. Optionally recalibrate the gate with restricted updates.

This is the preferred path for truly orthogonal domains.

### Case B: Near-neighbor or sub-domain

If the new domain is close to an existing domain or cluster, then the system may choose one of two strategies:

- **Sub-domain attachment** — add a child node with its own head, but initialize from the nearest existing head.
- **Patch adaptation** — keep the parent node but attach a local patch module or LoRA-style adapter to the parent head.

In either case, updates should be isolated or constrained. This preserves the README's original patch-network intuition while aligning with modern parameter-efficient continual-learning strategies.[cite:371][cite:379]

### Case C: Ambiguous novelty

If novelty is uncertain, the system should stage the domain in a provisional node with a probationary head and a temporary retrieval shard. Promotion to a full domain occurs only after enough evidence shows persistent separability from existing domains.

This reduces over-fragmentation.

## Catastrophic Forgetting Strategy

TRM v2 addresses catastrophic forgetting through **structural isolation first**, **selective adaptation second**, and **replay-based refresh last**.

### Structural isolation

Parameter isolation provides the strongest baseline guarantee because new learning is assigned to new parameters rather than overlapping parameters.[cite:376] In TRM v2, this means:

- Existing domain heads are frozen by default.
- The shared encoder is frozen after stabilization.
- Gate updates are minimized after convergence.[cite:375]

### Selective adaptation

When near-neighbor transfer is desirable, use local adaptation methods rather than global fine-tuning. Candidate mechanisms include:

- LoRA adapters on domain heads.
- Orthogonal gradient projection for local updates.[cite:379]
- Neighbor-initialized expert expansion similar to expandable continual-learning expert architectures.[cite:381]

### Replay-based refresh

The README already proposes mini-buffers and synthetic-sample rules in cold storage.[cite:371] TRM v2 treats these as replay buffers for safe reactivation or calibration. Replay is used only when necessary, and only for the affected domain or neighborhood, not for the whole system.

## Cold Storage in TRM v2

The README's cold storage section becomes a first-class memory lifecycle in TRM v2.[cite:371]

### Hot, warm, cold states

Every domain exists in one of three operational states:

- **Hot** — head and retrieval shard are active in fast memory.
- **Warm** — retrieval shard is available, but the head may be quantized or lazily loaded.
- **Cold** — head is quantized and offloaded, shard is persisted on disk, replay buffer is archived, and the domain is inactive until recalled.

### Cold-storage artifact bundle

A cold-stored domain should preserve a versioned bundle containing:

- Quantized domain head checkpoint.
- Domain metadata.
- SQLite retrieval shard snapshot.
- Replay mini-buffer.
- Synthetic-data rules or augmentation recipe.
- Drift statistics and calibration metrics.
- Parent/neighbor references from the graph.

### Remembering state

The README introduces a user-visible **Remembering** phase during reactivation.[cite:371] TRM v2 keeps this concept and formalizes it as a temporary state where:

1. The domain is identified as relevant.
2. Its head is reloaded or dequantized.
3. The SQLite shard is opened and warmed.
4. A drift test compares current input patterns to stored replay distributions.
5. Optional low-risk adaptation occurs if drift exceeds threshold.

This state is theoretically valuable because it makes delayed memory retrieval explicit rather than pretending all knowledge is always fully resident.

### Reactivation queue

One extension beyond the README is a **reactivation queue**. If several cold domains are simultaneously needed, the system should order recalls by a priority score based on relevance, urgency, recency, and estimated VRAM cost. This prevents memory thrash during multi-domain bursts.

## SQLite-Sharded Expert Theory

The SQLite-sharded expert is a domain knowledge store intended to replace many always-loaded BERT-style expert models in the common case where retrieval dominates inference cost.

### Why replace BERT experts

The README's domain expert abstraction predates the current lightweight retrieval ecosystem.[cite:371] In many real workloads, a BERT expert is serving two distinct roles:

1. It acts as a semantic retriever over domain knowledge.
2. It provides local domain reasoning.

TRM v2 proposes to split those roles.

- Local domain reasoning remains in the lightweight domain head.
- Domain knowledge retrieval moves into a **SQLite shard**.

This makes the system lighter, easier to cold-store, and cheaper to scale.

### Structure of a SQLite shard

A theoretical SQLite-sharded expert contains:

- A main table of chunks or records.
- An embedding column storing vectors as blobs.
- An FTS5 lexical index for prefiltering.
- Metadata tables for provenance, timestamps, and schema versioning.
- Optional graph tables for entity or relation edges.

A representative shard includes:

```text
chunks(id, text, embedding, source, created_at, metadata_json)
fts_chunks(text)
entities(entity_id, label, type, metadata_json)
relations(src_id, rel_type, dst_id, weight)
artifacts(key, value)
```

### Retrieval flow

The shard supports a layered retrieval policy:

1. Lexical prefilter via FTS5.
2. Vector similarity over candidate rows using SQLite vector extensions or external SIMD-assisted functions.[cite:377]
3. Graph or metadata reranking when structured relations exist.
4. Top-k handoff to the reasoning pipeline.

This is far more lightweight than keeping a transformer expert loaded for every domain.

### Relationship to cold storage

The SQLite shard is also the natural persistence unit for cold storage. When a domain is offloaded, the retrieval memory does not disappear; it simply becomes the durable knowledge substrate that can be reopened on demand.

In this sense:

- The **head** is the mutable reasoning shell.
- The **SQLite shard** is the durable knowledge substrate.

## Relation to Existing Mycelium Ideas

TRM v2 is not a rejection of the original README architecture. It is a formalization and modernization of it.[cite:371]

| README idea | TRM v2 interpretation |
|---|---|
| Meta-controller | Gate + DomainGraph registry |
| Domain experts | Frozen encoder + isolated domain heads + SQLite retrieval shard |
| Patch networks | Local adapters, child heads, or patch modules |
| Cold storage | Versioned head + shard + replay bundle lifecycle |
| Non-destructive learning | Structural parameter isolation + replay-assisted reactivation |

The original design goal of defeating catastrophic forgetting remains intact, but the mechanism becomes more explicit and more scalable.

## Research Alignment

The proposed TRM v2 aligns with several strands of current research:

- **MoE continual learning** supports the idea that expert specialization and gated routing reduce forgetting, but also indicates that unrestricted gate updating can hurt convergence.[cite:375]
- **Parameter isolation theory** supports the use of separate learnable parameters per domain or task to prevent destructive interference.[cite:376]
- **Pathway protection and expandable representations** reinforce the idea that growth should allocate new capacity instead of overwriting global capacity.[cite:373][cite:381]
- **Parameter-efficient continual learning** suggests that local adapters and orthogonal updates are appropriate for near-neighbor adaptation without retraining the whole system.[cite:379]
- **SQLite vector search systems** support replacing many heavy runtime retrievers with embedded, low-memory semantic stores.[cite:377]

## Open Theoretical Questions

Several questions remain intentionally open and should be resolved in experimentation rather than assumed in theory.

### How many domains should share one gate?

A single gate across too many domains may become unstable or too diffuse. A hierarchical or clustered gate may be better once the graph grows large.

### When should a close domain become a child domain instead of a patch?

This depends on persistence, semantic distance, retrieval overlap, and routing confusion. The graph should support both, but policy thresholds need empirical tuning.

### Can some domains become retrieval-only?

Not every domain may require its own trainable head. Some domains may eventually be represented entirely by retrieval memory plus generic reasoning, which would reduce training complexity further.

### How much replay is enough?

The README suggests 500 to 1000 historical samples.[cite:371] That is a sensible starting range, but optimal replay size likely depends on domain entropy, drift rate, and head capacity.

## Conclusion

TRM v2 restores Mycelium's original anti-forgetting mission by turning non-destructive growth into a structural property of the architecture rather than a training hope.[cite:371][cite:376] Its key commitments are a frozen shared encoder, isolated domain heads, a graph-backed domain registry, cold-storage lifecycle management, and SQLite-sharded experts that separate durable domain knowledge from lightweight reasoning logic.[cite:371][cite:375][cite:377]

This architecture remains faithful to the original Mycelium ideas of patching, modularity, and long-term retention, while updating them with a modern continual-learning interpretation suitable for the present reasoning stack.[cite:371]
