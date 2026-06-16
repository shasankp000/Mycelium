# Mycelium — TRM v0.2 Rebuild Notes

> **Authored:** 2026-06-16  
> **Status:** Pre-implementation planning document

---

## Why This Rebuild Exists

The `phase-d-wiring` branch accumulated deep structural problems exposed by the first backend-only run. The output for the query `"Is 9.9 bigger than 9.11?"` revealed:

- `engine_spec` appearing as a hardcoded fusion domain with no connection to the query
- `NO_EXPERT_AVAILABLE` for a simple numeric comparison — no `mathematics` or `general_reasoning` domain exists
- TRM checkpoint returning `chemistry` as primary domain (99.9997% probability) for a comparison question — the model is undertrained noise
- `phase2_pipeline` created a new `general` expert while `expert_decision` simultaneously returned `USE_EXISTING_EXPERT` for `physics` — the two subsystems are incoherent
- `evidence_dst` reporting `m_true=0, m_false=0, m_unknown=1.0` — DST belief fusion contributing nothing
- The BERT-based expert system (`UnifiedBERTExpert`, `k-medoids`, calibration) correctly registering `physics` as the most similar expert to a math question via cosine similarity on biology medoid text

The root cause is architectural: the current system bolts domain-expert ML (BERT, k-medoids, calibration) on top of routing (MultiLensRouter, spectral signatures) with no unified authority. TRM v0.2, as defined in `IMPLEMENTATION_SPEC_TRM_V2.md`, replaces this with a coherent design: a shared frozen encoder, per-domain lightweight heads, SQLite-sharded retrieval, and a DomainGraph registry as the single source of truth.

---

## What Gets Nuked (Full List)

These modules are **fully deleted or superseded** in the rebuild. No patches.

### The BERT Expert System — Core Target

| File | Why It Dies |
|---|---|
| `mycelium/pipeline/unified_bert_expert.py` | The entire BERT-based expert inference path. K-medoids clustering on biology text answering physics questions is the failure mode. SQLite-sharded heads replace its retrieval role. |
| `mycelium/pipeline/unified_expert_system.py` | Orchestrates BERT experts. Replaced by `TRMV2InferenceEngine` + `DomainGraphRegistry`. |
| `mycelium/pipeline/expert_filter.py` | Tag-based expert filtering with hardcoded `similarity_threshold=0.45`. Logic absorbed into `DomainNoveltyPolicy`. |
| `mycelium/pipeline/layer2_expert_loader.py` | Loads BERT experts from flat files. Replaced by `SQLiteDomainStore` loader. |
| `mycelium/pipeline/phase2/pipeline.py` | Phase2Pipeline as standalone decision authority — incoherent with expert_decision. Absorbed into `TRMV2InferenceEngine.route()`. |
| `mycelium/pipeline/phase3/pipeline.py` | Phase3To5Pipeline — runs after Phase2 produces a `create_new_expert` decision that Phase2 also makes separately. Eliminated. |
| `mycelium/pipeline/orchestration.py` | `combine_routing_and_expert_decisions()` — the combiner that fails to reconcile Phase2 and expert_system outputs. Replaced by single-authority routing. |

### Hardcoded Domain Pollution

| File | Problem | Action |
|---|---|---|
| `mycelium/pipeline/multi_lens_router.py` | `engine_spec` hardcoded in fusion score initialization. Domain whitelist baked in. | Full rewrite — dynamic domain registry lookup only, no hardcoded domain names anywhere. |
| `mycelium/pipeline/model_registry.py` | `STARTUP_SPECS` likely contains hardcoded domain list. | Audit fully; replace static spec list with `DomainGraphRegistry.all_hot_domains()` at warmup. |
| `mycelium/pipeline/dynamic_signature_manager.py` | `sync_signatures()` iterates over registered domains but uses a static `signatures/` directory. | Keep the file, remove hardcoded path defaults, make directory configurable via `config.toml`. |

### Legacy TRM v1 Artifacts (Superseded, Not Deleted)

These files are **kept as-is** but stop being called in the new pipeline. TRM v2 is a peer to `trm_engine.py`, not a replacement per the spec.

| File | Status in v0.2 |
|---|---|
| `mycelium/trm/trm_engine.py` | Legacy engine — retained, feature-flagged off |
| `mycelium/trm/reasoner.py` | TRMReasoner v1 — retained for reference, bypassed when `trm_v2.enabled=true` |
| `mycelium/trm/trm_routing_trace_writer.py` | Contains `DOMAIN_LIST` hardcoded to 3 domains — retained but not called in new path |
| `mycelium/trm/trm_checkpoints/trm_latest.pt` | Checkpoint that returns `chemistry` for everything — retained but not loaded in new path |

### The Monolith

| File | Problem | Action |
|---|---|---|
| `mycelium/pipeline/run_workflow.py` | 700+ line function with no CLI, no backend-only mode, incoherent multi-authority decisions, hardcoded phase2→phase3 flow | **Full rebuild** as thin orchestrator — see architecture below |

---

## What Gets Built (TRM v0.2)

Per `IMPLEMENTATION_SPEC_TRM_V2.md`, all new code lives in `mycelium/`. Nothing outside it is created.

### New Module Tree

```
mycelium/
├── trm/
│   └── v2/
│       ├── __init__.py
│       ├── encoder.py          SharedEncoder wrapping existing TRMNetwork (frozen)
│       ├── gating.py           DomainGate + GateState
│       ├── heads.py            DomainHead (lightweight MLP) + HeadRegistry
│       ├── halt.py             HaltControllerV2
│       ├── inference.py        TRMV2InferenceEngine — single routing authority
│       ├── trainer.py          TRMV2Trainer (bootstrap, new-head, gate-calibrate)
│       ├── checkpointing.py    HeadManifest + versioned I/O
│       └── policies.py         GateMode + expansion policies
│
├── domain_graph/
│   ├── __init__.py
│   ├── models.py               DomainNode, DomainEdge, DriftProfile
│   ├── state.py                DomainState, DomainMode, GateState, NoveltyDecision enums
│   ├── registry.py             DomainGraphRegistry — single source of truth for all domains
│   ├── novelty.py              DomainNoveltyPolicy (replaces shadow_domain_detector logic)
│   ├── events.py               Observability events via existing pipeline_event.py
│   └── persistence.py          JSON/SQLite-backed graph I/O
│
├── domain_store/
│   ├── __init__.py
│   ├── schema.py               SQL DDL (chunks, entities, relations, replay_samples, shard_meta)
│   ├── store.py                SQLiteDomainStore — one .sqlite file per domain
│   ├── embedder.py             OfflineEmbedder (runs at ingest time only, not inference)
│   ├── ingest.py               DomainIngestor (text corpus → SQLite shard)
│   ├── retrieval.py            DomainRetrievalEngine (FTS5 prefilter + Python cosine rerank)
│   ├── rerank.py               ResultReranker
│   └── migration.py            LegacyExpertMigrator (read-only from BERT side, offline only)
│
└── cold_storage/
    ├── __init__.py
    ├── manager.py              ColdStorageManager (cold-store + reactivate lifecycle)
    ├── quantization.py         INT8/FP16 quantization
    ├── reactivation.py         DomainReactivationService
    ├── replay_buffer.py        ReplayBufferManager
    ├── priority_queue.py       ReactivationQueue (min-heap)
    └── drift.py                DriftDetector (semantic, retrieval, routing, confidence, activation)
```

### New `run_workflow.py` Contract

The rebuilt `run_workflow.py` is a **thin orchestrator**, not a reasoning engine. Its responsibilities:

1. Parse CLI args (`argparse`) — backend-only mode from day one
2. Load feature flags from `config.toml` via `load_trm_v2_config()`
3. Call Layer 0 (`QuestionRouter`) — unchanged
4. Call `TRMV2InferenceEngine.route()` if `trm_v2.enabled=true`, else fall back to legacy `MultiLensRouter`
5. Call `DomainRetrievalEngine.retrieve()` if `sqlite_experts.enabled=true`
6. Emit SSE events via existing `EventEmitter`
7. Return structured result — no decision logic inline

**No phase2_pipeline, no phase3_pipeline, no expert_system, no expert_filter in the new path.**

The single routing authority is `TRMV2InferenceEngine`. Phase 2/3 are eliminated; their useful sub-components (confidence calibration, predicate extraction) become optional post-processors that the orchestrator can call independently.

### New CLI Entry Point

```bash
# Backend-only single query
python -m mycelium.pipeline.run_workflow \
    --input "Is 9.9 bigger than 9.11?" \
    --mode smart \
    --backend-only

# Backend-only batch from file
python -m mycelium.pipeline.run_workflow \
    --input-file queries.txt \
    --mode smart \
    --backend-only \
    --output results.json
```

---

## Architecture Contracts (New System)

### Single Authority Rule
`TRMV2InferenceEngine.route()` is the **only** function that makes routing decisions. No other module may return a `decision_type`, `selected_domain`, or `create_new_expert` signal. Any module that currently does this is a deletion target.

### Domain Registry Rule
All domain names come from `DomainGraphRegistry`. No module may hardcode a domain name (`engine_spec`, `physics`, `chemistry`, `general`, etc.) as a string literal. Domain membership is determined at runtime by the registry.

### Frozen Encoder Rule
`SharedEncoder` wraps the existing `TRMNetwork` backbone. It is frozen by default (`requires_grad=False`). It is only unfrozen during `GateState.THAWING` and only for the affected domain neighborhood — never globally.

### Per-Domain Head Rule
Each domain gets one `DomainHead` (lightweight MLP, ~3 layers, `hidden_dim=128`). Training a new domain head never touches existing head weights or the shared encoder. The `HeadManifest` records `domain_version` and `head_version` in every artifact filename for rollback safety.

### SQLite Shard Rule
Every domain's knowledge lives in `artifacts/sqlite_experts/{domain_id}.sqlite`. No flat file expert stores, no BERT embedding caches at inference time. `OfflineEmbedder` runs only at ingest time, never at inference time.

### Feature Flag Rule
All TRM v2 code paths are guarded by feature flags defaulting to `false`. The legacy path runs unchanged when flags are off. This enables shadow-mode comparison before any cutover.

---

## Build Order (Phase-by-Phase)

### Phase 1 — Data Layer (no model dependencies)
1. `mycelium/domain_graph/state.py` — enums
2. `mycelium/domain_graph/models.py` — `DomainNode`, `DomainEdge`, `DriftProfile`
3. `mycelium/domain_graph/persistence.py` — JSON/SQLite I/O
4. `mycelium/domain_graph/registry.py` — `DomainGraphRegistry`
5. `mycelium/domain_store/schema.py` — SQL DDL
6. `mycelium/domain_store/store.py` — `SQLiteDomainStore`
7. `mycelium/domain_store/embedder.py` — `OfflineEmbedder`
8. `mycelium/domain_store/ingest.py` — `DomainIngestor`
9. `mycelium/domain_store/retrieval.py` — `DomainRetrievalEngine`

### Phase 2 — Graph Intelligence
10. `mycelium/domain_graph/novelty.py` — `DomainNoveltyPolicy`
11. `mycelium/domain_graph/events.py` — observability via `pipeline_event.py`
12. `mycelium/domain_store/migration.py` — `LegacyExpertMigrator` (offline only)

### Phase 3 — Neural Layer
13. `mycelium/trm/v2/encoder.py` — `SharedEncoder`
14. `mycelium/trm/v2/heads.py` — `DomainHead` + `HeadRegistry`
15. `mycelium/trm/v2/gating.py` — `DomainGate`
16. `mycelium/trm/v2/halt.py` — `HaltControllerV2`
17. `mycelium/trm/v2/checkpointing.py` — `HeadManifest`
18. `mycelium/trm/v2/policies.py` — `GateMode` + expansion policies
19. `mycelium/trm/v2/trainer.py` — `TRMV2Trainer`
20. `mycelium/trm/v2/inference.py` — `TRMV2InferenceEngine`

### Phase 4 — Cold Storage
21. `mycelium/cold_storage/quantization.py`
22. `mycelium/cold_storage/replay_buffer.py`
23. `mycelium/cold_storage/drift.py` — `DriftDetector`
24. `mycelium/cold_storage/priority_queue.py` — `ReactivationQueue`
25. `mycelium/cold_storage/reactivation.py` — `DomainReactivationService`
26. `mycelium/cold_storage/manager.py` — `ColdStorageManager`

### Phase 5 — Scripts and Integration
27. `scripts/onboard_new_domain.py`
28. `scripts/migrate_experts_to_sqlite.py`
29. `config.toml` — add `[trm_v2]`, `[domain_graph]`, `[sqlite_experts]`, `[cold_storage]` tables
30. `mycelium/pipeline/config_loader.py` — extend with `load_trm_v2_config()`
31. `mycelium/pipeline/run_workflow.py` — **full rebuild** as thin orchestrator with CLI

### Phase 6 — Tests and Parity
32. All test files under `tests/trm_v2/`, `tests/domain_graph/`, `tests/domain_store/`, `tests/cold_storage/`
33. Shadow-mode comparison run
34. Parity report before PR merge

---

## Acceptance Criteria (from Spec)

- [ ] All feature flags default to `false`; no existing test regressions
- [ ] All five `NoveltyDecision` cases covered by integration test
- [ ] `SQLiteDomainStore` retrieval F1 within 5% of legacy expert on held-out query set
- [ ] `ColdStorageManager` cold-store + reactivation cycle completes without corrupting `DomainNode` state or `HeadManifest`
- [ ] New domain onboarding script runs end-to-end for one test domain without touching any existing domain head weights
- [ ] `TRMV2InferenceEngine.route()` top-1 domain agreement with legacy `MultiLensRouter` ≥ 80% on benchmark query set
- [ ] All `graph_domain_*` observability events contain required fields: `sequence_number`, `event_id`, `timestamp`, `graph_schema_version`, `domain_version`
- [ ] `run_workflow.py` has a working `--backend-only` CLI mode
- [ ] No module contains a hardcoded domain name string literal

---

## Files That Survive Unchanged

These modules are reused directly by the new architecture and are not modified:

| File | Role in v0.2 |
|---|---|
| `mycelium/trm/network.py` | Frozen encoder backbone — `SharedEncoder` wraps this |
| `mycelium/trm/config.py` | TRM v2 config extends this schema |
| `mycelium/trm/embeddings.py` | Embedding utilities reused by encoder |
| `mycelium/trm/graph_store.py` | `DomainGraph` registry is a sibling, not replaced |
| `mycelium/trm/graph_decay.py` | LRU decay logic consulted by `ColdStorageManager` |
| `mycelium/trm/trm_ood_fallback.py` | OOD fallback still applies when gate confidence is low |
| `mycelium/pipeline/dynamic_signature_manager.py` | Spectral signatures reused by `DomainNoveltyPolicy` |
| `mycelium/pipeline/spectral_analyzer.py` | `SpectralAnalyzer` passed into `DomainNoveltyPolicy` |
| `mycelium/pipeline/pipeline_event.py` | All observability events use this infrastructure |
| `mycelium/pipeline/shadow_domain_detector.py` | Novelty detection feeds `DomainGraph` provisional nodes |
| `mycelium/pipeline/patch_dag.py` | Patch lineage stored in `DomainGraph` edges |
| `mycelium/pipeline/layer1_router.py` | Tag extraction, spectral embedding — kept, called before TRM v2 |
| `mycelium/pipeline/layer0/router.py` | `QuestionRouter` — unchanged, runs before everything |
| `mycelium/pipeline/api_models.py` | `ReasoningMode`, `get_depth_config` — unchanged |

---

## Artifact Directory Layout

All new artifacts live under `artifacts/` (already in `.gitignore`):

```
artifacts/
├── trm_v2/
│   ├── encoder/base_v1.0.pt
│   ├── gate/gate_v1.0.pt
│   ├── heads/{domain_id}/
│   │   ├── head_v{HEAD}_d{DOMAIN}.pt
│   │   ├── head_v{HEAD}_d{DOMAIN}_int8.pt
│   │   └── manifest.json
│   └── manifests/global_manifest.json
├── domain_graph/
│   ├── graph_v1.0.json
│   └── snapshots/
├── sqlite_experts/
│   └── {domain_id}.sqlite
└── cold_storage/
    └── {domain_id}/
        ├── bundle_v{DOMAIN}_d{HEAD}.json
        └── replay_v{DOMAIN}.pkl
```

All artifact filenames embed `domain_version` and `head_version` for rollback safety.

---

## Known Technical Debt From `phase-d-wiring` (Do Not Port)

These specific patterns from the old branch must not appear anywhere in the rebuild:

- `_LABEL_TO_ROUTE_OBJ` — global mutation pattern used in DST tag patching
- `fusion_scores` dict with hardcoded keys (`engine_spec`, etc.)
- `phase2_pipeline.run()` + `expert_system.decide()` producing conflicting outputs
- `STARTUP_SPECS` list in `model_registry.py` with any hardcoded domain names
- Any `try/except` that silently swallows import errors and leaves `None` callables that crash mid-pipeline
- `_prev_phase2_result` inter-sentence state carried across the sentence loop without explicit propagation contract

