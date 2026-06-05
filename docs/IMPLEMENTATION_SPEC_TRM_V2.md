# Mycelium TRM v2 Implementation Specification (v0.2)

## Purpose and Scope

This document is the **repo-aligned** implementation specification for TRM v2 and SQLite-sharded domain experts in Project Mycelium. Every module path, class name, and integration point is derived from a scan of the actual repository structure on branch `web-ui-prototype`.

The goal is a plan you can implement on a new branch (`feature/trm-v2`) and PR-merge when evaluation parity is confirmed, without touching any existing production code paths.

### Existing modules this plan interacts with

| Existing file | Role in TRM v2 |
|---|---|
| `mycelium/trm/network.py` | Frozen encoder backbone initialized from here |
| `mycelium/trm/trainer.py` | TRM v2 trainer mirrors its training loop idiom |
| `mycelium/trm/trm_engine.py` | Legacy engine; TRM v2 engine is a peer, not a replacement |
| `mycelium/trm/config.py` | TRM v2 config extends this schema |
| `mycelium/trm/embeddings.py` | Embedding utilities reused by TRM v2 encoder |
| `mycelium/trm/graph_store.py` | DomainGraph registry is a sibling, not a replacement |
| `mycelium/trm/graph_decay.py` | Decay logic consulted for LRU cold-store scoring |
| `mycelium/trm/trm_ood_fallback.py` | OOD fallback still applies; TRM v2 routes to it when gate confidence is low |
| `mycelium/pipeline/dynamic_signature_manager.py` | Spectral signatures reused directly by DomainGraph novelty policy |
| `mycelium/pipeline/spectral_analyzer.py` | SpectralAnalyzer passed into DomainNoveltyPolicy |
| `mycelium/pipeline/multi_lens_router.py` | MultiLensRouter consults TRM v2 gate if `USE_TRM_V2=True` |
| `mycelium/pipeline/unified_bert_expert.py` | Migration source; SQLite shards replace its retrieval role |
| `mycelium/pipeline/layer2_expert_loader.py` | Expert loader extended to support SQLite-backed domains |
| `mycelium/pipeline/run_workflow.py` | Integration point — feature-flagged TRM v2 path added here |
| `mycelium/pipeline/pipeline_event.py` | DomainGraph observability events emitted using this infrastructure |
| `mycelium/pipeline/shadow_domain_detector.py` | Novelty detection feeds DomainGraph provisional node creation |
| `mycelium/pipeline/patch_dag.py` | Patch lineage stored in DomainGraph edges |
| `config.toml` | New `[trm_v2]`, `[domain_graph]`, `[sqlite_experts]` tables added here |

---

## Branch Strategy

```text
base branch:   web-ui-prototype
new branch:    feature/trm-v2
PR target:     web-ui-prototype (after parity evaluation)
```

No existing files are modified until Phase 5 (integration). All new code lives in new modules.

---

## Feature Flags

Add to `config.toml` under a new top-level table:

```toml
[trm_v2]
enabled = false
encoder_checkpoint = "artifacts/trm_v2/encoder/base.pt"
gate_checkpoint    = "artifacts/trm_v2/gate/gate.pt"
gate_top_k         = 3
gate_mode          = "frozen"            # bootstrap | expansion | frozen

[domain_graph]
enabled              = false
graph_path           = "artifacts/domain_graph/graph.json"
graph_schema_version = "1.0"

[sqlite_experts]
enabled          = false
root_dir         = "artifacts/sqlite_experts"
use_fts5         = true
vector_extension = "auto"              # auto | sqlite-vec | python-fallback

[cold_storage]
enabled                = false
quantize_on_store      = true
replay_min_samples     = 500
replay_max_samples     = 1000
reactivation_queue     = true
```

Read by extending `mycelium/pipeline/config_loader.py` with a `load_trm_v2_config()` function that returns a typed dataclass — consistent with how existing config sections are loaded in that file.

---

## New Module Layout

All new modules live inside `mycelium/`. Nothing outside `mycelium/` is created.

```text
mycelium/
├── trm/
│   └── v2/                                  ← NEW
│       ├── __init__.py
│       ├── encoder.py                       encoder wrapper + freeze logic
│       ├── gating.py                        DomainGate + GateState
│       ├── heads.py                         DomainHead + HeadRegistry
│       ├── halt.py                          HaltController v2
│       ├── inference.py                     TRMV2InferenceEngine
│       ├── trainer.py                       TRMV2Trainer
│       ├── checkpointing.py                 HeadManifest + manifest I/O
│       └── policies.py                      GateMode + expansion policies
│
├── domain_graph/                            ← NEW
│   ├── __init__.py
│   ├── models.py                            DomainNode, DomainEdge, DriftProfile
│   ├── state.py                             DomainState, DomainMode, GateState enums
│   ├── registry.py                          DomainGraphRegistry (read/write/query)
│   ├── novelty.py                           DomainNoveltyPolicy (uses SpectralAnalyzer)
│   ├── events.py                            DomainObservabilityEvent emitter
│   └── persistence.py                       JSON / SQLite-backed graph I/O
│
├── domain_store/                            ← NEW
│   ├── __init__.py
│   ├── schema.py                            SQL DDL constants
│   ├── store.py                             SQLiteDomainStore (CRUD + search)
│   ├── embedder.py                          OfflineEmbedder (runs at ingest time only)
│   ├── ingest.py                            DomainIngestor (text → shard)
│   ├── retrieval.py                         DomainRetrievalEngine (FTS5 + vector)
│   ├── rerank.py                            ResultReranker
│   └── migration.py                         LegacyExpertMigrator
│
└── cold_storage/                            ← NEW
    ├── __init__.py
    ├── manager.py                           ColdStorageManager
    ├── quantization.py                      QuantizationHelper (INT8/FP16)
    ├── reactivation.py                      DomainReactivationService
    ├── replay_buffer.py                     ReplayBufferManager
    ├── priority_queue.py                    ReactivationQueue
    └── drift.py                             DriftDetector
```

---

## Enums and State Models

### `mycelium/domain_graph/state.py`

```python
from enum import Enum

class DomainState(str, Enum):
    CREATING    = "CREATING"
    HOT         = "HOT"
    WARM        = "WARM"
    COLD        = "COLD"
    REMEMBERING = "REMEMBERING"
    DEPRECATED  = "DEPRECATED"
    ARCHIVED    = "ARCHIVED"
    FAILED      = "FAILED"

class DomainMode(str, Enum):
    RETRIEVAL_ONLY = "RETRIEVAL_ONLY"
    FULL_DOMAIN    = "FULL_DOMAIN"

class GateState(str, Enum):
    BOOTSTRAP     = "BOOTSTRAP"
    EXPANSION     = "EXPANSION"
    STABLE        = "STABLE"
    THAWING       = "THAWING"
    RECALIBRATING = "RECALIBRATING"

class NoveltyDecision(str, Enum):
    NEW_DOMAIN      = "NEW_DOMAIN"
    CHILD_DOMAIN    = "CHILD_DOMAIN"
    PATCH_EXISTING  = "PATCH_EXISTING"
    PROVISIONAL     = "PROVISIONAL"
    RETRIEVAL_ONLY  = "RETRIEVAL_ONLY"
```

---

## Data Models

### `mycelium/domain_graph/models.py`

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class DriftProfile:
    semantic_drift:    float = 0.0
    retrieval_drift:   float = 0.0
    routing_drift:     float = 0.0
    confidence_drift:  float = 0.0
    activation_drift:  float = 0.0

@dataclass
class DomainNode:
    domain_id:               str
    label:                   str
    domain_version:          str           = "1.0"
    schema_version:          str           = "1.0"
    head_version:            str           = "1"
    retrieval_version:       str           = "1"
    state:                   str           = "CREATING"
    mode:                    str           = "FULL_DOMAIN"
    semantic_centroid:       List[float]   = field(default_factory=list)
    spectral_signature_path: str | None    = None
    head_checkpoint_path:    str | None    = None
    sqlite_shard_path:       str | None    = None
    replay_buffer_path:      str | None    = None
    parent_domain_id:        str | None    = None
    parent_domains:          List[str]     = field(default_factory=list)
    derived_from:            List[str]     = field(default_factory=list)
    neighbor_ids:            List[str]     = field(default_factory=list)
    creation_reason:         str           = ""
    created_by:              str           = "system"
    creation_timestamp:      float         = 0.0
    updated_at:              float         = 0.0
    activation_stats:        Dict[str, Any]= field(default_factory=dict)
    drift_profile:           DriftProfile  = field(default_factory=DriftProfile)

@dataclass
class DomainEdge:
    src_domain_id: str
    dst_domain_id: str
    relation_type: str          # semantic_neighbor | child_of | patch_of | retrieval_overlap
    weight:        float = 1.0
    metadata:      Dict[str, Any] = field(default_factory=dict)
```

### `mycelium/trm/v2/checkpointing.py`

```python
@dataclass
class HeadManifest:
    domain_id:                  str
    encoder_version:            str
    head_version:               str
    checkpoint_path:            str
    state:                      str
    domain_version:             str
    quantized_checkpoint_path:  str | None = None
    sqlite_shard_path:          str | None = None
    replay_buffer_path:         str | None = None
    created_at:                 str = ""
    updated_at:                 str = ""
    train_data_signature:       str = ""
    parent_domain_id:           str | None = None
    metadata:                   Dict[str, Any] = field(default_factory=dict)
```

All manifests are written as JSON alongside checkpoints. Every artifact file name includes `domain_version` and `head_version` to guarantee rollback safety.

---

## Neural Components

### `mycelium/trm/v2/encoder.py`

The shared encoder wraps the existing `TRMNetwork` from `mycelium/trm/network.py` with a freeze toggle, so the initial TRM checkpoint can be reused as the base encoder without rewriting the backbone.

```python
import torch.nn as nn
from mycelium.trm.network import TRMNetwork

class SharedEncoder(nn.Module):
    def __init__(self, base_network: TRMNetwork) -> None:
        super().__init__()
        self.backbone = base_network

    def forward(self, features: dict) -> torch.Tensor:
        # Returns the pre-head latent z, not the final logits
        ...

    def freeze(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def unfreeze_for_calibration(self) -> None:
        # Used only during THAWING gate state
        for p in self.backbone.parameters():
            p.requires_grad_(True)
```

### `mycelium/trm/v2/heads.py`

```python
class DomainHead(nn.Module):
    """Lightweight MLP head per domain. Typical size: 3 linear layers, hidden_dim=128."""

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "domain_logit":    ...,
            "confidence":      ...,
            "halt_delta":      ...,
            "retrieval_bias":  ...,
        }


class HeadRegistry:
    """Loads, caches, and unloads DomainHead instances keyed by domain_id."""

    def get(self, domain_id: str) -> DomainHead: ...
    def load(self, manifest: HeadManifest) -> DomainHead: ...
    def unload(self, domain_id: str) -> None: ...
    def loaded_domains(self) -> list[str]: ...
```

### `mycelium/trm/v2/gating.py`

```python
class DomainGate(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Returns softmax weights over all registered domain slots
        ...

    def topk(self, z: torch.Tensor, k: int = 3) -> list[tuple[str, float]]:
        # Returns [(domain_id, weight), ...] for top-k domains
        ...

    def set_mode(self, mode: GateState) -> None:
        # Controls requires_grad on gate parameters
        ...
```

**Gate parameter freezing rule:** When `mode == GateState.STABLE` or `GateState.FROZEN`, all gate parameters have `requires_grad=False`. When `mode == GateState.THAWING` or `RECALIBRATING`, only gate weights for the affected domain neighborhood are unfrozen — determined by the DomainGraph neighbor list.

### `mycelium/trm/v2/halt.py`

```python
class HaltControllerV2(nn.Module):
    """
    Global halt head.
    Frozen by default after base training.
    Domain-specific calibration via optional per-domain halt_delta from DomainHead output.
    """

    def forward(
        self,
        z: torch.Tensor,
        head_outputs: dict[str, dict],
    ) -> torch.Tensor:
        # Weighted sum of global halt + per-domain halt_delta contributions
        ...
```

### `mycelium/trm/v2/inference.py`

```python
class TRMV2InferenceEngine:
    def __init__(
        self,
        encoder: SharedEncoder,
        gate: DomainGate,
        head_registry: HeadRegistry,
        domain_graph: DomainGraphRegistry,
        halt_controller: HaltControllerV2,
        cold_storage_manager: ColdStorageManager,
    ) -> None: ...

    def route(
        self,
        query: str,
        predicate_frames: list | None = None,   # PredicateFrame objects from Predicate Engine
        context: dict | None = None,
    ) -> dict:
        """
        Returns:
            {
                "top_domains": [(domain_id, weight), ...],
                "halt_score": float,
                "retrieval_hints": dict,
                "reactivation_needed": list[str],
                "gate_state": str,
            }
        """
        ...
```

**Predicate Engine compatibility:** When `predicate_frames` is supplied, the encoder uses the frame's structured representation as its primary input rather than raw query text. This matches the preferred flow defined in the theoretical spec.

---

## DomainGraph Registry

### `mycelium/domain_graph/registry.py`

```python
class DomainGraphRegistry:

    # CRUD
    def add_domain(self, node: DomainNode) -> None: ...
    def update_domain(self, domain_id: str, updates: dict) -> None: ...
    def get_domain(self, domain_id: str) -> DomainNode | None: ...
    def add_edge(self, edge: DomainEdge) -> None: ...
    def mark_state(self, domain_id: str, state: DomainState) -> None: ...

    # Query
    def nearest_domains(
        self,
        embedding: list[float],
        top_k: int = 5,
        exclude_states: list[DomainState] | None = None,
    ) -> list[DomainNode]: ...

    def all_hot_domains(self) -> list[DomainNode]: ...
    def all_cold_domains(self) -> list[DomainNode]: ...
    def neighbors(self, domain_id: str) -> list[DomainNode]: ...

    # Versioning
    def snapshot(self, path: str | None = None) -> str: ...
    def graph_version(self) -> str: ...
```

Nearest-domain lookup uses cosine similarity over `semantic_centroid` vectors. For small graphs (< 200 domains), numpy is sufficient. For larger graphs, delegate to the existing `SpectralAnalyzer` from `mycelium/pipeline/spectral_analyzer.py` — it already has signature distance computation.

### `mycelium/domain_graph/novelty.py`

Integrates directly with the existing `DynamicSignatureManager` and `SpectralAnalyzer` from `mycelium/pipeline/dynamic_signature_manager.py` and `mycelium/pipeline/spectral_analyzer.py`.

```python
class DomainNoveltyPolicy:
    def __init__(
        self,
        registry: DomainGraphRegistry,
        spectral_analyzer,                  # existing SpectralAnalyzer instance
        centroid_distance_threshold: float = 0.35,
        spectral_distance_threshold: float = 0.40,
        provisional_persistence_threshold: int = 50,
    ) -> None: ...

    def decide(
        self,
        query_embedding: list[float],
        spectral_signature: list[float] | None,
        candidate_samples: list[dict],
    ) -> tuple[NoveltyDecision, str | None]:
        """
        Returns (decision, nearest_domain_id_if_applicable).
        nearest_domain_id is populated for CHILD_DOMAIN and PATCH_EXISTING cases.
        """
        ...
```

This replaces the ad-hoc novelty check that currently lives dispersed across `shadow_domain_detector.py` and `run_workflow.py`, centralizing the decision logic without removing either existing file.

### `mycelium/domain_graph/events.py`

Emits observability events using the existing `pipeline_event.py` infrastructure already in the repo.

```python
from mycelium.pipeline.pipeline_event import PipelineEvent, emit_event

EVENT_TYPES = frozenset({
    "graph_domain_create",
    "graph_domain_activate",
    "graph_domain_coldstore",
    "graph_domain_remember",
    "graph_domain_promote",
    "graph_domain_deprecate",
    "graph_gate_recalibration",
    "graph_drift_detected",
})

def emit_domain_event(
    event_type: str,
    domain_id: str,
    domain_version: str,
    graph_schema_version: str,
    sequence_number: int,
    payload: dict | None = None,
) -> None:
    assert event_type in EVENT_TYPES
    emit_event(PipelineEvent(
        event_type=event_type,
        sequence_number=sequence_number,
        payload={
            "domain_id": domain_id,
            "domain_version": domain_version,
            "graph_schema_version": graph_schema_version,
            **(payload or {}),
        }
    ))
```

---

## SQLite-Sharded Domain Store

### `mycelium/domain_store/schema.py`

```python
SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS chunks (
    id           TEXT PRIMARY KEY,
    text         TEXT NOT NULL,
    embedding    BLOB,
    source       TEXT,
    created_at   TEXT,
    updated_at   TEXT,
    metadata_json TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
    USING fts5(id UNINDEXED, text, content='chunks', content_rowid='rowid');

CREATE TABLE IF NOT EXISTS entities (
    entity_id    TEXT PRIMARY KEY,
    label        TEXT NOT NULL,
    entity_type  TEXT,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS relations (
    relation_id  TEXT PRIMARY KEY,
    src_id       TEXT NOT NULL,
    rel_type     TEXT NOT NULL,
    dst_id       TEXT NOT NULL,
    weight       REAL DEFAULT 1.0,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS replay_samples (
    sample_id    TEXT PRIMARY KEY,
    text         TEXT NOT NULL,
    label_json   TEXT,
    embedding    BLOB,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS shard_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""
```

### `mycelium/domain_store/store.py`

```python
class SQLiteDomainStore:
    """
    One instance per domain SQLite file.
    File naming: artifacts/sqlite_experts/{domain_id}.sqlite
    """

    def __init__(self, db_path: Path) -> None: ...

    # Write
    def upsert_chunk(self, chunk: dict) -> None: ...
    def upsert_entity(self, entity: dict) -> None: ...
    def upsert_relation(self, relation: dict) -> None: ...
    def add_replay_sample(self, sample: dict) -> None: ...
    def set_meta(self, key: str, value: str) -> None: ...

    # Read
    def search_fts(self, query: str, limit: int = 50) -> list[dict]: ...
    def get_chunk_embeddings(self, ids: list[str]) -> list[tuple[str, bytes]]: ...
    def get_replay_samples(self, limit: int = 1000) -> list[dict]: ...
    def get_meta(self, key: str) -> str | None: ...

    # Vector search (Python fallback — no extension required)
    def cosine_search(
        self,
        query_embedding: list[float],
        candidate_ids: list[str],
        limit: int = 20,
    ) -> list[dict]: ...
```

Vector search is a **Python-side cosine similarity** over candidate embeddings fetched by rowid — no sqlite-vec extension required for the initial implementation. Extension support can be added transparently later via `vector_extension = "auto"`.

### `mycelium/domain_store/retrieval.py`

```python
class DomainRetrievalEngine:
    """
    Coordinates retrieval across one SQLite shard for a domain.
    Called from run_workflow.py when USE_SQLITE_SHARDED_EXPERTS=True.
    """

    def retrieve(
        self,
        domain_id: str,
        query: str,
        query_embedding: list[float],
        k: int = 10,
    ) -> list[dict]:
        """
        Flow:
        1. FTS5 prefilter → candidate_ids (up to 100)
        2. cosine_search over candidates
        3. Merge and rerank
        4. Return top-k with text, score, metadata
        """
        ...
```

### `mycelium/domain_store/migration.py`

```python
class LegacyExpertMigrator:
    """
    Reads the existing UnifiedBERTExpert knowledge stores and writes
    SQLite shards for each domain. Read-only from the legacy side.
    """

    def __init__(
        self,
        bert_expert,             # existing UnifiedBERTExpert instance
        store_root: Path,
        embedder: OfflineEmbedder,
    ) -> None: ...

    def export_domain_corpus(self, domain_id: str) -> list[dict]: ...
    def build_shard(self, domain_id: str) -> Path: ...
    def validate_shard(self, shard_path: Path) -> dict: ...
    def run_full_migration(self, domain_ids: list[str]) -> dict: ...
```

This is an **offline script only**, not called at inference time. Wrapped by `scripts/migrate_experts_to_sqlite.py`.

---

## Cold Storage

### `mycelium/cold_storage/manager.py`

```python
class ColdStorageManager:
    """
    Orchestrates cold-store and reactivation lifecycle.
    Consults graph_decay.py LRU scores for eligibility decisions,
    consistent with existing graph decay logic in mycelium/trm/graph_decay.py.
    """

    def should_cold_store(self, domain_id: str) -> bool:
        # Checks: LRU score, activation_stats.last_used, current DomainState
        ...

    def cold_store(self, domain_id: str) -> None:
        # 1. Flush head checkpoint
        # 2. Quantize (INT8 or FP16 depending on config)
        # 3. Snapshot SQLite shard
        # 4. Persist replay buffer
        # 5. Write versioned cold-storage bundle manifest
        # 6. Mark domain COLD in DomainGraphRegistry
        # 7. Emit graph_domain_coldstore event
        ...

    def reactivate(self, domain_id: str, context: dict | None = None) -> None:
        ...

    def current_state(self, domain_id: str) -> DomainState: ...
```

### `mycelium/cold_storage/reactivation.py`

```python
class DomainReactivationService:
    def reactivate(self, domain_id: str, query_context: dict | None = None) -> dict:
        """
        Steps:
        1. Mark domain REMEMBERING
        2. Emit graph_domain_remember event
        3. Reload or dequantize head checkpoint
        4. Open SQLite shard
        5. Run DriftDetector against replay buffer
        6. If drift exceeds threshold on semantic_drift or confidence_drift:
               run localized few-shot adaptation (low LR, head weights only)
        7. Promote to WARM or HOT
        8. Return {domain_id, state, drift_profile, adapted: bool}
        """
        ...
```

### `mycelium/cold_storage/drift.py`

```python
class DriftDetector:
    def compare(
        self,
        current_embeddings: list[list[float]],
        replay_embeddings: list[list[float]],
    ) -> DriftProfile:
        """
        semantic_drift:    mean cosine distance shift
        retrieval_drift:   FTS5 recall delta on held-out queries
        routing_drift:     gate weight distribution shift (KL divergence)
        confidence_drift:  mean confidence delta on replay samples
        activation_drift:  query volume trend change
        """
        ...
```

### `mycelium/cold_storage/priority_queue.py`

```python
import heapq
from dataclasses import dataclass, field

@dataclass(order=True)
class ReactivationRequest:
    priority:       float
    domain_id:      str = field(compare=False)
    query_id:       str = field(compare=False)
    estimated_cost: float = field(compare=False, default=0.0)
    created_at:     float = field(compare=False, default=0.0)

class ReactivationQueue:
    def push(self, request: ReactivationRequest) -> None: ...
    def pop(self) -> ReactivationRequest | None: ...
    def pending(self) -> list[str]: ...
```

---

## Training

### `mycelium/trm/v2/trainer.py`

Mirrors the training loop idiom from the existing `mycelium/trm/trainer.py` and `mycelium/trm/trm_training_sim.py` — same loss names (`l_dom`, `l_halt`), same checkpoint format, same trace writer interface so existing tooling works.

```python
class TRMV2Trainer:

    def train_bootstrap(
        self,
        dataset_bundle: dict,
        epochs: int,
        lr: float = 1e-4,
    ) -> dict:
        """Trains encoder + gate + initial domain heads from scratch."""
        ...

    def train_new_domain_head(
        self,
        domain_id: str,
        dataset_bundle: dict,
        epochs: int,
        lr: float = 5e-5,
        freeze_encoder: bool = True,
        freeze_gate: bool = True,
    ) -> HeadManifest:
        """Trains only the new domain head. All other parameters frozen by default."""
        ...

    def calibrate_gate(
        self,
        samples: list[dict],
        affected_domain_ids: list[str],
        lr: float = 1e-5,
        max_steps: int = 200,
    ) -> dict:
        """
        Restricted gate calibration.
        Only unfreezes weights connected to affected_domain_ids neighborhoods.
        """
        ...
```

New-domain training uses the existing trace generation infra. Add a new function
`generate_new_domain_traces(domain_id, sqlite_shard_path)` in `mycelium/trm/trm_training_sim.py`
that reads samples from the SQLite shard's `replay_samples` table rather than the flat `training_data/` files.

---

## Integration Into `run_workflow.py`

Add the following narrow integration block. **No existing logic is touched.**

```python
# Near top of run_mycelium_workflow()
_cfg = load_trm_v2_config()

# --- TRM v2 routing (shadow mode) ---
_trm_v2_result = None
if _cfg.trm_v2.enabled:
    _trm_v2_result = _get_trm_v2_engine().route(
        query=query,
        predicate_frames=predicate_frames,
        context=context,
    )

# --- Existing routing continues unchanged ---
router = MultiLensRouter(spectral_analyzer=spectral_analyzer)
...

# --- Expert retrieval (SQLite path) ---
if _cfg.sqlite_experts.enabled and _trm_v2_result:
    evidence_bundle = _sqlite_retrieval_engine.retrieve(
        domain_id=_trm_v2_result["top_domains"][0][0],
        query=query,
        query_embedding=query_embedding,
    )
else:
    evidence_bundle = None   # legacy path continues below
```

The legacy `unified_bert_expert.py` path runs unchanged when flags are off, enabling clean shadow-mode comparison before any cutover.

---

## Artifact Directory Layout

All new artifacts live under `artifacts/` (already in `.gitignore`).

```text
artifacts/
├── trm_v2/
│   ├── encoder/
│   │   └── base_v1.0.pt
│   ├── gate/
│   │   └── gate_v1.0.pt
│   ├── heads/
│   │   └── {domain_id}/
│   │       ├── head_v{HEAD}_d{DOMAIN}.pt
│   │       ├── head_v{HEAD}_d{DOMAIN}_int8.pt
│   │       └── manifest.json
│   └── manifests/
│       └── global_manifest.json
│
├── domain_graph/
│   ├── graph_v1.0.json
│   └── snapshots/
│
├── sqlite_experts/
│   ├── {domain_id}.sqlite
│   └── ...
│
└── cold_storage/
    └── {domain_id}/
        ├── bundle_v{DOMAIN}_d{HEAD}.json
        └── replay_v{DOMAIN}.pkl
```

All artifact file names embed `domain_version` and `head_version` to guarantee rollback safety and replay integrity.

---

## New Scripts

### `scripts/onboard_new_domain.py`

```text
Usage:
  python scripts/onboard_new_domain.py \
    --domain-id "quantum_physics" \
    --label "Quantum Physics" \
    --corpus-dir data/quantum_physics/ \
    --parent-domain "physics" \
    --reason "Persistent query volume on quantum topics" \
    --created-by "shasankp000"
```

Steps performed:
1. `DomainNoveltyPolicy.decide()` — confirm domain is genuinely new or sub-domain.
2. Embed corpus via `OfflineEmbedder`.
3. Build SQLite shard via `DomainIngestor`.
4. Create `DomainNode` and register in `DomainGraphRegistry`.
5. Train new `DomainHead` via `TRMV2Trainer.train_new_domain_head()`.
6. Write `HeadManifest`.
7. Emit `graph_domain_create` event.

### `scripts/migrate_experts_to_sqlite.py`

```text
Usage:
  python scripts/migrate_experts_to_sqlite.py \
    --domains physics history general \
    --validate
```

Reads the existing `UnifiedBERTExpert` knowledge stores (read-only) and writes SQLite shards. Does not modify the legacy expert in any way.

---

## Tests

Mirror the existing test layout under `tests/`. All TRM v2 tests go in `tests/trm_v2/`.

```text
tests/
├── trm_v2/
│   ├── test_encoder.py
│   ├── test_gating.py
│   ├── test_heads.py
│   ├── test_inference.py
│   └── test_checkpointing.py
├── domain_graph/
│   ├── test_registry.py
│   ├── test_novelty.py
│   └── test_events.py
├── domain_store/
│   ├── test_store.py
│   ├── test_retrieval.py
│   └── test_migration.py
└── cold_storage/
    ├── test_manager.py
    ├── test_reactivation.py
    └── test_drift.py
```

Minimum test coverage required before PR:
- All enum transitions in the `DomainState` state machine.
- `DomainNoveltyPolicy` for all five `NoveltyDecision` cases.
- `SQLiteDomainStore` CRUD and hybrid FTS5 + cosine search.
- `HeadManifest` roundtrip serialization.
- `ColdStorageManager` cold-store + reactivation end-to-end without state corruption.
- `TRMV2InferenceEngine.route()` with and without predicate frames.

---

## Build Order

**Phase 1 — Data layer** (no model dependencies):
1. `mycelium/domain_graph/state.py`
2. `mycelium/domain_graph/models.py`
3. `mycelium/domain_graph/persistence.py`
4. `mycelium/domain_graph/registry.py`
5. `mycelium/domain_store/schema.py`
6. `mycelium/domain_store/store.py`
7. `mycelium/domain_store/embedder.py`
8. `mycelium/domain_store/ingest.py`
9. `mycelium/domain_store/retrieval.py`

**Phase 2 — Graph intelligence:**
10. `mycelium/domain_graph/novelty.py`
11. `mycelium/domain_graph/events.py`
12. `mycelium/domain_store/migration.py`

**Phase 3 — Neural layer:**
13. `mycelium/trm/v2/encoder.py`
14. `mycelium/trm/v2/heads.py`
15. `mycelium/trm/v2/gating.py`
16. `mycelium/trm/v2/halt.py`
17. `mycelium/trm/v2/checkpointing.py`
18. `mycelium/trm/v2/policies.py`
19. `mycelium/trm/v2/trainer.py`
20. `mycelium/trm/v2/inference.py`

**Phase 4 — Cold storage:**
21. `mycelium/cold_storage/quantization.py`
22. `mycelium/cold_storage/replay_buffer.py`
23. `mycelium/cold_storage/drift.py`
24. `mycelium/cold_storage/priority_queue.py`
25. `mycelium/cold_storage/reactivation.py`
26. `mycelium/cold_storage/manager.py`

**Phase 5 — Scripts and integration:**
27. `scripts/onboard_new_domain.py`
28. `scripts/migrate_experts_to_sqlite.py`
29. `config.toml` — add new tables
30. `mycelium/pipeline/config_loader.py` — extend with `load_trm_v2_config()`
31. `mycelium/pipeline/run_workflow.py` — narrow feature-flagged integration block

**Phase 6 — Tests and parity evaluation:**
32. All test files
33. Shadow-mode comparison run against existing domains
34. Parity report before PR

---

## Acceptance Criteria for PR

- All feature flags default to `false`; no existing test regressions.
- All five `NoveltyDecision` cases covered by integration test.
- `SQLiteDomainStore` retrieval quality on migrated domain matches or exceeds legacy expert (target: F1 within 5% on held-out query set).
- `ColdStorageManager` cold-store + reactivation cycle completes without corrupting `DomainNode` state or `HeadManifest`.
- New domain onboarding script runs end-to-end for one test domain without touching any existing domain head weights.
- `TRMV2InferenceEngine.route()` top-1 domain agreement with legacy `MultiLensRouter` ≥ 80% on benchmark query set.
- All `graph_domain_*` observability events contain required fields: `sequence_number`, `event_id`, `timestamp`, `graph_schema_version`, `domain_version`.
