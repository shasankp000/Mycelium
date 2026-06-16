# Mycelium Repository Restructuring Plan (Hardened Revision)

> **Branch:** `semantic-architecture-impl`
> **Status:** Approved for execution — Hardened Revision
> **Scope:** Root-level `.py` clutter → organized package architecture
> **Goal:** Preserve runtime stability while preparing for long-term semantic runtime evolution and future Rust boundaries

---

# 1. Architectural Philosophy

This restructuring is NOT merely cosmetic repository cleanup.

It is:

> transition from research-style prototype sprawl into stable semantic systems engineering.

The restructuring must:

- preserve runtime stability,
- preserve import determinism,
- preserve semantic pipeline behavior,
- preserve graph/runtime contracts,
- preserve future modularity,
- and avoid introducing hidden architectural coupling.

---

# 2. Core Architectural Decisions

## 2.1 `core/` Remains at Root

`core/` remains:

```text
project-wide shared type contract
```

Used by:

- `layer0/`
- `mycelium/pipeline/`
- tests
- orchestration
- runtime systems

Moving it creates unnecessary dependency complexity.

---

## 2.2 `layer0/` Remains Isolated

`layer0/` remains a standalone package.

Reason:

- safety pre-routing layer,
- dependency isolation,
- prevents semantic-runtime circular imports.

---

## 2.3 `run_workflow.py` Remains at Root

`run_workflow.py` remains:

```text
primary orchestration entrypoint
```

Equivalent to:

- `main.py`
- `manage.py`
- runtime launch controller.

This is intentional.

---

## 2.4 `pipeline/` Is Transitional Runtime Consolidation

`mycelium/pipeline/` is currently:

```text
runtime operational consolidation layer
```

NOT the final long-term architecture.

It intentionally groups:

- routing,
- orchestration,
- inference,
- patch systems,
- MCP,
- graph operations,
- eventing,
- runtime tooling.

---

## IMPORTANT FUTURE NOTE

`pipeline/` is expected to become a future decomposition pressure point.

Potential future split:

```text
mycelium/
├── pipeline/
├── runtime/
├── graph/
├── orchestration/
├── inference/
├── patching/
└── tooling/
```

DO NOT perform this split now.

The current restructuring intentionally prioritizes:

- stabilization,
- import cleanup,
- operational consolidation,
- and runtime coherence.

---

# 3. Critical Runtime Safety Rules

## Rule 1 — Archive Must NOT Become Runtime Dependency

Archived code MUST NOT remain active runtime infrastructure.

This is extremely important.

---

## REQUIRED FIX

The following import:

```python
from archive.phase2_validation.pipeline import Phase2Pipeline
```

is NOT allowed in final runtime architecture.

---

## REQUIRED ACTION

If `Phase2Pipeline` is still runtime-active:

MOVE IT INTO:

```text
mycelium/pipeline/
```

or:

```text
mycelium/runtime/
```

before archival.

---

## Archive Definition

`archive/` means:

```text
historical non-runtime storage
```

NOT:

```text
live runtime dependency source
```

---

## Rule 2 — No Runtime Dependency on `tools/`

`tools/` is strictly:

- one-off utilities,
- developer scripts,
- analysis helpers,
- visualization helpers.

Production/runtime code MUST NOT import from:

```text
tools/
```

---

## Rule 3 — Preserve Import Layer Boundaries

To prevent future architectural spaghetti:

formal import boundaries are established.

---

# Allowed Import Matrix

| Layer | Allowed Imports |
|---|---|
| `core/` | standard library + external libs only |
| `layer0/` | `core/` only |
| `mycelium/pipeline/` | `core/`, `mycelium/*` |
| `mycelium/trainers/` | `core/`, `mycelium/pipeline/` |
| `tools/` | may import anything |
| `tests/` | may import anything |
| `archive/` | MUST NOT be imported by runtime |

---

## Rule 4 — Runtime-Generated Assets Must Stay Isolated

Runtime-generated semantic assets:

- signatures,
- traces,
- caches,
- patch batches,
- reports,
- profiling outputs

must remain operationally isolated.

---

## Future Recommendation

Eventually migrate:

```text
signatures/
```

into:

```text
runtime/signatures/
```

or:

```text
data/signatures/
```

for cleaner runtime separation.

Not required immediately.

---

# 4. Target Repository Layout

```text
Mycelium/
│
├── run_workflow.py
├── config.toml
├── requirements.txt
├── README.md
├── LICENSE / CITATION.cff
├── .gitignore / .gitattributes
├── *.md
│
├── core/
│   ├── __init__.py
│   └── types.py
│
├── mycelium/
│   ├── __init__.py
│   ├── canonicalization/
│   ├── contradiction/
│   ├── fusion/
│   ├── ir/
│   ├── router/
│   ├── trm/
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── api_models.py
│   │   ├── auto_semantic_clusterer.py
│   │   ├── broadcast_api.py
│   │   ├── calibration_api.py
│   │   ├── config_loader.py
│   │   ├── conversation_agent.py
│   │   ├── domain_tool_planner.py
│   │   ├── dynamic_signature_manager.py
│   │   ├── expert_filter.py
│   │   ├── fusion_engine.py
│   │   ├── layer1_router.py
│   │   ├── layer2_expert_loader.py
│   │   ├── lexis_bridge.py
│   │   ├── lexis_condenser.py
│   │   ├── llm_providers.py
│   │   ├── mcp_client.py
│   │   ├── mcp_tools_server.py
│   │   ├── model_registry.py
│   │   ├── multi_lens_router.py
│   │   ├── optimization_engine.py
│   │   ├── orchestration.py
│   │   ├── patch_batch_logger.py
│   │   ├── patch_dataset_logger.py
│   │   ├── patch_dag.py
│   │   ├── pipeline_event.py
│   │   ├── sandbox_manager.py
│   │   ├── sandbox_models.py
│   │   ├── spacy_bridge.py
│   │   ├── spacy_worker.py
│   │   ├── spectral_analyzer.py
│   │   ├── unified_bert_expert.py
│   │   └── unified_expert_system.py
│   │
│   └── trainers/
│       ├── __init__.py
│       ├── train_bert.py
│       └── tuning_config.py
│
├── layer0/
│
├── tools/
│   └── visualize_results.py
│
├── tests/
│   ├── test_multilens_system.py
│   └── fixtures/
│       ├── dummy_models/
│       └── dummy_bert_models/
│
├── archive/
│   ├── phase2_validation/
│   ├── phase3_validation/
│   └── interface_snapshot/
│
├── benchmarks/
├── evaluation_data/
├── training_data/
├── signatures/
├── docs/
├── scripts/
└── web-ui/
```

---

# 5. Additional Runtime Safety Requirements

## 5.1 Config Path Safety

After moving:

```text
config_loader.py
```

into:

```text
mycelium/pipeline/
```

its path resolution MUST be updated.

---

## REQUIRED CHANGE

Config lookup MUST walk upward to root:

```python
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.toml"
```

Hardcoded relative assumptions are forbidden.

---

# 5.2 Dynamic Path Safety

`layer2_expert_loader.py` currently references fixture paths dynamically.

After restructuring:

ALL runtime paths MUST use:

```python
Path(__file__).resolve()
```

NOT:

```python
os.getcwd()
```

NOT:

```python
relative cwd assumptions
```

This is critical for:

- test execution,
- Docker,
- subprocess workers,
- future Rust orchestration.

---

# 5.3 Absolute Internal Imports Only

After restructuring:

ALL internal imports MUST become:

```python
from mycelium.pipeline.foo import X
```

NOT:

```python
import foo
```

This prevents:

- runtime ambiguity,
- accidental shadowing,
- cwd-sensitive imports,
- subprocess import breakage.

---

# 6. Future Rust Boundary Preparation

This restructuring should also preserve future Rust migration boundaries.

Potential future Rust-native systems:

- graph traversal
- semantic hashing
- contradiction propagation
- DAG propagation
- event bus
- stabilization engine
- routing hot paths

Therefore:

all graph/runtime interfaces should remain:

- modular,
- serialization-safe,
- boundary-clean,
- and FFI-friendly.

---

# 7. Updated Execution Stages

| Stage | Action |
|---|---|
| **A** | Create `mycelium/pipeline/__init__.py` and `mycelium/trainers/__init__.py` |
| **B** | Move + rename `layer1_router.py` and `layer2_expert_loader.py` |
| **C** | Fix all internal imports inside those files BEFORE moving remaining modules |
| **D** | Move remaining operational `.py` files into `mycelium/pipeline/` |
| **E** | Rewrite ALL flat imports into absolute package imports |
| **F** | Move trainers into `mycelium/trainers/` |
| **G** | Move utility scripts into `tools/` |
| **H** | Move tests and fixtures |
| **I** | Verify NO runtime imports reference `archive/` |
| **J** | Move validation snapshots into `archive/` |
| **K** | Run global import audit (`grep "import .*prototype"`) |
| **L** | Run full runtime smoke test |
| **M** | Run end-to-end semantic pipeline test |
| **N** | Freeze stabilized import structure |

---

# 8. REQUIRED Verification Checklist

Before merge completion:

- [ ] No remaining flat imports
- [ ] No runtime imports from `archive/`
- [ ] No runtime imports from `tools/`
- [ ] `config.toml` resolves correctly
- [ ] Dynamic fixture paths resolve correctly
- [ ] `run_workflow.py` executes successfully
- [ ] Routing pipeline loads successfully
- [ ] TRM initialization succeeds
- [ ] MultiLensRouter initializes successfully
- [ ] All tests import correctly
- [ ] No circular imports introduced
- [ ] SSE/status emitter pipeline still functions
- [ ] Graph persistence still functions
- [ ] Semantic hashes unchanged after restructuring

---

# 9. Final Architectural Principle

This restructuring intentionally transforms the repository into:

```text
root = semantic control plane
packages = operational runtime substrate
```

The objective is NOT merely cleanliness.

It is:

- runtime determinism,
- architectural isolation,
- future scalability,
- and long-term semantic runtime maintainability.

