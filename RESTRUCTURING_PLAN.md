# Mycelium Repository Restructuring Plan

> **Branch:** `semantic-architecture-impl`  
> **Status:** Approved for execution — Option 1 (Full move + fix imports)  
> **Scope:** Root-level `.py` clutter → organised `mycelium/pipeline/` package  
> **Does NOT touch:** files already moved in Stages 1–6 of the semantic architecture impl

---

## 1. Design Rationale

The goal is a root that looks like [Lexis](https://github.com/shasankp000/Lexis): only the
primary entry point, project config, and repo metadata at the top level. All operational
source lives inside named packages.

### Key decisions from pre-execution audit

| Question | Finding | Decision |
|---|---|---|
| Do `core/` and `mycelium/` overlap? | **No.** `core/` = shared type contract (`Layer0Result`, `RoutingResult`, `ExpertDecisionResult`). `mycelium/` = semantic architecture sub-package (canonicalization, contradiction, fusion, IR, router, TRM). | Keep both as-is, unchanged. |
| Is `layer_1_prototype.py` live code? | **Yes — the most critical file in the repo.** Contains the entire multi-lens routing engine, domain ontology, temporal locality layer, tag extraction, LRU caches, telemetry events. Directly imported by `run_workflow.py`. | Rename → `layer1_router.py` and move to `mycelium/pipeline/`. |
| Is `layer_2_prototype.py` live code? | **Yes — transitional but active.** Provides `ExpertModel` + `get_expert_model()` (legacy SVM expert loader). Imported by `layer_1_prototype.py`. | Rename → `layer2_expert_loader.py` and move to `mycelium/pipeline/`. |
| Where does `train_bert.py` belong? | Training script, not pipeline. | Move to `mycelium/trainers/`. |
| Where does `tuning_config.py` belong? | Training hyperparameter config, used by trainers. | Move to `mycelium/trainers/`. |
| Where does `visualize_results.py` belong? | One-off utility / analysis script. | Move to `tools/`. |
| Where does `test_multilens_system.py` belong? | Stray test file at root. | Move to `tests/`. |
| Where do `phase2_validation/`, `phase3_validation/`, `interface_snapshot/` belong? | Historical validation runs / snapshots. | Move to `archive/`. |
| Where do `dummy_models/`, `dummy_bert_models/` belong? | Test fixtures only. | Move to `tests/fixtures/`. |

---

## 2. Target Directory Layout

```
Mycelium/
│
├── run_workflow.py                  ← entry point (STAYS at root)
├── config.toml                      ← project config (STAYS)
├── requirements.txt                 ← (STAYS)
├── README.md                        ← (STAYS)
├── LICENSE / CITATION.cff           ← (STAYS)
├── .gitignore / .gitattributes      ← (STAYS)
├── *.md  (ARCHITECTURE_OVERVIEW, OPERATIONAL_GUIDE, etc.) ← (STAYS)
│
├── core/                            ← shared type contract (UNCHANGED)
│   ├── __init__.py
│   └── types.py                     ← Layer0Result, RoutingResult, ExpertDecisionResult
│
├── mycelium/                        ← semantic architecture engine (UNCHANGED)
│   ├── __init__.py
│   ├── canonicalization/
│   ├── contradiction/
│   ├── fusion/
│   ├── ir/
│   ├── router/
│   ├── trm/
│   │
│   ├── pipeline/                    ← NEW: all operational root-level .py files land here
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
│   │   ├── layer1_router.py         ← renamed from layer_1_prototype.py ✨
│   │   ├── layer2_expert_loader.py  ← renamed from layer_2_prototype.py ✨
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
│   └── trainers/                   ← NEW: training scripts sub-package
│       ├── __init__.py
│       ├── train_bert.py            ← moved from root
│       └── tuning_config.py        ← moved from root
│
├── layer0/                         ← pre-routing safety layer (UNCHANGED)
│
├── tools/                          ← NEW: one-off utility scripts
│   └── visualize_results.py        ← moved from root
│
├── tests/                          ← (EXISTS — append to)
│   ├── test_multilens_system.py    ← moved from root
│   └── fixtures/                   ← NEW: test data / stub models
│       ├── dummy_models/           ← moved from root
│       └── dummy_bert_models/      ← moved from root
│
├── archive/                        ← (EXISTS — append to)
│   ├── phase2_validation/          ← moved from root
│   ├── phase3_validation/          ← moved from root
│   └── interface_snapshot/         ← moved from root
│
├── benchmarks/          ← (STAYS at root)
├── evaluation_data/     ← (STAYS at root)
├── training_data/       ← (STAYS at root)
├── signatures/          ← (STAYS at root — runtime-generated)
├── docs/                ← (STAYS)
├── scripts/             ← (STAYS)
├── trainers/            ← DELETE after merge into mycelium/trainers/
└── web-ui/              ← (STAYS at root)
```

---

## 3. File Move Table

Every root-level `.py` file and its destination.

| Current path | New path | Notes |
|---|---|---|
| `api_models.py` | `mycelium/pipeline/api_models.py` | |
| `auto_semantic_clusterer.py` | `mycelium/pipeline/auto_semantic_clusterer.py` | |
| `broadcast_api.py` | `mycelium/pipeline/broadcast_api.py` | |
| `calibration_api.py` | `mycelium/pipeline/calibration_api.py` | |
| `config_loader.py` | `mycelium/pipeline/config_loader.py` | |
| `conversation_agent.py` | `mycelium/pipeline/conversation_agent.py` | |
| `domain_tool_planner.py` | `mycelium/pipeline/domain_tool_planner.py` | |
| `dynamic_signature_manager.py` | `mycelium/pipeline/dynamic_signature_manager.py` | |
| `expert_filter.py` | `mycelium/pipeline/expert_filter.py` | |
| `fusion_engine.py` | `mycelium/pipeline/fusion_engine.py` | |
| `layer_1_prototype.py` | `mycelium/pipeline/layer1_router.py` | **Renamed** ✨ |
| `layer_2_prototype.py` | `mycelium/pipeline/layer2_expert_loader.py` | **Renamed** ✨ |
| `lexis_bridge.py` | `mycelium/pipeline/lexis_bridge.py` | |
| `lexis_condenser.py` | `mycelium/pipeline/lexis_condenser.py` | |
| `llm_providers.py` | `mycelium/pipeline/llm_providers.py` | |
| `mcp_client.py` | `mycelium/pipeline/mcp_client.py` | |
| `mcp_tools_server.py` | `mycelium/pipeline/mcp_tools_server.py` | |
| `model_registry.py` | `mycelium/pipeline/model_registry.py` | |
| `multi_lens_router.py` | `mycelium/pipeline/multi_lens_router.py` | |
| `optimization_engine.py` | `mycelium/pipeline/optimization_engine.py` | |
| `orchestration.py` | `mycelium/pipeline/orchestration.py` | |
| `patch_batch_logger.py` | `mycelium/pipeline/patch_batch_logger.py` | |
| `patch_dataset_logger.py` | `mycelium/pipeline/patch_dataset_logger.py` | |
| `patch_dag.py` | `mycelium/pipeline/patch_dag.py` | |
| `pipeline_event.py` | `mycelium/pipeline/pipeline_event.py` | |
| `sandbox_manager.py` | `mycelium/pipeline/sandbox_manager.py` | |
| `sandbox_models.py` | `mycelium/pipeline/sandbox_models.py` | |
| `spacy_bridge.py` | `mycelium/pipeline/spacy_bridge.py` | |
| `spacy_worker.py` | `mycelium/pipeline/spacy_worker.py` | |
| `spectral_analyzer.py` | `mycelium/pipeline/spectral_analyzer.py` | |
| `unified_bert_expert.py` | `mycelium/pipeline/unified_bert_expert.py` | |
| `unified_expert_system.py` | `mycelium/pipeline/unified_expert_system.py` | |
| `train_bert.py` | `mycelium/trainers/train_bert.py` | |
| `tuning_config.py` | `mycelium/trainers/tuning_config.py` | |
| `visualize_results.py` | `tools/visualize_results.py` | |
| `test_multilens_system.py` | `tests/test_multilens_system.py` | |
| `dummy_models/` | `tests/fixtures/dummy_models/` | |
| `dummy_bert_models/` | `tests/fixtures/dummy_bert_models/` | |
| `phase2_validation/` | `archive/phase2_validation/` | |
| `phase3_validation/` | `archive/phase3_validation/` | |
| `interface_snapshot/` | `archive/interface_snapshot/` | |
| `trainers/` (root) | `mycelium/trainers/` (merge) | Delete root `trainers/` after merge |

---

## 4. Import Rewrite Table

All files requiring import path changes after the move.  
Pattern: flat `import foo` / `from foo import X` → `from mycelium.pipeline.foo import X`.

### `run_workflow.py` (stays at root, imports `mycelium.pipeline.*`)

| Old import | New import |
|---|---|
| `from layer_1_prototype import (extract_tags_llama, ...)` | `from mycelium.pipeline.layer1_router import (extract_tags_llama, ...)` |
| `from multi_lens_router import MultiLensRouter` | `from mycelium.pipeline.multi_lens_router import MultiLensRouter` |
| `from phase2_validation.pipeline import Phase2Pipeline` | `from archive.phase2_validation.pipeline import Phase2Pipeline` |
| `import config_loader as cfg` | `from mycelium.pipeline import config_loader as cfg` |
| `from conversation_agent import ...` | `from mycelium.pipeline.conversation_agent import ...` |
| `from broadcast_api import ...` | `from mycelium.pipeline.broadcast_api import ...` |
| *(any other flat imports of moved modules)* | `from mycelium.pipeline.<module> import ...` |

### `mycelium/pipeline/layer1_router.py` (was `layer_1_prototype.py`)

| Old import | New import |
|---|---|
| `import config_loader as cfg` | `from mycelium.pipeline import config_loader as cfg` |
| `from layer_2_prototype import get_expert_model` | `from mycelium.pipeline.layer2_expert_loader import get_expert_model` |
| `from model_registry import embed_batch` | `from mycelium.pipeline.model_registry import embed_batch` |
| `from pipeline_event import build_event` | `from mycelium.pipeline.pipeline_event import build_event` |

### `mycelium/pipeline/layer2_expert_loader.py` (was `layer_2_prototype.py`)

| Old import | New import |
|---|---|
| `sys.path.append(os.path.join(..., 'dummy_models', ...))` | Update base path: `os.path.join(os.path.dirname(__file__), '..', '..', '..', 'tests', 'fixtures', 'dummy_models', ...)` |

### `mycelium/pipeline/multi_lens_router.py`

| Old import | New import |
|---|---|
| `from core.types import RoutingResult` | *(unchanged — `core/` stays at root, still importable as `core.types`)* |
| `from tuning_config import ...` | `from mycelium.trainers.tuning_config import ...` |
| `from layer_1_prototype import multi_lens_route` | `from mycelium.pipeline.layer1_router import multi_lens_route` |

### `mycelium/pipeline/orchestration.py`

| Old import | New import |
|---|---|
| `from core.types import ExpertDecisionResult, RoutingResult` | *(unchanged)* |
| *(any flat sibling imports)* | `from mycelium.pipeline.<module> import ...` |

### `mycelium/pipeline/unified_expert_system.py`

| Old import | New import |
|---|---|
| `from core.types import ExpertDecisionResult, RoutingResult` | *(unchanged)* |
| `from layer_2_prototype import get_expert_model` | `from mycelium.pipeline.layer2_expert_loader import get_expert_model` |
| *(any flat sibling imports)* | `from mycelium.pipeline.<module> import ...` |

### `layer0/router.py`

| Old import | New import |
|---|---|
| `from core.types import Layer0Result` | *(unchanged)* |

### `tests/test_multilens_system.py` (moved from root)

| Old import | New import |
|---|---|
| `from core.types import RoutingResult` | *(unchanged — `core/` still at root)* |
| `from multi_lens_router import MultiLensRouter` | `from mycelium.pipeline.multi_lens_router import MultiLensRouter` |
| `from layer_1_prototype import ...` | `from mycelium.pipeline.layer1_router import ...` |

### `tests/integration/test_routing_and_expert_flow.py`

| Old import | New import |
|---|---|
| `from core.types import ExpertDecisionResult, RoutingResult` | *(unchanged)* |
| `from orchestration import combine_routing_and_expert_decisions` | `from mycelium.pipeline.orchestration import combine_routing_and_expert_decisions` |

### `mycelium/trainers/train_bert.py` (moved from root)

| Old import | New import |
|---|---|
| `from tuning_config import ...` | `from mycelium.trainers.tuning_config import ...` |
| `from model_registry import ...` | `from mycelium.pipeline.model_registry import ...` |
| `from unified_bert_expert import ...` | `from mycelium.pipeline.unified_bert_expert import ...` |

---

## 5. New Files to Create

| File | Content |
|---|---|
| `mycelium/pipeline/__init__.py` | Empty (marks package) |
| `mycelium/trainers/__init__.py` | Empty (marks package) |
| `tools/__init__.py` | Empty (optional, marks package) |
| `tests/fixtures/.gitkeep` | Empty placeholder |

---

## 6. Execution Stages

Work will be carried out in atomic commits on `semantic-architecture-impl`.
Each stage is independently reviewable and bisectable.

| Stage | Action | Files touched |
|---|---|---|
| **A** | Create `mycelium/pipeline/__init__.py` and `mycelium/trainers/__init__.py` | 2 new files |
| **B** | Move + rename the two prototype files (`layer1_router.py`, `layer2_expert_loader.py`), fix their internal imports | 2 files |
| **C** | Move remaining 30 operational `.py` files to `mycelium/pipeline/`, fix internal cross-imports within the group | ~30 files |
| **D** | Move `train_bert.py` + `tuning_config.py` → `mycelium/trainers/`, fix their imports | 2 files |
| **E** | Move `visualize_results.py` → `tools/`, move `test_multilens_system.py` → `tests/` | 2 files |
| **F** | Fix `run_workflow.py` imports to point at `mycelium.pipeline.*` | 1 file |
| **G** | Fix `tests/integration/test_routing_and_expert_flow.py` imports | 1 file |
| **H** | Move `dummy_models/` + `dummy_bert_models/` → `tests/fixtures/`, update path in `layer2_expert_loader.py` | dirs + 1 file |
| **I** | Move `phase2_validation/`, `phase3_validation/`, `interface_snapshot/` → `archive/` | 3 dirs |
| **J** | Merge root `trainers/` into `mycelium/trainers/`, delete root `trainers/` | cleanup |
| **K** | Final sweep: verify no remaining flat imports of moved modules across all `.py` files | audit only |

---

## 7. Invariants (Must Not Change)

- `core/` **stays at root** — it is the shared type contract imported by both `layer0/` and
  `mycelium/pipeline/` files. Moving it would break `from core.types import ...` in 6+ files
  with no benefit.
- `run_workflow.py` **stays at root** — it is the primary entry point, equivalent to `main.py`
  in Lexis.
- `config.toml` **stays at root** — `config_loader.py` reads it relative to `__file__`; after
  `config_loader` moves to `mycelium/pipeline/`, the path resolving logic must be updated to
  walk up two levels to find `config.toml` at the project root.
- `layer0/` **stays at root** — it is a self-contained safety layer package already correctly
  structured; nesting it inside `mycelium/` would create circular dependency risk.
- `mycelium/` sub-packages (canonicalization, contradiction, fusion, ir, router, trm) **stay
  unchanged** — they were correctly placed in the semantic architecture work (Stages 1–5).

---

## 8. Post-Restructuring Root (Final State)

After all stages complete, `ls` at the project root should show **only**:

```
.gitattributes        .gitignore
ARCHITECTURE_OVERVIEW.md   BERT_SETUP_GUIDE.md
CITATION.cff         CONFIGURATION_REFERENCE.md
CONTRIBUTOR_AGREEMENT.md   CREDITS.md
DUAL_VENV_SETUP.md   IMPLEMENTATION_REPORT.md
LICENSE              LIMITATIONS_AND_FUTURE_WORK.md
MYCELIUM_SYSTEM_OVERVIEW.md   OPERATIONAL_GUIDE.md
OPTIMIZATION_SPEC.md QUICK_START.md
README.md            ROUTING_DECISION_TREE.md
RESTRUCTURING_PLAN.md
config.toml          requirements.txt
run_workflow.py

archive/    benchmarks/   core/      docs/
evaluation_data/   layer0/    mycelium/  scripts/
signatures/  tests/   tools/   training_data/   web-ui/
```

No loose `.py` files. No vestigial stage directories. Clean.
