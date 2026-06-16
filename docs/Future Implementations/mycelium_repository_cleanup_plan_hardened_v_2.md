# Mycelium — Repository Cleanup Plan (Hardened Revision)

**Branch:** `semantic-architecture-impl`
**Scanned:** 2026-05-20
**Revision:** Hardened Cleanup Pass

---

# Overview

The root directory currently contains ~70+ files and folders, many of which are stale phase-snapshot docs, loose test scripts, dummy scaffolding folders, and runtime-generated output directories that were committed by mistake.

This hardened revision modifies the original cleanup strategy to ensure:

- architectural lineage is preserved,
- historical optimization baselines are retained,
- debugging archaeology is not accidentally destroyed,
- replay/debugging assets remain available,
- and semantic/runtime interfaces are frozen before aggressive cleanup.

Cleanup actions are now split into:

- DELETE
- MOVE
- ARCHIVE
- GITIGNORE
- KEEP

rather than only delete/move.

---

# NEW GLOBAL RULES (CRITICAL)

## Rule 1 — No Immediate Permanent Deletion

Before permanent deletion:

```text
archive first
validate later
```

Any file with:

- architectural history,
- tuning data,
- implementation lineage,
- optimization rationale,
- debugging value,
- or semantic design history

MUST first move into:

```text
archive/
```

for at least one stabilization/testing cycle.

---

## Rule 2 — Interface Freeze Before Cleanup

Before refactoring or deleting:

Create:

```text
interface_snapshot/
```

containing:

- API payload examples
- graph serialization examples
- semantic hash examples
- SSE event schema examples
- IR schema examples
- lifecycle state examples

This prevents cleanup from accidentally breaking:

- replay systems,
- graph persistence,
- frontend synchronization,
- semantic hashing,
- cache compatibility.

---

## Rule 3 — Dependency Scan Before Deletion

Before deleting ANY file:

Run:

```bash
grep/import scan
```

to ensure:

- no imports remain,
- no runtime references exist,
- no dynamic loaders depend on it,
- no reflection/plugin systems reference it.

This is especially important for:

- config-driven systems,
- runtime module loading,
- plugin discovery,
- dynamic import systems.

---

## Rule 4 — Preserve Benchmark Baselines

Before cleanup:

Persist:

```text
benchmarks/pre_cleanup/
```

containing:

- wall-clock timings
- memory usage
- graph counts
- cache hit rates
- stabilization latency
- semantic routing timings

This allows post-cleanup regression validation.

---

# Category 1: Loose Test Files (root level)

These `test_*.py` files are sitting in the root instead of `tests/`.

| File | Action | Reason |
|---|---|---|
| `test_biobert_endtoend.py` | **Move → `tests/archive/`** or delete later | Historical BioBERT integration reference may still help future regression debugging |
| `test_expert_inference_clean.py` | **Move → `tests/archive/`** | One-off validation snapshot but still useful as historical inference reference |
| `test_multilens_system.py` | **Move → `tests/`** | Still architecturally important |

---

# Category 2: Dummy / Scaffold Folders

| Folder | Action | Reason |
|---|---|---|
| `dummy_bert_models/` | **Archive → `archive/deprecated_scaffolds/`** then delete later | Prevent accidental loss of development scaffolding references |
| `dummy_models/` | **Archive → `archive/deprecated_scaffolds/`** then delete later | Same |

---

# Category 3: One-Off Data Download Scripts

| File | Action | Reason |
|---|---|---|
| `download_chemistry_datasets.py` | **Move → `scripts/`** | Utility/setup script |
| `download_physics_datasets.py` | **Move → `scripts/`** | Same |
| `physics_datasets_sources.md` | **Move → `docs/archive/dataset_notes/`** | Historical sourcing notes may still be useful |

---

# Category 4: Stale Phase-Snapshot Markdown Files

The original cleanup plan deleted most phase files outright.

This hardened revision changes that.

These files contain:

- optimization history,
- architectural evolution,
- failed approaches,
- tuning rationale,
- debugging lineage.

Those may become valuable later.

---

## REQUIRED CHANGE

Move phase files into:

```text
archive/phases/
```

NOT direct deletion.

---

| File | New Action | Reason |
|---|---|---|
| `PHASE_1_SPECTRAL_ANALYSIS.md` | **Archive** | Historical spectral reasoning lineage |
| `PHASE_1_VERIFICATION.md` | **Archive** | Historical validation reference |
| `PHASE_2_FUSION_ENGINE.md` | **Archive** | Fusion-engine evolution history |
| `PHASE_3_OPTIMIZATION_ENGINE.md` | **Archive (IMPORTANT)** | Optimization archaeology may matter later |
| `PHASE_4_INTEGRATION.md` | **Archive** | Integration lineage |
| `PHASE_5_TESTING_VALIDATION.md` | **Archive (IMPORTANT)** | Historical test assumptions may help later debugging |
| `PHASE_6_TUNING_RESULTS.md` | **Move → `docs/archive/tuning/`** | Contains useful tuning history |
| `PHASE_7_DOCUMENTATION_COMPLETE.md` | **Archive** | Meta-history still useful for timeline reconstruction |
| `mycelium_brainstorming_notes.md` | **Archive → `archive/brainstorming/`** | Future rediscovery prevention |
| `IMPLEMENTATION_REPORT.md` | **Move → `docs/`** | Long-term permanent value |

---

# Category 5: Runtime Output Folders

The original plan only partially reorganized runtime artifacts.

This hardened revision centralizes ALL runtime-generated artifacts.

---

# REQUIRED NEW STRUCTURE

```text
runtime/
├── traces/
├── patch_batches/
├── reports/
├── cache/
├── profiling/
├── temp/
└── generated/
```

---

## Updated Actions

| Folder | Action | Reason |
|---|---|---|
| `patch_batches/` | **Move → `runtime/patch_batches/` + gitignore** | Runtime-generated |
| `traces/` | **Move → `runtime/traces/` + gitignore** | Runtime-generated |
| `reports/` | **Move → `runtime/reports/` + conditional gitignore** | Keep generated reports centralized |
| `evaluation_data/` | **Split into static/generated** | Prevent benchmark drift |

---

## REQUIRED `evaluation_data/` Split

```text
evaluation_data/
├── static/
├── generated/
└── benchmarks/
```

Where:

- `static/` → tracked
- `generated/` → gitignored
- `benchmarks/` → optionally tracked

---

# Category 6: Keep As-Is (Permanent Root-Level Docs)

These remain valid permanent docs.

No action needed.

- `README.md`
- `ARCHITECTURE_OVERVIEW.md`
- `CONFIGURATION_REFERENCE.md`
- `OPERATIONAL_GUIDE.md`
- `QUICK_START.md`
- `OPTIMIZATION_SPEC.md`
- `ROUTING_DECISION_TREE.md`
- `MYCELIUM_SYSTEM_OVERVIEW.md`
- `LIMITATIONS_AND_FUTURE_WORK.md`
- `DUAL_VENV_SETUP.md`
- `BERT_SETUP_GUIDE.md`
- `DOCUMENTATION_INDEX.md`
- `CITATION.cff`
- `CONTRIBUTOR_AGREEMENT.md`
- `CREDITS.md`
- `LICENSE`

---

# NEW Category 7: Archive Policy

Introduce:

```text
archive/
├── phases/
├── brainstorming/
├── deprecated_scaffolds/
├── obsolete_tests/
├── tuning/
├── snapshots/
└── dataset_notes/
```

---

## Purpose

Prevents:

- irreversible loss of historical reasoning,
- accidental deletion of optimization rationale,
- future architectural rediscovery loops.

---

# Proposed Final Root Structure (Updated)

```text
Mycelium/
├── [permanent .md docs]
├── [core .py source files]
├── scripts/
├── tests/
├── docs/
├── archive/
├── runtime/
├── benchmarks/
├── interface_snapshot/
├── core/
├── layer0/
├── mycelium/
├── web-ui/
├── trainers/
├── training_data/
├── signatures/
├── expert_post_check/
└── .gitignore
```

---

# Updated `.gitignore` Recommendations

```gitignore
runtime/**
!runtime/.gitkeep

benchmark_results/**
cache/**

*.trace
*.patchbatch
*.profile
```

---

# Updated Staged Cleanup Order

| Stage | Contents |
|---|---|
| **Stage 0** | Create `interface_snapshot/` + benchmark baselines |
| **Stage 1** | Dependency scans on all candidate deletions |
| **Stage 2** | Move phase docs → `archive/` instead of deleting |
| **Stage 3** | Move test files into `tests/` or `tests/archive/` |
| **Stage 4** | Centralize runtime artifacts into `runtime/` |
| **Stage 5** | Split `evaluation_data/` into static/generated/benchmarks |
| **Stage 6** | Update `.gitignore` |
| **Stage 7** | Full end-to-end validation |
| **Stage 8** | Only AFTER stabilization: permanent deletion review |

---

# Final Architectural Principle

This cleanup is NOT merely repository tidying.

It is:

> transition from rapid architecture evolution into stable semantic systems engineering.

Therefore:

- historical lineage matters,
- replayability matters,
- benchmark preservation matters,
- interface stability matters,
- and debugging archaeology matters.

