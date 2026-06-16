# Mycelium — Repository Cleanup Plan

**Branch:** `semantic-architecture-impl`
**Scanned:** 2026-05-20

---

## Overview

The root directory currently contains ~70+ files and folders, many of which are stale phase-snapshot docs, loose test scripts, dummy scaffolding folders, and runtime-generated output directories that were committed by mistake. This plan organises them into four cleanup actions: **delete**, **move**, **gitignore**, and **keep**.

---

## Category 1: Loose Test Files (root level)

These `test_*.py` files are sitting in the root instead of `tests/`.

| File | Action | Reason |
|---|---|---|
| `test_biobert_endtoend.py` | **Move → `tests/`** or **delete** | Ad-hoc BioBERT integration test; BioBERT is no longer the primary expert system |
| `test_expert_inference_clean.py` | **Move → `tests/`** or **delete** | One-off inference validation snapshot, likely stale |
| `test_multilens_system.py` | **Move → `tests/`** | Most relevant of the three — keep but properly house it |

---

## Category 2: Dummy / Scaffold Folders

| Folder | Action | Reason |
|---|---|---|
| `dummy_bert_models/` | **Delete** | Placeholder scaffold — real models are loaded from `config.toml` at runtime |
| `dummy_models/` | **Delete** | Same — development stubs with no runtime role |

---

## Category 3: One-Off Data Download Scripts (root clutter)

| File | Action | Reason |
|---|---|---|
| `download_chemistry_datasets.py` | **Move → `scripts/`** | Utility/setup script, not core pipeline logic |
| `download_physics_datasets.py` | **Move → `scripts/`** | Same |
| `physics_datasets_sources.md` | **Move → `docs/`** or **delete** | Dataset sourcing notes — not doc-worthy enough to stay root-level |

---

## Category 4: Stale Phase-Snapshot Markdown Files

These were one-time development journals documenting work-in-progress phases. The architecture has evolved well past all of them, and their content is either superseded by permanent docs or no longer relevant.

| File | Action | Reason |
|---|---|---|
| `PHASE_1_SPECTRAL_ANALYSIS.md` | **Delete** | Superseded by `ARCHITECTURE_OVERVIEW.md` |
| `PHASE_1_VERIFICATION.md` | **Delete** | One-time verification snapshot, no ongoing value |
| `PHASE_2_FUSION_ENGINE.md` | **Delete** | Covered in `ARCHITECTURE_OVERVIEW.md` |
| `PHASE_3_OPTIMIZATION_ENGINE.md` | **Delete** | Same |
| `PHASE_4_INTEGRATION.md` | **Delete** | Same |
| `PHASE_5_TESTING_VALIDATION.md` | **Delete** | Replaced by actual `tests/` directory |
| `PHASE_6_TUNING_RESULTS.md` | **Move → `docs/`** or **delete** | If kept, belongs in `docs/` not root; otherwise delete |
| `PHASE_7_DOCUMENTATION_COMPLETE.md` | **Delete** | Meta-completion marker — serves no ongoing purpose |
| `mycelium_brainstorming_notes.md` | **Delete** | Raw scratch brainstorm notes, nothing architectural |
| `IMPLEMENTATION_REPORT.md` | **Move → `docs/`** | Has lasting reference value but belongs in `docs/`, not root |

---

## Category 5: Runtime Output Folders (likely committed by mistake)

These folders appear to contain runtime-generated output and should not be tracked in git.

| Folder | Action | Reason |
|---|---|---|
| `patch_batches/` | **Add to `.gitignore` + remove from tracking** | Runtime-generated patch batch output |
| `traces/` | **Add to `.gitignore` + remove from tracking** | Runtime trace/debug dumps |
| `reports/` | **Add to `.gitignore`** (if auto-generated) | If reports are generated at runtime they should not be tracked |
| `evaluation_data/` | **Review** | If static checked-in test data → keep; if runtime-generated → gitignore |

---

## Category 6: Keep As-Is (legitimate root-level docs)

These are properly-scoped permanent documentation files. No action needed.

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

## Proposed Final Root Structure

```
Mycelium/
├── [permanent .md docs]          ← the 16 files listed in Category 6
├── [core .py source files]       ← api_models, broadcast_api, run_workflow, etc.
├── scripts/                      ← download_chemistry_datasets.py, download_physics_datasets.py
├── tests/                        ← all test_*.py consolidated here
├── docs/                         ← IMPLEMENTATION_REPORT.md (+ PHASE_6_TUNING_RESULTS.md if kept)
├── core/
├── layer0/
├── mycelium/
├── web-ui/
├── trainers/
├── training_data/
├── signatures/
├── expert_post_check/
└── .gitignore                    ← updated to exclude traces/, patch_batches/, reports/ (if generated)
```

---

## Suggested Staged Push Order

| Stage | Contents |
|---|---|
| **Stage 1** | Delete all `PHASE_*.md` files + `mycelium_brainstorming_notes.md` + `dummy_bert_models/` + `dummy_models/` |
| **Stage 2** | Move `test_*.py` → `tests/`; move `download_*.py` + `physics_datasets_sources.md` → `scripts/`; move `IMPLEMENTATION_REPORT.md` → `docs/` |
| **Stage 3** | Update `.gitignore` to exclude `traces/`, `patch_batches/`, `reports/` (if runtime-generated); remove from tracking with `git rm --cached` |

