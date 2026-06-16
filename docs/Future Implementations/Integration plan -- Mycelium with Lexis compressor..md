# Mycelium x Lexis Integration Plan

## Overview

This document describes the full integration strategy for embedding Lexis-E's linguistically-structured compression pipeline into Mycelium's expert routing architecture. It covers two interlinked proposals: (1) using Lexis as the backing store and extraction engine for a new structured patch format, and (2) extending the patch system into a traversable knowledge DAG rather than a flat list of model files.

The core thesis: a **CREATE_NEW_PATCH** signal should not produce a new sub-transformer model. Instead, it should produce a **compressed, structured, queryable knowledge node** -- a patch that lives in a DAG, stores its content as a `.lexi` binary, exposes its linguistic structure as queryable metadata, and extends the existing k-NN routing space without any retraining.

---

## Background: What Each System Contributes

### Mycelium (current state)

- `UnifiedExpertSystem` routes queries across domain experts (music SVM, physics/chemistry/medical BERT-based)
- OOD detection: three-method ensemble (SVM decision distance, k-NN distance, Isolation Forest) with majority-vote
- `CREATE_NEW_PATCH` fires when all experts report OOD -- currently a terminal routing signal with no materialization
- Expert centroids persisted as `.pkl` files; calibration run once and cached
- MiniLM (`all-MiniLM-L6-v2`) embeddings used for centroid similarity throughout

### Lexis-E (current state)

https://github.com/shasankp000/Lexis

- 12-stage linguistically-structured text compressor; achieves 2.7523 bpb on FineWeb with zero learned weights
- Produces `.lexi` binary files containing both the arithmetic-coded character stream and structured metadata layers (POS sequences, morphological codes, coreference chains, case flags, symbol tables)
- `compact_mode` reduces full-payload overhead by ~47–52% vs the main branch
- Stage 4 (Longformer coreference) produces an entity graph as a byproduct of compression
- Requires Python 3.11.x (spaCy compatibility constraint); runs in an isolated `.venv2` alongside Mycelium

---

## Part 1 -- Lexis as Context Condenser (Immediate, Low-Risk)

Before the full patch system is built, Lexis's upstream linguistic stages can be applied to conversation history to extend Mycelium's effective context window.

### Mechanism

Stages 1–4 of the Lexis pipeline produce linguistically cleaned, coreference-resolved text -- shorter than the raw input, with pronoun chains collapsed and sentence boundaries normalized. This text is still LLM-readable; it has not been arithmetically encoded. Running this pass on multi-turn conversation history before building the LLM prompt yields 15–30% token reduction on domain-heavy conversations without any information loss.

### Implementation

```python
# lexis_condenser.py
# Extract and adapt from Lexis pipeline_trace.py Stages 1–4

from lexis.stage1_segmentation import segment_sentences
from lexis.stage3_pos import annotate_pos
from lexis.stage4_discourse import resolve_coreferences

def condense_history(turns: list[dict]) -> str:
    """
    Applies Lexis Stage 1 + 3 + 4 to conversation history.
    Returns coreference-resolved, sentence-normalized text.
    Suitable for direct inclusion in LLM prompt context window.
    """
    raw = "\n".join(f"{t['role']}: {t['content']}" for t in turns)
    segmented = segment_sentences(raw)
    pos_annotated = annotate_pos(segmented)          # metadata only, not injected into text
    condensed = resolve_coreferences(pos_annotated)  # collapses pronoun chains
    return condensed
```

Stages 5–12 (arithmetic coding, factoriadic encoding, compact_mode binary packing) are **not used** in this path -- they produce binary output unsuitable for LLM input.

---

## Part 2 -- Structured Patch Architecture

### 2.1 What a Patch Should Be

A patch is not a sub-transformer. Training a new model for each novel domain is prohibitively expensive, produces opaque representations, and creates a combinatorial composition problem (how do two sub-transformers interact on a cross-domain query?).

A patch should instead be a **structured knowledge node** with four components:

| Component | Format | Purpose |
|---|---|---|
| `centroid_embedding` | `numpy` array (384-dim MiniLM) | Routing: extends k-NN space so future similar queries route here |
| `lexi_path` | Path to `.lexi` binary | Content: decompressed on query to inject into LLM context (RAG-style) |
| `pos_fingerprint` | `numpy` array (POS distribution histogram) | Structural similarity: identifies syntactically similar patches |
| `entity_graph` | Adjacency dict (term → co-referring terms) | Cross-patch relationships: Stage 4 coreference output |

The patch node is stored as a JSON sidecar alongside the `.lexi` file:

```json
{
  "patch_id": "qm-patch-001",
  "created_at": "2026-05-17T19:00:00Z",
  "domain_label": "quantum_mechanics",
  "centroid_embedding": "<path-to-.npy>",
  "lexi_path": "patches/qm-patch-001.lexi",
  "pos_fingerprint": "<path-to-.npy>",
  "entity_graph": "<path-to-.json>",
  "parent_domains": ["physics", "quantum"],
  "overlap_edges": {
    "chemistry/quantum-chem": 0.34,
    "physics/classical": 0.12
  },
  "confidence_threshold": 0.72,
  "novel_chunk_count": 14,
  "source_document": "input_doc.txt"
}
```

### 2.2 Patch Creation Pipeline

When `CREATE_NEW_PATCH` fires on a large input document, the following pipeline runs as an offline background job (never blocking query serving):

```
Large novel document
        │
        ▼
┌─────────────────────────────────┐
│  Step 1: Chunk + OOD Filter     │
│                                 │
│  Split into 512-token passages  │
│  (20% sliding overlap)          │
│                                 │
│  Per chunk, run UnifiedExpert:  │
│  • KNOWN + high conf → discard  │
│  • KNOWN + low conf  → boundary │
│  • OOD               → NOVEL ✓  │
└──────────────┬──────────────────┘
               │ novel chunks only
               ▼
┌─────────────────────────────────┐
│  Step 2: Lexis-E Full Pipeline  │
│                                 │
│  Stage 1–2: Sentence seg +      │
│             morphological roots │
│  Stage 3:   POS annotation      │
│  Stage 4:   Coreference chains  │
│             → entity graph      │
│  Stage 5:   Case flags          │
│  Stage 6–7: Arithmetic coding   │
│             → .lexi binary      │
└──────────────┬──────────────────┘
               │ .lexi + metadata
               ▼
┌─────────────────────────────────┐
│  Step 3: Embedding + Fingerprint│
│                                 │
│  MiniLM encode novel chunks     │
│  → centroid_embedding           │
│                                 │
│  POS histogram over all novel   │
│  chunks → pos_fingerprint       │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Step 4: DAG Placement          │
│                                 │
│  Find nearest existing patch    │
│  nodes by cosine similarity     │
│  → assign parent_domains        │
│                                 │
│  Compute overlap_edges to all   │
│  nodes with similarity > 0.25   │
│                                 │
│  Write patch node JSON          │
│  Insert centroid into k-NN index│
└─────────────────────────────────┘
```

### 2.3 The Patch DAG

Patches are stored as nodes in a directed acyclic graph. The graph is serialized as a JSON adjacency list and loaded into memory at startup alongside the existing expert centroids.

```
ROOT
├── physics
│   ├── quantum
│   │   ├── [patch] qm-001 (.lexi)
│   │   └── [patch] qft-002 (.lexi)
│   └── classical
│       └── [patch] thermo-001 (.lexi)
├── chemistry
│   ├── organic
│   └── quantum-chem ←─ overlap edge ─→ qm-001 (0.34)
└── medical
    └── pharmacology
        └── [patch] drug-mech-001 (.lexi)
```

**Node types:**

| Type | Description |
|---|---|
| Domain node | A trained expert (SVM or BERT-based); exists today |
| Patch node | A Lexis-backed knowledge node; new |
| Boundary node | A domain node with low-confidence coverage; flagged for patch creation |

**Edge types:**

| Type | Direction | Meaning |
|---|---|---|
| `is-a` | Child → Parent | Patch inherits domain context from parent |
| `overlap` | Bidirectional | Two patches share entity graph vocabulary above threshold |
| `spawned-from` | Patch → Domain | Which expert's OOD signal triggered this patch |

### 2.4 Query-Time Routing with Patch DAG

The modified routing logic in `UnifiedExpertSystem`:

```python
def route_with_patches(self, query: str) -> RoutingResult:
    # Existing expert routing
    result = self.route(query)

    if result.action == "CREATE_NEW_PATCH":
        # No patch exists yet -- return signal to caller
        # (patch creation is an async offline job, not query-time)
        return result

    elif result.action == "USE_EXPERT":
        # Existing path -- unchanged
        return result

    elif result.action == "USE_PATCH":
        # NEW: retrieve from patch DAG
        patch_node = self.patch_dag.nearest(
            query_embedding=self.encode(query),
            threshold=patch_node.confidence_threshold
        )
        # Decompress relevant spans from .lexi via bridge
        context_text = lexi_decompress(
            patch_node.lexi_path,
            max_chars=2048  # fit within context budget; sliced in bridge
        )
        return RoutingResult(
            action="USE_PATCH",
            patch_id=patch_node.patch_id,
            injected_context=context_text,
            entity_graph=patch_node.entity_graph
        )
```

The `injected_context` is prepended to the LLM prompt as retrieved knowledge -- semantically equivalent to RAG retrieval, but sourced from a Lexis-compressed store rather than a vector database.

---

## Part 3 -- Why Not a Sub-Transformer?

| Criterion | Sub-transformer patch | Lexis-backed patch node |
|---|---|---|
| Creation cost | Full fine-tuning run (GPU hours) | Lexis pipeline run (CPU, ~9 min / 100k chars) |
| Storage | 100M–1B parameters per patch | ~11 bpb × input size (compact_mode) |
| Routing | Requires a meta-router over N models | k-NN over centroid embeddings (already built) |
| Cross-domain composition | Undefined -- two models cannot be trivially merged | Overlap edges in DAG; entity graphs unioned |
| Interpretability | Black-box weights | POS fingerprint, entity graph, coreference chains all inspectable |
| Update / correction | Full re-fine-tune | Append new chunks, re-run Lexis pipeline, update centroid |
| Context injection | N/A | Decompress `.lexi` spans → prepend to prompt |

The sub-transformer path only becomes justified when a patch accumulates enough queries in a stable domain that the RAG-style injection latency becomes a bottleneck -- at which point the patch node's accumulated query log becomes the fine-tuning dataset for a proper expert. The patch DAG thus naturally stages domain maturation: novel domain → patch node → full expert, with the patch node as the intermediate state.

---

## Part 4 -- Venv Isolation Architecture

Mycelium and Lexis run in **separate, fully isolated virtual environments**. Mycelium's dependency tree (Python 3.14) is never modified. Lexis's dependency tree (Python 3.11, required by spaCy) lives in `.venv2` alongside the Mycelium project root.

```
mycelium/
├── .venv/          ← Mycelium (Python 3.14) -- untouched
├── .venv2/         ← Lexis-E (Python 3.11) -- renamed from Lexis project venv
├── lexis/          ← Lexis source tree (copied or symlinked)
├── lexis_bridge.py ← subprocess bridge (runs in .venv, shells out to .venv2)
└── ...
```

All Mycelium code calls Lexis exclusively through `lexis_bridge.py`, which shells out to `.venv2/bin/python`. No Lexis module is ever imported directly into Mycelium's process.

### Why not a shared venv?

| Concern | Shared venv | Isolated .venv2 |
|---|---|---|
| spaCy requires Python 3.11 | Forces Mycelium to 3.11 | No constraint on Mycelium |
| `fastcoref` dependency pins | May conflict with Mycelium's `transformers` | Fully isolated |
| `huggingface-hub` upper-bound patch | Must patch Mycelium's transformers | Only patches `.venv2` |
| Breakage risk | High -- one `pip install` can break both | Zero -- envs never interact |

### Lexis-E requirements (inside `.venv2` only)

```
python3.11
spacy>=3.7.0
en_core_web_sm   (python -m spacy download en_core_web_sm)
en_core_web_lg   (python -m spacy download en_core_web_lg)
fastcoref        (Stage 4 coreference)
lemminflect      (morphological inflection)
torch            (GPU used by Stage 3 + Stage 4 if available)
transformers     (patch huggingface-hub upper bound in dependency_versions_table.py)
zstandard        (optional -- LXZ1 zstd wrapping for smaller .lexi files)
cupy-cuda12x     (optional -- only if CUDA available)
```

> **Note:** `msgpack` is no longer required. Lexis-E now uses its own LEXI binary envelope
> format (`encode_metadata` / `decode_metadata`) for all new files. Legacy msgpack files
> are still readable (read-only fallback in `decompress()`), but never written.

---

## Part 5 -- `lexis_bridge.py` (Corrected)

The Lexis CLI uses **subcommands** (`compress`, `decompress`, `analyse`), not flags like
`--stdin` or `--max-chars`. All interaction goes through file paths; `decompress` prints
to stdout only. The bridge handles temp-file creation for compression input and applies
`max_chars` slicing post-stdout capture.

### Actual CLI reference

| Subcommand | Signature | Notes |
|---|---|---|
| `compress` | `compress <input_file> <output_file> [--model] [--compact-context] [--compact-profile default\|aggressive] [--compact-top-k N] [--compact-scale N]` | Writes LEXI binary envelope to `output_file` |
| `decompress` | `decompress <input_file>` | Prints reconstructed text to stdout; no truncation flag |
| `analyse` | `analyse <input_file> [--model]` | Prints per-stage stats to stdout |

### Implementation

```python
# lexis_bridge.py
import subprocess
import tempfile
from pathlib import Path

LEXIS_PYTHON = Path(__file__).parent / ".venv2/bin/python"
LEXIS_MAIN   = Path(__file__).parent / "lexis/main.py"


def lexi_compress(
    text: str,
    output_path: str,
    compact: bool = True,
    profile: str = "default",
) -> None:
    """
    Write text to a temp file, compress to output_path as .lexi binary.

    Uses --compact-context by default (profile='default' = k6s511).
    Temp file is always cleaned up regardless of success/failure.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(text)
        tmp_path = tmp.name

    cmd = [str(LEXIS_PYTHON), str(LEXIS_MAIN), "compress", tmp_path, output_path]
    if compact:
        cmd += ["--compact-context", "--compact-profile", profile]

    try:
        result = subprocess.run(cmd, capture_output=True)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(f"Lexis compress failed: {result.stderr.decode()}")


def lexi_decompress(lexi_path: str, max_chars: int | None = None) -> str:
    """
    Decompress a .lexi file; optionally truncate to max_chars.

    Truncation is applied after stdout capture -- there is no --max-chars flag
    in the Lexis CLI. Pass max_chars=2048 for LLM context injection use cases.
    """
    result = subprocess.run(
        [str(LEXIS_PYTHON), str(LEXIS_MAIN), "decompress", lexi_path],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Lexis decompress failed: {result.stderr.decode()}")

    text = result.stdout.decode("utf-8")
    return text[:max_chars] if max_chars else text


def lexi_analyse(text_path: str) -> str:
    """
    Run analyse mode on a text file; return stdout stats as a string.

    Useful for debugging patch creation -- shows per-stage bpb, POS Huffman
    summary, and context-mixing model stats.
    """
    result = subprocess.run(
        [str(LEXIS_PYTHON), str(LEXIS_MAIN), "analyse", text_path],
        capture_output=True,
    )
    return result.stdout.decode("utf-8")
```

---

## Part 6 -- Implementation Phases

### Phase 1 -- Foundation (prerequisite)
- [ ] Rename Lexis's existing Python 3.11 venv to `.venv2` inside the Mycelium project root -- Mycelium's existing venv (Python 3.14) remains `.venv` and is untouched
- [ ] Verify `.venv2` smoke-test: `source .venv2/bin/activate && python pipeline_trace.py` -- all 12 stages green
- [ ] Copy or symlink the Lexis source tree into `mycelium/lexis/`
- [ ] Add `lexis_bridge.py` to Mycelium root (see Part 5)
- [ ] Smoke-test the bridge from Mycelium's `.venv`:
  ```bash
  source .venv/bin/activate
  python -c "from lexis_bridge import lexi_compress, lexi_decompress; \
             lexi_compress('hello world', '/tmp/test.lexi'); \
             print(lexi_decompress('/tmp/test.lexi'))"
  ```

### Phase 2 -- Context Condenser
- [ ] Extract Stages 1–4 from Lexis `main.py` / `pipeline_trace.py` into `lexis_condenser.py` (called via bridge, not direct import)
- [ ] Wire `condense_history()` into Mycelium's context-building path (pre-LLM call)
- [ ] Measure token reduction on representative conversation histories

### Phase 3 -- Patch Node Format
- [ ] Define `PatchNode` dataclass (fields from §2.1)
- [ ] Implement `PatchCreationPipeline` (§2.2) as a background `asyncio` task triggered by `CREATE_NEW_PATCH`
- [ ] Implement `PatchDAG` with JSON serialization and k-NN index insertion
- [ ] Add `USE_PATCH` action to `RoutingResult` and `UnifiedExpertSystem.route_with_patches()`

### Phase 4 -- Lexis Backend
- [ ] `lexis_bridge.py` `lexi_compress()` and `lexi_decompress()` are the only call sites -- no direct Lexis imports anywhere in Mycelium
- [ ] Store `.lexi` files under `patches/` directory alongside sidecar JSON
- [ ] Implement patch retrieval: call `lexi_decompress(patch_node.lexi_path, max_chars=2048)` via bridge, inject into LLM prompt

### Phase 5 -- DAG Traversal + Cross-Patch Relationships
- [ ] Implement overlap edge computation (cosine similarity of POS fingerprints + entity graph Jaccard)
- [ ] Implement DAG traversal for multi-hop cross-domain queries
- [ ] Add patch visualization endpoint to the web UI (renders the DAG as a collapsible tree)

### Phase 6 -- Hardening
- [ ] Patch creation idempotency (same document does not create duplicate nodes)
- [ ] Patch expiry / stale detection (entity graph no longer referenced)
- [ ] Promotion path: patch node → full expert when query volume crosses threshold

---

## Open Questions

1. **Chunk boundary strategy** -- 512-token sliding window is a reasonable default, but domain boundaries rarely align with fixed-size chunks. A sentence-aware chunker (Lexis Stage 1 output) would produce cleaner novel-span isolation.

2. **Patch confidence threshold** -- the threshold at which a patch node activates vs. deferring to a parent domain expert needs empirical calibration on Mycelium's query distribution.

3. **Entity graph cross-patch unioning** -- when two patches share high entity graph overlap, should they be merged into a single node or kept separate with a strong overlap edge? Merging risks losing the provenance of each source document; separate nodes with edges preserves it.

4. **Longformer GPU memory** -- Stage 4 coreference (Longformer, 90.5M params) and Mycelium's BERT-based experts will contend for GPU VRAM during patch creation. Patch creation should be scheduled when the server is idle, or run on CPU with a flag.

5. **`decompress` stdout buffering** -- for very large `.lexi` files, `subprocess.run(..., capture_output=True)` buffers the entire decompressed output in memory before `max_chars` slicing. If patches grow large, switch to `subprocess.Popen` with streaming stdout and read only the first `max_chars` bytes directly.
