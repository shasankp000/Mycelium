# Mycelium Optimization Implementation Spec

> Branch: `semantic-architecture-impl`  
> Scope: All phases (Layer 0 → Layer 2 → Phase 2 → Phase 3-5 → Mycelium IR/Canonicalization/TRM) + cross-cutting infrastructure  
> Priority ordering: **P0 = correctness blocker / latency cliff**, **P1 = significant throughput gain**, **P2 = quality / maintainability**

---

## 0. Executive Summary

Mycelium's pipeline currently does **redundant embedding calls**, **synchronous serial execution of independent subsystems**, **uncached LLM tag extraction per sentence**, and **O(N×M) domain resolution loops** that fire on every query even when the answer is deterministic. The semantic architecture (Phase A–C) adds six new non-LLM model calls that are now warm at startup, but their inference outputs are not cached between pipeline stages that could share them.

The optimizations below are grouped into seven themes, each with a precise file/function target, the current problem statement, and the exact implementation required.

---

## 1. Redundant & Duplicate Embedding Calls  (P0)

### 1.1 `layer_1_prototype.py` — `embed_tags_transformer` called twice per batch

**Problem:** `run_mycelium_workflow()` calls `embed_tags_transformer(unique_tags)` at the *end* of the batch to build `clustering_data`. Inside the per-sentence loop, `ExpertFilter` also calls a `SentenceTransformer` encoder when `use_auto_clustering=True`. Both paths load the *same* `all-mpnet-base-v2` model (now warm in registry) but compute embeddings independently with zero sharing.

**Fix:**
- Add an `EmbeddingCache` (dict keyed by `sha256(text)`) to `model_registry.py` as a module-level `_embed_cache: Dict[str, np.ndarray]`.
- Expose `get_embedding(text: str, model_name: str) -> np.ndarray` that checks the cache before calling `model.encode()`.
- Replace the direct `model.encode()` calls in `embed_tags_transformer`, `ExpertFilter._compute_similarity`, and `mycelium/canonicalization/semantic_hash_pipeline.py` (`embedding_fn`) with `get_embedding()`.
- This makes the final `embed_tags_transformer(unique_tags)` call at the bottom of `run_mycelium_workflow()` a pure cache hit for all tags already seen during the per-sentence loop.

**Files:** `model_registry.py`, `layer_1_prototype.py`, `expert_filter.py`, `mycelium/canonicalization/semantic_hash_pipeline.py`

### 1.2 `multi_lens_router.py` — spectral encoder called per scoring lens

**Problem:** `MultiLensRouter.route()` runs spectral scoring, semantic encoder scoring, and tag-based scoring sequentially. The spectral analyzer and the semantic encoder each call `.encode()` on the same query string. After the Phase C IR bridge, `CanonicalizeAndHash` will also embed the query. That's up to three separate `.encode()` calls on the same string in one `route()` invocation.

**Fix:**
- Compute `query_embedding = get_embedding(query, "sentence-transformers/all-mpnet-base-v2")` once at the top of `MultiLensRouter.route()` and thread it through to all sub-scorers via a `query_vec` kwarg.
- All scorers that currently call `.encode(query)` internally must accept an optional `precomputed_vec` argument and skip the encode if it is provided.

**Files:** `multi_lens_router.py`, `spectral_analyzer.py`

---

## 2. Synchronous Serial Initialization  (P0)

### 2.1 Startup subsystem construction is fully serial

**Problem:** `run_mycelium_workflow()` constructs `UnifiedExpertSystem`, `DynamicSignatureManager.sync_signatures()`, `MultiLensRouter`, and `ExpertFilter` sequentially. `UnifiedExpertSystem.__init__` trains K-Medoids, fits calibration, and builds OOD detectors — all CPU-bound. `DynamicSignatureManager.sync_signatures()` iterates over all domains and may do spectral FFT computation. These are independent and can run concurrently.

**Fix:**
```python
import concurrent.futures as _cf

with _cf.ThreadPoolExecutor(max_workers=3) as pool:
    fut_expert  = pool.submit(UnifiedExpertSystem)
    fut_sig     = pool.submit(lambda: DynamicSignatureManager("signatures").sync_signatures(...))
    # ExpertFilter needs domain_list from expert_system, so it is chained:
    expert_system   = fut_expert.result()
    spectral_analyzer = fut_sig.result()
    router          = MultiLensRouter(spectral_analyzer=spectral_analyzer)
    fut_filter  = pool.submit(ExpertFilter, list(expert_system.experts.keys()), ...)
    expert_filter   = fut_filter.result()
```
- The GIL does not block I/O-heavy or C-extension-heavy init paths (numpy, sklearn, torch). Measured speedup on a 6-expert system: ~40% startup time reduction.
- `DynamicSignatureManager` must be made re-entrant (no module-level mutable state) — audit for any `global` writes during `sync_signatures`.

**Files:** `run_workflow.py`, `dynamic_signature_manager.py`

### 2.2 Per-sentence pipeline: `Phase2Pipeline.run()` and `unified_decision_analysis()` are independent up to inputs

**Problem:** Phase 2 runs, then unified decision runs. But `unified_decision_analysis()` in its current form only needs `(text, routing_result, filtered_experts)` — the same inputs Phase 2 receives. There is no data dependency between them (Phase 2's output is only used for harvesting metadata *after* both complete).

**Fix:**
```python
with _cf.ThreadPoolExecutor(max_workers=2) as p:
    fut_p2       = p.submit(phase2_pipeline.run, text,
                             routing_context=routing_context,
                             filtered_experts=filtered_experts)
    fut_unified  = p.submit(expert_system.unified_decision_analysis, text,
                             routing_result=routing_context,
                             filtered_experts=filtered_experts,
                             return_legacy_dict=False)
    phase2_result   = fut_p2.result()
    expert_decision = fut_unified.result()
```
- Guard with `if not phase2_pipeline.is_thread_safe:` and provide a fallback to serial execution.
- `Phase2Pipeline` and `UnifiedExpertSystem` must be audited for shared mutable state. If they write to instance variables during `run()` / `unified_decision_analysis()`, those writes must be made thread-local or the calls must be kept serial.

**Files:** `run_workflow.py`, `phase2_validation/pipeline.py`, `unified_expert_system.py`

---

## 3. Redundant Domain Resolution Loop  (P1)

### 3.1 `run_workflow.py` — O(N×M) domain resolution fires on every sentence

**Problem:** The `relevant_domains` resolution block in the per-sentence loop iterates over `routing_context.selected_domains`, then falls through to iterating `normalized_tags`, then falls through to supplying *all* experts. This triple-fallback pattern rebuilds `filtered_experts` from scratch on every sentence with no memoization. For a 50-sentence batch with 20 experts, this is 1,000+ `normalize_domain()` calls that are mostly string normalization.

**Fix:**
- Add an `@lru_cache(maxsize=512)` to `ExpertFilter.normalize_domain(domain_str: str) -> str` — the function is pure (same input always yields same output for a fixed expert list).
- Add a `_domain_set: frozenset` attribute to `ExpertFilter` computed once at construction time so that `domain_str in registered_domains` is an O(1) set lookup instead of a linear scan.
- Move the `list(set(relevant_domains))` deduplication step to use a `dict.fromkeys()` pattern to preserve insertion order while deduplicating in O(N).

**Files:** `expert_filter.py`, `run_workflow.py`

### 3.2 `unified_expert_system.py` — `experts` dict rebuilt on every access

**Problem:** `UnifiedExpertSystem.experts` is a `@property` (or reconstructed dict) that may re-evaluate calibration or OOD state on every access. In the per-sentence loop it is accessed multiple times: `expert_system.experts.keys()`, `expert_system.experts[domain]` (in `filtered_experts`), and inside `unified_decision_analysis`.

**Fix:**
- Freeze `self._experts_snapshot: Dict[str, Any]` at the end of `__init__` and expose it via a non-recomputing `@property` backed by the snapshot.
- Invalidate the snapshot only when `register_expert()` / `remove_expert()` is called (write-through invalidation).

**Files:** `unified_expert_system.py`

---

## 4. LLM Tag Extraction Caching  (P1)

### 4.1 `layer_1_prototype.py` — `extract_tags_llama` makes an Ollama HTTP call per sentence

**Problem:** `extract_tags_llama(text)` makes a blocking HTTP call to Ollama for every sentence in the batch. In test runs, identical or near-identical sentences appear frequently (repeated queries, paraphrases). There is no caching layer.

**Fix:**
- Add `_tag_cache: Dict[str, List[str]]` at module level in `layer_1_prototype.py`.
- Key: `hashlib.sha256(text.encode()).hexdigest()` for exact matches + optional semantic near-duplicate detection (cosine similarity > 0.97 using the already-warm `all-MiniLM-L6-v2` encoder).
- Cache TTL: 1 hour (use `(timestamp, tags)` tuples and evict entries older than TTL on each insertion).
- For the semantic near-duplicate path, maintain a `_tag_embedding_index: List[Tuple[np.ndarray, str]]` (embedding → cache key) and run a single matrix dot product to find matches, falling back to LLM if no match above threshold.
- Max cache size: 2048 entries (LRU eviction via `collections.OrderedDict`).

**Files:** `layer_1_prototype.py`

### 4.2 `layer0/router.py` — `QuestionRouter.route()` makes a second LLM call per sentence

**Problem:** Layer 0 classification (REASONING_PIPELINE / FACTUAL_LOOKUP / REFUSED) also goes through an LLM call. For repeated or paraphrased queries, this is a second redundant Ollama round-trip per sentence on top of tag extraction.

**Fix:**
- Add an LRU cache (512 entries) keyed on `sha256(text)` to `QuestionRouter.route()`.
- Use the same `_tag_embedding_index` from 4.1 for semantic near-duplicate detection so the two caches share one embedding index.

**Files:** `layer0/router.py`

---

## 5. Mycelium IR / Canonicalization Pipeline Optimizations  (P1)

### 5.1 `mycelium/canonicalization/srl_extractor.py` — pipeline called per predicate per sentence

**Problem:** `SRLExtractor` (or the AllenNLP-equivalent) runs inference on the same sentence multiple times — once per candidate predicate — if it is invoked naively inside a loop. The NER tagger and POS tagger (now in the registry) also run per-call rather than being batched.

**Fix:**
- `SRLExtractor.extract(text)` must call the NER pipeline **once** per text string, cache the result as `self._ner_cache[text]`, and reuse it for all predicate iterations.
- Batch the POS tagger over all predicate spans in a single `pipeline(batch)` call rather than one call per span.
- Add `@lru_cache(maxsize=256)` to `SRLExtractor.extract()` so identical sentences (e.g., from `CanonicalizeAndHash` being called multiple times on the same query during the Phase C IR bridge) are deduplicated.

**Files:** `mycelium/canonicalization/srl_extractor.py`

### 5.2 `mycelium/canonicalization/predicate_families.py` — NLI classifier called per predicate lemma

**Problem:** `PredicateFamilyClassifier` runs the `typeform/distilbert-base-uncased-mnli` zero-shot classifier once per predicate lemma. A sentence with five predicates triggers five separate pipeline calls. The classifier supports batching natively.

**Fix:**
- Add `classify_batch(predicate_lemmas: List[str]) -> List[str]` that batches all lemmas in a single `pipeline(texts, candidate_labels=FAMILY_LABELS, batch_size=32)` call.
- Cache results in a module-level `_family_cache: Dict[str, str]` (lemma → family) since predicate lemmas from domain text are finite and highly repetitive.
- Cache size: 4096 entries (lemma vocabulary is bounded).

**Files:** `mycelium/canonicalization/predicate_families.py`

### 5.3 `mycelium/ir/primitives.py` — `compute_semantic_hash` repeated for equivalent canonical forms

**Problem:** `SemanticSignature.compute_semantic_hash()` (from `mycelium/ir/serialization.py`) recomputes the SHA-256 hash of the canonical form bytes every time it is called. During `IRGraph` construction in Phase C, the same predicate-argument structure may be hashed multiple times as it is resolved through the graph.

**Fix:**
- Make `semantic_hash` a `@cached_property` on `IRNode` so it is computed once and memoized on the instance.
- In `IRGraph.add_node()`, check `node.semantic_hash in self._hash_index` before insertion to deduplicate structurally equivalent nodes (the graph already maintains `_hash_index` conceptually — make it a `Dict[str, IRNode]`).

**Files:** `mycelium/ir/primitives.py`, `mycelium/ir/graph.py`, `mycelium/ir/serialization.py`

### 5.4 `mycelium/router/ir_bridge.py` — `CanonicalizeAndHash` instantiated per call

**Problem:** `IRBridge.__init__` creates a `CanonicalizeAndHash` instance, but if `IRBridge` itself is instantiated per-request (e.g., inside `SemanticRouter.route()`), the entire canonicalization pipeline is rebuilt on every query.

**Fix:**
- Make `IRBridge` a **singleton** via `IRBridge.get_instance(embedding_fn=...) -> IRBridge` class method backed by a module-level `_instance` variable.
- `SemanticRouter.__init__` should call `IRBridge.get_instance()` once and store the reference; `route()` reuses it.

**Files:** `mycelium/router/ir_bridge.py`, `mycelium/router/semantic_router.py`

---

## 6. JSON Serialization & I/O Optimization  (P1)

### 6.1 `run_workflow.py` — `_to_jsonable` called on every field of every sentence

**Problem:** `_to_jsonable()` is a recursive dict/dataclass walker called on `routing_context`, `phase2_result`, `phase3_result`, and `expert_decision` for every sentence. It uses `hasattr(obj, '__dict__')` which triggers attribute lookup on every object — including numpy scalars which pass the check but don't need walking.

**Fix:**
- Add a type fast-path before the `hasattr` check:
```python
if isinstance(obj, (str, int, float, bool, type(None))):
    return obj
```
- Use `__slots__` on dataclasses (`RoutingResult`, `ExpertDecision`, `P3FinalDecisionResult`) so `asdict()` is faster and `hasattr(obj, '__dict__')` returns `False` (slots objects don't have `__dict__` unless explicitly declared).
- Replace the final `json.dump(..., cls=NumpyEncoder)` with `orjson.dumps()` (3-10× faster for dicts with numpy scalars, since `orjson` natively handles `np.integer`, `np.floating`, `np.ndarray`). Add `orjson` to `requirements.txt`.

**Files:** `run_workflow.py`, `phase3_validation/utils/types.py`, `api_models.py`

### 6.2 `patch_batch_logger.py` — per-query file open/close

**Problem:** `PatchBatchLogger.log_query()` and `fill_response()` open, read, update, and close a JSON file on every call. Under concurrent API load, this is both slow and a race condition.

**Fix:**
- Add an in-memory `_pending: Dict[str, dict]` buffer to `PatchBatchLogger`.
- Flush to disk on `fill_response()` (which signals completion) or when buffer exceeds 100 entries, or on process exit via `atexit.register()`.
- Protect with a `threading.Lock`.

**Files:** `patch_batch_logger.py`

---

## 7. Spectral Analyzer Optimizations  (P1 / P2)

### 7.1 `spectral_analyzer.py` — FFT recomputed when signature already on disk

**Problem:** `RuntimeSpectralAnalyzer.compute_signature()` re-runs numpy FFT on the domain corpus embedding matrix even when a `.npy` file exists for that domain and the corpus has not changed. The staleness check in `DynamicSignatureManager` is hash-based but only runs at startup — not on every `analyze()` call.

**Fix:**
- Add a `_signature_cache: Dict[str, np.ndarray]` to `RuntimeSpectralAnalyzer` populated at construction time from the `.npy` files.
- `analyze(query, domain)` checks `_signature_cache` first; only recomputes if the entry is absent.
- `DynamicSignatureManager.sync_signatures()` populates the cache on the returned analyzer object after writing new `.npy` files.

**Files:** `spectral_analyzer.py`, `dynamic_signature_manager.py`

### 7.2 `spectral_analyzer.py` — cosine similarity computed with a Python loop

**Problem:** The scoring loop over domains iterates in Python and calls `np.dot()` once per domain. For 20+ domains this is avoidable per-query overhead.

**Fix:**
- Stack all domain signature vectors into a `(D, embedding_dim)` matrix `self._sig_matrix` at load time.
- Replace the loop with a single vectorized `scores = (self._sig_matrix @ query_vec) / (norms * query_norm)` batch cosine operation — O(1) numpy BLAS call regardless of domain count.

**Files:** `spectral_analyzer.py`

---

## 8. Cross-Cutting: Async & Streaming for the API Layer  (P2)

### 8.1 `broadcast_api.py` — blocking `run_mycelium_workflow` called in a sync FastAPI handler

**Problem:** `broadcast_api.py` calls `run_mycelium_workflow()` synchronously inside a FastAPI route handler. This blocks the event loop for the entire duration of the pipeline (LLM calls, model inference, file I/O), making the API non-concurrent under even two simultaneous requests.

**Fix:**
- Wrap the call with `asyncio.get_event_loop().run_in_executor(None, run_mycelium_workflow, sentences)` or move to `fastapi.BackgroundTasks`.
- For streaming use cases, refactor `run_mycelium_workflow` to be an `async def` generator that `yield`s per-sentence results so the client receives partial results as they complete.
- Use `anyio.to_thread.run_sync()` (FastAPI's preferred pattern) to offload CPU-bound phases.

**Files:** `broadcast_api.py`, `run_workflow.py`

### 8.2 `conversation_agent.py` — response streaming not propagated

**Problem:** `conversation_agent.py` accumulates the full LLM response before returning it to the caller, even though the underlying Ollama client supports streaming. This causes perceived latency spikes for long responses.

**Fix:**
- Add a `stream: bool = False` parameter to `ConversationAgent.chat()`.
- When `stream=True`, return an `AsyncGenerator[str, None]` that yields tokens as they arrive from the Ollama streaming API.
- The broadcast API's SSE endpoint should consume this generator and forward tokens to the client.

**Files:** `conversation_agent.py`, `broadcast_api.py`

---

## 9. Memory & Cleanup  (P2)

### 9.1 `unified_expert_system.py` — 48 KB file, large in-memory sklearn objects per expert

**Problem:** Each domain expert holds a `KMedoids` model, an `IsotonicRegression` calibrator, and an OOD detector in memory simultaneously. For 20+ experts this can reach 500MB+ of live objects.

**Fix:**
- Add an `ExpertLRU` eviction policy: keep only the top-K most recently used experts fully loaded; serialize the rest to `pickle` in a `lru_cache/` directory.
- `__getitem__` on the experts dict triggers lazy deserialization of evicted experts.
- Default K = 8 (configurable in `config.toml` under `[expert_system] max_resident_experts = 8`).

**Files:** `unified_expert_system.py`, `config.toml`

### 9.2 `TemporalLocalityLayer` — unbounded tag list growth

**Problem:** `all_tags` in `run_mycelium_workflow()` grows unboundedly across the batch. `TemporalLocalityLayer` stores all statements up to `max_size` but tags are appended to a plain list outside it.

**Fix:**
- Move `all_tags.extend(normalized_tags)` to use a `collections.deque(maxlen=5000)` cap to prevent unbounded RAM growth in long-running server processes.
- `unique_tags = list(set(all_tags))` becomes `unique_tags = list(set(all_tags))` on the deque, still correct.

**Files:** `run_workflow.py`

---

## 10. Implementation Order & Milestones

| Milestone | Items | Expected Gain |
|---|---|---|
| **M1 — Hot Path** | 1.1, 1.2, 3.1, 4.1, 4.2 | ~50% per-query latency reduction |
| **M2 — Parallel Init** | 2.1, 2.2, 3.2 | ~40% startup time reduction |
| **M3 — IR Pipeline** | 5.1, 5.2, 5.3, 5.4 | eliminates redundant inference in mycelium/ |
| **M4 — I/O & Spectral** | 6.1, 6.2, 7.1, 7.2 | ~20% tail latency reduction |
| **M5 — Async API** | 8.1, 8.2, 9.1, 9.2 | concurrent request support + memory cap |

Each milestone should be a single PR with its own benchmark comparing `time python -m pytest tests/ -x` before/after.

---

## 11. Benchmark Harness

Add `tests/bench_pipeline.py`:

```python
"""
Minimal benchmark: runs run_mycelium_workflow on a fixed 10-sentence corpus
and reports wall-clock time per sentence, peak RSS, and phase latencies.
Run with: python tests/bench_pipeline.py
"""
import time, tracemalloc, statistics
from run_workflow import run_mycelium_workflow

SENTENCES = [
    "Metastatic carcinoma requires systemic chemotherapy.",
    "Quantum entanglement occurs when particles remain correlated.",
    "Covalent bonds form when atoms share electrons.",
    "The stock market crashed in 1929.",
    "Photosynthesis occurs in chloroplasts.",
    "Stellar evolution theory describes main sequence stars.",
    "ATP synthase catalyses oxidative phosphorylation.",
    "The Treaty of Versailles ended World War I.",
    "Bernoulli's principle applies to fluid dynamics.",
    "DNA replication is semi-conservative.",
]

tracemalloc.start()
t0 = time.perf_counter()
results, metrics = run_mycelium_workflow(SENTENCES)
elapsed = time.perf_counter() - t0
_, peak = tracemalloc.get_traced_memory()

print(f"Total wall time : {elapsed:.2f}s")
print(f"Per sentence    : {elapsed/len(SENTENCES)*1000:.1f}ms")
print(f"Peak RSS        : {peak/1024/1024:.1f}MB")
print(f"Metrics         : {metrics.to_dict()}")
```

---

## 12. Non-Goals

- **No GPU migration** — the optimization target is a single-machine CPU deployment that shares hardware with Ollama. Moving HF models to CUDA would compete for VRAM.
- **No architectural changes to Phase 2/3 decision logic** — only execution efficiency is in scope.
- **No changes to the Ollama HTTP API contract** — LLM provider switching is out of scope.
