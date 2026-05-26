# Mycelium — Architecture Diagram

> **WIP — updated in stages.** See `.diagram_wip.md` for expansion plan.  
> Stage 3: `multi_lens_router.py` internals expanded (spectral analysis, fusion engine, budget-select, classification).

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  CLIENT (Next.js / web-ui)                                                                         │
│                                                                                                    │
│   ReasoningGraph/index.tsx                                                                         │
│   └─ EventSource  POST /api/v1/chat/stream  ──────────────────────────────────────────────────┬───┐│
│               GET  /api/v1/chat/stream  (legacy shim)                        SSE stream        │   ││
│              POST  /api/v1/chat          (blocking JSON)                                       │   ││
└────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                                                                │   │
┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  FASTAPI BACKEND  (broadcast_api.py)                                                               │
│                                                                                                    │
│  Startup                                                                                           │
│  └─ preload_models()                                                                               │
│      └─ model_registry.warmup([mpnet, MiniLM])  ── _warmup_done.set()                              │
│      └─ _write_calibration_state()  →  runtime/calibration_state.json                              │
│                                                                                                    │
│  POST /api/v1/chat/stream  (primary SSE)                                               ◄───┘       │
│  GET  /api/v1/chat/stream  (legacy GET shim)                                                       │
│  POST /api/v1/chat         (blocking, returns ChatResponse)                                        │
│  POST /api/v1/query        (legacy, returns MyceliumRunSummary only)                               │
│  GET  /health  /api/v1/health                                                                      │
│  GET  /api/calibrate/ready                                                                         │
│  GET  /api/v1/traces/recent  GET /api/v1/traces/{trace_id}                                         │
│                                                                                                    │
│  _make_sse_response()                                                                              │
│  ├─ ReplayJournal (maxlen=200)  ←─ reconnect replay via Last-Event-ID header                       │
│  └─ ThreadPoolExecutor(1)  →  _producer()                                                          │
│       └─ _full_pipeline_generator()                                                                │
│            ├─ _build_run_summary()  ──────────────────────────────────────────────────────────────┐│
│            ├─ _run_sandbox()                                                                       ││
│            ├─ ConversationAgent.answer()                                                           ││
│            ├─ append_trace()  →  traces/*.jsonl                                                    ││
│            └─ patch_logger.log_query() / fill_response()  (patch queries)                         ││
│                                                                                                    │
│  asyncio.Queue ──────────────────────── pipeline events (SSE) ─────────────────────────────────────┘
│  (loop.call_soon_threadsafe)  →  StreamingResponse(_async_gen)                                     │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                            |
                               _build_run_summary()
                               └─ run_mycelium_workflow()  [run_workflow.py]
                                    |
┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  PIPELINE ORCHESTRATOR  (run_workflow.py :: run_mycelium_workflow)                                  │
│                                                                                                    │
│  INIT (once per request)                                                                           │
│  ├─ api_models.get_depth_config(reasoning_mode)                                                    │
│  ├─ EventEmitter / make_emitter()  →  pipeline_event.py                                            │
│  ├─ model_registry.warmup(STARTUP_SPECS)                                                           │
│  ├─ Phase2Pipeline()                                                                               │
│  ├─ Phase3To5Pipeline()                                                                            │
│  ├─ GraphStore(persistence_path)                                                                   │
│  ├─ MultiWorkerDFSLookup(graph_store)                                                              │
│  ├─ TRMEngine(store, dfs)                                                                          │
│  ├─ TRMLens()                                                                                      │
│  ├─ DAGDecomposer(graph_store, max_depth)                                                          │
│  ├─ TRMReasoner (optional, loaded from trm_latest.pt checkpoint)                                   │
│  ├─ CanonicalizeAndHash  (optional)                                                                │
│  ├─ DSTFusion  (optional)                                                                          │
│  ├─ ContradictionClassifier  (optional)                                                            │
│  ├─ get_unified_expert_system()  →  UnifiedExpertSystem                                            │
│  ├─ DynamicSignatureManager.sync_signatures()  →  SpectralAnalyzer                                │
│  ├─ MultiLensRouter(spectral_analyzer)                                                             │
│  ├─ TRMOODFallback (optional, wraps TRMOODHead + MultiLensRouter)                                 │
│  ├─ ExpertFilter(domain_list, use_auto_clustering=True)                                            │
│  └─ QuestionRouter()  [layer0/router.py]                                                           │
│                                                                                                    │
│  PER-SENTENCE LOOP  (╳ each sentence in the batch)                                                 │
│  │                                                                                                 │
│  ├── 1. TAG EXTRACTION  [layer1_router.py]                                                         │
│  │       extract_tags_llama(text)  →  raw tags                                                     │
│  │       normalize_tags(tags)  →  normalized_tags                                                  │
│  │       TemporalLocalityLayer.add_statement()                                                     │
│  │                                                                                                 │
│  ├── 2. LAYER 0 ROUTING  [layer0/router.py]                                                        │
│  │       QuestionRouter.route(text)                                                                │
│  │       ├─ route == REASONING_PIPELINE  →  continue to step 3                                     │
│  │       └─ other routes (refusal, direct)  →  emit SSE + skip remaining steps                     │
│  │                                                                                                 │
│  ├── 3. MULTI-LENS ROUTING  [multi_lens_router.py]  ←── see detail block below                    │
│  │       MultiLensRouter.route(text)  →  RoutingResult                                             │
│  │       TRMLens.refine(routing_context)  →  refined routing_context                               │
│  │       └─ shadow_signal PROMOTE_TO_EXPERT → reload_domain_ontology()                             │
│  │                                                                                                 │
│  ├── 4. DOMAIN RESOLUTION                                                                          │
│  │       routing_context.selected_domains  →  relevant_domains[]                                   │
│  │       ExpertFilter.normalize_domain()  (fallback normalization)                                 │
│  │       emit graph:dfs_step (one event per domain)                                                │
│  │                                                                                                 │
│  ├── 5. TRM REASONER  (optional)  [trm/reasoner.py]                                               │
│  │       _run_trm_reasoner(text, routing_context, relevant_domains)                                │
│  │       ├─ reranks relevant_domains by domain_probs                                               │
│  │       ├─ halt_confidence < trm_threshold  →  should_escalate = True                             │
│  │       └─ TRMOODFallback.should_trigger()  →  LEVEL_N ood_fallback_result                        │
│  │                                                                                                 │
│  ├── 6. LAYER 1 SPATIAL/TEMPORAL ANALYSIS  [layer1_router.py]                                      │
│  │       embed_tags_transformer(normalized_tags)  →  tag_vectors                                   │
│  │       cluster_tags_transformer()  →  tag_clusters                                               │
│  │       TemporalLocalityLayer.get_recent_statements()                                             │
│  │       analyze_spatial_locality()  →  spatial_analysis                                          │
│  │       assign_domain_patch()  →  domain_patch                                                   │
│  │                                                                                                 │
│  ├── 7. EXPERT PRE-FILTER  [expert_filter.py]                                                      │
│  │       ExpertFilter.filter_experts_by_tags()  →  pre_filter_result                               │
│  │       filtered_experts = {domain: expert}  →  only in relevant_domains                         │
│  │                                                                                                 │
│  ├── 8. PHASE 2 PIPELINE  [phase2/pipeline.py]                                                     │
│  │       Phase2Pipeline.run(text, routing_context, filtered_experts, depth_cfg)                   │
│  │       emit graph:tool_start / graph:tool_done                                                   │
│  │       └─ returns Phase2Result  →  _adapt_phase2_to_p3()  →  P3FinalDecisionResult               │
│  │                                                                                                 │
│  ├── 9. PHASE 3–5 PIPELINE  [phase3/pipeline.py]                                                   │
│  │       Phase3To5Pipeline.run_complete_pipeline(final_decision_result)                            │
│  │       emit graph:tool_start / graph:tool_done                                                   │
│  │       └─ returns phase3_result (validation + action_result + phase_latencies)                   │
│  │                                                                                                 │
│  ├── 10. UNIFIED EXPERT DECISION  [unified_expert_system.py]                                       │
│  │        emit graph:synthesis_start                                                               │
│  │        UnifiedExpertSystem.unified_decision_analysis(                                           │
│  │            text, routing_context, filtered_experts, depth_cfg)                                 │
│  │        └─ returns unified_decision  →  _adapt_unified_to_p3()                                   │
│  │                                                                                                 │
│  ├── 11. ORCHESTRATION COMBINE  [orchestration.py]                                                 │
│  │        combine_routing_and_expert_decisions(routing, expert)                                    │
│  │                                                                                                 │
│  ├── 12. TRM TRACE WRITER  [trm/trm_routing_trace_writer.py]  (optional)                          │
│  │        trm_trace_writer.record(text, routing_context, selected_domain, ...)                    │
│  │                                                                                                 │
│  └── →  append to all_sentence_data[]                                                             │
│           fields: sentence, tags, routing_context, phase2_result, phase3_result,                   │
│                   expert_decision, shadow_signal, trm_reasoner_result,                             │
│                   ood_fallback_result, should_escalate                                             │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘


                              ↑ step 1 + step 6 expand here ↓
┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  LAYER 1 ROUTER  (layer1_router.py)                                                                │
│                                                                                                    │
│  ┌─ DOMAIN ONTOLOGY ────────────────────────────────────────────────────────────────────────────┐  │
│  │  _BASE_ONTOLOGY — statically defined per-domain core+attribute vocabularies                  │  │
│  │  _build_domain_ontology()                                                                    │  │
│  │  └─ _discover_live_domains()  →  cfg.discover_live_domains()                                 │  │
│  │       ├─ scans [experts].dir for subdirs containing *.pt / *.bin / *.safetensors              │  │
│  │       ├─ strips _bert / _model / _expert suffixes                                             │  │
│  │       ├─ writes result back → config.toml [layer1.domain_list]  (non-fatal on failure)       │  │
│  │       └─ falls back to static [layer1.domain_list] if disk scan finds nothing                 │  │
│  │  Unknown live domains → minimal scaffold {core:[name], attributes:[]}                        │  │
│  └──────────────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                                    │
│  ┌─ TAG EXTRACTION ────────────────────────────────────────────────────────────────────────────┐  │
│  │  extract_tags_llama(text)   →  ollama.generate(model, prompt)                               │  │
│  │  extract_tags_openai(text)  →  requests.post(endpoint, ...)  (OpenAI-compat)                │  │
│  │  _tag_cache (LRU, maxlen=2048, TTL=3600s)  ←─ sha256 key                                    │  │
│  │  normalize_tags(tags)  →  fuzzywuzzy.process.extractOne()  (threshold from cfg)             │  │
│  └──────────────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                                    │
│  ┌─ EMBEDDING & CLUSTERING ───────────────────────────────────────────────────────────────────┐  │
│  │  embed_tags_transformer(tags)                                                               │  │
│  │  └─ model_registry.embed_batch(tags, model_name)  →  np.array of vectors                   │  │
│  │  cluster_tags_transformer(tags, embeddings)  →  cosine-similarity greedy merge             │  │
│  │  cluster_tags(tags, embeddings)  →  AgglomerativeClustering(cosine, average)               │  │
│  │  └─ distance_threshold from cfg (layer1.tag_cluster_distance_threshold)                    │  │
│  └──────────────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                                    │
│  ┌─ TEMPORAL LOCALITY LAYER ──────────────────────────────────────────────────────────────────┐  │
│  │  TemporalLocalityLayer(max_size, time_window_hours)                                         │  │
│  │  ├─ add_statement(sentence, tags, timestamp)  →  deque + tag_frequency counter              │  │
│  │  ├─ get_recent_statements(time_limit_hours)  →  entries within cutoff window                │  │
│  │  ├─ get_temporal_similarity(input_tags)  →  Jaccard(input ∩ recent / input ∪ recent)        │  │
│  │  └─ get_frequent_tags(min_frequency)  →  {tag: freq}                                       │  │
│  └──────────────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                                    │
│  ┌─ SPATIAL LOCALITY & DOMAIN PATCH ─────────────────────────────────────────────────────────┐  │
│  │  analyze_spatial_locality(recent_statements, clusters)                                     │  │
│  │  └─ cluster_counts  →  dominant_clusters[], cluster_distribution                           │  │
│  │  assign_domain_patch(spatial_analysis)                                                     │  │
│  │  ├─ dominance > threshold   →  use_existing_patch   (cluster_id)                          │  │
│  │  ├─ moderate dominance      →  create_hybrid_patch  (primary_cluster)                     │  │
│  │  └─ no dominance            →  create_new_patch                                           │  │
│  └──────────────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                                    │
│  ┌─ MULTI-LENS ROUTE (called by MultiLensRouter) ────────────────────────────────────────────┐  │
│  │  Lens 1  _lens1_embedding_candidates(text, top_k)                                          │  │
│  │          embed_weight*cosine_sim + lex_weight*token_overlap per domain anchor               │  │
│  │          _deduplicate_candidates()  →  cosine cluster, keep best scorer                    │  │
│  │  Lens 2  _lens2_ontology_explanations(text, candidates)                                    │  │
│  │          token overlap vs core+0.5×attr vocab; child/parent suppression                    │  │
│  │  Lens 3  _lens3_abstraction_signature(text, concepts)                                      │  │
│  │          {core:{domain:[hits]}, modifiers:{domain:[hits]}}                                  │  │
│  │          active_domains.core == 0  →  ATTRIBUTE_ONLY path                                  │  │
│  │  Shadow  _run_shadow_check(text, fusion_scores)                                            │  │
│  │          ShadowDomainDetector.observe()                                                    │  │
│  │          PROMOTE_TO_EXPERT → reload_domain_ontology() + emit promote_shadow_domain         │  │
│  └──────────────────────────────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘


                              ↑ step 3 expands here ↓
┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  MULTI-LENS ROUTER  (multi_lens_router.py :: MultiLensRouter.route)                                │
│                                                                                                    │
│  ┌─ CONSTRUCTION ───────────────────────────────────────────────────────────────────────────────┐  │
│  │  MultiLensRouter(spectral_analyzer=<synced analyzer from DynamicSignatureManager>)           │  │
│  │  ├─ _spectral_analyzer = RuntimeSpectralAnalyzer(sig_dir, model=all-MiniLM-L6-v2)           │  │
│  │  │   (or injected from run_workflow's DynamicSignatureManager.sync_signatures())             │  │
│  │  └─ _fusion_engine     = FusionEngine()                                                     │  │
│  └──────────────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                                    │
│  route(text) — full pipeline                                                                       │
│  │                                                                                                 │
│  ├─ A. multi_lens_route(text)  [layer1_router.py]                                                  │
│  │       → base_result { classification, lens1_candidates, lens2_explanations,                     │
│  │                        lens3_signature, primary_domain, shadow_signal }                         │
│  │       is_attribute_only = classification == "ATTRIBUTE_ONLY"                                    │
│  │       semantic_scores   = lens1_candidates  →  {domain: score}                                  │
│  │       object_level_domains = lens3.core hits + lens2 "object" level matches                     │
│  │                                                                                                 │
│  ├─ B. SPECTRAL ANALYSIS  [spectral_analyzer.py]                                                   │
│  │       RuntimeSpectralAnalyzer.analyze_text(text)  →  spectral_scores {domain: score}            │
│  │       (skipped silently if analyzer not ready)                                                  │
│  │                                                                                                 │
│  ├─ C. FUSION  [fusion_engine.py]                                                                  │
│  │       FusionEngine.fuse(semantic_scores, spectral_scores, confidence_scores)                    │
│  │       → fused_scores {domain: fused_score}                                                      │
│  │       FALLBACK if FusionEngine unavailable or raises:                                           │
│  │         _merge_scores_fallback()                                                                │
│  │         domain in both sources:  0.55×sem + 0.45×spc                                           │
│  │         domain in semantic only: 0.55×sem                                                       │
│  │         domain in spectral only: 0.45×spc                                                       │
│  │                                                                                                 │
│  ├─ D. PRE-SELECTION CLUSTERING                                                                    │
│  │       _preselect_cluster(fused_scores)                                                          │
│  │       └─ _deduplicate_candidates()  →  collapse near-synonym domains before budget-select       │
│  │                                                                                                 │
│  ├─ E. BUDGET SELECT                                                                               │
│  │       _budget_select(fused_scores, confidence_scores,                                           │
│  │                      min_score=DOMAIN_SCORE_THRESHOLD, max_experts=MAX_EXPERTS)                 │
│  │       sort by (fused_score DESC, confidence DESC, domain ASC)                                   │
│  │       → selected_experts[], budget_cap_applied                                                  │
│  │       create_new_expert = True when selected_experts=[] AND NOT attribute_only                  │
│  │                                                                                                 │
│  ├─ F. ATTRIBUTE OVERRIDE  (if ATTRIBUTE_ONLY and ENABLE_ATTRIBUTE_OVERRIDE)                       │
│  │       promote object_level_domains scoring ≥ threshold into selected_experts                   │
│  │       (respects max_experts budget cap)                                                         │
│  │                                                                                                 │
│  ├─ G. LAST-RESORT FALLBACK                                                                        │
│  │       if still empty → use base_result.primary_domain                                          │
│  │                                                                                                 │
│  └─ H. CLASSIFICATION                                                                              │
│         variance = var(fused_scores.values())                                                      │
│         ATTRIBUTE_ONLY        no override applied, is_attribute_only=True                          │
│         SINGLE_DOMAIN         len(selected)==1                                                     │
│         MULTI_DOMAIN          len(selected)>1  AND  variance >= 0.1                                │
│         AMBIGUOUS             len(selected)>1  AND  variance < 0.1                                 │
│         NO_EXPERT_AVAILABLE   selected empty after all fallbacks                                   │
│                                                                                                    │
│  Returns: RoutingResult {                                                                          │
│    classification, selected_domains, primary_domain, fusion_scores,                                │
│    coverage, create_new_expert,                                                                    │
│    metadata { lens_scores:{semantic,spectral,confidence},                                          │
│               fused_scores, variance, explanation }                                                │
│  }                                                                                                 │
│                                                                                                    │
│  Metrics tracked (in-process): total_requests, attribute_only, single_domain,                     │
│  multi_domain, ambiguous, no_expert, create_new_expert_true,                                       │
│  attribute_override_applied, budget_cap_applied                                                    │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘


                         SSE EVENTS EMITTED (phase_name  →  frontend meaning)
                         ───────────────────────────────────────────────────────
                         setting_up              Model registry warming up
                         environment_ready       Models loaded, pipeline starting
                         graph_expert_init       UnifiedExpertSystem initializing
                         graph_spectral_sync     SpectralAnalyzer syncing .npy files
                         graph_router_ready      MultiLensRouter + ExpertFilter ready
                         graph_layer0            Layer 0 classifying query
                         routing                 Multi-lens routing starting
                         graph_routing           Routing complete, classification known
                         graph:dfs_step          One domain explored (one event/domain)
                         graph_trm_decision      TRMReasoner halt/escalate decision
                         promote_shadow_domain   Shadow domain promoted to expert
                         graph:tool_start        Phase 2 or Phase 3 starting
                         graph:tool_done         Phase 2 or Phase 3 complete
                         graph:synthesis_start   UnifiedExpertSystem arbitrating
                         done                    Full pipeline complete (payload=ChatResponse)


                               SUPPORTING MODULES WIRED IN
                               ─────────────────────────────────

  config_loader.py          All TOML config accessors (typed, static methods)
  api_models.py             Pydantic models: ChatRequest, ChatResponse,
                            MyceliumRunSummary, ReasoningTrace, ReasoningMode,
                            get_depth_config()
  pipeline_event.py         EventEmitter, PipelineEvent, ReplayJournal
  model_registry.py         warmup(), loaded_models(), STARTUP_SPECS
  dynamic_signature_manager.py  sync_signatures()  →  SpectralAnalyzer (.npy)
  spectral_analyzer.py      RuntimeSpectralAnalyzer.analyze_text()  →  {domain:score}
  fusion_engine.py          FusionEngine.fuse(semantic, spectral, confidence)
                            fallback: _merge_scores_fallback() (0.55/0.45 weights)
  patch_batch_logger.py     log_query() / fill_response()  for CREATE_NEW_PATCH
  patch_dag.py              Patch DAG graph manager
  sandbox_manager.py        get_sandbox_manager()  →  MCP tool execution
  sandbox_models.py         build_sandbox_task_from_run()  →  SandboxTask
  mcp_client.py             MCP protocol client
  mcp_tools_server.py       MCP tool server (exposes tools to sandbox)
  conversation_agent.py     get_conversation_agent()  →  LLM answer synthesis
  llm_providers.py          OpenAI / Ollama provider abstraction
  orchestration.py          combine_routing_and_expert_decisions()
  lexis_bridge.py           Lexical analysis bridge
  lexis_condenser.py        Lexical condensation / summarisation
  spacy_bridge.py           spaCy NLP bridge
  spacy_worker.py           spaCy subprocess worker


                                TRM SUBSYSTEM  (mycelium/trm/)
                                ───────────────────────────────

  trm_engine.py             TRMEngine(store, dfs)  — top-level TRM coordinator
  graph_store.py            GraphStore  — persistent domain knowledge graph
  multi_worker_dfs.py       MultiWorkerDFSLookup  — parallel DFS over GraphStore
  integration.py            TRMLens.refine()  — post-routing context enrichment
  reasoner.py               TRMReasoner (PyTorch) — domain classifier + halt head
  config.py                 TRMConfig  — model hyperparams + checkpoint path
  trm_ood_fallback.py       TRMOODFallback + TRMOODHead  — OOD detection
  trm_routing_trace_writer.py  trm_trace_writer.record()  — training data logger
  trm_checkpoints/          trm_latest.pt  — latest checkpoint (gitignored)


                         OPTIONAL MODULES (graceful degradation if unavailable)
                         ───────────────────────────────────────────────────────
                         mycelium/contradiction/  ContradictionClassifier
                         mycelium/canonicalization/  CanonicalizeAndHash
                         mycelium/fusion/  DSTFusion, ConfidenceStateFusion
                         mycelium/reasoning/  DAGDecomposer
                         TRMReasoner / TRMOODFallback  (requires torch)
```

---

> **Stage 3 complete.** `multi_lens_router.py` expanded: full route() pipeline (A→H),
> spectral analysis, FusionEngine + fallback merge weights, pre-selection clustering,
> budget-select, attribute override, last-resort fallback, classification logic, metrics.
> Next: `unified_expert_system.py`, `phase2/pipeline.py`, `phase3/pipeline.py`, `layer0/router.py`, TRM internals, web-ui.
