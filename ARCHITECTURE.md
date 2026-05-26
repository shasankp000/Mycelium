# Mycelium — Architecture Diagram

> **WIP — updated in stages.** See `.diagram_wip.md` for expansion plan.  
> Stage 2: `layer1_router.py` internals expanded (domain discovery, multi-lens route, temporal/spatial layers).

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  CLIENT (Next.js / web-ui)                                                                         │
│                                                                                                    │
│   ReasoningGraph/index.tsx                                                                         │
│   └─ EventSource  POST /api/v1/chat/stream  ─────────────────────────────────────────────────┬────┐│
│               GET  /api/v1/chat/stream  (legacy shim)                      SSE stream         │    ││
│              POST  /api/v1/chat          (blocking JSON)                                      │    ││
└────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                                                               │    │
┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  FASTAPI BACKEND  (broadcast_api.py)                                                  │    │        │
│                                                                                                    │
│  Startup                                                                                           │
│  └─ preload_models()                                                                               │
│      └─ model_registry.warmup([mpnet, MiniLM])  ── _warmup_done.set()                              │
│      └─ _write_calibration_state()  →  runtime/calibration_state.json                              │
│                                                                                                    │
│  POST /api/v1/chat/stream  (primary SSE)                                              ◄────┘       │
│  GET  /api/v1/chat/stream  (legacy GET shim, same path)                                            │
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
│            ├─ _build_run_summary()  ───────────────────────────────────────────────────────────────┐│
│            ├─ _run_sandbox()                                                                       ││
│            ├─ ConversationAgent.answer()                                                           ││
│            ├─ append_trace()  →  traces/*.jsonl                                                    ││
│            └─ patch_logger.log_query() / fill_response()  (patch queries)                         ││
│                                                                                                    │
│  asyncio.Queue  ─────────────────────── pipeline events (SSE)  ────────────────────────────────────┘
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
│  ├── 3. MULTI-LENS ROUTING  [multi_lens_router.py]                                                 │
│  │       MultiLensRouter.route(text)  →  routing_context                                           │
│  │       TRMLens.refine(routing_context)  →  refined routing_context                               │
│  │       ShadowDomainDetector (embedded in MultiLensRouter)                                        │
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

                                    |
                         step 1 + step 6 detail
                                    |
┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  LAYER 1 ROUTER  (layer1_router.py)                                                                │
│                                                                                                    │
│  ┌─ DOMAIN ONTOLOGY (built once at import, refreshed by reload_domain_ontology()) ──────────────┐  │
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
│  │                                                                                             │  │
│  │  cluster_tags_transformer(tags, embeddings)  →  cosine-similarity greedy merge             │  │
│  │  └─ clusters: {cluster_id: [member_tags]}                                                  │  │
│  │                                                                                             │  │
│  │  cluster_tags(tags, embeddings)  →  AgglomerativeClustering(cosine, average)               │  │
│  │  └─ distance_threshold from cfg (layer1.tag_cluster_distance_threshold)                    │  │
│  └──────────────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                                    │
│  ┌─ TEMPORAL LOCALITY LAYER ─────────────────────────────────────────────────────────────────┐  │
│  │  TemporalLocalityLayer(max_size, time_window_hours)                                        │  │
│  │  ├─ add_statement(sentence, tags, timestamp)  →  deque + tag_frequency counter             │  │
│  │  ├─ get_recent_statements(time_limit_hours)  →  entries within cutoff window               │  │
│  │  ├─ get_temporal_similarity(input_tags)                                                    │  │
│  │  │    →  Jaccard(input ∩ recent / input ∪ recent)                                          │  │
│  │  └─ get_frequent_tags(min_frequency)  →  {tag: freq}                                      │  │
│  └──────────────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                                    │
│  ┌─ SPATIAL LOCALITY ANALYSIS ────────────────────────────────────────────────────────────────┐  │
│  │  analyze_spatial_locality(recent_statements, clusters)                                     │  │
│  │  └─ cluster_counts from recent tags  →  dominant_clusters[], cluster_distribution          │  │
│  │                                                                                             │  │
│  │  assign_domain_patch(spatial_analysis)                                                     │  │
│  │  ├─ dominance_ratio > threshold  →  use_existing_patch   (cluster_id)                     │  │
│  │  ├─ moderate dominance           →  create_hybrid_patch  (primary_cluster)                │  │
│  │  └─ no dominant clusters         →  create_new_patch                                      │  │
│  └──────────────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                                    │
│  ┌─ MULTI-LENS ROUTE (multi_lens_route) ─────────────────────────────────────────────────────┐  │
│  │                                                                                             │  │
│  │  should_escalate=False  →  _fast_lexical_route()  →  classification=FAST                  │  │
│  │                              pure token-overlap scoring, no embedding model                │  │
│  │                                                                                             │  │
│  │  should_escalate=True   →  full DFS                                                        │  │
│  │  │                                                                                         │  │
│  │  ├─ Lens 1  _lens1_embedding_candidates(text, top_k)                                       │  │
│  │  │          ├─ anchor text = core[:5] + attributes[:3] per domain                          │  │
│  │  │          ├─ embed_weight * cosine_sim + lex_weight * token_overlap                       │  │
│  │  │          └─ _deduplicate_candidates()  →  cosine cluster → keep best scorer             │  │
│  │  │               └─ _parent_dedup()  fallback if embedding unavailable                    │  │
│  │  │                                                                                         │  │
│  │  ├─ Lens 2  _lens2_ontology_explanations(text, candidates, max_candidates)                 │  │
│  │  │          ├─ token overlap against core + 0.5×attribute vocab per domain                 │  │
│  │  │          ├─ child-domain penalty: child_core ⊆ parent_core  →  ×0.7                    │  │
│  │  │          ├─ secondary dampening: parent_cov ≥ 0.9×child_cov  →  ×0.9                  │  │
│  │  │          └─ parent-child suppression: suppress child if parent scores ≥ 0.9×           │  │
│  │  │                                                                                         │  │
│  │  ├─ Lens 3  _lens3_abstraction_signature(text, concepts)                                   │  │
│  │  │          └─ {core: {domain:[hits]}, modifiers: {domain:[hits]}}                         │  │
│  │  │               active_domains.core == 0  →  ATTRIBUTE_ONLY path                         │  │
│  │  │                                                                                         │  │
│  │  └─ Shadow  _run_shadow_check(text, fusion_scores)                                         │  │
│  │             └─ ShadowDomainDetector.observe()                                              │  │
│  │                  ├─ ATTRIBUTE_ONLY path: always runs                                       │  │
│  │                  └─ NORMAL path: only if top lens1 score < shadow_threshold                │  │
│  │                       signal.status == PROMOTE_TO_EXPERT                                   │  │
│  │                       →  reload_domain_ontology() + emit promote_shadow_domain             │  │
│  │                                                                                             │  │
│  │  Output: {primary_domain, secondary_domains[], explanation,                                │  │
│  │           lens1_candidates, lens2_explanations, lens3_signature,                           │  │
│  │           classification: NORMAL|ATTRIBUTE_ONLY|SHADOW|FAST,                               │  │
│  │           shadow_signal}                                                                   │  │
│  └──────────────────────────────────────────────────────────────────────────────────────────────┘  │
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

> **Stage 2 complete.** `layer1_router.py` internals expanded: domain discovery/write-back,
> multi-lens route (Lens 1/2/3 + shadow check), temporal locality layer, spatial locality
> + domain patch assignment.
> Next: expand `multi_lens_router.py`, `unified_expert_system.py`, `phase2/`, `phase3/`, `layer0/`, TRM subsystem.
