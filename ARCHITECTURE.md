# Mycelium — Architecture Diagram

> **WIP — updated in stages.** See `.diagram_wip.md` for expansion plan.  
> Stage 5: `phase2/pipeline.py` internals expanded (6-phase orchestration, expert resolution, depth-cap).

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
│  ... unchanged from Stage 4 ...                                                                    │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                            |
                               _build_run_summary()
                               └─ run_mycelium_workflow()  [run_workflow.py]
                                    |
┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  PIPELINE ORCHESTRATOR  (run_workflow.py)                                                           │
│  ... unchanged from Stage 4 ...                                                                    │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘


                              ↑ step 3 expands here ↓
┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  MULTI-LENS ROUTER  (multi_lens_router.py)                                                          │
│  ... unchanged from Stage 3 ...                                                                    │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘


                              ↑ step 10 expands here ↓
┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  UNIFIED EXPERT SYSTEM  (unified_expert_system.py)                                                 │
│  ... unchanged from Stage 4 ...                                                                    │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘


                              ↑ step 8 expands here ↓
┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  PHASE 2 PIPELINE  (phase2/pipeline.py :: Phase2Pipeline)                                          │
│                                                                                                    │
│  Phase2Pipeline.__init__(config=Phase2Config())                                                    │
│  ├─ InputNormalizationPipeline(config)        phase_2_1_input_normalization.py                   │
│  ├─ SemanticUnderstandingPipeline(config)     phase_2_2_semantic_understanding.py                │
│  ├─ ExpertSelectionPipeline(config)           phase_2_3_expert_selection.py                      │
│  ├─ MultiExpertInferencePipeline(config)      phase_2_4_inference.py                             │
│  ├─ CalibrationPipeline(config)               phase_2_5_calibration.py                           │
│  └─ DecisionSynthesisPipeline(config)         phase_2_6_synthesis.py                             │
│                                                                                                    │
│  Phase2Pipeline.run(text, routing_context, filtered_experts, depth_config)                        │
│  │                                                                                                 │
│  ├─ 2.1  InputNormalizationPipeline.normalize(text)                                               │
│  │       → norm_result  { cleaned_text, domain_mapping, ... }                                      │
│  │                                                                                                 │
│  ├─ 2.2  SemanticUnderstandingPipeline.understand(norm_result, domains)                           │
│  │       → semantic_result                                                                         │
│  │                                                                                                 │
│  ├─ 2.3  Expert resolution (priority order):                                                       │
│  │       1. available_experts  (explicit list, highest priority)                                   │
│  │       2. filtered_experts   (dict → [{name:expert_{d}, domain:d}] conversion)                  │
│  │       3. ExpertSelectionPipeline.build_default_experts(semantic_result)                        │
│  │       ExpertSelectionPipeline.select_experts(semantic_result, experts,                         │
│  │                                              routing_context=routing_context)                  │
│  │       → selection_result { selected_experts[] }                                                 │
│  │       depth_config.expert_top_k caps selected_experts length                                   │
│  │         fast=1  balanced=2  deep=3                                                              │
│  │                                                                                                 │
│  ├─ 2.4  MultiExpertInferencePipeline.infer(cleaned_text, selected_experts)                       │
│  │       → inference_result                                                                        │
│  │                                                                                                 │
│  ├─ 2.5  CalibrationPipeline.calibrate_and_quantify(inference_result)                             │
│  │       → calibration_result                                                                      │
│  │                                                                                                 │
│  └─ 2.6  DecisionSynthesisPipeline.synthesize(                                                    │
│            norm_result, semantic_result, selection_result,                                         │
│            inference_result, calibration_result)                                                   │
│         → FinalDecisionResult { final_decision, decision_confidence, phase_outputs... }            │
│                                                                                                    │
│  On any exception → raises Phase2PipelineError (caller gets structured failure)                   │
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


                          PHASE 2 SUB-PHASES  (phase2/phases/)
                          ────────────────────────────────────────

  phase_2_1_input_normalization.py    InputNormalizationPipeline
                                      .normalize(text)  →  NormalizationResult
  phase_2_2_semantic_understanding.py SemanticUnderstandingPipeline
                                      .understand(norm, domains)  →  SemanticResult
  phase_2_3_expert_selection.py       ExpertSelectionPipeline
                                      .select_experts(semantic, experts, routing_context)
                                      .build_default_experts(semantic)  →  [{name,domain}]
  phase_2_4_inference.py              MultiExpertInferencePipeline
                                      .infer(text, selected_experts)  →  InferenceResult
  phase_2_5_calibration.py            CalibrationPipeline
                                      .calibrate_and_quantify(inference)  →  CalibrationResult
  phase_2_6_synthesis.py              DecisionSynthesisPipeline
                                      .synthesize(norm, semantic, selection,
                                                  inference, calibration)
                                      →  FinalDecisionResult { final_decision,
                                                               decision_confidence }


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

> **Stage 5 complete.** `phase2/pipeline.py` expanded: 6-phase sequence (2.1 input normalisation →
> 2.2 semantic understanding → 2.3 expert resolution with 3-priority fallback + depth-cap →
> 2.4 multi-expert inference → 2.5 calibration → 2.6 decision synthesis → FinalDecisionResult).
> Sub-phase registry added to supporting-modules section.
> Next: `phase3/pipeline.py`, `layer0/router.py`, TRM internals, web-ui.
