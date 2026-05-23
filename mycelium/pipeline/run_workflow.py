import json
import datetime
import time as _time
from collections import Counter, deque
from dataclasses import asdict, dataclass, is_dataclass
from typing import Callable, List, Dict, Any, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import random
from mycelium.pipeline.layer1_router import (
    extract_tags_llama,
    normalize_tags,
    embed_tags_transformer,
    cluster_tags_transformer,
    TemporalLocalityLayer,
    analyze_spatial_locality,
    assign_domain_patch,
)
from mycelium.pipeline.multi_lens_router import MultiLensRouter
from mycelium.pipeline.phase2.pipeline import Phase2Pipeline
from mycelium.pipeline.phase3.pipeline import Phase3To5Pipeline
from mycelium.pipeline.phase3.utils.types import (
    FinalDecisionResult as P3FinalDecisionResult,
)
from mycelium.pipeline.unified_expert_system import UnifiedExpertSystem
from mycelium.pipeline.expert_filter import ExpertFilter
from mycelium.pipeline.orchestration import combine_routing_and_expert_decisions
from mycelium.pipeline.layer0.router import QuestionRouter
from mycelium.trainers.tuning_config import ENABLE_LOGGING, LOG_SAMPLE_RATE
from mycelium.pipeline.patch_batch_logger import patch_logger
from mycelium.pipeline.dynamic_signature_manager import DynamicSignatureManager
from mycelium.pipeline.model_registry import warmup, loaded_models, STARTUP_SPECS
from mycelium.pipeline.pipeline_event import EventEmitter, make_emitter
# Phase 6 — reasoning mode / depth config
from mycelium.pipeline.api_models import ReasoningMode, get_depth_config
# Phase D — TRM: routing refinement + TRMEngine (owns GraphStore +
#            MultiWorkerDFSLookup + PromotionPolicy in one place)
from mycelium.trm.integration import TRMLens
from mycelium.trm.graph_store import GraphStore
from mycelium.trm.trm_engine import TRMEngine
# Phase F — MultiWorkerDFSLookup (injected into TRMEngine so WorkerPool
#            is shared; auto-selects single vs multi-worker by store size)
from mycelium.trm.multi_worker_dfs import MultiWorkerDFSLookup
# Phase D — DAGDecomposer
from mycelium.reasoning.dag_decomposer import DAGDecomposer
# Phase D (Option 2) — TRMReasoner: neural routing refinement
try:
    import torch as _torch
    from mycelium.trm.reasoner import TRMReasoner
    from mycelium.trm.config import TRMConfig
    from mycelium.trm.trm_routing_trace_writer import (
        trm_trace_writer,
        _encode_query as _trm_encode_query,
        _spectral_vec_to_tensor as _trm_spectral_vec,
        _domain_to_idx as _trm_domain_to_idx,
        _initial_domain_probs as _trm_init_probs,
        PREDICATE_FAMILY_MAP as _TRM_PRED_MAP,
        _DEFAULT_PREDICATE_FAMILY as _TRM_DEF_PRED,
        N_DOMAINS as _TRM_N_DOMAINS,
    )
    _TRM_AVAILABLE = True
except Exception as _trm_import_err:
    TRMReasoner = None  # type: ignore[assignment,misc]
    TRMConfig = None    # type: ignore[assignment,misc]
    trm_trace_writer = None  # type: ignore[assignment]
    _TRM_AVAILABLE = False
# Phase E — ContradictionClassifier (non-fatal fallback)
try:
    from mycelium.contradiction.classifier import ContradictionClassifier
    _CONTRADICTION_AVAILABLE = True
except Exception:
    ContradictionClassifier = None  # type: ignore[assignment,misc]
    _CONTRADICTION_AVAILABLE = False
# Phase B — canonicalization pipeline
try:
    from mycelium.canonicalization.semantic_hash_pipeline import CanonicalizeAndHash
    _CANON_AVAILABLE = True
except Exception:
    CanonicalizeAndHash = None  # type: ignore[assignment,misc]
    _CANON_AVAILABLE = False
# Phase F — DST confidence fusion
try:
    from mycelium.fusion.dst_fusion import DSTFusion, ConfidenceStateFusion, ConflictError
    _DST_AVAILABLE = True
except Exception:
    DSTFusion = None  # type: ignore[assignment,misc]
    ConfidenceStateFusion = None  # type: ignore[assignment,misc]
    ConflictError = None  # type: ignore[assignment,misc]
    _DST_AVAILABLE = False


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, deque)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "__dict__") and not isinstance(obj, bytes):
        return _to_jsonable(vars(obj))
    return obj


def _adapt_phase2_to_p3(p2: Any, original_text: str = "") -> P3FinalDecisionResult:
    def _get(*attrs: str, default: Any = "") -> Any:
        for attr in attrs:
            val = getattr(p2, attr, None)
            if val is None and isinstance(p2, dict):
                val = p2.get(attr)
            if val is not None:
                return val
        return default

    reasoning_raw = _get("reasoning_chain", "reasoning", default=[])
    if isinstance(reasoning_raw, list):
        reasoning_str = " ".join(str(r) for r in reasoning_raw)
    else:
        reasoning_str = str(reasoning_raw)

    extra_fields = (
        "original_text", "input_text", "query", "sentence",
        "action_details", "expert_predictions",
    )
    metadata: Dict[str, Any] = {}
    for f in extra_fields:
        val = getattr(p2, f, None)
        if val is None and isinstance(p2, dict):
            val = p2.get(f)
        if val is not None:
            metadata[f] = val

    if original_text:
        metadata["original_query"] = original_text
    elif not metadata.get("original_query"):
        for fallback_key in ("original_text", "input_text", "query", "sentence"):
            if metadata.get(fallback_key):
                metadata["original_query"] = metadata[fallback_key]
                break

    return P3FinalDecisionResult(
        decision=_get("decision_label", "prediction", "final_decision", "decision"),
        confidence=float(_get("confidence", "expert_confidence", default=0.5)),
        reasoning=reasoning_str,
        action=_get("action_type", "action", "recommended_action", default="use_existing"),
        expert_name=_get("selected_expert", "expert_name", "expert"),
        domain=_get("domain", "selected_domain", default=""),
        metadata=metadata,
    )


def _adapt_unified_to_p3(
    expert_decision: Any,
    original_text: str,
    phase2_metadata: Dict[str, Any],
) -> P3FinalDecisionResult:
    decision_type: str = getattr(expert_decision, "decision_type", "") or ""
    action_map = {
        "USE_EXISTING_EXPERT": "use_existing",
        "CREATE_NEW_PATCH": "create_new_patch",
        "CREATE_NEW_EXPERT": "create_new_expert",
    }
    action = action_map.get(decision_type.upper(), "use_existing")
    selected_experts: List[str] = list(
        getattr(expert_decision, "selected_experts", []) or []
    )
    expert_name = selected_experts[0] if selected_experts else ""
    domain = expert_name
    confidence = float(getattr(expert_decision, "expert_confidence", 0.0) or 0.0)
    reasoning = str(getattr(expert_decision, "reasoning", "") or "")
    metadata: Dict[str, Any] = dict(phase2_metadata)
    metadata["original_query"] = original_text
    metadata["unified_decision_type"] = decision_type
    metadata["unified_selected_experts"] = selected_experts
    return P3FinalDecisionResult(
        decision=decision_type,
        confidence=confidence,
        reasoning=reasoning,
        action=action,
        expert_name=expert_name,
        domain=domain,
        metadata=metadata,
    )


@dataclass
class WorkflowMetrics:
    layer0_routes: Counter = None
    routing_classifications: Counter = None
    expert_decisions: Counter = None
    domains: Counter = None

    def __post_init__(self) -> None:
        self.layer0_routes = Counter() if self.layer0_routes is None else self.layer0_routes
        self.routing_classifications = (
            Counter() if self.routing_classifications is None else self.routing_classifications
        )
        self.expert_decisions = Counter() if self.expert_decisions is None else self.expert_decisions
        self.domains = Counter() if self.domains is None else self.domains

    def to_dict(self) -> Dict[str, Dict[str, int]]:
        return {
            "layer0_routes": dict(self.layer0_routes),
            "routing_classifications": dict(self.routing_classifications),
            "expert_decisions": dict(self.expert_decisions),
            "domains": dict(self.domains),
        }


# ---------------------------------------------------------------------------
# Process-wide TRMReasoner singleton (Option 2 wiring)
# ---------------------------------------------------------------------------

_trm_reasoner: Optional[Any] = None  # TRMReasoner | None


def _get_trm_reasoner() -> Optional[Any]:
    """
    Lazily initialise and return the process-wide TRMReasoner.

    * Loads weights from TRMConfig.model_path when the file exists.
    * Falls back to random-initialised weights when no checkpoint is
      present (fallback_to_router=True guarantees safe operation).
    * Returns None if TRM is unavailable (import failed).
    """
    global _trm_reasoner
    if not _TRM_AVAILABLE:
        return None
    if _trm_reasoner is not None:
        return _trm_reasoner
    try:
        cfg = TRMConfig()
        reasoner = TRMReasoner(cfg)
        import os as _os
        if cfg.model_path and _os.path.isfile(cfg.model_path):
            state = _torch.load(cfg.model_path, map_location="cpu")
            reasoner.load_state_dict(state)
            print(f"\u2705 TRMReasoner: loaded weights from {cfg.model_path}")
        else:
            print(
                "\u26a0\ufe0f  TRMReasoner: no checkpoint found at "
                f"{cfg.model_path!r} — using random weights "
                "(fallback_to_router=True, safe to proceed)"
            )
        reasoner.eval()
        _trm_reasoner = reasoner
        return _trm_reasoner
    except Exception as _e:
        print(f"\u274c TRMReasoner init failed: {_e} — TRM disabled for this run")
        return None


def _run_trm_reasoner(
    reasoner: Any,
    text: str,
    routing_context: Any,
    relevant_domains: List[str],
) -> Optional[Dict[str, Any]]:
    """
    Run TRMReasoner on a single query and return a compact result dict.

    Returns None on any error so callers can safely ignore failures.

    Result keys
    -----------
    domain_probs        : list[float]  length == N_DOMAINS
    primary_domain_idx  : int          argmax of domain_probs
    primary_domain      : str          DOMAIN_LIST[primary_domain_idx]
    halt_confidence     : float        sigmoid(halt_logit)
    n_steps_taken       : int
    reranked_domains    : list[str]    relevant_domains re-ordered by TRM score
    """
    try:
        import torch as _t
        from mycelium.trm.trm_routing_trace_writer import DOMAIN_LIST as _DL

        # 1. token_ids  [1, context_len]
        ids = _trm_encode_query(text)
        token_ids = _t.tensor([ids], dtype=_t.long)

        # 2. spectral_vec  [1, N_DOMAINS]
        spec_scores = getattr(routing_context, "spectral_scores", None)
        sel_doms = list(getattr(routing_context, "selected_domains", []) or [])
        sv = _trm_spectral_vec(spec_scores, sel_doms)
        spectral_vec = _t.tensor([sv], dtype=_t.float32)

        # 3. predicate_family_id  [1]
        clf = str(getattr(routing_context, "classification", "UNKNOWN") or "UNKNOWN").upper()
        pred_id = _TRM_PRED_MAP.get(clf, _TRM_DEF_PRED)
        predicate_family_id = _t.tensor([pred_id], dtype=_t.long)

        # 4. initial_domain_probs  [1, N_DOMAINS]
        init_p = _trm_init_probs(sv)
        initial_domain_probs = _t.tensor([init_p], dtype=_t.float32)

        with _t.no_grad():
            out = reasoner(
                token_ids=token_ids,
                spectral_vec=spectral_vec,
                predicate_family_id=predicate_family_id,
                initial_domain_probs=initial_domain_probs,
            )

        domain_probs: List[float] = _t.softmax(out.domain_logits, dim=-1)[0].tolist()
        primary_domain_idx: int = int(out.primary_domain_idx[0].item())
        primary_domain: str = (
            _DL[primary_domain_idx]
            if 0 <= primary_domain_idx < len(_DL)
            else "unknown"
        )
        halt_conf: float = float(_t.sigmoid(out.halt_logit)[0].item())
        n_steps: int = int(getattr(out, "n_steps_taken", 1))

        # Re-rank relevant_domains by TRM domain_probs score.
        # If a domain is not in DOMAIN_LIST its TRM score is treated as 0.
        def _trm_score(d: str) -> float:
            idx = _trm_domain_to_idx(d)
            return domain_probs[idx] if 0 <= idx < _TRM_N_DOMAINS else 0.0

        reranked = sorted(relevant_domains, key=_trm_score, reverse=True)

        return {
            "domain_probs":       domain_probs,
            "primary_domain_idx": primary_domain_idx,
            "primary_domain":     primary_domain,
            "halt_confidence":    halt_conf,
            "n_steps_taken":      n_steps,
            "reranked_domains":   reranked,
        }
    except Exception as _re:
        print(f"\u26a0\ufe0f  TRMReasoner forward pass failed: {_re}")
        return None


def run_mycelium_workflow(
    sentences: Sequence[str],
    trace_id: Optional[str] = None,
    on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    reasoning_mode: ReasoningMode = "balanced",
) -> Tuple[List[Dict[str, Any]], WorkflowMetrics]:
    """
    Full Mycelium reasoning pipeline.

    Phase 6 (reasoning_mode):
        "fast"     -> dfs_max_depth=1, expert_top_k=1, phase3_passes=1
        "balanced" -> dfs_max_depth=2, expert_top_k=2, phase3_passes=2  (default)
        "deep"     -> dfs_max_depth=4, expert_top_k=3, phase3_passes=3

    Phase B wiring:
        CanonicalizeAndHash.process(text) is called per sentence to build
        real IRNodes with fully populated SemanticSignatures.

    Phase D wiring:
        TRMLens refines routing_context.  TRMEngine (owns GraphStore +
        MultiWorkerDFSLookup + PromotionPolicy) replaces the previous
        scattered inline plumbing:
            - trm_engine.persist(graph)          <- put+observe+promote
            - trm_engine.lookup(ir_nodes)        <- BY_HASH + equiv fallback
            - trm_engine.record_contradiction()  <- penalty+contest+revise
        DAGDecomposer is constructed with trm_engine.store so it writes
        into the same GraphStore instance.

    Phase D (Option 2) — TRMReasoner wiring:
        After MultiLensRouter.route() + TRMLens.refine(), TRMReasoner
        runs a neural forward pass over (token_ids, spectral_vec,
        predicate_family_id, initial_domain_probs) and produces:
            - domain_probs       : re-ranks relevant_domains
            - halt_confidence    : confidence in the routing decision
            - primary_domain_idx : TRM's top-1 domain prediction
        When fallback_to_router=True (default), a TRM failure or low
        halt_confidence degrades gracefully to MultiLensRouter order.
        TRMRoutingTraceWriter captures every routed sentence for training.

    Phase E wiring:
        ContradictionClassifier compares consecutive Phase 2 outputs.
        CONTESTED revision fires via trm_engine.record_contradiction().

    Phase F wiring:
        1. DSTFusion converts ConfidenceState -> DSTFrame.  The Pignistic
           scalar (to_net_confidence) drives graph confidence; m_unknown
           is surfaced in trm_lookup.
        2. MultiWorkerDFSLookup is injected into TRMEngine so the
           WorkerPool is shared across lookup() and the DAGDecomposer
           DFS calls.  Auto-selects single vs multi-worker by store size.
    """
    import uuid as _uuid
    import hashlib as _hashlib

    depth_cfg = get_depth_config(reasoning_mode)
    dfs_max_depth: int = depth_cfg["dfs_max_depth"]
    expert_top_k: int = depth_cfg["expert_top_k"]
    phase3_passes: int = depth_cfg["phase3_passes"]

    _wall_start = _time.monotonic()
    _request_id = trace_id or str(_uuid.uuid4())

    def _sse_callback(ev):
        if on_event is not None:
            on_event(ev.to_sse_dict())

    emitter: EventEmitter = make_emitter(
        request_id=_request_id,
        wall_start=_wall_start,
        on_public_event=_sse_callback,
        max_hz=10.0,
    )

    emitter.emit(
        phase_name="setting_up",
        message="Warming up model registry...",
        detail="Pre-flight: loading non-LLM model weights into registry",
        state="running",
        metadata={"reasoning_mode": reasoning_mode, "dfs_max_depth": dfs_max_depth},
    )

    print("\U0001f9e0 Pre-flight: loading non-LLM model weights into registry...")
    warmup(STARTUP_SPECS)
    _resident = loaded_models()
    print(
        f"\u2705 ModelRegistry warm -- {len(_resident)} model(s) resident: "
        f"{[k.split(':')[1] for k in _resident]}\n"
    )

    emitter.emit(
        phase_name="environment_ready",
        message="Environment ready",
        detail=f"{len(_resident)} model(s) resident",
        state="running",
        metadata={
            "model_count": len(_resident),
            "models": [k.split(":")[1] for k in _resident],
            "reasoning_mode": reasoning_mode,
        },
    )

    phase2_pipeline = Phase2Pipeline()
    phase3_pipeline = Phase3To5Pipeline()

    # ---------------------------------------------------------------
    # Phase D / F layer — single TRMEngine owns store + DFS + policy
    # ---------------------------------------------------------------
    _graph_store = GraphStore()
    _dfs_lookup  = MultiWorkerDFSLookup(_graph_store)
    trm_engine   = TRMEngine(store=_graph_store, dfs=_dfs_lookup)
    graph_store  = trm_engine.store
    trm_lens     = TRMLens()
    dag_decomposer = DAGDecomposer(
        graph_store=graph_store,
        max_depth=dfs_max_depth,
    )

    # ---------------------------------------------------------------
    # Phase D (Option 2) — TRMReasoner (lazy singleton, non-fatal)
    # ---------------------------------------------------------------
    trm_reasoner = _get_trm_reasoner()
    if trm_reasoner is not None:
        print("\u2705 TRMReasoner active (Option 2 wiring)")
    else:
        print("\u26a0\ufe0f  TRMReasoner unavailable — routing via MultiLensRouter only")

    # --- Phase B — shared canonicalizer (reuses SRL doc cache §5.1) ---
    _canonicalizer = CanonicalizeAndHash() if _CANON_AVAILABLE else None

    # --- Phase F — shared DST fusion engine ---
    _dst_fusion = DSTFusion() if _DST_AVAILABLE else None

    # --- Phase E — ContradictionClassifier ---
    contradiction_classifier = ContradictionClassifier() if _CONTRADICTION_AVAILABLE else None
    _prev_phase2_result: Optional[Any] = None

    from mycelium.pipeline.unified_expert_system import get_unified_expert_system, _unified_system
    _expert_msg = 'Setting up environment...' if _unified_system is None else 'Loading expert system...'
    emitter.emit(
        phase_name="graph_expert_init",
        message=_expert_msg,
        detail="K-Medoids + Calibration + OOD Detection",
        state="running",
    )
    print("Initializing unified expert system (K-Medoids + Calibration + OOD Detection)...")
    expert_system = get_unified_expert_system()
    registered_domains = set(expert_system.experts.keys())
    print(f"Initialized unified expert system with {len(registered_domains)} experts\n")

    emitter.emit(
        phase_name="graph_expert_init",
        message="Expert system ready",
        detail=f"{len(registered_domains)} expert domain(s) registered",
        state="running",
        metadata={"expert_count": len(registered_domains)},
    )

    emitter.emit(
        phase_name="graph_spectral_sync",
        message="Syncing spectral signatures...",
        detail="Checking for stale or missing .npy files",
        state="running",
    )
    print("Syncing spectral signatures with registered expert domains...")
    sig_manager = DynamicSignatureManager(signature_dir="signatures")
    spectral_analyzer = sig_manager.sync_signatures(registered_domains)
    _synced_domains = len(spectral_analyzer.get_available_domains())
    print(f"\u2705 Spectral signatures synced ({_synced_domains} domains loaded)\n")

    emitter.emit(
        phase_name="graph_spectral_sync",
        message="Spectral signatures synced",
        detail=f"{_synced_domains} domain signature(s) loaded",
        state="running",
        metadata={"synced_domains": _synced_domains},
    )

    router = MultiLensRouter(spectral_analyzer=spectral_analyzer)

    print("Initializing expert filter with automatic semantic clustering...")
    expert_filter = ExpertFilter(
        domain_list=list(registered_domains),
        use_auto_clustering=True,
        similarity_threshold=0.45,
    )
    print("\u2705 Expert filter initialized with auto-clustering\n")

    print("Initializing Layer0 question router...")
    question_router = QuestionRouter()
    print("\u2705 Layer0 router initialized\n")

    emitter.emit(
        phase_name="graph_router_ready",
        message="Routing layer ready",
        detail="MultiLens router + expert filter + Layer0 router initialised",
        state="running",
        metadata={"expert_filter_threshold": 0.45},
    )

    temporal_layer = TemporalLocalityLayer(max_size=50, time_window_hours=24)
    all_sentence_data: List[Dict[str, Any]] = []
    all_tags: deque = deque(maxlen=5000)
    metrics = WorkflowMetrics()

    for idx, text in enumerate(sentences, start=1):
        tags = extract_tags_llama(text)
        normalized_tags = normalize_tags(tags)
        timestamp = datetime.datetime.now().isoformat()
        temporal_layer.add_statement(text, normalized_tags, timestamp)

        emitter.emit(
            phase_name="graph_layer0",
            message="Classifying query...",
            detail=f"Sentence {idx}/{len(sentences)}: {text[:80]}",
            state="running",
            metadata={"sentence_index": idx, "sentence_count": len(sentences)},
        )

        layer0_result = question_router.route(text)
        metrics.layer0_routes[layer0_result.route] += 1

        if layer0_result.route != "REASONING_PIPELINE":
            if ENABLE_LOGGING and idx % LOG_SAMPLE_RATE == 0:
                print(f"\U0001f6ab Layer0 route: {layer0_result.route.upper()}\n")

            emitter.emit(
                phase_name="graph_layer0",
                message="Query handled by Layer 0",
                detail=f"Route: {layer0_result.route}",
                state="running",
                metadata={"layer0_route": layer0_result.route},
            )

            all_sentence_data.append(
                {
                    "sentence": text,
                    "tags": normalized_tags,
                    "timestamp": timestamp,
                    "layer0_routing": _to_jsonable(layer0_result),
                    "routing_context": {},
                    "phase2_result": {},
                    "phase3_result": {},
                    "expert_decision": {},
                    "expert_flag": "refused",
                    "selected_domain": "unknown",
                    "decision_confidence": 0.0,
                }
            )
            continue

        emitter.emit(
            phase_name="routing",
            message="Routing query through semantic lenses...",
            detail=f"{len(normalized_tags)} tag(s) extracted",
            state="running",
            metadata={"tag_count": len(normalized_tags)},
        )

        routing_context = router.route(text)
        routing_context = trm_lens.refine(routing_context)

        classification = getattr(routing_context, "classification", None)
        if classification:
            metrics.routing_classifications[classification] += 1

        emitter.emit(
            phase_name="graph_routing",
            message="Routing complete",
            detail=f"Classification: {classification}",
            state="running",
            metadata={"classification": str(classification) if classification else ""},
        )

        relevant_domains: List[str] = []
        for domain in getattr(routing_context, "selected_domains", []):
            domain_str = str(domain)
            if domain_str in registered_domains:
                relevant_domains.append(domain_str)
            else:
                normalized = expert_filter.normalize_domain(domain_str)
                if normalized:
                    relevant_domains.append(normalized)

        if not relevant_domains:
            for tag in normalized_tags:
                if tag in registered_domains:
                    relevant_domains.append(tag)
                else:
                    domain = expert_filter.normalize_domain(tag)
                    if domain:
                        relevant_domains.append(domain)

        if not relevant_domains:
            relevant_domains = list(registered_domains)
            if ENABLE_LOGGING and idx % LOG_SAMPLE_RATE == 0:
                print(
                    f'\u2139\ufe0f  No domain resolved from routing or tags for "{text[:60]}...". '
                    "Supplying all experts to reasoning pipeline.\n"
                )
            pre_check_result = expert_filter.filter_experts_by_tags(
                normalized_tags,
                expert_system,
                bert_manager=None,
                bert_domains=registered_domains,
            )
            if ENABLE_LOGGING and pre_check_result["missing_domains"]:
                print(
                    f"\u26a0\ufe0f  Pre-check (fallback path) -- missing domains: "
                    f"{pre_check_result['missing_domains']}\n"
                )

        relevant_domains = list(set(relevant_domains))

        # -------------------------------------------------------------------
        # Phase D (Option 2) — TRMReasoner: re-rank relevant_domains
        # -------------------------------------------------------------------
        trm_reasoner_result: Optional[Dict[str, Any]] = None
        if trm_reasoner is not None:
            trm_reasoner_result = _run_trm_reasoner(
                trm_reasoner, text, routing_context, relevant_domains
            )
            if trm_reasoner_result is not None:
                # Use TRM-reranked domain order
                relevant_domains = trm_reasoner_result["reranked_domains"]
                if ENABLE_LOGGING and idx % LOG_SAMPLE_RATE == 0:
                    print(
                        f"\U0001f9e0 TRMReasoner: primary={trm_reasoner_result['primary_domain']} "
                        f"halt_conf={trm_reasoner_result['halt_confidence']:.3f} "
                        f"steps={trm_reasoner_result['n_steps_taken']}"
                    )

        if expert_top_k < len(relevant_domains):
            relevant_domains = relevant_domains[:expert_top_k]

        filtered_experts = {
            domain: expert_system.experts[domain]
            for domain in relevant_domains
            if domain in expert_system.experts
        }

        emitter.emit(
            phase_name="graph_phase2",
            message="Running reasoning pipeline...",
            detail=f"{len(filtered_experts)} expert(s) active (top_k={expert_top_k})",
            state="running",
            metadata={
                "active_domains": list(filtered_experts.keys()),
                "expert_top_k": expert_top_k,
                "dfs_max_depth": dfs_max_depth,
                "trm_primary_domain": trm_reasoner_result["primary_domain"] if trm_reasoner_result else None,
                "trm_halt_confidence": trm_reasoner_result["halt_confidence"] if trm_reasoner_result else None,
            },
        )
        emitter.emit_heartbeat(idx % 5)

        phase2_result = phase2_pipeline.run(
            text,
            routing_context=routing_context,
            filtered_experts=filtered_experts,
            depth_config=depth_cfg,
        )

        # ------------------------------------------------------------------
        # Phase E — ContradictionClassifier on consecutive p2 outputs
        # ------------------------------------------------------------------
        contradiction_result: Optional[Dict[str, Any]] = None
        if contradiction_classifier is not None and _prev_phase2_result is not None:
            try:
                import types as _ct
                import hashlib as _ch

                def _p2_node(p2: Any, node_id: str) -> Any:
                    label = str(
                        getattr(p2, "final_decision", None)
                        or getattr(p2, "decision_label", None) or ""
                    )
                    canonical = label.lower().strip()
                    pred_family = str(
                        getattr(p2, "domain", None)
                        or getattr(p2, "selected_domain", None) or "UNKNOWN"
                    ).upper()
                    h = _ch.sha256(f"{canonical}|{pred_family}|0".encode()).hexdigest()
                    sig = _ct.SimpleNamespace(
                        semantic_hash=h, canonical_form=canonical,
                        predicate_family=pred_family, abstraction_level=0,
                        equivalence_family=[],
                    )
                    conf = float(getattr(p2, "decision_confidence", 0.5) or 0.5)
                    conf_state = _ct.SimpleNamespace(overall_confidence=conf)
                    t_state = _ct.SimpleNamespace(
                        type="UNKNOWN", start=None, end=None,
                        relative_relation=None, uncertainty=1.0,
                        historical_validity=True,
                    )
                    return _ct.SimpleNamespace(
                        id=node_id, semantic_signature=sig,
                        confidence_state=conf_state, temporal_state=t_state,
                        label=label,
                    )

                node_a = _p2_node(_prev_phase2_result, f"p2-prev-{idx - 1}")
                node_b = _p2_node(phase2_result, f"p2-curr-{idx}")
                c_edge = contradiction_classifier.classify(
                    node_a, node_b, context={"sentence_index": idx}
                )
                contradiction_result = {
                    "type": getattr(c_edge, "type", None),
                    "severity": getattr(c_edge, "severity", None),
                    "confidence": getattr(c_edge, "confidence", None),
                    "scope": getattr(c_edge, "scope", None),
                }
                _graph_id_prev = f"G-{_request_id[:8]}-{idx - 1:04d}"
                trm_engine.record_contradiction(_graph_id_prev, c_edge)
            except Exception as _ce:
                contradiction_result = {"error": str(_ce)}

        emitter.emit(
            phase_name="graph_unified_decision",
            message="Computing unified expert decision...",
            detail=f"{len(filtered_experts)} expert(s) evaluated",
            state="running",
            metadata={"active_domains": list(filtered_experts.keys())},
        )

        expert_decision = expert_system.unified_decision_analysis(
            text,
            routing_result=routing_context,
            filtered_experts=filtered_experts,
            return_legacy_dict=False,
            depth_config=depth_cfg,
        )
        expert_decision = combine_routing_and_expert_decisions(
            routing_context, expert_decision,
        )

        # ------------------------------------------------------------------
        # Phase B + D + F — build real IRNodes, DAG, DST, persist via TRMEngine
        # ------------------------------------------------------------------
        trm_lookup_result: Optional[Dict[str, Any]] = None
        dag_graph = None
        try:
            _graph_id = f"G-{_request_id[:8]}-{idx:04d}"
            _pred_family = str(
                getattr(routing_context, "classification", "UNKNOWN") or "UNKNOWN"
            )
            _raw_conf = float(
                getattr(expert_decision, "expert_confidence", 0.5) or 0.5
            )

            # Phase F — DST scalar
            _dst_conf: float = _raw_conf
            _dst_m_unknown: Optional[float] = None
            if _dst_fusion is not None:
                try:
                    import types as _t
                    _cs_stub = _t.SimpleNamespace(
                        overall_confidence=_raw_conf,
                        semantic_confidence=_raw_conf,
                        structural_confidence=_raw_conf,
                        epistemic_confidence=_raw_conf,
                        evidence_confidence=_raw_conf,
                        temporal_confidence=_raw_conf,
                        contradiction_penalty=0.0,
                        aggregation_method="dst",
                    )
                    _dst_frame = _dst_fusion.from_confidence_state(
                        _cs_stub, label=_graph_id
                    )
                    _dst_conf = DSTFusion.to_net_confidence(_dst_frame)
                    _dst_m_unknown = _dst_frame.m_unknown
                except Exception:
                    pass

            # Phase B — build real IRNodes
            _ir_nodes = []
            if _canonicalizer is not None:
                try:
                    _ir_nodes = _canonicalizer.process(
                        text,
                        abstraction_level=0,
                        source=_graph_id,
                    )
                except Exception:
                    _ir_nodes = []

            if not _ir_nodes:
                import types as _types
                _text_hash = _hashlib.sha256(text.encode()).hexdigest()[:16]
                _sig = _types.SimpleNamespace(
                    semantic_hash=_text_hash,
                    canonical_form=text.lower().strip(),
                    predicate_family=_pred_family,
                    equivalence_family=normalized_tags,
                )
                _ir_nodes = [_types.SimpleNamespace(
                    id=f"{_graph_id}-n0",
                    semantic_signature=_sig,
                )]

            root_node = _ir_nodes[0]
            try:
                dag_graph = dag_decomposer.decompose(
                    root_node, request_id=_graph_id
                )
            except Exception as _dag_exc:
                dag_graph = None

            if dag_graph is not None:
                _persisted, obs_count, _target_state = trm_engine.persist(dag_graph)
            else:
                import types as _types
                _conf_stub = _types.SimpleNamespace(
                    overall_confidence=_dst_conf,
                    contradiction_penalty=0.0,
                )
                _graph_stub = _types.SimpleNamespace(
                    graph_id=_graph_id, nodes=_ir_nodes, edges=[],
                    state="DRAFT", version="v1",
                    confidence_state=_conf_stub,
                    fingerprint=None, ontology_version="1.0",
                    created_at=timestamp, updated_at=timestamp,
                    parent_graph_id=None,
                )
                _persisted, obs_count, _target_state = trm_engine.persist(_graph_stub)

            _lookup = trm_engine.lookup(_ir_nodes)

            trm_lookup_result = {
                "found": _lookup.found,
                "strategy_used": _lookup.strategy_used,
                "query_hash": _lookup.query_hash,
                "explanation": _lookup.explanation,
                "primary": _lookup.primary,
                "candidate_count": len(_lookup.candidates),
                "dfs_max_depth_cfg": dfs_max_depth,
                "obs_count": obs_count,
                "promoted_to": _target_state,
                "dst_conf": round(_dst_conf, 4),
                "dst_m_unknown": round(_dst_m_unknown, 4) if _dst_m_unknown is not None else None,
                "dag_nodes": len(dag_graph.nodes) if dag_graph is not None else 0,
                "dag_edges": len(dag_graph.edges) if dag_graph is not None else 0,
                "phase_b_nodes": len(_ir_nodes),
                "multi_worker_dfs": True,
                "trm_engine": True,
                # Phase D (Option 2) — TRMReasoner fields surfaced in lookup result
                "trm_reasoner_active": trm_reasoner is not None,
                "trm_primary_domain": trm_reasoner_result["primary_domain"] if trm_reasoner_result else None,
                "trm_primary_domain_idx": trm_reasoner_result["primary_domain_idx"] if trm_reasoner_result else None,
                "halt_confidence": trm_reasoner_result["halt_confidence"] if trm_reasoner_result else None,
                "n_steps_taken": trm_reasoner_result["n_steps_taken"] if trm_reasoner_result else None,
            }
        except Exception as _trm_exc:
            trm_lookup_result = {"found": False, "error": str(_trm_exc)}

        # -------------------------------------------------------------------
        # Phase D (Option 2) — TRMRoutingTraceWriter: capture training sample
        # -------------------------------------------------------------------
        selected_domain_prelim = (
            expert_decision.selected_experts[0]
            if getattr(expert_decision, "selected_experts", None)
            else "unknown"
        )
        if trm_trace_writer is not None and selected_domain_prelim != "unknown":
            trm_trace_writer.record(
                text=text,
                routing_context=routing_context,
                selected_domain=selected_domain_prelim,
                trm_lookup_result=trm_lookup_result,
            )

        emitter.emit(
            phase_name="expert_decision",
            message="Expert decision reached",
            detail=(
                f"{getattr(expert_decision, 'decision_type', 'N/A')} · "
                f"confidence {float(getattr(expert_decision, 'expert_confidence', 0.0)):.2f}"
            ),
            state="running",
            metadata={
                "decision_type": getattr(expert_decision, "decision_type", ""),
                "selected_experts": list(
                    getattr(expert_decision, "selected_experts", []) or []
                ),
                "confidence": float(getattr(expert_decision, "expert_confidence", 0.0)),
                "trm_lookup_found": trm_lookup_result.get("found", False) if trm_lookup_result else False,
                "contradiction_type": contradiction_result.get("type") if contradiction_result else None,
                "dst_m_unknown": trm_lookup_result.get("dst_m_unknown") if trm_lookup_result else None,
                "trm_primary_domain": trm_lookup_result.get("trm_primary_domain") if trm_lookup_result else None,
                "halt_confidence": trm_lookup_result.get("halt_confidence") if trm_lookup_result else None,
            },
        )

        p2_fdr = _adapt_phase2_to_p3(phase2_result, original_text=text)
        phase2_extra_metadata: Dict[str, Any] = dict(p2_fdr.metadata or {})

        phase3_input = _adapt_unified_to_p3(
            expert_decision, original_text=text,
            phase2_metadata=phase2_extra_metadata,
        )

        if phase3_input.metadata is None:
            phase3_input.metadata = {}
        if trm_lookup_result is not None:
            phase3_input.metadata["trm_lookup"] = trm_lookup_result
        if contradiction_result is not None:
            phase3_input.metadata["contradiction"] = contradiction_result

        emitter.emit(
            phase_name="graph_phase3",
            message="Running validation pipeline...",
            detail=f"Phase 3-5: action execution + feedback (passes={phase3_passes})",
            state="running",
            metadata={"phase3_passes": phase3_passes},
        )

        phase3_result = phase3_pipeline.run_complete_pipeline(phase3_input)
        _prev_phase2_result = phase2_result

        metrics.expert_decisions[expert_decision.decision_type] += 1
        for dom in getattr(expert_decision, "selected_experts", []) or []:
            metrics.domains[dom] += 1

        decision_to_flag = {
            "USE_EXISTING_EXPERT": "use_existing_expert",
            "CREATE_NEW_PATCH": "create_new_patch",
            "CREATE_NEW_EXPERT": "create_new_expert",
        }
        flag = decision_to_flag.get(expert_decision.decision_type, "create_new_expert")
        selected_domain = (
            expert_decision.selected_experts[0]
            if expert_decision.selected_experts else "unknown"
        )
        confidence = float(getattr(expert_decision, "expert_confidence", 0.0))

        is_patch_decision = expert_decision.decision_type == "CREATE_NEW_PATCH"
        import uuid as _uuid
        sentence_trace_id = trace_id if (idx == 1 and trace_id) else str(_uuid.uuid4())

        if is_patch_decision:
            phase_latencies: Dict[str, float] = (
                _to_jsonable(phase3_result).get("phase_latencies", {})
                if isinstance(phase3_result, (dict,)) or hasattr(phase3_result, "__dict__")
                else {}
            )
            patch_logger.log_query(
                trace_id=sentence_trace_id,
                query=text,
                tags=normalized_tags,
                routing_classification=str(classification) if classification else "",
                phase_latencies_ms=phase_latencies,
            )
            if ENABLE_LOGGING:
                print(
                    f"\U0001f4dd CREATE_NEW_PATCH -- logged query to batch "
                    f"(trace_id={sentence_trace_id}). "
                    "Proceeding through reasoning pipeline.\n"
                )

        final_answer: str = ""
        p3_dict = _to_jsonable(phase3_result) if phase3_result else {}
        for _key in ("final_answer", "answer", "response", "output", "text"):
            val = p3_dict.get(_key)
            if val and isinstance(val, str):
                final_answer = val
                break
        if not final_answer:
            action = p3_dict.get("action_result") or {}
            for _key in ("final_answer", "answer", "response", "output", "text"):
                val = action.get(_key) if isinstance(action, dict) else None
                if val and isinstance(val, str):
                    final_answer = val
                    break

        if is_patch_decision and final_answer:
            patch_logger.fill_response(
                trace_id=sentence_trace_id,
                response=final_answer,
                phase_latencies_ms=phase_latencies,
            )

        emitter.emit(
            phase_name="graph_clustering",
            message="Updating tag cluster model...",
            detail=f"Sentence {idx}/{len(sentences)} complete",
            state="running",
            metadata={
                "sentence_index": idx,
                "flag": flag,
                "selected_domain": selected_domain,
                "confidence": round(confidence, 4),
                "reasoning_mode": reasoning_mode,
            },
        )

        all_sentence_data.append(
            {
                "sentence": text,
                "tags": normalized_tags,
                "timestamp": timestamp,
                "layer0_routing": _to_jsonable(layer0_result),
                "routing_context": _to_jsonable(routing_context),
                "phase2_result": _to_jsonable(phase2_result),
                "phase3_result": _to_jsonable(phase3_result),
                "expert_decision": _to_jsonable(expert_decision),
                "expert_flag": flag,
                "selected_domain": selected_domain,
                "decision_confidence": confidence,
                "trace_id": sentence_trace_id,
                "reasoning_mode": reasoning_mode,
                "trm_lookup": trm_lookup_result,
                "contradiction": contradiction_result,
                "trm_stats": trm_engine.stats(),
                "trm_reasoner": trm_reasoner_result,
            }
        )
        all_tags.extend(normalized_tags)

        if ENABLE_LOGGING and idx % LOG_SAMPLE_RATE == 0:
            _stats = trm_engine.stats()
            _trace_counts = trm_trace_writer.counts() if trm_trace_writer else {}
            print(
                "Sentence: {sent}\nTags: {tags}\nTimestamp: {ts}\n"
                "Unified Decision: {flag} (Domain: {dom}, Confidence: {conf:.3f})\n"
                "Reasoning Mode: {mode} (dfs_depth={dfs}, top_k={k})\n"
                "TRM: found={trm_found} obs={obs} promoted={promoted}\n"
                "TRMReasoner: active={trm_r_active} primary={trm_primary} "
                "halt_conf={halt_conf} steps={steps}\n"
                "TraceWriter: train={tw_train} eval={tw_eval}\n"
                "TRMEngine stats: graphs={g} obs_total={o} states={s}\n"
                "DST: conf={dst_conf} m_unknown={m_unk}\n"
                "DAG: nodes={dag_n} edges={dag_e} phase_b_nodes={pb_n}\n"
                "Contradiction: type={c_type} severity={c_sev}\n"
                "MultiWorkerDFS: active\n".format(
                    sent=text, tags=normalized_tags, ts=timestamp,
                    flag=flag, dom=selected_domain, conf=confidence,
                    mode=reasoning_mode, dfs=dfs_max_depth, k=expert_top_k,
                    trm_found=trm_lookup_result.get("found") if trm_lookup_result else False,
                    obs=trm_lookup_result.get("obs_count") if trm_lookup_result else 0,
                    promoted=trm_lookup_result.get("promoted_to") if trm_lookup_result else None,
                    trm_r_active=trm_reasoner is not None,
                    trm_primary=trm_lookup_result.get("trm_primary_domain") if trm_lookup_result else "N/A",
                    halt_conf=(
                        f"{trm_lookup_result.get('halt_confidence'):.3f}"
                        if trm_lookup_result and trm_lookup_result.get("halt_confidence") is not None
                        else "N/A"
                    ),
                    steps=trm_lookup_result.get("n_steps_taken") if trm_lookup_result else "N/A",
                    tw_train=_trace_counts.get("train", 0),
                    tw_eval=_trace_counts.get("eval", 0),
                    g=_stats["total_graphs"], o=_stats["total_observations"],
                    s=_stats["state_distribution"],
                    dst_conf=trm_lookup_result.get("dst_conf") if trm_lookup_result else "N/A",
                    m_unk=trm_lookup_result.get("dst_m_unknown") if trm_lookup_result else "N/A",
                    dag_n=trm_lookup_result.get("dag_nodes") if trm_lookup_result else 0,
                    dag_e=trm_lookup_result.get("dag_edges") if trm_lookup_result else 0,
                    pb_n=trm_lookup_result.get("phase_b_nodes") if trm_lookup_result else 0,
                    c_type=contradiction_result.get("type") if contradiction_result else "N/A",
                    c_sev=contradiction_result.get("severity") if contradiction_result else "N/A",
                )
            )

    with open("evaluation_data/sentence_tags.json", "w", encoding="utf-8") as f:
        json.dump(all_sentence_data, f, indent=4, cls=NumpyEncoder)

    expert_evaluation_results: Dict[str, Any] = {
        "evaluation_timestamp": datetime.datetime.now().isoformat(),
        "sentences_evaluated": len(all_sentence_data),
        "flag_summary": {},
        "detailed_results": all_sentence_data,
        "trm_final_stats": trm_engine.stats(),
        "trm_trace_counts": trm_trace_writer.counts() if trm_trace_writer else {},
    }
    for entry in all_sentence_data:
        flag = entry["expert_flag"]
        expert_evaluation_results["flag_summary"][flag] = (
            expert_evaluation_results["flag_summary"].get(flag, 0) + 1
        )

    with open("evaluation_data/expert_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(expert_evaluation_results, f, indent=4, cls=NumpyEncoder)
    print("Expert evaluation results saved to expert_evaluation_results.json")

    unique_tags = list(set(all_tags))
    if unique_tags:
        embeddings = embed_tags_transformer(unique_tags, model_name="all-mpnet-base-v2")
        clusters = cluster_tags_transformer(
            unique_tags, embeddings, similarity_threshold=0.5
        )
    else:
        embeddings = np.zeros((0, 0))
        clusters = {}

    clustering_data = {"tags": unique_tags, "clusters": clusters}
    with open("evaluation_data/tag_clusters_transformer.json", "w", encoding="utf-8") as f:
        json.dump(clustering_data, f, indent=4)
    print("Clusters saved to tag_clusters_transformer.json:", clusters)

    recent_statements = temporal_layer.get_recent_statements(time_limit_hours=1)
    spatial_analysis = analyze_spatial_locality(recent_statements, clusters)
    patch_assignment = assign_domain_patch(spatial_analysis, similarity_threshold=0.3)
    frequent_tags = temporal_layer.get_frequent_tags(min_frequency=2)
    temporal_analysis_data = {
        "recent_statements_count": len(recent_statements),
        "spatial_analysis": spatial_analysis,
        "patch_assignment": patch_assignment,
        "frequent_tags": frequent_tags,
        "analysis_timestamp": datetime.datetime.now().isoformat(),
    }
    with open("evaluation_data/temporal_analysis.json", "w", encoding="utf-8") as f:
        json.dump(temporal_analysis_data, f, indent=4)
    print("Temporal analysis saved to temporal_analysis.json")

    emitter.flush()
    return all_sentence_data, metrics


def _is_lfs_pointer(csv_path: str) -> bool:
    with open(csv_path, "r", encoding="utf-8") as csv_file:
        first_line = csv_file.readline().strip()
        return first_line.startswith("version https://git-lfs.github.com/spec/v1")


def _sample_column(df: pd.DataFrame, column: str, count: int, seed: int) -> list:
    if column not in df.columns:
        return []
    return df[column].dropna().sample(count, random_state=seed).tolist()


def get_random_samples():
    med_path = "dummy_models/Medical/medical_dataset.csv"
    med_samples: List[str] = []
    if not _is_lfs_pointer(med_path):
        med_df = pd.read_csv(med_path)
        med_samples = _sample_column(med_df, "sentence", 3, 42)

    music_path = "dummy_models/Music/music_classification_dataset.csv"
    music_samples: List[str] = []
    if not _is_lfs_pointer(music_path):
        music_df = pd.read_csv(music_path)
        music_samples = _sample_column(music_df, "sentence", 3, 43)

    phys_path = "dummy_models/Physics/physics_data.csv"
    phys_samples: List[str] = []
    if not _is_lfs_pointer(phys_path):
        phys_df = pd.read_csv(phys_path)
        phys_samples = _sample_column(phys_df, "Comment", 4, 44)

    all_samples = med_samples + music_samples + phys_samples
    if not all_samples:
        all_samples = [
            "Metastatic carcinoma requires systemic chemotherapy.",
            "Quantum entanglement occurs when particles remain correlated.",
            "Covalent bonds form when atoms share electrons.",
            "The stock market crashed in 1929.",
            "Photosynthesis occurs in plants.",
            "Red car with 700cc engine.",
            "High torque low noise.",
            "Earth orbits the Sun.",
            "Glossy metallic red finish.",
            "Stellar evolution theory.",
        ]
    random.shuffle(all_samples)
    return all_samples


if __name__ == "__main__":
    sentences = get_random_samples()
    run_mycelium_workflow(sentences)
