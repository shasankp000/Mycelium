import json
import os as _os
import datetime
import time as _time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path as _Path
from typing import Callable, List, Dict, Any, Optional, Sequence, Tuple, cast

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
    reload_domain_ontology,
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
from mycelium.pipeline import config_loader as _cfg
from mycelium.pipeline.api_models import ReasoningMode, get_depth_config
from mycelium.trm.integration import TRMLens
from mycelium.trm.graph_store import GraphStore
from mycelium.trm.trm_engine import TRMEngine
from mycelium.trm.multi_worker_dfs import MultiWorkerDFSLookup
from mycelium.reasoning.dag_decomposer import DAGDecomposer

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
    from mycelium.trm.trm_ood_fallback import TRMOODFallback, TRMOODHead

    _TRM_AVAILABLE = True
except Exception as _trm_import_err:
    _torch = None
    TRMReasoner = None
    TRMConfig = None
    TRMOODFallback = None
    TRMOODHead = None
    trm_trace_writer = None
    _trm_encode_query = None
    _trm_spectral_vec = None
    _trm_domain_to_idx = None
    _trm_init_probs = None
    _TRM_PRED_MAP = {}
    _TRM_DEF_PRED = 0
    _TRM_N_DOMAINS = 0
    _TRM_AVAILABLE = False
try:
    from mycelium.contradiction.classifier import ContradictionClassifier

    _CONTRADICTION_AVAILABLE = True
except Exception:
    ContradictionClassifier = None
    _CONTRADICTION_AVAILABLE = False
try:
    from mycelium.canonicalization.semantic_hash_pipeline import CanonicalizeAndHash

    _CANON_AVAILABLE = True
except Exception:
    CanonicalizeAndHash = None
    _CANON_AVAILABLE = False
try:
    from mycelium.fusion.dst_fusion import (
        DSTFusion,
        ConfidenceStateFusion,
        ConflictError,
    )

    _DST_AVAILABLE = True
except Exception:
    DSTFusion = None
    ConfidenceStateFusion = None
    ConflictError = None
    _DST_AVAILABLE = False

_GRAPH_STORE_DIR: str = _cfg.graph_store_persistence_dir()
_GRAPH_STORE_DECAY_ON_LOAD: bool = _cfg.graph_store_run_decay_on_load()
_os.makedirs(_GRAPH_STORE_DIR, exist_ok=True)

_TRM_CHECKPOINT_PATH: str = str(
    _Path(__file__).resolve().parents[1] / "trm" / "trm_checkpoints" / "trm_latest.pt"
)


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        return super().default(o)


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if is_dataclass(obj) and not isinstance(obj, type):
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
        "original_text",
        "input_text",
        "query",
        "sentence",
        "action_details",
        "expert_predictions",
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

    raw_conf = _get("confidence", "expert_confidence", default=0.5)
    p2_confidence = float(raw_conf if raw_conf is not None else 0.5)

    return P3FinalDecisionResult(
        decision=_get("decision_label", "prediction", "final_decision", "decision"),
        confidence=p2_confidence,
        reasoning=reasoning_str,
        action=_get(
            "action_type", "action", "recommended_action", default="use_existing"
        ),
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
    layer0_routes: Counter = field(default_factory=Counter)
    routing_classifications: Counter = field(default_factory=Counter)
    expert_decisions: Counter = field(default_factory=Counter)
    domains: Counter = field(default_factory=Counter)

    def to_dict(self) -> Dict[str, Dict[str, int]]:
        return {
            "layer0_routes": dict(self.layer0_routes),
            "routing_classifications": dict(self.routing_classifications),
            "expert_decisions": dict(self.expert_decisions),
            "domains": dict(self.domains),
        }


_trm_reasoner: Optional[Any] = None


def _get_trm_reasoner() -> Optional[Any]:
    global _trm_reasoner
    if not _TRM_AVAILABLE:
        return None
    if _trm_reasoner is not None:
        return _trm_reasoner
    try:
        assert TRMConfig is not None and TRMReasoner is not None and _torch is not None
        cfg = cast(Any, TRMConfig)()
        cfg.model_path = _TRM_CHECKPOINT_PATH
        reasoner = cast(Any, TRMReasoner)(cfg)
        if _os.path.isfile(cfg.model_path):
            ckpt = cast(Any, _torch).load(
                cfg.model_path, map_location="cpu", weights_only=False
            )
            if isinstance(ckpt, dict) and "model_state" in ckpt:
                state = ckpt["model_state"]
                _step = ckpt.get("step", "?")
                print(
                    f"\u2705 TRMReasoner: loaded checkpoint from "
                    f"{cfg.model_path} (step={_step})"
                )
            else:
                state = ckpt
                print(
                    f"\u2705 TRMReasoner: loaded legacy weights from {cfg.model_path}"
                )
            reasoner.load_state_dict(state)
        else:
            print(
                f"\u26a0\ufe0f  TRMReasoner: no checkpoint at {cfg.model_path!r} \u2014 "
                "using random weights (fallback_to_router=True, safe to proceed)"
            )
        reasoner.eval()
        _trm_reasoner = reasoner
        return _trm_reasoner
    except Exception as _e:
        print(f"\u274c TRMReasoner init failed: {_e} \u2014 TRM disabled for this run")
        return None


def _sanitize_spectral_scores(spectral_scores: Any) -> Any:
    if spectral_scores is None:
        return None
    if isinstance(spectral_scores, dict):
        cleaned = {}
        for k, v in spectral_scores.items():
            try:
                cleaned[k] = float(v)
            except (TypeError, ValueError):
                pass
        return cleaned if cleaned else None
    if isinstance(spectral_scores, (list, tuple, np.ndarray)):
        for v in spectral_scores:
            try:
                float(v)
            except (TypeError, ValueError):
                return None
        return spectral_scores
    return spectral_scores


def _run_trm_reasoner(
    reasoner: Any,
    text: str,
    routing_context: Any,
    relevant_domains: List[str],
) -> Optional[Dict[str, Any]]:
    try:
        if (
            _trm_encode_query is None
            or _trm_spectral_vec is None
            or _trm_domain_to_idx is None
            or _trm_init_probs is None
        ):
            return None
        trm_domain_to_idx = _trm_domain_to_idx
        import torch as _t
        from mycelium.trm.trm_routing_trace_writer import DOMAIN_LIST as _DL

        ids = cast(Any, _trm_encode_query)(text)
        token_ids = _t.tensor([ids], dtype=_t.long)

        raw_spec_scores = getattr(routing_context, "spectral_scores", None)
        spec_scores = _sanitize_spectral_scores(raw_spec_scores)
        sel_doms = list(getattr(routing_context, "selected_domains", []) or [])
        sv = cast(Any, _trm_spectral_vec)(spec_scores, sel_doms)
        spectral_vec = _t.tensor([sv], dtype=_t.float32)

        clf = str(
            getattr(routing_context, "classification", "UNKNOWN") or "UNKNOWN"
        ).upper()
        pred_id = _TRM_PRED_MAP.get(clf, _TRM_DEF_PRED)
        predicate_family_id = _t.tensor([pred_id], dtype=_t.long)

        init_p = cast(Any, _trm_init_probs)(sv)
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
            _DL[primary_domain_idx] if 0 <= primary_domain_idx < len(_DL) else "unknown"
        )
        halt_conf: float = float(out.halt_confidence[0].item())
        n_steps: int = int(getattr(out, "n_steps_taken", 1))

        def _trm_score(d: str) -> float:
            idx = cast(Any, trm_domain_to_idx)(d)
            return domain_probs[idx] if 0 <= idx < _TRM_N_DOMAINS else 0.0

        reranked = sorted(relevant_domains, key=_trm_score, reverse=True)

        return {
            "domain_probs": domain_probs,
            "primary_domain_idx": primary_domain_idx,
            "primary_domain": primary_domain,
            "halt_confidence": halt_conf,
            "n_steps_taken": n_steps,
            "reranked_domains": reranked,
            "_trm_output": out,
            "_spectral_tensor": spectral_vec,
        }
    except Exception as _re:
        print(f"\u26a0\ufe0f  TRMReasoner forward pass failed: {_re}")
        return None


# ---------------------------------------------------------------------------
# Shadow signal extraction helper
# ---------------------------------------------------------------------------

def _extract_shadow_signal(routing_context: Any) -> Optional[Dict[str, Any]]:
    sig = getattr(routing_context, "shadow_signal", None)
    if sig is None and isinstance(routing_context, dict):
        sig = routing_context.get("shadow_signal")
    if sig is None:
        return None
    if isinstance(sig, dict):
        return sig
    if hasattr(sig, "as_dict"):
        return sig.as_dict()
    return None


def run_mycelium_workflow(
    sentences: Sequence[str],
    trace_id: Optional[str] = None,
    on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    reasoning_mode: ReasoningMode = "smart",
) -> Tuple[List[Dict[str, Any]], WorkflowMetrics]:
    import uuid as _uuid
    import hashlib as _hashlib

    depth_cfg = get_depth_config(reasoning_mode)
    dfs_max_depth: int = depth_cfg["dfs_max_depth"]
    expert_top_k: int = depth_cfg["expert_top_k"]
    phase3_passes: int = depth_cfg["phase3_passes"]
    trm_threshold: float = float(depth_cfg["trm_threshold"])

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
        metadata={
            "reasoning_mode": reasoning_mode,
            "dfs_max_depth": dfs_max_depth,
            "trm_threshold": trm_threshold,
        },
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

    _graph_store = GraphStore(
        persistence_path=_GRAPH_STORE_DIR,
        run_decay_on_load=_GRAPH_STORE_DECAY_ON_LOAD,
    )
    _dfs_lookup = MultiWorkerDFSLookup(_graph_store)
    trm_engine = TRMEngine(store=_graph_store, dfs=_dfs_lookup)
    graph_store = trm_engine.store
    trm_lens = TRMLens()
    dag_decomposer = DAGDecomposer(
        graph_store=graph_store,
        max_depth=dfs_max_depth,
    )

    trm_reasoner = _get_trm_reasoner()
    if trm_reasoner is not None:
        print("\u2705 TRMReasoner active (Option 2 wiring)")
    else:
        print(
            "\u26a0\ufe0f  TRMReasoner unavailable \u2014 routing via MultiLensRouter only"
        )

    _ood_fallback: Optional[Any] = None

    _canonicalizer = (
        cast(Any, CanonicalizeAndHash)()
        if _CANON_AVAILABLE and CanonicalizeAndHash is not None
        else None
    )
    _dst_fusion = (
        cast(Any, DSTFusion)() if _DST_AVAILABLE and DSTFusion is not None else None
    )
    contradiction_classifier = (
        cast(Any, ContradictionClassifier)()
        if _CONTRADICTION_AVAILABLE and ContradictionClassifier is not None
        else None
    )
    _prev_phase2_result: Optional[Any] = None

    from mycelium.pipeline.unified_expert_system import (
        get_unified_expert_system,
        _unified_system,
    )

    _expert_msg = (
        "Setting up environment..."
        if _unified_system is None
        else "Loading expert system..."
    )
    emitter.emit(
        phase_name="graph_expert_init",
        message=_expert_msg,
        detail="K-Medoids + Calibration + OOD Detection",
        state="running",
    )
    print(
        "Initializing unified expert system (K-Medoids + Calibration + OOD Detection)..."
    )
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

    if trm_reasoner is not None and TRMOODFallback is not None:
        try:
            if TRMConfig is None or TRMOODHead is None:
                raise RuntimeError("TRMConfig/TRMOODHead unavailable")
            _trm_cfg = cast(Any, TRMConfig)()
            _ood_head = cast(Any, TRMOODHead)(
                hidden_size=_trm_cfg.hidden_size,
                bottleneck=_trm_cfg.ood_head_hidden or None,
            )
            _ood_fallback = TRMOODFallback(
                cfg=_trm_cfg,
                ood_head=_ood_head,
                multi_lens_router=router,
                canonicalizer=_canonicalizer,
                dag_decomposer=dag_decomposer,
                contradiction_classifier=contradiction_classifier,
            )
            print("\u2705 TRMOODFallback active (heuristic + OODHead)")
        except Exception as _oodf_err:
            print(
                f"\u26a0\ufe0f  TRMOODFallback init failed: {_oodf_err} \u2014 OOD fallback disabled"
            )
            _ood_fallback = None

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
                    "shadow_signal": None,
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

        # ----------------------------------------------------------------
        # Shadow domain handling
        # ----------------------------------------------------------------
        shadow_signal_dict = _extract_shadow_signal(routing_context)
        if shadow_signal_dict is not None:
            _shadow_status = shadow_signal_dict.get("status", "")
            _shadow_id     = shadow_signal_dict.get("shadow_id", "?")
            _shadow_evid   = shadow_signal_dict.get("evidence", 0)
            _shadow_tokens = shadow_signal_dict.get("top_tokens", [])

            if _shadow_status == "PROMOTE_TO_EXPERT":
                print(
                    f"\U0001f7e1 ShadowDomain PROMOTE_TO_EXPERT: "
                    f"{_shadow_id} \u2014 evidence={_shadow_evid} "
                    f"top_tokens={_shadow_tokens[:6]}"
                )
                emitter.emit(
                    phase_name="promote_shadow_domain",
                    message=f"Shadow domain ready for promotion: {_shadow_id}",
                    detail=(
                        f"evidence={_shadow_evid}  "
                        f"top_tokens={_shadow_tokens[:6]}"
                    ),
                    state="done",
                    metadata=shadow_signal_dict,
                )
                try:
                    reload_domain_ontology()
                except Exception as _rdo_err:
                    print(f"\u26a0\ufe0f  reload_domain_ontology failed: {_rdo_err}")
            else:
                print(
                    f"\U0001f7e4 ShadowDomain accumulating: "
                    f"{_shadow_id} evidence={_shadow_evid} "
                    f"tokens={_shadow_tokens[:4]}"
                )
        # ----------------------------------------------------------------

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

        # ----------------------------------------------------------------
        # graph:dfs_step  — one event per domain the router surfaced.
        # Emitted here, after relevant_domains is final, so the frontend
        # graph builder has the complete explored set before Phase 2 begins.
        # ----------------------------------------------------------------
        _fused_scores: Dict[str, float] = dict(
            getattr(routing_context, "fusion_scores", {}) or {}
        )
        for _step_idx, _domain in enumerate(relevant_domains):
            emitter.emit(
                phase_name="graph:dfs_step",
                message=f"Domain explored: {_domain}",
                detail=f"step {_step_idx + 1}/{len(relevant_domains)}",
                state="running",
                metadata={
                    "domain": _domain,
                    "step_index": _step_idx,
                    "total_steps": len(relevant_domains),
                    "fused_score": round(_fused_scores.get(_domain, 0.0), 4),
                    "dfs_max_depth": dfs_max_depth,
                    "classification": str(classification) if classification else "",
                },
            )
        # ----------------------------------------------------------------

        trm_reasoner_result: Optional[Dict[str, Any]] = None
        ood_fallback_result: Optional[Dict[str, Any]] = None
        should_escalate: bool = True

        if trm_reasoner is not None:
            trm_reasoner_result = _run_trm_reasoner(
                trm_reasoner, text, routing_context, relevant_domains
            )
            if trm_reasoner_result is not None:
                relevant_domains = trm_reasoner_result["reranked_domains"]
                halt_conf: float = trm_reasoner_result["halt_confidence"]
                should_escalate = halt_conf < trm_threshold

                if ENABLE_LOGGING and idx % LOG_SAMPLE_RATE == 0:
                    print(
                        f"\U0001f9e0 TRMReasoner: primary={trm_reasoner_result['primary_domain']} "
                        f"halt_conf={halt_conf:.3f} "
                        f"threshold={trm_threshold:.2f} "
                        f"should_escalate={should_escalate} "
                        f"steps={trm_reasoner_result['n_steps_taken']}"
                    )

                emitter.emit(
                    phase_name="graph_trm_decision",
                    message="TRM routing decision",
                    detail=(
                        f"halt_conf={halt_conf:.3f} threshold={trm_threshold:.2f} "
                        f"escalate={should_escalate}"
                    ),
                    state="running",
                    metadata={
                        "primary_domain": trm_reasoner_result["primary_domain"],
                        "halt_confidence": halt_conf,
                        "trm_threshold": trm_threshold,
                        "should_escalate": should_escalate,
                        "dfs_max_depth": dfs_max_depth,
                        "reasoning_mode": reasoning_mode,
                        "n_steps_taken": trm_reasoner_result["n_steps_taken"],
                    },
                )

                if _ood_fallback is not None:
                    _trm_out = trm_reasoner_result["_trm_output"]
                    _spec_t = trm_reasoner_result["_spectral_tensor"]
                    _triggered, _ood_conf, _reason = _ood_fallback.should_trigger(
                        _trm_out, _spec_t
                    )
                    if _triggered:
                        print(
                            f"\U0001f6a8 TRMOODFallback TRIGGERED: "
                            f"level=LEVEL_{_ood_fallback.cfg.ood_halt_threshold} "
                            f"ood_conf={_ood_conf:.3f} "
                            f"domain={trm_reasoner_result['primary_domain']} "
                            f"reason: {_reason}"
                        )
                        _ood_res = _ood_fallback.route(
                            query=text,
                            trm_output=_trm_out,
                            spectral_vec=_spec_t,
                            domain=trm_reasoner_result["primary_domain"],
                        )
                        ood_fallback_result = {
                            "triggered": True,
                            "level": str(_ood_res.fallback_level),
                            "ood_confidence": _ood_res.ood_head_confidence,
                            "selected_domain": _ood_res.selected_domain,
                            "reason": _ood_res.reason,
                        }

        tag_vectors = embed_tags_transformer(normalized_tags)
        tag_clusters = cluster_tags_transformer(normalized_tags, tag_vectors)
        recent_statements = temporal_layer.get_recent_statements()
        spatial_analysis = analyze_spatial_locality(recent_statements, tag_clusters)
        domain_patch = assign_domain_patch(spatial_analysis)

        pre_filter_result = expert_filter.filter_experts_by_tags(
            normalized_tags,
            expert_system,
            bert_manager=None,
            bert_domains=registered_domains,
        )

        if ENABLE_LOGGING and idx % LOG_SAMPLE_RATE == 0:
            if pre_filter_result["missing_domains"]:
                print(
                    f"\u26a0\ufe0f  Expert pre-check -- missing domains: "
                    f"{pre_filter_result['missing_domains']}\n"
                )

        filtered_experts = {
            d: expert_system.experts[d]
            for d in relevant_domains
            if d in expert_system.experts
        }

        # ----------------------------------------------------------------
        # graph:tool_start — Phase 2 validation
        # ----------------------------------------------------------------
        emitter.emit(
            phase_name="graph:tool_start",
            message="Starting Phase 2 validation",
            detail="Expert scoring + confidence calibration",
            state="running",
            metadata={
                "tool": "phase2_validation",
                "tag_count": len(normalized_tags),
                "expert_count": len(filtered_experts),
                "domains": list(filtered_experts.keys()),
            },
        )

        phase2_result = phase2_pipeline.run(
            text=text,
            routing_context=routing_context,
            filtered_experts=filtered_experts or None,
            depth_config=depth_cfg,
        )
        _prev_phase2_result = phase2_result

        # ----------------------------------------------------------------
        # graph:tool_done — Phase 2 validation complete
        # ----------------------------------------------------------------
        emitter.emit(
            phase_name="graph:tool_done",
            message="Phase 2 validation complete",
            detail="",
            state="running",
            metadata={
                "tool": "phase2_validation",
                "decision": str(getattr(phase2_result, "decision_label", "") or ""),
                "confidence": round(
                    float(getattr(phase2_result, "confidence", 0.0) or 0.0), 4
                ),
            },
        )

        p3_input = _adapt_phase2_to_p3(phase2_result, original_text=text)

        # ----------------------------------------------------------------
        # graph:tool_start — Phase 3 reasoning
        # ----------------------------------------------------------------
        emitter.emit(
            phase_name="graph:tool_start",
            message="Starting Phase 3 reasoning pipeline",
            detail=f"phase3_passes={phase3_passes}",
            state="running",
            metadata={
                "tool": "phase3_reasoning",
                "phase3_passes": phase3_passes,
                "action": str(getattr(p3_input, "action", "") or ""),
            },
        )

        phase3_result = phase3_pipeline.run_complete_pipeline(
            final_decision_result=p3_input,
        )

        # ----------------------------------------------------------------
        # graph:tool_done — Phase 3 reasoning complete
        # ----------------------------------------------------------------
        _p3_latencies: Dict[str, float] = {}
        if isinstance(phase3_result, dict):
            _p3_latencies = phase3_result.get("phase_latencies", {}) or {}
        elif hasattr(phase3_result, "phase_latencies"):
            _p3_latencies = dict(getattr(phase3_result, "phase_latencies", {}) or {})

        emitter.emit(
            phase_name="graph:tool_done",
            message="Phase 3 reasoning complete",
            detail="",
            state="running",
            metadata={
                "tool": "phase3_reasoning",
                "phase_latencies_ms": {k: round(v, 1) for k, v in _p3_latencies.items()},
            },
        )

        # ----------------------------------------------------------------
        # graph:synthesis_start — LLM synthesis about to begin
        # Fires before unified_decision_analysis so the frontend can show
        # a "thinking" node as soon as the expert arbiter starts.
        # ----------------------------------------------------------------
        emitter.emit(
            phase_name="graph:synthesis_start",
            message="Expert synthesis starting",
            detail=f"Arbitrating across {len(filtered_experts)} expert(s)",
            state="running",
            metadata={
                "expert_count": len(filtered_experts),
                "domains": list(filtered_experts.keys()),
                "reasoning_mode": reasoning_mode,
                "should_escalate": should_escalate,
            },
        )

        unified_decision = expert_system.unified_decision_analysis(
            input_text=text,
            routing_result=routing_context,
            filtered_experts=filtered_experts or None,
            depth_config=depth_cfg,
        )

        p3_from_unified = _adapt_unified_to_p3(
            unified_decision,
            original_text=text,
            phase2_metadata=_to_jsonable(phase2_result)
            if is_dataclass(phase2_result)
            else {},
        )

        final_phase3 = phase3_result or p3_from_unified

        combined_result = combine_routing_and_expert_decisions(
            routing=routing_context,
            expert=unified_decision,
        )

        decision_type: str = getattr(unified_decision, "decision_type", "") or ""
        metrics.expert_decisions[decision_type] += 1

        selected_domain: str = ""
        selected_experts = list(getattr(unified_decision, "selected_experts", []) or [])
        if selected_experts:
            selected_domain = str(selected_experts[0])
        for attr in ("domain", "selected_domain", "expert_name"):
            val = getattr(unified_decision, attr, None)
            if val:
                selected_domain = str(val)
                break
        metrics.domains[selected_domain or "unknown"] += 1

        decision_confidence: float = float(
            getattr(unified_decision, "expert_confidence", 0.0) or 0.0
        )

        if ENABLE_LOGGING and idx % LOG_SAMPLE_RATE == 0:
            print(
                f"\U0001f9e0 [{idx}] domain={selected_domain} "
                f"decision={decision_type} "
                f"conf={decision_confidence:.3f}\n"
            )

        if trm_trace_writer is not None:
            try:
                trm_trace_writer.record(
                    text=text,
                    routing_context=routing_context,
                    selected_domain=selected_domain,
                    trm_lookup_result=trm_reasoner_result,
                )
            except Exception as _tw_err:
                if ENABLE_LOGGING:
                    print(f"\u26a0\ufe0f  TraceWriter error: {_tw_err}")

        all_sentence_data.append(
            {
                "sentence": text,
                "tags": normalized_tags,
                "timestamp": timestamp,
                "layer0_routing": _to_jsonable(layer0_result),
                "routing_context": _to_jsonable(routing_context),
                "phase2_result": _to_jsonable(phase2_result),
                "phase3_result": _to_jsonable(final_phase3),
                "expert_decision": _to_jsonable(unified_decision),
                "expert_flag": decision_type,
                "selected_domain": selected_domain,
                "decision_confidence": decision_confidence,
                "shadow_signal": shadow_signal_dict,
                "should_escalate": should_escalate,
                "trm_reasoner_result": _to_jsonable(
                    {
                        k: v
                        for k, v in (trm_reasoner_result or {}).items()
                        if not k.startswith("_")
                    }
                ),
                "ood_fallback_result": _to_jsonable(ood_fallback_result or {}),
            }
        )

    return all_sentence_data, metrics
