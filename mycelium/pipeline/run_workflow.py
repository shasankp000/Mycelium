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
# Phase D — TRM integration: routing refinement + graph store + DFS lookup + promotion
from mycelium.trm.integration import TRMLens
from mycelium.trm.graph_store import GraphStore
from mycelium.trm.dfs_lookup import DFSLookup
from mycelium.trm.promotion import PromotionPolicy
# Phase E — Contradiction classifier (non-fatal fallback if IR stack unavailable)
try:
    from mycelium.contradiction.classifier import ContradictionClassifier
    _CONTRADICTION_AVAILABLE = True
except Exception:
    ContradictionClassifier = None  # type: ignore[assignment,misc]
    _CONTRADICTION_AVAILABLE = False


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

    return P3FinalDecisionResult(
        decision=_get("decision_label", "prediction", "final_decision", "decision"),
        confidence=float(_get("confidence", "expert_confidence", default=0.5)),
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
    layer0_routes: Counter = None
    routing_classifications: Counter = None
    expert_decisions: Counter = None
    domains: Counter = None

    def __post_init__(self) -> None:
        self.layer0_routes = (
            Counter() if self.layer0_routes is None else self.layer0_routes
        )
        self.routing_classifications = (
            Counter()
            if self.routing_classifications is None
            else self.routing_classifications
        )
        self.expert_decisions = (
            Counter() if self.expert_decisions is None else self.expert_decisions
        )
        self.domains = Counter() if self.domains is None else self.domains

    def to_dict(self) -> Dict[str, Dict[str, int]]:
        return {
            "layer0_routes": dict(self.layer0_routes),
            "routing_classifications": dict(self.routing_classifications),
            "expert_decisions": dict(self.expert_decisions),
            "domains": dict(self.domains),
        }


def run_mycelium_workflow(
    sentences: Sequence[str],
    trace_id: Optional[str] = None,
    on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    reasoning_mode: ReasoningMode = "balanced",
) -> Tuple[List[Dict[str, Any]], WorkflowMetrics]:
    """
    Phase 6: *reasoning_mode* controls the DAG/DFS traversal depth and expert
    top-k throughout the pipeline.  The mapping is defined in api_models.py::

        "fast"     → dfs_max_depth=1, expert_top_k=1, phase3_passes=1
        "balanced" → dfs_max_depth=2, expert_top_k=2, phase3_passes=2  (default)
        "deep"     → dfs_max_depth=4, expert_top_k=3, phase3_passes=3

    Phase D wiring:
        - TRMLens refines routing_context domain probabilities (passthrough
          fallback when no checkpoint exists).
        - GraphStore accumulates per-sentence graph stubs.
        - PromotionPolicy is run after every observe() call so the lifecycle
          ladder (DRAFT→CANDIDATE→STABILIZED→CANONICAL) is active from the
          first sentence.
        - DFSLookup result is injected into phase3_input.metadata["trm_lookup"].

    Phase E wiring:
        - ContradictionClassifier runs on consecutive phase2_result pairs
          (previous sentence vs current).  The ContradictionEdge classification
          is non-fatal and stored in all_sentence_data["contradiction"].
    """
    import uuid as _uuid

    depth_cfg = get_depth_config(reasoning_mode)
    dfs_max_depth: int   = depth_cfg["dfs_max_depth"]
    expert_top_k: int    = depth_cfg["expert_top_k"]
    phase3_passes: int   = depth_cfg["phase3_passes"]

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
        message="Warming up model registry\u2026",
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

    # Phase D — TRM layer (fallback-safe)
    trm_lens = TRMLens()
    graph_store = GraphStore()
    dfs_lookup = DFSLookup(graph_store)
    promotion_policy = PromotionPolicy()

    # Phase E — ContradictionClassifier (lazy singleton, fallback-safe)
    contradiction_classifier = ContradictionClassifier() if _CONTRADICTION_AVAILABLE else None
    _prev_phase2_result: Optional[Any] = None  # rolling previous sentence result

    from mycelium.pipeline.unified_expert_system import get_unified_expert_system, _unified_system
    _expert_msg = 'Setting up environment\u2026' if _unified_system is None else 'Loading expert system\u2026'
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
        message="Syncing spectral signatures\u2026",
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
            message="Classifying query\u2026",
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
            message="Routing query through semantic lenses\u2026",
            detail=f"{len(normalized_tags)} tag(s) extracted",
            state="running",
            metadata={"tag_count": len(normalized_tags)},
        )

        routing_context = router.route(text)

        # Phase D — TRMLens refines domain probabilities; passthrough if no checkpoint.
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

        # Phase 6 — cap the expert pool to expert_top_k when mode is fast/balanced
        if expert_top_k < len(relevant_domains):
            relevant_domains = relevant_domains[:expert_top_k]

        filtered_experts = {
            domain: expert_system.experts[domain]
            for domain in relevant_domains
            if domain in expert_system.experts
        }

        emitter.emit(
            phase_name="graph_phase2",
            message="Running reasoning pipeline\u2026",
            detail=f"{len(filtered_experts)} expert(s) active (top_k={expert_top_k})",
            state="running",
            metadata={
                "active_domains": list(filtered_experts.keys()),
                "expert_top_k": expert_top_k,
                "dfs_max_depth": dfs_max_depth,
            },
        )
        emitter.emit_heartbeat(idx % 5)

        # Phase 2 receives depth_config so sub-phases can respect DFS depth
        phase2_result = phase2_pipeline.run(
            text,
            routing_context=routing_context,
            filtered_experts=filtered_experts,
            depth_config=depth_cfg,
        )

        # Phase E — ContradictionClassifier: compare current vs previous p2 output.
        # Operates on text-level nodes built from phase2 decision fields.
        # Non-fatal — a classifier error must never block the main workflow.
        contradiction_result: Optional[Dict[str, Any]] = None
        if contradiction_classifier is not None and _prev_phase2_result is not None:
            try:
                import types as _ct
                import hashlib as _ch

                def _p2_node(p2: Any, node_id: str) -> Any:
                    """Minimal IRNode-compatible stub from a phase2 result."""
                    label = str(
                        getattr(p2, "final_decision", None)
                        or getattr(p2, "decision_label", None)
                        or ""
                    )
                    canonical = label.lower().strip()
                    pred_family = str(
                        getattr(p2, "domain", None)
                        or getattr(p2, "selected_domain", None)
                        or "UNKNOWN"
                    ).upper()
                    h = _ch.sha256(f"{canonical}|{pred_family}|0".encode()).hexdigest()
                    sig = _ct.SimpleNamespace(
                        semantic_hash=h,
                        canonical_form=canonical,
                        predicate_family=pred_family,
                        abstraction_level=0,
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
                        id=node_id,
                        semantic_signature=sig,
                        confidence_state=conf_state,
                        temporal_state=t_state,
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

                # If severity ≥ threshold, mark the stored graph as CONTESTED
                _graph_id_prev = f"G-{_request_id[:8]}-{idx - 1:04d}"
                prev_graph = graph_store.get_latest(_graph_id_prev)
                if prev_graph is not None:
                    target = promotion_policy.contest(
                        prev_graph,
                        severity=float(getattr(c_edge, "severity", 0.0) or 0.0),
                    )
                    if target is not None:
                        graph_store.add_revision(_graph_id_prev, new_state=target)

            except Exception as _ce:
                contradiction_result = {"error": str(_ce)}

        emitter.emit(
            phase_name="graph_unified_decision",
            message="Computing unified expert decision\u2026",
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
            routing_context,
            expert_decision,
        )

        # Phase D — GraphStore + DFSLookup + PromotionPolicy
        trm_lookup_result: Optional[Dict[str, Any]] = None
        try:
            import types as _types
            import hashlib as _hashlib

            _text_hash = _hashlib.sha256(text.encode()).hexdigest()[:16]
            _graph_id = f"G-{_request_id[:8]}-{idx:04d}"

            _sig = _types.SimpleNamespace(
                semantic_hash=_text_hash,
                canonical_form=text.lower().strip(),
                predicate_family=getattr(routing_context, "classification", "UNKNOWN") or "UNKNOWN",
                equivalence_family=normalized_tags,
            )
            _node = _types.SimpleNamespace(
                id=f"{_graph_id}-n0",
                semantic_signature=_sig,
            )
            _confidence_stub = _types.SimpleNamespace(overall_confidence=float(
                getattr(expert_decision, "expert_confidence", 0.5) or 0.5
            ))
            _graph_stub = _types.SimpleNamespace(
                graph_id=_graph_id,
                nodes=[_node],
                edges=[],
                state="DRAFT",
                version="v1",
                confidence_state=_confidence_stub,
                fingerprint=None,
                ontology_version="1.0",
                created_at=timestamp,
                updated_at=timestamp,
                parent_graph_id=None,
            )

            graph_store.put(_graph_stub)
            obs_count = graph_store.observe(_graph_id)

            # PromotionPolicy: run after every observe() — lifecycle ladder now active
            _target_state = promotion_policy.evaluate(_graph_stub, obs_count)
            if _target_state is not None and _target_state != _graph_stub.state:
                graph_store.add_revision(_graph_id, new_state=_target_state)

            # DFS search: BY_HASH first, then BY_EQUIVALENCE fallback
            _lookup = dfs_lookup.search(_node, strategy="BY_HASH", min_state_rank=0)
            if not _lookup.found:
                _lookup = dfs_lookup.find_equivalent(_node)

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
            }
        except Exception as _trm_exc:
            trm_lookup_result = {"found": False, "error": str(_trm_exc)}

        emitter.emit(
            phase_name="expert_decision",
            message="Expert decision reached",
            detail=(
                f"{getattr(expert_decision, 'decision_type', 'N/A')} \u00b7 "
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
            },
        )

        p2_fdr = _adapt_phase2_to_p3(phase2_result, original_text=text)
        phase2_extra_metadata: Dict[str, Any] = dict(p2_fdr.metadata or {})

        phase3_input = _adapt_unified_to_p3(
            expert_decision,
            original_text=text,
            phase2_metadata=phase2_extra_metadata,
        )

        # Inject TRM + contradiction context into Phase 3 metadata
        if phase3_input.metadata is None:
            phase3_input.metadata = {}
        if trm_lookup_result is not None:
            phase3_input.metadata["trm_lookup"] = trm_lookup_result
        if contradiction_result is not None:
            phase3_input.metadata["contradiction"] = contradiction_result

        emitter.emit(
            phase_name="graph_phase3",
            message="Running validation pipeline\u2026",
            detail=f"Phase 3-5: action execution + feedback collection (passes={phase3_passes})",
            state="running",
            metadata={"phase3_passes": phase3_passes},
        )

        phase3_result = phase3_pipeline.run_complete_pipeline(phase3_input)

        # Advance rolling window for next iteration
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
            if expert_decision.selected_experts
            else "unknown"
        )
        confidence = float(getattr(expert_decision, "expert_confidence", 0.0))

        is_patch_decision = expert_decision.decision_type == "CREATE_NEW_PATCH"
        import uuid as _uuid

        sentence_trace_id = trace_id if (idx == 1 and trace_id) else str(_uuid.uuid4())

        if is_patch_decision:
            phase_latencies: Dict[str, float] = (
                _to_jsonable(phase3_result).get("phase_latencies", {})
                if isinstance(phase3_result, (dict,))
                or hasattr(phase3_result, "__dict__")
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
            message="Updating tag cluster model\u2026",
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
            }
        )
        all_tags.extend(normalized_tags)

        if ENABLE_LOGGING and idx % LOG_SAMPLE_RATE == 0:
            print(
                "Sentence: {sent}\nTags: {tags}\nTimestamp: {ts}\n"
                "Unified Decision: {flag} (Domain: {dom}, Confidence: {conf:.3f})\n"
                "Reasoning Mode: {mode} (dfs_depth={dfs}, top_k={k})\n"
                "TRM Lookup: found={trm_found} strategy={trm_strat} obs={obs} promoted={promoted}\n"
                "Contradiction: type={c_type} severity={c_sev}\n".format(
                    sent=text,
                    tags=normalized_tags,
                    ts=timestamp,
                    flag=flag,
                    dom=selected_domain,
                    conf=confidence,
                    mode=reasoning_mode,
                    dfs=dfs_max_depth,
                    k=expert_top_k,
                    trm_found=trm_lookup_result.get("found") if trm_lookup_result else False,
                    trm_strat=trm_lookup_result.get("strategy_used") if trm_lookup_result else "N/A",
                    obs=trm_lookup_result.get("obs_count") if trm_lookup_result else 0,
                    promoted=trm_lookup_result.get("promoted_to") if trm_lookup_result else None,
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
    }
    for entry in all_sentence_data:
        flag = entry["expert_flag"]
        expert_evaluation_results["flag_summary"][flag] = (
            expert_evaluation_results["flag_summary"].get(flag, 0) + 1
        )

    with open(
        "evaluation_data/expert_evaluation_results.json", "w", encoding="utf-8"
    ) as f:
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
    with open(
        "evaluation_data/tag_clusters_transformer.json", "w", encoding="utf-8"
    ) as f:
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
