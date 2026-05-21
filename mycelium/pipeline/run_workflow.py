try:
    import orjson as _json_lib
    _USE_ORJSON = True
except ImportError:
    import json as _json_lib  # type: ignore[no-redef]
    _USE_ORJSON = False

import json
import datetime
import time as _time
from collections import Counter, deque
from dataclasses import asdict, dataclass, is_dataclass
from typing import Callable, List, Dict, Any, Optional, Sequence, Tuple

import numpy as np
from mycelium.pipeline.layer_1_prototype import (
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
from mycelium.pipeline.phase3.utils.types import FinalDecisionResult as P3FinalDecisionResult
from mycelium.pipeline.layer_2_prototype import get_expert_model
import mycelium.pipeline.layer_2_prototype as layer_2_prototype
from mycelium.pipeline.unified_expert_system import UnifiedExpertSystem
from mycelium.pipeline.expert_filter import ExpertFilter
from mycelium.pipeline.orchestration import combine_routing_and_expert_decisions
from layer0.router import QuestionRouter
from mycelium.pipeline.tuning_config import ENABLE_LOGGING, LOG_SAMPLE_RATE
from mycelium.pipeline.patch_batch_logger import patch_logger
from mycelium.pipeline.dynamic_signature_manager import DynamicSignatureManager
from mycelium.pipeline.model_registry import warmup, loaded_models, STARTUP_SPECS
from mycelium.pipeline.pipeline_event import EventEmitter, make_emitter


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

    extra_fields = ("original_text", "input_text", "query", "sentence",
                    "action_details", "expert_predictions")
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
) -> Tuple[List[Dict[str, Any]], WorkflowMetrics]:
    import uuid as _uuid
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
    )

    print("\U0001f9e0 Pre-flight: loading non-LLM model weights into registry...")
    warmup(STARTUP_SPECS)
    _resident = loaded_models()
    print(f"\u2705 ModelRegistry warm -- {len(_resident)} model(s) resident: "
          f"{[k.split(':')[1] for k in _resident]}\n")

    emitter.emit(
        phase_name="environment_ready",
        message="Environment ready",
        detail=f"{len(_resident)} model(s) resident",
        state="running",
        metadata={"model_count": len(_resident), "models": [k.split(':')[1] for k in _resident]},
    )

    phase2_pipeline = Phase2Pipeline()
    phase3_pipeline = Phase3To5Pipeline()

    emitter.emit(
        phase_name="graph_expert_init",
        message="Initialising expert system\u2026",
        detail="K-Medoids + Calibration + OOD Detection",
        state="running",
    )
    print("Initializing unified expert system (K-Medoids + Calibration + OOD Detection)...")
    expert_system = UnifiedExpertSystem()
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
                    f"\u2139\ufe0f  No domain resolved from routing or tags for \"{text[:60]}...\". "
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
        filtered_experts = {
            domain: expert_system.experts[domain]
            for domain in relevant_domains
            if domain in expert_system.experts
        }

        emitter.emit(
            phase_name="graph_phase2",
            message="Running reasoning pipeline\u2026",
            detail=f"{len(filtered_experts)} expert(s) active",
            state="running",
            metadata={"active_domains": list(filtered_experts.keys())},
        )
        emitter.emit_heartbeat(idx % 5)

        phase2_result = phase2_pipeline.run(
            text,
            routing_context=routing_context,
            filtered_experts=filtered_experts,
        )

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
        )
        expert_decision = combine_routing_and_expert_decisions(
            routing_context,
            expert_decision,
        )

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
                "selected_experts": list(getattr(expert_decision, "selected_experts", []) or []),
                "confidence": float(getattr(expert_decision, "expert_confidence", 0.0)),
            },
        )

        p2_fdr = _adapt_phase2_to_p3(phase2_result, original_text=text)
        phase2_extra_metadata: Dict[str, Any] = dict(p2_fdr.metadata or {})

        phase3_input = _adapt_unified_to_p3(
            expert_decision,
            original_text=text,
            phase2_metadata=phase2_extra_metadata,
        )

        emitter.emit(
            phase_name="graph_phase3",
            message="Running validation pipeline\u2026",
            detail="Phase 3-5: action execution + feedback collection",
            state="running",
        )

        phase3_result = phase3_pipeline.run_complete_pipeline(phase3_input)

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
        sentence_trace_id = (
            trace_id if (idx == 1 and trace_id) else str(_uuid.uuid4())
        )

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
            }
        )
        all_tags.extend(normalized_tags)

        if ENABLE_LOGGING and idx % LOG_SAMPLE_RATE == 0:
            print(
                "Sentence: {sent}\nTags: {tags}\nTimestamp: {ts}\n"
                "Unified Decision: {flag} (Domain: {dom}, Confidence: {conf:.3f})\n".format(
                    sent=text,
                    tags=normalized_tags,
                    ts=timestamp,
                    flag=flag,
                    dom=selected_domain,
                    conf=confidence,
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


import pandas as pd
import random


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
