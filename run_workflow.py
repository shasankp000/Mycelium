import json
import datetime
from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass
from typing import List, Dict, Any, Optional, Sequence, Tuple

import numpy as np
from layer_1_prototype import (
    extract_tags_llama,
    normalize_tags,
    embed_tags_transformer,
    cluster_tags_transformer,
    TemporalLocalityLayer,
    analyze_spatial_locality,
    assign_domain_patch,
)
from multi_lens_router import MultiLensRouter
from phase2_validation.pipeline import Phase2Pipeline
from phase3_validation.pipeline import Phase3To5Pipeline
from layer_2_prototype import get_expert_model
import layer_2_prototype
from unified_expert_system import UnifiedExpertSystem
from expert_filter import ExpertFilter
from orchestration import combine_routing_and_expert_decisions
from layer0.router import QuestionRouter
from tuning_config import ENABLE_LOGGING, LOG_SAMPLE_RATE
from patch_batch_logger import patch_logger


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""

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
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "__dict__") and not isinstance(obj, (str, bytes)):
        return _to_jsonable(vars(obj))
    return obj


@dataclass
class WorkflowMetrics:
    """Lightweight observability for end-to-end runs.

    This is intentionally simple so tests can assert on high-level behavior
    without depending on internal implementation details.
    """

    layer0_routes: Counter = None
    routing_classifications: Counter = None
    expert_decisions: Counter = None
    domains: Counter = None

    def __post_init__(self) -> None:
        self.layer0_routes = Counter() if self.layer0_routes is None else self.layer0_routes
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
) -> Tuple[List[Dict[str, Any]], WorkflowMetrics]:
    """Run the full Mycelium workflow on a batch of sentences.

    Returns the per-sentence analysis records and aggregated WorkflowMetrics.

    Args:
        sentences:  One or more user queries to process.
        trace_id:   Optional caller-supplied trace UUID.  When provided, any
                    CREATE_NEW_PATCH response for the first sentence will be
                    patched back into the batch log under this ID.
    """

    phase2_pipeline = Phase2Pipeline()
    phase3_pipeline = Phase3To5Pipeline()
    router = MultiLensRouter()

    # Step 0: Initialize unified expert system
    print("Initializing unified expert system (K-Medoids + Calibration + OOD Detection)...")
    expert_system = UnifiedExpertSystem()
    print(f"Initialized unified expert system with {len(expert_system.experts)} experts\n")

    # Initialize expert filter for tag-based routing (with auto-clustering)
    print("Initializing expert filter with automatic semantic clustering...")
    domain_list = ["music", "physics", "chemistry", "medical"]
    expert_filter = ExpertFilter(
        domain_list=domain_list,
        use_auto_clustering=True,
        similarity_threshold=0.45,
    )
    print("✅ Expert filter initialized with auto-clustering\n")

    temporal_layer = TemporalLocalityLayer(max_size=50, time_window_hours=24)
    all_sentence_data: List[Dict[str, Any]] = []
    all_tags: List[str] = []

    # Initialize Layer0 router and metrics
    print("Initializing Layer0 question router...")
    question_router = QuestionRouter()
    print("✅ Layer0 router initialized\n")
    metrics = WorkflowMetrics()

    for idx, text in enumerate(sentences, start=1):
        tags = extract_tags_llama(text)
        normalized_tags = normalize_tags(tags)
        timestamp = datetime.datetime.now().isoformat()
        temporal_layer.add_statement(text, normalized_tags, timestamp)

        # Step 1a: Layer0 classification
        layer0_result = question_router.route(text)
        metrics.layer0_routes[layer0_result.route] += 1

        # Short-circuit for non-REASONING_PIPELINE routes
        if layer0_result.route != "REASONING_PIPELINE":
            if ENABLE_LOGGING and idx % LOG_SAMPLE_RATE == 0:
                print(f"🚫 Layer0 route: {layer0_result.route.upper()}\n")

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

        # Routing and Phase 2/3 processing
        routing_context = router.route(text)
        classification = getattr(routing_context, "classification", None)
        if classification:
            metrics.routing_classifications[classification] += 1

        phase2_result = phase2_pipeline.run(text, routing_context=routing_context)
        phase3_result = phase3_pipeline.run_complete_pipeline(phase2_result)

        # Filter experts using routing-selected domains first, then semantic tags
        relevant_domains: List[str] = []
        for domain in getattr(routing_context, "selected_domains", []):
            normalized = expert_filter.normalize_domain(str(domain))
            if normalized:
                relevant_domains.append(normalized)

        if not relevant_domains:
            for tag in normalized_tags:
                domain = expert_filter.normalize_domain(tag)
                if domain:
                    relevant_domains.append(domain)

        relevant_domains = list(set(relevant_domains))
        filtered_experts = {
            domain: expert_system.experts[domain]
            for domain in relevant_domains
            if domain in expert_system.experts
        }

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

        # ------------------------------------------------------------------
        # CREATE_NEW_PATCH: no matching domain expert exists.
        #
        # 1. Log the query immediately into the daily batch file so the
        #    offline patch-model training pipeline can pick it up later.
        # 2. Do NOT short-circuit — let the full 6-phase reasoning pipeline
        #    run as normal (consequence generation, evidence grounding,
        #    sandbox, validation are all part of phases 2-3 and already
        #    ran above).  The LLM will still produce the best response it
        #    can without a specialist expert.
        # 3. After the response is available, fill it back into the log.
        # ------------------------------------------------------------------
        is_patch_decision = expert_decision.decision_type == "CREATE_NEW_PATCH"
        # Determine the trace_id for this sentence.  For single-query calls
        # from the API the caller passes its UUID; for batch calls we
        # generate a per-sentence one.
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
                    f"📝 CREATE_NEW_PATCH — logged query to batch "
                    f"(trace_id={sentence_trace_id}). "
                    "Proceeding through reasoning pipeline.\n"
                )

        # Retrieve the final LLM-generated answer from phase3 result so we
        # can attach it to the batch record.  Different pipeline versions
        # surface the answer under different keys; we try the most common
        # ones in order.
        final_answer: str = ""
        p3_dict = _to_jsonable(phase3_result) if phase3_result else {}
        for _key in ("final_answer", "answer", "response", "output", "text"):
            val = p3_dict.get(_key)
            if val and isinstance(val, str):
                final_answer = val
                break
        # Fallback: check nested action_result
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

    # Step 2: Save all sentences, tags, and timestamps to JSON
    with open("evaluation_data/sentence_tags.json", "w", encoding="utf-8") as f:
        json.dump(all_sentence_data, f, indent=4, cls=NumpyEncoder)

    # Save expert evaluation results separately
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

    # Step 3: Remove duplicates for clustering
    unique_tags = list(set(all_tags))
    if unique_tags:
        embeddings = embed_tags_transformer(unique_tags, model_name="all-mpnet-base-v2")
        clusters = cluster_tags_transformer(
            unique_tags, embeddings, similarity_threshold=0.5
        )
    else:
        embeddings = np.zeros((0, 0))
        clusters = {}

    clustering_data = {
        "tags": unique_tags,
        "clusters": clusters,
    }
    with open("evaluation_data/tag_clusters_transformer.json", "w", encoding="utf-8") as f:
        json.dump(clustering_data, f, indent=4)
    print("Clusters saved to tag_clusters_transformer.json:", clusters)

    # Step 4: Temporal and spatial locality analysis
    recent_statements = temporal_layer.get_recent_statements(time_limit_hours=1)
    spatial_analysis = analyze_spatial_locality(recent_statements, clusters)
    patch_assignment = assign_domain_patch(
        spatial_analysis, similarity_threshold=0.3
    )
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
    # Medical
    med_path = "dummy_models/Medical/medical_dataset.csv"
    med_samples: List[str] = []
    if not _is_lfs_pointer(med_path):
        med_df = pd.read_csv(med_path)
        med_samples = _sample_column(med_df, "sentence", 3, 42)

    # Music
    music_path = "dummy_models/Music/music_classification_dataset.csv"
    music_samples: List[str] = []
    if not _is_lfs_pointer(music_path):
        music_df = pd.read_csv(music_path)
        music_samples = _sample_column(music_df, "sentence", 3, 43)

    # Physics
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
