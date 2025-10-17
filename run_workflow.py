import json
import datetime
import numpy as np
from layer_1_prototype import (
    extract_tags_llama, normalize_tags, embed_tags_transformer, cluster_tags_transformer,
    TemporalLocalityLayer, analyze_spatial_locality, assign_domain_patch
)
from layer_2_prototype import get_expert_model
import layer_2_prototype
from unified_expert_system import UnifiedExpertSystem
from expert_filter import ExpertFilter

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def run_mycelium_workflow(sentences):
    # Step 0: Initialize unified expert system
    print("Initializing unified expert system (K-Medoids + Calibration + OOD Detection)...")
    expert_system = UnifiedExpertSystem()
    print(f"Initialized unified expert system with {len(expert_system.experts)} experts\n")
    
    # Initialize expert filter for tag-based routing (with auto-clustering)
    print("Initializing expert filter with automatic semantic clustering...")
    domain_list = ['music', 'physics', 'chemistry', 'medical']
    expert_filter = ExpertFilter(
        domain_list=domain_list,
        use_auto_clustering=True,
        similarity_threshold=0.45
    )
    print("✅ Expert filter initialized with auto-clustering\n")
    
    # Step 1: Tag generation and normalization
    temporal_layer = TemporalLocalityLayer(max_size=50, time_window_hours=24)
    all_sentence_data = []
    all_tags = []
    
    for text in sentences:
        tags = extract_tags_llama(text)
        normalized_tags = normalize_tags(tags)
        timestamp = datetime.datetime.now().isoformat()
        temporal_layer.add_statement(text, normalized_tags, timestamp)
        
        # Filter experts by tags using semantic clustering
        relevant_domains = []
        for tag in normalized_tags:
            domain = expert_filter.normalize_domain(tag)
            if domain:
                relevant_domains.append(domain)
        relevant_domains = list(set(relevant_domains))  # Remove duplicates
        
        if not relevant_domains:
            # No expert for this domain
            expert_decision = {
                'unified_decision': {
                    'decision_flag': 'create_new_expert',
                    'selected_domain': 'unknown',
                    'confidence_in_decision': 0.9,
                    'reasoning': [f"No expert available for tags: {normalized_tags}"]
                },
                'expert_analyses': {},
                'system_summary': {'experts_analyzed': 0}
            }
        else:
            # Evaluate only relevant experts
            filtered_experts = {domain: expert_system.experts[domain] for domain in relevant_domains if domain in expert_system.experts}
            expert_decision = expert_system.unified_decision_analysis(text, filtered_experts=filtered_experts)
        
        flag = expert_decision['unified_decision']['decision_flag']
        selected_domain = expert_decision['unified_decision']['selected_domain']
        confidence = expert_decision['unified_decision']['confidence_in_decision']
        
        all_sentence_data.append({
            "sentence": text,
            "tags": normalized_tags,
            "timestamp": timestamp,
            "expert_decision": expert_decision,
            "expert_flag": flag,
            "selected_domain": selected_domain,
            "decision_confidence": confidence
        })
        all_tags.extend(normalized_tags)
        print(f"Sentence: {text}\nTags: {normalized_tags}\nTimestamp: {timestamp}\nUnified Decision: {flag} (Domain: {selected_domain}, Confidence: {confidence:.3f})\n")
    
    # Step 2: Save all sentences, tags, and timestamps to JSON
    with open("evaluation_data/sentence_tags.json", "w", encoding="utf-8") as f:
        json.dump(all_sentence_data, f, indent=4, cls=NumpyEncoder)
    
    # Save expert evaluation results separately
    expert_evaluation_results = {
        "evaluation_timestamp": datetime.datetime.now().isoformat(),
        "sentences_evaluated": len(all_sentence_data),
        "flag_summary": {},
        "detailed_results": all_sentence_data
    }
    
    # Count flag occurrences
    for entry in all_sentence_data:
        flag = entry["expert_flag"]
        expert_evaluation_results["flag_summary"][flag] = expert_evaluation_results["flag_summary"].get(flag, 0) + 1
    
    with open("evaluation_data/expert_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(expert_evaluation_results, f, indent=4, cls=NumpyEncoder)
    print("Expert evaluation results saved to expert_evaluation_results.json")
    
    # Step 3: Remove duplicates for clustering
    unique_tags = list(set(all_tags))
    embeddings = embed_tags_transformer(unique_tags, model_name="all-mpnet-base-v2")
    clusters = cluster_tags_transformer(unique_tags, embeddings, similarity_threshold=0.5)
    clustering_data = {
        "tags": unique_tags,
        "clusters": clusters
    }
    with open("evaluation_data/tag_clusters_transformer.json", "w", encoding="utf-8") as f:
        json.dump(clustering_data, f, indent=4)
    print("Clusters saved to tag_clusters_transformer.json:", clusters)
    
    # Step 4: Temporal and spatial locality analysis
    recent_statements = temporal_layer.get_recent_statements(time_limit_hours=1)
    spatial_analysis = analyze_spatial_locality(recent_statements, clusters)
    patch_assignment = assign_domain_patch(spatial_analysis, similarity_threshold=0.3)
    frequent_tags = temporal_layer.get_frequent_tags(min_frequency=2)
    temporal_analysis_data = {
        "recent_statements_count": len(recent_statements),
        "spatial_analysis": spatial_analysis,
        "patch_assignment": patch_assignment,
        "frequent_tags": frequent_tags,
        "analysis_timestamp": datetime.datetime.now().isoformat()
    }
    with open("evaluation_data/temporal_analysis.json", "w", encoding="utf-8") as f:
        json.dump(temporal_analysis_data, f, indent=4)
    print("Temporal analysis saved to temporal_analysis.json")

import pandas as pd
import random

def get_random_samples():
    base_dir = __file__
    # Medical
    med_df = pd.read_csv("dummy_models/Medical/medical_dataset.csv")
    med_samples = med_df['sentence'].dropna().sample(3, random_state=42).tolist()
    # Music
    music_df = pd.read_csv("dummy_models/Music/music_classification_dataset.csv")
    music_samples = music_df['sentence'].dropna().sample(3, random_state=43).tolist()
    # Physics
    phys_df = pd.read_csv("dummy_models/Physics/physics_data.csv")
    phys_samples = phys_df['Comment'].dropna().sample(4, random_state=44).tolist()
    # Mix and shuffle
    all_samples = med_samples + music_samples + phys_samples
    random.shuffle(all_samples)
    return all_samples

if __name__ == "__main__":
    sentences = get_random_samples()
    run_mycelium_workflow(sentences)