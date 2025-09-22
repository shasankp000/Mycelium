import json
import datetime
from layer_1_prototype import (
    extract_tags_llama, normalize_tags, embed_tags_transformer, cluster_tags_transformer,
    TemporalLocalityLayer, analyze_spatial_locality, assign_domain_patch
)
from layer_2_prototype import get_expert_model

def run_mycelium_workflow(sentences):
    # Step 1: Tag generation and normalization
    temporal_layer = TemporalLocalityLayer(max_size=50, time_window_hours=24)
    all_sentence_data = []
    all_tags = []
    
    for text in sentences:
        tags = extract_tags_llama(text)
        normalized_tags = normalize_tags(tags)
        timestamp = datetime.datetime.now().isoformat()
        temporal_layer.add_statement(text, normalized_tags, timestamp)
        
        # Expert evaluation
        expert_scores = {}
        matched_experts = []
        for tag in normalized_tags:
            expert = get_expert_model(tag)
            if expert:
                score = expert.score(text, normalized_tags)
                expert_scores[tag] = score
                matched_experts.append(tag)
        
        # Flag logic
        flag = None
        if not matched_experts:
            flag = "create_new_expert"
        elif len(matched_experts) == 1:
            score = expert_scores[matched_experts[0]]
            if score["f1"] > 0.7:
                flag = "use_existing_expert"
            else:
                flag = "create_new_expert"
        else:
            good_experts = [tag for tag in matched_experts if expert_scores[tag]["f1"] > 0.7]
            if len(good_experts) > 1:
                flag = "create_hybrid_expert"
            elif len(good_experts) == 1:
                flag = "use_existing_expert"
            else:
                flag = "create_new_expert"
        
        all_sentence_data.append({
            "sentence": text,
            "tags": normalized_tags,
            "timestamp": timestamp,
            "expert_scores": expert_scores,
            "expert_flag": flag
        })
        all_tags.extend(normalized_tags)
        print(f"Sentence: {text}\nTags: {normalized_tags}\nTimestamp: {timestamp}\nExpert Scores: {expert_scores}\nExpert Flag: {flag}\n")
    
    # Step 2: Save all sentences, tags, and timestamps to JSON
    with open("evaluation_data/sentence_tags.json", "w", encoding="utf-8") as f:
        json.dump(all_sentence_data, f, indent=4)
    
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
        json.dump(expert_evaluation_results, f, indent=4)
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

if __name__ == "__main__":
    sentences = [
        "AI-driven systems can analyze data, improve healthcare, and optimize logistics",
        "Quantum computers are revolutionizing physics and mathematics research",
        "Financial markets are influenced by global politics and economic trends",
        "Music and art therapy are used in modern healthcare for mental wellness",
        "Education technology platforms leverage AI to personalize learning",
        "Sports analytics use big data to optimize team performance",
        "Biology and chemistry are foundational for pharmaceutical innovations",
        "History and literature provide context for understanding political movements",
        "Mathematics is essential for advancements in physics and finance",
        "Logistics companies use AI and IoT to streamline supply chains"
    ]
    run_mycelium_workflow(sentences)