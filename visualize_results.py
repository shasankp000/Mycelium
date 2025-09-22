import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import Counter

def visualize_mycelium_results():
    """Create visualizations for Mycelium workflow results"""
    
    # Load data
    with open("evaluation_data/expert_evaluation_results.json", "r") as f:
        expert_data = json.load(f)
    
    with open("evaluation_data/tag_clusters_transformer.json", "r") as f:
        cluster_data = json.load(f)
    
    with open("evaluation_data/temporal_analysis.json", "r") as f:
        temporal_data = json.load(f)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Mycelium Expert Evaluation & Clustering Analysis', fontsize=16)
    
    # 1. Expert Flag Distribution
    flags = expert_data["flag_summary"]
    axes[0, 0].pie(flags.values(), labels=flags.keys(), autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('Expert Flag Distribution')
    
    # 2. F1 Score Distribution by Expert
    f1_scores = []
    expert_names = []
    for result in expert_data["detailed_results"]:
        for expert, scores in result["expert_scores"].items():
            f1_scores.append(scores["f1"])
            expert_names.append(expert)
    
    if f1_scores:
        df_scores = pd.DataFrame({"Expert": expert_names, "F1_Score": f1_scores})
        sns.boxplot(data=df_scores, x="Expert", y="F1_Score", ax=axes[0, 1])
        axes[0, 1].set_title('F1 Score Distribution by Expert')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Cluster Size Distribution
    cluster_sizes = [len(tags) for tags in cluster_data["clusters"].values()]
    axes[0, 2].hist(cluster_sizes, bins=10, edgecolor='black')
    axes[0, 2].set_title('Cluster Size Distribution')
    axes[0, 2].set_xlabel('Cluster Size')
    axes[0, 2].set_ylabel('Frequency')
    
    # 4. Tag Frequency
    all_tags = []
    for result in expert_data["detailed_results"]:
        all_tags.extend(result["tags"])
    
    tag_counts = Counter(all_tags)
    top_tags = dict(tag_counts.most_common(10))
    
    axes[1, 0].bar(top_tags.keys(), top_tags.values())
    axes[1, 0].set_title('Top 10 Most Frequent Tags')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 5. Expert Performance Heatmap
    experts = list(set(expert_names))
    metrics = ['precision', 'recall', 'f1', 'mse', 'mae']
    
    if experts and len(experts) > 1:
        heatmap_data = []
        for expert in experts:
            expert_metrics = []
            for metric in metrics:
                scores = []
                for result in expert_data["detailed_results"]:
                    if expert in result["expert_scores"]:
                        scores.append(result["expert_scores"][expert][metric])
                expert_metrics.append(np.mean(scores) if scores else 0)
            heatmap_data.append(expert_metrics)
        
        sns.heatmap(heatmap_data, annot=True, xticklabels=metrics, 
                   yticklabels=experts, ax=axes[1, 1], cmap='RdYlBu_r')
        axes[1, 1].set_title('Expert Performance Heatmap (Average Scores)')
    
    # 6. Temporal Analysis
    if temporal_data["frequent_tags"]:
        freq_tags = temporal_data["frequent_tags"]
        axes[1, 2].bar(freq_tags.keys(), freq_tags.values())
        axes[1, 2].set_title('Frequent Tags (Recent Activity)')
        axes[1, 2].tick_params(axis='x', rotation=45)
    else:
        axes[1, 2].text(0.5, 0.5, 'No frequent tags data', 
                       horizontalalignment='center', verticalalignment='center')
        axes[1, 2].set_title('Frequent Tags (Recent Activity)')
    
    plt.tight_layout()
    plt.savefig('evaluation_data/mycelium_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("=== MYCELIUM ANALYSIS SUMMARY ===")
    print(f"Total sentences evaluated: {expert_data['sentences_evaluated']}")
    print(f"Flag distribution: {expert_data['flag_summary']}")
    print(f"Number of clusters: {len(cluster_data['clusters'])}")
    print(f"Average cluster size: {np.mean(cluster_sizes):.2f}")
    print(f"Recent statements analyzed: {temporal_data['recent_statements_count']}")
    print(f"Patch assignment: {temporal_data['patch_assignment']['action']}")
    
    if f1_scores:
        print(f"Average F1 score across all experts: {np.mean(f1_scores):.3f}")
        print(f"Best performing expert: {expert_names[np.argmax(f1_scores)]} (F1: {max(f1_scores):.3f})")

if __name__ == "__main__":
    try:
        visualize_mycelium_results()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the workflow first to generate the required JSON files.")
    except Exception as e:
        print(f"Visualization error: {e}")
        print("Make sure you have matplotlib, seaborn, pandas, and numpy installed.")