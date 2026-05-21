import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EVAL_DATA_DIR = _PROJECT_ROOT / "evaluation_data"

def visualize_mycelium_results():
    """Create visualizations for Mycelium unified expert system results"""

    results_path = _EVAL_DATA_DIR / "expert_evaluation_results.json"
    try:
        with open(results_path, "r") as f:
            expert_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: expert_evaluation_results.json not found at {results_path}")
        return
    except json.JSONDecodeError:
        print("Error: Invalid JSON in expert_evaluation_results.json")
        return

    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Mycelium Unified Expert System Analysis', fontsize=16, fontweight='bold')

    # 1. Expert Decision Distribution (Pie Chart)
    if "flag_summary" in expert_data:
        flags = expert_data["flag_summary"]
        colors = ['#ff7f7f', '#87ceeb', '#98fb98', '#dda0dd']
        if flags:
            wedges, texts, autotexts = axes[0, 0].pie(
                flags.values(),
                labels=flags.keys(),
                autopct='%1.1f%%',
                colors=colors[:len(flags)],
                startangle=90
            )
            axes[0, 0].set_title('Expert Decision Distribution', fontweight='bold')
        else:
            axes[0, 0].text(0.5, 0.5, 'No flag data available', ha='center', va='center')
            axes[0, 0].set_title('Expert Decision Distribution (No Data)', fontweight='bold')
    else:
        axes[0, 0].text(0.5, 0.5, 'No flag summary found', ha='center', va='center')
        axes[0, 0].set_title('Expert Decision Distribution (No Data)', fontweight='bold')

    # 2. Extract data for analysis
    similarity_scores = []
    confidence_scores = []
    expert_labels = []
    decision_types = []
    ood_confidences = []

    if "detailed_results" in expert_data:
        for result in expert_data["detailed_results"]:
            if "expert_decision" in result:
                expert_decision = result["expert_decision"]

                if "unified_decision" in expert_decision:
                    unified_decision = expert_decision["unified_decision"]
                    selected_domain = unified_decision.get("selected_domain", "unknown")
                    decision_type = unified_decision.get("decision_flag", "unknown")

                    if "expert_analyses" in expert_decision and selected_domain in expert_decision["expert_analyses"]:
                        domain_analysis = expert_decision["expert_analyses"][selected_domain]

                        if "systems_analysis" in domain_analysis:
                            systems = domain_analysis["systems_analysis"]

                            if "k_medoids" in systems:
                                similarity_scores.append(systems["k_medoids"].get("similarity_score", 0.0))
                            else:
                                similarity_scores.append(0.0)

                            if "calibration" in systems:
                                confidence_scores.append(systems["calibration"].get("confidence_score", 0.0))
                            else:
                                confidence_scores.append(0.0)

                            if "ood_detection" in systems:
                                ood_confidences.append(systems["ood_detection"].get("ood_confidence", 0.0))
                            else:
                                ood_confidences.append(0.0)

                            expert_labels.append(selected_domain)
                            decision_types.append(decision_type)

    # 3. System Performance Scatter Plot
    if similarity_scores and confidence_scores:
        scatter = axes[0, 1].scatter(
            similarity_scores,
            confidence_scores,
            c=ood_confidences,
            cmap='viridis',
            alpha=0.7,
            s=100
        )
        axes[0, 1].set_xlabel('Similarity Score')
        axes[0, 1].set_ylabel('Confidence Score')
        axes[0, 1].set_title('System Performance Analysis', fontweight='bold')
        plt.colorbar(scatter, ax=axes[0, 1], label='OOD Confidence')
    else:
        axes[0, 1].text(0.5, 0.5, 'No performance data available', ha='center', va='center')
        axes[0, 1].set_title('System Performance Analysis (No Data)', fontweight='bold')

    # 4. Decision Type Distribution
    if decision_types:
        decision_counts = Counter(decision_types)
        axes[0, 2].bar(decision_counts.keys(), decision_counts.values(), color='skyblue')
        axes[0, 2].set_title('Decision Type Distribution', fontweight='bold')
        axes[0, 2].set_xlabel('Decision Type')
        axes[0, 2].set_ylabel('Count')
        plt.setp(axes[0, 2].xaxis.get_majorticklabels(), rotation=45)
    else:
        axes[0, 2].text(0.5, 0.5, 'No decision data available', ha='center', va='center')
        axes[0, 2].set_title('Decision Type Distribution (No Data)', fontweight='bold')

    # 5. Domain Distribution
    if expert_labels:
        domain_counts = Counter(expert_labels)
        axes[1, 0].bar(domain_counts.keys(), domain_counts.values(), color='lightcoral')
        axes[1, 0].set_title('Domain Distribution', fontweight='bold')
        axes[1, 0].set_xlabel('Domain')
        axes[1, 0].set_ylabel('Count')
        plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)
    else:
        axes[1, 0].text(0.5, 0.5, 'No domain data available', ha='center', va='center')
        axes[1, 0].set_title('Domain Distribution (No Data)', fontweight='bold')

    # 6. System Performance Heatmap
    if similarity_scores and confidence_scores and ood_confidences:
        data_matrix = []
        metrics = ['Similarity', 'Confidence', 'OOD Confidence']
        unique_domains = list(set(expert_labels))
        for domain in unique_domains:
            domain_indices = [j for j, label in enumerate(expert_labels) if label == domain]
            if domain_indices:
                data_matrix.append([
                    np.mean([similarity_scores[j] for j in domain_indices]),
                    np.mean([confidence_scores[j] for j in domain_indices]),
                    np.mean([ood_confidences[j] for j in domain_indices]),
                ])
        if data_matrix:
            heatmap_data = pd.DataFrame(data_matrix, index=unique_domains, columns=metrics)
            sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', ax=axes[1, 1], fmt='.3f')
            axes[1, 1].set_title('System Performance Heatmap', fontweight='bold')
        else:
            axes[1, 1].text(0.5, 0.5, 'Insufficient data for heatmap', ha='center', va='center')
            axes[1, 1].set_title('System Performance Heatmap (No Data)', fontweight='bold')
    else:
        axes[1, 1].text(0.5, 0.5, 'No performance metrics available', ha='center', va='center')
        axes[1, 1].set_title('System Performance Heatmap (No Data)', fontweight='bold')

    # 7. Score Distribution Histogram
    if similarity_scores:
        axes[1, 2].hist(similarity_scores, bins=10, alpha=0.7, color='blue', label='Similarity')
        axes[1, 2].hist(confidence_scores, bins=10, alpha=0.7, color='red', label='Confidence')
        axes[1, 2].set_title('Score Distribution', fontweight='bold')
        axes[1, 2].set_xlabel('Score Value')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].legend()
    else:
        axes[1, 2].text(0.5, 0.5, 'No score data available', ha='center', va='center')
        axes[1, 2].set_title('Score Distribution (No Data)', fontweight='bold')

    plt.tight_layout()
    output_path = _EVAL_DATA_DIR / "mycelium_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\u2713 Visualization saved to {output_path}")

    print("\n=== Mycelium Analysis Summary ===")
    if similarity_scores:
        print(f"Average Similarity Score: {np.mean(similarity_scores):.3f}")
        print(f"Average Confidence Score: {np.mean(confidence_scores):.3f}")
        print(f"Average OOD Confidence: {np.mean(ood_confidences):.3f}")
        print(f"Similarity Score Range: [{min(similarity_scores):.3f}, {max(similarity_scores):.3f}]")
        print(f"Confidence Score Range: [{min(confidence_scores):.3f}, {max(confidence_scores):.3f}]")
    if expert_labels:
        print(f"Total Decisions Analyzed: {len(expert_labels)}")
        print(f"Unique Domains: {len(set(expert_labels))}")
        print(f"Domain Distribution: {dict(Counter(expert_labels))}")
    if decision_types:
        print(f"Decision Type Distribution: {dict(Counter(decision_types))}")

    plt.show()

if __name__ == "__main__":
    visualize_mycelium_results()
