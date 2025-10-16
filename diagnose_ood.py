"""
Diagnostic script to analyze OOD detection behavior.
"""

import json
import pandas as pd

# Load evaluation results
with open('evaluation_data/expert_evaluation_results.json', 'r') as f:
    results = json.load(f)

print("="*80)
print("🔍 OOD DETECTION ANALYSIS")
print("="*80)

# Analyze each decision
for i, result in enumerate(results['detailed_results'], 1):
    sentence = result['sentence'][:80] + "..." if len(result['sentence']) > 80 else result['sentence']
    decision = result['expert_flag']
    domain = result['selected_domain']
    
    print(f"\n{'='*80}")
    print(f"Sample #{i}: {sentence}")
    print(f"Decision: {decision} | Domain: {domain}")
    print(f"{'='*80}")
    
    # Check if expert analysis exists
    if 'expert_analyses' in result['expert_decision'] and result['expert_decision']['expert_analyses']:
        for expert_domain, analysis in result['expert_decision']['expert_analyses'].items():
            print(f"\n🔧 {expert_domain.upper()} Expert Analysis:")
            
            # K-Medoids
            k_medoids = analysis['systems_analysis']['k_medoids']
            print(f"   📊 K-Medoids Similarity: {k_medoids['similarity_score']:.4f}")
            
            # Calibration
            calibration = analysis['systems_analysis']['calibration']
            print(f"   🎯 Calibrated Confidence: {calibration['confidence_score']:.4f}")
            print(f"      Raw Confidence: {calibration['raw_confidence']:.4f}")
            print(f"      Calibration Quality: {calibration['calibration_quality']:.4f}")
            
            # OOD Detection
            ood = analysis['systems_analysis']['ood_detection']
            print(f"   🚨 OOD Detection:")
            print(f"      Is OOD: {ood['is_ood']}")
            print(f"      OOD Confidence: {ood['ood_confidence']:.4f}")
            print(f"      Reason: {ood['reason']}")
            
            if 'ood_scores' in ood:
                print(f"      OOD Scores:")
                print(f"         SVM Distance: {ood['ood_scores']['svm_distance']:.4f} (threshold: < 0.5 = OOD)")
                print(f"         NN Distance: {ood['ood_scores']['nn_distance']:.4f} (threshold: > 0.5 = OOD)")
                print(f"         Isolation Score: {ood['ood_scores']['isolation_score']:.4f} (threshold: < 0.0 = OOD)")
            
            # Unified Scores
            unified = analysis['unified_scores']
            print(f"   📈 Unified Scores:")
            print(f"      Base Similarity: {unified['base_similarity']:.4f}")
            print(f"      Base Confidence: {unified['base_confidence']:.4f}")
            print(f"      OOD Penalty: {unified['ood_penalty']:.4f}")
            print(f"      Adjusted Similarity: {unified['adjusted_similarity']:.4f}")
            print(f"      Adjusted Confidence: {unified['adjusted_confidence']:.4f}")
            print(f"      Composite Score: {unified['composite_score']:.4f}")
            print(f"      Quality Score: {unified['quality_score']:.4f}")
            
            # Recommendation
            recommendation = analysis['recommendation']
            print(f"   ✅ Recommendation:")
            print(f"      Decision: {recommendation['decision']}")
            print(f"      Confidence: {recommendation['confidence_in_decision']:.4f}")
            print(f"      Reasoning: {recommendation['reasoning']}")
            print(f"      Thresholds:")
            print(f"         High: {recommendation['thresholds_used']['high_threshold']:.4f}")
            print(f"         Medium: {recommendation['thresholds_used']['medium_threshold']:.4f}")
            print(f"         OOD Rejection: {recommendation['thresholds_used']['ood_rejection_threshold']:.4f}")
    else:
        print(f"   ℹ️ No expert analysis (no matching expert for tags: {result['tags']})")

print("\n" + "="*80)
print("📊 SUMMARY STATISTICS")
print("="*80)

# Calculate statistics
ood_penalties = []
composite_scores = []
decisions = {'use_existing_expert': 0, 'create_new_patch': 0, 'create_new_expert': 0}

for result in results['detailed_results']:
    decisions[result['expert_flag']] += 1
    
    if 'expert_analyses' in result['expert_decision'] and result['expert_decision']['expert_analyses']:
        for analysis in result['expert_decision']['expert_analyses'].values():
            ood_penalties.append(analysis['unified_scores']['ood_penalty'])
            composite_scores.append(analysis['unified_scores']['composite_score'])

print(f"\n📈 Decision Distribution:")
for decision, count in decisions.items():
    print(f"   {decision}: {count} ({count/len(results['detailed_results'])*100:.1f}%)")

if ood_penalties:
    print(f"\n🚨 OOD Penalty Statistics:")
    print(f"   Mean: {sum(ood_penalties)/len(ood_penalties):.4f}")
    print(f"   Min: {min(ood_penalties):.4f}")
    print(f"   Max: {max(ood_penalties):.4f}")
    print(f"   Count > 0.2 (blocks use_existing_expert): {sum(1 for p in ood_penalties if p > 0.2)}/{len(ood_penalties)}")

if composite_scores:
    print(f"\n📊 Composite Score Statistics:")
    print(f"   Mean: {sum(composite_scores)/len(composite_scores):.4f}")
    print(f"   Min: {min(composite_scores):.4f}")
    print(f"   Max: {max(composite_scores):.4f}")

print("\n" + "="*80)
