"""
End-to-End Validation: Expert Pre-Check → BioBERT Inference
Uses actual samples from BioBERT's test.csv to validate the complete workflow
"""

import json
import datetime

import pandas as pd
import pytest
from unified_expert_system import UnifiedExpertSystem
from expert_filter import ExpertFilter
from layer_1_prototype import extract_tags_llama, normalize_tags

def test_with_biobert_samples():
    """Test using actual BioBERT test samples"""
    
    print("="*100)
    print("END-TO-END VALIDATION: Pre-Check → BioBERT Inference")
    print("Using actual samples from BioBERT's test.csv")
    print("="*100)
    
    # Load BioBERT test data
    print("\n📂 Loading BioBERT test data...")
    test_csv_path = "dummy_models/Medical_BERT/test.csv"

    with open(test_csv_path, "r", encoding="utf-8") as test_file:
        first_line = test_file.readline().strip()
        if first_line.startswith("version https://git-lfs.github.com/spec/v1"):
            pytest.skip(
                "BioBERT test.csv is a Git LFS pointer; fetch LFS content to run this test."
            )

    test_df = pd.read_csv(test_csv_path)
    if "Type" not in test_df.columns or "Text" not in test_df.columns:
        pytest.skip("BioBERT test.csv missing expected columns (Type, Text).")
    print(f"   Loaded {len(test_df)} test samples")
    
    # Get balanced sample: 5 Biology, 5 Non-Biology
    biology_samples = test_df[test_df['Type'] == 'Biology'].sample(n=5, random_state=42)
    nonbiology_samples = test_df[test_df['Type'] == 'Non-Biology'].sample(n=5, random_state=42)
    
    test_samples = pd.concat([biology_samples, nonbiology_samples]).reset_index(drop=True)
    
    print(f"\n✅ Selected 10 balanced samples:")
    print(f"   - 5 Biology samples")
    print(f"   - 5 Non-Biology samples")
    
    # Initialize expert system
    print("\n🚀 Initializing expert system...")
    expert_system = UnifiedExpertSystem()
    expert_filter = ExpertFilter()
    
    print("\n" + "="*100)
    print("RUNNING END-TO-END TESTS")
    print("="*100)
    
    results = []
    correct_precheck = 0
    correct_inference = 0
    correct_alignment = 0
    
    for idx, row in test_samples.iterrows():
        text = row['Text']
        true_label = row['Type']
        
        print(f"\n{'='*100}")
        print(f"TEST CASE #{idx + 1}")
        print(f"{'='*100}")
        print(f"Input: {text[:150]}...")
        print(f"True Label: {true_label}")
        
        # Step 1: Extract tags
        tags = normalize_tags(extract_tags_llama(text))
        print(f"Tags: {tags}")
        
        # Step 2: Expert pre-check decision
        class DummyBERTManager:
            def get_available_domains(self):
                return ['physics', 'chemistry', 'medical']
        
        bert_manager = DummyBERTManager()
        filter_result = expert_filter.filter_experts_by_tags(tags, expert_system, bert_manager)
        relevant_domains = filter_result['relevant_svm_experts'] + filter_result['relevant_bert_experts']
        
        print(f"Relevant experts: {relevant_domains}")
        
        if not relevant_domains or 'medical' not in relevant_domains:
            print(f"\n❌ PRE-CHECK DECISION: No medical expert matched")
            print(f"   Decision: create_new_expert")
            
            results.append({
                'text': text[:100],
                'true_label': true_label,
                'tags': tags,
                'precheck_decision': 'create_new_expert',
                'precheck_correct': False,
                'inference_result': None,
                'final_correct': False
            })
            continue
        
        # Run unified decision analysis
        decision_result = expert_system.unified_decision_analysis(
            text,
            filtered_experts={'medical': expert_system.experts['medical']}
        )
        
        # Access the correct keys in unified_decision dictionary
        precheck_flag = decision_result['unified_decision'].get('decision_flag', 'create_new_expert')
        precheck_domain = decision_result['unified_decision'].get('selected_domain', 'unknown')
        precheck_confidence = decision_result['unified_decision'].get('confidence_in_decision', 0.0)
        
        print(f"\n✅ PRE-CHECK DECISION:")
        print(f"   Flag: {precheck_flag}")
        print(f"   Domain: {precheck_domain}")
        print(f"   Confidence: {precheck_confidence:.4f}")
        
        # Determine if pre-check was correct
        if true_label == 'Biology':
            precheck_should_be = 'use_existing_expert' if precheck_domain == 'medical' else 'create_new_expert'
        else:
            precheck_should_be = 'create_new_expert'
        
        precheck_correct = (precheck_flag == precheck_should_be)
        if precheck_correct:
            correct_precheck += 1
            print(f"   Pre-check Assessment: ✅ CORRECT")
        else:
            print(f"   Pre-check Assessment: ❌ WRONG (should be {precheck_should_be})")
        
        # Step 3: Model inference (if use_existing_expert)
        if precheck_flag == 'use_existing_expert' and precheck_domain == 'medical':
            print(f"\n🔬 RUNNING BIOBERT INFERENCE:")
            
            medical_expert = expert_system.experts['medical']
            prediction = medical_expert.predict(text)
            probs = medical_expert.predict_proba(text)
            
            prob_bio = float(probs[0][0])
            prob_nonbio = float(probs[0][1])
            
            if prediction == 0:
                predicted_class = "Biology"
                confidence = prob_bio
            else:
                predicted_class = "Non-Biology"
                confidence = prob_nonbio
            
            print(f"   Predicted: {predicted_class}")
            print(f"   Confidence: {confidence:.4f}")
            print(f"   Probabilities: Biology={prob_bio:.4f}, Non-Biology={prob_nonbio:.4f}")
            
            inference_correct = (predicted_class == true_label)
            if inference_correct:
                correct_inference += 1
                print(f"   Inference Assessment: ✅ CORRECT")
            else:
                print(f"   Inference Assessment: ❌ WRONG")
            
            # Check alignment
            alignment_correct = precheck_correct and inference_correct
            if alignment_correct:
                correct_alignment += 1
            
            results.append({
                'text': text[:100],
                'true_label': true_label,
                'tags': tags,
                'precheck_decision': precheck_flag,
                'precheck_domain': precheck_domain,
                'precheck_confidence': precheck_confidence,
                'precheck_correct': precheck_correct,
                'predicted_class': predicted_class,
                'model_confidence': confidence,
                'inference_correct': inference_correct,
                'final_correct': alignment_correct
            })
        else:
            results.append({
                'text': text[:100],
                'true_label': true_label,
                'tags': tags,
                'precheck_decision': precheck_flag,
                'precheck_domain': precheck_domain,
                'precheck_confidence': precheck_confidence,
                'precheck_correct': precheck_correct,
                'inference_result': 'N/A (not routed to medical expert)',
                'final_correct': precheck_correct
            })
    
    # Summary
    print(f"\n{'='*100}")
    print("VALIDATION SUMMARY")
    print(f"{'='*100}")
    
    total = len(test_samples)
    inference_cases = sum(1 for r in results if r.get('predicted_class') is not None)
    
    print(f"\n📊 PRE-CHECK LAYER PERFORMANCE:")
    print(f"   Correct decisions: {correct_precheck}/{total} ({100*correct_precheck/total:.1f}%)")
    
    if inference_cases > 0:
        print(f"\n📊 MODEL INFERENCE PERFORMANCE:")
        print(f"   Correct predictions: {correct_inference}/{inference_cases} ({100*correct_inference/inference_cases:.1f}%)")
    
    print(f"\n📊 END-TO-END PERFORMANCE:")
    print(f"   Fully correct (pre-check + inference): {correct_alignment}/{total} ({100*correct_alignment/total:.1f}%)")
    
    # Detailed breakdown
    print(f"\n📋 BREAKDOWN BY TRUE LABEL:")
    
    biology_results = [r for r in results if r['true_label'] == 'Biology']
    nonbiology_results = [r for r in results if r['true_label'] == 'Non-Biology']
    
    bio_correct = sum(1 for r in biology_results if r.get('final_correct', False))
    nonbio_correct = sum(1 for r in nonbiology_results if r.get('final_correct', False))
    
    print(f"\n   Biology samples (should route to medical expert):")
    print(f"      Correct: {bio_correct}/{len(biology_results)} ({100*bio_correct/len(biology_results):.1f}%)")
    
    print(f"\n   Non-Biology samples (should NOT route to medical expert):")
    print(f"      Correct: {nonbio_correct}/{len(nonbiology_results)} ({100*nonbio_correct/len(nonbiology_results):.1f}%)")
    
    # Save results
    output_file = 'evaluation_data/biobert_endtoend_validation.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.datetime.now().isoformat(),
            'test_samples': total,
            'precheck_accuracy': correct_precheck / total,
            'inference_accuracy': correct_inference / inference_cases if inference_cases > 0 else 0,
            'endtoend_accuracy': correct_alignment / total,
            'detailed_results': results
        }, f, indent=4, ensure_ascii=False)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    print(f"{'='*100}")
    
    return results

if __name__ == "__main__":
    test_with_biobert_samples()
