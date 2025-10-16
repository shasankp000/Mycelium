"""
Expert Inference Validation Script

This script tests the complete workflow:
1. Expert pre-check layer makes decision (use_existing_expert, create_new_patch, create_new_expert)
2. If use_existing_expert  Load model and get actual prediction
3. Compare pre-check confidence vs actual model output
4. Analyze if decisions were correct

This helps us determine:
- Is the pre-check layer accurate?
- Are the models overfitting/underfitting?
- Do high-confidence decisions match actual predictions?
"""

import json
import torch
from unified_expert_system import UnifiedExpertSystem
from expert_filter import ExpertFilter
from layer_1_prototype import extract_tags_llama, normalize_tags
import datetime

def analyze_model_prediction(expert, text, domain):
    """
    Get detailed prediction from the model.
    
    Returns:
        dict with prediction, confidence, probabilities, and interpretation
    """
    # Get prediction (works for both SVM and BERT experts)
    if hasattr(expert, 'predict'):
        # BERT expert has predict method
        prediction = expert.predict(text)
        probs = expert.predict_proba(text)
        confidence = float(max(probs[0]))
    else:
        # SVM expert uses calibrated_model
        text_tfidf = expert.vectorizer.transform([text])
        prediction = expert.calibrated_model.predict(text_tfidf)[0]
        probs = expert.calibrated_model.predict_proba(text_tfidf)
        confidence = float(max(probs[0]))
    
    # Get class labels (0 = negative, 1 = positive)
    # Handle both numeric (BERT: 0,1) and string (SVM: "Yes"/"No") predictions
    if isinstance(prediction, str):
        predicted_class = 1 if prediction.lower() in ['yes', domain.lower(), domain.capitalize()] else 0
    else:
        predicted_class = int(prediction)
    
    positive_class = 1
    negative_class = 0
    
    # Interpret prediction
    if predicted_class == positive_class:
        interpretation = f" POSITIVE - Input belongs to {domain.upper()} domain"
        class_label = f"{domain.capitalize()}"
    else:
        interpretation = f" NEGATIVE - Input does NOT belong to {domain.upper()} domain"
        class_label = f"Not {domain.capitalize()}"
    
    return {
        'predicted_class': predicted_class,
        'class_label': class_label,
        'confidence': confidence,
        'probabilities': {
            f'P(Not {domain.capitalize()})': float(probs[0][0]),
            f'P({domain.capitalize()})': float(probs[0][1])
        },
        'interpretation': interpretation,
        'is_positive': predicted_class == positive_class
    }

def test_expert_workflow(test_sentences):
    """
    Test complete expert workflow with detailed analysis.
    """
    print("="*100)
    print("EXPERT INFERENCE VALIDATION TEST")
    print("="*100)
    print("\nThis test validates:")
    print("1. Expert pre-check layer decisions")
    print("2. Actual model predictions when use_existing_expert is called")
    print("3. Alignment between pre-check confidence and model output")
    print("="*100)
    
    # Initialize system
    print("\n Initializing expert system...")
    expert_system = UnifiedExpertSystem()
    expert_filter = ExpertFilter()
    print(f" Initialized {len(expert_system.experts)} experts\n")
    
    results = []
    
    for i, sentence in enumerate(test_sentences, 1):
        print("\n" + "="*100)
        print(f" TEST CASE #{i}")
        print("="*100)
        print(f"Input: {sentence[:100]}{'...' if len(sentence) > 100 else ''}")
        
        # Extract tags
        tags = normalize_tags(extract_tags_llama(sentence))
        print(f"Tags: {tags}")
        
        # Filter experts by tags
        class DummyBERTManager:
            def get_available_domains(self):
                return [d for d in expert_system.experts.keys() if d in ['physics', 'chemistry']]
        
        bert_manager = DummyBERTManager()
        filter_result = expert_filter.filter_experts_by_tags(tags, expert_system, bert_manager)
        relevant_domains = filter_result['relevant_svm_experts'] + filter_result['relevant_bert_experts']
        
        if not relevant_domains:
            print(f"\n No expert available for tags: {tags}")
            print(f"Decision: create_new_expert")
            results.append({
                'sentence': sentence,
                'tags': tags,
                'decision': 'create_new_expert',
                'reason': 'No matching expert'
            })
            continue
        
        print(f"Relevant experts: {relevant_domains}")
        
        # Get expert decision
        filtered_experts = {domain: expert_system.experts[domain] for domain in relevant_domains if domain in expert_system.experts}
        expert_decision = expert_system.unified_decision_analysis(sentence, filtered_experts=filtered_experts)
        
        decision_flag = expert_decision['unified_decision']['decision_flag']
        selected_domain = expert_decision['unified_decision']['selected_domain']
        decision_confidence = expert_decision['unified_decision']['confidence_in_decision']
        
        print(f"\n PRE-CHECK DECISION:")
        print(f"   Flag: {decision_flag}")
        print(f"   Domain: {selected_domain}")
        print(f"   Confidence: {decision_confidence:.4f}")
        
        # Get detailed scores
        if selected_domain in expert_decision['expert_analyses']:
            analysis = expert_decision['expert_analyses'][selected_domain]
            print(f"\n PRE-CHECK METRICS:")
            print(f"   Similarity: {analysis['unified_scores']['base_similarity']:.4f}")
            print(f"   Confidence: {analysis['unified_scores']['base_confidence']:.4f}")
            print(f"   OOD Penalty: {analysis['unified_scores']['ood_penalty']:.4f}")
            print(f"   Composite: {analysis['unified_scores']['composite_score']:.4f}")
            print(f"   Reasoning: {analysis['recommendation']['reasoning']}")
        
        # If use_existing_expert, get actual model prediction
        if decision_flag == 'use_existing_expert':
            print(f"\n ACTUAL MODEL INFERENCE:")
            print(f"   Loading {selected_domain.upper()} expert for validation...")
            
            expert = expert_system.experts[selected_domain]
            model_result = analyze_model_prediction(expert, sentence, selected_domain)
            
            print(f"\n    MODEL PREDICTION:")
            print(f"      {model_result['interpretation']}")
            print(f"      Predicted Class: {model_result['class_label']}")
            print(f"      Confidence: {model_result['confidence']:.4f}")
            print(f"      Probabilities:")
            for class_name, prob in model_result['probabilities'].items():
                print(f"         {class_name}: {prob:.4f}")
            
            # Analyze alignment
            print(f"\n    ALIGNMENT ANALYSIS:")
            
            # Check if model agrees with pre-check
            if model_result['is_positive']:
                alignment = " CORRECT"
                alignment_text = f"Model confirms input belongs to {selected_domain} (as pre-check suggested)"
            else:
                alignment = " MISMATCH"
                alignment_text = f"Model says input does NOT belong to {selected_domain} (pre-check may be wrong!)"
            
            print(f"      Pre-check: use_existing_expert ({selected_domain})")
            print(f"      Model: {model_result['class_label']}")
            print(f"      Alignment: {alignment}")
            print(f"      Analysis: {alignment_text}")
            
            # Confidence comparison
            precheck_conf = decision_confidence
            model_conf = model_result['confidence']
            conf_diff = abs(precheck_conf - model_conf)
            
            print(f"\n    CONFIDENCE COMPARISON:")
            print(f"      Pre-check confidence: {precheck_conf:.4f}")
            print(f"      Model confidence: {model_conf:.4f}")
            print(f"      Difference: {conf_diff:.4f}")
            
            if conf_diff < 0.1:
                conf_alignment = " Well aligned"
            elif conf_diff < 0.2:
                conf_alignment = " Moderate difference"
            else:
                conf_alignment = " Large discrepancy"
            
            print(f"      Assessment: {conf_alignment}")
            
            results.append({
                'sentence': sentence,
                'tags': tags,
                'decision': decision_flag,
                'domain': selected_domain,
                'precheck_confidence': precheck_conf,
                'model_prediction': model_result,
                'alignment': alignment,
                'confidence_diff': conf_diff
            })
        else:
            print(f"\n    Decision is '{decision_flag}' - no model inference needed")
            results.append({
                'sentence': sentence,
                'tags': tags,
                'decision': decision_flag,
                'domain': selected_domain,
                'precheck_confidence': decision_confidence
            })
    
    # Summary
    print("\n" + "="*100)
    print(" VALIDATION SUMMARY")
    print("="*100)
    
    use_existing_cases = [r for r in results if r.get('decision') == 'use_existing_expert']
    
    if use_existing_cases:
        print(f"\n use_existing_expert cases: {len(use_existing_cases)}")
        
        correct = sum(1 for r in use_existing_cases if r.get('alignment') == ' CORRECT')
        mismatches = sum(1 for r in use_existing_cases if r.get('alignment') == ' MISMATCH')
        
        print(f"\n   Model Alignment:")
        print(f"      Correct predictions: {correct}/{len(use_existing_cases)} ({correct/len(use_existing_cases)*100:.1f}%)")
        print(f"      Mismatches: {mismatches}/{len(use_existing_cases)} ({mismatches/len(use_existing_cases)*100:.1f}%)")
        
        avg_conf_diff = sum(r.get('confidence_diff', 0) for r in use_existing_cases) / len(use_existing_cases)
        print(f"\n   Average confidence difference: {avg_conf_diff:.4f}")
        
        if mismatches > 0:
            print(f"\n    MISMATCHES DETECTED:")
            for r in use_existing_cases:
                if r.get('alignment') == ' MISMATCH':
                    print(f"\n      Input: {r['sentence'][:80]}...")
                    print(f"      Pre-check said: use {r['domain']} expert")
                    print(f"      Model said: {r['model_prediction']['interpretation']}")
                    print(f"       Pre-check layer may have FALSE POSITIVE")
    
    other_decisions = [r for r in results if r.get('decision') != 'use_existing_expert']
    if other_decisions:
        print(f"\n Other decisions: {len(other_decisions)}")
        for decision in set(r.get('decision') for r in other_decisions):
            count = sum(1 for r in other_decisions if r.get('decision') == decision)
            print(f"   {decision}: {count}")
    
    # Save detailed results
    with open('evaluation_data/inference_validation_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.datetime.now().isoformat(),
            'total_cases': len(results),
            'use_existing_expert_cases': len(use_existing_cases),
            'correct_predictions': correct if use_existing_cases else 0,
            'mismatches': mismatches if use_existing_cases else 0,
            'detailed_results': results
        }, f, indent=4, ensure_ascii=False)
    
    print(f"\n Detailed results saved to: evaluation_data/inference_validation_results.json")
    print("="*100)
    
    return results

if __name__ == "__main__":
    # Test sentences covering different domains
    test_sentences = [
        # Physics (should match physics expert)
        "The photoelectric effect demonstrates that light has particle properties, as Einstein showed in 1905.",
        "Quantum entanglement occurs when particles interact and remain correlated regardless of distance.",
        "The Heisenberg uncertainty principle states that you cannot simultaneously know both position and momentum.",
        
        # Chemistry (should match chemistry expert)
        "The Haber process synthesizes ammonia from nitrogen and hydrogen using an iron catalyst.",
        "Covalent bonds form when atoms share electrons to achieve stable electron configurations.",
        "Oxidation-reduction reactions involve the transfer of electrons between chemical species.",
        
        # Medicine (should match medical expert)
        "Metastatic carcinoma requires systemic chemotherapy rather than localized radiation treatment.",
        "Immunotherapy has revolutionized cancer treatment by harnessing the body's immune system.",
        
        # Ambiguous cases (testing edge cases)
        "Radiation therapy uses high-energy particles to destroy cancer cells.",  # Physics + Medical overlap
        "People still got cancer before we knew about radiation it was causing harm.",  # Previous test case
        
        # Off-domain (should be create_new_expert)
        "The stock market crashed in 1929, leading to the Great Depression.",
        "Artificial intelligence models require large datasets for training."
    ]
    
    test_expert_workflow(test_sentences)
