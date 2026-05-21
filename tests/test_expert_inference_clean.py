"""
Expert Inference Validation Script

Tests the complete workflow:
1. Expert pre-check layer makes decision
2. If use_existing_expert → load model and get actual prediction
3. Compare pre-check confidence vs actual model output
4. Analyze if decisions were correct

.. note::
   Moved from repository root to tests/ during cleanup pass (2026-05-21).
"""

import json
from mycelium.pipeline.unified_expert_system import UnifiedExpertSystem
from mycelium.pipeline.expert_filter import ExpertFilter
from mycelium.pipeline.layer1_router import extract_tags_llama, normalize_tags
import datetime

import pytest


def analyze_model_prediction(expert, text, domain):
    """Get detailed prediction from the model."""
    if hasattr(expert, 'predict'):
        prediction = expert.predict(text)
        probs = expert.predict_proba(text)
        confidence = float(max(probs[0]))
    else:
        text_tfidf = expert.vectorizer.transform([text])
        prediction = expert.calibrated_model.predict(text_tfidf)[0]
        probs = expert.calibrated_model.predict_proba(text_tfidf)
        confidence = float(max(probs[0]))

    if isinstance(prediction, str):
        predicted_class = 1 if prediction.lower() in ['yes', domain.lower(), domain.capitalize()] else 0
    else:
        predicted_class = int(prediction)

    if predicted_class == 1:
        interpretation = f"POSITIVE - Input belongs to {domain.upper()} domain"
        class_label = f"{domain.capitalize()}"
    else:
        interpretation = f"NEGATIVE - Input does NOT belong to {domain.upper()} domain"
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
        'is_positive': predicted_class == 1
    }


def test_expert_workflow(test_sentences):
    """Test complete expert workflow with detailed analysis."""
    try:
        expert_system = UnifiedExpertSystem()
    except Exception as exc:
        pytest.skip(f"UnifiedExpertSystem initialization failed: {exc}")
    expert_filter = ExpertFilter()

    results = []

    for i, sentence in enumerate(test_sentences, 1):
        tags = normalize_tags(extract_tags_llama(sentence))

        class DummyBERTManager:
            def get_available_domains(self):
                return [d for d in expert_system.experts.keys() if d in ['physics', 'chemistry']]

        bert_manager = DummyBERTManager()
        filter_result = expert_filter.filter_experts_by_tags(tags, expert_system, bert_manager)
        relevant_domains = filter_result['relevant_svm_experts'] + filter_result['relevant_bert_experts']

        if not relevant_domains:
            results.append({'sentence': sentence, 'tags': tags, 'decision': 'create_new_expert', 'reason': 'No matching expert'})
            continue

        filtered_experts = {domain: expert_system.experts[domain] for domain in relevant_domains if domain in expert_system.experts}
        expert_decision = expert_system.unified_decision_analysis(sentence, filtered_experts=filtered_experts)

        decision_flag = expert_decision['unified_decision']['decision_flag']
        selected_domain = expert_decision['unified_decision']['selected_domain']
        decision_confidence = expert_decision['unified_decision']['confidence_in_decision']

        if decision_flag == 'use_existing_expert':
            expert = expert_system.experts[selected_domain]
            model_result = analyze_model_prediction(expert, sentence, selected_domain)
            alignment = "CORRECT" if model_result['is_positive'] else "MISMATCH"
            conf_diff = abs(decision_confidence - model_result['confidence'])
            results.append({
                'sentence': sentence, 'tags': tags,
                'decision': decision_flag, 'domain': selected_domain,
                'precheck_confidence': decision_confidence,
                'model_prediction': model_result, 'alignment': alignment, 'confidence_diff': conf_diff
            })
        else:
            results.append({
                'sentence': sentence, 'tags': tags,
                'decision': decision_flag, 'domain': selected_domain,
                'precheck_confidence': decision_confidence
            })

    with open('evaluation_data/inference_validation_results.json', 'w', encoding='utf-8') as f:
        use_existing_cases = [r for r in results if r.get('decision') == 'use_existing_expert']
        correct = sum(1 for r in use_existing_cases if r.get('alignment') == 'CORRECT')
        mismatches = sum(1 for r in use_existing_cases if r.get('alignment') == 'MISMATCH')
        json.dump({
            'timestamp': datetime.datetime.now().isoformat(),
            'total_cases': len(results),
            'use_existing_expert_cases': len(use_existing_cases),
            'correct_predictions': correct,
            'mismatches': mismatches,
            'detailed_results': results
        }, f, indent=4, ensure_ascii=False)

    return results


@pytest.fixture
def test_sentences():
    return [
        "The photoelectric effect demonstrates that light has particle properties, as Einstein showed in 1905.",
        "Quantum entanglement occurs when particles interact and remain correlated regardless of distance.",
        "The Heisenberg uncertainty principle states that you cannot simultaneously know both position and momentum.",
        "The Haber process synthesizes ammonia from nitrogen and hydrogen using an iron catalyst.",
        "Covalent bonds form when atoms share electrons to achieve stable electron configurations.",
        "Oxidation-reduction reactions involve the transfer of electrons between chemical species.",
        "Metastatic carcinoma requires systemic chemotherapy rather than localized radiation treatment.",
        "Immunotherapy has revolutionized cancer treatment by harnessing the body's immune system.",
        "Radiation therapy uses high-energy particles to destroy cancer cells.",
        "People still got cancer before we knew about radiation it was causing harm.",
        "The stock market crashed in 1929, leading to the Great Depression.",
        "Artificial intelligence models require large datasets for training.",
    ]


if __name__ == "__main__":
    sentences = [
        "The photoelectric effect demonstrates that light has particle properties, as Einstein showed in 1905.",
        "The Haber process synthesizes ammonia from nitrogen and hydrogen using an iron catalyst.",
        "Metastatic carcinoma requires systemic chemotherapy rather than localized radiation treatment.",
        "Radiation therapy uses high-energy particles to destroy cancer cells.",
        "The stock market crashed in 1929, leading to the Great Depression.",
    ]
    test_expert_workflow(sentences)
