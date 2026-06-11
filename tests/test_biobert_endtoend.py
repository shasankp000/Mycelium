"""
End-to-End Validation: Expert Pre-Check → BioBERT Inference
Uses actual samples from BioBERT's test.csv to validate the complete workflow.

.. note::
   Moved from repository root to tests/ during cleanup pass (2026-05-21).
"""

import json
import datetime
import os

import pandas as pd
import pytest
from mycelium.pipeline.unified_expert_system import UnifiedExpertSystem
from mycelium.pipeline.expert_filter import ExpertFilter
from mycelium.pipeline.layer1_router import extract_tags_llama, normalize_tags


def test_with_biobert_samples():
    """Test using actual BioBERT test samples"""

    print("=" * 100)
    print("END-TO-END VALIDATION: Pre-Check → BioBERT Inference")
    print("Using actual samples from BioBERT's test.csv")
    print("=" * 100)

    # Load BioBERT test data
    print("\n📂 Loading BioBERT test data...")
    test_csv_path = "dummy_models/Medical_BERT/test.csv"

    if not os.path.exists(test_csv_path):
        pytest.skip(
            "BioBERT test.csv not found; fetch dummy_models/Medical_BERT to run this test."
        )

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

    biology_samples = test_df[test_df["Type"] == "Biology"].sample(n=5, random_state=42)
    nonbiology_samples = test_df[test_df["Type"] == "Non-Biology"].sample(
        n=5, random_state=42
    )
    test_samples = pd.concat([biology_samples, nonbiology_samples]).reset_index(
        drop=True
    )

    expert_system = UnifiedExpertSystem()
    expert_filter = ExpertFilter()

    results = []
    correct_precheck = 0
    correct_inference = 0
    correct_alignment = 0

    for idx, row in test_samples.iterrows():
        text = row["Text"]
        true_label = row["Type"]
        tags = normalize_tags(extract_tags_llama(text))

        class DummyBERTManager:
            def get_available_domains(self):
                return ["physics", "chemistry", "medical"]

        bert_manager = DummyBERTManager()
        filter_result = expert_filter.filter_experts_by_tags(
            tags, expert_system, bert_manager
        )
        relevant_domains = (
            filter_result["relevant_svm_experts"]
            + filter_result["relevant_bert_experts"]
        )

        if not relevant_domains or "medical" not in relevant_domains:
            results.append(
                {
                    "text": text[:100],
                    "true_label": true_label,
                    "tags": tags,
                    "precheck_decision": "create_new_expert",
                    "precheck_correct": False,
                    "inference_result": None,
                    "final_correct": False,
                }
            )
            continue

        decision_result = expert_system.unified_decision_analysis(
            text, filtered_experts={"medical": expert_system.experts["medical"]}
        )

        precheck_flag = decision_result["unified_decision"].get(
            "decision_flag", "create_new_expert"
        )
        precheck_domain = decision_result["unified_decision"].get(
            "selected_domain", "unknown"
        )
        precheck_confidence = decision_result["unified_decision"].get(
            "confidence_in_decision", 0.0
        )

        if true_label == "Biology":
            precheck_should_be = (
                "use_existing_expert"
                if precheck_domain == "medical"
                else "create_new_expert"
            )
        else:
            precheck_should_be = "create_new_expert"

        precheck_correct = precheck_flag == precheck_should_be
        if precheck_correct:
            correct_precheck += 1

        if precheck_flag == "use_existing_expert" and precheck_domain == "medical":
            medical_expert = expert_system.experts["medical"]
            prediction = medical_expert.predict(text)
            probs = medical_expert.predict_proba(text)
            prob_bio = float(probs[0][0])
            prob_nonbio = float(probs[0][1])
            predicted_class = "Biology" if prediction == 0 else "Non-Biology"
            confidence = prob_bio if prediction == 0 else prob_nonbio
            inference_correct = predicted_class == true_label
            if inference_correct:
                correct_inference += 1
            alignment_correct = precheck_correct and inference_correct
            if alignment_correct:
                correct_alignment += 1
            results.append(
                {
                    "text": text[:100],
                    "true_label": true_label,
                    "tags": tags,
                    "precheck_decision": precheck_flag,
                    "precheck_domain": precheck_domain,
                    "precheck_confidence": precheck_confidence,
                    "precheck_correct": precheck_correct,
                    "predicted_class": predicted_class,
                    "model_confidence": confidence,
                    "inference_correct": inference_correct,
                    "final_correct": alignment_correct,
                }
            )
        else:
            results.append(
                {
                    "text": text[:100],
                    "true_label": true_label,
                    "tags": tags,
                    "precheck_decision": precheck_flag,
                    "precheck_domain": precheck_domain,
                    "precheck_confidence": precheck_confidence,
                    "precheck_correct": precheck_correct,
                    "inference_result": "N/A (not routed to medical expert)",
                    "final_correct": precheck_correct,
                }
            )

    total = len(test_samples)
    inference_cases = sum(1 for r in results if r.get("predicted_class") is not None)

    output_file = "evaluation_data/biobert_endtoend_validation.json"
    # Ensure the output directory exists before writing (mirrors the pattern
    # in test_expert_inference_clean.py).
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.datetime.now().isoformat(),
                "test_samples": total,
                "precheck_accuracy": correct_precheck / total,
                "inference_accuracy": correct_inference / inference_cases
                if inference_cases > 0
                else 0,
                "endtoend_accuracy": correct_alignment / total,
                "detailed_results": results,
            },
            f,
            indent=4,
            ensure_ascii=False,
        )

    return results


if __name__ == "__main__":
    test_with_biobert_samples()
