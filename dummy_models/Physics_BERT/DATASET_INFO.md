# Physics BERT Expert - Training Information

## Model Performance

**Test Results:**
- **Accuracy**: 91.01%
- **Precision**: 91.07%
- **Recall**: 91.01%
- **F1 Score**: 91.00%

**Per-Class Performance:**
- **Non-Physics**: Precision 0.90, Recall 0.93, F1 0.91 (184 samples)
- **Physics**: Precision 0.93, Recall 0.89, F1 0.91 (183 samples)

## Training Configuration

- **Model**: bert-base-uncased
- **Epochs**: 3
- **Batch Size**: 16
- **Max Sequence Length**: 128 tokens
- **Learning Rate**: 2e-5
- **Optimizer**: AdamW
- **Device**: CUDA (GPU)

## Datasets Used

### 1. Hendrycks MMLU (Massive Multitask Language Understanding)
- **Source**: HuggingFace `cais/mmlu`
- **Physics Subjects**:
  - college_physics
  - high_school_physics
- **Non-Physics Subjects**:
  - college_biology
  - college_chemistry
  - high_school_biology
- **Total Samples**: 562 (281 Physics, 281 Non-Physics)
- **Description**: Competition-level academic questions from standardized tests
- **License**: MIT License

### 2. SciQ Dataset
- **Source**: HuggingFace `sciq`
- **Total Samples**: 4,000 (2,000 Physics, 2,000 Non-Physics)
- **Description**: Science question-answering dataset with support text
- **Filtering**: Keyword-based classification for physics vs non-physics science
- **License**: CC BY 4.0

## Combined Dataset Statistics

**After Preprocessing:**
- **Total Samples**: 3,659
- **Training Set**: 2,927 samples (1,544 Physics, 1,383 Non-Physics)
- **Validation Set**: 365 samples
- **Test Set**: 367 samples

**Preprocessing Steps:**
1. Combined Hendrycks MMLU and SciQ datasets
2. Removed 24 duplicate samples
3. Filtered by length (200-2,000 characters for optimal BERT processing)
4. Split into train/val/test (80/10/10)

## Dataset Files in This Directory

- `train.csv` - Training dataset (2,927 samples)
- `validation.csv` - Validation dataset (365 samples)
- `test.csv` - Test dataset (367 samples)
- `physics_combined_dataset.csv` - Full combined dataset before splitting
- `hendrycks_physics.csv` - Original Hendrycks MMLU subset
- `sciq_physics.csv` - Original SciQ subset

## Model Files

- `model_best/` - Best model checkpoint (highest F1 score during training)
- `model_final/` - Final model after all epochs
- `training_history.json` - Training metrics per epoch
- `test_results.json` - Final test set evaluation results

## Usage

To load this model for inference:

```python
from bert_physics_expert import BERTPhysicsExpert

expert = BERTPhysicsExpert("dummy_models/Physics_BERT/model_best")
result = expert.score("What is quantum entanglement?")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.4f}")
```

## Future Improvements (Phase 2)

Planned dataset additions for enhanced performance:
- arXiv physics paper abstracts (~400K samples)
- Physics Stack Exchange Q&A (~200K samples)
- Physics textbook corpus
- OpenCourseWare physics content

**Target Performance**: 95%+ accuracy with larger, more diverse dataset

---

**Training Date**: October 16, 2025
**Model Version**: Phase 1
**Status**: ✅ Ready for production use
