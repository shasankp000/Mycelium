# BERT Physics Expert - Phase 1 Setup Guide

## Quick Start Guide

### Step 1: Install Dependencies

```bash
pip install torch transformers datasets sentence-transformers pandas numpy scikit-learn tqdm
```

### Step 2: Download Physics Datasets

```bash
python download_physics_datasets.py
```

This will:
- Download Hendrycks MMLU physics dataset (~2K samples)
- Download SciQ physics questions (~3K samples)
- Combine and preprocess the datasets
- Create train/validation/test splits (80/10/10)
- Save to `training_data/physics/`

**Expected output**: ~5K+ physics text samples ready for training

### Step 3: Train BERT Physics Expert

```bash
python train_physics_bert.py
```

**Training Configuration:**
- Model: `bert-base-uncased` (110M parameters)
- Batch size: 16
- Max sequence length: 128 tokens
- Epochs: 3
- Learning rate: 2e-5
- Optimizer: AdamW with linear warmup

**Expected training time:**
- With GPU: ~10-15 minutes
- With CPU: ~1-2 hours

**Output:**
- Best model: `dummy_models/Physics_BERT/model_best/`
- Final model: `dummy_models/Physics_BERT/model_final/`
- Training history: `dummy_models/Physics_BERT/training_history.json`
- Test results: `dummy_models/Physics_BERT/test_results.json`

### Step 4: Test the BERT Expert

```bash
python bert_physics_expert.py
```

This will:
- Load the trained BERT model
- Test with sample physics queries
- Convert to unified expert system format
- Save configuration for integration

### Step 5: Integration with Unified Expert System

The BERT model can now be integrated into your unified expert system:

```python
from bert_physics_expert import BERTPhysicsExpert

# Load BERT expert
bert_expert = BERTPhysicsExpert("dummy_models/Physics_BERT/model_best")

# Use for prediction
result = bert_expert.score("What is quantum mechanics?")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.4f}")
```

## Expected Performance

### Target Metrics (Phase 1)
- **Accuracy**: 75-85%
- **F1 Score**: 70-80%
- **Confidence on physics queries**: 80-85%
- **Confidence on non-physics queries**: Lower (OOD detection)

## Directory Structure After Setup

```
Mycelium/
├── training_data/
│   └── physics/
│       ├── train.csv
│       ├── validation.csv
│       ├── test.csv
│       └── physics_combined_dataset.csv
├── dummy_models/
│   └── Physics_BERT/
│       ├── model_best/
│       │   ├── pytorch_model.bin
│       │   ├── config.json
│       │   ├── tokenizer_config.json
│       │   └── label_map.json
│       ├── model_final/
│       ├── training_history.json
│       └── test_results.json
├── download_physics_datasets.py
├── train_physics_bert.py
└── bert_physics_expert.py
```

## Troubleshooting

### Issue: CUDA out of memory
**Solution**: Reduce batch size in `train_physics_bert.py`:
```python
BATCH_SIZE = 8  # or even 4
```

### Issue: Download fails for datasets
**Solution**: Check internet connection and try again. Datasets are downloaded from HuggingFace.

### Issue: Training is too slow
**Solution**: 
- Use `distilbert-base-uncased` instead of `bert-base-uncased` (faster, slightly lower accuracy)
- Reduce max_length to 64
- Use GPU if available

## Next Steps (Phase 2)

1. Add arXiv physics abstracts (~400K samples)
2. Include Physics Stack Exchange Q&A
3. Fine-tune on domain-specific physics subfields
4. Implement advanced calibration techniques
5. Add patch training system for continuous learning

## Monitoring Training

Watch for:
- **Training loss should decrease**: Indicates model is learning
- **Validation accuracy should increase**: Indicates generalization
- **Gap between train and val**: Large gap = overfitting
- **Target**: Val accuracy 75-85%, F1 70-80%

## Configuration Options

You can modify these in `train_physics_bert.py`:

```python
# For faster training (lower accuracy)
MODEL_NAME = "distilbert-base-uncased"
EPOCHS = 2
BATCH_SIZE = 32

# For better accuracy (slower training)
MODEL_NAME = "bert-base-uncased"
EPOCHS = 5
BATCH_SIZE = 8
MAX_LENGTH = 256
```

---

**Last Updated**: October 16, 2025
