# Quick Start Guide - Updated Mycelium System

## System Overview

Mycelium now uses a **hybrid expert architecture**:
- **1 Legacy SVM Expert**: Medical (K-Medoids + Calibration + OOD)
- **2 BERT Experts**: Physics and Chemistry (neural classifiers)

## Running the System

### Prerequisites
```bash
# Ensure Ollama is running (for tag extraction)
# Ollama should have llama3.2:1b model installed

# Python environment with dependencies installed
pip install -r requirements.txt
```

### Execute Main Workflow
```bash
python run_workflow.py
```

This will:
1. Load Medical SVM expert
2. Load Physics and Chemistry BERT experts
3. Process 11 test samples (3 Medical, 4 Physics, 4 Chemistry)
4. Generate evaluation results and temporal analysis

## Expected Output

```
📂 Loading test samples from active expert domains...
✅ Medical (SVM): Loaded 3 samples
✅ Physics (BERT): Loaded 4 samples
✅ Chemistry (BERT): Loaded 4 samples

Initialized legacy expert system with 1 SVM experts
Initialized BERT expert system with 2 BERT experts

[Processing samples...]

Final Decision: use_existing_expert (Domain: physics, Type: BERT, Confidence: 0.997)
  BERT Match: physics (confidence: 0.997)
```

## Output Files

- `evaluation_data/sentence_tags.json` - Sentence processing results
- `evaluation_data/expert_evaluation_results.json` - Expert decisions
- `evaluation_data/tag_clusters_transformer.json` - Tag clustering
- `evaluation_data/temporal_analysis.json` - Temporal patterns

## Current Expert Domains

| Domain | Type | Status | Accuracy |
|--------|------|--------|----------|
| Medical | SVM | ✅ Active | Legacy |
| Physics | BERT | ✅ Active | 91.01% |
| Chemistry | BERT | ✅ Active | 95.07% |
| Music | SVM | ⚠️ Deprecated | - |

## Training New BERT Experts

### Physics (Already Trained)
```bash
python download_physics_datasets.py
python train_physics_bert.py
```

### Chemistry (Already Trained)
```bash
python download_chemistry_datasets.py
python train_chemistry_bert.py
```

### Medical (Pending)
Your team member should follow the same pattern:
```bash
python download_medical_datasets.py
python train_medical_bert.py
```

## Key Files

### Core System
- `unified_expert_system.py` - Legacy SVM expert system (Medical only)
- `bert_experts.py` - BERT expert class definitions
- `bert_integration.py` - Integration logic between SVM and BERT
- `run_workflow.py` - Main execution workflow

### Training Scripts
- `download_physics_datasets.py` - Physics data preparation
- `train_physics_bert.py` - Physics BERT training
- `download_chemistry_datasets.py` - Chemistry data preparation
- `train_chemistry_bert.py` - Chemistry BERT training

### Layer Systems
- `layer_1_prototype.py` - Tag extraction and clustering
- `layer_2_prototype.py` - Legacy expert model loading

## Decision Logic

The system uses this priority order:

1. **BERT High Confidence** (>0.7): Use BERT expert
2. **BERT Low Confidence** (<0.5): Request clarification
3. **Non-BERT Domain**: Use legacy SVM (Medical)
4. **No Expert Match**: Suggest creating new expert

## Troubleshooting

### Ollama Connection Error
```
ConnectionError: Failed to connect to Ollama
```
**Solution**: Start Ollama service
```bash
# Windows: Start Ollama app
# Linux/Mac: ollama serve
```

### Model Not Found Error
```
Physics BERT model not found at dummy_models/Physics_BERT
```
**Solution**: Ensure model files are in the correct location:
- `dummy_models/Physics_BERT/config.json`
- `dummy_models/Physics_BERT/model.safetensors`
- `dummy_models/Chemistry_BERT/config.json`
- `dummy_models/Chemistry_BERT/model.safetensors`

### CUDA Out of Memory
**Solution**: Reduce batch size in training scripts or set device to CPU:
```python
device = torch.device('cpu')
```

## Testing Individual Components

### Test BERT Expert Directly
```python
from bert_experts import BERTPhysicsExpert, BERTChemistryExpert

# Physics
physics_expert = BERTPhysicsExpert()
label, conf = physics_expert.predict("What is Newton's second law?")
print(f"Label: {label}, Confidence: {conf}")

# Chemistry
chem_expert = BERTChemistryExpert()
label, conf = chem_expert.predict("What is the periodic table?")
print(f"Label: {label}, Confidence: {conf}")
```

### Test Legacy Medical Expert
```python
from unified_expert_system import UnifiedExpertSystem

system = UnifiedExpertSystem()
result = system.unified_decision_analysis("What is diabetes?")
print(result['unified_decision'])
```

## Dataset Locations

### Active Datasets
- `dummy_models/Medical/medical_dataset.csv` (500 samples)
- `dummy_models/Physics_BERT/test.csv` (3,659 total, 203 test)
- `dummy_models/Chemistry_BERT/test.csv` (2,018 total, 203 test)

### Training Data Archives
- `training_data/physics/` - Physics BERT training splits
- `training_data/chemistry/` - Chemistry BERT training splits

## Performance Benchmarks

### System Initialization
- Medical SVM: ~2 seconds
- Physics BERT: ~3 seconds  
- Chemistry BERT: ~3 seconds
- **Total startup**: ~8 seconds

### Inference Speed (per sample)
- Medical SVM: ~50ms
- Physics BERT: ~100ms (CPU) / ~20ms (GPU)
- Chemistry BERT: ~100ms (CPU) / ~20ms (GPU)

## Configuration Options

### Disable Calibration/OOD
```python
system = UnifiedExpertSystem(
    enable_calibration=False,
    enable_ood_detection=False
)
```

### Change BERT Device
```python
bert_manager = BERTExpertManager(
    device=torch.device('cpu')  # or 'cuda'
)
```

## Next Steps

1. ✅ Medical SVM expert working
2. ✅ Physics BERT expert integrated
3. ✅ Chemistry BERT expert integrated
4. ⏳ Waiting for Medical BERT from team
5. 📋 Phase 1 testing and evaluation
6. 🚀 Production deployment

---
**Last Updated**: October 16, 2025  
**System Status**: ✅ Operational
