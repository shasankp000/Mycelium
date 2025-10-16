# Chemistry BERT Expert - Dataset Information

## Overview
This document describes the datasets used to train the Chemistry BERT expert model for Phase 1 of the Mycelium project.

## Model Performance (Test Set)
- **Accuracy**: 95.07%
- **Precision**: 95.23%
- **Recall**: 95.07%
- **F1 Score**: 95.07%

### Per-Class Performance
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Chemistry | 93% | 98% | 95% | 102 |
| Non-Chemistry | 98% | 92% | 95% | 101 |

## Dataset Statistics
- **Total Samples**: 2,018 (balanced)
- **Chemistry Samples**: 1,009
- **Non-Chemistry Samples**: 1,009
- **Train Split**: 1,614 samples (80%)
- **Validation Split**: 201 samples (10%)
- **Test Split**: 203 samples (10%)

## Dataset Sources

### 1. Hendrycks MMLU Chemistry (1,380 samples)
**Source**: `cais/mmlu` dataset on HuggingFace

**Chemistry Subjects**:
- `college_chemistry`: College-level chemistry questions
- `high_school_chemistry`: High school chemistry questions

**Non-Chemistry Subjects** (for contrast):
- `college_physics`: Physics questions
- `high_school_physics`: Physics questions  
- `college_mathematics`: Mathematics questions
- `high_school_mathematics`: Mathematics questions
- `college_biology`: Biology questions
- `high_school_biology`: Biology questions

**Format**: Multiple-choice questions with answers, formatted as:
```
Question: [question text] Choices: [A, B, C, D]. Answer: [correct answer]. 
This is a chemistry question testing understanding of chemical principles, reactions, molecular structures, and laboratory techniques.
```

### 2. SciQ Chemistry Dataset (1,712 samples - balanced to 856 Chemistry, 856 Non-Chemistry)
**Source**: `allenai/sciq` dataset on HuggingFace

**Filtering Method**: Keyword-based classification
- **Chemistry Keywords**: atom, molecule, chemical, reaction, element, compound, bond, electron, proton, neutron, ion, acid, base, pH, oxidation, reduction, catalyst, solution, solvent, periodic table, carbon, hydrogen, oxygen, nitrogen, metal, nonmetal, organic, inorganic, chemistry, mole, molarity, concentration, titration, precipitate, solubility, valence, isotope, radioactive, nuclear
- **Non-Chemistry Keywords**: force, motion, velocity, acceleration, gravity, electromagnetic, light wave, sound wave, frequency, cell, organism, DNA, protein synthesis, evolution, photosynthesis, mitosis, meiosis, ecosystem

**Format**: Science Q&A with context:
```
Question: [question] Answer: [answer]. Context: [supporting text]
```

## Data Processing Pipeline

1. **Download**: Retrieved data from HuggingFace datasets
2. **Filtering**: 
   - Removed duplicate samples (12 duplicates found)
   - Filtered by text length (200-2000 characters for BERT compatibility)
   - Applied keyword-based classification for SciQ
3. **Balancing**: Balanced classes to equal size (1,009 per class)
4. **Splitting**: 80% train, 10% validation, 10% test

## Text Length Distribution
- **Mean**: 596.8 characters
- **Min**: 200 characters
- **Max**: 2,000 characters
- **25th percentile**: 333 characters
- **75th percentile**: 718.5 characters

## Category Distribution
After combining and balancing:
- Science Q&A (chemistry): ~42% of chemistry samples
- College Chemistry: ~3% of chemistry samples
- High School Chemistry: ~10% of chemistry samples
- Science Q&A (other): ~42% of non-chemistry samples
- Physics/Math/Biology: ~16% of non-chemistry samples

## Comparison with Physics Expert
| Metric | Chemistry | Physics |
|--------|-----------|---------|
| Total Samples | 2,018 | 3,659 |
| Test Accuracy | 95.07% | 91.01% |
| Dataset Size Ratio | 55% | 100% |

The chemistry model achieves slightly higher accuracy despite having only 55% of the physics dataset size, suggesting the chemistry/non-chemistry distinction may be somewhat clearer in the training data.

## Training Configuration
- **Model**: bert-base-uncased (109M parameters)
- **Batch Size**: 16
- **Max Sequence Length**: 128 tokens
- **Learning Rate**: 2e-5
- **Optimizer**: AdamW
- **Epochs**: 3
- **Best Validation F1**: 97.51% (Epoch 2)
- **Final Test F1**: 95.07%

## Usage Notes
- This is a Phase 1 model using synthetic/academic datasets
- Phase 2 will incorporate real-world chemistry content and user feedback
- Model shows good generalization (validation performance close to test performance)
- No signs of overfitting (unlike initial music model with 100% accuracy)

## Files in This Directory
- `pytorch_model.bin`: Trained BERT model weights
- `config.json`: Model configuration
- `vocab.txt`: BERT tokenizer vocabulary
- `training_history.json`: Training metrics per epoch
- `chemistry_combined_dataset.csv`: Full dataset (all samples)
- `train.csv`: Training split
- `validation.csv`: Validation split
- `test.csv`: Test split
- `hendrycks_chemistry.csv`: Raw Hendrycks MMLU data
- `sciq_chemistry.csv`: Raw SciQ filtered data

## Date Created
October 16, 2025
