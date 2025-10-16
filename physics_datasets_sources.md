# High-Quality Physics Datasets for BERT Training

## Recommended Datasets

### 1. **arXiv Physics Papers Dataset**
- **Source**: https://www.kaggle.com/datasets/Cornell-University/arxiv
- **Size**: 1.7M+ papers, physics subset ~400K papers
- **Quality**: High - peer-reviewed academic papers
- **Format**: JSON with abstracts, full text, categories
- **Physics Categories**: astro-ph, cond-mat, gr-qc, hep-ex, hep-lat, hep-ph, hep-th, math-ph, nlin, nucl-ex, nucl-th, physics, quant-ph
- **Best for**: Domain-specific BERT pre-training

### 2. **PhysicsOverflow / Physics Stack Exchange**
- **Source**: https://archive.org/details/stackexchange (physics.stackexchange.com dump)
- **Size**: ~200K questions and answers
- **Quality**: High - community-moderated Q&A
- **Format**: XML dumps
- **Best for**: Question-answering and conversational physics understanding

### 3. **S2ORC (Semantic Scholar Open Research Corpus)**
- **Source**: https://github.com/allenai/s2orc
- **Size**: Physics subset ~1M papers
- **Quality**: Very high - includes citations and full text
- **Format**: JSON
- **Best for**: Large-scale scientific text understanding

### 4. **CORE (COnnecting REpositories)**
- **Source**: https://core.ac.uk/
- **Size**: 30M+ open access papers (physics subset ~2M)
- **Quality**: High - academic papers
- **Format**: API or bulk download
- **Best for**: Open access physics literature

### 5. **OpenPhysics Dataset (Custom Built)**
- **Source**: Curated from multiple sources
- **Components**:
  - Physics textbooks (public domain)
  - Wikipedia physics articles
  - MIT OpenCourseWare physics content
  - Khan Academy physics transcripts
- **Best for**: Educational physics content

### 6. **PubMed Central (Physics-related papers)**
- **Source**: https://www.ncbi.nlm.nih.gov/pmc/
- **Size**: ~500K physics/biophysics papers
- **Quality**: High - peer-reviewed
- **Format**: XML, full text available
- **Best for**: Biophysics and interdisciplinary physics

## Quick Start Datasets (Smaller, Ready-to-Use)

### 1. **Physics Questions Dataset**
- **HuggingFace**: `hendrycks/competition_physics` (STEM dataset)
- **Size**: 10K+ physics problems
- **Format**: JSON with questions and answers

### 2. **SciQ Dataset**
- **HuggingFace**: `sciq`
- **Size**: 13,679 science questions (physics subset ~3K)
- **Format**: JSON

### 3. **TruthfulQA Physics Subset**
- **HuggingFace**: `truthful_qa`
- **Size**: Physics questions with verified answers
- **Format**: JSON

## Recommended Approach for Mycelium

### Phase 1: Quick Start (This Week)
1. Download **Physics Stack Exchange** dump (~200K Q&A pairs)
2. Use **hendrycks/competition_physics** from HuggingFace (10K problems)
3. Combined dataset: ~210K high-quality physics text samples

### Phase 2: Enhanced Training (Next Month)
1. Add **arXiv physics abstracts** (400K papers)
2. Include **Wikipedia physics articles** (~50K articles)
3. Total: ~660K samples for robust BERT fine-tuning

### Phase 3: Production-Ready (Long-term)
1. Full **S2ORC physics corpus** (1M+ papers)
2. Custom curated physics textbook corpus
3. Continuous learning from user interactions

## Data Preprocessing Pipeline

```python
# Recommended preprocessing steps
1. Remove duplicates
2. Filter by length (50-512 tokens ideal for BERT)
3. Label quality filtering (remove low-quality posts)
4. Domain verification (ensure physics content)
5. Train/validation/test split (80/10/10)
```

## Dataset Statistics Targets

For a robust physics expert:
- **Minimum**: 50K samples
- **Recommended**: 200K+ samples
- **Optimal**: 500K+ samples

## License Considerations

- arXiv: Free to use for research
- Stack Exchange: CC BY-SA 4.0
- Wikipedia: CC BY-SA 3.0
- S2ORC: Research use allowed
- PubMed Central: Varies by article

## Next Steps

1. Download Physics Stack Exchange dump
2. Download hendrycks/competition_physics from HuggingFace
3. Preprocess and combine datasets
4. Create train/val/test splits
5. Fine-tune BERT model

---

**Last Updated**: October 16, 2025
