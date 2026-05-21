# High-Quality Physics Datasets for BERT Training

> **Moved from repository root** to `scripts/` during cleanup pass (2026-05-21).
> Last content update: October 16, 2025.

## Recommended Datasets

### 1. arXiv Physics Papers Dataset
- **Source:** https://www.kaggle.com/datasets/Cornell-University/arxiv
- **Size:** 1.7M+ papers; physics subset ~400K
- **Quality:** High — peer-reviewed academic papers
- **Physics Categories:** astro-ph, cond-mat, gr-qc, hep-ex, hep-ph, hep-th, math-ph, nlin, nucl-ex, nucl-th, physics, quant-ph

### 2. PhysicsOverflow / Physics Stack Exchange
- **Source:** https://archive.org/details/stackexchange
- **Size:** ~200K questions and answers
- **License:** CC BY-SA 4.0

### 3. S2ORC (Semantic Scholar Open Research Corpus)
- **Source:** https://github.com/allenai/s2orc
- **Size:** ~1M physics papers

### 4. CORE (COnnecting REpositories)
- **Source:** https://core.ac.uk/
- **Size:** 30M+ open access papers (physics subset ~2M)

## Quick Start (HuggingFace)

| Dataset | HF ID | Size | Notes |
|---|---|---|---|
| Physics Questions | `hendrycks/competition_physics` | 10K+ | STEM problems |
| SciQ | `sciq` | 13,679 | Physics subset ~3K |
| TruthfulQA | `truthful_qa` | Varies | Verified answers |

## Recommended Approach for Mycelium

**Phase 1 (Quick Start):** Physics Stack Exchange dump + hendrycks/competition_physics → ~210K samples  
**Phase 2 (Enhanced):** + arXiv abstracts + Wikipedia articles → ~660K samples  
**Phase 3 (Production):** Full S2ORC physics corpus → 1M+ samples

## Dataset Statistics Targets

- **Minimum:** 50K samples
- **Recommended:** 200K+ samples
- **Optimal:** 500K+ samples

## License Summary

| Source | License |
|---|---|
| arXiv | Free for research |
| Stack Exchange | CC BY-SA 4.0 |
| Wikipedia | CC BY-SA 3.0 |
| S2ORC | Research use |
| PubMed Central | Varies by article |
