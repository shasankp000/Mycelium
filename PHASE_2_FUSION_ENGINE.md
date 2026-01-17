# Phase 2: Fusion Engine (Score Combination & Superposition Detection)

This document describes the design and implementation of the Phase 2 Fusion Engine for Project Mycelium, Layer 1 routing. It strictly adheres to the constraints: standalone, optional, deterministic, and non‑breaking. It does not modify or integrate into existing routing, and does not perform any optimization or expert selection.

## Why Fusion Is Needed

- Multiple lenses produce weak, complementary signals (e.g., semantic similarity, spectral similarity, confidence). Each lens captures different aspects of the input.
- A deterministic fusion of these signals creates an interpretable, consolidated score per domain, enabling downstream stages to reason about single‑domain vs. multi‑domain vs. ambiguous inputs.
- Fusion provides explainability by exposing per‑lens contributions to the final fused score.

## Why Weighted Averaging (Not Voting)

- Weighted averaging is deterministic and continuous, preserving nuance in score magnitudes.
- Voting discards magnitude and can be brittle when signals are close or noisy.
- Weights reflect lens reliability and can be tuned while retaining determinism; they are auto‑normalized to sum to 1.

## Normalization

- Input lens scores are normalized into [0, 1].
- Rules:
  - Empty dict → return empty dict
  - Negative values → clipped to 0
  - Values > 1 → clipped to 1
  - Non‑numeric or `None` → treated as 0
  - Deterministic (no randomness, no state).

Implementation: `ScoreNormalizer.normalize(scores: Dict[str, float]) -> Dict[str, float]`.

## Weighted Fusion

- Fusion rule per domain:
  
  fused_score = w1 * semantic + w2 * spectral + w3 * confidence

- Weights auto‑normalize to sum to 1; if total ≤ 0, fallback to equal weighting.
- Domains are the union of all keys across inputs; missing lens/domain defaults to 0.
- Output is explainable: it includes `fused_score` and per‑lens contributions (each contribution = weight × normalized lens score).

Implementation: `FusionEngine.fuse(semantic_scores, spectral_scores, confidence_scores) -> Dict[str, Dict]`.

### Output Format Example

```
{
  "astronomy": {
    "fused_score": 0.71,
    "semantic": 0.35,
    "spectral": 0.22,
    "confidence": 0.14
  }
}
```

## Superposition Detection

- Operates on fused scores (domain → fused_score).
- Algorithm:
  - If `fused_scores` is empty → `classification = NO_EXPERT_AVAILABLE`.
  - Compute mean and variance (population variance) of fused scores.
  - Candidate domains: fused_score ≥ `score_threshold`.
  - Decision rules:
    - No scores ≥ threshold → `NO_EXPERT_AVAILABLE`
    - One score ≥ threshold → `SINGLE_DOMAIN`
    - Multiple ≥ threshold AND variance ≥ `variance_threshold` → `MULTI_DOMAIN`
    - Multiple ≥ threshold AND variance < `variance_threshold` → `AMBIGUOUS`

Implementation: `SuperpositionDetector.detect(fused_scores: Dict[str, float], score_threshold: float = 0.5, variance_threshold: float = 0.1) -> Dict`.

### Output Format Example

```
{
  "classification": "MULTI_DOMAIN",
  "candidate_domains": ["astronomy", "physics"],
  "variance": 0.14,
  "fused_scores": { ... }
}
```

## Determinism

- No randomness, no learning, no gradient updates.
- All computations are deterministic: clipping to [0,1], auto‑normalized weights, union of keys, sorted iteration where applicable, population variance.
- Error handling uses graceful degradation; module never raises to callers.

## Graceful Degradation

- Empty inputs never crash; return empty fused results or `NO_EXPERT_AVAILABLE` classification.
- Missing lenses and domains default to 0.
- Any internal error in detection returns a valid `NO_EXPERT_AVAILABLE` structure.

## Minimal Tests (No PyTest Required)

Run the module directly to exercise four cases:

1. Single‑domain case
2. Multi‑domain case
3. Ambiguous case
4. No‑expert case

It prints fused scores and classification for each case.

Quick start:

```
python3 fusion_engine.py
```

## Explicit Non‑Goals (Phase 2)

- No modification to routing (`layer_1_prototype.py`).
- No integration with routing yet.
- No expert selection.
- No calls to `expert_filter`.
- No optimization (Phase 3).
- No changes to Phase 1 `spectral_analyzer.py`.

## Files

- Module: `fusion_engine.py`
- Documentation: `PHASE_2_FUSION_ENGINE.md`

Phase 2 is COMPLETE as a standalone fusion component and ready for future integration when requested.
