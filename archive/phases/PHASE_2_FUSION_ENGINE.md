# Phase 2: Fusion Engine (Score Combination & Superposition Detection)

This document describes the design and implementation of the Phase 2 Fusion Engine for Project Mycelium, Layer 1 routing. Standalone, optional, deterministic, and non‑breaking.

## Why Fusion Is Needed

- Multiple lenses produce weak, complementary signals (semantic similarity, spectral similarity, confidence). Each lens captures different aspects of the input.
- A deterministic fusion creates an interpretable, consolidated score per domain.
- Fusion provides explainability by exposing per‑lens contributions.

## Why Weighted Averaging

- Weighted averaging is deterministic and continuous, preserving nuance in score magnitudes.
- Weights are auto‑normalized to sum to 1.

## Weighted Fusion Rule

```
fused_score = w1 * semantic + w2 * spectral + w3 * confidence
```

Output format:
```json
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

- `NO_EXPERT_AVAILABLE` → empty fused scores
- `SINGLE_DOMAIN` → one score ≥ threshold
- `MULTI_DOMAIN` → multiple ≥ threshold AND variance ≥ variance_threshold
- `AMBIGUOUS` → multiple ≥ threshold AND variance < variance_threshold

## Explicit Non-Goals

- No modification to routing (`layer_1_prototype.py`)
- No expert selection
- No optimization (Phase 3)

## Files

- Module: `fusion_engine.py`
- Documentation: `PHASE_2_FUSION_ENGINE.md`

> **Archived from root** during cleanup pass (2026-05-21). Original file: `PHASE_2_FUSION_ENGINE.md`.
