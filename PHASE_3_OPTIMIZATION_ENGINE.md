# Phase 3: Optimization Engine (Minimal Expert Selection)

This document describes the design and implementation of the Phase 3 Optimization Engine for Project Mycelium, Layer 1 routing. It strictly adheres to the constraints: standalone, optional, deterministic, and non-breaking. It does not modify or integrate into existing routing, and does not perform ML or dynamic threshold tuning.

## Why Optimization Is Needed

- After Phase 2 fuses scores, many domains may have non-zero relevance, but not all are useful or necessary.
- An optimization layer must select the **minimal set of experts** that together provide sufficient coverage.
- This minimizes downstream costs (computational, query routing, expert invocations) while maintaining quality.
- It also signals when existing experts are insufficient → **create new expert**.

## What "Coverage" Means

- **Coverage** is the cumulative value of selected experts.
- For each expert: `value = fused_score × confidence`
  - `fused_score`: relevance from Phase 2.
  - `confidence`: domain expert reliability (how trusted is this expert).
- **Coverage threshold**: a target cumulative value (e.g., 0.8).
- When selected experts' values sum to ≥ threshold → sufficient coverage.

## Why Greedy Selection Is Acceptable

- **Greedy algorithm**: at each step, pick the highest-value expert and add to selection.
- **Justification**:
  - Problem is deterministic and low-cost (domains are few, typically < 20).
  - Greedy guarantees a valid selection and deterministic output.
  - Ties are broken explicitly (confidence, score, alphabetical) for reproducibility.
  - Optimal solutions in low-cost cases are not necessary; greedy is good enough.
- **Optimality**: not guaranteed, but acceptable for Layer 1 routing (coverage is approximate anyway).

## Coverage Calculator

- Rule: `value = fused_score × confidence`, clipped to [0, 1].
- Handles missing confidence → default to 1.0.
- Deterministic, no side effects.

Implementation: `CoverageCalculator.compute_value(fused_score, confidence) -> float`.

## Greedy Expert Selector

- Inputs:
  - `fused_scores`: domain → [0, 1] (from Phase 2).
  - `confidence_scores`: domain → [0, 1] (optional, defaults to 1.0).
- Parameters:
  - `coverage_threshold`: target coverage value (default 0.8).
  - `max_experts`: maximum experts to select (default 3, prevents runaway selection).
- Algorithm:
  1. If `fused_scores` is empty → return `create_new_expert = True`.
  2. For each domain, compute `value = fused_score × confidence` (cost = 1.0 for all).
  3. Sort domains by: `value / cost` (descending) → confidence (descending) → fused_score (descending) → alphabetical.
  4. Greedy loop: add best experts sequentially until coverage ≥ threshold or max_experts reached.
  5. If coverage ≥ threshold → return selected experts.
  6. Else → return `create_new_expert = True`.

### Output Format

```json
{
  "selected_experts": ["astronomy", "physics"],
  "total_coverage": 1.325,
  "coverage_threshold": 0.8,
  "coverage_met": true,
  "create_new_expert": false
}
```

If failure:

```json
{
  "selected_experts": ["astronomy"],
  "total_coverage": 0.42,
  "coverage_threshold": 0.8,
  "coverage_met": false,
  "create_new_expert": true
}
```

## Determinism & Tie-Breaking

- No randomness, no learning, no gradient updates.
- Deterministic sorting:
  1. By value (descending).
  2. By confidence (descending).
  3. By fused score (descending).
  4. Alphabetically (domain name).
- All outputs are reproducible given identical inputs.

## Graceful Degradation

- Empty inputs → return valid result with `create_new_expert = True`.
- Missing confidence → default to 1.0.
- Missing fused score → treated as 0.0.
- Any internal error → return valid result, never raise.

## Minimal Tests

Run the module directly:

```bash
python3 optimization_engine.py
```

Tests demonstrate:
1. **Single expert sufficient**: one expert meets coverage threshold.
2. **Multiple experts required**: several experts needed to reach threshold.
3. **Insufficient coverage**: even with max experts, threshold not met → signal `create_new_expert`.
4. **Empty input**: graceful handling of no fused scores.

Each test prints selected experts, coverage, and decision.

## Explicit Non-Goals (Phase 3)

- No integration with routing (`layer_1_prototype.py`).
- No modification to fusion engine (`fusion_engine.py`).
- No calls to `expert_filter` or optimization of expert selection logic.
- No dynamic threshold tuning or learning.
- No workflow changes.

## Files

- Module: `optimization_engine.py`
- Documentation: `PHASE_3_OPTIMIZATION_ENGINE.md`

Phase 3 is COMPLETE as a standalone optimization component and ready for future integration when requested.
