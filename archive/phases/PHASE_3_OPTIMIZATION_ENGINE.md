# Phase 3: Optimization Engine (Minimal Expert Selection)

Standalone, optional, deterministic, non-breaking. No ML or dynamic threshold tuning.

## Why Optimization Is Needed

- After Phase 2 fuses scores, many domains may have non-zero relevance, but not all are necessary.
- Select the **minimal set of experts** that together provide sufficient coverage.
- Also signals when existing experts are insufficient → **create new expert**.

## Coverage Formula

- `value = fused_score × confidence` clipped to [0, 1]
- `coverage_threshold`: target cumulative value (default 0.8)

## Greedy Selection Algorithm

1. If `fused_scores` is empty → return `create_new_expert = True`
2. For each domain, compute `value = fused_score × confidence`
3. Sort: by value/cost desc → confidence desc → fused_score desc → alphabetical
4. Greedy loop: add best experts until coverage ≥ threshold or max_experts reached
5. If coverage not met → `create_new_expert = True`

## Output Format

```json
{
  "selected_experts": ["astronomy", "physics"],
  "total_coverage": 1.325,
  "coverage_threshold": 0.8,
  "coverage_met": true,
  "create_new_expert": false
}
```

## Explicit Non-Goals

- No integration with routing yet
- No dynamic threshold tuning or learning

## Files

- Module: `optimization_engine.py`
- Documentation: `PHASE_3_OPTIMIZATION_ENGINE.md`

> **Archived from root** during cleanup pass (2026-05-21). Original file: `PHASE_3_OPTIMIZATION_ENGINE.md`.
