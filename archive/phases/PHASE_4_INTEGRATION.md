# Phase 4: Integration (Orchestration Layer)

Phase 4 connects Phases 1–3 into existing Layer 1 routing without breaking backward compatibility.

## What Phase 4 Does

Pure orchestration pipeline:

```
INPUT TEXT
   ↓
Layer 1 (multi_lens_route) [existing, always]
   ↓
Phase 1: Spectral Analysis [optional]
   ↓
Phase 2: Fusion Engine [optional]
   ↓
Phase 3: Optimization Engine [optional]
   ↓
STRUCTURED ROUTING RESULT
```

## Backward Compatibility Guarantees

1. ATTRIBUTE_ONLY returns immediately — no later phases invoked
2. `use_multi_lens=False` restores old behavior
3. Existing files untouched
4. Optional components: any phase can be disabled without crash
5. Existing tests continue to pass

## Fallback Behavior

| Failure Point | Action |
|---|---|
| Layer 1 unavailable | Return `NO_EXPERT_AVAILABLE` |
| Spectral unavailable | Continue without spectral scores |
| Fusion fails | Use base semantic scores only |
| Optimization fails | Set `create_new_expert=True` |
| Any exception | Return safe fallback result |

## Files

- Module: `multi_lens_router.py`
- Documentation: `PHASE_4_INTEGRATION.md`

> **Archived from root** during cleanup pass (2026-05-21). Original file: `PHASE_4_INTEGRATION.md`.
