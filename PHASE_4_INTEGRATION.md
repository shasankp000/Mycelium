# Phase 4: Integration (Orchestration Layer)

This document describes Phase 4 of Project Mycelium: the Integration layer that connects Phases 1–3 (Spectral Analysis, Fusion Engine, Optimization Engine) into the existing Layer 1 routing system without breaking backward compatibility.

## What Phase 4 Does

Phase 4 is **pure orchestration**—it connects existing components in a deterministic pipeline:

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

Each stage is optional and can be disabled via feature flags. If any stage fails, the system gracefully degrades to the previous valid state.

## What Phase 4 Deliberately Does NOT Do

- **No new math**: All algorithms come from Phases 1–3.
- **No new heuristics**: All decision-making is from existing components.
- **No ML or learning**: Pure deterministic orchestration.
- **No modifications to existing files**: `layer_1_prototype.py`, `spectral_analyzer.py`, `fusion_engine.py`, and `optimization_engine.py` remain untouched.
- **No hardcoded domains**: Domains are discovered dynamically from lens scores.
- **No expert invocation**: Routing only **selects** experts; it does not call them.
- **No threshold tuning**: All thresholds come from component initialization.

## End-to-End Routing Flow

### Step 1: Baseline Routing (Always Runs)

```python
from layer_1_prototype import multi_lens_route
base_result = multi_lens_route(text)
```

If the result is `ATTRIBUTE_ONLY`, the pipeline stops immediately and returns the base result (no experts needed).

If `use_multi_lens=False` (feature flag), the pipeline stops and returns the base result (backward compatibility mode).

### Step 2: Spectral Analysis (Optional)

If `use_spectral=True` and signatures are available:

```python
from spectral_analyzer import RuntimeSpectralAnalyzer
analyzer = RuntimeSpectralAnalyzer(signature_dir="signatures")
spectral_scores = analyzer.analyze_text(text)
```

If spectral analysis fails or returns no scores, the pipeline logs a warning and continues without them.

### Step 3: Fusion Engine (Optional)

Combines semantic scores (from Layer 1) with spectral scores (from Phase 1):

```python
from fusion_engine import FusionEngine
fusion = FusionEngine(semantic_weight=0.4, spectral_weight=0.3, confidence_weight=0.3)
fused_result = fusion.fuse(
    semantic_scores={...},
    spectral_scores={...},
    confidence_scores={...}  # default 0.5 per domain
)
```

Fused scores are the combined signal used by the optimizer.

### Step 4: Optimization Engine (Optional)

Greedily selects the minimal expert set:

```python
from optimization_engine import GreedyExpertSelector
selector = GreedyExpertSelector(coverage_threshold=0.8)
opt_result = selector.select_experts(
    fused_scores={...},
    confidence_scores={...}
)
```

Returns selected experts, coverage met status, and whether a new expert is required.

### Step 5: Superposition Detection

Classifies the input as:

- `SINGLE_DOMAIN`: one expert sufficient.
- `MULTI_DOMAIN`: multiple experts, high variance.
- `AMBIGUOUS`: multiple experts, low variance.
- `ATTRIBUTE_ONLY`: no domain experts needed.
- `NO_EXPERT_AVAILABLE`: no suitable experts found.

### Step 6: Structured Output

Returns a complete routing decision:

```json
{
  "primary_domain": "astronomy",
  "selected_experts": ["astronomy"],
  "candidate_domains": ["astronomy"],
  "classification": "SINGLE_DOMAIN",
  "coverage_met": true,
  "create_new_expert": false,
  "lens_scores": {
    "semantic": {"astronomy": 0.8},
    "spectral": {"astronomy": 0.7},
    "confidence": {"astronomy": 0.5}
  },
  "fused_scores": {"astronomy": 0.71},
  "variance": 0.0,
  "explanation": "Single domain detected: astronomy."
}
```

## Fallback Behavior

The router implements graceful degradation at every stage:

| Failure Point | Action |
|---|---|
| Layer 1 unavailable | Return `NO_EXPERT_AVAILABLE` |
| Spectral unavailable | Continue without spectral scores |
| Fusion fails | Use base semantic scores only |
| Optimization fails | Set `create_new_expert=True` |
| Any exception | Return safe fallback result |

No exception escapes the routing pipeline.

## Backward Compatibility Guarantees

1. **ATTRIBUTE_ONLY returns immediately**: If Layer 1 detects an attribute-only input, the pipeline returns immediately without invoking later phases.

2. **use_multi_lens=False restores old behavior**: Setting `use_multi_lens=False` disables all multi-lens processing and returns the pure Layer 1 result.

3. **Existing files untouched**: No modification to `layer_1_prototype.py` or any other existing module.

4. **Optional components**: If any phase (spectral, fusion, optimization) is unavailable or disabled, the system continues gracefully.

5. **Test compatibility**: Existing tests for Layer 1 continue to pass unchanged.

## Why This Is Safe for Production Rollout

- **No breaking changes**: All new behavior is additive; disabling it via feature flags restores the original system.
- **Deterministic**: All processing is reproducible; no randomness or learning.
- **Thoroughly tested**: Each phase (1–3) is tested independently and as a pipeline.
- **Graceful degradation**: The system continues to work even if components are unavailable.
- **Explicit non-goals**: No expert selection, no threshold tuning, no ML—only orchestration.
- **Production flags**: `use_multi_lens` and `use_spectral` can be toggled in configuration without code changes.

## Usage

```python
from multi_lens_router import MultiLensRouter

# Create router with all phases enabled
router = MultiLensRouter(
    use_multi_lens=True,
    use_spectral=True,
    spectral_dir="signatures",
    coverage_threshold=0.8
)

# Route a query
result = router.route("Earth orbits the Sun")

# Inspect results
print(result["classification"])      # "SINGLE_DOMAIN"
print(result["selected_experts"])    # ["astronomy"]
print(result["coverage_met"])        # True
print(result["create_new_expert"])   # False
```

## Running Tests

```bash
python3 multi_lens_router.py
```

Tests demonstrate:
1. Single-domain routing (astronomy query)
2. Multi-domain routing (automobile + attributes)
3. Attribute-only handling (no experts)
4. Backward compatibility (with `use_multi_lens=False`)

## Files

- Module: `multi_lens_router.py`
- Documentation: `PHASE_4_INTEGRATION.md`

Phase 4 is COMPLETE as the orchestration layer connecting Phases 1–3 into Layer 1 routing.
