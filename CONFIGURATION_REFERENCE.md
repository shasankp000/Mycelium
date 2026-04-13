# Configuration Reference

**Project Mycelium — Tuning & Configuration Guide**  
**Version:** 1.0  
**Date:** January 17, 2026

---

## Overview

This document describes all configuration parameters used by the system, their defaults, safe ranges, and impacts.

**Key Principle:** All tuning is centralized in `tuning_config.py`. Modify this file to adjust system behavior **without changing algorithms or code logic**.

---

## Configuration File: `tuning_config.py`

### Location
```
/Users/abhinaygiri/Documents/Projects/Mycelium/tuning_config.py
```

### Structure
```python
# Fusion engine weights
FUSION_WEIGHTS = {
    "semantic": float,    # 0.0-1.0
    "spectral": float,    # 0.0-1.0
    "confidence": float   # 0.0-1.0
}

# Thresholds
SUPERPOSITION_VARIANCE_THRESHOLD = float     # 0.0-1.0
DOMAIN_SCORE_THRESHOLD = float               # 0.0-1.0
COVERAGE_THRESHOLD = float                   # 0.0-1.0
SOFT_STOP_PERCENTAGE = float                 # 0.0-1.0

# Limits
MAX_EXPERTS = int                            # 1-10

def validate_config():
    """Validates all parameters on import."""
```

### How Configuration Loads

1. **On import:**
   ```python
   from tuning_config import (
       FUSION_WEIGHTS,
       DOMAIN_SCORE_THRESHOLD,
       ...
   )
   ```

2. **If import fails:**
   - Each module has hardcoded defaults
   - System continues with fallback values
   - No crash (graceful degradation)

3. **Validation:**
   - On import, `validate_config()` checks all constraints
   - If invalid, raises `ConfigurationError` with details
   - Fix constraints and re-import

---

## Parameter Details

### 1. Fusion Weights

#### Purpose
Control how much each lens contributes to final domain scores.

#### Parameter
```python
FUSION_WEIGHTS = {
    "semantic": 0.45,
    "spectral": 0.35,
    "confidence": 0.20
}
```

#### Constraints
```
semantic + spectral + confidence == 1.0
0 <= each weight <= 1.0
Each weight > 0.0 (no zero weights)
```

#### Safe Ranges
| Weight | Min | Default | Max | Rationale |
|--------|-----|---------|-----|-----------|
| semantic | 0.3 | 0.45 | 0.7 | Layer 1 is most reliable (keyword-based) |
| spectral | 0.1 | 0.35 | 0.5 | Spectral is supplementary (optional) |
| confidence | 0.05 | 0.20 | 0.35 | Confidence is least reliable (flat 0.5) |

#### Impact of Changes
```
↑ semantic weight:
  - Favor keywords and Layer 1 definitions
  - More predictable, less flexible
  - Better for well-defined domains

↑ spectral weight:
  - Favor semantic similarity (SentenceTransformer)
  - More flexible, less deterministic
  - Better for nuanced/novel queries

↑ confidence weight:
  - Favor flat confidence (not recommended)
  - Currently unused; no effect
```

#### Tuning Example
**Problem:** System too rigid, missing related domains

**Solution:** Increase spectral, decrease semantic
```python
FUSION_WEIGHTS = {
    "semantic": 0.40,
    "spectral": 0.40,   # ↑ from 0.35
    "confidence": 0.20
}
# Result: More flexibility for semantic similarity
```

---

### 2. Superposition Variance Threshold

#### Purpose
Distinguish between MULTI_DOMAIN (high variance, distinct domains) and AMBIGUOUS (low variance, unclear).

#### Parameter
```python
SUPERPOSITION_VARIANCE_THRESHOLD = 0.08
```

#### Formula
```
variance = sum((score - mean)² for score in all_scores) / num_scores
if variance >= threshold:
    classification = MULTI_DOMAIN
else:
    classification = AMBIGUOUS
```

#### Safe Range
| Value | Interpretation |
|-------|-----------------|
| 0.05 | Very strict; rarely MULTI_DOMAIN |
| 0.08 | Default; balance (current) |
| 0.12 | Loose; frequently MULTI_DOMAIN |
| 0.20+ | Very loose; almost always MULTI_DOMAIN |

#### Impact of Changes
```
↓ threshold (e.g., 0.05):
  - Stricter: More AMBIGUOUS, fewer MULTI_DOMAIN
  - Better for focused routing
  - Risk: Miss legitimate multi-domain queries

↑ threshold (e.g., 0.15):
  - Looser: More MULTI_DOMAIN, fewer AMBIGUOUS
  - Better for flexible queries
  - Risk: Classify ambiguous as multi-domain
```

#### Examples

**Variance = 0.00125:**
- Scores: [0.456, 0.520, 0.450], all close
- With threshold=0.08: variance < threshold → AMBIGUOUS ✓

**Variance = 0.0234:**
- Scores: [0.35, 0.65, 0.40], spread
- With threshold=0.08: variance >= threshold → MULTI_DOMAIN ✓

---

### 3. Domain Score Threshold (Override Rule)

#### Purpose
Minimum fused score required to override ATTRIBUTE_ONLY classification (Phase 6 feature).

#### Parameter
```python
DOMAIN_SCORE_THRESHOLD = 0.45
```

#### When Used
```python
if base_classification == "ATTRIBUTE_ONLY" and object_level_domains:
    for domain in object_level_domains:
        fused_score = fused_scores.get(domain, 0.0)
        if fused_score >= DOMAIN_SCORE_THRESHOLD:  # ← Here
            selected_experts.append(domain)
            break
```

#### Safe Range
| Value | Effect |
|-------|--------|
| 0.35 | Very loose; most ATTRIBUTE_ONLY → SINGLE_DOMAIN |
| 0.45 | Default; medium (current) |
| 0.55 | Strict; few overrides |
| 0.70+ | Very strict; almost no overrides |

#### Impact of Changes
```
↓ threshold (e.g., 0.35):
  - More ATTRIBUTE_ONLY → SINGLE_DOMAIN promotions
  - Better for domain extraction
  - Risk: Promote unrelated domains

↑ threshold (e.g., 0.60):
  - Fewer overrides; stay ATTRIBUTE_ONLY longer
  - More conservative
  - Risk: Miss valid domain matches
```

#### Tuning Example
**Problem:** "Glossy red finish" should promote to aesthetics expert

**Current:**
- Fused score for aesthetics: 0.48
- Threshold: 0.45
- 0.48 >= 0.45? YES → Override applied ✓

**If threshold=0.50:**
- 0.48 >= 0.50? NO → Override NOT applied ✗

---

### 4. Coverage Threshold

#### Purpose
Minimum total coverage (sum of selected experts' values) needed to declare routing success.

#### Parameter
```python
COVERAGE_THRESHOLD = 0.70
```

#### When Used
```python
total_coverage = sum(selected_experts_values)
coverage_met = (total_coverage >= COVERAGE_THRESHOLD)
create_new_expert = not coverage_met
```

#### Safe Range
| Value | Effect |
|-------|--------|
| 0.50 | Very loose; easy to meet |
| 0.70 | Default; strict (current) |
| 0.90 | Very strict; hard to meet |

#### Impact of Changes
```
↓ threshold (e.g., 0.50):
  - Easier to declare success
  - Fewer create_new_expert=True
  - Risk: Insufficient expert coverage

↑ threshold (e.g., 0.85):
  - Harder to meet; more create_new_expert=True
  - Signal for domain expansion
  - Risk: Too aggressive expert creation
```

---

### 5. Soft-Stop Percentage

#### Purpose
Early termination of expert selection when coverage reaches (threshold * percentage).

#### Parameter
```python
SOFT_STOP_PERCENTAGE = 0.90
```

#### Formula
```
soft_stop_threshold = COVERAGE_THRESHOLD * SOFT_STOP_PERCENTAGE
                    = 0.70 * 0.90 = 0.63
```

#### When Used
```python
for each candidate:
    if coverage >= soft_stop_threshold and len(selected) > 0:
        break  # Stop early, don't need all experts
```

#### Safe Range
| Value | Stop At | Effect |
|-------|---------|--------|
| 0.80 | 0.70 * 0.80 = 0.56 | Very aggressive; stop early |
| 0.90 | 0.70 * 0.90 = 0.63 | Default; balance (current) |
| 1.00 | 0.70 * 1.00 = 0.70 | No early stop; select all needed |

#### Impact of Changes
```
↓ percentage (e.g., 0.75):
  - Stop at 52.5% of threshold
  - Faster; fewer experts selected
  - Risk: Low coverage (might still hit threshold)

↑ percentage (e.g., 1.00):
  - No early stop; select all experts up to threshold
  - Higher coverage but more experts
  - Risk: Unnecessary expert routing
```

#### Example
**Scores:** [automobile: 0.26, aesthetics: 0.25, astronomy: 0.225]

With SOFT_STOP_PERCENTAGE=0.90:
```
1. Add automobile, coverage=0.26, check: 0.26 >= 0.63? NO
2. Add aesthetics, coverage=0.51, check: 0.51 >= 0.63? NO
3. Add astronomy, coverage=0.735, check: 0.735 >= 0.63? YES → STOP
Final: 3 experts selected
```

With SOFT_STOP_PERCENTAGE=1.00:
```
1. Add automobile, coverage=0.26, coverage_met=False
2. Add aesthetics, coverage=0.51, coverage_met=False
3. Add astronomy, coverage=0.735, coverage_met=True → STOP
Final: 3 experts selected (same)
```

With SOFT_STOP_PERCENTAGE=0.75:
```
1. Add automobile, coverage=0.26, check: 0.26 >= 0.525? NO
2. Add aesthetics, coverage=0.51, check: 0.51 >= 0.525? NO
3. Add astronomy, coverage=0.735, check: 0.735 >= 0.525? YES → STOP
Final: 3 experts selected (same)
```

---

### 6. Max Experts

#### Purpose
Hard cap on number of experts selected per query.

#### Parameter
```python
MAX_EXPERTS = 3
```

#### When Used
```python
for candidate in sorted_candidates:
    if len(selected_experts) >= MAX_EXPERTS:
        break  # Hard limit
    selected_experts.append(candidate)
```

#### Safe Range
| Value | Effect |
|-------|--------|
| 1 | Only 1 expert always (too restrictive) |
| 2-3 | Default range (2-3) |
| 5 | Allow many experts (looser) |
| 10+ | Very loose; almost no cap |

#### Impact of Changes
```
↓ max (e.g., 1):
  - Only 1 expert selected
  - Very focused routing
  - Risk: Miss important domains

↑ max (e.g., 5):
  - Up to 5 experts per query
  - Broader coverage
  - Risk: Distributed routing, harder to manage
```

#### Current Setting: 3
Rationale: Balance between focus (not too many) and coverage (enough experts).

---

## Feature Flags (Module-Level)

These are not in `tuning_config.py` but in individual modules.

### In `multi_lens_router.py`

#### ENABLE_LOGGING
```python
ENABLE_LOGGING = False
```
- If True: Log every request detail (debug mode)
- If False: No logging (production)

#### LOG_SAMPLE_RATE
```python
LOG_SAMPLE_RATE = 1
```
- Log every Nth request
- 1 = all requests (if ENABLE_LOGGING=True)
- 100 = every 100th request (for high-volume)

#### ENABLE_ATTRIBUTE_OVERRIDE
```python
ENABLE_ATTRIBUTE_OVERRIDE = True
```
- If True: Use Phase 6 override rule
- If False: Skip override, keep ATTRIBUTE_ONLY

### In Module Initialization

#### use_multi_lens
```python
router = MultiLensRouter(use_multi_lens=True)
```
- If True: Use Phases 1-6
- If False: Use Layer 1 only (no fusion/optimization)

#### use_spectral
```python
router = MultiLensRouter(use_spectral=True)
```
- If True: Include spectral analysis (Phase 1)
- If False: Skip spectral (spectral_scores={})

---

## Configuration Validation

### What Gets Validated

**On `import tuning_config`:**

1. **Weights sum to 1.0**
   ```python
   assert abs(sum(FUSION_WEIGHTS.values()) - 1.0) < 0.0001
   ```

2. **All weights > 0**
   ```python
   assert all(w > 0.0 for w in FUSION_WEIGHTS.values())
   ```

3. **Thresholds in [0, 1]**
   ```python
   assert 0.0 <= SUPERPOSITION_VARIANCE_THRESHOLD <= 1.0
   assert 0.0 <= DOMAIN_SCORE_THRESHOLD <= 1.0
   assert 0.0 <= COVERAGE_THRESHOLD <= 1.0
   assert 0.0 <= SOFT_STOP_PERCENTAGE <= 1.0
   ```

4. **MAX_EXPERTS >= 1**
   ```python
   assert MAX_EXPERTS >= 1
   ```

### Error Messages

**Example:**
```
ConfigurationError: FUSION_WEIGHTS must sum to 1.0
  semantic: 0.50
  spectral: 0.35
  confidence: 0.20
  sum: 1.05 (invalid)
```

### How to Fix

1. Edit `tuning_config.py`
2. Re-import the module (or restart Python)
3. Validation runs again
4. If still invalid, get error message with details

---

## Safe Configuration Changes

### Recommended Tuning Workflow

**Step 1: Identify problem**
- E.g., "Too many ATTRIBUTE_ONLY classifications"

**Step 2: Map to parameter**
- E.g., DOMAIN_SCORE_THRESHOLD controls overrides

**Step 3: Adjust incrementally**
- E.g., 0.45 → 0.40 (10% change)

**Step 4: Test**
- Run test suite
- Check metrics: `router.get_metrics()`

**Step 5: Repeat or revert**
- If better: keep change
- If worse: revert to previous value

### Common Tuning Scenarios

#### Scenario 1: Too Many ATTRIBUTE_ONLY

**Symptoms:**
- Queries that should have domains classified as ATTRIBUTE_ONLY
- Metrics: attribute_only > 30% of total

**Solution:**
```python
# Lower override threshold to promote more ATTRIBUTE_ONLY → SINGLE_DOMAIN
DOMAIN_SCORE_THRESHOLD = 0.40  # from 0.45
```

#### Scenario 2: Too Many AMBIGUOUS

**Symptoms:**
- Similar-scoring domains classified as AMBIGUOUS instead of MULTI_DOMAIN
- Metrics: ambiguous > multi_domain

**Solution:**
```python
# Lower variance threshold to distinguish better
SUPERPOSITION_VARIANCE_THRESHOLD = 0.06  # from 0.08
```

#### Scenario 3: Low Coverage

**Symptoms:**
- Metrics: coverage_met = False frequently
- Many create_new_expert = True

**Solution:**
```python
# Lower coverage threshold or raise soft-stop
COVERAGE_THRESHOLD = 0.65  # from 0.70
# OR
SOFT_STOP_PERCENTAGE = 1.0  # from 0.90 (select all experts)
```

#### Scenario 4: Too Rigid Routing

**Symptoms:**
- Well-defined domains only, missing related concepts
- No MULTI_DOMAIN queries

**Solution:**
```python
# Increase spectral weight for semantic flexibility
FUSION_WEIGHTS = {
    "semantic": 0.40,   # from 0.45
    "spectral": 0.40,   # from 0.35
    "confidence": 0.20
}
```

---

## Monitoring Configuration Health

### Check Current Configuration

**In Python:**
```python
from tuning_config import (
    FUSION_WEIGHTS,
    SUPERPOSITION_VARIANCE_THRESHOLD,
    DOMAIN_SCORE_THRESHOLD,
    COVERAGE_THRESHOLD,
    SOFT_STOP_PERCENTAGE,
    MAX_EXPERTS,
    validate_config
)

validate_config()  # Raises if invalid
print(f"Weights: {FUSION_WEIGHTS}")
print(f"Variance threshold: {SUPERPOSITION_VARIANCE_THRESHOLD}")
print(f"Domain score threshold: {DOMAIN_SCORE_THRESHOLD}")
print(f"Coverage threshold: {COVERAGE_THRESHOLD}")
print(f"Soft-stop: {SOFT_STOP_PERCENTAGE}")
print(f"Max experts: {MAX_EXPERTS}")
```

### System Metrics

**After routing requests:**
```python
metrics = router.get_metrics()
print(f"Total requests: {metrics['total_requests']}")
print(f"ATTRIBUTE_ONLY: {metrics['attribute_only']}")
print(f"SINGLE_DOMAIN: {metrics['single_domain']}")
print(f"MULTI_DOMAIN: {metrics['multi_domain']}")
print(f"AMBIGUOUS: {metrics['ambiguous']}")
print(f"NO_EXPERT: {metrics['no_expert']}")
print(f"Override applied: {metrics['attribute_override_applied']}")
```

---

## Configuration in Production

### Deployment Best Practices

1. **Version control:**
   - Keep `tuning_config.py` in version control
   - Tag configuration with version numbers

2. **Document changes:**
   - Always include reason for change
   - Record before/after metrics

3. **Gradual rollout:**
   - Test configuration changes thoroughly
   - Rollout in stages (10% → 50% → 100%)

4. **Monitor:**
   - Watch metrics continuously
   - Alert if thresholds deviate unexpectedly

5. **Fallback:**
   - If problems: revert to last known-good config
   - All modules have hardcoded defaults

---

## Quick Reference Card

```
FUSION_WEIGHTS:
  semantic:   0.45  (Layer 1 definitions)
  spectral:   0.35  (Semantic similarity)
  confidence: 0.20  (Flat confidence)

Thresholds:
  SUPERPOSITION_VARIANCE_THRESHOLD:  0.08   (MULTI_DOMAIN vs AMBIGUOUS)
  DOMAIN_SCORE_THRESHOLD:            0.45   (Override rule)
  COVERAGE_THRESHOLD:                0.70   (Success criterion)
  SOFT_STOP_PERCENTAGE:              0.90   (Early stop at 63%)

Limits:
  MAX_EXPERTS: 3

Flags:
  ENABLE_LOGGING:        False
  LOG_SAMPLE_RATE:       1
  ENABLE_ATTRIBUTE_OVERRIDE: True
```

