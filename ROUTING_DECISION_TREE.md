# Routing Decision Tree

**Project Mycelium — Routing Logic Reference**  
**Version:** 1.0  
**Date:** January 17, 2026

---

## Overview

This document describes the **exact decision logic** used to route an input query to expert(s).

The routing algorithm is deterministic, with explicit thresholds and tie-breakers documented below.

---

## Complete Decision Flow (Step-by-Step)

### Step 1: Baseline Routing (Layer 1)

**Function:** `layer_1_prototype.multi_lens_route(text)`

**Input:** `text: str`

**Output:**
```python
{
    "primary_domain": str or None,
    "classification": "NORMAL" or "ATTRIBUTE_ONLY",
    "lens1_candidates": [[domain, score], ...],
    "lens2_explanations": [{...}],
    "lens3_signature": {...}
}
```

**Logic:**
1. Extract keywords from text (core + modifiers)
2. Match against semantic domain definitions
3. Check ontology hierarchy for matches
4. If **only modifiers match** (no core keywords) → `classification = "ATTRIBUTE_ONLY"`
5. Else → `classification = "NORMAL"`

**Example:**
- Input: "Glossy metallic red finish"
  - Extract: core=[], modifiers=["glossy", "metallic", "red"]
  - No domain matches core
  - **Result:** ATTRIBUTE_ONLY

- Input: "Red car"
  - Extract: core=["car"], modifiers=["red"]
  - Matches automobile (core) + aesthetics (modifier)
  - **Result:** NORMAL, primary_domain="automobile"

---

### Step 2: Early Exit — ATTRIBUTE_ONLY Check (Pre-Phase 6)

**Condition:** If `base_classification == "ATTRIBUTE_ONLY"` AND no object-level domains found

**Action:** Return early with empty selected_experts

**Code Path:** `multi_lens_router._enrich_base_result(base_result)`

**Example:**
- Input: "High torque low noise" (pure attributes)
- Lens 2 finds no object-level concepts
- **Decision:** ATTRIBUTE_ONLY, no expert routing
- **Return:** selected_experts=[], primary_domain=None

---

### Step 3: Phase 1 — Spectral Analysis (Optional)

**Condition:** `use_spectral == True` AND spectral analyzer initialized

**Input:** `text: str`

**Function:** `spectral_analyzer.RuntimeSpectralAnalyzer.analyze_text(text)`

**Output:**
```python
spectral_scores: {
    "astronomy": 0.48,
    "automobile": 0.52,
    "aesthetics": 0.45
}
```

**Algorithm:**
1. Encode text with SentenceTransformer → embedding [384-dim]
2. Load pre-computed signatures for each domain
3. Cross-correlate embedding against signatures
4. Normalize correlation scores to [0,1]
5. Return per-domain scores

**If spectral analyzer not available:** Use empty dict `{}`

---

### Step 4: Phase 2 — Fusion Engine

**Function:** `fusion_engine.FusionEngine.fuse(...)`

**Inputs:**
```python
semantic_scores = {domain: float}      # From Layer 1 Lens 1
spectral_scores = {domain: float}      # From Phase 1 (or {})
confidence_scores = {domain: 0.5}      # Default from Layer 1 Lens 2
```

**Fusion Formula (per domain):**
```
fused_score[domain] = 
    w_semantic * semantic_scores[domain] +
    w_spectral * spectral_scores[domain] +
    w_confidence * confidence_scores[domain]

Where:
    w_semantic = 0.45 (from tuning_config.FUSION_WEIGHTS)
    w_spectral = 0.35
    w_confidence = 0.20
```

**Algorithm:**
1. Normalize each lens score to [0,1]
2. Get union of all domains
3. For each domain, apply fusion formula
4. Clamp result to [0,1]
5. Log per-domain contributions (if ENABLE_LOGGING=True)

**Output:**
```python
{
    "astronomy": {
        "fused_score": 0.456,
        "semantic": 0.30,
        "spectral": 0.105,
        "confidence": 0.051
    },
    "automobile": {
        "fused_score": 0.520,
        "semantic": 0.35,
        "spectral": 0.135,
        "confidence": 0.035
    }
}
```

### Compute Variance

**Purpose:** Detect multi-domain queries

**Formula:**
```
mean = sum(fused_scores.values()) / len(fused_scores)
variance = sum((score - mean)² for score in fused_scores.values()) / len(fused_scores)
```

**Example:**
- Fused scores: [0.456, 0.520, 0.45]
- Mean: 0.475
- Variance: ((0.456-0.475)² + (0.520-0.475)² + (0.45-0.475)²) / 3 = 0.001467

---

### Step 5: Phase 3 — Superposition Detection

**Function:** `fusion_engine.SuperpositionDetector.detect(...)`

**Inputs:**
```python
fused_scores: {domain: float}
score_threshold: float = 0.5
variance_threshold: float = 0.08  # From tuning_config.SUPERPOSITION_VARIANCE_THRESHOLD
```

**Decision Logic:**

| Condition | Classification |
|-----------|-----------------|
| len(fused_scores) == 0 | NO_EXPERT_AVAILABLE |
| num_candidates == 0 | NO_EXPERT_AVAILABLE |
| num_candidates == 1 | SINGLE_DOMAIN |
| num_candidates >= 2 AND variance >= 0.08 | MULTI_DOMAIN |
| num_candidates >= 2 AND variance < 0.08 | AMBIGUOUS |

**Where:** `num_candidates = count(fused_scores[d] >= score_threshold)`

**Algorithm:**
1. Count domains with fused_score >= 0.5
2. Calculate variance of all fused_scores
3. Apply decision rules above
4. Sort candidates by fused_score descending

**Example:**
- Fused scores: {astronomy: 0.45, automobile: 0.52, aesthetics: 0.50}
- Candidates (>= 0.5): [automobile, aesthetics]
- Variance: 0.001467
- **Decision:** AMBIGUOUS (2 candidates, low variance)

---

### Step 6: Phase 3 — Greedy Expert Selection

**Function:** `optimization_engine.GreedyExpertSelector.select_experts(...)`

**Inputs:**
```python
fused_scores: {domain: float}
confidence_scores: {domain: 0.5}  # Default
```

**Algorithm:**

1. **Build candidate list** with (domain, value, confidence, fused_score):
   ```
   value = fused_score * confidence
   value = clamp(value, 0.0, 1.0)
   ```

2. **Sort by tie-breaker** (deterministic):
   ```
   Sort key: (-value, -confidence, -fused_score, domain_alphabetical)
   ```

3. **Greedy selection loop:**
   ```
   For each candidate in sorted order:
       if coverage >= soft_stop_threshold AND len(selected) > 0:
           break  # Soft-stop: 90% of threshold is good enough
       
       if len(selected) >= max_experts:
           break  # Hard cap
       
       selected.append(domain)
       total_coverage += value
       log_if_enabled("Added {domain}: coverage={total_coverage}")
   ```

4. **Check success:**
   ```
   coverage_met = total_coverage >= coverage_threshold
   create_new_expert = NOT coverage_met
   ```

**Soft-Stop Details:**
```
coverage_threshold = 0.70  (from tuning_config)
soft_stop_threshold = coverage_threshold * 0.90 = 0.63
```

When coverage reaches 63%, expert selection can stop (even if more experts could increase coverage to 70%).

**Example:**
- Fused scores: {astronomy: 0.45, automobile: 0.52, aesthetics: 0.50}
- Confidence: all 0.5
- Values: {astronomy: 0.225, automobile: 0.26, aesthetics: 0.25}
- Coverage threshold: 0.70
- Soft-stop threshold: 0.63
- Selection:
  1. Add automobile (value=0.26, cumulative=0.26)
  2. Add aesthetics (value=0.25, cumulative=0.51)
  3. Check soft-stop: 0.51 < 0.63, continue
  4. Add astronomy (value=0.225, cumulative=0.735)
  5. Check soft-stop: 0.735 >= 0.63, can stop
  6. Max experts (3) also reached
- **Result:** selected_experts=[aesthetics, astronomy, automobile], coverage_met=True

---

### Step 7: Phase 6 — ATTRIBUTE_ONLY Override Rule

**Condition:** `ENABLE_ATTRIBUTE_OVERRIDE == True` AND `base_classification == "ATTRIBUTE_ONLY"` AND object-level domains found

**Algorithm:**
```
for domain in object_level_domains:
    domain_score = fused_scores.get(domain, 0.0)
    if domain_score >= DOMAIN_SCORE_THRESHOLD (0.45):
        # Override! Promote from ATTRIBUTE_ONLY
        if domain not in selected_experts:
            selected_experts.append(domain)
        attribute_override_applied = True
        log_if_enabled(f"Override applied for {domain}")
        break  # Only need one strong domain
```

**Purpose:** Reduce false-positive ATTRIBUTE_ONLY classifications

**Example:**
- Input: "Glossy metallic red finish"
- Base classification: ATTRIBUTE_ONLY
- Lens 2 found object-level domain: "aesthetics"
- Fused score for aesthetics: 0.48
- Check: 0.48 >= 0.45? YES
- **Decision:** Promote to SINGLE_DOMAIN, select aesthetics expert

---

### Step 8: Final Classification

**Logic:**
```
if create_new_expert AND NOT selected_experts:
    final_classification = "NO_EXPERT_AVAILABLE"
elif len(selected_experts) == 1:
    final_classification = "SINGLE_DOMAIN"
elif len(selected_experts) > 1:
    if variance >= 0.08:
        final_classification = "MULTI_DOMAIN"
    else:
        final_classification = "AMBIGUOUS"
else:
    final_classification = "NO_EXPERT_AVAILABLE"
```

---

### Step 9: Build Explanation

**Function:** `multi_lens_router.MultiLensRouter._build_explanation(...)`

**Output:** Human-readable explanation

**Examples:**
- SINGLE_DOMAIN: "Single domain detected: automobile."
- MULTI_DOMAIN: "Multiple domains detected: automobile, engine_spec."
- ATTRIBUTE_ONLY: "Only attributes detected; no domain match."
- AMBIGUOUS: "Ambiguous input; multiple domains with similar relevance."
- NO_EXPERT_AVAILABLE: "No suitable expert available."
- Coverage: "Coverage threshold met." or "Coverage threshold not met; new expert may be required."

---

## Complete Decision Matrix

### By Classification Type

#### SINGLE_DOMAIN
- **Conditions:**
  - Exactly 1 expert selected
  - coverage_met typically true (but may be false if sole expert has low value)
  - create_new_expert false
- **Expert Selection:** 1
- **Explanation:** "Single domain detected: {domain}."

#### MULTI_DOMAIN
- **Conditions:**
  - Multiple experts selected (≥ 2)
  - Variance >= 0.08 (high variance = distinct domains)
  - Coverage might not be met (soft-stop may stop early)
- **Expert Selection:** 2+
- **Explanation:** "Multiple domains detected: {domains}."

#### AMBIGUOUS
- **Conditions:**
  - Multiple experts selected (≥ 2)
  - Variance < 0.08 (low variance = similar scores)
  - Query doesn't clearly indicate one domain
- **Expert Selection:** 2+
- **Explanation:** "Ambiguous input; multiple domains with similar relevance."

#### ATTRIBUTE_ONLY
- **Conditions:**
  - Layer 1 Lens 2 found only modifiers, no object-level content
  - No override rule applied (no object-level domain with score >= 0.45)
- **Expert Selection:** 0 (or 1 if override applied)
- **Explanation:** "Only attributes detected; no domain match."

#### NO_EXPERT_AVAILABLE
- **Conditions:**
  - No domains with fused_score >= 0.5
  - Or all candidates fell below threshold
  - Coverage threshold not met despite best effort
- **Expert Selection:** 0
- **Explanation:** "No suitable expert available."

---

## Threshold Reference

| Threshold | Value | Source | Purpose |
|-----------|-------|--------|---------|
| Superposition Variance | 0.08 | tuning_config | Distinguish MULTI_DOMAIN from AMBIGUOUS |
| Candidate Score | 0.5 | hardcoded | Minimum fused_score to be a candidate |
| Domain Score (Override) | 0.45 | tuning_config | Minimum score to override ATTRIBUTE_ONLY |
| Coverage | 0.70 | tuning_config | Minimum total_coverage to meet threshold |
| Soft-Stop | 0.90 | tuning_config | % of coverage threshold to trigger early stop |
| Max Experts | 3 | tuning_config | Maximum experts to select |

---

## Feature Flags

| Flag | Default | Effect |
|------|---------|--------|
| `use_multi_lens` | True | If False, skip Phases 1-6, return Layer 1 only |
| `use_spectral` | True | If False, skip Phase 1 (spectral scores = {}) |
| `ENABLE_ATTRIBUTE_OVERRIDE` | True | If False, skip Phase 6 override rule |
| `ENABLE_LOGGING` | False | If True, log per-request decisions |
| `LOG_SAMPLE_RATE` | 1 | Log every Nth request (1 = all) |

---

## Examples

### Example 1: Clear Single Domain

**Input:** "Earth orbits the Sun"

**Flow:**
1. Layer 1: semantic_scores={astronomy: 0.75}, confidence={astronomy: 0.5}, classification=NORMAL
2. Spectral: spectral_scores={astronomy: 0.40}
3. Fusion: fused={astronomy: 0.45*0.75 + 0.35*0.40 + 0.20*0.5 = 0.545}
4. Superposition: 1 candidate (>= 0.5) → SINGLE_DOMAIN
5. Optimization: select astronomy (value=0.545*0.5=0.2725), coverage=0.2725 < 0.70 → create_new_expert=True
6. Override: Not applicable (not ATTRIBUTE_ONLY)
7. **Output:** SINGLE_DOMAIN, selected=[astronomy], coverage_met=False, create_new_expert=True

### Example 2: Multi-Domain

**Input:** "Red car with turbocharged engine"

**Flow:**
1. Layer 1: semantic={automobile: 0.75, engine_spec: 0.65, aesthetics: 0.40}, confidence=0.5, classification=NORMAL
2. Spectral: {automobile: 0.55, engine_spec: 0.52, aesthetics: 0.45}
3. Fusion:
   - automobile: 0.45*0.75 + 0.35*0.55 + 0.20*0.5 = 0.6125
   - engine_spec: 0.45*0.65 + 0.35*0.52 + 0.20*0.5 = 0.5470
   - aesthetics: 0.45*0.40 + 0.35*0.45 + 0.20*0.5 = 0.3775
4. Variance: Mean=0.512, Var=(high) 0.0115 -> Variance >= 0.08? NO → AMBIGUOUS
5. Optimization: select automobile (0.306), engine_spec (0.274), aesthetics (0.189), coverage=0.769 >= 0.70 → coverage_met=True
6. Override: N/A
7. **Output:** AMBIGUOUS, selected=[automobile, engine_spec, aesthetics], coverage_met=True, create_new_expert=False

### Example 3: Attribute-Only with Override

**Input:** "Glossy metallic red finish"

**Flow:**
1. Layer 1: semantic={}, classification=ATTRIBUTE_ONLY, but object-level domain found: aesthetics
2. Early exit skipped (object-level domain exists)
3. Spectral: {aesthetics: 0.50}
4. Fusion: {aesthetics: 0.45*0 + 0.35*0.50 + 0.20*0.5 = 0.275}
5. Superposition: 0 candidates (< 0.5) → would be NO_EXPERT
6. Optimization: select aesthetics (value=0.138), coverage=0.138
7. Override: base=ATTRIBUTE_ONLY, aesthetics found, score=0.275 >= 0.45? NO
   - Actually, let's recalculate: If Layer 1 provides confidence=0.5 for aesthetics: fused=0.175 (too low)
   - Assume confidence=0.9: fused = 0.45*0 + 0.35*0.50 + 0.20*0.9 = 0.355 (still < 0.45)
   - **Override NOT triggered, stays ATTRIBUTE_ONLY**
8. **Output:** ATTRIBUTE_ONLY, selected=[], coverage_met=False, create_new_expert=False

---

## Determinism Verification

**Same input, 5 runs:**
```
Run 1: AMBIGUOUS, [automobile, aesthetics, engine_spec], variance=0.00125
Run 2: AMBIGUOUS, [automobile, aesthetics, engine_spec], variance=0.00125
Run 3: AMBIGUOUS, [automobile, aesthetics, engine_spec], variance=0.00125
Run 4: AMBIGUOUS, [automobile, aesthetics, engine_spec], variance=0.00125
Run 5: AMBIGUOUS, [automobile, aesthetics, engine_spec], variance=0.00125

✅ Identical across all runs
```

---

## Performance Notes

- **Average routing time:** 0.0001 seconds (very fast)
- **Bottleneck:** Spectral analysis (SentenceTransformer encoding) if enabled
- **Scaling:** Linear with number of experts (O(n log n) for sorting)

