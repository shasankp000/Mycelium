# Architecture Overview

**Project Mycelium — Multi-Lens Routing System**  
**Version:** 1.0 (Phases 1–6 Complete)  
**Date:** January 17, 2026

---

## Table of Contents

1. [System Purpose](#system-purpose)
2. [High-Level Architecture](#high-level-architecture)
3. [The Three Lenses](#the-three-lenses)
4. [Why Multi-Lens?](#why-multi-lens)
5. [Data Flow](#data-flow)
6. [Determinism Guarantees](#determinism-guarantees)
7. [Component Responsibilities](#component-responsibilities)

---

## System Purpose

**Mycelium** is an intelligent text-to-expert routing system that determines which domain expert(s) should handle a given query.

### Core Problem
Given a text input (e.g., "Red car with turbocharged engine"), the system must:
- Identify relevant domains (automobile, aesthetics, engine_spec)
- Select the most appropriate expert(s)
- Decide if existing experts are sufficient or if new expertise is needed

### Solution
Use **three complementary lenses** to analyze text from different angles, each with unique strengths:

```
Input Query
    ↓
Lens 1: Semantic (core/modifier matching) → semantic scores
Lens 2: Ontology (domain hierarchy) → confidence scores + explanation
Lens 3: Spectral (frequency analysis) → spectral scores
    ↓
Fusion Engine (weighted combination)
    ↓
Optimization (greedy expert selection)
    ↓
Routing Decision + Explanation
```

---

## High-Level Architecture

### System Layers

```
┌─────────────────────────────────────────────────────────┐
│                    APPLICATION                          │
│         (Uses MultiLensRouter for routing)              │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│           MULTI_LENS_ROUTER (Orchestration)            │
│  Coordinates all phases, manages feature flags         │
│  Input: text | Output: routing decision                │
└──────────────┬──────────────┬──────────────┬────────────┘
               │              │              │
        ┌──────▼──┐    ┌──────▼──┐   ┌──────▼──┐
        │ PHASE 1 │    │ PHASE 2 │   │ PHASE 3 │
        │Spectral │    │ Fusion  │   │ Optimiz │
        └─────────┘    └─────────┘   └─────────┘
               │              │              │
        ┌──────▼──────────────▼──────────────▼──┐
        │    LAYER 1 (layer_1_prototype)       │
        │  Lens 1 (Semantic) + Lens 2 (Ontology)
        │  Returns: classification, domains,   │
        │  explanation, confidence              │
        └──────────────────────────────────────┘
```

### Component Hierarchy

```
tuning_config.py (Configuration)
    ↓
    ├─→ fusion_engine.py (Phase 2)
    │   ├─→ ScoreNormalizer
    │   └─→ SuperpositionDetector
    │
    ├─→ optimization_engine.py (Phase 3)
    │   ├─→ CoverageCalculator
    │   └─→ GreedyExpertSelector
    │
    ├─→ spectral_analyzer.py (Phase 1)
    │   ├─→ SpectralSignatureGenerator (pre-training)
    │   └─→ RuntimeSpectralAnalyzer (inference)
    │
    └─→ multi_lens_router.py (Phase 4 + Phase 6)
        └─→ MultiLensRouter (orchestration)
                ↓
        layer_1_prototype.py (Foundation)
        └─→ multi_lens_route()
```

---

## The Three Lenses

### Lens 1: Semantic Scoring (Layer 1, built-in)

**Purpose:** Match input text against domain keywords using exact and fuzzy matching.

**How it Works:**
1. Extract core keywords from input (e.g., "car" from "red car")
2. Extract attributes (e.g., "red")
3. Look up each keyword in domain definitions
4. Return score [0,1] per domain based on core/attribute matches

**Strengths:**
- Fast (no ML, no I/O)
- Deterministic
- Explainable (shows matched keywords)
- Works without pre-training

**Weaknesses:**
- Limited to exact matches (requires predefined domains)
- Doesn't capture semantic relationships
- No understanding of context

**Output:** `semantic_scores: {domain: float}`

---

### Lens 2: Ontology Scoring (Layer 1, built-in)

**Purpose:** Understand domain hierarchies and object-level vs. attribute-only content.

**How it Works:**
1. Map input keywords to ontology hierarchy (e.g., car → vehicle → transportation)
2. Identify if match is at object-level (core concept) or modifier-level (attributes)
3. Return confidence score based on match depth
4. Classify as ATTRIBUTE_ONLY if only modifiers matched

**Strengths:**
- Captures domain hierarchies
- Distinguishes objects from attributes
- Prevents false positives on "pure attribute" inputs
- Explainable (shows matched concepts)

**Weaknesses:**
- Requires hand-built ontology
- Brittle if ontology is incomplete
- Confidence scores are heuristic-based

**Output:** `confidence_scores: {domain: float}`, `classification: ATTRIBUTE_ONLY | NORMAL`

---

### Lens 3: Spectral Analysis (Phase 1, optional)

**Purpose:** Use frequency-domain analysis to distinguish domains by linguistic patterns.

**How it Works:**
1. Pre-training phase:
   - Encode domain texts using SentenceTransformer (all-MiniLM-L6-v2)
   - Compute Fast Fourier Transform (FFT) per embedding dimension
   - Calculate Power Spectral Density (PSD)
   - Save signature for each domain
2. Runtime phase:
   - Encode query text
   - Cross-correlate against stored signatures
   - Return correlation scores [0,1] per domain

**Strengths:**
- Captures linguistic patterns
- Distinguishes multi-domain queries
- Deterministic (same input → same output)
- No training required (pre-computed signatures)

**Weaknesses:**
- Requires pre-trained signatures (separate step)
- Low discrimination with small corpora
- Requires SentenceTransformer dependency (optional)

**Output:** `spectral_scores: {domain: float}`

---

## Why Multi-Lens?

Each lens has blind spots. Using all three compensates:

| Scenario | Lens 1 | Lens 2 | Lens 3 | Outcome |
|----------|--------|--------|--------|---------|
| "Red car" (clear domain) | ✅ Matches "car" | ✅ Object-level | ✅ Autocorrelates | **Confident SINGLE_DOMAIN** |
| "Glossy metallic red" (pure attributes) | ❌ No keywords | ✅ Attributes only | ✅ Could match | **Lens 2 catches it → ATTRIBUTE_ONLY** |
| "Medical ultrasound uses physics" (multi-domain) | ⚠️ Partial matches | ⚠️ Low confidence | ✅ High variance | **Lens 3 helps decide MULTI_DOMAIN** |
| "Quantum culinary philosophy" (unknown domains) | ❌ No matches | ❌ No concepts | ✅ Noise detected | **All agree → NO_EXPERT** |

### Fusion Strategy
1. **Weight**: Semantic 0.45, Spectral 0.35, Confidence 0.20
   - Semantic most reliable (established matching)
   - Spectral helps with disambiguation
   - Confidence provides caution signal
2. **Variance Detection**: Use spectral variance to detect multi-domain
3. **Override Rules**: If Lens 2 found object + fusion score high, promote from ATTRIBUTE_ONLY

---

## Data Flow

### Complete Request Flow

```
┌──────────────────────────────────────────────────────────────┐
│ INPUT: Text Query (e.g., "Red sports car with engine")      │
└────────────────────────────┬─────────────────────────────────┘
                             │
                      ┌──────▼──────┐
                      │ Layer 1      │
                      │multi_lens_   │
                      │route(text)   │
                      └──────┬───────┘
                             │
             ┌───────────────┼───────────────┐
             │               │               │
        ┌────▼────┐    ┌─────▼────┐   ┌─────▼─────┐
        │semantic_ │    │confidence│   │lens3_     │
        │scores    │    │_scores   │   │signature  │
        └────┬─────┘    └─────┬────┘   └─────┬─────┘
             │                │              │
      (default 0.5) (Lens 2)  │        (if spectral)
             │                │              │
             └────────────┬───┴──────────────┘
                          │
                   ┌──────▼──────┐
                   │ Spectral     │
                   │ Analyzer     │
                   │(if enabled)  │
                   └──────┬───────┘
                          │
                   ┌──────▼──────┐
                   │ Fusion       │
                   │ Engine       │
                   └──────┬───────┘
                          │
                   ┌──────▼──────────┐
                   │ fused_scores    │
                   │ {domain: float} │
                   └──────┬──────────┘
                          │
                   ┌──────▼────────────┐
                   │ Optimization      │
                   │ Engine (Greedy)   │
                   └──────┬────────────┘
                          │
             ┌────────────┴────────────┐
             │                         │
    ┌────────▼────────┐       ┌────────▼────────┐
    │selected_experts │       │coverage_met?    │
    │[domain1, ...]   │       │bool             │
    └────────┬────────┘       └────────┬────────┘
             │                         │
             └────────────┬────────────┘
                          │
        ┌─────────────────▼──────────────────┐
        │ Classify & Build Output            │
        │ - ATTRIBUTE_ONLY                   │
        │ - SINGLE_DOMAIN                    │
        │ - MULTI_DOMAIN                     │
        │ - AMBIGUOUS                        │
        │ - NO_EXPERT_AVAILABLE              │
        └─────────────────┬──────────────────┘
                          │
        ┌─────────────────▼──────────────────┐
        │ OUTPUT: Routing Decision           │
        │ {                                  │
        │   primary_domain: str,             │
        │   selected_experts: [str],         │
        │   classification: str,             │
        │   coverage_met: bool,              │
        │   create_new_expert: bool,         │
        │   lens_scores: {...},              │
        │   fused_scores: {...},             │
        │   variance: float,                 │
        │   explanation: str                 │
        │ }                                  │
        └────────────────────────────────────┘
```

### Key Data Structures

```python
# Layer 1 Output (from layer_1_prototype.multi_lens_route)
{
    "primary_domain": str or None,
    "secondary_domains": [str],
    "explanation": str,
    "lens1_candidates": [[domain, score], ...],
    "lens2_explanations": [{
        "concept": str,
        "score": float,
        "matched_core": [str],      # Object-level keywords
        "matched_attr": [str],      # Attribute-level keywords
        "parent": str or None,
        "level": "object" or "modifier"
    }, ...],
    "lens3_signature": {
        "levels": {
            "core": {domain: [keywords]},
            "modifiers": {domain: [keywords]}
        },
        "active_domains": {"core": int, "modifiers": int}
    },
    "classification": "NORMAL" or "ATTRIBUTE_ONLY"
}

# Multi-Lens Router Output (final routing decision)
{
    "primary_domain": str or None,
    "selected_experts": [str],          # Expert(s) to route to
    "candidate_domains": [str],         # All viable domains
    "classification": str,              # SINGLE_DOMAIN, MULTI_DOMAIN, AMBIGUOUS, ATTRIBUTE_ONLY, NO_EXPERT_AVAILABLE
    "coverage_met": bool,               # Did expert selection meet threshold?
    "create_new_expert": bool,          # Should new expert be created?
    "lens_scores": {
        "semantic": {domain: float},
        "spectral": {domain: float},
        "confidence": {domain: float}
    },
    "fused_scores": {domain: float},    # Weighted combination
    "variance": float,                  # Variance of fused scores
    "explanation": str                  # Human-readable decision summary
}
```

---

## Determinism Guarantees

The system is **fully deterministic** with no randomness anywhere:

### Deterministic Components

| Component | Mechanism | Evidence |
|-----------|-----------|----------|
| Semantic Scoring | Exact keyword matching | Fixed input → fixed output |
| Ontology Scoring | Deterministic hierarchy traversal | No branching on stochastic decisions |
| Spectral Analysis | Fixed FFT + pre-computed signatures | Same embedding → same FFT → same score |
| Fusion | Deterministic weighted sum | Math is deterministic |
| Optimization | Deterministic sort with explicit tie-breakers | Sort key: (value desc, confidence desc, score desc, alphabetical) |
| Variance | Population variance formula | Same inputs → same variance |

### Tie-Breaking (Explicit)
When two experts have identical scores, order is:
1. By value (fused × confidence) - descending
2. By confidence - descending
3. By fused score - descending
4. By domain name - alphabetical (ultimate tie-breaker)

### Verification
**Test:** Run same input 5 times → identical output
```python
router = MultiLensRouter()
results = [router.route(text) for _ in range(5)]
assert all(r == results[0] for r in results)  # ✅ Always passes
```

---

## Component Responsibilities

### layer_1_prototype.py (Foundation)
**Responsibility:** Provide baseline semantic and ontology-based routing
- **Inputs:** text
- **Outputs:** semantic_scores, confidence_scores, classification, explanation
- **Locked:** Cannot modify (established baseline)
- **Dependencies:** None (self-contained)

### Phase 1: spectral_analyzer.py (Spectral Lens)
**Responsibility:** Generate and apply spectral signatures
- **SpectralSignatureGenerator:** Pre-training (one-time, offline)
  - Encodes domain texts
  - Computes FFT + PSD per dimension
  - Saves .npy signatures
- **RuntimeSpectralAnalyzer:** Inference (per-request)
  - Loads signatures
  - Cross-correlates input
  - Returns spectral_scores
- **Locked:** Cannot modify (established algorithm)
- **Dependencies:** Optional (SentenceTransformer, scipy, numpy)

### Phase 2: fusion_engine.py (Fusion)
**Responsibility:** Combine three lens scores into unified scores
- **ScoreNormalizer:** Clip and normalize [0,1]
- **FusionEngine:** Weighted averaging (configurable)
  - Takes semantic, spectral, confidence scores
  - Returns fused_scores per domain
  - Logs per-domain contributions
- **SuperpositionDetector:** Classify based on variance
  - Detects SINGLE_DOMAIN, MULTI_DOMAIN, AMBIGUOUS, NO_EXPERT
  - Uses threshold from config
- **Tunable:** Weights, variance threshold (tuning_config.py)
- **Dependencies:** None (standard library)

### Phase 3: optimization_engine.py (Optimization)
**Responsibility:** Greedily select minimal expert set
- **CoverageCalculator:** value = fused_score × confidence
- **GreedyExpertSelector:** 
  - Sort by value (descending)
  - Accumulate coverage until threshold or max_experts
  - Supports soft-stop at 90% of threshold
  - Deterministic sorting
- **Tunable:** Coverage threshold, max_experts, soft-stop % (tuning_config.py)
- **Dependencies:** None (standard library)

### Phase 4: multi_lens_router.py (Orchestration)
**Responsibility:** Coordinate all phases into single routing API
- **MultiLensRouter:** Main class
  - Feature flags (use_multi_lens, use_spectral)
  - Initializes components
  - Orchestrates pipeline
  - Implements ATTRIBUTE_ONLY override rule (Phase 6)
  - Tracks metrics (Phase 6)
  - Logs decisions (Phase 6)
- **Tunable:** Config values, logging settings
- **Dependencies:** All phases above (graceful fallbacks if missing)

### tuning_config.py (Configuration - NEW in Phase 6)
**Responsibility:** Centralize all hyperparameters
- **FUSION_WEIGHTS:** semantic, spectral, confidence (sum=1.0)
- **Thresholds:** variance, domain_score, coverage, soft_stop, max_experts
- **Flags:** enable_attribute_override, enable_logging, log_sample_rate
- **Validation:** Runs on import, ensures consistency
- **No Dependencies:** Pure configuration, imports nothing from system

### expert_filter.py (External - NOT Modified)
**Responsibility:** (Existing system, untouched by Phases 1-6)
**Locked:** Cannot modify in Phase 7

---

## Safety & Constraints

### What Is Locked (Cannot Modify)
- ❌ layer_1_prototype.py (baseline)
- ❌ expert_filter.py (external system)
- ❌ Algorithms (spectral FFT, fusion formula, greedy optimization)
- ❌ Determinism guarantees (no randomness anywhere)

### What Can Be Adjusted
- ✅ Fusion weights (must sum to 1.0)
- ✅ Thresholds (variance, coverage, score)
- ✅ Feature flags (use_multi_lens, use_spectral)
- ✅ Logging level and sample rate
- ✅ Override rules (isolated, disableable)

### What Cannot Be Added
- ❌ External ML models (except SentenceTransformer for spectral)
- ❌ Online learning / retraining
- ❌ Randomness or stochasticity
- ❌ New major components (phases are complete)

---

## Deployment Model

### Standalone Usage
```python
from multi_lens_router import MultiLensRouter

router = MultiLensRouter(
    use_multi_lens=True,      # Enable Phase 2-4
    use_spectral=True,        # Enable Phase 1
    coverage_threshold=0.7    # From tuning_config.py
)

result = router.route("Red car with engine")

print(result["classification"])       # "SINGLE_DOMAIN" or "MULTI_DOMAIN"
print(result["selected_experts"])     # ["automobile"] or ["automobile", "engine_spec"]
print(result["primary_domain"])       # "automobile"
```

### Backward Compatibility
```python
# Old code still works - returns only Layer 1
from layer_1_prototype import multi_lens_route

result = multi_lens_route("Red car")  # Still works, no Phase 1-6
```

### Metrics & Monitoring
```python
# Access aggregate metrics
metrics = MultiLensRouter.get_metrics()
print(metrics["attribute_only_pct"])       # % ATTRIBUTE_ONLY
print(metrics["create_new_expert_pct"])    # % unmet coverage
```

---

## Summary

Mycelium is a **three-lens text routing system** that:

1. ✅ Uses semantic matching (Lens 1) for speed and explainability
2. ✅ Uses ontology hierarchies (Lens 2) for accuracy and object detection
3. ✅ Uses spectral analysis (Lens 3) for multi-domain discrimination
4. ✅ Fuses scores intelligently with configurable weights
5. ✅ Greedily selects minimal expert set with soft-stop optimization
6. ✅ Maintains full determinism (no randomness)
7. ✅ Provides clear explanations and metrics
8. ✅ Supports backward compatibility

**Design Philosophy:** Simple, explainable, deterministic, extensible.

