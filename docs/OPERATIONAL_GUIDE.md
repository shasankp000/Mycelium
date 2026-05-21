# Operational Guide

**Project Mycelium — How to Use, Extend, and Operate**  
**Version:** 1.0  
**Date:** January 17, 2026

---

## Overview

This guide covers practical operation of the system: how to use it, test it, add domains, debug, and handle common situations.

---

## Quick Start

### Installation

1. **Prerequisites:**
   ```bash
   Python 3.8+
   pip install numpy scipy sentence-transformers
   ```

2. **Setup:**
   ```bash
   cd /Users/abhinaygiri/Documents/Projects/Mycelium
   python3 -c "from unified_expert_system import *; print('OK')"
   ```

### Basic Usage

```python
from multi_lens_router import MultiLensRouter

# Initialize router
router = MultiLensRouter(use_multi_lens=True, use_spectral=True)

# Route a query
result = router.route("Red car with turbocharged engine")

print(result["classification"])      # "MULTI_DOMAIN"
print(result["selected_experts"])    # ["automobile", "engine_spec", ...]
print(result["primary_domain"])      # "automobile"
print(result["reasoning"])           # Explanation
```

### Output Format

```python
{
    "classification": str,           # SINGLE_DOMAIN, MULTI_DOMAIN, etc.
    "selected_experts": [str, ...],  # Expert domains to consult
    "primary_domain": str,           # Most relevant domain (or None)
    "reasoning": str,                # Human-readable explanation
    "confidence": float,             # Confidence in classification (0-1)
    "metadata": {
        "fusion_scores": {domain: float},
        "coverage": float,
        "variance": float,
        "create_new_expert": bool
    }
}
```

---

## Running Tests

### Full Test Suite

```bash
cd /Users/abhinaygiri/Documents/Projects/Mycelium
python3 -m pytest test_multilens_system.py -v
# OR
python3 test_multilens_system.py
```

### Expected Output

```
test_color_finish_attribute ... ok
test_performance_attribute ... ok
test_earth_science_single ... ok
...
Ran 17 tests in 0.001s
OK
```

### Test Groups

| Group | Tests | Purpose |
|-------|-------|---------|
| Single-Domain | 2 | Basic routing to one expert |
| Multi-Domain | 3 | Routing to multiple experts |
| Attribute-Only | 2 | Handling pure-attribute queries |
| No-Expert | 1 | Graceful degradation |
| Backward Compat | 2 | Layer 1 fallback mode |
| Degradation | 3 | Robustness without spectral |
| Determinism | 2 | Identical output (5 runs) |
| Performance | 2 | Routing speed |

### Running Specific Tests

```bash
# Single test
python3 -m pytest test_multilens_system.py::TestMultiLensSystem::test_earth_science_single -v

# Test group
python3 -m pytest test_multilens_system.py::TestMultiLensSystem::test_color_finish_attribute -v
python3 -m pytest test_multilens_system.py::TestMultiLensSystem::test_performance_attribute -v
```

### Debug a Test

```python
# Add to test_multilens_system.py temporarily
result = router.route("Red car with turbocharged engine")
print(f"Result: {result}")
print(f"Fusion scores: {result['metadata']['fusion_scores']}")
print(f"Variance: {result['metadata']['variance']}")
print(f"Coverage: {result['metadata']['coverage']}")
```

---

## Adding a New Domain

### Step 1: Define Domain in Layer 1

**File:** `layer_1_prototype.py`

In the `DOMAIN_DEFINITIONS` dict:

```python
DOMAIN_DEFINITIONS = {
    ...
    "new_domain": {
        "keywords": ["keyword1", "keyword2", ...],
        "modifiers": ["modifier1", ...],  # Optional attributes
        "related_concepts": {...},         # Optional ontology
        "confidence": 0.5                  # Default
    }
}
```

**Example: Adding "electronics"**

```python
"electronics": {
    "keywords": ["circuit", "transistor", "capacitor", "resistor", "semiconductor", "electronics"],
    "modifiers": ["compact", "low-power", "high-frequency"],
    "related_concepts": {
        "embedded_systems": 0.8,
        "computer_science": 0.7
    },
    "confidence": 0.5
}
```

### Step 2: Create Expert Filter

**File:** `expert_filter.py`

Add expert definition:

```python
class ElectronicsExpert:
    """Expert for electronics domain."""
    
    def __init__(self):
        self.name = "ElectronicsExpert"
        self.domain = "electronics"
    
    def filter_results(self, results):
        """Filter results for electronics expertise."""
        return [r for r in results if self._is_relevant(r)]
    
    def _is_relevant(self, result):
        """Check if result is electronics-related."""
        keywords = {"circuit", "transistor", "component", "signal"}
        return any(kw in result.lower() for kw in keywords)
```

### Step 3: Generate Spectral Signature (Optional)

**If using spectral analysis:**

```python
from spectral_analyzer import RuntimeSpectralAnalyzer

analyzer = RuntimeSpectralAnalyzer()

# Example domain texts for signature generation
electronics_texts = [
    "Semiconductor circuit design",
    "Electronic components and transistors",
    "Integrated circuit fabrication",
    "Electronics and electrical systems"
]

# Analyzer stores signatures; you can export:
# export_signatures() → tuning_config.py or file
```

### Step 4: Update Tests

**File:** `test_multilens_system.py`

Add test for new domain:

```python
def test_electronics_single(self):
    """Test routing to electronics expert."""
    result = self.router.route("Circuit design with low-power transistors")
    
    self.assertEqual(result["classification"], "SINGLE_DOMAIN")
    self.assertIn("electronics", result["selected_experts"])
    self.assertEqual(result["primary_domain"], "electronics")
```

### Step 5: Run Tests

```bash
python3 test_multilens_system.py TestMultiLensSystem.test_electronics_single
```

### Common Domain Structure Template

```python
"{domain_name}": {
    "keywords": [
        "core_keyword_1",
        "core_keyword_2",
        "unique_term_3"
    ],
    "modifiers": [
        "quality_attribute_1",
        "quality_attribute_2"
    ],
    "related_concepts": {
        "related_domain_1": 0.8,
        "related_domain_2": 0.6
    },
    "confidence": 0.5
}
```

---

## Debugging Guide

### Problem: Query Routes to Wrong Domain

**Symptoms:**
```
router.route("query") → wrong_domain
Expected: correct_domain
```

**Debug Steps:**

1. **Check Layer 1 definitions:**
   ```python
   from layer_1_prototype import DOMAIN_DEFINITIONS
   
   # Verify keywords match
   if "query_keyword" in DOMAIN_DEFINITIONS["correct_domain"]["keywords"]:
       print("✓ Keyword in definitions")
   else:
       print("✗ Keyword missing!")
   ```

2. **Check fusion scores:**
   ```python
   result = router.route("query")
   fusion_scores = result["metadata"]["fusion_scores"]
   print(f"Scores: {fusion_scores}")
   # wrong_domain score should be lower
   ```

3. **Verify spectral analysis (if enabled):**
   ```python
   from spectral_analyzer import RuntimeSpectralAnalyzer
   
   analyzer = RuntimeSpectralAnalyzer()
   spectral = analyzer.analyze_text("query")
   print(f"Spectral scores: {spectral}")
   # Check if spectral is causing the issue
   ```

4. **Check fusion weights:**
   ```python
   from tuning_config import FUSION_WEIGHTS
   print(FUSION_WEIGHTS)
   # Adjust if needed
   ```

### Problem: Query Classified as ATTRIBUTE_ONLY

**Symptoms:**
```
router.route("glossy red finish") → ATTRIBUTE_ONLY
Expected: should route to aesthetics expert
```

**Debug Steps:**

1. **Check Layer 1 output:**
   ```python
   from layer_1_prototype import multi_lens_route
   
   base = multi_lens_route("glossy red finish")
   print(f"Base classification: {base['classification']}")
   print(f"Lens 2 results: {base['lens2_explanations']}")
   ```

2. **Check object-level domains:**
   ```python
   result = router.route("glossy red finish")
   # Look for object_level_domains in internal state
   ```

3. **Check override threshold:**
   ```python
   from tuning_config import DOMAIN_SCORE_THRESHOLD
   fusion_scores = result["metadata"]["fusion_scores"]
   
   for domain, score in fusion_scores.items():
       if score >= DOMAIN_SCORE_THRESHOLD:
           print(f"✓ {domain} score {score} >= threshold {DOMAIN_SCORE_THRESHOLD}")
       else:
           print(f"✗ {domain} score {score} < threshold {DOMAIN_SCORE_THRESHOLD}")
   ```

### Problem: Inconsistent Results

**Symptoms:**
```
Run 1: MULTI_DOMAIN [a, b, c]
Run 2: MULTI_DOMAIN [a, b]
Expected: identical
```

**Debug Steps:**

1. **Check randomness:**
   ```python
   import numpy as np
   print(f"NumPy seed: {np.random.get_state()[0]}")
   
   # Verify no random() calls in routing
   grep -r "random\|randint\|shuffle" *.py  # Should return nothing
   ```

2. **Check sorting determinism:**
   ```python
   # Tie-breaker in optimization_engine.py:
   # Sort by: (-value, -confidence, -fused_score, domain_alphabetical)
   ```

3. **Run determinism test:**
   ```bash
   python3 -m pytest test_multilens_system.py -k determinism -v
   ```

### Problem: Coverage Not Met

**Symptoms:**
```
router.route("query") → create_new_expert=True
Expected: create_new_expert=False
```

**Debug Steps:**

1. **Check coverage calculation:**
   ```python
   result = router.route("query")
   print(f"Coverage: {result['metadata']['coverage']}")
   print(f"Threshold: 0.70")
   
   if result['metadata']['coverage'] < 0.70:
       print("✗ Coverage below threshold")
   ```

2. **Check selected experts:**
   ```python
   print(f"Selected: {result['selected_experts']}")
   print(f"Fusion scores: {result['metadata']['fusion_scores']}")
   
   # Sum should be < 0.70
   ```

3. **Possible solutions:**
   - Add domain with related concept
   - Lower COVERAGE_THRESHOLD in tuning_config.py
   - Increase FUSION_WEIGHTS["spectral"] for more flexibility

---

## Performance Tuning

### Benchmarking

```python
import time
from multi_lens_router import MultiLensRouter

router = MultiLensRouter(use_multi_lens=True, use_spectral=True)

# Warm-up
router.route("test")

# Benchmark
queries = [
    "Red car",
    "Earth orbits sun",
    "Glossy finish",
    # ... 100 more queries
]

start = time.time()
for q in queries:
    router.route(q)
elapsed = time.time() - start

print(f"Total: {elapsed:.4f}s")
print(f"Per query: {elapsed/len(queries):.6f}s")
```

### Typical Performance

| Mode | Speed | Notes |
|------|-------|-------|
| Layer 1 only | 0.00001s | Keyword matching only |
| With fusion | 0.00005s | Scoring and optimization |
| With spectral | 0.0001s+ | SentenceTransformer encoding |

### Optimization Tips

1. **Disable spectral in production (if not needed):**
   ```python
   router = MultiLensRouter(use_spectral=False)
   # Fastest: 0.00005s per query
   ```

2. **Batch queries:**
   ```python
   queries = [...]
   results = [router.route(q) for q in queries]
   # Parallelizable if using async
   ```

3. **Cache spectral embeddings:**
   ```python
   # Currently computed fresh each time
   # Future: pre-compute and cache
   ```

---

## Logging & Observability

### Enable Logging

**File:** `multi_lens_router.py`

```python
ENABLE_LOGGING = True
LOG_SAMPLE_RATE = 1  # Log all requests
```

### Example Output

```
Request 1: "Red car"
  Base: NORMAL, primary=automobile
  Spectral: {automobile: 0.55, aesthetics: 0.45}
  Fusion: {automobile: 0.6125, aesthetics: 0.3775}
  Superposition: AMBIGUOUS (1 candidate, variance=0.00125)
  Selected: [automobile, aesthetics]
  Coverage: 0.495 (< 0.70)
  create_new_expert: True
```

### Get Metrics

```python
metrics = router.get_metrics()

print(f"Total requests: {metrics['total_requests']}")
print(f"Distribution:")
print(f"  ATTRIBUTE_ONLY: {metrics['attribute_only']}")
print(f"  SINGLE_DOMAIN: {metrics['single_domain']}")
print(f"  MULTI_DOMAIN: {metrics['multi_domain']}")
print(f"  AMBIGUOUS: {metrics['ambiguous']}")
print(f"  NO_EXPERT: {metrics['no_expert']}")
print(f"Override applied: {metrics['attribute_override_applied']}")
```

---

## Troubleshooting

### Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'sentence_transformers'
```

**Solution:**
```bash
pip install sentence-transformers
```

### Configuration Errors

**Error:**
```
ConfigurationError: FUSION_WEIGHTS must sum to 1.0
```

**Solution:**
```python
# Edit tuning_config.py
FUSION_WEIGHTS = {
    "semantic": 0.45,
    "spectral": 0.35,
    "confidence": 0.20  # Must sum to 1.0
}
```

### Spectral Analysis Errors

**Error:**
```
RuntimeError: SentenceTransformer model not downloaded
```

**Solution:**
```python
# Will auto-download on first use
# Or pre-download:
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
```

### Test Failures

**Error:**
```
AssertionError: SINGLE_DOMAIN != MULTI_DOMAIN
```

**Solution:**
1. Check test expectations (may need update for Phase 6)
2. Check tuning_config.py (parameters may have changed)
3. Run determinism test first
4. Check git diff to see what changed

---

## Common Operations

### Reset Metrics

```python
# Metrics reset automatically on router creation
router = MultiLensRouter()

# Or manually:
from multi_lens_router import MultiLensRouter
MultiLensRouter._METRICS = {
    "total_requests": 0,
    "attribute_only": 0,
    ...
}
```

### Export Metrics to JSON

```python
import json
from multi_lens_router import MultiLensRouter

metrics = MultiLensRouter.get_metrics()
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
```

### Validate Configuration

```python
from tuning_config import validate_config

try:
    validate_config()
    print("✓ Configuration valid")
except Exception as e:
    print(f"✗ Configuration error: {e}")
```

### Compare Routing Modes

```python
from multi_lens_router import MultiLensRouter

# Mode 1: Full multi-lens
router_full = MultiLensRouter(use_multi_lens=True, use_spectral=True)
result1 = router_full.route("Red car")

# Mode 2: No spectral
router_no_spectral = MultiLensRouter(use_multi_lens=True, use_spectral=False)
result2 = router_no_spectral.route("Red car")

# Mode 3: Layer 1 only
router_layer1 = MultiLensRouter(use_multi_lens=False)
result3 = router_layer1.route("Red car")

# Compare
print(f"Full: {result1['classification']}")
print(f"No spectral: {result2['classification']}")
print(f"Layer 1 only: {result3['classification']}")
```

---

## File Reference

### Core Files

| File | Purpose | Modify? |
|------|---------|---------|
| `layer_1_prototype.py` | Semantic + ontology lenses | Add domains only |
| `spectral_analyzer.py` | FFT-based spectral analysis | No |
| `fusion_engine.py` | Weighted score fusion | Via tuning_config |
| `optimization_engine.py` | Greedy expert selection | Via tuning_config |
| `multi_lens_router.py` | Orchestration | Flags only |
| `expert_filter.py` | Domain experts | Add experts |
| `tuning_config.py` | Configuration | YES (tuning) |

### Test Files

| File | Purpose |
|------|---------|
| `test_multilens_system.py` | Full test suite (17 tests) |

### Documentation Files (This Phase)

| File | Purpose |
|------|---------|
| `ARCHITECTURE_OVERVIEW.md` | System design |
| `ROUTING_DECISION_TREE.md` | Routing logic |
| `CONFIGURATION_REFERENCE.md` | Tuning parameters |
| `OPERATIONAL_GUIDE.md` | This file |
| `LIMITATIONS_AND_FUTURE_WORK.md` | Limitations & extensions |

---

## Getting Help

### Check the Docs

1. **System design:** See ARCHITECTURE_OVERVIEW.md
2. **Routing logic:** See ROUTING_DECISION_TREE.md
3. **Configuration:** See CONFIGURATION_REFERENCE.md
4. **Future ideas:** See LIMITATIONS_AND_FUTURE_WORK.md

### Run Diagnostics

```bash
# Full test suite
python3 test_multilens_system.py

# Specific test
python3 -m pytest test_multilens_system.py::TestMultiLensSystem::test_earth_science_single -v

# Debug mode
python3 << 'EOF'
from multi_lens_router import MultiLensRouter
router = MultiLensRouter()
result = router.route("your query")
print(result)
EOF
```

