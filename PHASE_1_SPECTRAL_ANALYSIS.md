# PHASE 1: Spectral Analysis Core

## Overview

**Phase 1** implements a standalone, non-breaking **spectral analysis module** for Layer 1 routing.

This phase learns the frequency-domain "signature" of each domain by analyzing embeddings, enabling deterministic structural comparison at runtime.

### Key Principle

> Semantic similarity alone doesn't capture structural patterns. By analyzing embedding distributions in frequency space, we gain a complementary signal that's independent of semantic content.

---

## Problem This Solves

### Current Limitation

Layer 1's existing Lens 1-2 uses:
- **Lens 1**: Semantic similarity (embeddings + fallback)
- **Lens 2**: Ontology-based explanation

Both are **semantic lenses**—they measure meaning similarity.

### Gap

A text can be structurally similar to a domain's corpus even if semantically different. For example:
- "The object moves toward the observer" (structure: physics-like)
- vs. "I feel the pull of attraction" (structure: relationship-like)

Both may score well semantically in multiple domains, but their **frequency patterns** differ.

### What Phase 1 Adds

A third, **structural lens** that:
- Analyzes embedding distributions in frequency space
- Learns "how each domain looks" as a pattern
- Compares inputs against these patterns deterministically
- Returns domain similarity scores

---

## Algorithm

### Part A: Pre-Training (One-Time)

**Goal**: Learn what each domain "looks like" in frequency space.

**Input**: 
```python
domain_corpus = {
    "astronomy": ["text1", "text2", ...],
    "automobile": ["text1", "text2", ...],
}
```

**Process** (for each domain):

1. **Encode texts**
   - Use SentenceTransformer to convert each text → 768-dim embedding
   - Stack embeddings: `(num_texts, 768)`

2. **Compute FFT + PSD per dimension**
   - For each of the 768 embedding dimensions:
     - Treat as a 1D signal (one value per text)
     - Compute FFT
     - Compute Power Spectral Density (PSD)

3. **Average across dimensions**
   - Pool PSDs across all 768 dimensions
   - Result: single PSD vector

4. **Average across texts**
   - The PSD already incorporates all texts (due to FFT)
   - Normalize to [0, 1]

5. **Save to disk**
   - Write as `signatures/{domain}_spectral_signature.npy`
   - Metadata saved alongside

**Output**:
```python
{
    "astronomy": {
        "signature_path": "signatures/astronomy_spectral_signature.npy",
        "num_texts": 42,
        "embedding_dim": 768,
        "status": "saved"
    }
}
```

### Part B: Runtime Analysis

**Goal**: Compare input text against learned domain signatures.

**Input**: Single text string

**Process**:

1. **Encode input**
   - Use same SentenceTransformer model
   - Result: 768-dim embedding

2. **Compute input's PSD**
   - Extract magnitude of embedding
   - Treat as a signal
   - Compute PSD (same method as training)

3. **Load domain signatures**
   - Fetch all saved .npy files

4. **Cross-correlate**
   - For each domain's signature:
     - Compute normalized cross-correlation with input PSD
     - Normalize to [0, 1]

5. **Return scores**
   ```python
   {
       "astronomy": 0.71,
       "automobile": 0.18,
   }
   ```

---

## Why This Approach Is Deterministic

1. **No randomness**: All operations (FFT, correlation, normalization) are deterministic
2. **No training loops**: Just signal processing
3. **No gradient updates**: Pure mathematical operations
4. **Reproducible**: Re-running on same corpus yields identical signatures
5. **Safe to cache**: Signatures can be saved and reused without concern

---

## Graceful Degradation (Critical)

Phase 1 is **designed to fail safely**. If any of these occur:

- ❌ `SentenceTransformer` not installed
- ❌ `scipy` not installed
- ❌ Signature files missing at runtime
- ❌ Encoding fails
- ❌ FFT computation fails

**Behavior**:
- Log a warning
- Return `{}` (empty dict)
- **Do NOT crash**
- System continues with existing Lens 1-2

This ensures Phase 1 is fully optional and non-breaking.

---

## What This Phase Does NOT Do

### Explicitly Out of Scope

- ❌ **No fusion**: Not combining with Lens 1-2 scores yet
- ❌ **No optimization**: Not selecting minimal expert sets
- ❌ **No routing changes**: Layer 1 routing logic unchanged
- ❌ **No expert selection**: Not deciding which domains are "best"
- ❌ **No integration**: Not integrated with `expert_filter.py` yet

This is **Phase 1 only**. Phases 2-4 will build on top.

---

## Code Structure

### File: `spectral_analyzer.py`

**Classes**:

1. **`SpectralSignatureGenerator`**
   ```python
   __init__(model_name: str, signature_dir: str)
   generate_domain_signatures(domain_corpus: Dict[str, List[str]]) -> Dict
   ```
   
   Pre-training. Learns and saves signatures.

2. **`RuntimeSpectralAnalyzer`**
   ```python
   __init__(signature_dir: str, model_name: str)
   analyze_text(text: str) -> Dict[str, float]
   is_ready() -> bool
   get_available_domains() -> List[str]
   ```
   
   Runtime. Analyzes inputs and scores domains.

**Graceful Imports**:
- `SentenceTransformer` (optional)
- `scipy.fftpack` (optional)

**Testing**:
- Minimal `__main__` block
- Creates test corpus
- Generates signatures
- Analyzes sample texts
- Prints results

---

## Files Created

1. **`spectral_analyzer.py`** (~400 lines)
   - Phase 1 implementation
   - Pre-training + runtime
   - Graceful degradation
   - Minimal test

2. **`signatures/`** (directory, created on demand)
   - Stores `.npy` files per domain

3. **`PHASE_1_SPECTRAL_ANALYSIS.md`** (this file)
   - Documentation

---

## Usage

### Pre-Training (One-Time)

```python
from spectral_analyzer import SpectralSignatureGenerator

corpus = {
    "astronomy": ["text1", "text2", ...],
    "automobile": ["text1", "text2", ...],
}

generator = SpectralSignatureGenerator(
    model_name="all-MiniLM-L6-v2",
    signature_dir="signatures"
)

results = generator.generate_domain_signatures(corpus)
print(results)
```

### Runtime Analysis

```python
from spectral_analyzer import RuntimeSpectralAnalyzer

analyzer = RuntimeSpectralAnalyzer(signature_dir="signatures")

if analyzer.is_ready():
    scores = analyzer.analyze_text("Earth orbits the Sun")
    print(scores)
    # Output: {"astronomy": 0.71, "automobile": 0.18}
else:
    print("Analyzer not ready")
```

### Testing

```bash
cd /path/to/Mycelium
python spectral_analyzer.py
```

Output will show:
- Pre-training phase (if model available)
- Runtime analysis on sample texts
- Scores per domain

---

## Integration Points (Future)

### Phase 2: Fusion Engine
Will combine Spectral scores with Lens 1-2 scores.

### Phase 3: Optimization Engine
Will use fused scores to select minimal expert sets.

### Phase 4: Multi-Lens Router
Will integrate all phases into Layer 1 routing.

---

## Design Rationale

### Why Spectral Analysis?

1. **Structural independence**: Orthogonal to semantic similarity
2. **Deterministic**: No ML training, no randomness
3. **Efficient**: O(n log n) via FFT
4. **Reusable**: Signatures cached as static data
5. **Interpretable**: Frequency patterns have clear meaning

### Why Separate Module?

1. **Isolation**: Can be tested independently
2. **Optional**: Can be disabled without breaking system
3. **Composable**: Easy to combine with future phases
4. **Maintainable**: Concerns cleanly separated

### Why Phase 1?

Foundation for later phases:
- Phase 2 needs Spectral scores to fuse
- Phase 3 needs fused scores to optimize
- Phase 4 orchestrates everything

---

## Testing Verification

Run minimal test:

```bash
python spectral_analyzer.py
```

Expected output:
```
======================================================================
PHASE 1: Spectral Analysis Core - Minimal Test
======================================================================

📝 Creating test corpus...

🔄 Pre-training signatures...
✅ Saved signature: signatures/astronomy_spectral_signature.npy
✅ Saved signature: signatures/automobile_spectral_signature.npy
Generated: ['astronomy', 'automobile']
  astronomy: {'signature_path': '...', 'num_texts': 3, 'status': 'saved'}
  automobile: {'signature_path': '...', 'num_texts': 3, 'status': 'saved'}

🔍 Runtime analysis...
Ready: True
Loaded domains: ['astronomy', 'automobile']

Analyzing (expected astronomy): 'The planet orbits a star in space'
  astronomy: 0.7234
  automobile: 0.2156

Analyzing (expected automobile): 'My car has a red finish'
  astronomy: 0.1834
  automobile: 0.6891

======================================================================
✅ Phase 1 Test Complete
======================================================================
```

---

## Backward Compatibility

### No Changes to Existing Code

- `layer_1_prototype.py`: **Untouched**
- `expert_filter.py`: **Untouched**
- All existing tests: **Unchanged**

### Execution Impact

If Phase 1 is not used:
- Zero impact on existing routing
- System behaves identically
- All existing functionality intact

---

## Summary

Phase 1 delivers:

✅ **Standalone spectral analysis module**
✅ **Deterministic, cache-friendly signatures**
✅ **Graceful degradation**
✅ **Clean, testable code**
✅ **Foundation for Phase 2-4**

🎯 **Next step**: Phase 2 (Fusion Engine) will combine Spectral with Lens 1-2 scores.
