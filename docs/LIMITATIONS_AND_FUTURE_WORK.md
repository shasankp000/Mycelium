# Limitations and Future Work

**Project Mycelium — Design Constraints, Honest Assessment, and Extension Ideas**  
**Version:** 1.0  
**Date:** January 17, 2026

---

## Overview

This document honestly describes what the system **cannot do**, why by design, and how to extend it responsibly in the future.

---

## System Design Constraints

### Constraint 1: Determinism First

**Principle:** Every routing decision is deterministic and reproducible.

**Why:** Debugging requires reproducibility. Non-deterministic systems hide bugs.

**Implication:** No randomness allowed anywhere:
- No random tie-breaking
- No probabilistic sampling
- No Monte Carlo methods
- No stochastic gradient descent

**Cost:** Cannot use modern ML approaches that require stochasticity.

**Extension:** Future systems can add randomness IF fully isolated and toggled.

---

### Constraint 2: No Online Learning

**Principle:** System parameters are static, configured via `tuning_config.py`.

**Why:** Online learning requires maintaining state, which breaks determinism and complicates debugging.

**Implication:**
- Cannot adapt to user feedback during operation
- Cannot auto-tune from queries
- Cannot personalize per user/domain

**Cost:** Must manually tune; no automatic improvement.

**Extension:** Future systems can add offline learning (batch retraining between deployments).

---

### Constraint 3: Finite Domain Set

**Principle:** Expert system has fixed set of domains (Layer 1 DOMAIN_DEFINITIONS).

**Why:** Automatic domain discovery is hard; explicit is better.

**Implication:**
- Queries don't map to undefined domains
- Must add domains manually
- NO infinite domain expansion

**Cost:** Scaling requires manual curation.

**Extension:** Future systems can add domain discovery (with human review).

---

### Constraint 4: Flat Confidence Scores

**Principle:** Lens 2 returns confidence=0.5 for all domains (placeholder).

**Why:** Confidence calibration requires labeled data; out of scope.

**Implication:**
- Cannot distinguish "high confidence" from "low confidence" in ontology matches
- Confidence weight in fusion has minimal effect

**Cost:** Suboptimal fusion weighting.

**Extension:** Future systems can learn confidence from labeled evaluation data.

---

### Constraint 5: No Architecture Changes

**Principle:** Three lenses, spectral, fusion, optimization pipeline is fixed.

**Why:** Architecture changes break all testing and guarantees.

**Implication:**
- Cannot add new lenses mid-project
- Cannot change pipeline order
- Cannot modify algorithm flow

**Cost:** Limited to tuning, not redesign.

**Extension:** Future systems can propose alternative architectures (with fresh validation).

---

### Constraint 6: No Domain Hardcoding

**Principle:** Routing logic is domain-agnostic.

**Why:** Domain-specific logic doesn't scale; it bloats code.

**Implication:**
- Special cases for aesthetics vs. engine vs. astronomy are forbidden
- All domains treated equally
- Logic must work for ANY domain set

**Cost:** Cannot optimize for specific domains.

**Extension:** If domain-specific optimization needed, create domain-specific expert filters (not routers).

---

## Honest Limitations

### Limitation 1: Attribute-Only Queries

**Problem:** Queries with only modifiers (no core concepts) don't route.

**Example:**
```
Input: "Glossy red finish"
Output: ATTRIBUTE_ONLY, no expert selected
```

**Why:**
- Layer 1 Lens 1 requires keyword matches
- Pure attributes don't define domain
- Override rule (Phase 6) helps but isn't perfect

**When It Hurts:**
- Fashion: "Plaid, wool, waterproof" (no domain keyword)
- Design: "Minimalist, sleek, modern"
- Arts: "Abstract, surreal, impressionistic"

**Workaround:**
- Increase DOMAIN_SCORE_THRESHOLD to be more aggressive with overrides
- Add more keywords to domain definitions
- Use expert filters to handle attribute-only queries post-routing

### Limitation 2: Multi-Domain Ambiguity

**Problem:** Queries with multiple similar-scoring domains classified as AMBIGUOUS.

**Example:**
```
Input: "Vehicle mechanics and engineering"
Output: AMBIGUOUS [automobile, engine_spec]
Variance: 0.005 (too low for MULTI_DOMAIN)
```

**Why:**
- Variance threshold (0.08) is fixed
- Cannot distinguish "intentional multi-domain" from "ambiguous query"

**When It Hurts:**
- Interdisciplinary queries: "Physics and chemistry experiment"
- Compound products: "Electronic vehicle propulsion"
- Cross-domain comparisons: "Astronomy vs. geology"

**Workaround:**
- Lower SUPERPOSITION_VARIANCE_THRESHOLD (make more liberal)
- Add query context to distinguish intent
- Use secondary classifier for disambiguati

### Limitation 3: Coverage Threshold Issues

**Problem:** Some queries can never meet 0.70 coverage, triggering create_new_expert=True.

**Example:**
```
Input: "Obscure scientific term"
Output: coverage=0.38, create_new_expert=True
```

**Why:**
- Limited domain expertise
- Spectral analysis may not have signature match
- Single weak expert can't reach threshold alone

**When It Hurts:**
- Novel domains: queries outside training set
- Emerging fields: new terminology
- Niche expertise: very specialized topics

**Workaround:**
- Lower COVERAGE_THRESHOLD (accept lower coverage)
- Add more related domains
- Create catch-all expert

### Limitation 4: Spectral Dependency

**Problem:** Spectral analysis (Phase 1) requires SentenceTransformer model (~100MB).

**Why:**
- Model must be downloaded once (~5 min first run)
- Adds ~0.0001s per query
- Requires internet for download

**When It Hurts:**
- Offline systems (no internet)
- Embedded devices (size constraints)
- Latency-critical systems (millisecond scale)

**Workaround:**
- Disable spectral: `use_spectral=False`
- Pre-download model
- Use quantized/distilled model

### Limitation 5: No Semantic Relationships

**Problem:** Domain relationships are stored as ontology but not used for routing.

**Current:**
```python
"related_concepts": {
    "astronomy": 0.7
}
```

**Not used in:** Fusion, optimization, or selection.

**Why:** Would add complexity; kept simple.

**Impact:** Miss opportunities for cross-domain routing.

**Workaround:**
- Add related domains to keywords
- Use expert filters to do relationship lookup

### Limitation 6: Linear Scaling with Domains

**Problem:** Routing time scales linearly with number of domains O(n).

**Currently:** ~20 domains → 0.0001s

**At 10,000 domains:** ~0.0005s (still fast, but grows)

**Why:** Must compute scores for all domains.

**When It Hurts:** Very large expert systems (>1000 domains).

**Workaround:**
- Use domain hierarchy (master categories only)
- Pre-filter domains before routing
- Shard into sub-routers per category

### Limitation 7: No User Feedback Loop

**Problem:** System cannot learn from routing errors.

**Example:**
```
User: "Route 'red car' to automobile"
System: routes to [automobile, aesthetics]
User: "wrong, should be [automobile] only"
System: (no change, static config)
```

**Why:** Online learning violates determinism constraint.

**Impact:** Cannot adapt to domain-specific user preferences.

**Workaround:**
- Manual evaluation → adjust tuning_config.py
- Batch retraining between deployments
- Create user-specific configuration profiles

---

## Not Implemented (Intentionally)

These features were considered but rejected for scope/design reasons:

### 1. Query Expansion
**Why rejected:** Would require domain-specific knowledge, hard to generalize.

### 2. Spell Correction
**Why rejected:** Adds uncertainty; determinism requires exact matches.

### 3. Synonym Handling
**Why rejected:** Synonym sets are domain-specific; no generic solution.

### 4. Negation ("NOT operator")
**Why rejected:** Adds logic complexity; queries rarely use negation.

### 5. Hierarchical Routing
**Why rejected:** Adds complexity; flat routing simpler to debug.

### 6. Temporal Routing ("latest" vs "historical")
**Why rejected:** Out of scope; expert selection, not result ordering.

### 7. User Personalization
**Why rejected:** Requires per-user state; violates determinism.

### 8. Multi-Language Support
**Why rejected:** Would need language-specific domain definitions.

### 9. Reasoning Explanations
**Why rejected:** Possible but adds significant complexity; current explanations sufficient.

### 10. Confidence Scores
**Why rejected:** Requires labeled evaluation data; out of scope.

---

## Future Work

### Phase 8: Offline Learning (Medium Difficulty)

**Objective:** Improve tuning through batch retraining.

**Approach:**
1. Collect query logs (what users asked)
2. Collect evaluations (what they said was correct)
3. Analyze disagreements (where system wrong)
4. Propose new tuning values
5. A/B test before deployment

**Constraints:**
- Must maintain determinism
- No online learning during operation
- Human review of proposed changes

**Example:**
```
Collected logs:
  Query: "red car"
  System: AMBIGUOUS [automobile, aesthetics]
  User: should be SINGLE_DOMAIN [automobile]

Proposed tuning:
  SUPERPOSITION_VARIANCE_THRESHOLD: 0.08 → 0.06
  (make MULTI_DOMAIN stricter)
```

### Phase 9: Domain Discovery (Hard Difficulty)

**Objective:** Automatically propose new domains from unmatched queries.

**Approach:**
1. Track create_new_expert=True queries
2. Cluster similar queries
3. Extract keywords from clusters
4. Propose domain definition
5. Human review & addition

**Constraints:**
- Human review required (don't auto-add)
- Determinism maintained
- Manual domain definition

**Example:**
```
Unmatched queries (cluster):
  "Circuit design"
  "Semiconductor fabrication"
  "Transistor behavior"

Proposed domain:
  "electronics": {
    "keywords": ["circuit", "transistor", "semiconductor"],
    "modifiers": ["low-power", "high-frequency"],
    ...
  }
```

### Phase 10: Confidence Calibration (Hard Difficulty)

**Objective:** Learn when to trust which lens.

**Approach:**
1. Collect labeled evaluation data (correct domains)
2. Train lightweight model: (semantic, spectral, ontology) → correct?
3. Output per-domain confidence instead of flat 0.5
4. Adjust FUSION_WEIGHTS based on lens reliability

**Constraints:**
- Labeled data required (~500+ examples)
- Model must be tiny (no neural nets)
- Determinism maintained (no randomness)

**Example:**
```
Training data:
  Query: "red car", Lens1: automobile, Lens2: aesthetics → Correct: automobile
  Query: "glossy finish", Lens1: [], Lens2: aesthetics → Correct: aesthetics

Learned:
  If Lens1 found AND confidence > X: trust Lens1 (0.9)
  If Lens1 empty: trust Lens2 (0.6)
```

### Phase 11: Hierarchical Domains (Medium Difficulty)

**Objective:** Support domain hierarchy (master categories + subcategories).

**Approach:**
1. Organize domains into tree (e.g., STEM → Physics, Chemistry, Biology)
2. Route first to master category
3. Then to subcategory
4. Return path as routing result

**Example:**
```
STEM
├── Physics
│   ├── Classical Mechanics
│   ├── Thermodynamics
│   └── Astronomy
├── Chemistry
│   ├── Organic
│   └── Inorganic
└── Biology
    ├── Botany
    └── Zoology
```

**Constraint:** Must update all phases.

### Phase 12: Multi-Language Support (Hard Difficulty)

**Objective:** Route queries in multiple languages.

**Approach:**
1. Detect language (textblob or similar)
2. Translate to English (or use multilingual model)
3. Route as normal
4. Return results

**Constraint:** Translation adds latency & error.

### Phase 13: Temporal Stability (Medium Difficulty)

**Objective:** Ensure routing doesn't change over time.

**Approach:**
1. Log tuning_config.py version with each result
2. Alert if config changed unexpectedly
3. Support config versioning (v1.0, v1.1, etc.)
4. Ability to revert to previous config

**Constraint:** Requires git integration.

### Phase 14: Expert Performance Monitoring (Easy Difficulty)

**Objective:** Track which experts perform well.

**Approach:**
1. Route query to experts
2. Collect user feedback on expert results
3. Track expert success rate per domain
4. Alert if expert performance degrades
5. Suggest expert replacement if needed

**Constraint:** Requires feedback collection infrastructure.

---

## Safe Extensions

These extensions maintain system constraints and can be added safely:

### Extension 1: Expert Filters (Safe)

Add domain-specific result filtering post-routing:

```python
# In expert_filter.py
class AutomobileExpert:
    def filter_results(self, results):
        # Filter for automobile-relevant results
        return [r for r in results if self._is_relevant(r)]
```

**Why safe:** Doesn't change routing logic, only output filtering.

### Extension 2: Query Preprocessing (Safe)

Add query normalization before routing:

```python
def preprocess_query(text):
    text = text.lower()
    text = remove_common_stopwords(text)
    return text
```

**Why safe:** Doesn't change routing logic, input normalization only.

### Extension 3: Configuration Profiles (Safe)

Add multiple configuration variants:

```python
# tuning_config_conservative.py
# tuning_config_aggressive.py
# tuning_config_balanced.py
```

**Why safe:** Doesn't change logic, just configuration variants.

### Extension 4: Result Caching (Safe)

Cache routing results for identical queries:

```python
cache = {}
def route_cached(query):
    if query in cache:
        return cache[query]
    result = router.route(query)
    cache[query] = result
    return result
```

**Why safe:** Performance optimization, doesn't change logic.

### Extension 5: Batch Routing (Safe)

Add bulk routing for efficiency:

```python
def batch_route(queries):
    return [router.route(q) for q in queries]
```

**Why safe:** No logic changes, just convenience.

### Extension 6: Metrics Dashboard (Safe)

Add visualization of metrics:

```python
metrics = router.get_metrics()
# Visualize in UI: pie chart, trends, alerts
```

**Why safe:** Observability only, doesn't affect logic.

---

## Unsafe Extensions (Violate Constraints)

These would require revisiting core constraints:

### ❌ Online Learning
- Would break determinism
- Use batch learning instead

### ❌ Randomized Tie-Breaking
- Would break reproducibility
- Use deterministic tie-breaker

### ❌ User Personalization
- Would require per-user state
- Use configuration profiles instead

### ❌ Probabilistic Routing
- Would break determinism
- Use ensemble if needed

### ❌ Dynamic Domain Addition
- Would break completeness checking
- Use manual domain addition + retest

---

## Design Rationale

### Why Determinism?

**Benefit:** Debugging is tractable.
- Same query always routes same way
- Errors are reproducible
- Can pinpoint root cause

**Cost:** Cannot use modern stochastic ML.

**Justification:** Expert routing is not ML problem; it's logic problem.

---

### Why No Online Learning?

**Benefit:** Configuration is explicit and auditable.
- Can read tuning_config.py and understand behavior
- Can diff configurations across deployments
- Can revert safely

**Cost:** Manual tuning required.

**Justification:** Expert systems should be transparent, not black boxes.

---

### Why Flat Confidence?

**Benefit:** Simpler, fewer parameters.
- No overfitting to confidence data
- Easier to understand

**Cost:** Suboptimal fusion weights.

**Justification:** Confidence is orthogonal to routing; can be added later if needed.

---

### Why Three Lenses?

**Benefit:** Redundancy + diversity.
- Layer 1 is reliable (keyword-based)
- Lens 3 is novel (spectral)
- Fusion reduces bias

**Cost:** Added complexity.

**Justification:** Three is Goldilocks number; enough diversity without explosion.

---

## Lessons Learned

### Lesson 1: Determinism Matters
- Started assuming randomness OK
- Discovered it makes debugging impossible
- Now: deterministic by design

### Lesson 2: Simple Is Better
- Considered many complex features
- Realized most not needed
- Now: implement only essential

### Lesson 3: Configuration Over Code
- Tempted to hardcode thresholds
- Realized flexibility needed
- Now: all tuning in config file

### Lesson 4: Testing Early, Often
- Built Phase 5 test suite early
- Caught regressions immediately
- Now: 17 tests validate each change

### Lesson 5: Documentation Matters
- Code alone doesn't explain decisions
- Design rationale easy to forget
- Now: comprehensive documentation in Phase 7

---

## Conclusion

This system is **production-ready within its constraints**:
- Deterministic routing
- Tunable configuration
- Comprehensive testing
- Clear documentation

It is **NOT**:
- A general ML classifier
- A semantic search engine
- A personalization system
- An automatic learner

If you need those, consider different architectures. If you need a **deterministic, debuggable, tunable expert router**, this is it.

---

## Questions? Next Steps?

1. **Understand limitations:** Read this doc carefully
2. **Try extensions:** Implement safe extensions (Section above)
3. **Propose changes:** Document in terms of Phase 8+ (Future Work)
4. **Contribute:** Follow existing patterns, don't violate constraints

