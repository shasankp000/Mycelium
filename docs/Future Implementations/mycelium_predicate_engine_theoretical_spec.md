# Mycelium Predicate Engine — Hardened Theoretical Specification (v0.2)

**Version:** 0.2-hardened  
**Branch target:** `web-ui-prototype`  
**Status:** Pre-implementation design  
**Date:** 2026-05-27

---

# 1. Motivation

The Mycelium pipeline (L0 → L6) currently routes and classifies queries with
high sophistication but cannot *reason over structured claims*. Phases 2–3
operate on raw text, meaning every sub-phase re-parses independently and the
contradiction/fusion layer (ContradictionClassifier, DSTFusion) has no
structured input to work with.

This specification defines the missing semantic reasoning substrate required
for true structured reasoning:

1. Predicate Engine
2. Negation Engine
3. Evidence Finder (Refutation-First)
4. PredicateStore
5. Predicate Provenance Layer
6. Structured Evidence Fusion

This transitions Mycelium from:

```text
text → text orchestration
```

into:

```text
text → semantic IR → reasoning
```

---

# 2. Architectural Philosophy

## Core Principle

Knowledge is NOT reasoning.

The system is decomposed into:

| Responsibility | Component |
|---|---|
| Semantic extraction | PredicateExtractor |
| Knowledge retrieval | EvidenceFinder |
| Semantic storage | PredicateStore |
| Contradiction handling | DSTFusion |
| Reasoning orchestration | TRM + DAG/DFS |
| Ontology alignment | ontology governance |
| Semantic stabilization | leverage propagation |

PredicateFrames become:

```text
first-class semantic reasoning units
```

rather than:
```text
NLP metadata blobs
```

---

# 3. Predicate Engine Pipeline

```text
Raw query
    ↓
nlp_preprocessor
    ↓
L0 Gate
    ↓
PredicateExtractor
    ↓
PredicateNegator
    ↓
PredicateStore
    ↓
EvidenceFinder
    ↓
DSTFusion
    ↓
Semantic stabilization
```

---

# 4. PredicateFrame — Semantic IR

```python
PredicateFrame
  ├── predicate_id
  ├── surface_form
  ├── predicate_type
  ├── subject
  ├── relation
  ├── object
  ├── negated
  ├── quantifier_scope
  ├── scope_conditions
  ├── presupposition_attack
  ├── presupposition_frame
  ├── falsifiable
  ├── refutation_burden
  ├── negated_form
  ├── domain_hint
  ├── source_span

  # Hardened additions
  ├── derived_from
  ├── generated_by_phase
  ├── transformation_history
  ├── semantic_revision
  ├── provenance_confidence
  ├── graph_links
```

---

# 5. NEW — Predicate Provenance Layer

## Motivation

Predicates will evolve over time due to:

- contradiction propagation
- ontology updates
- patch DAG corrections
- scope refinement
- stabilization
- semantic reinterpretation

Without provenance lineage:
semantic debugging becomes extremely difficult.

---

## Provenance Fields

```python
derived_from: List[predicate_id]
```

Tracks:
- parent predicates
- decomposition lineage
- presupposition derivation
- contradiction-derived predicates

---

```python
generated_by_phase: str
```

Tracks:
- extraction origin
- reasoning phase source

Examples:
- PredicateExtractor
- PredicateNegator
- StabilizationEngine
- PatchResolver

---

```python
transformation_history: List[TransformationRecord]
```

Tracks:
- negation transforms
- scope narrowing
- ontology rewrites
- contradiction rewrites
- patch applications

---

## TransformationRecord

```python
TransformationRecord
  ├── timestamp
  ├── transformation_type
  ├── source_predicate
  ├── resulting_predicate
  ├── reason
  ├── triggered_by
```

---

# 6. Predicate Relations (Future-Compatible Foundation)

## Motivation

Predicates do not exist independently.

Eventually:
- predicates support each other
- contradict each other
- specialize each other
- imply each other

This specification introduces the foundational structure
without enabling full cross-predicate inference yet.

---

## PredicateRelation

```python
PredicateRelation
  ├── source_predicate_id
  ├── target_predicate_id
  ├── relation_type
  ├── confidence
```

---

## Supported Relation Types

```text
SUPPORTS
CONTRADICTS
PRESUPPOSES
SPECIALIZES
GENERALIZES
DEPENDS_ON
TEMPORALLY_PRECEDES
```

---

## IMPORTANT

Cross-predicate inference remains OUT OF SCOPE for v0.2.

However:
PredicateRelation exists now so later systems can integrate
without redesigning PredicateFrame itself.

---

# 7. PredicateStore — Semantic Working Memory

## Architectural Evolution

PredicateStore is NOT merely:
```text
request-scoped storage
```

It is evolving toward:
```text
transient semantic working memory
```

---

## Responsibilities

PredicateStore provides:

- indexed predicate retrieval
- request-scoped semantic state
- contradiction-aware lookup
- replay support
- provenance lookup
- patch integration
- stabilization visibility

---

## Query API

```python
PredicateStore.get(predicate_id)
PredicateStore.by_type(predicate_type)
PredicateStore.by_domain(domain)
PredicateStore.by_falsifiability(flag)
PredicateStore.by_relation(relation_type)
PredicateStore.related(predicate_id)
```

---

# 8. Negation Rules (Deterministic)

## Core Principle

Negation MUST remain:

- deterministic
- replayable
- non-generative
- structurally analyzable

LLM-generated negation is forbidden.

Reason:
- hallucination risk
- replay instability
- contradiction inconsistency
- non-deterministic stabilization

---

## Rule N1 — FACTIVE

```text
Flip polarity
```

---

## Rule N2 — COMPARATIVE

```text
WordNet antonym lookup
fallback = not_<relation>
```

---

## Rule N3 — CAUSAL

```text
Invert causal relation
adjust refutation burden
```

---

## Rule N4 — EXISTENTIAL

```text
Apply De Morgan inversion
```

---

## Rule N5 — NORMATIVE

```text
Mark NON_FALSIFIABLE
skip retrieval
route to value reasoning
```

---

## Rule N6 — Presupposition Attack

```text
Attack hidden assumption frame
NOT surface wording
```

---

# 9. EvidenceFinder — Refutation First

## Retrieval Strategy

```text
Pass 1:
Search for evidence supporting ¬C

Pass 2:
Search for evidence supporting C

Pass 3:
DSTFusion combines both
```

---

## Epistemic Motivation

Refutation-first retrieval reduces:

- confirmation bias
- retrieval anchoring
- semantic overfitting
- cherry-picked evidence

---

## Early Halt Rule

For:

```text
UNIVERSAL claims
```

with:

```text
SINGLE_COUNTEREXAMPLE
```

refutation burden:

```text
halt retrieval immediately
```

after one strong counterexample.

---

# 10. EvidenceBundle

```python
EvidenceBundle
  ├── predicate_id
  ├── supports
  ├── refutes
  ├── confirmation_score
  ├── refutation_score
  ├── net_confidence
  ├── scope_qualified
  ├── verdict
  ├── tool_log

  # Hardened additions
  ├── contradiction_trace
  ├── stabilization_notes
  ├── evidence_lineage
```

---

# 11. DSTFusion Integration

DSTFusion no longer operates on:

```text
raw retrieval text
```

Instead it operates on:

```text
structured semantic evidence
```

This enables:
- scoped confidence propagation
- contradiction-aware fusion
- leverage weighting
- semantic stabilization
- future ontology arbitration

---

# 12. Semantic Stabilization Integration

PredicateFrames become first-class graph objects.

This allows:
- contradiction propagation
- leverage decay
- ontology revision
- graph replay
- semantic snapshots
- future distributed reasoning

---

# 13. Explicitly Out of Scope (v0.2)

Still excluded:

- generative LLM negation
- vector databases
- cross-predicate theorem proving
- temporal sequence logic
- autonomous ontology rewriting
- distributed predicate synchronization

These remain future-phase systems.

---

# 14. Success Criteria

The hardened Predicate Engine is correct when:

1. PredicateFrames become reusable semantic IR objects
2. Negation remains deterministic and replay-safe
3. NORMATIVE predicates never enter contradiction trees
4. UNIVERSAL claims halt on first strong counterexample
5. Predicate provenance lineage is traceable
6. PredicateStore acts as semantic working memory
7. DSTFusion receives structured EvidenceBundles
8. Predicate relations are graph-compatible
9. Contradiction propagation can reference predicate lineage
10. Semantic replay reconstructs the full predicate evolution chain

---

# 15. Final Architectural Principle

Before:

```text
Mycelium reasoned over text
```

Now:

```text
Mycelium reasons over formalized semantic claims
```

This transforms Mycelium from:

```text
LLM orchestration framework
```

into:

```text
semantic reasoning runtime
```
