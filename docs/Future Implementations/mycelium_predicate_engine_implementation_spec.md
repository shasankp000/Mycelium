# Mycelium Predicate Engine — Hardened Implementation Specification (v0.2)

**Version:** 0.2-hardened  
**Target branch:** `web-ui-prototype`  
**Status:** Build plan  
**Date:** 2026-05-27

---

# 1. Goal

Implement a deterministic semantic predicate engine that:

1. Detects claim-bearing propositions
2. Produces structured `PredicateFrame` semantic IR objects
3. Computes deterministic structural negations
4. Tracks semantic provenance lineage
5. Builds request-scoped semantic working memory
6. Performs refutation-first evidence search
7. Produces structured `EvidenceBundle` outputs
8. Feeds semantic evidence into DSTFusion + stabilization systems
9. Enables future contradiction propagation and semantic replay

This implementation formally transitions Mycelium from:

```text
text → text orchestration
```

into:

```text
text → semantic IR → reasoning
```

---

# 2. Architectural Principles

## Principle 1 — Knowledge ≠ Reasoning

The predicate engine is NOT a knowledge system.

It is:
```text
semantic reasoning infrastructure
```

Knowledge retrieval remains:
- tools
- SQLite
- retrieval layers
- graph stores
- patch DAGs

Reasoning remains:
- TRM
- DAG/DFS
- stabilization
- contradiction propagation
- ontology governance

---

## Principle 2 — Determinism First

Predicate processing MUST remain:

- deterministic
- replayable
- graph-safe
- stabilization-safe

LLM-generated semantic transforms are forbidden in v0.2.

This includes:
- negation
- contradiction generation
- scope inversion
- ontology rewrites

---

## Principle 3 — PredicateFrames Are First-Class Runtime Objects

PredicateFrames are NOT temporary parsing artifacts.

They are:
```text
first-class semantic reasoning units
```

that survive across:
- retrieval
- contradiction
- stabilization
- replay
- patch propagation

---

# 3. Recommended Delivery Order

## Phase A — Semantic IR Foundation

1. `predicate_types.py`
2. `predicate_relations.py`
3. `predicate_store.py`

Goal:
- stable semantic object layer

---

## Phase B — Extraction + Negation

4. `predicate_extractor.py`
5. `predicate_negator.py`
6. provenance tracking

Goal:
- deterministic semantic claim construction

---

## Phase C — Evidence Layer

7. `evidence_types.py`
8. `evidence_finder.py`
9. retrieval tools

Goal:
- refutation-first semantic evidence

---

## Phase D — Fusion + Stabilization

10. DSTFusion adapter
11. contradiction integration
12. semantic replay hooks

Goal:
- structured reasoning over semantic claims

---

# 4. New Modules

---

## 4.1 `mycelium/pipeline/predicates/predicate_types.py`

Defines all semantic IR structures.

```python
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Literal
```

---

### PredicateType

```python
class PredicateType(str, Enum):
    FACTIVE = "FACTIVE"
    COMPARATIVE = "COMPARATIVE"
    CAUSAL = "CAUSAL"
    EXISTENTIAL = "EXISTENTIAL"
    TEMPORAL = "TEMPORAL"
    MODAL = "MODAL"
    NORMATIVE = "NORMATIVE"
    DEFINITIONAL = "DEFINITIONAL"
```

---

### QuantifierScope

```python
class QuantifierScope(str, Enum):
    UNIVERSAL = "UNIVERSAL"
    EXISTENTIAL = "EXISTENTIAL"
    SCOPED = "SCOPED"
    AMBIGUOUS = "AMBIGUOUS"
```

---

### RefutationBurden

```python
class RefutationBurden(str, Enum):
    SINGLE_COUNTEREXAMPLE = "SINGLE_COUNTEREXAMPLE"
    EXHAUSTIVE = "EXHAUSTIVE"
    SCOPE_BOUNDED = "SCOPE_BOUNDED"
    NON_FALSIFIABLE = "NON_FALSIFIABLE"
```

---

### PredicateEntity

```python
@dataclass
class PredicateEntity:
    text: str
    entity_type: str = ""
    confidence: float = 0.8
```

---

### PredicateRelation

```python
@dataclass
class PredicateRelation:
    lemma: str
    polarity: Literal["POSITIVE", "NEGATIVE"] = "POSITIVE"
    tense: str = "PRESENT"
    modality: str = "CERTAIN"
```

---

### TransformationRecord

NEW HARDENED STRUCTURE.

Tracks semantic evolution lineage.

```python
@dataclass
class TransformationRecord:
    timestamp: float
    transformation_type: str
    source_predicate: str
    resulting_predicate: str
    reason: str
    triggered_by: str
```

---

### PredicateFrame

```python
@dataclass
class PredicateFrame:
    predicate_id: str
    surface_form: str

    predicate_type: PredicateType

    subject: PredicateEntity
    relation: PredicateRelation
    object: PredicateEntity | str

    negated: bool = False

    quantifier_scope: QuantifierScope = QuantifierScope.AMBIGUOUS

    scope_conditions: List[str] = field(default_factory=list)

    presupposition_attack: bool = False
    presupposition_frame: Optional["PredicateFrame"] = None

    falsifiable: bool = True

    refutation_burden: RefutationBurden = RefutationBurden.SCOPE_BOUNDED

    negated_form: Optional["PredicateFrame"] = None

    domain_hint: str = "general"

    source_span: Tuple[int, int] = (0, 0)

    # HARDENED ADDITIONS

    derived_from: List[str] = field(default_factory=list)

    generated_by_phase: str = "PredicateExtractor"

    transformation_history: List[TransformationRecord] = field(default_factory=list)

    semantic_revision: int = 0

    provenance_confidence: float = 1.0

    graph_links: List[str] = field(default_factory=list)
```

---

## 4.2 `mycelium/pipeline/predicates/predicate_relations.py`

NEW MODULE.

Provides future-compatible graph linkage primitives.

```python
class PredicateRelationType(str, Enum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    PRESUPPOSES = "PRESUPPOSES"
    SPECIALIZES = "SPECIALIZES"
    GENERALIZES = "GENERALIZES"
    DEPENDS_ON = "DEPENDS_ON"
    TEMPORALLY_PRECEDES = "TEMPORALLY_PRECEDES"
```

---

```python
@dataclass
class PredicateGraphRelation:
    source_predicate_id: str
    target_predicate_id: str
    relation_type: PredicateRelationType
    confidence: float
```

IMPORTANT:
- no theorem proving yet
- no autonomous inference yet
- relation graph only

---

## 4.3 `predicate_extractor.py`

Consumes:
```python
SentenceAnalysis
```

Returns:
```python
List[PredicateFrame]
```

---

### Extraction Rules

Initial extraction remains:
- heuristic-first
- classifier-second

DO NOT build a second NLP stack.

Consume existing:
- dep triples
- SRL
- coercive signals
- presupposition triggers
- L0 reasoning metadata

---

### REQUIRED HARDENING

Extractor MUST emit:
- provenance metadata
- semantic revision = 0
- generated_by_phase
- transformation history root node

---

### IMPORTANT

Initial implementation target:
```text
ONE primary predicate per sentence
```

Multi-predicate decomposition remains future work.

---

## 4.4 `predicate_negator.py`

Implements:

```python
negate(predicate: PredicateFrame)
```

Returns:
```python
PredicateFrame | None
```

---

### HARD REQUIREMENTS

Negation MUST:
- preserve provenance
- append transformation record
- increment semantic_revision
- preserve source lineage

---

### REQUIRED RULES

| Predicate Type | Rule |
|---|---|
| FACTIVE | polarity flip |
| COMPARATIVE | WordNet antonym |
| CAUSAL | invert causal polarity |
| EXISTENTIAL | De Morgan inversion |
| NORMATIVE | return None |
| MODAL | confidence-capped inversion |
| TEMPORAL | temporal inversion |
| DEFINITIONAL | scoped definitional negation |

---

### WordNet Rules

```python
from nltk.corpus import wordnet as wn
```

---

### REQUIRED CACHE

Antonym lookup MUST use:
```python
LRU cache or memoized dict
```

to avoid repeated traversal cost.

---

### FORBIDDEN

```python
LLM negation generation
```

Reason:
- replay instability
- contradiction nondeterminism
- semantic drift

---

## 4.5 `predicate_store.py`

PredicateStore becomes:
```text
semantic working memory
```

NOT:
```text
temporary request list
```

---

### Required API

```python
class PredicateStore:

    def all(self): ...
    def get(self, predicate_id): ...
    def primary(self): ...

    def by_type(self, ptype): ...
    def by_domain(self, domain): ...

    def falsifiable(self): ...

    def related(self, predicate_id): ...

    def lineage(self, predicate_id): ...

    def descendants(self, predicate_id): ...
```

---

### REQUIRED HARDENING

Store MUST support:
- provenance traversal
- relation lookup
- replay-safe ordering
- deterministic iteration

---

### IMPORTANT

Predicate insertion order MUST use:
```python
sequence_number
```

NOT:
```python
dict insertion ordering assumptions
```

---

## 4.6 `evidence_types.py`

### EvidenceItem

```python
@dataclass
class EvidenceItem:
    text: str
    source: str
    relevance: float
    recency_score: float
    supports_original: bool
    scope_match: float
```

---

### ToolCallRecord

```python
@dataclass
class ToolCallRecord:
    tool_name: str
    success: bool
    latency_ms: float
    error: str = ""
```

---

### EvidenceBundle

```python
@dataclass
class EvidenceBundle:

    predicate_id: str

    supports: list[EvidenceItem]
    refutes: list[EvidenceItem]

    confirmation_score: float
    refutation_score: float

    net_confidence: float

    scope_qualified: bool

    verdict: str

    tool_log: list[ToolCallRecord]

    # HARDENED ADDITIONS

    contradiction_trace: list[str]

    stabilization_notes: list[str]

    evidence_lineage: list[str]
```

---

## 4.7 `evidence_finder.py`

Implements:
```text
refutation-first retrieval
```

---

### REQUIRED FLOW

```python
for predicate in predicate_store.falsifiable():

    neg = predicate.negated_form

    refutes = search_supporting(neg)

    if single_counterexample(refutes):
        supports = []
    else:
        supports = search_supporting(predicate)

    bundle = score_and_fuse(...)
```

---

### REQUIRED HARDENING

Evidence search MUST:
- preserve provenance
- preserve evidence lineage
- preserve retrieval ordering
- emit replay-safe event IDs

---

### IMPORTANT

Retrieval MUST remain:
```text
bounded
```

NO unbounded DFS retrieval expansion.

---

# 5. Tool Layer

---

## 5.1 CalculatorTool

Safe arithmetic only.

FORBIDDEN:
```python
eval()
```

Use:
- AST whitelist
- bounded operations
- safe parsing

---

## 5.2 WikipediaRetrieverTool

Initial factual retrieval substrate.

MUST:
- score lexical overlap
- preserve provenance
- emit SSE events
- preserve source URLs

---

# 6. Workflow Integration

---

## 6.1 Integration Point

Insert after:
```text
Layer1 routing
```

before:
```text
Phase 2 reasoning
```

---

### Required Flow

```python
analysis = preprocessor.analyse(query)

predicates = extract_predicates(...)

predicates = [attach_negation(p) for p in predicates]

context.predicate_store = PredicateStore(predicates)
```

---

## 6.2 SSE Event Additions

Required events:

```text
graph_predicate_extract
graph_predicate_negate
graph_predicate_relation
graph_evidence_start
graph_evidence_done
graph_stabilization_update
```

---

### REQUIRED HARDENING

ALL events MUST include:

```text
sequence_number
event_id
timestamp
graph_schema_version
```

---

# 7. DSTFusion Integration

DSTFusion now operates over:
```text
structured semantic evidence
```

NOT:
```text
raw retrieval text
```

---

### Initial Fusion Input

```python
fusion_input = {
    "expert_outputs": existing_outputs,
    "evidence_bundles": [...],
    "predicate_graph": predicate_relations,
}
```

---

### REQUIRED HARDENING

Fusion MUST preserve:
- contradiction trace
- predicate lineage
- semantic revision history

---

# 8. Dependency Additions

```text
nltk>=3.8
wikipedia>=1.4.0
```

---

### Runtime Warmup

```python
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
```

---

### IMPORTANT

Warmup MUST happen:
```text
once at startup
```

NOT:
```text
per request
```

---

# 9. Tests

---

## 9.1 Unit Tests — Negator

Cases:
- factual inversion
- comparative antonym
- existential inversion
- normative rejection
- presupposition attack

---

## 9.2 Unit Tests — Provenance

Cases:
- lineage preservation
- semantic revision increment
- transformation history append
- derived_from integrity

---

## 9.3 Unit Tests — Predicate Relations

Cases:
- relation insertion
- relation traversal
- contradiction lookup
- support lookup

---

## 9.4 Integration Tests — Evidence

Cases:
- universal halt on counterexample
- replay-safe event ordering
- bounded retrieval
- EvidenceBundle lineage

---

## 9.5 Replay Tests

NEW HARDENED TEST CATEGORY.

Ensure:
```text
same input
→ same PredicateFrames
→ same negation
→ same event ordering
→ same replay reconstruction
```

---

# 10. Milestones

---

## Milestone A — Semantic IR

Deliver:
- PredicateFrame
- PredicateRelation
- PredicateStore

DONE when:
- semantic objects exist independently of retrieval

---

## Milestone B — Deterministic Negation

Deliver:
- negator
- provenance
- revision tracking

DONE when:
- all falsifiable predicates produce replay-safe negations

---

## Milestone C — Evidence Layer

Deliver:
- EvidenceFinder
- CalculatorTool
- WikipediaRetrieverTool

DONE when:
- structured EvidenceBundles populate DSTFusion

---

## Milestone D — Semantic Stabilization Hooks

Deliver:
- contradiction trace
- stabilization notes
- replay lineage

DONE when:
- predicates survive into stabilization/replay systems

---

# 11. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| WordNet weak antonyms | awkward negation | fallback structural inversion |
| predicate misclassification | wrong retrieval strategy | classifier fallback |
| retrieval explosion | latency spikes | bounded evidence budget |
| replay nondeterminism | broken semantic archaeology | sequence_number ordering |
| provenance drift | semantic debugging impossible | mandatory lineage tracking |
| normalization instability | contradictory graph states | semantic revisioning |
| contradiction storms | graph explosion | bounded stabilization propagation |

---

# 12. Explicitly Forbidden (v0.2)

The following are forbidden:

- generative negation
- unrestricted theorem proving
- autonomous ontology rewriting
- infinite retrieval loops
- recursive unbounded predicate decomposition
- LLM-generated contradiction synthesis

---

# 13. First PR Scope

First PR MUST remain intentionally small.

Allowed:
- predicate types
- provenance tracking
- extractor
- negator
- PredicateStore
- unit tests
- SSE extraction events

NOT allowed yet:
- retrieval tools
- DSTFusion integration
- stabilization propagation
- predicate theorem proving

Goal:
```text
visibility + deterministic semantic IR correctness
```

---

# 14. Final Architectural Principle

Before:
```text
Mycelium reasoned over text
```

Now:
```text
Mycelium reasons over formalized semantic claims
```

This transitions Mycelium from:
```text
LLM orchestration framework
```

into:
```text
semantic reasoning runtime
```
