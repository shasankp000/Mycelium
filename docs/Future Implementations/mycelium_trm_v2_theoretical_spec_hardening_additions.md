# TRM v2 Hardened Specification Additions (v0.2)

> These sections are intended to be merged into the existing TRM v2 Theoretical Specification.
>
> They do NOT replace the original document.
>
> They extend it with:
>
> - lifecycle formalization
> - provenance
> - replayability
> - graph versioning
> - retrieval-only domain support
> - gate recovery mechanisms
> - frontend observability compatibility
> - alignment with Predicate Engine v0.2

---

# ADDITION A
# Domain Lifecycle State Machine

## Motivation

The original specification defines:

```text
Hot
Warm
Cold
Remembering
```

states.

However future domain evolution requires:

- deprecation
- archival
- failed creation handling
- replay compatibility

Therefore domain lifecycle becomes a formal state machine.

---

## DomainState

```python
class DomainState(str, Enum):
    CREATING = "CREATING"
    HOT = "HOT"
    WARM = "WARM"
    COLD = "COLD"
    REMEMBERING = "REMEMBERING"
    DEPRECATED = "DEPRECATED"
    ARCHIVED = "ARCHIVED"
    FAILED = "FAILED"
```

---

## Lifecycle

```text
CREATING
    ↓
HOT
    ↓
WARM
    ↓
COLD
    ↓
REMEMBERING
    ↓
HOT
```

Additional transitions:

```text
HOT
 ↓
DEPRECATED
 ↓
ARCHIVED
```

and

```text
CREATING
 ↓
FAILED
```

---

## Design Rule

Domains MUST NEVER be hard-deleted.

Domain retirement occurs through:

```text
DEPRECATED
→ ARCHIVED
```

This preserves:

- replay
- provenance
- semantic archaeology
- patch lineage

---

# ADDITION B
# Domain Versioning

## Motivation

A domain is not a static entity.

Over time:

- retrieval evolves
- calibration evolves
- topology evolves
- storage evolves

Replay and debugging require explicit versioning.

---

## DomainVersion Fields

Add to DomainNode:

```python
domain_version: str

schema_version: str

head_version: str

retrieval_version: str
```

---

## Example

```text
Physics
├── domain_version = 1.3
├── head_version = 4
└── retrieval_version = 8
```

---

## Rule

All persisted artifacts MUST carry:

```text
domain_version
```

including:

- snapshots
- cold-storage bundles
- replay archives
- graph exports

---

# ADDITION C
# Domain Provenance

## Motivation

The Predicate Engine now supports lineage tracking.

DomainGraph should support equivalent ancestry tracking.

---

## Add To DomainNode

```python
parent_domains: List[str]

derived_from: List[str]

creation_reason: str

created_by: str

creation_timestamp: float
```

---

## Example

```text
Physics
 ↓
Quantum Physics
 ↓
Quantum Error Correction
```

All ancestry remains recoverable.

---

## Design Rule

Domain creation MUST be explainable.

Every domain must answer:

```text
Why was I created?
```

without external metadata.

---

# ADDITION D
# Structured Drift Profile

## Motivation

Current specification references:

```text
drift_profile
```

but does not define it.

Different forms of drift must remain distinguishable.

---

## DriftProfile

```python
@dataclass
class DriftProfile:

    semantic_drift: float

    retrieval_drift: float

    routing_drift: float

    confidence_drift: float

    activation_drift: float
```

---

## Interpretation

### semantic_drift

Change in domain meaning.

### retrieval_drift

Change in retrieval quality.

### routing_drift

Change in gate behavior.

### confidence_drift

Calibration degradation.

### activation_drift

Usage-pattern shift.

---

## Rule

Drift responses must be localized.

Example:

```text
retrieval_drift high
routing_drift low
```

MUST NOT trigger gate recalibration.

---

# ADDITION E
# Retrieval-Only Domains

## Motivation

Not every domain requires a dedicated head.

Some domains are primarily retrieval problems.

---

## DomainMode

```python
class DomainMode(str, Enum):

    RETRIEVAL_ONLY = "RETRIEVAL_ONLY"

    FULL_DOMAIN = "FULL_DOMAIN"
```

---

## Retrieval-Only Characteristics

Contains:

```text
SQLite shard
metadata
domain graph node
```

Does NOT contain:

```text
specialized head
```

---

## Promotion Policy

```text
RETRIEVAL_ONLY
      ↓
Persistent usage
      ↓
Reasoning failures
      ↓
Need for calibration
      ↓
FULL_DOMAIN
```

---

## Demotion Policy

```text
FULL_DOMAIN
      ↓
Long inactivity
      ↓
Head archived
      ↓
RETRIEVAL_ONLY
```

---

## Benefit

Prevents:

```text
domain-head explosion
```

while preserving:

```text
knowledge accessibility
```

---

# ADDITION F
# Gate Recovery Protocol

## Motivation

Frozen gates prevent forgetting.

Frozen forever prevents adaptation.

A recovery mechanism is required.

---

## GateState

```python
class GateState(str, Enum):

    BOOTSTRAP = "BOOTSTRAP"

    EXPANSION = "EXPANSION"

    STABLE = "STABLE"

    THAWING = "THAWING"

    RECALIBRATING = "RECALIBRATING"
```

---

## Recovery Lifecycle

```text
STABLE
   ↓
Routing instability detected
   ↓
THAWING
   ↓
RECALIBRATING
   ↓
STABLE
```

---

## Trigger Conditions

Possible triggers:

- routing confusion spike
- repeated misrouting
- domain overlap increase
- confidence collapse

---

## Rule

Only local recalibration is allowed.

Global retraining remains forbidden.

---

# ADDITION G
# DomainGraph Versioning

## Motivation

Domain topology evolves.

Replay requires graph-level versioning.

---

## Add To DomainGraph

```python
graph_version: str

schema_version: str

creation_timestamp: float

last_stabilized: float
```

---

## Rule

Every snapshot MUST contain:

```text
graph_version
```

This supports:

- migration
- replay
- graph comparison
- topology archaeology

---

# ADDITION H
# Domain Observability Events

## Motivation

Frontend redesign now includes:

- graph replay
- graph snapshots
- semantic observability

TRM v2 must expose domain lifecycle events.

---

## Required Events

```text
graph_domain_create

graph_domain_activate

graph_domain_coldstore

graph_domain_remember

graph_domain_promote

graph_domain_deprecate

graph_gate_recalibration

graph_drift_detected
```

---

## Required Metadata

Every event MUST contain:

```text
sequence_number

event_id

timestamp

graph_schema_version

domain_version
```

---

## Purpose

Supports:

- frontend graph visualization
- replay
- semantic archaeology
- domain lifecycle debugging

---

# ADDITION I
# Reasoning Responsibility Clarification

## Motivation

Predicate Engine v0.2 fundamentally changes where reasoning occurs.

---

## Previous Interpretation

```text
Domain Head
    ↓
Reasoning Memory
```

---

## Revised Interpretation

### SQLite Shards

Responsible for:

```text
knowledge memory
```

Examples:

- retrieval corpus
- embeddings
- FTS indexes
- metadata
- provenance
- domain artifacts

---

### Domain Heads

Responsible for:

```text
routing calibration
confidence calibration
retrieval policy hints
local adaptation parameters
```

NOT:

```text
knowledge storage
```

NOT:

```text
primary reasoning
```

---

### Predicate Engine

Responsible for:

```text
semantic representation
```

---

### TRM

Responsible for:

```text
reasoning
```

including:

- domain selection
- evidence synthesis
- contradiction handling
- stabilization
- cross-domain coordination

---

## Final Separation

```text
SQLite Shards
=
Knowledge Memory

Predicate Engine
=
Semantic Representation

TRM
=
Reasoning

Domain Heads
=
Calibration + Routing
```

---

# ADDITION J
# Future Compatibility With Predicate Engine

TRM v2 SHALL treat PredicateFrames as the canonical reasoning substrate.

Domain heads MUST NOT operate directly on raw text when PredicateFrames are available.

Preferred flow:

```text
Query
 ↓
Predicate Engine
 ↓
PredicateFrames
 ↓
TRM
 ↓
Domain Selection
 ↓
Retrieval
 ↓
DSTFusion
 ↓
Stabilization
```

This guarantees compatibility with:

- contradiction propagation
- semantic replay
- provenance tracking
- graph reasoning
- ontology governance
- leverage propagation

and aligns TRM v2 with the broader Mycelium semantic runtime architecture.
