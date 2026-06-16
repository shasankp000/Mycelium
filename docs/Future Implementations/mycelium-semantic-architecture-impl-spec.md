# Mycelium — Semantic Architecture Implementation Specification

**Version:** 1.0  
**Date:** May 2026  
**Branch:** `web-ui-prototype`  
**Author:** Shasank Prasad  
**Based on:** Mycelium Semantic Architecture Consolidation Notes + Reasoning Architecture Layers 1–6 Spec

---

## Overview

This specification defines the implementation plan for layering the full semantic reasoning architecture onto the existing `web-ui-prototype` branch. The current branch implements the **Unified Expert Pre-Check Layer** (OOD detection, K-Medoids calibration, SVM/BERT-based experts, and the async calibration job pattern). Everything described in this document sits *above and around* that foundation — it does not replace it.

The architecture being specified here transforms Mycelium from an expert routing system into a **semantic reasoning substrate with epistemic stabilization**. The work is organized in strict dependency order: later phases depend on earlier ones being stable.

---

## Current Branch Baseline

Before reading this spec, the following are already present in `web-ui-prototype`:

| Component | Status |
|---|---|
| `UnifiedExpertSystem` with OOD 3-method ensemble | ✅ Implemented |
| K-Medoids clustering + calibration (CalibratedClassifierCV) | ✅ Implemented |
| Expert centroid persistence (`.pkl` files) | ✅ Implemented |
| `CREATE_NEW_PATCH` / `USE_EXPERT` routing signals | ✅ Partially wired |
| Async calibration job (`/api/calibrate/start` + poll) | ✅ Implemented (per previous session) |
| SentenceTransformer lazy-cache fix (single instance) | ✅ Implemented (per previous session) |
| `TRANSFORMERS_OFFLINE=1` boot-time env var | ✅ Implemented (per previous session) |
| Layers 1–2 (Contradiction Detector, Claim Decomposer) | ✅ Implemented per reasoning arch plan |
| Layers 3–6 (Consequence, Evidence, Evaluator, Synthesizer) | ✅ Implemented per reasoning arch plan |
| Web UI prototype (FastAPI + frontend) | ✅ In progress |

**What is NOT yet implemented:** everything in this document.

---

## Architecture Overview

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│  Phase A: Formal IR Layer                   │  ← NEW
│  Node / Edge / Graph / SemanticSignature    │
│  ConfidenceState / TemporalState /          │
│  ProvenanceChain / GraphFingerprint         │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│  Phase B: Canonicalization Pipeline         │  ← NEW
│  SRL + relation extraction                 │
│  MultiLensRouter arbitration               │
│  Ontology alignment → canonical form        │
│  Semantic hash generation                   │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│  Phase C: MultiLensRouter Evolution         │  ← UPGRADE existing router
│  Lens 1: Semantic Similarity (existing)    │
│  Lens 2: Spectral + Epistemic Polarity     │  ← Upgrade
│  Lens 3: Confidence Fusion (existing)      │
│  Lens 4: Relation-Family Routing           │  ← NEW
│  Lens 5: Ontology Grounding                │  ← NEW
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│  Existing Expert / Layers 0–6 Pipeline      │  ← UNCHANGED
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│  Phase D: DAG Reasoning + TRM               │  ← NEW
│  DFS decomposition with cycle detection     │
│  Graph lifecycle states                     │
│  TRM Stabilization (semantic convergence)   │
│  Competing graph coexistence                │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│  Phase E: Contradiction Ontology            │  ← NEW
│  7-class contradiction classifier           │
│  ContradictionEdge / LeverageEdge           │
│  Leverage propagation + decay               │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│  Phase F: Ontology Governance               │  ← NEW
│  Dynamic ontology formation                 │
│  Domain-based decay rates                   │
│  Adversarial robustness layer               │
└─────────────────────────────────────────────┘
```

---

## Phase A — Formal Intermediate Representation (IR)

**Goal:** Establish the stable internal semantic substrate that all later phases operate upon. Nothing in Phases B–F works correctly without a stable IR.

**Priority:** Mandatory first. No other phase starts until IR is stable and tested.

### A.1 — Core Primitives

Create `mycelium/ir/primitives.py`:

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class SemanticSignature:
    semantic_hash: str
    embedding_signature: list       # 384-dim MiniLM vector (reuse existing)
    spectral_signature: list        # from Lens 2
    predicate_family: str           # CAUSAL | CORRELATIONAL | TEMPORAL | etc.
    abstraction_level: int          # 0 = most concrete, higher = more abstract
    canonical_form: str             # normalized predicate string
    equivalence_family: list        # list of semantically equivalent forms

@dataclass
class ConfidenceState:
    overall_confidence: float
    semantic_confidence: float      = 0.0
    structural_confidence: float    = 0.0
    epistemic_confidence: float     = 0.0
    evidence_confidence: float      = 0.0
    temporal_confidence: float      = 0.0
    contradiction_penalty: float    = 0.0
    aggregation_method: str         = "weighted_average"
    confidence_sources: list        = field(default_factory=list)

    # Phase A initial math — upgrade to Dempster-Shafer in Phase F
    def net_confidence(self) -> float:
        return max(0.0, self.overall_confidence - self.contradiction_penalty)

@dataclass
class TemporalState:
    type: str           # EXACT | INTERVAL | APPROXIMATE | RELATIVE |
                        # HISTORICAL_ESTIMATE | UNKNOWN
    start: Optional[str]            = None
    end: Optional[str]              = None
    relative_relation: Optional[str] = None   # BEFORE | AFTER | DURING
    uncertainty: float              = 1.0
    historical_validity: bool       = True

@dataclass
class ProvenanceChain:
    sources: list                   # source URLs / tool IDs
    reasoning_paths: list           # ordered list of layer decisions
    decomposition_origin: str       # which Layer 2 template was used
    evidence_nodes: list            # EvidenceItem IDs from Layer 4
    ontology_resolution_path: list  # sequence of ontology alignments
    worker_threads: list            # which reasoning workers contributed
    timestamp: str                  # ISO 8601

@dataclass
class GraphFingerprint:
    semantic_hash: str
    structural_hash: str
    predicate_family_hash: str
    temporal_signature: str
    ontology_signature: str
    canonicalization_version: str   = "v1.0"
```

### A.2 — Node, Edge, Graph

```python
# mycelium/ir/graph.py

NODE_TYPES = [
    "ENTITY", "CLAIM", "PREDICATE", "RELATION", "EVIDENCE",
    "HYPOTHESIS", "TEMPORAL_RELATION", "CONTRADICTION",
    "ABSTRACTION", "REASONING_MOTIF"
]

RELATION_FAMILIES = [
    "SUPPORTS", "CONTRADICTS", "CAUSES", "CORRELATES_WITH",
    "IMPLIES", "PRECEDES", "PART_OF", "SUBTYPE_OF",
    "INSTANCE_OF", "DEPENDS_ON", "DERIVED_FROM"
]

GRAPH_STATES = [
    "DRAFT", "CANDIDATE", "STABILIZED",
    "CANONICAL", "CONTESTED", "DEPRECATED", "ARCHIVED"
]

@dataclass
class IRNode:
    id: str
    type: str                       # from NODE_TYPES
    label: str
    semantic_signature: SemanticSignature
    confidence_state: ConfidenceState
    temporal_state: TemporalState
    provenance: ProvenanceChain
    metadata: dict                  = field(default_factory=dict)
    ontology_version: str           = "v1.0"
    state: str                      = "DRAFT"

@dataclass
class IREdge:
    id: str
    source: str                     # IRNode.id
    target: str                     # IRNode.id
    relation_type: str              # from RELATION_FAMILIES
    confidence_state: ConfidenceState
    temporal_state: TemporalState
    weight: float                   = 1.0
    metadata: dict                  = field(default_factory=dict)

@dataclass
class IRGraph:
    graph_id: str
    nodes: list                     # list[IRNode]
    edges: list                     # list[IREdge]
    fingerprint: GraphFingerprint
    state: str                      = "DRAFT"
    version: str                    = "v1"
    confidence_state: ConfidenceState = field(default_factory=lambda: ConfidenceState(0.0))
    ontology_version: str           = "v1.0"
    created_at: str                 = ""
    updated_at: str                 = ""
    parent_graph_id: Optional[str]  = None    # for revision trees (Phase D)
```

### A.3 — LeverageEdge

```python
# mycelium/ir/leverage.py

@dataclass
class LeverageEdge:
    source: str                     # IRNode.id
    target: str                     # IRNode.id
    direct_leverage: float
    transitive_leverage: float
    stability: float
    decay_rate: float
    dependency_depth: int
    verification_state: str         # UNVERIFIED | VERIFIED | CONTESTED | SEVERED

    def effective_leverage(self) -> float:
        """Prevents cyclic confidence amplification."""
        damping = 0.85
        return self.direct_leverage * (damping ** self.dependency_depth)
```

### A.4 — ContradictionEdge

```python
# mycelium/ir/contradiction.py

CONTRADICTION_CLASSES = [
    "DIRECT_CONTRADICTION",
    "PARTIAL_CONTRADICTION",
    "CONTEXTUAL_CONTRADICTION",
    "TEMPORAL_CONTRADICTION",
    "PROBABILISTIC_DISAGREEMENT",
    "TRADEOFF_RELATION",
    "NON_CONTRADICTORY_DIVERGENCE"
]

@dataclass
class ContradictionEdge:
    type: str                       # from CONTRADICTION_CLASSES
    severity: float                 # 0–1
    confidence: float               # confidence in the contradiction classification
    scope: str                      # LOCAL | DOMAIN | GLOBAL
    temporal_validity: TemporalState
    source_graph_id: str
    target_graph_id: str
```

### A.5 — IR Serialization

All IR objects must be:

- JSON-serializable via `dataclasses.asdict()`
- versioned (carry `ontology_version` and `canonicalization_version`)
- deterministically hashable: given the same inputs, the same `semantic_hash` must be produced

Implement `mycelium/ir/serialization.py` with:
- `ir_to_json(graph: IRGraph) -> str`
- `ir_from_json(data: str) -> IRGraph`
- `compute_semantic_hash(node: IRNode) -> str` — SHA-256 of canonical form + predicate family + abstraction level

### A.6 — Tests (Phase A)

- Given identical inputs → same `semantic_hash` produced (determinism check)
- `ConfidenceState.net_confidence()` never goes below 0
- `LeverageEdge.effective_leverage()` decreases monotonically with depth
- JSON round-trip: `ir_from_json(ir_to_json(g)) == g`

---

## Phase B — Canonicalization Pipeline

**Goal:** Build the deterministic normalization pipeline that transforms raw Layer 2 decomposition output into stable IR nodes and edges.

**Dependency:** Phase A complete.

### B.1 — Pipeline Structure

```
Raw Layer 2 Output
    ↓
B.1 Normalization (lowercase, punctuation, unicode normalization)
    ↓
B.2 SRL + Relation Extraction
    ↓
B.3 Predicate Family Classification
    ↓
B.4 Ontology Alignment (v1.0: rule-based; v2.0: learned)
    ↓
B.5 Canonical Form Generation
    ↓
B.6 Semantic Hash Generation   ← AFTER canonicalization, BEFORE TRM
    ↓
B.7 IR Node/Edge Construction
    ↓
B.8 TRM Persistence (Phase D)
```

### B.2 — Semantic Role Labeling (SRL)

**Why NER alone is insufficient:** NER extracts entities but not roles. `John gave Mary a book` requires `giver=John`, `receiver=Mary`, `object=book`, `relation=TRANSFER` to be semantically correct.

Implementation in `mycelium/canonicalization/srl.py`:

```python
import spacy

nlp = None  # lazy-load, same pattern as SentenceTransformer fix

def get_nlp():
    global nlp
    if nlp is None:
        nlp = spacy.load("en_core_web_lg")
    return nlp

def extract_semantic_roles(text: str) -> dict:
    """
    Returns:
        {
            "entities": list[dict],      # {text, label, role}
            "relations": list[dict],     # {subject, predicate, object, family}
            "events": list[dict],        # {trigger, args}
            "dependency_tree": list      # head/dep structure
        }
    """
    doc = get_nlp()(text)
    entities = _extract_entities(doc)
    relations = _extract_relations(doc)
    events = _extract_events(doc)
    return {
        "entities": entities,
        "relations": relations,
        "events": events,
        "dependency_tree": _dep_tree(doc)
    }
```

**Note:** Phase B uses spaCy `en_core_web_lg` which is available in the existing venv. Full SRL (AllenNLP-style) can be added in Phase F hardening.

### B.3 — Predicate Family Classification

```python
# mycelium/canonicalization/predicate_families.py

PREDICATE_FAMILY_MAP = {
    "CAUSAL": ["causes", "leads_to", "produces", "results_in", "triggers"],
    "CORRELATIONAL": ["correlates_with", "associated_with", "linked_to"],
    "RISK": ["increases_risk", "reduces_risk", "predicts"],
    "TEMPORAL": ["precedes", "follows", "during", "before", "after"],
    "DEFINITIONAL": ["is_a", "subtype_of", "instance_of", "defined_as"],
    "CONTRADICTORY": ["contradicts", "refutes", "disproves"],
    "SUPPORTIVE": ["supports", "confirms", "validates"],
    "COMPARATIVE": ["outperforms", "exceeds", "is_less_than"]
}

def classify_predicate_family(predicate_str: str) -> str:
    """Rule-based v1.0. Upgrade to embedding-based in Phase F."""
    normalized = predicate_str.lower().replace(" ", "_")
    for family, members in PREDICATE_FAMILY_MAP.items():
        if normalized in members:
            return family
    return "UNKNOWN"
```

**Critical rule:** `CAUSES` ≠ `INCREASES_RISK`. These are separate predicate families. Canonicalization must preserve this distinction and never collapse them.

### B.4 — Canonical Form Generator

```python
# mycelium/canonicalization/canonical_form.py

def generate_canonical_form(
    subject: str,
    predicate_family: str,
    predicate_str: str,
    obj: str,
    abstraction_level: int = 0
) -> str:
    """
    Produces a stable, normalized string used as the basis for semantic_hash.
    Format: FAMILY::subject::normalized_predicate::object::depth{N}
    Example: CAUSAL::smoking::causes::cancer::depth0
    """
    subj_norm = subject.lower().strip()
    obj_norm = obj.lower().strip()
    pred_norm = predicate_str.lower().replace(" ", "_").strip()
    return f"{predicate_family}::{subj_norm}::{pred_norm}::{obj_norm}::depth{abstraction_level}"
```

### B.5 — Canonicalization Versioning

All canonical forms carry a `canonicalization_version` string (e.g., `"v1.0"`). When the canonicalization logic is upgraded:

1. Bump the version string
2. Existing graphs retain their original `canonicalization_version`
3. A migration pass re-canonicalizes old graphs and creates new graph revisions (never mutating originals — see Phase D immutability rules)

### B.6 — Tests (Phase B)

- `Orbits(Earth, Sun)` and `MovesAround(Earth, Sun)` and `OrbitalMotion(Earth, Sun)` → same `semantic_hash` after canonicalization
- `CAUSES(Smoking, Cancer)` and `INCREASES_RISK(Smoking, Cancer)` → **different** `semantic_hash` (regression test — must never collapse)
- `canonical_form` is identical for two structurally different but semantically equivalent inputs

---

## Phase C — MultiLensRouter Evolution

**Goal:** Upgrade the existing `MultiLensRouter` from a 3-lens expert selector into a 5-lens semantic arbitration system.

**Dependency:** Phase B complete (Lens 4 needs canonical predicate families from B.3).

### C.1 — Existing Lenses (Unchanged Interface)

| Lens | Current Function | Change |
|---|---|---|
| Lens 1: Semantic Similarity | SentenceTransformer cosine routing | None — now uses cached singleton from previous fix |
| Lens 2: Spectral Structural Analysis | Vocabulary/syntactic profiles | Upgrade (see C.2) |
| Lens 3: Confidence Fusion | Weighted score aggregation | Upgrade (see C.3) |

### C.2 — Lens 2 Upgrade: Epistemic Polarity

Add spectral axes for epistemic signal detection. These are lightweight — regex + token frequency distributions, no model inference.

```python
# mycelium/router/lens2_spectral.py

CERTAINTY_MARKERS = [
    "definitively", "proves", "undeniably", "unquestionably",
    "it is certain", "established fact"
]
SPECULATION_MARKERS = [
    "suggests", "may", "could", "possibly", "evidence weakly",
    "appears to", "preliminary"
]
MANIPULATION_MARKERS = [
    "everyone knows", "obvious that", "only a fool",
    "undeniable truth", "mainstream lies"
]

def compute_epistemic_polarity(text: str) -> dict:
    """
    Returns {
        certainty_score: float,
        speculation_score: float,
        manipulation_score: float,
        propaganda_score: float,
        probabilistic_density: float
    }
    Used by Layer 0 classification and Lens 3 confidence weighting.
    """
```

This feeds directly into `ConfidenceState.epistemic_confidence` on the IR node.

### C.3 — Lens 3 Upgrade: Structured Confidence Fusion

Replace the scalar confidence output with a full `ConfidenceState` object:

```python
def fuse_confidence(
    semantic_score: float,
    spectral_score: float,
    epistemic_polarity: dict,
    evidence_confidence: float,
    temporal_confidence: float
) -> ConfidenceState:
    """
    Phase C: weighted averaging.
    Phase F upgrade: Dempster-Shafer Theory fusion.
    """
    weights = {
        "semantic": 0.35,
        "spectral": 0.20,
        "epistemic": 0.15,
        "evidence": 0.20,
        "temporal": 0.10
    }
    overall = (
        weights["semantic"] * semantic_score +
        weights["spectral"] * spectral_score +
        weights["epistemic"] * (1.0 - epistemic_polarity["manipulation_score"]) +
        weights["evidence"] * evidence_confidence +
        weights["temporal"] * temporal_confidence
    )
    return ConfidenceState(
        overall_confidence=overall,
        semantic_confidence=semantic_score,
        structural_confidence=spectral_score,
        epistemic_confidence=1.0 - epistemic_polarity["manipulation_score"],
        evidence_confidence=evidence_confidence,
        temporal_confidence=temporal_confidence
    )
```

### C.4 — Lens 4 (NEW): Relation-Family Routing

Route not just domain (physics, medical) but also *relation type* — because different relation families require different inference rules.

```python
# mycelium/router/lens4_relation_routing.py

RELATION_INFERENCE_RULES = {
    "CAUSAL": "causal_reasoning_pipeline",
    "TEMPORAL": "temporal_ordering_pipeline",
    "PROBABILISTIC": "bayesian_inference_pipeline",
    "DEFINITIONAL": "ontology_lookup_pipeline",
    "CONTRADICTORY": "contradiction_resolution_pipeline",
}

def route_by_relation_family(
    canonical_form: str,
    predicate_family: str
) -> str:
    """Returns the pipeline identifier that should handle this relation type."""
    return RELATION_INFERENCE_RULES.get(predicate_family, "general_reasoning_pipeline")
```

### C.5 — Lens 5 (NEW): Ontology Grounding

```python
# mycelium/router/lens5_ontology.py

ONTOLOGY_HIERARCHY = {
    "virus": ["pathogen", "microorganism", "biological_entity"],
    "string_theory": ["theoretical_physics", "physics", "science"],
    "smoking": ["behavior", "substance_use", "health_risk_factor"],
    # ... extended via Phase F dynamic ontology
}

def resolve_ontology_path(entity: str) -> list:
    """
    Returns abstraction path from entity to root concept.
    Phase C: static dict lookup.
    Phase F upgrade: dynamic emergent ontology from reasoning workers.
    """
    entity_norm = entity.lower().strip()
    return ONTOLOGY_HIERARCHY.get(entity_norm, [entity_norm])
```

### C.6 — Updated Router Output

The router now returns a `RouterResult` IR object instead of a plain dict:

```python
@dataclass
class RouterResult:
    domain: str
    relation_family: str            # NEW — from Lens 4
    ontology_path: list             # NEW — from Lens 5
    confidence_state: ConfidenceState  # UPGRADED — full structured confidence
    epistemic_polarity: dict        # NEW — from Lens 2 upgrade
    pipeline_id: str                # which reasoning pipeline to invoke
    routing_trace: dict             # full lens scores for audit
```

### C.7 — Tests (Phase C)

- `"definitively proves"` → high `certainty_score`, low `speculation_score`
- `"evidence weakly suggests"` → high `speculation_score`, penalized `epistemic_confidence`
- `CAUSES` relation → routes to `causal_reasoning_pipeline`
- `TEMPORAL` relation → routes to `temporal_ordering_pipeline`
- Router output passes Phase A IR serialization round-trip

---

## Phase D — DAG Reasoning, Graph Lifecycle, and TRM

**Goal:** Build the graph lifecycle system, DFS/DAG decomposition, graph revision trees, and the TRM semantic convergence layer.

**Dependency:** Phases A, B, C complete.

### D.1 — DFS + DAG Decomposition

```python
# mycelium/reasoning/dag_decomposer.py

class DAGDecomposer:
    """
    DFS-style recursive claim decomposition, constrained by DAG structure.
    Prevents: infinite cycles, redundant expansion, semantic duplication.
    """

    def __init__(self, max_depth: int = 3, info_gain_threshold: float = 0.15):
        self.max_depth = max_depth
        self.info_gain_threshold = info_gain_threshold
        self._visited: set = set()      # cycle detection

    def decompose(self, claim_node: IRNode, current_depth: int = 0) -> IRGraph:
        if current_depth >= self.max_depth:
            return self._leaf_graph(claim_node)

        semantic_hash = claim_node.semantic_signature.semantic_hash
        if semantic_hash in self._visited:
            return self._existing_node_ref(claim_node)  # DAG reuse, no duplication
        self._visited.add(semantic_hash)

        if not self._should_expand(claim_node):
            return self._leaf_graph(claim_node)

        sub_claims = self._generate_sub_claims(claim_node)
        sub_graphs = [
            self.decompose(sc, current_depth + 1)
            for sc in sub_claims
        ]
        return self._merge_sub_graphs(claim_node, sub_graphs)

    def _should_expand(self, node: IRNode) -> bool:
        """Expand only if information gain is above threshold."""
        # Phase D v1: heuristic based on confidence uncertainty
        uncertainty = 1.0 - node.confidence_state.overall_confidence
        return uncertainty > self.info_gain_threshold
```

**Key property:** When two decomposition paths share a subgraph (e.g., `Smoking → DNA damage` and `Radiation → DNA damage`), the `DNA damage` node is *reused* (same `semantic_hash`), not duplicated. This is the DAG reuse guarantee.

### D.2 — Graph Lifecycle Management

```python
# mycelium/reasoning/graph_lifecycle.py

class GraphStore:
    """
    Append-only immutable graph store.
    Graphs are never mutated — new revisions are created instead.
    """

    def __init__(self, persistence_path: str):
        self._graphs: dict[str, list[IRGraph]] = {}  # graph_id → revision list
        self._path = persistence_path

    def add_revision(self, graph: IRGraph) -> IRGraph:
        """
        Creates a new revision. Never mutates existing graphs.
        Version naming: G123-v1 → G123-v2, G123-v2B for branches.
        """
        base_id = graph.graph_id.split("-v")[0]
        existing = self._graphs.get(base_id, [])
        new_version = f"v{len(existing) + 1}"
        new_graph = replace(graph, version=new_version, updated_at=_now())
        self._graphs.setdefault(base_id, []).append(new_graph)
        self._persist(new_graph)
        return new_graph

    def get_latest(self, graph_id: str) -> Optional[IRGraph]:
        revisions = self._graphs.get(graph_id, [])
        return revisions[-1] if revisions else None

    def get_revision_tree(self, graph_id: str) -> list[IRGraph]:
        return self._graphs.get(graph_id, [])

    def transition_state(self, graph_id: str, new_state: str) -> IRGraph:
        """Creates a new revision with updated state. Never mutates in place."""
        latest = self.get_latest(graph_id)
        return self.add_revision(replace(latest, state=new_state))
```

### D.3 — TRM Stabilization Layer

TRM (Temporal Reasoning Module / semantic convergence layer) answers: *"How has this meaning historically stabilized?"*

```python
# mycelium/reasoning/trm.py

class TRMStabilizer:
    """
    Semantic convergence through repeated independent reasoning.
    Promotes graphs from DRAFT → CANDIDATE → STABILIZED → CANONICAL.
    """

    PROMOTION_THRESHOLDS = {
        "DRAFT":       1,   # created once
        "CANDIDATE":   3,   # seen independently 3 times
        "STABILIZED": 10,   # reached convergence
        "CANONICAL":  25    # heavily reinforced + evidenced
    }

    def __init__(self, graph_store: GraphStore):
        self._store = graph_store
        self._observation_counts: dict[str, int] = {}

    def observe(self, graph: IRGraph) -> IRGraph:
        """
        Called every time a reasoning worker produces a graph.
        Uses GraphFingerprint.semantic_hash to detect re-derivation.
        """
        fprint = graph.fingerprint.semantic_hash
        self._observation_counts[fprint] = self._observation_counts.get(fprint, 0) + 1
        count = self._observation_counts[fprint]

        new_state = graph.state
        for state, threshold in sorted(
            self.PROMOTION_THRESHOLDS.items(), key=lambda x: x[1], reverse=True
        ):
            if count >= threshold:
                new_state = state
                break

        if new_state != graph.state:
            return self._store.transition_state(graph.graph_id, new_state)
        return graph

    def arbitrate(self, graph_a: IRGraph, graph_b: IRGraph) -> str:
        """
        Determines relationship between two competing graphs.
        Returns: MERGE_IF_SIMILAR | COEXIST_IF_COMPETING |
                 DEPRECATE_IF_DOMINATED | ESCALATE_IF_UNCERTAIN
        """
        similarity = self._semantic_similarity(graph_a, graph_b)
        if similarity > 0.92:
            return "MERGE_IF_SIMILAR"
        elif similarity > 0.60:
            return "COEXIST_IF_COMPETING"
        elif graph_a.confidence_state.overall_confidence - \
             graph_b.confidence_state.overall_confidence > 0.40:
            return "DEPRECATE_IF_DOMINATED"
        else:
            return "ESCALATE_IF_UNCERTAIN"
```

**Decomposition Attractors:** When TRM promotes a graph to `CANONICAL`, it flags the associated reasoning path as a *reusable motif* — future decompositions that reach the same subgraph short-circuit to the canonical form rather than re-deriving it.

### D.4 — Semantic Branching

The `IRGraph.parent_graph_id` field (defined in Phase A) enables branch tracking:

```python
def branch_graph(parent: IRGraph, new_claim: IRNode) -> IRGraph:
    """
    Creates a new competing hypothesis branch from an existing graph.
    The branch inherits parent context but diverges from the new claim onward.
    """
    branch_id = f"{parent.graph_id}-branch-{uuid4().hex[:6]}"
    return IRGraph(
        graph_id=branch_id,
        nodes=[new_claim],
        edges=[],
        fingerprint=compute_fingerprint([new_claim]),
        state="DRAFT",
        parent_graph_id=parent.graph_id
    )
```

### D.5 — Tests (Phase D)

- DFS decomposer halts on cycle: `Economy ↔ Politics` loop doesn't infinite-recurse
- Same subgraph is reused (not duplicated): `DNA damage` node has same `semantic_hash` whether reached from `smoking` or `radiation` path
- `GraphStore.add_revision()` never mutates existing revision objects
- TRM promotes `DRAFT → CANDIDATE` after 3 independent observations of same `semantic_hash`
- TRM promotes to `CANONICAL` after 25 observations
- `COEXIST_IF_COMPETING` correctly assigned when `Coffee improves cognition` vs `Coffee increases anxiety`

---

## Phase E — Contradiction Ontology and Leverage Propagation

**Goal:** Implement the 7-class contradiction classifier, leverage propagation system, and contradiction-aware confidence decay.

**Dependency:** Phases A, B, C, D complete.

### E.1 — Contradiction Classifier

```python
# mycelium/contradiction/classifier.py

class ContradictionClassifier:
    """
    Classifies the semantic relationship between two conflicting claims.
    Not all disagreement is contradiction.
    """

    def classify(
        self,
        claim_a: IRNode,
        claim_b: IRNode,
        context: dict
    ) -> ContradictionEdge:
        """
        Pipeline:
        1. Semantic analysis (embedding distance)
        2. Predicate family comparison
        3. Temporal scope analysis
        4. Ontology overlap analysis
        5. Classification
        """
        sem_distance = self._semantic_distance(claim_a, claim_b)
        pred_match = claim_a.semantic_signature.predicate_family == \
                     claim_b.semantic_signature.predicate_family
        temporal_overlap = self._temporal_overlap(
            claim_a.temporal_state, claim_b.temporal_state
        )
        ontology_conflict = self._ontology_conflict(claim_a, claim_b)

        contradiction_type = self._classify_type(
            sem_distance, pred_match, temporal_overlap, ontology_conflict
        )

        return ContradictionEdge(
            type=contradiction_type,
            severity=self._severity(sem_distance, pred_match),
            confidence=self._classification_confidence(
                sem_distance, temporal_overlap
            ),
            scope="LOCAL" if sem_distance < 0.3 else "DOMAIN",
            temporal_validity=self._combined_temporal(
                claim_a.temporal_state, claim_b.temporal_state
            ),
            source_graph_id=claim_a.id,
            target_graph_id=claim_b.id
        )

    def _classify_type(
        self, sem_dist, pred_match, temporal_overlap, ontology_conflict
    ) -> str:
        if pred_match and sem_dist < 0.15 and temporal_overlap:
            return "DIRECT_CONTRADICTION"
        if pred_match and sem_dist < 0.30:
            return "PARTIAL_CONTRADICTION"
        if not temporal_overlap:
            return "TEMPORAL_CONTRADICTION"
        if not ontology_conflict and sem_dist > 0.50:
            return "NON_CONTRADICTORY_DIVERGENCE"
        if sem_dist < 0.40:
            return "CONTEXTUAL_CONTRADICTION"
        # Coffee improves focus + Coffee increases anxiety pattern
        return "TRADEOFF_RELATION"
```

**Critical regression test:** `Coffee improves focus` vs `Coffee increases anxiety` must classify as `TRADEOFF_RELATION`, never `DIRECT_CONTRADICTION`.

### E.2 — Leverage Propagation

```python
# mycelium/contradiction/leverage.py

class LeveragePropagator:
    """
    Manages epistemic leverage chains: A supports B supports C.
    When A is contradicted, B weakens, C destabilizes progressively.
    """

    def propagate_contradiction(
        self,
        contested_graph_id: str,
        graph_store: GraphStore,
        leverage_edges: list[LeverageEdge]
    ) -> dict:
        """
        Returns { graph_id: new_confidence_state } for all affected graphs.
        Does NOT immediately deprecate — enters CONTESTED state first.
        Actual deprecation only after TRM verification or TTL expiry.
        """
        affected = {}
        queue = [e for e in leverage_edges if e.source == contested_graph_id]

        while queue:
            edge = queue.pop(0)
            target = graph_store.get_latest(edge.target)
            if target is None:
                continue

            # Gradual decay, not instant invalidation
            decay = edge.effective_leverage() * edge.decay_rate
            new_confidence = max(
                0.0,
                target.confidence_state.overall_confidence - decay
            )
            new_conf_state = replace(
                target.confidence_state,
                overall_confidence=new_confidence,
                contradiction_penalty=target.confidence_state.contradiction_penalty + decay
            )
            affected[edge.target] = new_conf_state

            # Continue propagation if confidence dropped significantly
            if decay > 0.05:
                next_edges = [
                    e for e in leverage_edges if e.source == edge.target
                ]
                queue.extend(next_edges)

        return affected

    def verify_contradiction(
        self, contested_graph_id: str, evidence: list
    ) -> str:
        """
        Returns: CONFIRMED | DISPROVEN | TIMEOUT
        CONFIRMED → DEPRECATED + leverage severed
        DISPROVEN → exit CONTESTED, restore confidence
        TIMEOUT   → auto-invalidate + recursive re-evaluation
        """
```

### E.3 — Tests (Phase E)

- `DIRECT_CONTRADICTION` requires same predicate family + low semantic distance + temporal overlap (all three conditions)
- `Coffee improves focus` vs `Coffee increases anxiety` → `TRADEOFF_RELATION` (regression test)
- Leverage decay is monotonically decreasing with `dependency_depth`
- Contradiction propagation halts when decay drops below 0.01 (prevents infinite propagation)
- `CONTESTED` state does not immediately drop to `DEPRECATED` — requires `verify_contradiction()` call

---

## Phase F — Ontology Governance and Adversarial Robustness

**Goal:** Build the dynamic ontology formation system, domain-based decay rates, and adversarial input detection.

**Dependency:** Phases A–E complete.

### F.1 — Dynamic Ontology Formation

Replace Phase C's static `ONTOLOGY_HIERARCHY` dict with an emergent system:

```python
# mycelium/ontology/governor.py

@dataclass
class OntologyRelation:
    subtype: str
    supertype: str
    confidence: float
    stabilized: bool
    decay_rate: float           # domain-specific
    last_verified: str          # ISO 8601 timestamp

class OntologyGovernor:
    """
    Ontology emerges from reasoning, not from hardcoded labels.
    Formation pipeline:
        facts → reasoning workers → DAG decomposition
        → MultiLensRouter arbitration → TRM stabilization
        → ontology convergence → semantic hierarchy formation
    """

    DOMAIN_DECAY_RATES = {
        "physics":          0.002,   # slow
        "mathematics":      0.001,   # very slow
        "medicine":         0.010,   # medium
        "social_trends":    0.050,   # fast
        "politics":         0.040,   # fast
        "historical_facts": 0.0005,  # very slow
    }

    def propose_relation(
        self, subtype: str, supertype: str,
        evidence: list[IRNode], domain: str
    ) -> OntologyRelation:
        """
        Proposed relations are DRAFT until TRM stabilizes them.
        Example: Virus ⊂ Pathogen emerges from biological definitions.
        """
        confidence = self._compute_proposal_confidence(evidence)
        decay_rate = self.DOMAIN_DECAY_RATES.get(domain, 0.020)
        return OntologyRelation(
            subtype=subtype,
            supertype=supertype,
            confidence=confidence,
            stabilized=False,
            decay_rate=decay_rate,
            last_verified=_now()
        )

    def decay_pass(self) -> list[OntologyRelation]:
        """
        Periodic decay pass. Relations that drop below threshold
        require re-reasoning with fresh evidence to re-stabilize.
        Prevents semantic fossilization.
        """
```

### F.2 — Adversarial Robustness

```python
# mycelium/ontology/adversarial.py

@dataclass
class AuthorTrustProfile:
    author_id: str
    trust_score: float          # starts at 1.0, decays on false factual claims
    fact_history: list[dict]    # {claim, verified, timestamp}
    flagged: bool               = False
    leverage_ceiling: float     = 1.0

class AdversarialGuard:
    """
    Prevents ontology poisoning, confidence laundering,
    and coordinated semantic manipulation.
    """

    def verify_factual_input(
        self, claim: IRNode, author: AuthorTrustProfile,
        ground_truth_result: dict
    ) -> AuthorTrustProfile:
        if not ground_truth_result["verified"]:
            # Trust decays on false factual claims
            new_trust = author.trust_score * 0.90
            updated_author = replace(author, trust_score=new_trust)

            if new_trust < 0.50:
                # Silent pattern analysis begins
                return self._analyze_poisoning_pattern(updated_author)

            return updated_author
        return author

    def _analyze_poisoning_pattern(
        self, author: AuthorTrustProfile
    ) -> AuthorTrustProfile:
        """
        Analyzes historical factual claims for coordinated poisoning.
        If poisoning pattern detected:
        - author flagged
        - factual inputs down-weighted
        - leverage ceiling reduced
        """
        false_rate = sum(
            1 for f in author.fact_history if not f["verified"]
        ) / max(len(author.fact_history), 1)

        if false_rate > 0.40:
            return replace(
                author,
                flagged=True,
                leverage_ceiling=0.20
            )
        return author
```

### F.3 — Dempster-Shafer Confidence Upgrade

Upgrade `ConfidenceState` aggregation from Phase C's weighted average to Dempster-Shafer Theory (DST):

```python
# mycelium/reasoning/dst_fusion.py

def dst_fuse(belief_functions: list[dict]) -> ConfidenceState:
    """
    Dempster-Shafer fusion of multiple reasoning worker outputs.
    Preserves explicit unknown states — unlike Bayesian averaging,
    DST can represent "genuinely don't know" without forcing a probability.

    belief_functions: list of {positive: float, negative: float, unknown: float}
    where positive + negative + unknown == 1.0 per function.
    """
```

### F.4 — Full SRL Upgrade

Upgrade Phase B's spaCy-based SRL to AllenNLP-style structured SRL if VRAM budget allows. If not (single-GPU constraint), maintain spaCy but add explicit event extraction for common patterns.

### F.5 — Tests (Phase F)

- Ontology relation `Virus ⊂ Pathogen` stabilizes after sufficient evidence, not from hardcoded lookup
- `DOMAIN_DECAY_RATES["physics"] < DOMAIN_DECAY_RATES["politics"]`
- Author trust decays correctly on repeated false claims
- Author with `false_rate > 0.40` gets `leverage_ceiling = 0.20`
- DST fusion produces explicit `UNKNOWN` state when evidence genuinely conflicts, not forced probability

---

## Integration Architecture

### How New Phases Connect to Existing Layers 0–6

```
Existing Layer 0 (Question Classification)
    → feeds RouterResult domain + epistemic_polarity (Phase C.2 Lens 2 output)

Existing Layer 1 (Contradiction Detector)
    → output is now an IRNode of type CONTRADICTION (Phase A)
    → ContradictionEdge created by Phase E classifier

Existing Layer 2 (Claim Decomposer)
    → output passed to Phase B canonicalization before IR node construction
    → DFS decomposer (Phase D) wraps Layer 2 for recursive expansion

Existing Layer 3 (Consequence Generator)
    → SearchPlan now carries IRNode references for provenance tracking

Existing Layer 4 (Evidence Grounding)
    → EvidenceItems stored as IRNode type=EVIDENCE
    → ProvenanceChain.evidence_nodes populated here

Existing Layer 5 (Hypothesis Evaluator)
    → ConfidenceState from Phase A replaces scalar float output
    → LeverageEdges created here for SUPPORTS/CONTRADICTS relations

Existing Layer 6 (Reasoning Synthesizer)
    → Reads from GraphStore (Phase D) for TRM-stabilized canonical graphs
    → pipelinetrace dict now includes IRGraph.fingerprint
```

### `run_reasoning_pipeline()` Updated Signature

```python
async def run_reasoning_pipeline(
    user_query: str,
    precheck_result: dict,
    expert_metadata: dict,
    domain: str,
    graph_store: GraphStore,         # NEW — Phase D
    trm: TRMStabilizer,              # NEW — Phase D
    ontology_governor: OntologyGovernor  # NEW — Phase F
) -> FinalResponse:
```

---

## Performance Targets

All targets measured on existing hardware. Phases A–C add negligible latency. Phases D–E add DAG traversal overhead. Phase F adds periodic background work.

| Phase | Component | Target Latency | Notes |
|---|---|---|---|
| A | IR construction | < 5ms | Pure Python dataclass ops |
| B | Canonicalization pipeline | < 20ms | spaCy already loaded |
| C | Full 5-lens router | < 50ms | Lens 1 dominant; Lens 4/5 are dict lookups |
| D | DAG decomposition (depth 3) | < 100ms | Bounded by `max_depth` and `info_gain_threshold` |
| D | TRM `observe()` | < 10ms | Dict lookup + state transition |
| E | Contradiction classifier | < 30ms | Embedding distance + rule-based |
| E | Leverage propagation | < 20ms | BFS over leverage graph |
| F | Ontology decay pass | background | Scheduled idle-time job |
| F | DST fusion | < 15ms | Per-query, replaces weighted avg |

---

## File Structure (New Files)

```
mycelium/
├── ir/
│   ├── __init__.py
│   ├── primitives.py          # Phase A — SemanticSignature, ConfidenceState,
│   │                          #           TemporalState, ProvenanceChain, GraphFingerprint
│   ├── graph.py               # Phase A — IRNode, IREdge, IRGraph
│   ├── leverage.py            # Phase A — LeverageEdge
│   ├── contradiction.py       # Phase A — ContradictionEdge
│   └── serialization.py       # Phase A — JSON round-trip, semantic_hash
│
├── canonicalization/
│   ├── __init__.py
│   ├── srl.py                 # Phase B — SRL + relation extraction
│   ├── predicate_families.py  # Phase B — family classification
│   └── canonical_form.py      # Phase B — canonical form generator
│
├── router/
│   ├── lens2_spectral.py      # Phase C — epistemic polarity upgrade
│   ├── lens3_confidence.py    # Phase C — structured ConfidenceState fusion
│   ├── lens4_relation.py      # Phase C NEW — relation-family routing
│   └── lens5_ontology.py      # Phase C NEW — ontology grounding
│
├── reasoning/
│   ├── dag_decomposer.py      # Phase D — DFS + DAG
│   ├── graph_lifecycle.py     # Phase D — GraphStore, immutable revisions
│   ├── trm.py                 # Phase D — TRM stabilization
│   └── dst_fusion.py          # Phase F — Dempster-Shafer upgrade
│
├── contradiction/
│   ├── classifier.py          # Phase E — 7-class classifier
│   └── leverage.py            # Phase E — propagation + decay
│
└── ontology/
    ├── governor.py            # Phase F — dynamic ontology + decay rates
    └── adversarial.py         # Phase F — trust profiles + poisoning detection
```

---

## Implementation Order and Gate Conditions

| Phase | Gate Condition to Proceed |
|---|---|
| **A — IR Layer** | All Phase A tests green; JSON round-trip deterministic |
| **B — Canonicalization** | `CAUSES ≠ INCREASES_RISK` regression test must pass before Phase C |
| **C — Router Upgrade** | Router returns `RouterResult` IR object; existing Layer 0–6 tests still pass |
| **D — DAG + TRM** | Cycle detection confirmed; GraphStore immutability confirmed; TRM promotion thresholds tested |
| **E — Contradiction** | `TRADEOFF_RELATION` regression test must pass; leverage decay is bounded |
| **F — Ontology + Adversarial** | Ontology decay rates confirmed; DST fusion replaces weighted average cleanly |

---

## Open Items Before Starting Phase A

1. **spaCy model size decision** — `en_core_web_sm` (12MB) vs `en_core_web_lg` (560MB). Phase B SRL quality depends heavily on this. Recommend `en_core_web_lg` for Phases B–D, with `sm` as fallback for low-memory environments.

2. **GraphStore persistence format** — JSON flat files per graph (simple, human-readable) vs SQLite (queryable, faster for large graph counts). Recommend JSON for Phases D–E, migrate to SQLite in Phase F if graph count exceeds ~10,000 nodes.

3. **TRM promotion thresholds** — The values (3 → CANDIDATE, 10 → STABILIZED, 25 → CANONICAL) are initial estimates. These need empirical calibration on Mycelium's actual query distribution once Phase D is running.

4. **Reasoning workers parallelism** — Section 31 of the consolidation notes proposes multiple independent reasoning workers. Phase D implements single-worker DFS. Multi-worker aggregation (with DST fusion) is a Phase F concern — do not add it earlier.
