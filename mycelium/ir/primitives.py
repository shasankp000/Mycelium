"""
Phase A — Core IR Primitives.

These dataclasses form the semantic substrate for all reasoning phases.
Design follows the Mycelium Semantic Architecture Consolidation Notes
(sections 25-29) with strict alignment to:
  - Section 27: ConfidenceState — net_confidence uses support/contradiction
    decomposition, NOT scalar subtraction of penalty from overall.
  - Section 28: TemporalState — six temporal types, historical_validity bool.
  - Section 29: ProvenanceChain — worker_threads list for multi-worker support
    (Phase F), even though Phase D uses single-worker DFS.

All objects are:
  - JSON-serialisable via dataclasses.asdict()
  - Versioned (carry ontology_version / canonicalization_version fields)
  - Deterministically hashable (see serialization.compute_semantic_hash)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SemanticSignature:
    """Represents the semantic identity of a node.

    Fields
    ------
    semantic_hash : str
        SHA-256 of canonical_form + predicate_family + abstraction_level.
        Computed by serialization.compute_semantic_hash after Phase B
        canonicalization — never set manually.
    embedding_signature : list
        384-dim MiniLM vector (reuses the cached singleton from existing code).
    spectral_signature : list
        Spectral profile from Lens 2 (vocabulary / syntactic axes).
        Extended with epistemic-polarity axes in Phase C.
    predicate_family : str
        One of the PREDICATE_FAMILY_MAP keys defined in
        mycelium.canonicalization.predicate_families.  Examples:
        CAUSAL | CORRELATIONAL | TEMPORAL | DEFINITIONAL | …
    abstraction_level : int
        0 = most concrete (leaf node), higher = more abstract.
        Used in canonical_form generation (depth{N} suffix).
    canonical_form : str
        Stable normalised string produced by
        mycelium.canonicalization.canonical_form.generate_canonical_form.
        Format: FAMILY::subject::predicate::object::depth{N}
    equivalence_family : list[str]
        Semantically equivalent canonical forms that map to the same
        semantic_hash.  Populated by Phase B canonicalization.
    """

    semantic_hash: str
    embedding_signature: list
    spectral_signature: list
    predicate_family: str
    abstraction_level: int
    canonical_form: str
    equivalence_family: list = field(default_factory=list)


@dataclass
class ConfidenceState:
    """Represents multi-dimensional uncertainty for a node or edge.

    The five confidence axes map to fundamentally different uncertainty
    sources (Consolidation Notes §27).  They must never be collapsed into
    a single float before explicit fusion.

    net_confidence() implements the initial contradiction mathematics from
    §27:

        net_confidence = support_score − contradiction_score

    where:
        support_score    = overall_confidence (weighted evidence support)
        contradiction_score = contradiction_penalty (accumulated penalty)

    This will be upgraded to Dempster-Shafer Theory in Phase F
    (dst_fusion.py), which can represent genuine UNKNOWN states without
    forcing a probability.
    """

    overall_confidence: float
    semantic_confidence: float   = 0.0
    structural_confidence: float = 0.0
    epistemic_confidence: float  = 0.0
    evidence_confidence: float   = 0.0
    temporal_confidence: float   = 0.0
    contradiction_penalty: float = 0.0
    aggregation_method: str      = "weighted_average"
    confidence_sources: list     = field(default_factory=list)

    def net_confidence(self) -> float:
        """support_score − contradiction_score, clamped to [0, 1].

        Per §27 of the consolidation notes, this is the Phase A/C
        implementation.  Phase F replaces this with DST fusion which
        produces an explicit unknown mass rather than forcing a
        probability.
        """
        support_score = self.overall_confidence
        contradiction_score = self.contradiction_penalty
        return max(0.0, min(1.0, support_score - contradiction_score))


@dataclass
class TemporalState:
    """Represents temporal semantics for a claim or relation.

    Consolidation Notes §28.  Static timestamps are insufficient;
    the system must support all six temporal types.

    Supported types
    ---------------
    EXACT             — precise timestamp known
    INTERVAL          — start and end both known
    APPROXIMATE       — fuzzy, uncertainty > 0
    RELATIVE          — BEFORE | AFTER | DURING relationship
    HISTORICAL_ESTIMATE — inferred from historical context
    UNKNOWN           — temporal information unavailable

    historical_validity tracks whether a claim is still true at query time
    (e.g., PlutoIsPlanet was True historically, False after reclassification).
    This is required for reasoning cache integrity and contradiction analysis.
    """

    type: str  # EXACT | INTERVAL | APPROXIMATE | RELATIVE | HISTORICAL_ESTIMATE | UNKNOWN
    start: Optional[str]             = None
    end: Optional[str]               = None
    relative_relation: Optional[str] = None   # BEFORE | AFTER | DURING
    uncertainty: float               = 1.0
    historical_validity: bool        = True


@dataclass
class ProvenanceChain:
    """Tracks full reasoning lineage for auditability.

    Consolidation Notes §29.  Without lineage, hallucination propagation
    becomes opaque and reasoning is non-auditable.

    worker_threads is populated in Phase F (multi-worker architecture).
    In Phase D (single-worker DFS) it will contain a single entry.
    """

    sources: list                      # source URLs / tool IDs
    reasoning_paths: list              # ordered list of layer decisions
    decomposition_origin: str          # which Layer 2 template was used
    evidence_nodes: list               # EvidenceItem IDs from Layer 4
    ontology_resolution_path: list     # sequence of ontology alignments
    worker_threads: list               # which reasoning workers contributed
    timestamp: str                     # ISO 8601


@dataclass
class GraphFingerprint:
    """Canonical identity of a reasoning graph.

    Consolidation Notes §35 / §16.  Enables:
      - graph deduplication,
      - cache reuse,
      - TRM stabilisation,
      - contradiction lineage tracking,
      - graph inheritance detection.

    canonicalization_version is bumped when Phase B logic changes.
    Existing graphs retain their original version; a migration pass
    re-canonicalises and creates new revisions (Phase D immutability).
    """

    semantic_hash: str
    structural_hash: str
    predicate_family_hash: str
    temporal_signature: str
    ontology_signature: str
    canonicalization_version: str = "v1.0"
