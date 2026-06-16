"""
mycelium/trm_v2/types.py

Foundational data types for TRM v2.
All enums, dataclasses, and type aliases used across the TRM v2 subsystem
are defined here. No other trm_v2 module imports from outside this file
(except stdlib + numpy for embeddings).

Spec refs:
  - mycelium_trm_v2_theoretical_spec_v0.2.md  (DomainState, DomainMode,
    GateState, DriftProfile, DomainNode, DomainGraph)
  - mycelium_trm_v2_theoretical_spec_hardening_additions.md  (Additions A-J)
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class DomainState(str, Enum):
    """Full lifecycle state machine for a domain node.

    Allowed transitions (hard rule: no domain is ever hard-deleted):
      CREATING  -> HOT | FAILED
      HOT       -> WARM | DEPRECATED
      WARM      -> COLD | HOT
      COLD      -> REMEMBERING
      REMEMBERING -> WARM | HOT
      HOT / WARM / COLD -> DEPRECATED
      DEPRECATED -> ARCHIVED
    """
    CREATING    = "CREATING"
    HOT         = "HOT"
    WARM        = "WARM"
    COLD        = "COLD"
    REMEMBERING = "REMEMBERING"
    DEPRECATED  = "DEPRECATED"
    ARCHIVED    = "ARCHIVED"
    FAILED      = "FAILED"


class DomainMode(str, Enum):
    """Whether a domain has a trained head or is retrieval-only.

    Promotion policy  : RETRIEVAL_ONLY -> FULL_DOMAIN
                        (triggered by persistent usage + reasoning failures)
    Demotion policy   : FULL_DOMAIN -> RETRIEVAL_ONLY
                        (triggered by long inactivity; head is archived)
    """
    RETRIEVAL_ONLY = "RETRIEVAL_ONLY"
    FULL_DOMAIN    = "FULL_DOMAIN"


class GateState(str, Enum):
    """Operational state of the gating network.

    Recovery lifecycle:
      BOOTSTRAP -> EXPANSION -> STABLE
      STABLE -> THAWING -> RECALIBRATING -> STABLE

    Rule: only *local* recalibration is permitted during THAWING/RECALIBRATING.
    Global retraining is forbidden.
    """
    BOOTSTRAP     = "BOOTSTRAP"
    EXPANSION     = "EXPANSION"
    STABLE        = "STABLE"
    THAWING       = "THAWING"
    RECALIBRATING = "RECALIBRATING"


# ---------------------------------------------------------------------------
# Drift profile
# ---------------------------------------------------------------------------

@dataclass
class DriftProfile:
    """Five independent drift dimensions for a domain.

    Rule: drift responses MUST be localised.
    e.g. retrieval_drift=HIGH + routing_drift=LOW must NOT trigger
    gate recalibration — only retrieval remediation is warranted.
    """
    semantic_drift:    float = 0.0   # change in domain meaning
    retrieval_drift:   float = 0.0   # change in retrieval quality
    routing_drift:     float = 0.0   # change in gate routing behaviour
    confidence_drift:  float = 0.0   # calibration degradation
    activation_drift:  float = 0.0   # usage-pattern shift


# ---------------------------------------------------------------------------
# Domain node
# ---------------------------------------------------------------------------

@dataclass
class DomainNode:
    """A single node in the DomainGraph.

    Versioning fields (all persisted artifacts must carry domain_version):
      domain_version, schema_version, head_version, retrieval_version

    Provenance fields (every domain must answer "Why was I created?"
    without external metadata):
      parent_domains, derived_from, creation_reason, created_by,
      creation_timestamp

    Routing / storage refs:
      head_ref          – path or identifier for the domain head checkpoint
      cold_storage_ref  – path for the cold-storage bundle
      sqlite_shard_ref  – path for the per-domain SQLite shard
    """

    # ---- Identity ----------------------------------------------------------
    domain_id:   str = field(default_factory=lambda: str(uuid.uuid4()))
    name:        str = ""
    description: str = ""

    # ---- Versioning --------------------------------------------------------
    domain_version:    str = "0.1"
    schema_version:    str = "1.0"
    head_version:      str = "0"
    retrieval_version: str = "0"

    # ---- State -------------------------------------------------------------
    state: DomainState = DomainState.CREATING
    mode:  DomainMode  = DomainMode.RETRIEVAL_ONLY

    # ---- Semantic identity -------------------------------------------------
    # Stored as a plain list so the dataclass stays JSON-serialisable.
    # Callers convert to/from np.ndarray as needed.
    semantic_centroid:  List[float] = field(default_factory=list)
    spectral_signature: List[float] = field(default_factory=list)

    # ---- Storage refs ------------------------------------------------------
    head_ref:          Optional[str] = None
    cold_storage_ref:  Optional[str] = None
    sqlite_shard_ref:  Optional[str] = None

    # ---- Graph topology ----------------------------------------------------
    neighbors: List[str] = field(default_factory=list)  # domain_id list

    # ---- Provenance --------------------------------------------------------
    parent_domains:     List[str] = field(default_factory=list)
    derived_from:       List[str] = field(default_factory=list)
    creation_reason:    str = ""
    created_by:         str = ""
    creation_timestamp: float = field(default_factory=time.time)

    # ---- Runtime stats -----------------------------------------------------
    activation_stats: Dict[str, float] = field(default_factory=dict)
    drift_profile:    DriftProfile = field(default_factory=DriftProfile)

    # ---- Lineage (free-form audit trail) -----------------------------------
    lineage: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Domain graph
# ---------------------------------------------------------------------------

@dataclass
class DomainGraph:
    """Registry and memory topology for all domains.

    Every snapshot must carry graph_version to support:
      migration, replay, graph comparison, topology archaeology.

    Hard rule: domains are NEVER hard-deleted.
    Retirement path: DEPRECATED -> ARCHIVED only.
    """

    graph_version:      str = "0.1"
    schema_version:     str = "1.0"
    creation_timestamp: float = field(default_factory=time.time)
    last_stabilized:    float = 0.0

    # node map: domain_id -> DomainNode
    nodes: Dict[str, DomainNode] = field(default_factory=dict)

    # edge map: domain_id -> list of (neighbour_id, edge_type, weight)
    edges: Dict[str, List[tuple]] = field(default_factory=dict)

    # ---- Mutation helpers --------------------------------------------------

    def add_node(self, node: DomainNode) -> None:
        """Register a new domain node. Raises if domain_id already exists."""
        if node.domain_id in self.nodes:
            raise ValueError(
                f"DomainGraph.add_node: domain_id '{node.domain_id}' already registered."
            )
        self.nodes[node.domain_id] = node
        self.edges.setdefault(node.domain_id, [])

    def get_node(self, domain_id: str) -> Optional[DomainNode]:
        return self.nodes.get(domain_id)

    def add_edge(
        self,
        src_id: str,
        dst_id: str,
        edge_type: str,
        weight: float = 1.0,
    ) -> None:
        """Add a typed directed edge between two domain nodes."""
        if src_id not in self.nodes:
            raise KeyError(f"DomainGraph.add_edge: source '{src_id}' not in graph.")
        if dst_id not in self.nodes:
            raise KeyError(f"DomainGraph.add_edge: destination '{dst_id}' not in graph.")
        self.edges.setdefault(src_id, []).append((dst_id, edge_type, weight))

    def transition_state(self, domain_id: str, new_state: DomainState) -> None:
        """Apply a lifecycle state transition to a domain node.

        Enforces the allowed transition table. Raises ValueError on illegal
        transitions so callers cannot silently corrupt the state machine.
        """
        _ALLOWED: Dict[DomainState, List[DomainState]] = {
            DomainState.CREATING:    [DomainState.HOT, DomainState.FAILED],
            DomainState.HOT:         [DomainState.WARM, DomainState.DEPRECATED],
            DomainState.WARM:        [DomainState.COLD, DomainState.HOT, DomainState.DEPRECATED],
            DomainState.COLD:        [DomainState.REMEMBERING, DomainState.DEPRECATED],
            DomainState.REMEMBERING: [DomainState.WARM, DomainState.HOT],
            DomainState.DEPRECATED:  [DomainState.ARCHIVED],
            DomainState.ARCHIVED:    [],   # terminal – no transitions out
            DomainState.FAILED:      [],   # terminal – no transitions out
        }
        node = self.nodes.get(domain_id)
        if node is None:
            raise KeyError(f"DomainGraph.transition_state: '{domain_id}' not found.")
        allowed = _ALLOWED.get(node.state, [])
        if new_state not in allowed:
            raise ValueError(
                f"Illegal state transition for domain '{domain_id}': "
                f"{node.state.value} -> {new_state.value}. "
                f"Allowed: {[s.value for s in allowed]}"
            )
        node.state = new_state

    def all_active(self) -> List[DomainNode]:
        """Return all nodes that are not DEPRECATED, ARCHIVED, or FAILED."""
        inactive = {DomainState.DEPRECATED, DomainState.ARCHIVED, DomainState.FAILED}
        return [n for n in self.nodes.values() if n.state not in inactive]


# ---------------------------------------------------------------------------
# Observability events
# ---------------------------------------------------------------------------

@dataclass
class DomainEvent:
    """Structured lifecycle event emitted by TRM v2 subsystems.

    Required by the frontend observability spec (Addition H).
    Every event must carry: sequence_number, event_id, timestamp,
    graph_schema_version, domain_version.
    """
    event_type:           str        # one of the graph_domain_* constants below
    sequence_number:      int
    event_id:             str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp:            float = field(default_factory=time.time)
    graph_schema_version: str = "1.0"
    domain_version:       str = "0.1"
    domain_id:            Optional[str] = None
    payload:              Dict       = field(default_factory=dict)


# Event type constants (use these instead of bare strings)
EVT_DOMAIN_CREATE        = "graph_domain_create"
EVT_DOMAIN_ACTIVATE      = "graph_domain_activate"
EVT_DOMAIN_COLDSTORE     = "graph_domain_coldstore"
EVT_DOMAIN_REMEMBER      = "graph_domain_remember"
EVT_DOMAIN_PROMOTE       = "graph_domain_promote"
EVT_DOMAIN_DEPRECATE     = "graph_domain_deprecate"
EVT_GATE_RECALIBRATION   = "graph_gate_recalibration"
EVT_DRIFT_DETECTED       = "graph_drift_detected"
