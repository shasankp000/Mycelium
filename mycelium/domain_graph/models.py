from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from mycelium.domain_graph.state import DomainState, DomainMode, GateState, DriftType


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class DriftProfile:
    drift_type: DriftType
    detected_at: str = field(default_factory=_now)
    magnitude: float = 0.0
    resolved: bool = False
    resolution_note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drift_type": self.drift_type.value,
            "detected_at": self.detected_at,
            "magnitude": self.magnitude,
            "resolved": self.resolved,
            "resolution_note": self.resolution_note,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DriftProfile":
        return cls(
            drift_type=DriftType(d["drift_type"]),
            detected_at=d.get("detected_at", _now()),
            magnitude=d.get("magnitude", 0.0),
            resolved=d.get("resolved", False),
            resolution_note=d.get("resolution_note"),
        )


@dataclass
class DomainNode:
    domain_id: str
    label: str
    state: DomainState = DomainState.CREATING
    mode: DomainMode = DomainMode.BOOTSTRAP
    gate: GateState = GateState.CLOSED
    domain_version: int = 1
    head_version: int = 0
    query_count: int = 0
    last_active: Optional[str] = None
    created_at: str = field(default_factory=_now)
    tags: List[str] = field(default_factory=list)
    drift_history: List[DriftProfile] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def mark_active(self) -> None:
        self.last_active = _now()
        self.query_count += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain_id": self.domain_id,
            "label": self.label,
            "state": self.state.value,
            "mode": self.mode.value,
            "gate": self.gate.value,
            "domain_version": self.domain_version,
            "head_version": self.head_version,
            "query_count": self.query_count,
            "last_active": self.last_active,
            "created_at": self.created_at,
            "tags": self.tags,
            "drift_history": [d.to_dict() for d in self.drift_history],
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DomainNode":
        return cls(
            domain_id=d["domain_id"],
            label=d["label"],
            state=DomainState(d.get("state", "CREATING")),
            mode=DomainMode(d.get("mode", "BOOTSTRAP")),
            gate=GateState(d.get("gate", "CLOSED")),
            domain_version=d.get("domain_version", 1),
            head_version=d.get("head_version", 0),
            query_count=d.get("query_count", 0),
            last_active=d.get("last_active"),
            created_at=d.get("created_at", _now()),
            tags=d.get("tags", []),
            drift_history=[DriftProfile.from_dict(x) for x in d.get("drift_history", [])],
            meta=d.get("meta", {}),
        )


@dataclass
class DomainEdge:
    source_id: str
    target_id: str
    relation: str          # e.g. "extends", "overlaps", "conflicts", "patch_parent"
    weight: float = 1.0
    created_at: str = field(default_factory=_now)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation": self.relation,
            "weight": self.weight,
            "created_at": self.created_at,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DomainEdge":
        return cls(
            source_id=d["source_id"],
            target_id=d["target_id"],
            relation=d["relation"],
            weight=d.get("weight", 1.0),
            created_at=d.get("created_at", _now()),
            meta=d.get("meta", {}),
        )
