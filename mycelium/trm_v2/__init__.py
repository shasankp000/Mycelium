"""
mycelium/trm_v2
===============
TRM v2 — modular, non-destructively expandable Tiny Recursive Model substrate.

Import order for downstream modules:
  1. types      — enums, dataclasses, DomainGraph, DomainEvent
  2. domain_shard  — SQLite-backed per-domain knowledge store  (Phase 1)
  3. encoder       — SharedEncoder + DomainHead                (Phase 2)
  4. gate          — GatingNetwork                             (Phase 2)
  5. cold_storage  — Hot/Warm/Cold lifecycle manager           (Phase 2)
  6. pipeline      — TRMV2Pipeline wiring                      (Phase 2)
"""

from mycelium.trm_v2.types import (
    DomainState,
    DomainMode,
    GateState,
    DriftProfile,
    DomainNode,
    DomainGraph,
    DomainEvent,
    EVT_DOMAIN_CREATE,
    EVT_DOMAIN_ACTIVATE,
    EVT_DOMAIN_COLDSTORE,
    EVT_DOMAIN_REMEMBER,
    EVT_DOMAIN_PROMOTE,
    EVT_DOMAIN_DEPRECATE,
    EVT_GATE_RECALIBRATION,
    EVT_DRIFT_DETECTED,
)

__all__ = [
    "DomainState",
    "DomainMode",
    "GateState",
    "DriftProfile",
    "DomainNode",
    "DomainGraph",
    "DomainEvent",
    "EVT_DOMAIN_CREATE",
    "EVT_DOMAIN_ACTIVATE",
    "EVT_DOMAIN_COLDSTORE",
    "EVT_DOMAIN_REMEMBER",
    "EVT_DOMAIN_PROMOTE",
    "EVT_DOMAIN_DEPRECATE",
    "EVT_GATE_RECALIBRATION",
    "EVT_DRIFT_DETECTED",
]
