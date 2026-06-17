"""
mycelium/trm_v2/drift.py

DriftMonitor — rolling drift estimation for TRM v2 domains.

Tracked drift dimensions (all local, never globally remediated):
    semantic_drift    : centroid shift in query latent space
    retrieval_drift   : degradation in retrieval hit-rate / chunk yield
    routing_drift     : instability in gate confidence over time
    confidence_drift  : mismatch between gate confidence and retrieval success
    activation_drift  : shift in domain activation frequency

Design principles:
    - All monitoring is per-domain and windowed.
    - Any remediation is local to the affected domain(s).
    - Global retraining is forbidden; monitor only recommends local actions.
    - Drift values are normalised to [0, 1].

Typical integration:
    monitor = DriftMonitor(graph, gate)
    ...
    result = pipeline.query(...)
    monitor.update_from_pipeline_result(result)
    actions = monitor.recommended_actions()
"""

from __future__ import annotations

import math
import statistics
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from mycelium.trm_v2.types import (
    DriftProfile,
    DomainGraph,
    EVT_DRIFT_DETECTED,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    sim = float(np.dot(a, b) / (na * nb))
    sim = max(-1.0, min(1.0, sim))
    return (1.0 - sim) / 2.0  # map cosine similarity [-1,1] -> distance [0,1]

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def _safe_mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0

def _normalised_std(xs: List[float], scale: float = 1.0) -> float:
    if len(xs) < 2:
        return 0.0
    return _clamp01(statistics.pstdev(xs) / max(scale, 1e-9))


# ---------------------------------------------------------------------------
# Domain-local rolling buffers
# ---------------------------------------------------------------------------

@dataclass
class _DomainDriftBuffer:
    latent_history: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=64))
    retrieval_yield_history: Deque[float] = field(default_factory=lambda: deque(maxlen=64))
    gate_score_history: Deque[float] = field(default_factory=lambda: deque(maxlen=64))
    confidence_gap_history: Deque[float] = field(default_factory=lambda: deque(maxlen=64))
    activation_timestamps: Deque[float] = field(default_factory=lambda: deque(maxlen=128))
    last_update: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Action recommendation
# ---------------------------------------------------------------------------

@dataclass
class DriftAction:
    domain_id: str
    drift_type: str
    severity: float
    recommended_action: str


# ---------------------------------------------------------------------------
# DriftMonitor
# ---------------------------------------------------------------------------

class DriftMonitor:
    """Tracks rolling drift for all domains in a DomainGraph.

    Parameters
    ----------
    graph : DomainGraph
    semantic_threshold   : float   event threshold for semantic drift
    retrieval_threshold  : float   event threshold for retrieval drift
    routing_threshold    : float   event threshold for routing drift
    confidence_threshold : float   event threshold for confidence drift
    activation_threshold : float   event threshold for activation drift
    activation_ref_rate  : float   expected activations / minute baseline
    """

    def __init__(
        self,
        graph: DomainGraph,
        semantic_threshold: float = 0.35,
        retrieval_threshold: float = 0.45,
        routing_threshold: float = 0.35,
        confidence_threshold: float = 0.40,
        activation_threshold: float = 0.50,
        activation_ref_rate: float = 2.0,
    ) -> None:
        self.graph = graph
        self.semantic_threshold = semantic_threshold
        self.retrieval_threshold = retrieval_threshold
        self.routing_threshold = routing_threshold
        self.confidence_threshold = confidence_threshold
        self.activation_threshold = activation_threshold
        self.activation_ref_rate = activation_ref_rate

        self._buffers: Dict[str, _DomainDriftBuffer] = {}
        self._events: List[Dict] = []

    # ---- Internal ---------------------------------------------------------

    def _buf(self, domain_id: str) -> _DomainDriftBuffer:
        if domain_id not in self._buffers:
            self._buffers[domain_id] = _DomainDriftBuffer()
        return self._buffers[domain_id]

    def _update_profile(self, domain_id: str) -> DriftProfile:
        node = self.graph.get_node(domain_id)
        if node is None:
            return DriftProfile()

        buf = self._buf(domain_id)

        # 1) semantic_drift: distance between stored centroid and recent centroid
        semantic = 0.0
        if len(buf.latent_history) >= 3:
            recent_centroid = np.mean(np.stack(list(buf.latent_history)), axis=0)
            if node.semantic_centroid:
                stored = np.asarray(node.semantic_centroid, dtype=np.float32)
                semantic = _cosine_distance(stored, recent_centroid)
            else:
                node.semantic_centroid = recent_centroid.astype(np.float32).tolist()
                semantic = 0.0

        # 2) retrieval_drift: low chunk yield = high drift
        retrieval = 0.0
        if buf.retrieval_yield_history:
            avg_yield = _safe_mean(list(buf.retrieval_yield_history))
            retrieval = _clamp01(1.0 - avg_yield)

        # 3) routing_drift: instability in gate score
        routing = _normalised_std(list(buf.gate_score_history), scale=0.25)

        # 4) confidence_drift: confidence high but retrieval poor, or vice versa
        confidence = _safe_mean(list(buf.confidence_gap_history))
        confidence = _clamp01(confidence)

        # 5) activation_drift: compare current per-minute activation rate to baseline
        activation = 0.0
        now = time.time()
        recent = [t for t in buf.activation_timestamps if now - t <= 300.0]  # last 5 min
        if len(recent) >= 5:
            per_min = len(recent) / 5.0
            activation = _clamp01(abs(per_min - self.activation_ref_rate) / max(self.activation_ref_rate, 1e-9))

        profile = DriftProfile(
            semantic_drift=semantic,
            retrieval_drift=retrieval,
            routing_drift=routing,
            confidence_drift=confidence,
            activation_drift=activation,
        )
        node.drift_profile = profile
        return profile

    def _record_event(self, domain_id: str, drift_type: str, severity: float) -> None:
        self._events.append({
            "event_type": EVT_DRIFT_DETECTED,
            "domain_id": domain_id,
            "drift_type": drift_type,
            "severity": float(severity),
            "timestamp": time.time(),
        })

    # ---- Public update API ------------------------------------------------

    def update_domain(
        self,
        domain_id: str,
        latent: np.ndarray,
        gate_score: float,
        chunks_retrieved: int,
        retrieve_k: int,
        timestamp: Optional[float] = None,
    ) -> DriftProfile:
        """Update one domain from a single activation."""
        node = self.graph.get_node(domain_id)
        if node is None:
            raise KeyError(f"DriftMonitor.update_domain: unknown domain '{domain_id}'")

        ts = timestamp or time.time()
        buf = self._buf(domain_id)

        latent = np.asarray(latent, dtype=np.float32).reshape(-1)
        buf.latent_history.append(latent)
        buf.gate_score_history.append(float(gate_score))
        buf.activation_timestamps.append(ts)

        retrieval_yield = 0.0 if retrieve_k <= 0 else float(chunks_retrieved) / float(retrieve_k)
        buf.retrieval_yield_history.append(_clamp01(retrieval_yield))

        confidence_gap = abs(float(gate_score) - retrieval_yield)
        buf.confidence_gap_history.append(_clamp01(confidence_gap))
        buf.last_update = ts

        profile = self._update_profile(domain_id)

        if profile.semantic_drift >= self.semantic_threshold:
            self._record_event(domain_id, "semantic_drift", profile.semantic_drift)
        if profile.retrieval_drift >= self.retrieval_threshold:
            self._record_event(domain_id, "retrieval_drift", profile.retrieval_drift)
        if profile.routing_drift >= self.routing_threshold:
            self._record_event(domain_id, "routing_drift", profile.routing_drift)
        if profile.confidence_drift >= self.confidence_threshold:
            self._record_event(domain_id, "confidence_drift", profile.confidence_drift)
        if profile.activation_drift >= self.activation_threshold:
            self._record_event(domain_id, "activation_drift", profile.activation_drift)

        return profile

    def update_from_pipeline_result(self, result, retrieve_k: Optional[int] = None) -> Dict[str, DriftProfile]:
        """Update all activated domains from a PipelineResult.

        This avoids importing pipeline.py here and keeps the module decoupled.
        """
        rk = retrieve_k if retrieve_k is not None else max(
            [len(dr.chunks) for dr in result.domain_results] + [1]
        )
        latent = result.latent.detach().cpu().numpy().mean(axis=0)

        out: Dict[str, DriftProfile] = {}
        for dr in result.domain_results:
            out[dr.domain_id] = self.update_domain(
                domain_id=dr.domain_id,
                latent=latent,
                gate_score=float(dr.gate_score),
                chunks_retrieved=len(dr.chunks),
                retrieve_k=rk,
            )
        return out

    # ---- Recommendations --------------------------------------------------

    def recommended_actions(self) -> List[DriftAction]:
        """Return local remediation recommendations only."""
        actions: List[DriftAction] = []
        for domain_id, node in self.graph.nodes.items():
            p = node.drift_profile

            if p.semantic_drift >= self.semantic_threshold:
                actions.append(DriftAction(
                    domain_id, "semantic_drift", p.semantic_drift,
                    "refresh semantic centroid and run local replay"
                ))
            if p.retrieval_drift >= self.retrieval_threshold:
                actions.append(DriftAction(
                    domain_id, "retrieval_drift", p.retrieval_drift,
                    "reindex shard and inspect FTS/vector retrieval quality"
                ))
            if p.routing_drift >= self.routing_threshold:
                actions.append(DriftAction(
                    domain_id, "routing_drift", p.routing_drift,
                    "begin local gate recalibration for this domain only"
                ))
            if p.confidence_drift >= self.confidence_threshold:
                actions.append(DriftAction(
                    domain_id, "confidence_drift", p.confidence_drift,
                    "adjust gate temperature and compare confidence vs yield"
                ))
            if p.activation_drift >= self.activation_threshold:
                actions.append(DriftAction(
                    domain_id, "activation_drift", p.activation_drift,
                    "review lifecycle placement (HOT/WARM/COLD) for this domain"
                ))

        actions.sort(key=lambda a: a.severity, reverse=True)
        return actions

    # ---- Introspection ----------------------------------------------------

    def snapshot(self) -> Dict[str, DriftProfile]:
        return {
            did: node.drift_profile
            for did, node in self.graph.nodes.items()
        }

    def pop_events(self) -> List[Dict]:
        events = list(self._events)
        self._events.clear()
        return events
