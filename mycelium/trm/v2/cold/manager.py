from __future__ import annotations
import logging
from typing import Dict, List, Optional

from mycelium.domain_graph.registry import DomainGraphRegistry
from mycelium.domain_graph.state import DomainState, GateState
from mycelium.domain_graph.events import emit_cold_store, emit_reactivation, emit_drift_detected
from mycelium.trm.v2.cold.drift import DriftDetector, DriftConfig, DriftSignal
from mycelium.trm.v2.cold.replay import ReplayBufferManager, ReplayEntry
from mycelium.trm.v2.cold.reactivation import ReactivationQueue, ReactivationTicket

logger = logging.getLogger(__name__)


class ColdStorageManager:
    """
    Orchestrates drift detection, cold storage transitions, replay buffers,
    and the reactivation queue for all domains.

    Responsibilities:
      - Maintain one DriftDetector per HOT domain.
      - On every routed query: update detector, push to replay buffer.
      - Periodically evaluate drift; emit emit_drift_detected and open a
        thaw window via GateModeController when drift triggers.
      - Send domains to COLD state and emit emit_cold_store when they
        fall below activity thresholds.
      - Enqueue COLD domains into ReactivationQueue when they receive
        sufficient demand; emit emit_reactivation on pop.

    This class does NOT train or run inference — it only manages state
    and signals to the Trainer and InferenceEngine.
    """

    def __init__(
        self,
        registry: DomainGraphRegistry,
        replay_manager: Optional[ReplayBufferManager] = None,
        drift_config: Optional[DriftConfig] = None,
        cold_inactivity_threshold: int = 500,
        reactivation_demand_threshold: int = 5,
    ) -> None:
        self._registry = registry
        self._replay = replay_manager or ReplayBufferManager()
        self._drift_config = drift_config or DriftConfig()
        self._cold_threshold = cold_inactivity_threshold
        self._reactivation_threshold = reactivation_demand_threshold

        self._detectors: Dict[str, DriftDetector] = {}
        self._reactivation_queue = ReactivationQueue(registry)
        self._cold_demand: Dict[str, int] = {}

        self._initialise_detectors()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialise_detectors(self) -> None:
        for node in self._registry.all_hot_domains():
            centroid = node.meta.get("centroid")
            if centroid:
                self._detectors[node.domain_id] = DriftDetector(
                    node.domain_id, centroid, self._drift_config
                )

    def ensure_detector(self, domain_id: str) -> None:
        if domain_id in self._detectors:
            return
        node = self._registry.get(domain_id)
        if node is None:
            return
        centroid = node.meta.get("centroid", [0.0])
        self._detectors[domain_id] = DriftDetector(domain_id, centroid, self._drift_config)

    # ------------------------------------------------------------------
    # Per-query update
    # ------------------------------------------------------------------

    def on_query_routed(
        self,
        domain_id: str,
        embedding: List[float],
        query_text: str,
        label: int = 1,
    ) -> Optional[DriftSignal]:
        """
        Call after every successful domain routing.
        Returns a DriftSignal if drift was triggered, else None.
        """
        self.ensure_detector(domain_id)
        self._detectors[domain_id].update(embedding)

        self._replay.push(ReplayEntry(
            embedding=embedding,
            query_text=query_text,
            domain_id=domain_id,
            label=label,
        ))

        signal = self._detectors[domain_id].evaluate()
        if signal.triggered:
            node = self._registry.get(domain_id)
            dv = node.domain_version if node else 1
            # emit_drift_detected expects: domain_id, domain_version, drift_type, magnitude
            emit_drift_detected(
                domain_id=domain_id,
                domain_version=dv,
                drift_type="centroid_shift" if signal.centroid_delta >= self._drift_config.centroid_delta_threshold else "variance_explosion",
                magnitude=signal.drift_score,
            )
            logger.info("Drift triggered for domain '%s': %s", domain_id, signal.reason)
            return signal

        return None

    # ------------------------------------------------------------------
    # Cold store / reactivation
    # ------------------------------------------------------------------

    def cold_store(self, domain_id: str, reason: str = "") -> None:
        """Transition a domain to COLD state and emit the event."""
        node = self._registry.get(domain_id)
        if node is None:
            return
        self._registry.set_state(domain_id, DomainState.COLD)
        self._registry.set_gate(domain_id, GateState.CLOSED)
        # emit_cold_store expects: domain_id, domain_version, head_version, reason
        emit_cold_store(
            domain_id=domain_id,
            domain_version=node.domain_version,
            head_version=node.head_version,
            reason=reason,
        )
        logger.info("Domain '%s' moved to COLD storage: %s", domain_id, reason)

    def on_cold_domain_hit(
        self,
        domain_id: str,
        novelty_score: float,
        query_text: str,
    ) -> bool:
        """
        Called when a COLD domain receives a query hit.
        Returns True if reactivation was queued.
        """
        count = self._cold_demand.get(domain_id, 0) + 1
        self._cold_demand[domain_id] = count

        if count >= self._reactivation_threshold:
            self._reactivation_queue.enqueue(
                domain_id=domain_id,
                novelty_score=novelty_score,
                query_count=count,
                reason=f"demand threshold reached ({count} hits)",
            )
            logger.info("Domain '%s' queued for reactivation (demand=%d).", domain_id, count)
            return True

        return False

    def pop_reactivation(self) -> Optional[ReactivationTicket]:
        """
        Pop the next domain for reactivation.
        Caller must: load its head, call registry.set_state(HOT) + set_gate(OPEN).
        """
        ticket = self._reactivation_queue.pop()
        if ticket:
            node = self._registry.get(ticket.domain_id)
            dv = node.domain_version if node else 1
            hv = node.head_version if node else 0
            # emit_reactivation expects: domain_id, domain_version, head_version, trigger_similarity
            emit_reactivation(
                domain_id=ticket.domain_id,
                domain_version=dv,
                head_version=hv,
                trigger_similarity=ticket.novelty_score,
            )
        return ticket

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def drift_detector(self, domain_id: str) -> Optional[DriftDetector]:
        return self._detectors.get(domain_id)

    def reactivation_queue_size(self) -> int:
        return len(self._reactivation_queue)

    def replay_sizes(self) -> Dict[str, int]:
        return self._replay.sizes()
