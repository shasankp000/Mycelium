from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mycelium.domain_graph.registry import DomainGraphRegistry
from mycelium.domain_graph.state import DomainState, GateState
from mycelium.trm.v2.inference import TRMV2InferenceEngine, RouteResult
from mycelium.trm.v2.cold.manager import ColdStorageManager
from mycelium.trm.v2.cold.drift import DriftSignal
from mycelium.trm.v2.store.query_store import QueryStore

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    route: RouteResult
    stored_query_id: Optional[int] = None
    drift_signal: Optional[DriftSignal] = None
    reactivation_triggered: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)


class TRMV2Pipeline:
    """
    End-to-end TRM v2 pipeline.

    This is the ONLY public entry point for production inference.
    Callers supply a query; this class:
      1. Routes via TRMV2InferenceEngine  (single routing authority)
      2. Persists to per-domain QueryStore shard
      3. Updates ColdStorageManager drift detector
      4. Handles COLD domain demand tracking
      5. Surfaces drift signals and reactivation tickets to the caller

    Contract:
      - No routing logic lives here — only orchestration.
      - Feature-flagged: disabled by default, enabled via `enabled=True`.
        Legacy pipeline continues to run while this flag is off.
    """

    def __init__(
        self,
        engine: TRMV2InferenceEngine,
        cold_manager: ColdStorageManager,
        query_store: QueryStore,
        registry: DomainGraphRegistry,
        *,
        enabled: bool = False,          # feature flag — must be explicitly set True
        store_ood_queries: bool = False, # whether to persist OOD queries
    ) -> None:
        self._engine = engine
        self._cold = cold_manager
        self._store = query_store
        self._registry = registry
        self.enabled = enabled
        self._store_ood = store_ood_queries

        if not enabled:
            logger.info(
                "TRMV2Pipeline instantiated but feature flag `enabled=False`. "
                "Call pipeline.enabled = True to activate."
            )

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def run(
        self,
        query_text: str,
        input_ids=None,
        attention_mask=None,
        return_embedding: bool = False,
    ) -> PipelineResult:
        """
        Run one query through the full pipeline.

        Raises RuntimeError if called while `enabled=False`.
        """
        if not self.enabled:
            raise RuntimeError(
                "TRMV2Pipeline is disabled. Set pipeline.enabled = True to activate. "
                "Ensure legacy pipeline parity before switching."
            )

        # --- Step 1: route ---
        route = self._engine.route(
            query_text,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_embedding=True,          # always need embedding for store + drift
        )

        embedding: List[float] = route.embedding or []
        stored_id: Optional[int] = None
        drift_signal: Optional[DriftSignal] = None
        reactivation_triggered = False

        # --- Step 2: persist ---
        if route.selected_domain_id is not None:
            node = self._registry.get(route.selected_domain_id)
            dv = node.domain_version if node else 1
            hv = node.head_version if node else 0

            stored_id = self._store.record_query(
                domain_id=route.selected_domain_id,
                query_text=query_text,
                embedding=embedding,
                label=1,
                confidence=route.gate_confidence,
                novelty_sim=route.novelty_similarity,
                domain_version=dv,
                head_version=hv,
            )

            # --- Step 3: drift update ---
            drift_signal = self._cold.on_query_routed(
                domain_id=route.selected_domain_id,
                embedding=embedding,
                query_text=query_text,
                label=1,
            )

            # If drift triggered, also record it in the shard
            if drift_signal:
                self._store.record_drift(
                    domain_id=route.selected_domain_id,
                    drift_type=(
                        "centroid_shift"
                        if drift_signal.centroid_delta
                           >= self._cold._drift_config.centroid_delta_threshold
                        else "variance_explosion"
                    ),
                    magnitude=drift_signal.drift_score,
                    centroid_delta=drift_signal.centroid_delta,
                    variance_ratio=drift_signal.variance_ratio,
                    sample_count=drift_signal.sample_count,
                    domain_version=dv,
                    notes=drift_signal.reason,
                )

        elif self._store_ood and embedding:
            # OOD query — optionally persist to a dedicated shard
            stored_id = self._store.record_query(
                domain_id="__ood__",
                query_text=query_text,
                embedding=embedding,
                label=0,
                confidence=route.gate_confidence,
                novelty_sim=route.novelty_similarity,
            )

        # --- Step 4: COLD domain demand tracking ---
        if route.ood_fallback and route.novelty_similarity > 0:
            # Check if any cold domain was close (novelty policy set matched_domain_id)
            cold_candidate = route.route.matched_domain_id if hasattr(route, "route") else None
            # RouteResult carries matched_domain_id indirectly via novelty_decision;
            # for REACTIVATE_COLD the selected_domain_id is already set above.
            # For DEFER_OOD with a cold candidate we check the registry.
            if cold_candidate is None:
                # scan for COLD domains — future: novelty policy will surface this directly
                pass

        return PipelineResult(
            route=route,
            stored_query_id=stored_id,
            drift_signal=drift_signal,
            reactivation_triggered=reactivation_triggered,
        )

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def store_stats(self) -> Dict[str, Any]:
        return self._store.stats()

    def reactivation_queue_size(self) -> int:
        return self._cold.reactivation_queue_size()

    def pop_reactivation(self):
        return self._cold.pop_reactivation()
