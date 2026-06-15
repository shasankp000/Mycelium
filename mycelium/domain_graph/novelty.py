from __future__ import annotations
import logging
import math
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from mycelium.domain_graph.state import NoveltyDecision, DomainState, GateState
from mycelium.domain_graph.registry import DomainGraphRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _unpack(blob: bytes) -> List[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class NoveltyResult:
    decision: NoveltyDecision
    matched_domain_id: Optional[str] = None      # set for ROUTE/EXPAND/REACTIVATE
    similarity: float = 0.0                       # best cosine score found
    novelty_score: float = 0.0                    # 1.0 - similarity (how new it is)
    spectral_signal: Optional[str] = None         # optional tag from SpectralAnalyzer
    reason: str = ""
    meta: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Thresholds (configurable at construction time)
# ---------------------------------------------------------------------------

@dataclass
class NoveltyThresholds:
    route_existing: float = 0.72    # sim >= this → ROUTE_EXISTING
    expand_existing: float = 0.50   # sim in [expand, route) → EXPAND_EXISTING
    reactivate_cold: float = 0.65   # sim >= this against a COLD domain → REACTIVATE_COLD
    ood_floor: float = 0.25         # sim < this → DEFER_OOD (not even CREATE_NEW yet)
    # sim in [ood_floor, expand) → CREATE_NEW


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class DomainNoveltyPolicy:
    """
    Decides what to do with a query embedding given the current DomainGraphRegistry.

    Decision flow:
      1. Compute cosine sim against all HOT+OPEN domain centroids.
      2. If best_sim >= route_existing  → ROUTE_EXISTING
      3. If best_sim >= expand_existing → EXPAND_EXISTING
      4. Check COLD domains:  if best_cold_sim >= reactivate_cold → REACTIVATE_COLD
      5. If best_sim >= ood_floor       → CREATE_NEW
      6. Otherwise                      → DEFER_OOD

    Centroids are stored as BLOB in the DomainNode.meta["centroid"] field.
    If a domain has no centroid yet (e.g. still bootstrapping), it is skipped.
    """

    def __init__(
        self,
        registry: DomainGraphRegistry,
        thresholds: Optional[NoveltyThresholds] = None,
        spectral_analyzer=None,   # optional: mycelium.pipeline.spectral_analyzer.SpectralAnalyzer
    ) -> None:
        self._registry = registry
        self._thresholds = thresholds or NoveltyThresholds()
        self._spectral = spectral_analyzer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        query_embedding: List[float],
        query_text: str = "",
    ) -> NoveltyResult:
        t = self._thresholds

        hot_sims = self._score_domains(query_embedding, states={DomainState.HOT})
        cold_sims = self._score_domains(query_embedding, states={DomainState.COLD})

        spectral_tag: Optional[str] = None
        if self._spectral is not None and query_text:
            try:
                spectral_tag = self._spectral.dominant_tag(query_text)
            except Exception as exc:
                logger.debug("SpectralAnalyzer failed (non-fatal): %s", exc)

        # --- best HOT match ---
        best_hot = max(hot_sims, key=lambda x: x[1], default=(None, 0.0))
        best_domain_id, best_sim = best_hot

        if best_sim >= t.route_existing and best_domain_id:
            return NoveltyResult(
                decision=NoveltyDecision.ROUTE_EXISTING,
                matched_domain_id=best_domain_id,
                similarity=best_sim,
                novelty_score=1.0 - best_sim,
                spectral_signal=spectral_tag,
                reason=f"sim={best_sim:.4f} >= route_threshold={t.route_existing}",
            )

        if best_sim >= t.expand_existing and best_domain_id:
            return NoveltyResult(
                decision=NoveltyDecision.EXPAND_EXISTING,
                matched_domain_id=best_domain_id,
                similarity=best_sim,
                novelty_score=1.0 - best_sim,
                spectral_signal=spectral_tag,
                reason=f"sim={best_sim:.4f} in expand range [{t.expand_existing}, {t.route_existing})",
            )

        # --- check COLD domains for reactivation ---
        best_cold = max(cold_sims, key=lambda x: x[1], default=(None, 0.0))
        cold_domain_id, cold_sim = best_cold
        if cold_sim >= t.reactivate_cold and cold_domain_id:
            return NoveltyResult(
                decision=NoveltyDecision.REACTIVATE_COLD,
                matched_domain_id=cold_domain_id,
                similarity=cold_sim,
                novelty_score=1.0 - cold_sim,
                spectral_signal=spectral_tag,
                reason=f"cold domain sim={cold_sim:.4f} >= reactivate_threshold={t.reactivate_cold}",
            )

        # --- below expand but above OOD floor → genuinely new ---
        if best_sim >= t.ood_floor or cold_sim >= t.ood_floor:
            return NoveltyResult(
                decision=NoveltyDecision.CREATE_NEW,
                matched_domain_id=None,
                similarity=max(best_sim, cold_sim),
                novelty_score=1.0 - max(best_sim, cold_sim),
                spectral_signal=spectral_tag,
                reason=f"sim={max(best_sim, cold_sim):.4f} in create range [{t.ood_floor}, {t.expand_existing})",
            )

        # --- below OOD floor → defer ---
        return NoveltyResult(
            decision=NoveltyDecision.DEFER_OOD,
            matched_domain_id=None,
            similarity=max(best_sim, cold_sim),
            novelty_score=1.0,
            spectral_signal=spectral_tag,
            reason=f"sim={max(best_sim, cold_sim):.4f} < ood_floor={t.ood_floor}",
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _score_domains(
        self,
        query_embedding: List[float],
        states: set,
    ) -> List[Tuple[str, float]]:
        results = []
        for node in self._registry.all_domains():
            if node.state not in states:
                continue
            if node.gate == GateState.CLOSED and DomainState.HOT in states:
                continue
            centroid_blob = node.meta.get("centroid")
            if not centroid_blob:
                continue
            if isinstance(centroid_blob, list):
                centroid = centroid_blob
            else:
                try:
                    centroid = _unpack(centroid_blob)
                except Exception:
                    continue
            sim = _cosine(query_embedding, centroid)
            results.append((node.domain_id, sim))
        return results

    def update_centroid(
        self,
        domain_id: str,
        new_embedding: List[float],
        alpha: float = 0.1,
    ) -> None:
        """
        Exponential moving average update of the domain centroid.
        centroid = (1 - alpha) * centroid + alpha * new_embedding
        """
        node = self._registry.get(domain_id)
        if node is None:
            return
        old = node.meta.get("centroid")
        if old is None:
            node.meta["centroid"] = new_embedding
        else:
            if not isinstance(old, list):
                old = _unpack(old)
            updated = [
                (1.0 - alpha) * o + alpha * n
                for o, n in zip(old, new_embedding)
            ]
            norm = math.sqrt(sum(x * x for x in updated)) or 1.0
            node.meta["centroid"] = [x / norm for x in updated]
        self._registry.update(node)
