from __future__ import annotations
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from mycelium.domain_graph.state import DomainState, NoveltyDecision

logger = logging.getLogger(__name__)

# Schema version carried in every graph event — bump when event shape changes.
GRAPH_SCHEMA_VERSION = "1.0"

# Try to hook into existing pipeline_event infrastructure.
# Graceful fallback to logger-only if not available.
try:
    from mycelium.pipeline.pipeline_event import emit_event as _pipeline_emit
    _HAS_PIPELINE_EMIT = True
except ImportError:
    _HAS_PIPELINE_EMIT = False
    logger.debug("pipeline_event not available — domain graph events will log only.")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _base_payload(event_type: str, domain_id: Optional[str] = None) -> Dict[str, Any]:
    return {
        "event_id": str(uuid.uuid4()),
        "event_type": event_type,
        "timestamp": _now(),
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
        "domain_id": domain_id,
    }


def _emit(payload: Dict[str, Any]) -> None:
    if _HAS_PIPELINE_EMIT:
        try:
            _pipeline_emit(payload)
            return
        except Exception as exc:
            logger.debug("pipeline_event emit failed (%s) — falling back to log.", exc)
    logger.info("graph_event | %s", payload)


# ---------------------------------------------------------------------------
# Public event emitters — all required fields per REBUILD_NOTES.md contract
# ---------------------------------------------------------------------------

def emit_domain_registered(
    domain_id: str,
    label: str,
    domain_version: int,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    payload = _base_payload("graph_domain_registered", domain_id)
    payload.update({
        "label": label,
        "domain_version": domain_version,
        "meta": meta or {},
    })
    _emit(payload)


def emit_state_transition(
    domain_id: str,
    from_state: DomainState,
    to_state: DomainState,
    domain_version: int,
    reason: str = "",
) -> None:
    payload = _base_payload("graph_domain_state_transition", domain_id)
    payload.update({
        "from_state": from_state.value,
        "to_state": to_state.value,
        "domain_version": domain_version,
        "reason": reason,
    })
    _emit(payload)


def emit_novelty_decision(
    domain_id: Optional[str],
    decision: NoveltyDecision,
    similarity: float,
    novelty_score: float,
    query_text: str = "",
    spectral_signal: Optional[str] = None,
    reason: str = "",
) -> None:
    payload = _base_payload("graph_domain_novelty_decision", domain_id)
    payload.update({
        "decision": decision.value,
        "similarity": round(similarity, 6),
        "novelty_score": round(novelty_score, 6),
        "query_text_preview": query_text[:120],
        "spectral_signal": spectral_signal,
        "reason": reason,
    })
    _emit(payload)


def emit_cold_store(
    domain_id: str,
    domain_version: int,
    head_version: int,
    reason: str = "",
) -> None:
    payload = _base_payload("graph_domain_cold_store", domain_id)
    payload.update({
        "domain_version": domain_version,
        "head_version": head_version,
        "reason": reason,
    })
    _emit(payload)


def emit_reactivation(
    domain_id: str,
    domain_version: int,
    head_version: int,
    trigger_similarity: float = 0.0,
) -> None:
    payload = _base_payload("graph_domain_reactivation", domain_id)
    payload.update({
        "domain_version": domain_version,
        "head_version": head_version,
        "trigger_similarity": round(trigger_similarity, 6),
    })
    _emit(payload)


def emit_drift_detected(
    domain_id: str,
    domain_version: int,
    drift_type: str,
    magnitude: float,
) -> None:
    payload = _base_payload("graph_domain_drift_detected", domain_id)
    payload.update({
        "domain_version": domain_version,
        "drift_type": drift_type,
        "magnitude": round(magnitude, 6),
    })
    _emit(payload)
