# mycelium/pipeline/phase2/evidence_finder.py
# Stage 7 — Evidence retrieval for every falsifiable PredicateFrame.
#
# EvidenceFinder is the Phase 2 entry point.  It receives a PredicateStore
# (produced by PredicatePipeline in Milestone B) and returns an EvidenceResult
# containing one EvidenceBundle per falsifiable frame.
#
# Retrieval strategy
# ------------------
# For each frame in store.falsifiable():
#   1. Build a keyword query string from the frame's semantic fields
#   2. Primary path  : FusionEngine.query()
#   3. Fallback path : UnifiedBertExpert.query()
#   4. Stub path     : empty bundle (FALLBACK) -- offline / test mode
#
# NORMATIVE frames (falsifiable=False) are silently skipped.
#
# Errors in retrieval for any individual frame are caught and produce an
# empty-items bundle with method=FALLBACK.  The pipeline never raises.
#
# Spec ref: implementation spec v0.2.1 — Section 8 (Evidence Layer)

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from mycelium.pipeline.phase2.evidence_types import (
    EvidenceBundle,
    EvidenceItem,
    EvidenceResult,
    RetrievalMethod,
)
from mycelium.pipeline.predicates.predicate_store import PredicateStore
from mycelium.pipeline.predicates.predicate_types import PredicateEntity, PredicateFrame


# ---------------------------------------------------------------------------
# Retriever protocol — both FusionEngine and UnifiedBertExpert satisfy this.
# Defined structurally so EvidenceFinder does not import either module at
# the top level (avoids heavy model-loading at import time).
# ---------------------------------------------------------------------------

@runtime_checkable
class _Retriever(Protocol):
    """Structural protocol for any retriever EvidenceFinder can call.

    A retriever must expose a .query() method that accepts a plain string
    and returns a list of result dicts.  Each dict must contain at minimum:
        'text'  or  'snippet'   : str   — the retrieved passage
        'score' or 'relevance'  : float — relevance in [0.0, 1.0]
    Optional keys: 'id', 'url', 'title', 'metadata'.
    """
    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]: ...


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------

def _entity_text(value: Any) -> str:
    """Safely extract a plain string from a field that may be a
    ``PredicateEntity`` dataclass, a raw string, or None.

    PredicateFrame.subject and PredicateFrame.object are typed as
    ``PredicateEntity | str``.  Passing a PredicateEntity directly to
    ``str.join()`` raised:
        TypeError: sequence item 0: expected str instance, PredicateEntity found
    This helper centralises the extraction so _build_query stays readable.
    """
    if value is None:
        return ""
    if isinstance(value, PredicateEntity):
        return value.text or ""
    return str(value)


def _build_query(frame: PredicateFrame) -> str:
    """Construct a keyword search query from the semantic fields of a frame.

    Extracts plain-string representations of subject, relation verb, and
    object.  Both subject and object may be ``PredicateEntity`` instances
    or raw strings; the relation is carried by ``frame.relation.lemma``.

    Falls back to the first 12 tokens of raw_text if all three fields are
    empty.  Always returns a non-empty string.
    """
    parts: List[str] = []

    subject_text = _entity_text(getattr(frame, "subject", None))
    if subject_text:
        parts.append(subject_text)

    # The relation verb lives on frame.relation.lemma, not frame.predicate_verb.
    relation = getattr(frame, "relation", None)
    relation_text = (relation.lemma if relation is not None else "") or ""
    if relation_text:
        parts.append(relation_text)

    object_text = _entity_text(getattr(frame, "object", None))
    if object_text:
        parts.append(object_text)

    if parts:
        return " ".join(parts).strip()

    # Fallback: first 12 words of raw_text
    words = frame.raw_text.split()
    return " ".join(words[:12]).strip() or frame.predicate_id


# ---------------------------------------------------------------------------
# Result normaliser
# ---------------------------------------------------------------------------

def _normalise_result(
    raw: Dict[str, Any],
    predicate_id: str,
    method: RetrievalMethod,
) -> EvidenceItem:
    """Convert a raw retriever result dict to an EvidenceItem.

    Handles the two common key conventions (text/snippet, score/relevance)
    from FusionEngine and UnifiedBertExpert.
    """
    snippet = raw.get("snippet") or raw.get("text") or ""
    score = float(raw.get("relevance") or raw.get("score") or 0.0)
    # Clamp to [0.0, 1.0]
    score = max(0.0, min(1.0, score))
    return EvidenceItem(
        predicate_id=predicate_id,
        source_id=str(raw.get("id") or raw.get("source_id") or ""),
        snippet=snippet,
        relevance_score=score,
        retrieval_method=method,
        url=str(raw.get("url") or ""),
        title=str(raw.get("title") or ""),
        metadata={k: v for k, v in raw.items()
                  if k not in ("snippet", "text", "relevance", "score",
                               "id", "source_id", "url", "title")},
    )


# ---------------------------------------------------------------------------
# EvidenceFinder
# ---------------------------------------------------------------------------

class EvidenceFinder:
    """Retrieve supporting evidence for every falsifiable PredicateFrame.

    Typical usage
    -------------
    ::
        from mycelium.pipeline.phase2.evidence_finder import get_evidence_finder

        result = get_evidence_finder().find(context.predicate_store)
        context.evidence_result = result

    Constructor
    -----------
    EvidenceFinder accepts optional injected retrievers for testing:

        finder = EvidenceFinder(
            primary_retriever=MyFusionStub(),
            fallback_retriever=MyBertStub(),
        )

    If primary_retriever is None, EvidenceFinder attempts to lazily import
    FusionEngine from mycelium.pipeline.fusion_engine.  If that import fails
    (offline / not installed), it falls back to UnifiedBertExpert.
    If both fail, it uses the FALLBACK stub.
    """

    def __init__(
        self,
        primary_retriever: Optional[_Retriever] = None,
        fallback_retriever: Optional[_Retriever] = None,
        top_k: int = 5,
    ) -> None:
        self._primary = primary_retriever
        self._fallback = fallback_retriever
        self._top_k = top_k
        self._primary_method = RetrievalMethod.FUSION
        self._fallback_method = RetrievalMethod.BERT_EXPERT

    # ------------------------------------------------------------------
    # Lazy retriever resolution
    # ------------------------------------------------------------------

    def _get_primary(self) -> Optional[_Retriever]:
        if self._primary is not None:
            return self._primary
        try:
            from mycelium.pipeline.fusion_engine import FusionEngine  # noqa: PLC0415
            self._primary = FusionEngine()
            return self._primary
        except Exception:  # noqa: BLE001
            return None

    def _get_fallback(self) -> Optional[_Retriever]:
        if self._fallback is not None:
            return self._fallback
        try:
            raise ImportError("unified_bert_expert removed in TRM v0.2 — use TRMV2InferenceEngine")  # noqa: PLC0415
            self._fallback = UnifiedBertExpert()
            self._fallback_method = RetrievalMethod.BERT_EXPERT
            return self._fallback
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------
    # Per-frame retrieval
    # ------------------------------------------------------------------

    def find_for_frame(
        self,
        frame: PredicateFrame,
        domain_hint: str = "general",
    ) -> EvidenceBundle:
        """Retrieve evidence for a single PredicateFrame.

        Tries primary retriever first, then fallback, then returns an
        empty-items FALLBACK bundle on any error.

        Parameters
        ----------
        frame:
            A falsifiable PredicateFrame to retrieve evidence for.
        domain_hint:
            Domain string from L1 routing — passed as metadata to the
            retriever when the retriever protocol supports it.

        Returns
        -------
        EvidenceBundle with items sorted descending by relevance_score.
        Never raises.
        """
        query = _build_query(frame)
        t0 = time.perf_counter()

        raw_results: List[Dict[str, Any]] = []
        method = RetrievalMethod.FALLBACK

        primary = self._get_primary()
        if primary is not None:
            try:
                raw_results = primary.query(query, top_k=self._top_k)
                method = self._primary_method
            except Exception:  # noqa: BLE001
                raw_results = []

        if not raw_results:
            fallback = self._get_fallback()
            if fallback is not None:
                try:
                    raw_results = fallback.query(query, top_k=self._top_k)
                    method = self._fallback_method
                except Exception:  # noqa: BLE001
                    raw_results = []

        items = [
            _normalise_result(r, frame.predicate_id, method)
            for r in raw_results
        ]
        items.sort(key=lambda i: i.relevance_score, reverse=True)

        retrieval_ms = (time.perf_counter() - t0) * 1000.0
        return EvidenceBundle(
            frame=frame,
            items=items,
            query_used=query,
            retrieval_method=method if items else RetrievalMethod.FALLBACK,
            retrieval_ms=retrieval_ms,
        )

    # ------------------------------------------------------------------
    # Full-store pass
    # ------------------------------------------------------------------

    def find(
        self,
        store: PredicateStore,
        domain_hint: str = "general",
    ) -> EvidenceResult:
        """Run evidence retrieval for every falsifiable frame in the store.

        NORMATIVE frames (falsifiable=False) are silently skipped and
        recorded in EvidenceResult.skipped_ids.

        Parameters
        ----------
        store:
            Populated PredicateStore from Milestone B pipeline.
        domain_hint:
            Domain string propagated to each per-frame retrieval call.

        Returns
        -------
        EvidenceResult with one bundle per falsifiable frame.
        Never raises.
        """
        t0 = time.perf_counter()

        bundles: List[EvidenceBundle] = []
        skipped_ids: List[str] = []

        for frame in store:
            if not frame.falsifiable:
                skipped_ids.append(frame.predicate_id)
                continue
            bundle = self.find_for_frame(frame, domain_hint=domain_hint)
            bundles.append(bundle)

        total_ms = (time.perf_counter() - t0) * 1000.0
        return EvidenceResult(
            bundles=bundles,
            total_ms=total_ms,
            store_summary=store.summary(),
            skipped_ids=skipped_ids,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"EvidenceFinder("
            f"primary={type(self._primary).__name__ if self._primary else None}, "
            f"fallback={type(self._fallback).__name__ if self._fallback else None}, "
            f"top_k={self._top_k})"
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_finder_instance: Optional[EvidenceFinder] = None


def get_evidence_finder() -> EvidenceFinder:
    """Return the module-level EvidenceFinder singleton.

    Lazily instantiated on first call.  Uses lazy retriever resolution
    so no heavy models are loaded at import time.
    """
    global _finder_instance
    if _finder_instance is None:
        _finder_instance = EvidenceFinder()
    return _finder_instance
