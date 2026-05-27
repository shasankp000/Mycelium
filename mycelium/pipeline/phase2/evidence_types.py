# mycelium/pipeline/phase2/evidence_types.py
# Data types for the evidence retrieval layer (Stage 7).
#
# These are pure data containers — no logic, no I/O.
# Used by EvidenceFinder (evidence_finder.py) and consumed by
# the Phase 2 scoring / synthesis steps downstream.
#
# Spec ref: implementation spec v0.2.1 — Section 8 (Evidence Layer)

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from mycelium.pipeline.predicates.predicate_types import PredicateFrame


class RetrievalMethod(str, Enum):
    """How the evidence items for a bundle were retrieved."""
    FUSION = "fusion"              # FusionEngine primary path
    BERT_EXPERT = "bert_expert"    # UnifiedBertExpert fallback
    FALLBACK = "fallback"          # Stub / offline / error path


@dataclass
class EvidenceItem:
    """A single retrieved passage or document fragment.

    Attributes
    ----------
    predicate_id:
        ID of the PredicateFrame this item is evidence for.
    source_id:
        Unique identifier for the source document / chunk.
    snippet:
        The retrieved text passage (may be empty on retrieval failure).
    relevance_score:
        Float in [0.0, 1.0].  Higher = more relevant to the query.
    retrieval_method:
        Which retrieval path produced this item.
    url:
        Optional canonical URL for the source. Empty string if unknown.
    title:
        Optional document title. Empty string if unknown.
    metadata:
        Arbitrary extra fields from the retriever (e.g. chunk_index,
        model_name, score_breakdown).  Not used by downstream scoring
        directly but preserved for auditability.
    """
    predicate_id: str
    source_id: str
    snippet: str
    relevance_score: float
    retrieval_method: RetrievalMethod
    url: str = ""
    title: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceBundle:
    """All retrieved evidence for a single PredicateFrame.

    Attributes
    ----------
    frame:
        The PredicateFrame this bundle covers.
    items:
        Retrieved EvidenceItems, sorted descending by relevance_score.
        Empty list if retrieval failed or produced no results.
    query_used:
        The search query string that was sent to the retriever.
        Recorded for auditability and debugging.
    retrieval_method:
        Overall method used for this bundle (mirrors items[0].retrieval_method
        or FALLBACK if items is empty).
    retrieval_ms:
        Wall-clock time in milliseconds for this frame's retrieval.
    """
    frame: PredicateFrame
    items: List[EvidenceItem]
    query_used: str
    retrieval_method: RetrievalMethod
    retrieval_ms: float = 0.0

    def best(self) -> Optional[EvidenceItem]:
        """Return the highest-relevance item, or None if items is empty."""
        return self.items[0] if self.items else None

    def above_threshold(self, threshold: float = 0.5) -> List[EvidenceItem]:
        """Return items with relevance_score >= threshold."""
        return [i for i in self.items if i.relevance_score >= threshold]


@dataclass
class EvidenceResult:
    """Top-level container for one full pipeline evidence pass.

    Produced by EvidenceFinder.find() and attached to the workflow
    context as context.evidence_result.

    Attributes
    ----------
    bundles:
        One EvidenceBundle per falsifiable PredicateFrame, in the same
        order as store.falsifiable().
    total_ms:
        Total wall-clock time in milliseconds for the entire find() call.
    store_summary:
        Output of PredicateStore.summary() at the time find() was called.
        Carried here for SSE event emission without re-querying the store.
    skipped_ids:
        predicate_ids of frames that were skipped (NORMATIVE / non-falsifiable).
    """
    bundles: List[EvidenceBundle]
    total_ms: float
    store_summary: Dict[str, Any]
    skipped_ids: List[str] = field(default_factory=list)

    def bundle_for(self, predicate_id: str) -> Optional[EvidenceBundle]:
        """Return the bundle for the given predicate_id, or None."""
        for b in self.bundles:
            if b.frame.predicate_id == predicate_id:
                return b
        return None

    def all_items(self) -> List[EvidenceItem]:
        """Flatten all items across all bundles into a single list."""
        result = []
        for b in self.bundles:
            result.extend(b.items)
        return result

    def summary(self) -> Dict[str, Any]:
        """Lightweight dict for SSE event emission."""
        return {
            "bundles": len(self.bundles),
            "total_items": len(self.all_items()),
            "total_ms": round(self.total_ms, 2),
            "skipped": len(self.skipped_ids),
            "store": self.store_summary,
        }
