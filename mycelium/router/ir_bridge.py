"""
Phase C — IR Bridge
====================
Translates the raw output of multi_lens_route() (Layer 1 / Phase B
canonicalization) plus RuntimeSpectralAnalyzer scores (Phase 1) into
IRNode / IRGraph structures so the routing pipeline can reason over
canonicalized, hash-stable claim representations instead of raw score dicts.

Design Notes (Consolidation §46, Phase C):
    Critical ordering rule enforced here:
        1. SRL extraction from raw text          ← CanonicalizeAndHash.process()
        2. Predicate family classification        ← CanonicalizeAndHash.process()
        3. Canonical form generation              ← CanonicalizeAndHash.process()
        4. Semantic hash computation              ← CanonicalizeAndHash.process()
        5. Spectral signature population          ← THIS MODULE (Phase C)
        6. IRGraph construction with fingerprint  ← THIS MODULE (Phase C)
        7. TRM lookup                             ← Phase D (not yet)

    spectral_signature on each IRNode is filled here by packing the
    per-domain spectral float scores (from RuntimeSpectralAnalyzer) into
    a sorted list of [domain, score] pairs.  This fulfils the placeholder
    left in semantic_hash_pipeline.py:
        spectral_signature=[]  # Phase C fills this

    IRBridge.build() is the single entry-point.  It is import-safe: all
    mycelium.* imports are guarded so the file can be loaded in stripped
    test environments (graceful degradation returns empty structures).

Optimisations (§5.4):
    IRBridge now constructs a single SRLExtractor and CanonicalFormGenerator
    at __init__ time and passes them into CanonicalizeAndHash, so all
    build() calls share the same spaCy model instance, NER doc cache, and
    equivalence registry.  Previously a fresh CanonicalizeAndHash (and
    therefore a fresh SRLExtractor + fresh spaCy load) was created each
    time build() was called, burning the full spaCy load cost per request.
"""

from __future__ import annotations

import datetime
import hashlib
import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# Optional imports — Phase C gracefully degrades if mycelium stack is absent
# ---------------------------------------------------------------------------
try:
    from mycelium.canonicalization.srl_extractor import SRLExtractor
    from mycelium.canonicalization.canonical_form import CanonicalFormGenerator
    from mycelium.canonicalization.semantic_hash_pipeline import CanonicalizeAndHash
    _CANONICALIZER_AVAILABLE = True
except ImportError:
    SRLExtractor = None          # type: ignore[assignment,misc]
    CanonicalFormGenerator = None  # type: ignore[assignment,misc]
    CanonicalizeAndHash = None   # type: ignore[assignment,misc]
    _CANONICALIZER_AVAILABLE = False

try:
    from mycelium.ir.graph import IRGraph, IREdge
    from mycelium.ir.primitives import (
        GraphFingerprint,
        ConfidenceState,
        SemanticSignature,
    )
    from mycelium.ir.serialization import compute_graph_fingerprint_hashes
    _IR_AVAILABLE = True
except ImportError:
    IRGraph = None        # type: ignore[assignment,misc]
    IREdge = None         # type: ignore[assignment,misc]
    GraphFingerprint = None  # type: ignore[assignment,misc]
    ConfidenceState = None   # type: ignore[assignment,misc]
    SemanticSignature = None  # type: ignore[assignment,misc]
    compute_graph_fingerprint_hashes = None  # type: ignore[assignment]
    _IR_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pack_spectral_signature(spectral_scores: Dict[str, float]) -> List[Any]:
    """Convert {domain: score} dict to a sorted [[domain, score], ...] list.

    Stored in SemanticSignature.spectral_signature so Phase D can read
    per-domain scores without re-running the spectral analyzer.
    Sorted by domain name for determinism.
    """
    return sorted(
        [domain, round(score, 6)]
        for domain, score in spectral_scores.items()
    )


def _graph_id_from_nodes(nodes: List[Any]) -> str:
    """Stable graph ID derived from the sorted semantic hashes of its nodes."""
    hashes = sorted(
        n.semantic_signature.semantic_hash
        for n in nodes
        if hasattr(n, "semantic_signature")
    )
    payload = "|".join(hashes).encode("utf-8")
    short = hashlib.sha256(payload).hexdigest()[:16]
    return f"G_phaseC_{short}"


# ---------------------------------------------------------------------------
# IRBridge
# ---------------------------------------------------------------------------

class IRBridge:
    """Converts raw routing pipeline artefacts into IR-layer structures.

    Parameters
    ----------
    embedding_fn : callable, optional
        (str) -> list[float] embedding function forwarded to
        CanonicalizeAndHash.  If None, embedding_signature will be empty.

    Optimisation (§5.4):
        A single SRLExtractor and CanonicalFormGenerator are constructed
        once at __init__ time and shared across every build() call.  This
        avoids the repeated spaCy model load that occurred when
        CanonicalizeAndHash was instantiated fresh on each request.

    Usage
    -----
    bridge = IRBridge()
    result = bridge.build(
        text="Smoking causes lung cancer.",
        spectral_scores={"medicine": 0.82, "biology": 0.61},
        fused_scores={"medicine": 0.78, "biology": 0.55},
    )
    # result["ir_nodes"]    — list[IRNode]
    # result["ir_graph"]    — IRGraph (state=DRAFT)
    # result["graph_id"]    — str
    # result["ir_available"] — bool (False when stack missing)
    """

    def __init__(self, *, embedding_fn: Optional[Any] = None) -> None:
        self._embedding_fn = embedding_fn
        self._canonicalizer: Optional[Any] = None

        if _CANONICALIZER_AVAILABLE:
            try:
                # §5.4: shared extractor + generator instances so spaCy is
                # loaded exactly once and the NER doc cache / equivalence
                # registry persist across all build() calls on this bridge.
                shared_srl = SRLExtractor()
                shared_gen = CanonicalFormGenerator()
                self._canonicalizer = CanonicalizeAndHash(
                    embedding_fn=embedding_fn,
                    srl_extractor=shared_srl,
                    canonical_generator=shared_gen,
                )
            except Exception as exc:
                logger.warning("IRBridge: failed to init CanonicalizeAndHash: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        text: str,
        *,
        spectral_scores: Optional[Dict[str, float]] = None,
        fused_scores: Optional[Dict[str, float]] = None,
        source: str = "phase_c_router",
    ) -> Dict[str, Any]:
        """Run the §46 pipeline steps 1-6 and return IR artefacts.

        Parameters
        ----------
        text : str
            Raw query text (the same string passed to MultiLensRouter.route).
        spectral_scores : dict, optional
            {domain: float} from RuntimeSpectralAnalyzer.  Packed into
            each IRNode's SemanticSignature.spectral_signature.
        fused_scores : dict, optional
            {domain: float} final fused scores.  Used to set
            ConfidenceState.overall_confidence on the IRGraph.
        source : str
            Provenance source tag written into each IRNode's ProvenanceChain.

        Returns
        -------
        dict with keys:
            ir_nodes    : list[IRNode]  — one per SRL triple (may be empty)
            ir_graph    : IRGraph | None — DRAFT graph wrapping all nodes
            graph_id    : str           — stable ID for this request's graph
            ir_available: bool          — False when the IR stack is missing
        """
        spectral_scores = spectral_scores or {}
        fused_scores = fused_scores or {}

        if not _CANONICALIZER_AVAILABLE or not _IR_AVAILABLE or self._canonicalizer is None:
            return {
                "ir_nodes": [],
                "ir_graph": None,
                "graph_id": "",
                "ir_available": False,
            }

        # ── Steps 1-4: SRL → predicate family → canonical form → hash ────
        try:
            nodes: List[Any] = self._canonicalizer.process(
                text,
                node_type="CLAIM",
                temporal_type="UNKNOWN",
                abstraction_level=0,
                source=source,
            )
        except Exception as exc:
            logger.warning("IRBridge.build: canonicalization failed: %s", exc)
            return {
                "ir_nodes": [],
                "ir_graph": None,
                "graph_id": "",
                "ir_available": False,
            }

        # ── Step 5: populate spectral_signature on every query IRNode ──────
        packed_spectral = _pack_spectral_signature(spectral_scores)
        for node in nodes:
            try:
                sig = node.semantic_signature
                node.semantic_signature = SemanticSignature(
                    semantic_hash=sig.semantic_hash,
                    embedding_signature=sig.embedding_signature,
                    spectral_signature=packed_spectral,
                    predicate_family=sig.predicate_family,
                    abstraction_level=sig.abstraction_level,
                    canonical_form=sig.canonical_form,
                    equivalence_family=sig.equivalence_family,
                )
            except Exception as exc:
                logger.warning(
                    "IRBridge.build: could not set spectral_signature on node %s: %s",
                    getattr(node, "id", "?"), exc,
                )

        # ── Step 6: IRGraph construction ───────────────────────────────
        graph_id = _graph_id_from_nodes(nodes)
        now = datetime.datetime.utcnow().isoformat() + "Z"

        graph_confidence: float = 0.0
        if fused_scores:
            top_scores = sorted(fused_scores.values(), reverse=True)[:3]
            graph_confidence = sum(top_scores) / len(top_scores)

        try:
            fp_hashes = compute_graph_fingerprint_hashes(
                IRGraph(
                    graph_id=graph_id,
                    nodes=nodes,
                    edges=[],
                    fingerprint=GraphFingerprint(
                        semantic_hash="",
                        structural_hash="",
                        predicate_family_hash="",
                        temporal_signature="",
                        ontology_signature="",
                    ),
                    confidence_state=ConfidenceState(
                        overall_confidence=graph_confidence
                    ),
                    created_at=now,
                    updated_at=now,
                )
            )
            fingerprint = GraphFingerprint(
                semantic_hash=fp_hashes["semantic_hash"],
                structural_hash=fp_hashes["structural_hash"],
                predicate_family_hash=fp_hashes["predicate_family_hash"],
                temporal_signature=fp_hashes["temporal_signature"],
                ontology_signature=fp_hashes["ontology_signature"],
                canonicalization_version="v1.0",
            )
        except Exception as exc:
            logger.warning("IRBridge.build: fingerprint computation failed: %s", exc)
            fingerprint = GraphFingerprint(
                semantic_hash="",
                structural_hash="",
                predicate_family_hash="",
                temporal_signature="",
                ontology_signature="",
            )

        ir_graph = IRGraph(
            graph_id=graph_id,
            nodes=nodes,
            edges=[],
            fingerprint=fingerprint,
            state="DRAFT",
            version="v1",
            confidence_state=ConfidenceState(
                overall_confidence=graph_confidence
            ),
            created_at=now,
            updated_at=now,
        )

        return {
            "ir_nodes": nodes,
            "ir_graph": ir_graph,
            "graph_id": graph_id,
            "ir_available": True,
        }

    # ------------------------------------------------------------------
    # Convenience serialization helper
    # ------------------------------------------------------------------

    @staticmethod
    def nodes_to_dicts(nodes: List[Any]) -> List[Dict[str, Any]]:
        """Convert a list of IRNode dataclasses to plain dicts (for metadata)."""
        result: List[Dict[str, Any]] = []
        for node in nodes:
            try:
                result.append(asdict(node))
            except Exception:
                result.append({"id": getattr(node, "id", "?"), "error": "serialization_failed"})
        return result
