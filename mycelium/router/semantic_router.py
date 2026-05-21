"""
Phase C — SemanticRouter
=========================
A thin wrapper around MultiLensRouter that enriches every RoutingResult
with IR-layer data produced by IRBridge.

Public contract
---------------
    SemanticRouter is a *strict superset* of MultiLensRouter:
    - route(text) returns the same RoutingResult dataclass
    - RoutingResult.metadata gains five new keys (ir_nodes, ir_graph_id,
      semantic_hash, canonical_form, predicate_family) but ALL existing
      keys are preserved unchanged
    - get_metrics() and all MultiLensRouter parameters work identically

IR enrichment added to RoutingResult.metadata:
    ir_nodes        : list[dict]  — one per SRL triple; each is an
                                     asdict(IRNode) for JSON-safe output
    ir_graph_id     : str         — stable ID of the DRAFT IRGraph for
                                     this request (empty string if IR
                                     stack unavailable)
    semantic_hash   : str         — semantic_hash of the primary IRNode
    canonical_form  : str         — canonical_form of the primary IRNode
    predicate_family: str         — predicate_family of the primary IRNode
    ir_available    : bool        — True when the full IR stack is present

The raw IRGraph object is NOT stored in metadata (not JSON-safe); callers
that need it can construct IRBridge and call build() directly.

Design note
-----------
    SemanticRouter does NOT replace MultiLensRouter.  Both live on
    the public path:
        from multi_lens_router import MultiLensRouter    # unchanged
        from mycelium import SemanticRouter              # Phase C
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from mycelium.core.types import RoutingResult
from .ir_bridge import IRBridge

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# Optional imports — graceful degradation preserved
# ---------------------------------------------------------------------------
try:
    from multi_lens_router import MultiLensRouter
    _ROUTER_AVAILABLE = True
except ImportError:
    MultiLensRouter = None  # type: ignore[assignment,misc]
    _ROUTER_AVAILABLE = False


class SemanticRouter:
    """MultiLensRouter + IRBridge: Phase C integrated routing pipeline.

    Parameters
    ----------
    All parameters are forwarded to MultiLensRouter unchanged.
    Additional:
        embedding_fn : callable, optional
            (str) -> list[float] embedding function for IRBridge.

    Example
    -------
    >>> router = SemanticRouter(use_multi_lens=True, use_spectral=True)
    >>> result = router.route("Smoking causes lung cancer.")
    >>> result.metadata["semantic_hash"]
    'a3f9...'
    >>> result.metadata["canonical_form"]
    'CAUSAL::smoking::cause::lung cancer::depth0'
    >>> result.metadata["predicate_family"]
    'CAUSAL'
    """

    def __init__(
        self,
        *,
        use_multi_lens: bool = True,
        use_spectral: bool = True,
        spectral_dir: str = "signatures",
        coverage_threshold: float = 0.8,
        max_experts: Optional[int] = None,
        spectral_analyzer: Optional[Any] = None,
        embedding_fn: Optional[Any] = None,
    ) -> None:
        self._embedding_fn = embedding_fn
        self._bridge = IRBridge(embedding_fn=embedding_fn)

        if not _ROUTER_AVAILABLE:
            logger.error(
                "SemanticRouter: multi_lens_router.py not found — "
                "all route() calls will return NO_EXPERT_AVAILABLE."
            )
            self._router: Optional[Any] = None
            return

        kwargs: Dict[str, Any] = dict(
            use_multi_lens=use_multi_lens,
            use_spectral=use_spectral,
            spectral_dir=spectral_dir,
            coverage_threshold=coverage_threshold,
        )
        if max_experts is not None:
            kwargs["max_experts"] = max_experts
        if spectral_analyzer is not None:
            kwargs["spectral_analyzer"] = spectral_analyzer

        try:
            self._router = MultiLensRouter(**kwargs)
        except Exception as exc:
            logger.error("SemanticRouter: failed to init MultiLensRouter: %s", exc)
            self._router = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, text: str) -> RoutingResult:
        """Run the full Phase C pipeline.

        Steps:
            1. Delegate to MultiLensRouter.route(text) — phases 1-4
            2. Extract spectral_scores from the result metadata
            3. Run IRBridge.build(text, ...) — phase C steps 5-6
            4. Inject IR artefacts into RoutingResult.metadata

        The returned RoutingResult is the SAME object produced by
        MultiLensRouter.route(); metadata is mutated in-place to add
        the five new IR-layer keys.
        """
        # ── Step 1: base routing ──────────────────────────────────────────
        if self._router is None:
            return RoutingResult(
                classification="NO_EXPERT_AVAILABLE",
                selected_domains=[],
                primary_domain=None,
                fusion_scores={},
                coverage=0.0,
                create_new_expert=True,
                metadata={
                    "selected_experts": [],
                    "candidate_domains": [],
                    "coverage_met": False,
                    "budget_cap_applied": False,
                    "lens_scores": {"semantic": {}, "spectral": {}, "confidence": {}},
                    "fused_scores": {},
                    "variance": 0.0,
                    "explanation": "SemanticRouter: underlying MultiLensRouter unavailable.",
                    "ir_nodes": [],
                    "ir_graph_id": "",
                    "semantic_hash": "",
                    "canonical_form": "",
                    "predicate_family": "",
                    "ir_available": False,
                },
            )

        result: RoutingResult = self._router.route(text)

        # ── Step 2: extract spectral scores from metadata ─────────────────
        spectral_scores: Dict[str, float] = {}
        fused_scores: Dict[str, float] = {}
        try:
            lens = result.metadata.get("lens_scores") or {}
            spectral_scores = dict(lens.get("spectral") or {})
            fused_scores = dict(result.metadata.get("fused_scores") or {})
        except Exception as exc:
            logger.debug("SemanticRouter: could not extract scores from metadata: %s", exc)

        # ── Step 3: run IR bridge (§46 steps 5-6) ────────────────────────
        ir_result: Dict[str, Any] = {}
        try:
            ir_result = self._bridge.build(
                text,
                spectral_scores=spectral_scores,
                fused_scores=fused_scores,
                source="phase_c_semantic_router",
            )
        except Exception as exc:
            logger.warning("SemanticRouter: IRBridge.build failed: %s", exc)
            ir_result = {
                "ir_nodes": [],
                "ir_graph": None,
                "graph_id": "",
                "ir_available": False,
            }

        # ── Step 4: inject IR artefacts into metadata ────────────────────
        ir_nodes = ir_result.get("ir_nodes") or []
        graph_id: str = ir_result.get("graph_id") or ""
        ir_available: bool = bool(ir_result.get("ir_available", False))

        # Extract primary node fields (first node = first SRL triple)
        semantic_hash = ""
        canonical_form = ""
        predicate_family = ""
        if ir_nodes:
            primary_node = ir_nodes[0]
            try:
                sig = primary_node.semantic_signature
                semantic_hash = sig.semantic_hash
                canonical_form = sig.canonical_form
                predicate_family = sig.predicate_family
            except Exception:
                pass

        result.metadata["ir_nodes"] = IRBridge.nodes_to_dicts(ir_nodes)
        result.metadata["ir_graph_id"] = graph_id
        result.metadata["semantic_hash"] = semantic_hash
        result.metadata["canonical_form"] = canonical_form
        result.metadata["predicate_family"] = predicate_family
        result.metadata["ir_available"] = ir_available

        return result

    # ------------------------------------------------------------------
    # Pass-through helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_metrics() -> Dict[str, Any]:
        """Delegate to MultiLensRouter.get_metrics()."""
        if not _ROUTER_AVAILABLE or MultiLensRouter is None:
            return {}
        return MultiLensRouter.get_metrics()

    @property
    def request_count(self) -> int:
        """Total requests processed by the underlying MultiLensRouter."""
        if self._router is None:
            return 0
        return getattr(self._router, "request_count", 0)
