"""
Phase D — TRMLens
==================
Backward-compatible shim that replaces the local LLM call in the TRM
layer with the Samsung TRM neural reasoner (arXiv:2510.04871).

Drop-in replacement
-------------------
Anywhere in web-ui-prototype (run_workflow.py, conversation_agent.py,
layer_1_prototype.py) that previously called the local LLM to synthesise
an answer from fused router output, replace with:

    # Before (local LLM):
    answer = llm.generate(trm_prompt)

    # After (TRM):
    result = trm_lens.refine(semantic_router_output)
    # result.primary_domain is now TRM-refined
    # result.metadata["trm_domain_probs"]    — refined softmax
    # result.metadata["trm_halt_confidence"] — answer stability [0,1]
    # result.metadata["trm_steps_taken"]     — recursion depth used

The TRMLens does NOT replace the downstream LLM that generates the
user-facing response text — it only refines the domain routing decision
that determines *which* expert/context is used for that generation.

Fallback behaviour
------------------
If TRMConfig.fallback_to_router is True (default) and the model file is
missing or torch is unavailable, TRMLens gracefully passes through the
raw MultiLensRouter scores unchanged.  This ensures the system continues
to function during development before the TRM checkpoint is trained.

Pre-TRM pipeline fallback
-------------------------
When the TRM checkpoint has not yet been trained, call
    trm_lens.run_pre_trm_pipeline(query, routing_context, ood_fallback)
to run the full L1-L6 reasoning chain (CanonicalizeAndHash → DAGDecomposer
→ Predicate/Evidence/Hypothesis/Synthesizer → MultiLensRouter) unconditionally.
This does NOT require a TRMOutput — the absence of the checkpoint is itself
treated as the OOD trigger.  The call returns an OODFallbackResult (or None
on hard failure) and is idempotent with the trained path: once the checkpoint
is available, the normal trm_reasoner → TRMOODFallback.should_trigger() path
takes over and run_pre_trm_pipeline() is no longer called.

Token encoding
--------------
By default TRMLens uses a simple whitespace-tokeniser with a fixed
8192-token vocabulary built from the canonical query vocabulary.
For production, pass a `tokenize_fn` callable: str → list[int].
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Optional torch import — system stays functional if torch missing
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from .config import TRMConfig
    from .reasoner import TRMReasoner, TRMOutput
    _REASONER_AVAILABLE = True
except ImportError:
    TRMConfig = None   # type: ignore[assignment,misc]
    TRMReasoner = None # type: ignore[assignment]
    TRMOutput = None   # type: ignore[assignment]
    _REASONER_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Simple fallback tokeniser                                                   #
# --------------------------------------------------------------------------- #

_VOCAB: Dict[str, int] = {}  # built lazily on first call
_VOCAB_SIZE = 8192


def _default_tokenize(text: str, max_len: int) -> List[int]:
    """Whitespace tokeniser with a lazy-built hash vocabulary."""
    tokens = text.lower().split()
    ids = []
    for tok in tokens:
        if tok not in _VOCAB:
            if len(_VOCAB) < _VOCAB_SIZE - 1:
                _VOCAB[tok] = len(_VOCAB) + 1  # 0 reserved for PAD
            else:
                _VOCAB[tok] = hash(tok) % (_VOCAB_SIZE - 1) + 1
        ids.append(_VOCAB[tok])
    # Pad / truncate to max_len
    ids = ids[:max_len]
    ids += [0] * (max_len - len(ids))
    return ids


# --------------------------------------------------------------------------- #
# TRMLens                                                                     #
# --------------------------------------------------------------------------- #

class TRMLens:
    """
    Phase D integration shim — wraps TRMReasoner with the SemanticRouter
    interface from Phase C.

    Parameters
    ----------
    cfg          : TRMConfig  (optional; defaults constructed if None)
    tokenize_fn  : callable str → list[int], optional
                   Custom tokeniser.  Defaults to simple whitespace hash vocab.
    """

    def __init__(
        self,
        cfg: Optional[Any] = None,
        tokenize_fn: Optional[Callable[[str, int], List[int]]] = None,
    ) -> None:
        self._available = _TORCH_AVAILABLE and _REASONER_AVAILABLE
        self._model: Optional[Any] = None
        self._cfg: Optional[Any] = cfg
        self._tokenize_fn = tokenize_fn or _default_tokenize

        if not self._available:
            logger.warning(
                "TRMLens: torch or reasoner unavailable — fallback mode active."
            )
            return

        if cfg is None:
            cfg = TRMConfig()
            self._cfg = cfg

        # Only build model if a checkpoint exists
        if cfg.model_path:
            try:
                self._model = TRMReasoner(cfg).to(cfg.device).eval()
                ckpt = torch.load(cfg.model_path, map_location=cfg.device)
                state = ckpt.get("ema_state") or ckpt.get("model_state") or ckpt
                self._model.load_state_dict(state)
                logger.info("TRMLens: loaded TRM checkpoint from %s", cfg.model_path)
            except Exception as exc:
                logger.warning("TRMLens: failed to load checkpoint (%s) — fallback mode.", exc)
                self._model = None
        else:
            logger.info(
                "TRMLens: no model_path configured — fallback mode (raw router scores used)."
            )

    # ------------------------------------------------------------------ #
    # Main API                                                             #
    # ------------------------------------------------------------------ #

    def refine(self, semantic_router_output: Any) -> Any:
        """
        Accepts a SemanticRouter RoutingResult (Phase C) and returns an
        enriched copy with TRM-refined domain probabilities.

        If TRM is unavailable or not trained, returns the input unchanged
        (fallback behaviour controlled by TRMConfig.fallback_to_router).

        Parameters
        ----------
        semantic_router_output : RoutingResult
            Must have:
                .query_text         str
                .domain_scores      dict[str, float]   raw fused scores
                .metadata           dict               Phase C IR data

        Returns
        -------
        RoutingResult  (same type, mutated metadata)
        """
        if self._model is None or not self._available:
            return self._fallback(semantic_router_output)

        try:
            return self._run_trm(semantic_router_output)
        except Exception as exc:
            logger.warning("TRMLens.refine: TRM inference failed (%s) — fallback.", exc)
            return self._fallback(semantic_router_output)

    # ------------------------------------------------------------------ #
    # Pre-TRM pipeline fallback  (no checkpoint required)                 #
    # ------------------------------------------------------------------ #

    def run_pre_trm_pipeline(
        self,
        query: str,
        routing_context: Any,
        ood_fallback: Any,
        domain: str = "unknown",
        expert_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """
        Run the full L1-L6 reasoning chain unconditionally via
        TRMOODFallback, without requiring a trained TRM checkpoint or
        a TRMOutput object.

        This is the pre-TRM wiring path: the absence of a checkpoint is
        treated as a guaranteed OOD signal, so we skip
        TRMOODFallback.should_trigger() entirely and call .route()
        directly at LEVEL_2 (or the highest level the injected components
        support).

        The method is a no-op (returns None) if ood_fallback is None,
        so callers don't need to guard against a missing TRMOODFallback.

        Parameters
        ----------
        query : str
            Raw query text for this turn.
        routing_context : RoutingResult
            Output of MultiLensRouter.route() already refined by
            TRMLens.refine() (which is a passthrough here).
        ood_fallback : TRMOODFallback
            A fully constructed TRMOODFallback instance (canonicalizer,
            dag_decomposer, and router must be injected for L1+ to fire).
            Pass None to skip silently.
        domain : str
            Hint domain from the router's primary_domain or classification.
        expert_metadata : dict, optional
            Any extra metadata to pass through to the fallback chain.

        Returns
        -------
        OODFallbackResult | None
            The result of the fallback chain, or None if ood_fallback is
            None or a hard exception occurs.
        """
        if ood_fallback is None:
            logger.debug(
                "TRMLens.run_pre_trm_pipeline: ood_fallback is None — skipping."
            )
            return None

        # Derive best domain hint from routing context if not supplied
        if domain == "unknown":
            domain = (
                getattr(routing_context, "primary_domain", None)
                or getattr(routing_context, "classification", None)
                or "unknown"
            )

        try:
            logger.info(
                "TRMLens.run_pre_trm_pipeline: invoking L1-L6 chain "
                "(pre-TRM mode, no checkpoint) for query=%r domain=%s",
                query[:80],
                domain,
            )
            # Bypass should_trigger(): call route() directly.
            # TRMOODFallback.route() internally calls should_trigger()
            # which needs a real TRMOutput, so we use the internal
            # _run_layers_1_2 / _run_layers_3_6 / _call_router path
            # by constructing a minimal forced-trigger invocation.
            result = _run_ood_chain_no_trm(
                ood_fallback=ood_fallback,
                query=query,
                domain=str(domain),
                expert_metadata=expert_metadata or {},
            )
            logger.info(
                "TRMLens.run_pre_trm_pipeline: completed — "
                "level=%s selected_domain=%s",
                result.fallback_level if result is not None else "N/A",
                result.selected_domain if result is not None else "N/A",
            )
            return result
        except Exception as exc:
            logger.warning(
                "TRMLens.run_pre_trm_pipeline: chain failed (%s) — "
                "returning None (pipeline continues via router scores).",
                exc,
            )
            return None

    # ------------------------------------------------------------------ #
    # Internal: TRM inference path                                         #
    # ------------------------------------------------------------------ #

    def _run_trm(self, router_out: Any) -> Any:
        cfg = self._cfg
        import torch  # guaranteed available here

        query_text = getattr(router_out, "query_text", "") or ""
        domain_scores: Dict[str, float] = getattr(router_out, "domain_scores", {}) or {}
        metadata: Dict[str, Any] = dict(getattr(router_out, "metadata", {}) or {})

        # Build ordered domain list from config or from router scores
        domain_names: List[str] = list(domain_scores.keys())
        n_domains = cfg.n_domains

        # --- token_ids ---
        token_ids = self._tokenize_fn(query_text, cfg.context_len)
        token_ids_t = torch.tensor([token_ids], dtype=torch.long, device=cfg.device)  # [1, L]

        # --- spectral_vec ---
        scores = [domain_scores.get(d, 0.0) for d in domain_names[:n_domains]]
        # Pad to n_domains if router returned fewer domains
        while len(scores) < n_domains:
            scores.append(0.0)
        spectral_t = torch.tensor([scores], dtype=torch.float32, device=cfg.device)  # [1, n]

        # --- predicate_family_id ---
        pred_id = metadata.get("predicate_family_id", 0)
        pred_t = torch.tensor([pred_id], dtype=torch.long, device=cfg.device)  # [1]

        # --- initial_domain_probs (normalised router scores) ---
        total = sum(scores) or 1.0
        probs = [s / total for s in scores]
        probs_t = torch.tensor([probs], dtype=torch.float32, device=cfg.device)  # [1, n]

        # --- Run TRM ---
        with torch.no_grad():
            out: TRMOutput = self._model(
                token_ids=token_ids_t,
                spectral_vec=spectral_t,
                predicate_family_id=pred_t,
                initial_domain_probs=probs_t,
            )

        # --- Write refined results back into metadata ---
        refined_probs = out.domain_probs.squeeze(0).tolist()  # list[float]
        metadata["trm_domain_probs"]    = dict(zip(domain_names[:n_domains], refined_probs))
        metadata["trm_halt_confidence"] = float(out.halt_confidence.squeeze(0).item())
        metadata["trm_steps_taken"]     = out.n_steps_taken
        metadata["trm_primary_domain"]  = (
            domain_names[out.primary_domain_idx.item()]
            if out.primary_domain_idx.item() < len(domain_names)
            else "unknown"
        )

        # Mutate the RoutingResult in-place if it supports it; otherwise
        # create a shallow copy via dataclasses.replace if available.
        try:
            import dataclasses
            if dataclasses.is_dataclass(router_out):
                return dataclasses.replace(
                    router_out,
                    primary_domain=metadata["trm_primary_domain"],
                    metadata=metadata,
                )
        except Exception:
            pass

        # Fallback: set attributes directly (works for SimpleNamespace / custom classes)
        try:
            router_out.primary_domain = metadata["trm_primary_domain"]
            router_out.metadata = metadata
        except AttributeError:
            pass

        return router_out

    # ------------------------------------------------------------------ #
    # Fallback path                                                        #
    # ------------------------------------------------------------------ #

    def _fallback(self, router_out: Any) -> Any:
        """Pass-through: annotate metadata to indicate TRM was not used."""
        try:
            meta = dict(getattr(router_out, "metadata", {}) or {})
            meta["trm_domain_probs"]    = None
            meta["trm_halt_confidence"] = None
            meta["trm_steps_taken"]     = 0
            meta["trm_primary_domain"]  = getattr(router_out, "primary_domain", "unknown")
            try:
                router_out.metadata = meta
            except AttributeError:
                pass
        except Exception:
            pass
        return router_out


# --------------------------------------------------------------------------- #
# Module-level helper: run the OOD chain without a TRMOutput               #
# --------------------------------------------------------------------------- #

def _run_ood_chain_no_trm(
    ood_fallback: Any,
    query: str,
    domain: str,
    expert_metadata: Dict[str, Any],
) -> Any:
    """
    Execute TRMOODFallback's internal L1-L6 chain directly, bypassing
    the should_trigger() heuristic that requires a live TRMOutput.

    Escalation level is chosen by what components are wired into
    ood_fallback:
        - canonicalizer AND dag_decomposer present  → LEVEL_2 attempted
        - only canonicalizer present                → LEVEL_1
        - neither                                   → LEVEL_0 (router only)

    Always returns an OODFallbackResult dataclass.  Never raises.
    """
    try:
        from .trm_ood_fallback import FallbackLevel, OODFallbackResult
    except ImportError as exc:
        logger.error("_run_ood_chain_no_trm: cannot import fallback types (%s)", exc)
        return None

    # Determine achievable level
    has_canon = getattr(ood_fallback, "canonicalizer", None) is not None
    has_dag   = getattr(ood_fallback, "dag_decomposer", None) is not None
    has_l3    = getattr(ood_fallback, "_l3", None) is not None
    has_l4    = getattr(ood_fallback, "_l4", None) is not None
    has_l5    = getattr(ood_fallback, "_l5", None) is not None
    has_l6    = getattr(ood_fallback, "_l6", None) is not None
    has_full_chain = has_l3 or has_l4 or has_l5 or has_l6

    if has_canon and has_dag:
        level = FallbackLevel.LEVEL_2 if has_full_chain else FallbackLevel.LEVEL_1
    elif has_canon:
        level = FallbackLevel.LEVEL_1
    else:
        level = FallbackLevel.LEVEL_0

    ir_graph:         Any               = None
    dag_ctx:          Optional[Dict]    = None
    reasoning_trace:  Optional[Dict]    = None

    # Layer 1-2
    if level >= FallbackLevel.LEVEL_1:
        ir_graph, dag_ctx, level = ood_fallback._run_layers_1_2(
            query, domain, expert_metadata, level
        )

    # Layers 3-6
    if level >= FallbackLevel.LEVEL_2 and dag_ctx is not None:
        reasoning_trace = ood_fallback._run_layers_3_6(query, domain, dag_ctx)

    # Terminal: MultiLensRouter
    router_result   = ood_fallback._call_router(query, dag_ctx)
    selected_domain = _extract_domain_idx(router_result)

    return OODFallbackResult(
        triggered=True,
        fallback_level=level,
        ood_head_confidence=0.0,   # no OOD head without TRMOutput
        selected_domain=selected_domain,
        router_result=router_result,
        ir_graph=ir_graph,
        dag_context=dag_ctx,
        reasoning_trace=reasoning_trace,
        reason="pre-trm-mode: checkpoint not yet trained",
    )


def _extract_domain_idx(router_result: Any) -> int:
    """Pull a domain index from whatever the router returned; default 0."""
    for attr in ("selected_domain", "domain_idx", "domain_index"):
        val = getattr(router_result, attr, None)
        if val is not None:
            return int(val)
    if isinstance(router_result, dict):
        for key in ("selected_domain", "domain_idx", "domain_index"):
            if key in router_result:
                return int(router_result[key])
    return 0
