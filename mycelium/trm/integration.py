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
