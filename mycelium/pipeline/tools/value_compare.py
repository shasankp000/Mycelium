"""
mycelium/pipeline/tools/value_compare.py
=========================================
Deterministic value comparison tool for Mycelium COMPARATIVE predicates.

Supports:
  - int   : exact integer comparison
  - float : Decimal-exact comparison (avoids IEEE-754 traps like 9.9 vs 9.11)
  - str   : three modes:
              literal  - exact string equality (case-insensitive)
              fuzzy    - SequenceMatcher ratio  (0.0-1.0)
              semantic - cosine similarity via ModelRegistry all-MiniLM-L6-v2
                         (falls back to fuzzy if encoder unavailable)

No LLM is invoked anywhere in this module.
"""
from __future__ import annotations

import logging
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from typing import Any, Dict, Literal, Optional, Union

logger = logging.getLogger(__name__)

ValueType = Literal["int", "float", "str"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coerce_int(v: Any) -> int:
    if isinstance(v, int):
        return v
    try:
        return int(str(v).strip())
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Cannot coerce {v!r} to int: {exc}") from exc


def _coerce_decimal(v: Any) -> Decimal:
    if isinstance(v, Decimal):
        return v
    try:
        return Decimal(str(v).strip())
    except (InvalidOperation, TypeError) as exc:
        raise ValueError(f"Cannot coerce {v!r} to Decimal: {exc}") from exc


def _fuzzy_ratio(a: str, b: str) -> float:
    """SequenceMatcher similarity ratio in [0.0, 1.0]."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _semantic_similarity(a: str, b: str) -> float:
    """
    Cosine similarity via ModelRegistry all-MiniLM-L6-v2.
    Falls back to fuzzy ratio when encoder is unavailable.
    """
    try:
        from model_registry import ModelRegistry
        import numpy as np
        registry = ModelRegistry.instance()
        model = registry.get("all-MiniLM-L6-v2")
        emb = model.encode([a, b], convert_to_numpy=True, normalize_embeddings=True)
        score = float(np.dot(emb[0], emb[1]))
        return max(0.0, min(1.0, score))
    except Exception as exc:
        logger.warning(
            "value_compare: semantic encoder unavailable (%s); "
            "falling back to fuzzy ratio.", exc,
        )
        return _fuzzy_ratio(a, b)


# ---------------------------------------------------------------------------
# Core comparison functions
# ---------------------------------------------------------------------------

def _compare_int(a: Any, b: Any) -> Dict[str, Any]:
    ia, ib = _coerce_int(a), _coerce_int(b)
    return {
        "type": "int",
        "a": ia,
        "b": ib,
        "a_gt_b": ia > ib,
        "a_lt_b": ia < ib,
        "a_eq_b": ia == ib,
        "normalized": f"{ia} {'>' if ia > ib else '<' if ia < ib else '='} {ib}",
        "difference": ia - ib,
    }


def _compare_float(a: Any, b: Any) -> Dict[str, Any]:
    da, db = _coerce_decimal(a), _coerce_decimal(b)
    diff = da - db
    return {
        "type": "float",
        "a": float(da),
        "b": float(db),
        "a_decimal": str(da),
        "b_decimal": str(db),
        "a_gt_b": da > db,
        "a_lt_b": da < db,
        "a_eq_b": da == db,
        "normalized": f"{da} {'>' if da > db else '<' if da < db else '='} {db}",
        "difference": float(diff),
        "difference_exact": str(diff),
    }


def _compare_str(a: Any, b: Any) -> Dict[str, Any]:
    sa, sb = str(a), str(b)
    fuzzy = _fuzzy_ratio(sa, sb)
    semantic = _semantic_similarity(sa, sb)
    literal_eq = sa.lower() == sb.lower()
    return {
        "type": "str",
        "a": sa,
        "b": sb,
        "literal_equal": literal_eq,
        "fuzzy_similarity": round(fuzzy, 4),
        "semantic_similarity": round(semantic, 4),
        "fuzzy_match": fuzzy >= 0.85,
        "semantic_match": semantic >= 0.80,
    }


# ---------------------------------------------------------------------------
# Auto-detect type
# ---------------------------------------------------------------------------

def _detect_type(a: str, b: str) -> ValueType:
    """
    Infer comparison type from raw string representations.
    Priority: int > float > str.  Both values must agree; else str.
    """
    def _is_int(v: str) -> bool:
        try:
            int(v.strip())
            return True
        except ValueError:
            return False

    def _is_float(v: str) -> bool:
        try:
            Decimal(v.strip())
            return True
        except InvalidOperation:
            return False

    sa, sb = str(a).strip(), str(b).strip()
    if _is_int(sa) and _is_int(sb):
        return "int"
    if _is_float(sa) and _is_float(sb):
        return "float"
    return "str"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compare_values(
    a: Union[str, int, float],
    b: Union[str, int, float],
    value_type: Optional[ValueType] = None,
) -> Dict[str, Any]:
    """
    Compare two values deterministically.

    Args:
        a:          First value (subject).
        b:          Second value (object).
        value_type: Force a specific type. If None, auto-detected.

    Returns:
        int/float keys: a, b, a_gt_b, a_lt_b, a_eq_b, normalized, difference
        str keys:       a, b, literal_equal, fuzzy_similarity,
                        semantic_similarity, fuzzy_match, semantic_match

    Raises:
        ValueError: coercion failed for requested type.
        TypeError:  value_type not in {"int", "float", "str"}.
    """
    resolved_type: ValueType = value_type or _detect_type(str(a), str(b))
    logger.debug(
        "value_compare: a=%r b=%r type=%s (explicit=%s)",
        a, b, resolved_type, value_type is not None,
    )
    if resolved_type == "int":
        return _compare_int(a, b)
    if resolved_type == "float":
        return _compare_float(a, b)
    if resolved_type == "str":
        return _compare_str(a, b)
    raise TypeError(
        f"value_compare: unsupported value_type={resolved_type!r}. "
        "Must be 'int', 'float', or 'str'."
    )


# ---------------------------------------------------------------------------
# mcp_tools_server adapter
# ---------------------------------------------------------------------------

def _tool_handler(expression: str) -> Dict[str, Any]:
    """
    MCP tool handler.

    expression format: "<a> vs <b>"  or  "<a>, <b>"
    e.g.  "9.9 vs 9.11"  |  "hello, world"  |  "42 vs 7"

    Returns the compare_values dict, or an error dict on bad input.
    """
    for sep in (" vs ", ", ", " VS ", " VS. "):
        if sep in expression:
            parts = expression.split(sep, 1)
            a, b = parts[0].strip(), parts[1].strip()
            break
    else:
        return {
            "error": (
                f"value_compare: could not parse expression {expression!r}. "
                "Expected format: '<a> vs <b>' or '<a>, <b>'."
            )
        }
    try:
        return compare_values(a, b)
    except Exception as exc:
        return {"error": f"value_compare failed: {exc}"}
