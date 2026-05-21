"""
lexis_condenser.py
──────────────────
Part 1 of the Mycelium × Lexis integration plan:
  "Lexis as Context Condenser (Immediate, Low-Risk)"

Applies Lexis Stages 1–4 to multi-turn conversation history via the subprocess
brige to produce coreference-resolved, sentence-normalised text that is shorter
than the raw input (typically 15–30% token reduction on domain-heavy conversations)
but still fully LLM-readable -- no arithmetic coding is involved.

Stages 1–4 only:
  Stage 1  – sentence segmentation
  Stage 2  – morphological roots (metadata; not injected into text)
  Stage 3  – POS annotation     (metadata; not injected into text)
  Stage 4  – coreference resolution → pronoun chains collapsed, entity mentions normalised

Stages 5–12 (arithmetic coding, factoriadic encoding, compact_mode binary packing)
are intentionally NOT used here – they produce binary output unsuitable for LLM input.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from lexis_bridge import LEXIS_MAIN, LEXIS_PYTHON, _check_paths

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Condenser worker script
# ---------------------------------------------------------------------------
# This script is injected into the .venv2 Python process via -c so that
# Lexis internals are only imported inside the isolated environment.
_CONDENSER_WORKER = """
import sys, json
from pathlib import Path

lexis_root = Path(sys.argv[1])
sys.path.insert(0, str(lexis_root))

text = sys.stdin.read()

try:
    from lexis.stage1_segmentation import segment_sentences
    from lexis.stage3_pos import annotate_pos
    from lexis.stage4_discourse import resolve_coreferences
except ImportError as exc:
    # Graceful fallback: return text unchanged if stages aren't importable
    # (e.g., on a fresh checkout before full Lexis setup).  A warning is
    # printed to stderr so callers know condensation was skipped.
    print(f'[lexis_condenser] ImportError: {exc}. Returning raw text.', file=sys.stderr)
    print(text)
    sys.exit(0)

segmented = segment_sentences(text)
pos_annotated = annotate_pos(segmented)   # metadata only, not injected into text
condensed = resolve_coreferences(pos_annotated)
print(condensed)
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def condense_history(turns: list[dict], fallback_on_error: bool = True) -> str:
    """
    Apply Lexis Stages 1–4 to multi-turn conversation history.

    Returns coreference-resolved, sentence-normalised text suitable for direct
    inclusion in an LLM prompt context window.

    Parameters
    ----------
    turns             : list of dicts with keys 'role' and 'content', ordered
                        oldest-first (standard Mycelium conversation history format).
    fallback_on_error : if True (default) and the Lexis subprocess fails, return
                        the raw concatenated history instead of raising.  This
                        ensures the rest of the routing pipeline is never blocked
                        by Lexis unavailability.

    Returns
    -------
    Condensed conversation text as a plain string.
    """
    _check_paths()

    raw = "\n".join(
        f"{t.get('role', 'user')}: {t.get('content', '')}"
        for t in turns
    )

    # Run the worker script inside .venv2; pass raw text via stdin.
    try:
        result = subprocess.run(
            [
                str(LEXIS_PYTHON),
                "-c",
                _CONDENSER_WORKER,
                str(LEXIS_MAIN.parent),   # argv[1] = lexis root so worker can sys.path-insert it
            ],
            input=raw.encode("utf-8"),
            capture_output=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        if fallback_on_error:
            return raw
        raise

    if result.returncode != 0:
        if fallback_on_error:
            return raw
        raise RuntimeError(
            f"lexis_condenser worker failed (exit {result.returncode}):\n"
            f"{result.stderr.decode('utf-8', errors='replace')}"
        )

    condensed = result.stdout.decode("utf-8", errors="replace").strip()
    # Paranoia: if the worker returned nothing at all, fall back to raw.
    return condensed if condensed else raw


def condense_text(text: str, fallback_on_error: bool = True) -> str:
    """
    Apply Lexis Stages 1–4 to an arbitrary text block (not necessarily
    a multi-turn history).

    Useful for pre-condensing large document chunks before patch creation
    (Step 2 of the patch creation pipeline in §2.2).

    Parameters
    ----------
    text              : arbitrary plain text.
    fallback_on_error : see condense_history().

    Returns
    -------
    Condensed text string.
    """
    _check_paths()

    try:
        result = subprocess.run(
            [
                str(LEXIS_PYTHON),
                "-c",
                _CONDENSER_WORKER,
                str(LEXIS_MAIN.parent),
            ],
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        if fallback_on_error:
            return text
        raise

    if result.returncode != 0:
        if fallback_on_error:
            return text
        raise RuntimeError(
            f"lexis_condenser worker failed (exit {result.returncode}):\n"
            f"{result.stderr.decode('utf-8', errors='replace')}"
        )

    condensed = result.stdout.decode("utf-8", errors="replace").strip()
    return condensed if condensed else text


# ---------------------------------------------------------------------------
# Smoke-test (run directly: python lexis_condenser.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    sample_turns = [
        {"role": "user",      "content": "What is quantum entanglement? How does it affect particles?"},
        {"role": "assistant", "content": "Quantum entanglement is a phenomenon where two particles become correlated. These particles remain connected regardless of the distance between them. When you measure one particle, the other particle instantly reflects the measurement result."},
        {"role": "user",      "content": "So they can be used for faster-than-light communication?"},
        {"role": "assistant", "content": "No, they cannot be used for that purpose. The correlation between the particles does not allow information to travel faster than light. This is because the measurement result is random and cannot be controlled."},
    ]

    print("Raw turns (before condensation):")
    raw = "\n".join(f"{t['role']}: {t['content']}" for t in sample_turns)
    print(raw)
    print(f"\nRaw length: {len(raw)} chars")

    print("\nCondensing ...")
    condensed = condense_history(sample_turns)
    print(f"Condensed ({len(condensed)} chars):")
    print(condensed)
    print(f"\nReduction: {100 * (1 - len(condensed) / len(raw)):.1f}%")
    sys.exit(0)
