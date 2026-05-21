"""
lexis_bridge.py
───────────────
Subprocess bridge between Mycelium (.venv, Python 3.14) and Lexis-E (.venv2, Python 3.11).

All Lexis operations in Mycelium MUST go through this module.
No Lexis module is ever imported directly into Mycelium's process.

CLI reference (Lexis subcommands):
  compress   <input_file> <output_file> [--compact-context] [--compact-profile default|aggressive]
  decompress <input_file>                   → prints reconstructed text to stdout
  analyse    <input_file>                   → prints per-stage bpb stats to stdout
"""

import subprocess
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths -- resolved relative to this file so the bridge works from any cwd.
# ---------------------------------------------------------------------------
_ROOT         = Path(__file__).parent
LEXIS_PYTHON  = _ROOT / ".venv2" / "bin" / "python"
LEXIS_MAIN    = _ROOT / "lexis" / "main.py"


def _check_paths() -> None:
    """Raise a clear error early if the venv or Lexis source tree is missing."""
    if not LEXIS_PYTHON.exists():
        raise FileNotFoundError(
            f"Lexis venv not found at {LEXIS_PYTHON}. "
            "Run the Phase 1 setup: rename the Lexis Python 3.11 venv to .venv2 "
            "inside the Mycelium project root."
        )
    if not LEXIS_MAIN.exists():
        raise FileNotFoundError(
            f"Lexis main.py not found at {LEXIS_MAIN}. "
            "Copy or symlink the Lexis source tree into mycelium/lexis/."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def lexi_compress(
    text: str,
    output_path: str,
    compact: bool = True,
    profile: str = "default",
) -> None:
    """
    Write *text* to a temp file, compress it to *output_path* as a .lexi binary.

    Parameters
    ----------
    text        : raw input text to compress.
    output_path : destination path for the .lexi file (created or overwritten).
    compact     : if True, passes --compact-context to Lexis (recommended; ~47-52% overhead reduction).
    profile     : compact profile, either 'default' (k6s511) or 'aggressive'.

    Raises
    ------
    RuntimeError if Lexis returns a non-zero exit code.
    FileNotFoundError if the bridge paths are not configured.
    """
    _check_paths()

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(text)
        tmp_path = tmp.name

    cmd = [str(LEXIS_PYTHON), str(LEXIS_MAIN), "compress", tmp_path, output_path]
    if compact:
        cmd += ["--compact-context", "--compact-profile", profile]

    try:
        result = subprocess.run(cmd, capture_output=True)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Lexis compress failed (exit {result.returncode}):\n"
            f"{result.stderr.decode('utf-8', errors='replace')}"
        )


def lexi_decompress(lexi_path: str, max_chars: int | None = None) -> str:
    """
    Decompress a .lexi file and return the reconstructed text.

    Parameters
    ----------
    lexi_path : path to the .lexi binary file.
    max_chars : if set, truncate the output to this many characters before returning.
                Truncation is applied after full stdout capture -- there is no CLI flag
                for partial decompression. For very large files see the note in the
                integration plan (§Open Questions #5) about switching to Popen streaming.

    Returns
    -------
    Decompressed text string (optionally truncated).

    Raises
    ------
    RuntimeError if Lexis returns a non-zero exit code.
    FileNotFoundError if the bridge paths are not configured.
    """
    _check_paths()

    result = subprocess.run(
        [str(LEXIS_PYTHON), str(LEXIS_MAIN), "decompress", lexi_path],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Lexis decompress failed (exit {result.returncode}):\n"
            f"{result.stderr.decode('utf-8', errors='replace')}"
        )

    text = result.stdout.decode("utf-8", errors="replace")
    return text[:max_chars] if max_chars else text


def lexi_analyse(text_path: str) -> str:
    """
    Run analyse mode on an existing text file and return per-stage stats as a string.

    Parameters
    ----------
    text_path : path to an uncompressed text file (not a .lexi binary).

    Returns
    -------
    Stats string printed by the Lexis analyse subcommand (bpb per stage,
    POS Huffman summary, context-mixing model stats).

    Notes
    -----
    Useful for debugging patch creation.  Does NOT modify the file.
    """
    _check_paths()

    result = subprocess.run(
        [str(LEXIS_PYTHON), str(LEXIS_MAIN), "analyse", text_path],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Lexis analyse failed (exit {result.returncode}):\n"
            f"{result.stderr.decode('utf-8', errors='replace')}"
        )

    return result.stdout.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Convenience smoke-test (run directly: python lexis_bridge.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, textwrap

    sample = textwrap.dedent("""\
        The mitochondria is the powerhouse of the cell.
        Quantum entanglement describes correlations between particles that persist
        regardless of the distance separating them.
    """)

    out = "/tmp/mycelium_bridge_smoke_test.lexi"
    print("Compressing sample text ...")
    lexi_compress(sample, out, compact=True)
    print(f"  → written to {out}")

    print("Decompressing ...")
    recovered = lexi_decompress(out)
    print(f"  → recovered ({len(recovered)} chars):\n{recovered}")

    print("Bridge smoke-test PASSED.")
    sys.exit(0)
