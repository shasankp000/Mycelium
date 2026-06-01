"""
Regression test — fusion_engine.py must import tuning_config from the
correct package path (mycelium.trainers.tuning_config), not as a bare name.

Background: the bare `from tuning_config import ...` was silently caught by
an `except ImportError` block, causing fusion_engine to run with hardcoded
fallback defaults instead of the user's config.toml values.  This test
guards against that regression.
"""
import importlib
import sys
import types

import pytest


# ---------------------------------------------------------------------------
# 1. Module imports without error from a clean sys.modules state
# ---------------------------------------------------------------------------

def test_fusion_engine_importable():
    """fusion_engine must be importable without raising."""
    try:
        import mycelium.trainers.fusion_engine as fe  # noqa: F401
    except ImportError as exc:
        pytest.fail(f"fusion_engine could not be imported: {exc}")


# ---------------------------------------------------------------------------
# 2. FUSION_WEIGHTS comes from tuning_config, not the hardcoded fallback
# ---------------------------------------------------------------------------

def test_fusion_engine_reads_tuning_config_not_fallback():
    """
    FUSION_WEIGHTS must originate from mycelium.trainers.tuning_config,
    not from the except-branch hardcoded dict inside fusion_engine.py.

    Strategy: import tuning_config first, read its FUSION_WEIGHTS, then
    import fusion_engine and compare.  If they match the module value
    (not {'semantic': 0.4, 'spectral': 0.3, 'confidence': 0.3}) we know
    the correct path was resolved.
    """
    FALLBACK = {"semantic": 0.4, "spectral": 0.3, "confidence": 0.3}

    try:
        from mycelium.trainers import tuning_config as tc
        from mycelium.trainers import fusion_engine as fe
    except ImportError as exc:
        pytest.skip(f"Module not available in this environment: {exc}")

    # If tuning_config defines a non-fallback value, fusion_engine must match it.
    if hasattr(tc, "FUSION_WEIGHTS") and tc.FUSION_WEIGHTS != FALLBACK:
        assert fe.FUSION_WEIGHTS == tc.FUSION_WEIGHTS, (
            f"fusion_engine.FUSION_WEIGHTS {fe.FUSION_WEIGHTS!r} does not match "
            f"tuning_config.FUSION_WEIGHTS {tc.FUSION_WEIGHTS!r} — "
            "bare import path regression"
        )
    else:
        # tuning_config also uses the same default values; we can only confirm
        # the import did not raise and the module attribute exists.
        assert hasattr(fe, "FUSION_WEIGHTS"), "fusion_engine missing FUSION_WEIGHTS"


# ---------------------------------------------------------------------------
# 3. Bare name 'tuning_config' must NOT appear on sys.modules after import
#    (it would only be there if Python found it as a top-level module, which
#    would mean we're accidentally relying on the working directory)
# ---------------------------------------------------------------------------

def test_bare_tuning_config_not_in_sys_modules():
    """
    After importing fusion_engine, 'tuning_config' (bare name, no package
    prefix) must not be present in sys.modules.  Its presence would indicate
    that the bare import succeeded — meaning the cwd happens to be on
    sys.path, which masks the bug rather than fixing it.
    """
    try:
        import mycelium.trainers.fusion_engine  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"Module not available in this environment: {exc}")

    assert "tuning_config" not in sys.modules, (
        "'tuning_config' found as a top-level module in sys.modules after importing "
        "fusion_engine — the bare import path is being accidentally resolved via "
        "the working directory.  This masks the bug instead of fixing it."
    )


# ---------------------------------------------------------------------------
# 4. SUPERPOSITION_VARIANCE_THRESHOLD and ENABLE_LOGGING are present
# ---------------------------------------------------------------------------

def test_fusion_engine_exports_all_required_tuning_names():
    """All three names imported from tuning_config must be present on fusion_engine."""
    try:
        import mycelium.trainers.fusion_engine as fe
    except ImportError as exc:
        pytest.skip(f"Module not available in this environment: {exc}")

    for name in ("FUSION_WEIGHTS", "SUPERPOSITION_VARIANCE_THRESHOLD", "ENABLE_LOGGING"):
        assert hasattr(fe, name), (
            f"fusion_engine is missing {name!r} — tuning_config import may have failed silently"
        )
