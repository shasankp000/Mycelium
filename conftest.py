"""
conftest.py  —  project-root pytest configuration

Two responsibilities:

1. Stub out heavy optional dependencies (torch, transformers, sentence_transformers)
   BEFORE any collection-time import can trigger them.  This prevents the
   mycelium/__init__.py → mycelium.trm → TRMCell → torch import-chain from
   crashing the test runner when torch is not installed in the test environment.

2. Register custom pytest marks so pytest doesn't emit PytestUnknownMarkWarning.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# 1.  Stub heavy optional deps before any test module is imported
# ---------------------------------------------------------------------------

def _stub_module(name: str) -> MagicMock:
    """Create a MagicMock module and register it under *name* and all parents."""
    mock = MagicMock(name=name)
    sys.modules[name] = mock
    # Ensure parent packages are also registered so `import a.b.c` works
    parts = name.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        if parent not in sys.modules:
            parent_mock = MagicMock(name=parent)
            sys.modules[parent] = parent_mock
    return mock


_HEAVY_DEPS = [
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.optim",
    "torch.utils",
    "torch.utils.data",
    "transformers",
    "transformers.modeling_outputs",
    "sentence_transformers",
    "sentence_transformers.util",
    "sklearn",
    "sklearn.metrics",
    "sklearn.metrics.pairwise",
]

for _dep in _HEAVY_DEPS:
    if _dep not in sys.modules:
        _stub_module(_dep)

# torch.Tensor is referenced as a type in several modules — give it something
# plausible so isinstance() checks don't explode.
_torch_mock = sys.modules["torch"]
_torch_mock.Tensor = type("Tensor", (), {})
_torch_mock.nn.Module = type("Module", (), {"__init__": lambda self, *a, **kw: None})
_torch_mock.device = str  # device("cpu") → just a string in tests
_torch_mock.no_grad = MagicMock(return_value=MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False)))
_torch_mock.float32 = "float32"
_torch_mock.zeros = MagicMock(return_value=MagicMock())
_torch_mock.stack = MagicMock(return_value=MagicMock())


# ---------------------------------------------------------------------------
# 2.  Register custom marks
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "semantic: tests that verify semantic / behavioural correctness (not just syntax)",
    )
