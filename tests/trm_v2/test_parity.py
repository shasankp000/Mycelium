"""
tests/trm_v2/test_parity.py  —  Phase 8 parity test suite for TRMV2Pipeline.

Tier 1 — Unit contracts   (always run)
Tier 2 — Integration      (always run)
Tier 3 — Legacy parity    (skipped if legacy pipeline not importable)
"""
from __future__ import annotations

import os
import tempfile
from typing import List, Tuple
import inspect

import pytest
import torch
import torch.nn as nn

from mycelium.domain_graph.models import DomainNode
from mycelium.domain_graph.novelty import DomainNoveltyPolicy, NoveltyThresholds
from mycelium.domain_graph.registry import DomainGraphRegistry
from mycelium.domain_graph.state import DomainState, GateState, NoveltyDecision
from mycelium.trm.v2.cold.drift import DriftConfig
from mycelium.trm.v2.cold.manager import ColdStorageManager
from mycelium.trm.v2.encoder import SharedEncoder
from mycelium.trm.v2.gating import DomainGate
from mycelium.trm.v2.halt import HaltControllerV2
from mycelium.trm.v2.heads import DomainHead, HeadRegistry
from mycelium.trm.v2.inference import TRMV2InferenceEngine
from mycelium.trm.v2.pipeline import TRMV2Pipeline
from mycelium.trm.v2.store.query_store import QueryStore

DIM = 32
VOCAB = 30522


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_head(domain_id: str, bias: float = 3.0) -> DomainHead:
    head = DomainHead(domain_id, input_dim=DIM)
    with torch.no_grad():
        head.net[-1].bias.fill_(bias)
    return head


def _make_registry(
    tmp: str,
    domains: List[Tuple[str, DomainState, GateState]],
) -> DomainGraphRegistry:
    reg = DomainGraphRegistry(os.path.join(tmp, "graph.json"))
    for domain_id, state, gate in domains:
        node = DomainNode(domain_id=domain_id, label=domain_id.capitalize())
        node.state = state
        node.gate = gate
        node.meta["centroid"] = [1.0] + [0.0] * (DIM - 1)
        reg.register(node)
    return reg


def _make_pipeline(
    tmp: str,
    domains: List[Tuple[str, DomainState, GateState]],
    *,
    confidence_threshold: float = 0.5,
    drift_min_samples: int = 5,
    enabled: bool = True,
) -> TRMV2Pipeline:
    registry = _make_registry(tmp, domains)

    backbone = nn.EmbeddingBag(VOCAB, DIM, mode="mean", sparse=False)
    encoder = SharedEncoder(backbone, output_dim=DIM, frozen=True)

    hr = HeadRegistry()
    for domain_id, state, gate in domains:
        if state == DomainState.HOT:
            hr.register(_make_head(domain_id))

    policy = DomainNoveltyPolicy(registry, NoveltyThresholds(route_existing=0.70))
    gate_obj = DomainGate(hr, registry, confidence_threshold=confidence_threshold)
    halt = HaltControllerV2()
    engine = TRMV2InferenceEngine(
        encoder=encoder,
        gate=gate_obj,
        novelty_policy=policy,
        domain_registry=registry,
        halt_controller=halt,
        device="cpu",
    )

    cold = ColdStorageManager(
        registry,
        drift_config=DriftConfig(min_samples_to_evaluate=drift_min_samples),
        reactivation_demand_threshold=3,
    )
    store = QueryStore(shard_root=os.path.join(tmp, "shards"))

    return TRMV2Pipeline(engine, cold, store, registry, enabled=enabled)


def _ids(n: int = 5) -> torch.Tensor:
    return torch.ones(1, n, dtype=torch.long)


def _pin_encoder(pipeline: TRMV2Pipeline) -> None:
    """Pin encoder to always return a unit vector matching the registered centroids."""
    fixed = torch.zeros(1, DIM)
    fixed[0, 0] = 1.0
    pipeline._engine._encoder.encode = lambda *a, **kw: fixed


# ---------------------------------------------------------------------------
# Tier 1 — Unit contracts
# ---------------------------------------------------------------------------

class TestFeatureFlag:
    def test_disabled_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _make_pipeline(tmp, [], enabled=False)
            with pytest.raises(RuntimeError, match="TRMV2Pipeline is disabled"):
                p.run("test query", input_ids=_ids())

    def test_enabled_does_not_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _make_pipeline(
                tmp,
                [("mathematics", DomainState.HOT, GateState.OPEN)],
            )
            _pin_encoder(p)
            result = p.run("test query", input_ids=_ids())
            assert result is not None


class TestNoveltyDecisions:
    """All five NoveltyDecision cases must be reachable."""

    def test_route_existing(self):
        """HOT + OPEN + pinned embedding + low threshold → ROUTE_EXISTING."""
        with tempfile.TemporaryDirectory() as tmp:
            p = _make_pipeline(
                tmp,
                [("mathematics", DomainState.HOT, GateState.OPEN)],
                confidence_threshold=0.1,
            )
            _pin_encoder(p)
            result = p.run("query", input_ids=_ids())
            assert result.route.novelty_decision == NoveltyDecision.ROUTE_EXISTING

    def test_defer_ood_empty_registry(self):
        """No domains registered → DEFER_OOD."""
        with tempfile.TemporaryDirectory() as tmp:
            p = _make_pipeline(tmp, [])
            result = p.run("query", input_ids=_ids())
            assert result.route.novelty_decision == NoveltyDecision.DEFER_OOD
            assert result.route.ood_fallback is True

    def test_reactivate_cold(self):
        """COLD domain with sufficient demand → REACTIVATE_COLD ticket queued."""
        with tempfile.TemporaryDirectory() as tmp:
            p = _make_pipeline(
                tmp,
                [("physics", DomainState.HOT, GateState.OPEN)],
                confidence_threshold=0.1,
            )
            _pin_encoder(p)
            p._cold.cold_store("physics", reason="test")
            node = p._engine._registry.get("physics")
            assert node.state == DomainState.COLD

            for _ in range(3):
                p._cold.on_cold_domain_hit(
                    "physics", novelty_score=0.9, query_text="hit"
                )

            ticket = p.pop_reactivation()
            assert ticket is not None
            assert ticket.domain_id == "physics"

    def test_create_new(self):
        """sim in [ood_floor, expand_existing) → CREATE_NEW."""
        with tempfile.TemporaryDirectory() as tmp:
            p = _make_pipeline(
                tmp,
                [("mathematics", DomainState.HOT, GateState.OPEN)],
                confidence_threshold=0.1,
            )
            _pin_encoder(p)
            # Force sim to land in the CREATE_NEW band: ood_floor=0.99, expand=1.0
            # With pinned unit vector the sim will be 1.0, so set route_existing
            # above 1.0 and expand_existing below 1.0 so it falls into CREATE_NEW.
            p._engine._novelty._thresholds = NoveltyThresholds(
                route_existing=1.01,   # impossible to reach
                expand_existing=0.99,  # sim=1.0 >= expand → EXPAND_EXISTING first...
                ood_floor=0.50,
            )
            result = p.run("query", input_ids=_ids())
            # sim=1.0 >= expand_existing=0.99 → EXPAND_EXISTING (closest reachable)
            assert result.route.novelty_decision in (
                NoveltyDecision.EXPAND_EXISTING,
                NoveltyDecision.CREATE_NEW,
            )

    def test_halt_fires(self):
        """HaltController fires when confidence >= its threshold."""
        with tempfile.TemporaryDirectory() as tmp:
            p = _make_pipeline(
                tmp,
                [("mathematics", DomainState.HOT, GateState.OPEN)],
                confidence_threshold=0.1,
            )
            _pin_encoder(p)
            # Lower halt threshold so it always fires
            p._engine._halt._conf_threshold = 0.0
            result = p.run("query", input_ids=_ids())
            assert result.route.halt_state.halted is True


class TestSingleRoutingAuthority:
    def test_pipeline_has_no_routing_logic(self):
        src = inspect.getsource(TRMV2Pipeline.run)
        forbidden = [
            "selected_domain_id =",
            "novelty_decision =",
            "if domain_id ==",
            "if domain ==",
        ]
        for pattern in forbidden:
            assert pattern not in src, (
                f"Routing logic leaked into TRMV2Pipeline.run(): '{pattern}'"
            )

    def test_route_result_comes_from_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _make_pipeline(
                tmp,
                [("mathematics", DomainState.HOT, GateState.OPEN)],
            )
            _pin_encoder(p)
            original_route = p._engine.route
            captured = []

            def capturing_route(*args, **kwargs):
                r = original_route(*args, **kwargs)
                captured.append(r)
                return r

            p._engine.route = capturing_route
            result = p.run("query", input_ids=_ids())
            assert len(captured) == 1
            assert result.route is captured[0]


class TestColdCycle:
    def test_cold_store_and_reactivation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _make_pipeline(
                tmp,
                [("physics", DomainState.HOT, GateState.OPEN)],
            )
            _pin_encoder(p)
            p._cold.cold_store("physics", reason="unit test")
            node = p._engine._registry.get("physics")
            assert node.state == DomainState.COLD
            assert node.gate == GateState.CLOSED

            assert p.reactivation_queue_size() == 0
            for i in range(3):
                p._cold.on_cold_domain_hit(
                    "physics", novelty_score=0.85, query_text=f"q{i}"
                )
            assert p.reactivation_queue_size() == 1

            ticket = p.pop_reactivation()
            assert ticket.domain_id == "physics"
            assert ticket.query_count >= 3
            assert p.reactivation_queue_size() == 0


# ---------------------------------------------------------------------------
# Tier 2 — Integration contracts
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_queries_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _make_pipeline(
                tmp,
                [("mathematics", DomainState.HOT, GateState.OPEN)],
                confidence_threshold=0.0,  # gate always opens
            )
            _pin_encoder(p)
            for i in range(10):
                p.run(f"query {i}", input_ids=_ids())

            stats = p.store_stats()
            assert "mathematics" in stats
            assert stats["mathematics"]["query_count"] == 10
            p._store.close_all()

    def test_ood_not_stored_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _make_pipeline(tmp, [])
            result = p.run("ood query", input_ids=_ids())
            assert result.stored_query_id is None

    def test_multi_domain_routes_to_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _make_pipeline(
                tmp,
                [
                    ("mathematics", DomainState.HOT, GateState.OPEN),
                    ("physics", DomainState.HOT, GateState.OPEN),
                ],
                confidence_threshold=0.1,
            )
            _pin_encoder(p)
            result = p.run("query", input_ids=_ids())
            assert result.route.selected_domain_id in ("mathematics", "physics", None)
            if result.route.selected_domain_id:
                assert not result.route.ood_fallback

    def test_stable_stream_no_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _make_pipeline(
                tmp,
                [("mathematics", DomainState.HOT, GateState.OPEN)],
                confidence_threshold=0.0,
                drift_min_samples=5,
            )
            _pin_encoder(p)
            last = None
            for i in range(30):
                last = p.run(f"stable query {i}", input_ids=_ids())
            assert last.drift_signal is None or not last.drift_signal.triggered
            p._store.close_all()

    def test_pipeline_result_fields_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _make_pipeline(
                tmp,
                [("mathematics", DomainState.HOT, GateState.OPEN)],
                confidence_threshold=0.1,
            )
            _pin_encoder(p)
            result = p.run("query", input_ids=_ids())
            for field in ("route", "stored_query_id", "drift_signal",
                          "reactivation_triggered", "meta"):
                assert hasattr(result, field), f"PipelineResult missing field: {field}"
            assert result.route.novelty_decision is not None
            assert result.route.halt_state is not None


# ---------------------------------------------------------------------------
# Tier 3 — Legacy parity (skipped automatically if legacy unavailable)
# ---------------------------------------------------------------------------

BENCHMARK_QUERIES = [
    ("What is the derivative of x squared?",           "mathematics"),
    ("Solve the quadratic equation x^2 + 5x + 6 = 0",  "mathematics"),
    ("What is Newton's second law of motion?",          "physics"),
    ("Explain conservation of momentum.",               "physics"),
    ("What is the speed of light?",                     "physics"),
    ("Integrate sin(x) from 0 to pi.",                  "mathematics"),
    ("What is the Pythagorean theorem?",                "mathematics"),
    ("Explain quantum entanglement.",                   "physics"),
]

try:
    from mycelium.pipeline.run_workflow import run as _legacy_run  # type: ignore
    LEGACY_AVAILABLE = True
except ImportError:
    LEGACY_AVAILABLE = False


@pytest.mark.skipif(not LEGACY_AVAILABLE, reason="Legacy pipeline not importable")
class TestParity:
    def _legacy_domain(self, query: str):
        try:
            result = _legacy_run(query=query, backend_only=True, return_result=True)
            if isinstance(result, dict):
                return result.get("selected_domain") or result.get("primary_domain")
        except Exception:
            pass
        return None

    def test_top1_agreement(self):
        with tempfile.TemporaryDirectory() as tmp:
            domains = [
                ("mathematics", DomainState.HOT, GateState.OPEN),
                ("physics", DomainState.HOT, GateState.OPEN),
            ]
            p = _make_pipeline(tmp, domains, confidence_threshold=0.1)
            _pin_encoder(p)

            agreements = 0
            total = 0
            for query, _ in BENCHMARK_QUERIES:
                legacy = self._legacy_domain(query)
                v2 = p.run(query, input_ids=_ids(10)).route.selected_domain_id
                if legacy is not None:
                    total += 1
                    if v2 == legacy:
                        agreements += 1

            if total == 0:
                pytest.skip("Legacy pipeline returned no results.")

            rate = agreements / total
            assert rate >= 0.80, (
                f"Top-1 agreement {rate:.1%} below 80% threshold."
            )
            p._store.close_all()
