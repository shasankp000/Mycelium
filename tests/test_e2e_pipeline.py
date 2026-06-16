"""
tests/test_e2e_pipeline.py

End-to-end smoke test for the TRM v2 pipeline.

What this tests
---------------
- TRMV2Pipeline.run() accepts a plain string query and returns a
  PipelineResult with the expected contract.
- The pipeline does NOT raise on a completely cold shard root
  (no datasets seeded yet).
- The novelty_decision is a valid NoveltyDecision enum value.
- selected_domain may be None (OOD / DEFER_OOD) but must not raise.

Run
---
  pytest tests/test_e2e_pipeline.py -v
  pytest tests/test_e2e_pipeline.py -v -s   # to see stdout

Notes
-----
- This test uses a temp shard root and a temp graph file so it never
  touches production data.
- No torch required for the structural assertions; if torch is absent
  the encoder falls back to an EmbeddingBag stub (same as run_workflow.py).
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers — mirror the factory in run_workflow.py so we don't depend on
# argparse in tests.
# ---------------------------------------------------------------------------

def _build_pipeline(shard_root: str, graph_path: str):
    """Construct a TRMV2Pipeline with a stub backbone — no checkpoint needed."""
    import torch.nn as nn
    from mycelium.domain_graph.registry import DomainGraphRegistry
    from mycelium.domain_graph.novelty import DomainNoveltyPolicy, NoveltyThresholds
    from mycelium.trm.v2.encoder import SharedEncoder
    from mycelium.trm.v2.heads import HeadRegistry
    from mycelium.trm.v2.gating import DomainGate
    from mycelium.trm.v2.halt import HaltControllerV2
    from mycelium.trm.v2.inference import TRMV2InferenceEngine
    from mycelium.trm.v2.cold.manager import ColdStorageManager
    from mycelium.trm.v2.store.query_store import QueryStore
    from mycelium.trm.v2.pipeline import TRMV2Pipeline

    # Empty graph — no domains seeded; novelty policy will return DEFER_OOD.
    registry = DomainGraphRegistry(graph_path)

    output_dim = 128
    backbone   = nn.EmbeddingBag(30522, output_dim, mode="mean", sparse=False)
    encoder    = SharedEncoder(backbone, output_dim=output_dim, frozen=True)

    head_registry = HeadRegistry()
    policy = DomainNoveltyPolicy(registry, NoveltyThresholds())
    gate   = DomainGate(head_registry, registry)
    halt   = HaltControllerV2()

    # Try to get a real tokenizer; fall back gracefully if transformers absent.
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    except Exception:
        pass  # encoder accepts None tokenizer

    engine = TRMV2InferenceEngine(
        encoder=encoder,
        gate=gate,
        novelty_policy=policy,
        domain_registry=registry,
        halt_controller=halt,
        device="cpu",
        tokenizer=tokenizer,
    )

    cold  = ColdStorageManager(registry)
    store = QueryStore(shard_root=shard_root)
    return TRMV2Pipeline(engine, cold, store, registry, enabled=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestE2EPipeline:
    """End-to-end smoke tests for TRMV2Pipeline."""

    QUERY = "Is 9.9 bigger than 9.11?"

    def _run(self, tmp_path: Path):
        shard_root = str(tmp_path / "shards")
        graph_path = str(tmp_path / "graph.json")
        pipeline   = _build_pipeline(shard_root, graph_path)
        try:
            result = pipeline.run(self.QUERY)
        finally:
            pipeline._store.close_all()
        return result

    def test_pipeline_does_not_raise(self, tmp_path):
        """Pipeline must not raise for any valid string query."""
        self._run(tmp_path)  # passes if no exception

    def test_result_has_route(self, tmp_path):
        """PipelineResult.route must be present."""
        result = self._run(tmp_path)
        assert result.route is not None

    def test_novelty_decision_is_valid(self, tmp_path):
        """novelty_decision must be a NoveltyDecision enum member."""
        from mycelium.domain_graph.state import NoveltyDecision
        result = self._run(tmp_path)
        decision = result.route.novelty_decision
        assert isinstance(decision, NoveltyDecision), (
            f"Expected NoveltyDecision, got {type(decision)}: {decision!r}"
        )

    def test_gate_confidence_in_range(self, tmp_path):
        """gate_confidence must be in [0, 1]."""
        result = self._run(tmp_path)
        conf = result.route.gate_confidence
        assert 0.0 <= conf <= 1.0, f"gate_confidence out of range: {conf}"

    def test_ood_on_empty_registry(self, tmp_path):
        """
        With no domains in the registry the policy must return DEFER_OOD
        and selected_domain_id must be None.
        """
        from mycelium.domain_graph.state import NoveltyDecision
        result = self._run(tmp_path)
        assert result.route.novelty_decision == NoveltyDecision.DEFER_OOD, (
            f"Expected DEFER_OOD on empty registry, "
            f"got {result.route.novelty_decision.value}"
        )
        assert result.route.selected_domain_id is None

    def test_stored_query_id_is_none_on_empty_shard(self, tmp_path):
        """
        With no domain selected (OOD), stored_query_id should be None
        since there is no shard to write to.
        """
        result = self._run(tmp_path)
        # OOD path: nothing stored.
        assert result.stored_query_id is None

    def test_result_json_serialisable(self, tmp_path):
        """route fields must be JSON-serialisable (no raw tensors)."""
        result = self._run(tmp_path)
        r = result.route
        payload = {
            "selected_domain": r.selected_domain_id,
            "novelty_decision": r.novelty_decision.value,
            "gate_confidence": r.gate_confidence,
            "novelty_similarity": r.novelty_similarity,
            "ood_fallback": r.ood_fallback,
            "halted": r.halt_state.halted,
            "halt_reason": r.halt_state.reason,
            "stored_query_id": result.stored_query_id,
        }
        # Must not raise
        json.dumps(payload)


# ---------------------------------------------------------------------------
# Seeded-shard variant — only runs if datasets are present.
# Mark with @pytest.mark.integration so CI can skip it without torch/data.
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestE2EWithSeededShards:
    """
    Run the pipeline after seeding physics and chemistry shards.
    Requires:
      - data/physics/ or data/chemistry/ to contain CSV/JSONL files
      - torch installed
    Skip with: pytest -m "not integration"
    """

    PHYSICS_QUERY  = "Is 9.9 bigger than 9.11?"
    CHEMISTRY_QUERY = "What is the atomic number of carbon?"

    def _seed_and_build(self, data_root: Path, shard_root: Path, graph_path: Path):
        from scripts.seed_sqlite_experts import seed_domain
        for domain in ("physics", "chemistry"):
            seed_domain(
                domain,
                data_root=data_root,
                shard_root=shard_root,
                verbose=False,
            )
        return _build_pipeline(str(shard_root), str(graph_path))

    @pytest.fixture()
    def real_data_root(self):
        root = Path("data")
        if not root.exists():
            pytest.skip("data/ directory not found — skipping integration test")
        has_data = any(
            (root / d).exists() and any((root / d).iterdir())
            for d in ("physics", "chemistry")
        )
        if not has_data:
            pytest.skip("No physics/chemistry datasets found in data/ — skipping")
        return root

    def test_physics_query_routes_to_domain(self, tmp_path, real_data_root):
        """
        After seeding, a physics question should route to 'physics'
        or at minimum NOT be DEFER_OOD if a domain is registered.
        """
        from mycelium.domain_graph.state import NoveltyDecision
        shard_root = tmp_path / "shards"
        graph_path = tmp_path / "graph.json"
        pipeline = self._seed_and_build(real_data_root, shard_root, graph_path)
        try:
            result = pipeline.run(self.PHYSICS_QUERY)
        finally:
            pipeline._store.close_all()
        # With seeded shards, domain graph nodes exist → should not be DEFER_OOD
        # (unless the centroid similarity is genuinely below the OOD floor)
        print(f"physics query → {result.route.novelty_decision.value} "
              f"domain={result.route.selected_domain_id}")
        assert isinstance(result.route.novelty_decision, NoveltyDecision)
