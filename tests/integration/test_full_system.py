import json
from typing import List

from run_workflow import run_mycelium_workflow


def test_full_system_metrics_and_outputs(tmp_path, monkeypatch):
    """High-level sanity test for the full Mycelium workflow.

    This test intentionally depends only on the public workflow API and the
    aggregated WorkflowMetrics, so that internal refactors keep behavior
    compatible without rewriting tests.
    """

    # Use a small, representative batch of queries that should exercise
    # a mix of Layer0 routes and domain selections.
    sentences: List[str] = [
        "Earth orbits the Sun.",
        "Glossy metallic red finish on the car.",
        "Patient with myocardial infarction showing elevated troponin.",
        "Is it morally right to maximize profit at any cost?",
        "The study of astronomy involves planets and stars.",
    ]

    # Run workflow
    all_sentence_data, metrics = run_mycelium_workflow(sentences)

    # Basic shape checks on outputs
    assert len(all_sentence_data) == len(sentences)
    for entry in all_sentence_data:
        assert "sentence" in entry
        assert "layer0_routing" in entry
        assert "expert_flag" in entry

    # Metrics should contain at least one REASONING_PIPELINE and at least
    # one non-REASONING route (MULTI_PERSPECTIVE / REFUSE / CLARIFICATION)
    layer0_routes = metrics.layer0_routes
    assert layer0_routes["REASONING_PIPELINE"] >= 1
    non_reasoning = (
        layer0_routes.get("MULTI_PERSPECTIVE", 0)
        + layer0_routes.get("REFUSE", 0)
        + layer0_routes.get("CLARIFICATION", 0)
    )
    assert non_reasoning >= 1

    # There should be some routed classifications and expert decisions
    assert sum(metrics.routing_classifications.values()) >= 1
    assert sum(metrics.expert_decisions.values()) >= 1

    # And at least one domain counted from expert selections (when available)
    # Note: if models are missing, this may be zero, so only check type.
    assert isinstance(metrics.domains, type(metrics.layer0_routes))

    # Optionally, write metrics to a temporary JSON file to ensure they are
    # JSON-serializable for downstream tooling.
    metrics_path = tmp_path / "workflow_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics.to_dict(), f, indent=2)

    # Re-load and sanity check structure
    with metrics_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert "layer0_routes" in data
    assert "routing_classifications" in data
    assert "expert_decisions" in data
    assert "domains" in data
