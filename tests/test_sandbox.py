"""tests/test_sandbox.py  (Milestone 6 — sandbox unit tests)

Tests:
  1. SandboxTask round-trip serialisation
  2. SandboxStep round-trip serialisation
  3. SandboxResult round-trip serialisation
  4. build_sandbox_task_from_run with a minimal run-summary stub dict
  5. _calculator tool via _TOOL_REGISTRY (no network)
  6. SandboxManager._stub_result path
  7. SandboxManager._execute_step with unknown tool (no network)
"""

from __future__ import annotations

from datetime import datetime, timezone


from sandbox_models import (
    SandboxResult,
    SandboxStep,
    SandboxTask,
    build_sandbox_task_from_run,
)
from sandbox_manager import SandboxManager, _TOOL_REGISTRY


# ---------------------------------------------------------------------------
# 1-3. Pydantic round-trips
# ---------------------------------------------------------------------------


class TestSandboxModelsSerialisation:
    def test_task_round_trip(self):
        task = SandboxTask(
            trace_id="abc-123",
            user_query="What is the Higgs boson?",
            domains=["physics"],
            hypotheses=["It is the particle that gives mass."],
        )
        raw = task.model_dump_json()
        restored = SandboxTask.model_validate_json(raw)
        assert restored.trace_id == task.trace_id
        assert restored.domains == ["physics"]

    def test_step_round_trip(self):
        step = SandboxStep(
            tool="academic_search",
            input={"query": "Higgs boson"},
            output={"papers": [{"title": "Discovery of the Higgs", "year": 2012}]},
            commentary="Look for original discovery paper.",
            status="ok",
            started_at=datetime.now(tz=timezone.utc),
            finished_at=datetime.now(tz=timezone.utc),
        )
        raw = step.model_dump_json()
        restored = SandboxStep.model_validate_json(raw)
        assert restored.tool == "academic_search"
        assert restored.status == "ok"

    def test_result_round_trip(self):
        result = SandboxResult(
            trace_id="abc-123",
            started_at=datetime.now(tz=timezone.utc),
            finished_at=datetime.now(tz=timezone.utc),
            steps=[],
            summary="No steps executed.",
        )
        as_dict = result.model_dump()
        restored = SandboxResult.model_validate(as_dict)
        assert restored.summary == "No steps executed."
        assert restored.trace_id == "abc-123"


# ---------------------------------------------------------------------------
# 4. build_sandbox_task_from_run
# ---------------------------------------------------------------------------


class TestBuildSandboxTask:
    def _make_run_dict(self):
        return {
            "trace_id": "run-999",
            "sentence": "Explain black holes.",
            "routing": {
                "classification": "SCIENTIFIC",
                "selected_domains": ["physics", "astrophysics"],
            },
            "expert_decision": {
                "decision_type": "USE_EXISTING",
                "selected_experts": ["physics_expert"],
                "expert_confidence": 0.91,
            },
            "phase3": {
                "validation_decision": {"result_class": "VALID"},
                "action_result": {
                    "validated_answer": "A black hole is a region where gravity is so strong that nothing can escape."
                },
                "phase_latencies_ms": {"phase3": 340, "phase4": 210},
                "raw": {},
            },
        }

    def test_derives_domains(self):
        task = build_sandbox_task_from_run(self._make_run_dict())
        assert "physics" in task.domains
        assert "astrophysics" in task.domains

    def test_derives_trace_id(self):
        task = build_sandbox_task_from_run(self._make_run_dict())
        assert task.trace_id == "run-999"

    def test_uses_explicit_user_query(self):
        task = build_sandbox_task_from_run(
            self._make_run_dict(), user_query="Custom question?"
        )
        assert task.user_query == "Custom question?"

    def test_falls_back_to_sentence(self):
        task = build_sandbox_task_from_run(self._make_run_dict())
        assert task.user_query == "Explain black holes."

    def test_extracts_hypothesis(self):
        task = build_sandbox_task_from_run(self._make_run_dict())
        assert any("black hole" in h.lower() for h in task.hypotheses)

    def test_empty_run_does_not_crash(self):
        task = build_sandbox_task_from_run({}, user_query="anything")
        assert task.user_query == "anything"
        assert task.domains == []


# ---------------------------------------------------------------------------
# 5. Calculator tool (no network)
# ---------------------------------------------------------------------------


class TestCalculatorTool:
    def test_basic_arithmetic(self):
        result = _TOOL_REGISTRY["calculator"]("2 ** 10")
        assert result["result"] == 1024
        assert result["source"] == "calculator"

    def test_math_function(self):
        result = _TOOL_REGISTRY["calculator"]("sqrt(144)")
        assert abs(result["result"] - 12.0) < 1e-9

    def test_bad_expression(self):
        result = _TOOL_REGISTRY["calculator"]("open('/etc/passwd')")
        assert "error" in result

    def test_division_by_zero(self):
        result = _TOOL_REGISTRY["calculator"]("1 / 0")
        assert "error" in result


# ---------------------------------------------------------------------------
# 6. SandboxManager._stub_result
# ---------------------------------------------------------------------------


class TestSandboxManagerStub:
    def test_stub_result_shape(self):
        task = SandboxTask(trace_id="t-1", user_query="test")
        result = SandboxManager._stub_result(task, datetime.utcnow())
        assert isinstance(result, SandboxResult)
        assert result.steps == []
        assert (
            "failed" in result.summary.lower()
            or "unavailable" in result.summary.lower()
        )


# ---------------------------------------------------------------------------
# 7. _execute_step with unknown tool (no network)
# ---------------------------------------------------------------------------


class TestExecuteStepUnknownTool:
    def test_unknown_tool_returns_error_step(self):
        manager = SandboxManager.__new__(SandboxManager)
        step = manager._execute_step("nonexistent_tool", "some query", "test")
        assert step.status == "error"
        assert "Unknown tool" in step.output.get("error", "")
        assert step.tool == "nonexistent_tool"
