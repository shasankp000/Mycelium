"""Tests for Phase 3.1 — Action Executor components."""

import pytest

from phase3_validation.config.phase3_config import Phase3Config
from phase3_validation.phases.phase_3_1_action_executor import (
    ActionExecutor,
    ActionExecutionPipeline,
    ExistingExpertRouter,
    ExpertPatchCreator,
    NewExpertCreator,
    ActionExecutionError,
)
from phase3_validation.utils.types import (
    ActionResult,
    CreatedResource,
    FinalDecisionResult,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_decision(**kwargs) -> FinalDecisionResult:
    """Build a FinalDecisionResult with sensible defaults."""
    defaults = {
        "decision": "test decision",
        "confidence": 0.85,
        "reasoning": "unit-test reasoning",
        "action": "use_existing",
        "expert_name": "test_expert",
        "domain": "chemistry",
        "metadata": {},
    }
    defaults.update(kwargs)
    return FinalDecisionResult(**defaults)


# ------------------------------------------------------------------
# TestActionExecutor
# ------------------------------------------------------------------


class TestActionExecutor:
    """Tests for the ActionExecutor class."""

    def setup_method(self):
        self.config = Phase3Config()
        self.executor = ActionExecutor(config=self.config)

    def test_execute_create_new_expert(self):
        decision = _make_decision(action="create_new", domain="physics")
        result = self.executor.execute_action(decision)

        assert isinstance(result, ActionResult)
        assert result.action_type == "create_new"
        assert result.status == "success"
        assert result.execution_time_ms > 0
        assert len(result.created_resources) == 1
        assert result.created_resources[0].resource_type == "expert"
        assert len(result.resource_ids) == 1
        assert result.error_message is None

    def test_execute_use_existing_expert(self):
        decision = _make_decision(
            action="use_existing",
            expert_name="bert_expert",
            metadata={"input_data": {"text": "sample"}},
        )
        result = self.executor.execute_action(decision)

        assert result.action_type == "use_existing"
        assert result.status == "success"
        assert "routing" in result.metadata
        assert result.metadata["routing"]["expert_name"] == "bert_expert"
        assert result.metadata["routing"]["status"] == "routed"
        assert result.rollback_available is False

    def test_execute_create_patch(self):
        decision = _make_decision(
            action="create_patch",
            expert_name="chem_expert",
            metadata={
                "patch_type": "parameter_update",
                "modifications": {"learning_rate": 0.0005},
            },
        )
        result = self.executor.execute_action(decision)

        assert result.action_type == "create_patch"
        assert result.status == "success"
        assert len(result.created_resources) == 1
        assert result.created_resources[0].resource_type == "patch"
        assert result.rollback_available == self.config.enable_rollback

    def test_validate_action_feasibility_valid(self):
        for action in ("create_new", "use_existing", "create_patch"):
            assert self.executor.validate_action_feasibility(action) is True

    def test_validate_action_feasibility_invalid(self):
        assert self.executor.validate_action_feasibility("delete") is False
        assert self.executor.validate_action_feasibility("") is False
        assert self.executor.validate_action_feasibility("CREATE_NEW") is False

    def test_execute_invalid_action_returns_failed(self):
        decision = _make_decision(action="nonexistent")
        result = self.executor.execute_action(decision)

        assert result.status == "failed"
        assert result.action_type == "nonexistent"
        assert result.error_message is not None
        assert "Invalid action" in result.error_message

    def test_execute_with_rollback_success(self):
        decision = _make_decision(action="create_new", domain="biology")
        success, msg = self.executor.execute_with_rollback(
            "create_new", decision
        )

        assert success is True
        assert "successfully" in msg.lower()

    def test_execute_with_rollback_failure(self):
        decision = _make_decision(action="bad_action")
        success, msg = self.executor.execute_with_rollback(
            "bad_action", decision
        )

        assert success is False
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_get_execution_status_initial(self):
        status = self.executor.get_execution_status()

        assert status["total_executions"] == 0
        assert status["successful"] == 0
        assert status["failed"] == 0

    def test_get_execution_status_after_executions(self):
        self.executor.execute_action(
            _make_decision(action="create_new", domain="math")
        )
        self.executor.execute_action(
            _make_decision(action="invalid_action")
        )

        status = self.executor.get_execution_status()
        assert status["total_executions"] == 2
        assert status["successful"] == 1
        assert status["failed"] == 1


# ------------------------------------------------------------------
# TestNewExpertCreator
# ------------------------------------------------------------------


class TestNewExpertCreator:
    """Tests for the NewExpertCreator class."""

    def setup_method(self):
        self.creator = NewExpertCreator()

    def test_create_expert(self):
        resource = self.creator.create_expert(
            domain="chemistry", config={"epochs": 20}, training_data=[1, 2]
        )

        assert isinstance(resource, CreatedResource)
        assert resource.resource_type == "expert"
        assert resource.domain == "chemistry"
        assert resource.status == "active"
        assert resource.resource_id.startswith("expert_")
        assert resource.configuration["training_data_size"] == 2
        assert resource.configuration["epochs"] == 20
        assert resource.created_at != ""

    def test_create_expert_no_training_data(self):
        resource = self.creator.create_expert(
            domain="physics", config={}, training_data=None
        )

        assert resource.configuration["training_data_size"] == 0

    def test_generate_expert_id(self):
        eid = self.creator.generate_expert_id()

        assert eid.startswith("expert_")
        assert len(eid) == len("expert_") + 12

    def test_generate_expert_id_unique(self):
        ids = {self.creator.generate_expert_id() for _ in range(50)}
        assert len(ids) == 50, "IDs should be unique"

    def test_configure_expert_parameters_defaults(self):
        params = self.creator.configure_expert_parameters("bio", {})

        assert params["learning_rate"] == 0.001
        assert params["batch_size"] == 32
        assert params["epochs"] == 10
        assert params["domain"] == "bio"

    def test_configure_expert_parameters_overrides(self):
        params = self.creator.configure_expert_parameters(
            "bio", {"learning_rate": 0.01, "batch_size": 64}
        )

        assert params["learning_rate"] == 0.01
        assert params["batch_size"] == 64
        assert params["epochs"] == 10  # default kept

    def test_validate_expert_creation_valid(self):
        assert self.creator.validate_expert_creation({"domain": "chem"}) is True

    def test_validate_expert_creation_invalid_missing(self):
        assert self.creator.validate_expert_creation({}) is False

    def test_validate_expert_creation_invalid_empty(self):
        assert self.creator.validate_expert_creation({"domain": ""}) is False

    def test_validate_expert_creation_invalid_type(self):
        assert self.creator.validate_expert_creation({"domain": 123}) is False


# ------------------------------------------------------------------
# TestExistingExpertRouter
# ------------------------------------------------------------------


class TestExistingExpertRouter:
    """Tests for the ExistingExpertRouter class."""

    def setup_method(self):
        self.router = ExistingExpertRouter()

    def test_route_to_expert(self):
        result = self.router.route_to_expert(
            "bert_expert", {"text": "hello"}
        )

        assert result["expert_name"] == "bert_expert"
        assert result["status"] == "routed"
        assert result["input_received"] is True
        assert "timestamp" in result

    def test_verify_expert_exists_unregistered(self):
        assert self.router.verify_expert_exists("ghost") is False

    def test_verify_expert_exists_registered(self):
        self.router.register_expert("bert", "nlp")
        assert self.router.verify_expert_exists("bert") is True

    def test_register_and_verify(self):
        self.router.register_expert("chem_expert", "chemistry")
        assert self.router.verify_expert_exists("chem_expert") is True
        assert self.router.verify_expert_exists("other") is False

    def test_get_expert_performance_history_unknown(self):
        history = self.router.get_expert_performance_history("nope")
        assert history == {}

    def test_get_expert_performance_history_registered(self):
        self.router.register_expert("x", "math")
        history = self.router.get_expert_performance_history("x")
        assert isinstance(history, dict)


# ------------------------------------------------------------------
# TestExpertPatchCreator
# ------------------------------------------------------------------


class TestExpertPatchCreator:
    """Tests for the ExpertPatchCreator class."""

    def setup_method(self):
        self.patcher = ExpertPatchCreator()

    def test_create_patch(self):
        resource = self.patcher.create_patch(
            expert_name="bert",
            patch_type="retraining",
            modifications={"epochs": 5},
        )

        assert isinstance(resource, CreatedResource)
        assert resource.resource_type == "patch"
        assert resource.resource_id.startswith("patch_")
        assert resource.status == "active"
        assert resource.configuration["patch_type"] == "retraining"
        assert resource.configuration["expert_name"] == "bert"

    def test_generate_patch_id(self):
        pid = self.patcher.generate_patch_id()
        assert pid.startswith("patch_")
        assert len(pid) == len("patch_") + 12

    def test_test_patch_compatibility_valid(self):
        for pt in ("parameter_update", "retraining", "augmentation"):
            assert self.patcher.test_patch_compatibility(
                "bert", {"patch_type": pt}
            ) is True

    def test_test_patch_compatibility_invalid(self):
        assert self.patcher.test_patch_compatibility(
            "bert", {"patch_type": "delete_all"}
        ) is False
        assert self.patcher.test_patch_compatibility(
            "bert", {}
        ) is False

    def test_apply_patch(self):
        assert self.patcher.apply_patch("bert", {"lr": 0.01}) is True


# ------------------------------------------------------------------
# TestActionExecutionPipeline
# ------------------------------------------------------------------


class TestActionExecutionPipeline:
    """Tests for the ActionExecutionPipeline orchestrator."""

    def setup_method(self):
        self.pipeline = ActionExecutionPipeline()

    def test_pipeline_create_new(self):
        decision = _make_decision(action="create_new", domain="physics")
        result = self.pipeline.execute(decision)

        assert isinstance(result, ActionResult)
        assert result.status == "success"
        assert result.action_type == "create_new"
        assert len(result.created_resources) == 1

    def test_pipeline_use_existing(self):
        decision = _make_decision(
            action="use_existing", expert_name="router_expert"
        )
        result = self.pipeline.execute(decision)

        assert result.status == "success"
        assert result.action_type == "use_existing"

    def test_pipeline_create_patch(self):
        decision = _make_decision(
            action="create_patch",
            expert_name="patch_target",
            metadata={"patch_type": "augmentation", "modifications": {}},
        )
        result = self.pipeline.execute(decision)

        assert result.status == "success"
        assert result.action_type == "create_patch"
        assert len(result.resource_ids) == 1

    def test_pipeline_metadata_populated(self):
        decision = _make_decision(action="create_new", domain="bio")
        self.pipeline.execute(decision)
        meta = self.pipeline.get_execution_metadata()

        assert "action" in meta
        assert meta["action"] == "create_new"
        assert "status" in meta
        assert meta["status"] == "success"
        assert "execution_time_ms" in meta
        assert meta["execution_time_ms"] >= 0
        assert "timestamp" in meta

    def test_pipeline_metadata_empty_before_execution(self):
        meta = self.pipeline.get_execution_metadata()
        assert meta == {}

    def test_pipeline_invalid_action(self):
        decision = _make_decision(action="explode")
        result = self.pipeline.execute(decision)

        assert result.status == "failed"
        assert result.error_message is not None
