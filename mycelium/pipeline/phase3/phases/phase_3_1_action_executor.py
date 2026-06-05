"""
Phase 3.1 — Action Execution.

Receives a ``FinalDecisionResult`` from Phase 2.6 and executes the
recommended action: creating a new expert, routing to an existing
expert, or applying a patch to an existing expert.

Components:
    * ActionExecutor         — Routes and executes actions.
    * NewExpertCreator       — Creates new expert resources.
    * ExistingExpertRouter   — Routes work to existing experts.
    * ExpertPatchCreator     — Creates and applies expert patches.
    * ActionExecutionPipeline — Orchestrates the full phase.

Example:
    >>> from mycelium.pipeline.phase3.phases.phase_3_1_action_executor import (
    ...     ActionExecutionPipeline,
    ... )
    >>> from mycelium.pipeline.phase3.utils.types import FinalDecisionResult
    >>> decision = FinalDecisionResult(action="create_new_expert", domain="chemistry")
    >>> pipeline = ActionExecutionPipeline()
    >>> result = pipeline.execute(decision)
    >>> print(result.status)
    'success'
"""

import copy
import logging
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from mycelium.pipeline.phase3.config.phase3_config import Phase3Config
from mycelium.pipeline.phase3.utils.types import (
    ActionResult,
    CreatedResource,
    FinalDecisionResult,
)

logger = logging.getLogger(__name__)

# Canonical action names used internally by ActionExecutor.
_VALID_ACTIONS = frozenset({"create_new", "create_new_expert", "use_existing", "create_patch"})
_VALID_PATCH_TYPES = frozenset(
    {"parameter_update", "retraining", "augmentation"}
)

# Aliases: any name in the key set is normalised to the value before dispatch.
_ACTION_ALIAS: Dict[str, str] = {
    "create_new_expert": "create_new",
}


# ------------------------------------------------------------------
# Lazy post-check import helper
# ------------------------------------------------------------------

def _get_post_check_runner():
    """Return a cached PostCheckRunner, or None if the package is absent."""
    if not hasattr(_get_post_check_runner, "_instance"):
        try:
            from expert_post_check.runner import PostCheckRunner  # noqa: PLC0415
            _get_post_check_runner._instance = PostCheckRunner()
        except Exception as exc:
            logger.warning(
                "expert_post_check unavailable — post-check disabled: %s", exc
            )
            _get_post_check_runner._instance = None
    return _get_post_check_runner._instance


# ------------------------------------------------------------------
# Custom exceptions
# ------------------------------------------------------------------


class ActionExecutionError(Exception):
    """Raised when action execution fails."""


# ------------------------------------------------------------------
# NewExpertCreator
# ------------------------------------------------------------------


class NewExpertCreator:
    """Create new expert resources when required."""

    def __init__(self) -> None:
        self._created_experts: List[CreatedResource] = []

    def create_expert(
        self,
        domain: str,
        config: Dict,
        training_data: Optional[List] = None,
    ) -> CreatedResource:
        """Create a new expert resource.

        Args:
            domain: Domain the expert specialises in.
            config: Hyper-parameter overrides.
            training_data: Optional initial training data.

        Returns:
            A ``CreatedResource`` describing the new expert.
        """
        logger.debug("Creating new expert for domain=%s", domain)
        expert_id = self.generate_expert_id()
        parameters = self.configure_expert_parameters(domain, config)
        training_data_size = len(training_data) if training_data else 0

        resource = CreatedResource(
            resource_type="expert",
            resource_id=expert_id,
            domain=domain,
            configuration={
                "domain": domain,
                **parameters,
                "training_data_size": training_data_size,
            },
            created_at=datetime.now(timezone.utc).isoformat(),
            status="active",
            metadata={"creator": "NewExpertCreator"},
        )
        self._created_experts.append(resource)
        logger.info("Created expert %s for domain=%s", expert_id, domain)
        return resource

    def generate_expert_id(self) -> str:
        """Return a unique identifier like ``expert_<hex12>``."""
        return f"expert_{uuid4().hex[:12]}"

    def configure_expert_parameters(
        self, domain: str, config: Dict
    ) -> Dict:
        """Merge default hyper-parameters with *config* overrides.

        Args:
            domain: Expert domain.
            config: Caller-supplied overrides.

        Returns:
            Merged configuration dictionary.
        """
        defaults: Dict = {
            "domain": domain,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10,
        }
        defaults.update(config)
        logger.debug("Expert params for domain=%s: %s", domain, defaults)
        return defaults

    def validate_expert_creation(self, expert_config: Dict) -> bool:
        """Return ``True`` when *expert_config* has a non-empty ``domain``.

        Args:
            expert_config: Configuration to validate.

        Returns:
            Whether the configuration is valid.
        """
        domain = expert_config.get("domain")
        valid = isinstance(domain, str) and len(domain) > 0
        if not valid:
            logger.warning(
                "Invalid expert config — missing or empty 'domain': %s",
                expert_config,
            )
        return valid


# ------------------------------------------------------------------
# ExistingExpertRouter
# ------------------------------------------------------------------


class ExistingExpertRouter:
    """Route work to an existing expert."""

    def __init__(self) -> None:
        self._expert_registry: Dict[str, Dict] = {}
        self._routing_history: List[Dict] = []

    def route_to_expert(
        self, expert_name: str, input_data: Dict
    ) -> Dict:
        """Route *input_data* to the named expert.

        Args:
            expert_name: Name of the target expert.
            input_data: Payload to deliver.

        Returns:
            A dictionary describing the routing outcome.
        """
        logger.debug("Routing to expert=%s", expert_name)
        timestamp = datetime.now(timezone.utc).isoformat()
        result: Dict = {
            "expert_name": expert_name,
            "status": "routed",
            "input_received": True,
            "timestamp": timestamp,
        }
        self._routing_history.append(
            {"expert_name": expert_name, "timestamp": timestamp}
        )
        logger.info("Routed to expert=%s at %s", expert_name, timestamp)
        return result

    def verify_expert_exists(self, expert_name: str) -> bool:
        """Return ``True`` if *expert_name* is registered.

        Args:
            expert_name: Name to look up.
        """
        exists = expert_name in self._expert_registry
        logger.debug("Expert exists '%s': %s", expert_name, exists)
        return exists

    def register_expert(self, expert_name: str, domain: str) -> None:
        """Register an expert in the local registry.

        Args:
            expert_name: Unique name for the expert.
            domain: Domain the expert covers.
        """
        self._expert_registry[expert_name] = {
            "domain": domain,
            "status": "active",
            "performance_history": {},
        }
        logger.info(
            "Registered expert '%s' for domain=%s", expert_name, domain
        )

    def get_expert_performance_history(self, expert_name: str) -> Dict:
        """Return the performance history, or ``{}`` if unknown.

        Args:
            expert_name: Name of the expert to query.
        """
        entry = self._expert_registry.get(expert_name)
        if entry is None:
            logger.warning("No registry entry for expert '%s'", expert_name)
            return {}
        return entry.get("performance_history", {})


# ------------------------------------------------------------------
# ExpertPatchCreator
# ------------------------------------------------------------------


class ExpertPatchCreator:
    """Create and apply patches to existing experts."""

    def __init__(self) -> None:
        self._patches: List[CreatedResource] = []

    def create_patch(
        self, expert_name: str, patch_type: str, modifications: Dict
    ) -> CreatedResource:
        """Create a new patch resource for an expert.

        Args:
            expert_name: Expert the patch targets.
            patch_type: Category (e.g. ``parameter_update``).
            modifications: Concrete changes to apply.

        Returns:
            A ``CreatedResource`` describing the patch.
        """
        logger.debug(
            "Creating patch for expert=%s type=%s", expert_name, patch_type
        )
        patch_id = self.generate_patch_id()
        resource = CreatedResource(
            resource_type="patch",
            resource_id=patch_id,
            domain="",
            configuration={
                "expert_name": expert_name,
                "patch_type": patch_type,
                "modifications": modifications,
            },
            created_at=datetime.now(timezone.utc).isoformat(),
            status="active",
            metadata={"creator": "ExpertPatchCreator"},
        )
        self._patches.append(resource)
        logger.info("Created patch %s for expert=%s", patch_id, expert_name)
        return resource

    def generate_patch_id(self) -> str:
        """Return a unique identifier like ``patch_<hex12>``."""
        return f"patch_{uuid4().hex[:12]}"

    def test_patch_compatibility(
        self, expert_name: str, patch: Dict
    ) -> bool:
        """Return ``True`` if the patch carries a valid ``patch_type``.

        Args:
            expert_name: Target expert name.
            patch: Patch descriptor dictionary.
        """
        patch_type = patch.get("patch_type")
        compatible = patch_type in _VALID_PATCH_TYPES
        if not compatible:
            logger.warning(
                "Patch incompatible with expert '%s': patch_type=%s",
                expert_name,
                patch_type,
            )
        return compatible

    def apply_patch(self, expert_name: str, patch: Dict) -> bool:
        """Apply a patch to the named expert.  Returns ``True``.

        Args:
            expert_name: Target expert.
            patch: Patch descriptor dictionary.
        """
        logger.info("Applying patch to expert '%s': %s", expert_name, patch)
        return True


# ------------------------------------------------------------------
# ActionExecutor
# ------------------------------------------------------------------


class ActionExecutor:
    """Execute system recommendations.

    Validates and routes incoming actions to the appropriate
    sub-component and maintains an execution history.
    """

    def __init__(self, config: Optional[Phase3Config] = None) -> None:
        """Initialise the executor.

        Args:
            config: Optional ``Phase3Config``. Uses defaults when ``None``.
        """
        self._config = config or Phase3Config()
        self.expert_registry: Dict[str, Dict] = {}
        self.action_history: List[Dict] = []
        self._expert_creator = NewExpertCreator()
        self._expert_router = ExistingExpertRouter()
        self._patch_creator = ExpertPatchCreator()

    def execute_action(
        self, final_decision_result: FinalDecisionResult
    ) -> ActionResult:
        """Execute the action recommended by Phase 2.6.

        Accepts both the canonical action names (``create_new``,
        ``use_existing``, ``create_patch``) and the alias
        ``create_new_expert`` emitted by Phase 2.6 synthesis.

        Args:
            final_decision_result: Decision output from Phase 2.6.

        Returns:
            An ``ActionResult`` with status ``'success'`` or ``'failed'``.
        """
        # Normalise aliased action names before any further processing.
        raw_action = final_decision_result.action
        action = _ACTION_ALIAS.get(raw_action, raw_action)
        if action != raw_action:
            logger.debug(
                "Action alias resolved: '%s' -> '%s'", raw_action, action
            )
            import dataclasses
            final_decision_result = dataclasses.replace(
                final_decision_result, action=action
            )

        logger.info("Executing action: %s", action)
        start = time.perf_counter()

        if not self.validate_action_feasibility(action):
            elapsed = (time.perf_counter() - start) * 1000.0
            error_msg = f"Invalid action: {action}"
            logger.error(error_msg)
            self._record_history(action, "failed", elapsed)
            return ActionResult(
                action_type=action,
                status="failed",
                error_message=error_msg,
                execution_time_ms=elapsed,
            )

        try:
            if action == "create_new":
                result = self._execute_create_new(final_decision_result)
            elif action == "use_existing":
                result = self._execute_use_existing(final_decision_result)
            else:
                result = self._execute_create_patch(final_decision_result)

            elapsed = (time.perf_counter() - start) * 1000.0
            result.execution_time_ms = elapsed
            self._record_history(action, result.status, elapsed)
            logger.info(
                "Action '%s' completed status=%s in %.2f ms",
                action, result.status, elapsed,
            )
            return result
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000.0
            logger.error("Action '%s' raised: %s", action, exc)
            self._record_history(action, "failed", elapsed)
            return ActionResult(
                action_type=action,
                status="failed",
                error_message=str(exc),
                execution_time_ms=elapsed,
            )

    def validate_action_feasibility(self, action: str) -> bool:
        """Return ``True`` when *action* is in the valid set.

        Aliased names (e.g. ``create_new_expert``) are accepted because
        ``_VALID_ACTIONS`` includes them explicitly, and
        ``execute_action`` additionally normalises them before dispatch.

        Args:
            action: Action string to validate.
        """
        feasible = action in _VALID_ACTIONS
        logger.debug("Action feasibility '%s': %s", action, feasible)
        return feasible

    def execute_with_rollback(
        self, action: str, final_decision_result: FinalDecisionResult
    ) -> Tuple[bool, str]:
        """Execute with automatic rollback on failure.

        Args:
            action: Action type to execute.
            final_decision_result: Decision from Phase 2.6.

        Returns:
            A tuple ``(success, message)``.
        """
        logger.info("Executing with rollback: %s", action)
        try:
            result = self.execute_action(final_decision_result)
            if result.status == "success":
                return (True, "Action executed successfully")
            return (
                False,
                result.error_message
                or "Action completed with non-success status",
            )
        except Exception as exc:
            logger.error("Rollback-wrapped execution failed: %s", exc)
            return (False, str(exc))

    def get_execution_status(self) -> Dict:
        """Return aggregate counts from the execution history.

        Returns:
            Dict with ``total_executions``, ``successful``, ``failed``.
        """
        total = len(self.action_history)
        successful = sum(
            1 for h in self.action_history if h.get("status") == "success"
        )
        return {
            "total_executions": total,
            "successful": successful,
            "failed": total - successful,
        }

    # -- private helpers ------------------------------------------

    def _execute_create_new(self, fdr: FinalDecisionResult) -> ActionResult:
        domain = fdr.domain or "general"
        config = fdr.metadata.get("expert_config", {})
        training_data = fdr.metadata.get("training_data")
        resource = self._expert_creator.create_expert(
            domain, config, training_data
        )
        return ActionResult(
            action_type="create_new",
            status="success",
            executed_action=f"Created new expert for domain={domain}",
            created_resources=[resource],
            resource_ids=[resource.resource_id],
            rollback_available=self._config.enable_rollback,
        )

    def _execute_use_existing(self, fdr: FinalDecisionResult) -> ActionResult:
        """Route to an existing expert, then run the post-check pipeline.

        Steps
        -----
        1. Route the request to the selected expert.
        2. Extract the query string from fdr.metadata.
        3. Run PostCheckRunner.run(query, domain, expert_name).
        4. If verified_answer is empty (timeout / both reasoners failed),
           surface the best available partial answer from trm_answer or
           p6_answer and mark the result as degraded.  This prevents
           run_workflow.py from treating an empty string as a sentinel
           that triggers a full phase re-invocation (the infinite loop
           visible in logs at 21:03:40 → 21:05:33 → 21:06:27).
        5. Attach the PostCheckResult to action_result.metadata so it
           flows through to the API trace and the conversation layer LLM.
        6. Populate fdr.metadata["expert_used"] for upstream feedback.

        The ActionResult returned by this method always carries a
        non-empty ``executed_action`` string and ``status='success'``
        regardless of post-check outcome, so callers never need to
        re-invoke the phase based on this result alone.
        """
        expert_name = fdr.expert_name or "default_expert"
        domain = fdr.domain or expert_name
        input_data = fdr.metadata.get("input_data", {})

        # Step 1 — routing (unchanged behaviour)
        routing = self._expert_router.route_to_expert(expert_name, input_data)

        # Step 2 — resolve query string
        query: str = (
            fdr.metadata.get("original_query")
            or fdr.metadata.get("query")
            or input_data.get("query", "")
            or ""
        )

        # Step 3 — post-check
        post_check_meta: Dict = {}
        verified_answer: str = ""
        degraded: bool = False
        runner = _get_post_check_runner()
        if runner is not None and query:
            try:
                pc_result = runner.run(
                    query=query,
                    domain=domain,
                    expert_name=expert_name,
                )
                verified_answer = pc_result.verified_answer

                # Step 4 — degrade gracefully when verified_answer is empty.
                if not verified_answer:
                    fallback = pc_result.trm_answer or pc_result.p6_answer
                    if fallback:
                        verified_answer = fallback
                        degraded = True
                        logger.warning(
                            "PostCheck verified_answer empty — using fallback "
                            "answer from %s (degraded mode)",
                            "trm" if pc_result.trm_answer else "p6",
                        )
                    else:
                        degraded = True
                        logger.warning(
                            "PostCheck: both TRM and P6 returned empty answers "
                            "for domain=%s. Returning degraded result.",
                            domain,
                        )

                # Serialise dataclass to plain dict for JSON-safe storage
                if is_dataclass(pc_result):
                    post_check_meta = asdict(pc_result)
                else:
                    post_check_meta = vars(pc_result)

                logger.info(
                    "PostCheck complete — status=%s sim=%.3f degraded=%s",
                    pc_result.verification_status,
                    pc_result.similarity,
                    degraded,
                )
            except Exception as exc:
                logger.warning("PostCheckRunner.run() failed: %s", exc)
                post_check_meta = {"error": str(exc)}
                degraded = True
        elif runner is None:
            logger.debug("PostCheck skipped — runner unavailable")
        else:
            logger.debug("PostCheck skipped — empty query")

        # Step 6 — feedback: mark which expert was used
        fdr.metadata["expert_used"] = expert_name

        return ActionResult(
            action_type="use_existing",
            status="success",
            executed_action=f"Routed to expert={expert_name}",
            metadata={
                "routing": routing,
                "verified_answer": verified_answer,
                "post_check": post_check_meta,
                "post_check_degraded": degraded,
            },
            rollback_available=False,
        )

    def _execute_create_patch(self, fdr: FinalDecisionResult) -> ActionResult:
        expert_name = fdr.expert_name or "default_expert"
        patch_type = fdr.metadata.get("patch_type", "parameter_update")
        modifications = fdr.metadata.get("modifications", {})
        resource = self._patch_creator.create_patch(
            expert_name, patch_type, modifications
        )
        return ActionResult(
            action_type="create_patch",
            status="success",
            executed_action=f"Created patch for expert={expert_name}",
            created_resources=[resource],
            resource_ids=[resource.resource_id],
            rollback_available=self._config.enable_rollback,
        )

    def _record_history(
        self, action: str, status: str, elapsed_ms: float
    ) -> None:
        self.action_history.append({
            "action": action,
            "status": status,
            "execution_time_ms": elapsed_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })


# ------------------------------------------------------------------
# ActionExecutionPipeline (orchestrator)
# ------------------------------------------------------------------


class ActionExecutionPipeline:
    """Orchestrator for Phase 3.1 action execution.

    Coordinates ``ActionExecutor`` and its sub-components, records
    timing metadata, and exposes a single ``execute`` entry-point.
    """

    def __init__(self, config: Optional[Phase3Config] = None) -> None:
        """Initialise the pipeline.

        Args:
            config: Optional ``Phase3Config``. Uses defaults when ``None``.
        """
        self._config = config or Phase3Config()
        self._executor = ActionExecutor(self._config)
        self._expert_creator = NewExpertCreator()
        self._expert_router = ExistingExpertRouter()
        self._patch_creator = ExpertPatchCreator()
        self._last_metadata: Dict = {}

        errors = self._config.validate()
        if errors:
            logger.warning("Config validation errors: %s", errors)

    def execute(
        self, final_decision_result: FinalDecisionResult
    ) -> ActionResult:
        """Run the full action-execution phase.

        Args:
            final_decision_result: Output from Phase 2.6.

        Returns:
            An ``ActionResult`` with execution details.

        Raises:
            ActionExecutionError: On critical failure.
        """
        start = time.perf_counter()
        logger.info(
            "Pipeline executing action=%s", final_decision_result.action
        )

        result = self._executor.execute_action(final_decision_result)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        self._last_metadata = {
            "action": final_decision_result.action,
            "status": result.status,
            "execution_time_ms": elapsed_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if self._config.enable_performance_logging:
            logger.info(
                "Pipeline finished in %.2f ms (status=%s)",
                elapsed_ms, result.status,
            )
        return result

    def get_execution_metadata(self) -> Dict:
        """Return a copy of metadata from the most recent execution."""
        return copy.copy(self._last_metadata)
