"""
Phase 3 — Validation Orchestrator.

The critical decision point after Phase 2.6 (Synthesis) that
determines if the TRM (Thinking and Reasoning Model) output is
valid.  Implements the four validation result paths and the
Complete Failure re-run mechanism.

Components:
    * ValidationResultClassifier — classifies Layer 1 result
    * CompleteFailureHandler    — handles complete failure path

Example:
    >>> from phase3_validation.phases.phase_3_validation import (
    ...     ValidationResultClassifier,
    ...     CompleteFailureHandler,
    ... )
    >>> from phase3_validation.phases.phase_3_layer_1_contradiction import (
    ...     ContradictionAnalyzer,
    ... )
    >>> from phase2_validation.phases.phase_2_6_synthesis import (
    ...     FinalDecisionResult,
    ... )
    >>> decision = FinalDecisionResult(
    ...     original_text="Query",
    ...     final_decision="use_existing",
    ...     reasoning_chain=[{"step": 1, "content": "Reason A"}],
    ... )
    >>> analyzer = ContradictionAnalyzer()
    >>> classifier = ValidationResultClassifier()
    >>> # Extract answer and reasoning from the decision
    >>> answer = decision.final_decision
    >>> steps = [s.get("content", "") for s in decision.reasoning_chain]
    >>> layer1 = analyzer.analyze(answer, steps, [])
    >>> val_decision = classifier.classify(layer1, decision)
    >>> print(val_decision.result_class)
    'all_pass'
"""

import logging
import time
from typing import Any, Callable, List, Optional

from phase3_validation.config.validation_config import ValidationConfig
from phase3_validation.phases.layer_1_types import (
    CompleteFailureInfo,
    Layer1Result,
    ValidationDecision,
)
from phase3_validation.phases.phase_3_layer_1_contradiction import (
    ContradictionAnalyzer,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ValidationResultClassifier
# ---------------------------------------------------------------------------


class ValidationResultClassifier:
    """Classify a Layer1Result into one of four result classes.

    Decision logic (Phase 3 Week 5.1 — Complete Failure focus):

    * IF ``contradiction_detected`` AND ``severity == CRITICAL``
      → ``complete_failure``  (action: ``reject_and_rerun``)
    * ELSE  → ``all_pass``  (other paths built in future weeks)

    Attributes:
        config: Validation configuration.
    """

    def __init__(
        self,
        config: Optional[ValidationConfig] = None,
    ) -> None:
        """Initialise the classifier.

        Args:
            config: Validation configuration.  Uses defaults when
                not provided.
        """
        self.config = config or ValidationConfig()

    def classify(
        self,
        layer1_result: Layer1Result,
        original_decision: Any = None,
    ) -> ValidationDecision:
        """Classify *layer1_result* and return a ValidationDecision.

        Args:
            layer1_result: Output of the Layer 1 contradiction
                analyzer.
            original_decision: The original FinalDecisionResult that
                was validated (stored in CompleteFailureInfo when
                classification is ``complete_failure``).

        Returns:
            :class:`ValidationDecision` with result class, action,
            next step, confidence, and optional failure info.
        """
        if (
            layer1_result.contradiction_detected
            and layer1_result.severity == "CRITICAL"
        ):
            return self._build_complete_failure(
                layer1_result, original_decision
            )

        # Other paths (major_failure, minor_discrepancy) built in
        # future weeks; for now everything else passes.
        return self._build_all_pass(layer1_result)

    # ------------------------------------------------------------------
    # Private builders
    # ------------------------------------------------------------------

    def _build_complete_failure(
        self,
        layer1_result: Layer1Result,
        original_decision: Any,
    ) -> ValidationDecision:
        """Build a ValidationDecision for the complete_failure path.

        Confidence is reduced based on the number of affected
        components (each additional failed component reduces
        confidence by 0.1, floor 0.1).

        Args:
            layer1_result: Layer1Result with severity CRITICAL.
            original_decision: The FinalDecisionResult being rejected.

        Returns:
            ValidationDecision with result_class ``complete_failure``.
        """
        base_confidence = 0.95
        penalty = 0.1 * len(layer1_result.affected_components)
        confidence = max(0.1, base_confidence - penalty)

        failure_info = CompleteFailureInfo(
            original_decision=original_decision,
            failure_reason=layer1_result.reason,
            affected_layers=list(
                range(1, self.config.total_reasoning_layers + 1)
            ),
            recommended_action="rerun_full_pipeline",
            retry_count=0,
            max_retries=self.config.max_retries,
        )

        logger.warning(
            "ValidationResultClassifier: COMPLETE FAILURE detected "
            "(reason=%s, components=%s)",
            layer1_result.reason,
            layer1_result.affected_components,
        )

        return ValidationDecision(
            result_class="complete_failure",
            action="reject_and_rerun",
            next_step=(
                "Reject TRM output entirely and re-run full "
                "reasoning pipeline (Layers 1–6)"
            ),
            confidence=confidence,
            layer1_result=layer1_result,
            failure_info=failure_info,
        )

    def _build_all_pass(
        self,
        layer1_result: Layer1Result,
    ) -> ValidationDecision:
        """Build a ValidationDecision for the all_pass path.

        Args:
            layer1_result: Layer1Result with no critical issues.

        Returns:
            ValidationDecision with result_class ``all_pass``.
        """
        return ValidationDecision(
            result_class="all_pass",
            action="pass_to_action_executor",
            next_step="Proceed to Phase 3.1 action execution",
            confidence=1.0,
            layer1_result=layer1_result,
            failure_info=None,
        )


# ---------------------------------------------------------------------------
# CompleteFailureHandler
# ---------------------------------------------------------------------------


class CompleteFailureHandler:
    """Handle the Complete Failure path from the validation flowchart.

    Flow:
    1. Log failure with full details.
    2. Reject TRM entirely.
    3. Prepare clean slate (record that cache should be skipped).
    4. Re-run full reasoning pipeline (Layers 1–6) via *pipeline_fn*.
    5. Return to validation (loop back).
    6. Prevent infinite loops (max *config.max_retries* attempts).

    The handler accepts a *pipeline_fn* callable with the signature
    ``(text: str, skip_cache: bool) -> FinalDecisionResult`` so that
    it can be used with the real ``Phase2Pipeline.run`` or with a
    mock during testing.

    Attributes:
        config: Validation configuration.
        pipeline_fn: Callable that re-runs the full pipeline.
        analyzer: ContradictionAnalyzer for re-validating results.
        classifier: ValidationResultClassifier for re-validation.
        _retry_count: Current number of retry attempts.
    """

    def __init__(
        self,
        config: Optional[ValidationConfig] = None,
        pipeline_fn: Optional[
            Callable[..., Any]
        ] = None,
    ) -> None:
        """Initialise the handler.

        Args:
            config: Validation configuration.
            pipeline_fn: Callable used to re-run the pipeline.
                Signature: ``(text: str, skip_cache: bool=False)``
                returning a ``FinalDecisionResult``-like object.
                When not provided the handler will import
                ``Phase2Pipeline`` lazily and use ``run()``.
        """
        self.config = config or ValidationConfig()
        self.pipeline_fn = pipeline_fn
        self.analyzer = ContradictionAnalyzer(config=self.config)
        self.classifier = ValidationResultClassifier(
            config=self.config
        )
        self._retry_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle(
        self,
        validation_decision: ValidationDecision,
    ) -> ValidationDecision:
        """Handle a complete_failure ValidationDecision.

        Logs the failure, clears the retry counter on the first call,
        and delegates to :meth:`rerun_pipeline` with loop protection.

        Args:
            validation_decision: The ValidationDecision with
                ``result_class == "complete_failure"``.

        Returns:
            A new ValidationDecision after re-running the pipeline
            (may still be ``complete_failure`` if the re-run also
            fails and retries are exhausted).

        Raises:
            ValueError: If *validation_decision* is not a complete
                failure.
        """
        if validation_decision.result_class != "complete_failure":
            raise ValueError(
                "CompleteFailureHandler.handle() expects a "
                f"'complete_failure' decision, got "
                f"'{validation_decision.result_class}'"
            )

        self._log_failure(validation_decision)
        return self.rerun_pipeline(validation_decision)

    def rerun_pipeline(
        self,
        validation_decision: ValidationDecision,
    ) -> ValidationDecision:
        """Re-run the full pipeline and re-validate the result.

        Implements the re-run loop with a maximum of
        ``config.max_retries`` attempts.

        Args:
            validation_decision: The current failure decision.

        Returns:
            Updated ValidationDecision after re-run attempt.
        """
        failure_info = validation_decision.failure_info
        if failure_info is None:
            # Create minimal failure info if missing
            failure_info = CompleteFailureInfo(
                original_decision=None,
                failure_reason="unknown",
                max_retries=self.config.max_retries,
            )

        if self._retry_count >= self.config.max_retries:
            logger.error(
                "CompleteFailureHandler: max retries (%d) reached; "
                "escalating failure.",
                self.config.max_retries,
            )
            return self._build_max_retries_exceeded(
                validation_decision
            )

        self._retry_count += 1
        failure_info.retry_count = self._retry_count

        logger.info(
            "CompleteFailureHandler: retry %d/%d — re-running pipeline",
            self._retry_count,
            self.config.max_retries,
        )

        original = failure_info.original_decision
        text = self._extract_text(original)

        # Re-run the pipeline with skip_cache=True (clean slate)
        try:
            new_decision = self._run_pipeline(text, skip_cache=True)
        except Exception as exc:
            logger.exception(
                "CompleteFailureHandler: pipeline re-run failed: %s",
                exc,
            )
            return self._build_pipeline_error(
                validation_decision, str(exc)
            )

        # Re-validate the new decision
        new_layer1 = self._revalidate(new_decision)
        new_val_decision = self.classifier.classify(
            new_layer1, new_decision
        )

        if new_val_decision.result_class == "complete_failure":
            logger.warning(
                "CompleteFailureHandler: re-run produced another "
                "complete_failure (attempt %d)",
                self._retry_count,
            )
            # Preserve original request for next retry
            if new_val_decision.failure_info:
                new_val_decision.failure_info.original_decision = (
                    original
                )
                new_val_decision.failure_info.retry_count = (
                    self._retry_count
                )

        return new_val_decision

    def reset_retry_count(self) -> None:
        """Reset the internal retry counter to zero."""
        self._retry_count = 0

    @property
    def retry_count(self) -> int:
        """Current number of retry attempts performed."""
        return self._retry_count

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _log_failure(self, decision: ValidationDecision) -> None:
        """Log full failure details.

        Args:
            decision: The complete_failure ValidationDecision.
        """
        fi = decision.failure_info
        l1 = decision.layer1_result

        logger.error(
            "COMPLETE FAILURE: reason=%s, "
            "affected_layers=%s, "
            "severity=%s, "
            "contradiction=%s, "
            "ood_score=%.3f, "
            "chain_score=%.3f, "
            "evidence_score=%.3f",
            fi.failure_reason if fi else "unknown",
            fi.affected_layers if fi else [],
            l1.severity if l1 else "UNKNOWN",
            l1.contradiction_detected if l1 else False,
            l1.ood_score if l1 else 0.0,
            l1.chain_validity_score if l1 else 0.0,
            l1.evidence_alignment_score if l1 else 0.0,
        )

    def _run_pipeline(
        self,
        text: str,
        skip_cache: bool = True,
    ) -> Any:
        """Run the Phase 2 pipeline.

        Uses the injected *pipeline_fn* if provided, otherwise
        lazily imports and instantiates ``Phase2Pipeline``.

        Args:
            text: Input text to re-run.
            skip_cache: Whether to skip internal caches.

        Returns:
            FinalDecisionResult from the pipeline.
        """
        if self.pipeline_fn is not None:
            return self.pipeline_fn(text, skip_cache=skip_cache)

        # Lazy import to avoid circular imports
        from phase2_validation.pipeline import Phase2Pipeline  # noqa

        pipeline = Phase2Pipeline()
        return pipeline.run(text, skip_cache=skip_cache)

    def _revalidate(self, new_decision: Any) -> Layer1Result:
        """Re-run Layer 1 analysis on *new_decision*.

        Args:
            new_decision: FinalDecisionResult from the re-run.

        Returns:
            Layer1Result for the new decision.
        """
        answer = self._extract_answer(new_decision)
        steps = self._extract_steps(new_decision)
        evidence = self._extract_evidence(new_decision)
        return self.analyzer.analyze(answer, steps, evidence)

    @staticmethod
    def _extract_text(decision: Any) -> str:
        """Extract the original input text from a FinalDecisionResult.

        Args:
            decision: FinalDecisionResult-like object.

        Returns:
            The original text string.
        """
        if decision is None:
            return ""
        return getattr(decision, "original_text", "") or getattr(
            decision, "decision", ""
        ) or ""

    @staticmethod
    def _extract_answer(decision: Any) -> str:
        """Extract the answer string from a FinalDecisionResult.

        Args:
            decision: FinalDecisionResult-like object.

        Returns:
            Answer string.
        """
        if decision is None:
            return ""
        answer = getattr(decision, "final_decision", None)
        if answer:
            return str(answer)
        pred = getattr(decision, "aggregated_prediction", None)
        if pred:
            return str(pred)
        return getattr(decision, "reasoning_text", "") or ""

    @staticmethod
    def _extract_steps(decision: Any) -> List[str]:
        """Extract reasoning steps from a FinalDecisionResult.

        Args:
            decision: FinalDecisionResult-like object.

        Returns:
            List of reasoning step strings.
        """
        if decision is None:
            return []
        chain = getattr(decision, "reasoning_chain", None) or []
        steps: List[str] = []
        for item in chain:
            if isinstance(item, dict):
                content = (
                    item.get("content")
                    or item.get("description")
                    or item.get("text")
                    or str(item)
                )
                steps.append(str(content))
            elif isinstance(item, str):
                steps.append(item)
        return steps

    @staticmethod
    def _extract_evidence(decision: Any) -> List[str]:
        """Extract evidence strings from a FinalDecisionResult.

        Args:
            decision: FinalDecisionResult-like object.

        Returns:
            List of evidence strings.
        """
        if decision is None:
            return []
        evidence: List[str] = []

        # Try action_details.evidence
        action_details = getattr(decision, "action_details", {}) or {}
        ev = action_details.get("evidence", [])
        if isinstance(ev, list):
            evidence.extend(str(e) for e in ev)

        # Fall back to expert predictions as evidence
        if not evidence:
            predictions = (
                getattr(decision, "expert_predictions", {}) or {}
            )
            for expert, pred in predictions.items():
                if isinstance(pred, dict):
                    ev_val = pred.get("prediction", "")
                    if ev_val:
                        evidence.append(str(ev_val))
                elif pred:
                    evidence.append(str(pred))

        return evidence

    def _build_max_retries_exceeded(
        self,
        decision: ValidationDecision,
    ) -> ValidationDecision:
        """Build a ValidationDecision indicating max retries exceeded.

        Args:
            decision: The last complete_failure decision.

        Returns:
            ValidationDecision with escalation info.
        """
        fi = decision.failure_info
        return ValidationDecision(
            result_class="complete_failure",
            action="escalate",
            next_step=(
                f"Max retries ({self.config.max_retries}) exceeded; "
                "escalate to human review"
            ),
            confidence=0.0,
            layer1_result=decision.layer1_result,
            failure_info=CompleteFailureInfo(
                original_decision=(
                    fi.original_decision if fi else None
                ),
                failure_reason=(
                    (fi.failure_reason if fi else "")
                    + " [max_retries_exceeded]"
                ),
                affected_layers=(
                    fi.affected_layers if fi else list(range(1, 7))
                ),
                recommended_action="escalate_to_human_review",
                retry_count=self._retry_count,
                max_retries=self.config.max_retries,
            ),
        )

    def _build_pipeline_error(
        self,
        decision: ValidationDecision,
        error_msg: str,
    ) -> ValidationDecision:
        """Build a ValidationDecision for a pipeline re-run error.

        Args:
            decision: The last complete_failure decision.
            error_msg: Error message from the failed re-run.

        Returns:
            ValidationDecision indicating the pipeline error.
        """
        fi = decision.failure_info
        return ValidationDecision(
            result_class="complete_failure",
            action="escalate",
            next_step=(
                f"Pipeline re-run failed ({error_msg}); escalate"
            ),
            confidence=0.0,
            layer1_result=decision.layer1_result,
            failure_info=CompleteFailureInfo(
                original_decision=(
                    fi.original_decision if fi else None
                ),
                failure_reason=(
                    f"pipeline_rerun_error: {error_msg}"
                ),
                affected_layers=(
                    fi.affected_layers if fi else list(range(1, 7))
                ),
                recommended_action="escalate_to_human_review",
                retry_count=self._retry_count,
                max_retries=self.config.max_retries,
            ),
        )


# ---------------------------------------------------------------------------
# Convenience: ValidationOrchestrator
# ---------------------------------------------------------------------------


class ValidationOrchestrator:
    """Full Phase 3 Validation Orchestrator.

    Combines the ContradictionAnalyzer, ValidationResultClassifier,
    and CompleteFailureHandler into a single entrypoint.

    Attributes:
        config: Shared validation configuration.
        analyzer: Layer 1 contradiction analyzer.
        classifier: Validation result classifier.
        handler: Complete failure handler.
    """

    def __init__(
        self,
        config: Optional[ValidationConfig] = None,
        pipeline_fn: Optional[Callable[..., Any]] = None,
    ) -> None:
        """Initialise the orchestrator.

        Args:
            config: Validation configuration.
            pipeline_fn: Callable used to re-run Phase 2 pipeline.
        """
        self.config = config or ValidationConfig()
        self.analyzer = ContradictionAnalyzer(config=self.config)
        self.classifier = ValidationResultClassifier(
            config=self.config
        )
        self.handler = CompleteFailureHandler(
            config=self.config,
            pipeline_fn=pipeline_fn,
        )

    def validate(
        self,
        final_decision: Any,
    ) -> ValidationDecision:
        """Run the full validation flow on *final_decision*.

        1. Extract answer, reasoning steps, and evidence.
        2. Run Layer 1 analysis.
        3. Classify the result.
        4. If complete_failure, hand off to CompleteFailureHandler.
        5. Return the final ValidationDecision.

        Args:
            final_decision: FinalDecisionResult from Phase 2.6.

        Returns:
            :class:`ValidationDecision` for Phase 3.1.
        """
        start = time.perf_counter()
        logger.info("ValidationOrchestrator.validate() started")

        answer = CompleteFailureHandler._extract_answer(
            final_decision
        )
        steps = CompleteFailureHandler._extract_steps(final_decision)
        evidence = CompleteFailureHandler._extract_evidence(
            final_decision
        )

        layer1_result = self.analyzer.analyze(answer, steps, evidence)
        validation_decision = self.classifier.classify(
            layer1_result, final_decision
        )

        if validation_decision.result_class == "complete_failure":
            # Reset retry counter for fresh handling
            self.handler.reset_retry_count()
            validation_decision = self.handler.handle(
                validation_decision
            )

        elapsed = (time.perf_counter() - start) * 1000.0
        logger.info(
            "ValidationOrchestrator complete: class=%s time=%.1fms",
            validation_decision.result_class,
            elapsed,
        )
        return validation_decision
