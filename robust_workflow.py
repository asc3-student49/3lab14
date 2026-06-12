"""
Robust Workflow Orchestration with Error Handling and Fallbacks

Advanced orchestration with:
- Compensation actions for rollback
- Fallback strategies
- Circuit breakers
- Timeout management
- Comprehensive error recovery

Run: python robust_workflow.py
"""

import asyncio
import json
import inspect
from collections import defaultdict
from typing import Dict, Any, Optional, Protocol, runtime_checkable
from dataclasses import dataclass, field
from datetime import datetime
from pydantic_ai import Agent
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def is_retryable_error(exc: Exception) -> bool:
    """Classify whether an exception is safe to retry."""
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError, OSError)):
        return True

    if isinstance(exc, (ValueError, PermissionError)):
        return False

    error_type = exc.__class__.__name__.lower()
    error_message = str(exc).lower()

    non_retryable_markers = (
        "auth",
        "authentication",
        "permission",
        "credential",
        "validation",
        "schema",
        "config",
        "configuration",
    )
    if any(marker in error_type or marker in error_message for marker in non_retryable_markers):
        return False

    retryable_markers = ("timeout", "network", "transport", "connection", "temporary")
    if any(marker in error_type or marker in error_message for marker in retryable_markers):
        return True

    # Default to retryable for unknown operational failures.
    return True


@dataclass
class CircuitBreaker:
    """Circuit breaker to prevent cascading failures."""

    failure_threshold: int = 3
    timeout_seconds: float = 60.0

    failures: int = 0
    last_failure: Optional[datetime] = None
    is_open: bool = False
    is_half_open: bool = False
    probe_in_flight: bool = False
    total_calls: int = 0
    successful_calls: int = 0
    retryable_errors: int = 0
    non_retryable_errors: int = 0
    error_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def record_attempt(self):
        """Record an execution attempt for observability."""
        self.total_calls += 1

    def record_observed_error(self, error_type: str, retryable: bool):
        """Track error frequencies for future retry/circuit tuning."""
        self.error_counts[error_type] += 1
        if retryable:
            self.retryable_errors += 1
        else:
            self.non_retryable_errors += 1

    def record_success(self):
        """Record successful execution."""
        self.successful_calls += 1
        self.failures = 0
        self.is_open = False
        self.is_half_open = False
        self.probe_in_flight = False

    def record_failure(self, error_type: str, retryable: bool):
        """Record failure."""
        self.failures += 1
        self.last_failure = datetime.now()

        # Any failure while half-open re-opens immediately.
        if self.is_half_open or self.failures >= self.failure_threshold:
            self.is_open = True
            self.is_half_open = False
            self.probe_in_flight = False

    def snapshot(self) -> Dict[str, Any]:
        """Return lightweight diagnostics for retry-policy tuning."""
        success_rate = 0.0
        if self.total_calls > 0:
            success_rate = self.successful_calls / self.total_calls
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failures": self.failures,
            "retryable_errors": self.retryable_errors,
            "non_retryable_errors": self.non_retryable_errors,
            "success_rate": round(success_rate, 3),
            "error_counts": dict(self.error_counts),
        }

    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        # Closed state: execute normally.
        if not self.is_open and not self.is_half_open:
            return True

        # Open state: allow transition to half-open only after timeout window.
        if self.is_open:
            if not self.last_failure:
                return False

            elapsed = (datetime.now() - self.last_failure).total_seconds()
            if elapsed < self.timeout_seconds:
                return False

            # Transition to half-open and allow one probe call.
            self.is_open = False
            self.is_half_open = True
            self.probe_in_flight = False

        # Half-open state: allow exactly one in-flight probe.
        if self.is_half_open:
            if self.probe_in_flight:
                return False
            self.probe_in_flight = True
            return True

        return True


@runtime_checkable
class SagaStep(Protocol):
    """Contract for saga-aware steps with forward and compensating actions."""

    name: str

    async def execute(self, context: Dict[str, Any]) -> str:
        ...

    async def compensate(self, context: Dict[str, Any]) -> None:
        ...


class LLMSagaStep:
    """Reusable saga step implementation with retries, fallback, and circuit breaker."""

    def __init__(
        self,
        name: str,
        agent,
        prompt_template: str,
        fallback_agent=None,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
    ):
        self.name = name
        self.agent = agent
        self.prompt_template = prompt_template
        self.fallback_agent = fallback_agent
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.circuit_breaker = CircuitBreaker()

    async def execute(self, context: Dict[str, Any]) -> str:
        """Execute step with error handling."""
        if not self.circuit_breaker.can_execute():
            raise RuntimeError(f"Step {self.name} cannot execute: circuit breaker is open")

        prompt = self.prompt_template.format(**context)
        attempts = self.max_retries + 1
        self.circuit_breaker.record_attempt()

        metadata = context.setdefault("_metadata", {})
        step_diagnostics = metadata.setdefault("step_diagnostics", {})
        step_diagnostics[self.name] = {
            "attempts": 0,
            "retryable_errors": 0,
            "non_retryable_errors": 0,
            "final_status": "in_progress",
            "last_error_type": None,
            "last_error_message": None,
            "short_circuited_non_retryable": False,
            "used_fallback": False,
        }
        diag = step_diagnostics[self.name]

        last_error: Optional[Exception] = None
        last_error_type = "UnknownError"
        last_error_retryable = True

        for attempt in range(attempts):
            attempt_number = attempt + 1
            diag["attempts"] = attempt_number

            try:
                result = await asyncio.wait_for(
                    self.agent.run(prompt), timeout=self.timeout_seconds
                )
                self.circuit_breaker.record_success()
                diag["final_status"] = "success"
                diag["circuit_breaker"] = self.circuit_breaker.snapshot()
                return result.output

            except Exception as exc:
                last_error = exc
                last_error_type = exc.__class__.__name__
                last_error_retryable = is_retryable_error(exc)

                diag["last_error_type"] = last_error_type
                diag["last_error_message"] = str(exc)
                if last_error_retryable:
                    diag["retryable_errors"] += 1
                else:
                    diag["non_retryable_errors"] += 1

                self.circuit_breaker.record_observed_error(
                    error_type=last_error_type,
                    retryable=last_error_retryable,
                )

                if not last_error_retryable:
                    print(
                        f"  Non-retryable error on attempt {attempt_number}/{attempts}: {exc}; "
                        "skipping remaining retries"
                    )
                    diag["short_circuited_non_retryable"] = True
                    break

                if attempt < self.max_retries:
                    print(f"  Retryable error on attempt {attempt_number}/{attempts}: {exc}")
                    await asyncio.sleep(2**attempt)
                else:
                    print(f"  Retryable error on final attempt {attempt_number}/{attempts}: {exc}")

        self.circuit_breaker.record_failure(
            error_type=last_error_type,
            retryable=last_error_retryable,
        )

        if self.fallback_agent is not None:
            print("  Using fallback agent...")
            try:
                fallback_result = await asyncio.wait_for(
                    self.fallback_agent.run(prompt), timeout=self.timeout_seconds
                )
                diag["used_fallback"] = True
                diag["final_status"] = "fallback_success"
                diag["circuit_breaker"] = self.circuit_breaker.snapshot()
                return fallback_result.output
            except asyncio.TimeoutError:
                print("  Fallback timed out")
            except Exception as exc:
                print(f"  Fallback failed: {exc}")

        diag["final_status"] = "failed"
        diag["circuit_breaker"] = self.circuit_breaker.snapshot()
        raise RuntimeError(f"Step {self.name} failed completely") from last_error

    async def compensate(self, context: Dict[str, Any]):
        """Default no-op compensation for steps without side effects."""
        print(f"  ↶ No compensation required for {self.name}")


class ExtractSagaStep(LLMSagaStep):
    """Extract stage with concrete compensation behavior."""

    def __init__(self, agent, fallback_agent=None):
        super().__init__(
            name="extract",
            agent=agent,
            prompt_template="Extract key information from: {input_data}",
            fallback_agent=fallback_agent,
            timeout_seconds=10.0,
            max_retries=2,
        )

    async def compensate(self, context: Dict[str, Any]) -> None:
        print(f"  ↶ Running compensation for {self.name}")
        context["active_resources"] = context.get("active_resources", 0) - 1
        event = {
            "step": "extract",
            "action": "resource_decrement",
            "active_resources": context["active_resources"],
            "timestamp": datetime.now().isoformat(),
        }

        context.setdefault("rollback_order", []).append("extract")
        print("    Undo extract: decremented active_resources")

        artifacts = Path("rollback_artifacts")
        artifacts.mkdir(exist_ok=True)
        with (artifacts / "undo_extract.json").open("w", encoding="utf-8") as f:
            json.dump(event, f, indent=2)

        await asyncio.sleep(0.2)


class TransformSagaStep(LLMSagaStep):
    """Transform stage with concrete compensation behavior."""

    def __init__(self, agent, fallback_agent=None):
        super().__init__(
            name="transform",
            agent=agent,
            prompt_template="Transform this data: {extract_result}",
            fallback_agent=fallback_agent,
            timeout_seconds=10.0,
            max_retries=2,
        )

    async def compensate(self, context: Dict[str, Any]) -> None:
        print(f"  ↶ Running compensation for {self.name}")
        event = {
            "step": "transform",
            "action": "reversal_event_posted",
            "timestamp": datetime.now().isoformat(),
        }

        context.setdefault("reversal_events", []).append(event)
        context.setdefault("rollback_order", []).append("transform")
        print("    Undo transform: posted simulated reversal event")

        artifacts = Path("rollback_artifacts")
        artifacts.mkdir(exist_ok=True)
        with (artifacts / "reversal_events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

        await asyncio.sleep(0.2)


class SummarizeSagaStep(LLMSagaStep):
    """Summary stage that intentionally fails to demonstrate rollback."""

    def __init__(self, agent, fallback_agent=None):
        super().__init__(
            name="summarize",
            agent=agent,
            prompt_template="Summarize: {transform_result}",
            fallback_agent=fallback_agent,
            timeout_seconds=10.0,
            max_retries=2,
        )


class FakeResult:
    """Simple result wrapper matching the pydantic_ai Agent result shape."""

    def __init__(self, output: str):
        self.output = output


class EchoAgent:
    """Agent stub that always succeeds with deterministic output."""

    async def run(self, prompt: str):
        return FakeResult(f"OK: {prompt}")


class AlwaysFailAgent:
    """Agent stub that always fails to trigger rollback."""

    async def run(self, prompt: str):
        raise RuntimeError("Forced summarize failure for rollback validation")


class FlakyTransientAgent:
    """Fails with a transient transport error before eventually succeeding."""

    def __init__(self, failures_before_success: int = 1):
        self.failures_before_success = failures_before_success
        self.calls = 0

    async def run(self, prompt: str):
        self.calls += 1
        if self.calls <= self.failures_before_success:
            raise ConnectionError("Simulated transient transport failure")
        return FakeResult(f"Recovered on call {self.calls}: {prompt}")


class FatalValidationAgent:
    """Always fails with a non-retryable validation/configuration error."""

    async def run(self, prompt: str):
        raise ValueError("Invalid configuration schema for step input")


class RobustWorkflowOrchestrator:
    """Orchestrator with comprehensive error handling."""

    def __init__(self, name: str):
        self.name = name
        self.steps: list[SagaStep] = []
        self.executed_steps: list[SagaStep] = []

    @staticmethod
    def _is_valid_saga_step(step: object) -> bool:
        """Ensure execute/compensate are both implemented as coroutines."""
        execute_method = getattr(step, "execute", None)
        compensate_method = getattr(step, "compensate", None)
        return (
            hasattr(step, "name")
            and callable(execute_method)
            and callable(compensate_method)
            and inspect.iscoroutinefunction(execute_method)
            and inspect.iscoroutinefunction(compensate_method)
        )

    def add_step(self, step: SagaStep):
        """Add step to workflow with upfront saga-contract validation."""
        if not self._is_valid_saga_step(step):
            raise TypeError(
                "Invalid saga step. Each step must provide async execute(context) "
                "and async compensate(context) methods."
            )
        self.steps.append(step)
        return self

    async def execute(self, initial_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow with error handling."""
        context = initial_context.copy()
        self.executed_steps.clear()
        start_time = datetime.now()
        context["_metadata"] = {
            "start_time": start_time,
            "steps_completed": [],
            "steps_failed": [],
        }

        print(f"\n{'='*60}")
        print(f"Starting workflow: {self.name}")
        print(f"{'='*60}")

        try:
            total_steps = len(self.steps)
            for i, step in enumerate(self.steps, 1):
                print(f"\nStep {i}/{total_steps}: {step.name}")

                try:
                    output = await step.execute(context)
                    context[f"{step.name}_result"] = output
                    context["_metadata"]["steps_completed"].append(step.name)
                    self.executed_steps.append(step)
                except Exception:
                    context["_metadata"]["steps_failed"].append(step.name)
                    await self._rollback(context)
                    context["_metadata"]["end_time"] = datetime.now()
                    context["_metadata"]["status"] = "failed"
                    return context

            context["_metadata"]["end_time"] = datetime.now()
            context["_metadata"]["status"] = "completed"
            duration = (
                context["_metadata"]["end_time"] - context["_metadata"]["start_time"]
            ).total_seconds()
            print(f"\nWorkflow completed in {duration:.2f}s")
            return context

        except Exception:
            await self._rollback(context)
            raise

    async def _rollback(self, context: Dict[str, Any]):
        """Rollback executed steps."""
        if len(self.executed_steps) == 0:
            return

        print(f"\n{'='*60}")
        print(f"  Rolling Back {len(self.executed_steps)} Steps")
        print(f"{'='*60}\n")

        for step in reversed(self.executed_steps):
            await step.compensate(context)

        self.executed_steps.clear()


async def demo_transient_retry_success() -> None:
    """Demonstrate transient retry path with exponential backoff and eventual success."""
    print("\n" + "=" * 60)
    print("Demo: Transient error retries and succeeds")
    print("=" * 60)

    context: Dict[str, Any] = {"payload": "customer_feedback_chunk"}
    step = LLMSagaStep(
        name="transient_demo",
        agent=FlakyTransientAgent(failures_before_success=1),
        prompt_template="Process payload: {payload}",
        fallback_agent=EchoAgent(),
        timeout_seconds=5.0,
        max_retries=3,
    )

    output = await step.execute(context)
    diagnostics = context["_metadata"]["step_diagnostics"]["transient_demo"]
    print(f"Result: {output}")
    print(f"Transient diagnostics: {json.dumps(diagnostics, indent=2)}")


async def demo_fatal_abort_to_fallback() -> None:
    """Demonstrate non-retryable path that short-circuits retries and goes to fallback."""
    print("\n" + "=" * 60)
    print("Demo: Fatal error aborts retries immediately")
    print("=" * 60)

    context: Dict[str, Any] = {"payload": "customer_feedback_chunk"}
    step = LLMSagaStep(
        name="fatal_demo",
        agent=FatalValidationAgent(),
        prompt_template="Validate payload: {payload}",
        fallback_agent=EchoAgent(),
        timeout_seconds=5.0,
        max_retries=3,
    )

    output = await step.execute(context)
    diagnostics = context["_metadata"]["step_diagnostics"]["fatal_demo"]
    print(f"Result (from fallback): {output}")
    print(f"Fatal diagnostics: {json.dumps(diagnostics, indent=2)}")


# Example workflow
async def example():
    """Demonstrate retry classification behavior."""
    await demo_transient_retry_success()
    await demo_fatal_abort_to_fallback()


if __name__ == "__main__":
    asyncio.run(example())
