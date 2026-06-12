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
from typing import Dict, Any, Optional, Protocol, runtime_checkable
from dataclasses import dataclass
from datetime import datetime
from pydantic_ai import Agent
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


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

    def record_success(self):
        """Record successful execution."""
        self.failures = 0
        self.is_open = False
        self.is_half_open = False
        self.probe_in_flight = False

    def record_failure(self):
        """Record failure."""
        self.failures += 1
        self.last_failure = datetime.now()

        # Any failure while half-open re-opens immediately.
        if self.is_half_open or self.failures >= self.failure_threshold:
            self.is_open = True
            self.is_half_open = False
            self.probe_in_flight = False

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

        for attempt in range(attempts):
            attempt_number = attempt + 1

            try:
                result = await asyncio.wait_for(
                    self.agent.run(prompt), timeout=self.timeout_seconds
                )
                self.circuit_breaker.record_success()
                return result.output

            except asyncio.TimeoutError:
                if attempt < self.max_retries:
                    print(f"  Timeout on attempt {attempt_number}/{attempts}; retrying...")
                else:
                    print(f"  Timeout on final attempt {attempt_number}/{attempts}")

            except Exception as exc:
                if attempt < self.max_retries:
                    print(f"  Error on attempt {attempt_number}/{attempts}: {exc}")
                    await asyncio.sleep(2**attempt)
                else:
                    print(f"  Error on final attempt {attempt_number}/{attempts}: {exc}")

        self.circuit_breaker.record_failure()

        if self.fallback_agent is not None:
            print("  Using fallback agent...")
            try:
                fallback_result = await asyncio.wait_for(
                    self.fallback_agent.run(prompt), timeout=self.timeout_seconds
                )
                return fallback_result.output
            except asyncio.TimeoutError:
                print("  Fallback timed out")
            except Exception as exc:
                print(f"  Fallback failed: {exc}")

        raise RuntimeError(f"Step {self.name} failed completely")

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


# Example workflow
async def example():
    """Demonstrate robust workflow."""

    # Create deterministic local agents for reliable demonstration.
    primary_agent = EchoAgent()
    fallback_agent = EchoAgent()

    # Build workflow
    workflow = RobustWorkflowOrchestrator("Data Processing Pipeline")

    workflow.add_step(ExtractSagaStep(agent=primary_agent, fallback_agent=fallback_agent))
    workflow.add_step(TransformSagaStep(agent=primary_agent, fallback_agent=fallback_agent))
    workflow.add_step(SummarizeSagaStep(agent=AlwaysFailAgent(), fallback_agent=None))

    # Demonstrate early contract rejection: missing compensate() is blocked before run.
    class InvalidExecuteOnlyStep:
        name = "invalid"

        async def execute(self, context: Dict[str, Any]) -> str:
            return "should never run"

    try:
        workflow.add_step(InvalidExecuteOnlyStep())
    except TypeError as exc:
        print(f"Contract check: rejected invalid step registration ({exc})")

    # Execute
    result = await workflow.execute(
        {
            "input_data": "Sample customer feedback data...",
            "active_resources": 1,
        }
    )

    print("\nFinal Result:")
    print(f"Status: {result['_metadata']['status']}")
    print(f"Steps completed: {result['_metadata']['steps_completed']}")
    print(f"Rollback order: {result.get('rollback_order', [])}")


if __name__ == "__main__":
    asyncio.run(example())
