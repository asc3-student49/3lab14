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
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from pydantic_ai import Agent
import os
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

    def record_success(self):
        """Record successful execution."""
        self.failures = 0
        self.is_open = False

    def record_failure(self):
        """Record failure."""
        self.failures += 1
        self.last_failure = datetime.now()

        if self.failures >= self.failure_threshold:
            self.is_open = True

    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        if not self.is_open:
            return True

        # Check if timeout has passed
        if self.last_failure:
            elapsed = (datetime.now() - self.last_failure).total_seconds()
            if elapsed >= self.timeout_seconds:
                # Half-open state - allow one attempt
                self.is_open = False
                self.failures = self.failure_threshold - 1
                return True

        return False


class RobustWorkflowStep:
    """Workflow step with error handling."""

    def __init__(
        self,
        name: str,
        agent: Agent,
        prompt_template: str,
        fallback_agent: Optional[Agent] = None,
        compensation_action: Optional[Callable] = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
    ):
        self.name = name
        self.agent = agent
        self.prompt_template = prompt_template
        self.fallback_agent = fallback_agent
        self.compensation_action = compensation_action
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.circuit_breaker = CircuitBreaker()

    async def execute(self, context: Dict[str, Any]) -> str:
        """Execute step with error handling."""
        # Check circuit breaker
        if not self.circuit_breaker.can_execute():
            raise RuntimeError(f"Circuit breaker open for {self.name}")

        # Format prompt
        prompt = self.prompt_template.format(**context)

        # Try main agent with retries
        for attempt in range(self.max_retries + 1):
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    self.agent.run(prompt), timeout=self.timeout_seconds
                )

                self.circuit_breaker.record_success()
                return result.output

            except asyncio.TimeoutError:
                if attempt < self.max_retries:
                    print(f"  ⏱ Timeout on attempt {attempt + 1}, retrying...")
                    continue
                else:
                    print(f"  ⏱ Timeout after {self.max_retries + 1} attempts")
                    break

            except Exception as e:
                if attempt < self.max_retries:
                    print(f"  ⚠ Error on attempt {attempt + 1}: {e}")
                    await asyncio.sleep(2**attempt)  # Exponential backoff
                    continue
                else:
                    print(f"  ✗ Failed after {self.max_retries + 1} attempts")
                    break

        # Record failure
        self.circuit_breaker.record_failure()

        # Try fallback agent
        if self.fallback_agent:
            print(f"  🔄 Using fallback agent...")
            try:
                result = await asyncio.wait_for(
                    self.fallback_agent.run(prompt), timeout=self.timeout_seconds
                )
                return result.output

            except Exception as e:
                print(f"  ✗ Fallback also failed: {e}")

        raise RuntimeError(f"Step {self.name} failed completely")

    async def compensate(self, context: Dict[str, Any]):
        """Execute compensation action."""
        if self.compensation_action:
            print(f"  ↶ Running compensation for {self.name}")
            await self.compensation_action(context)


class RobustWorkflowOrchestrator:
    """Orchestrator with comprehensive error handling."""

    def __init__(self, name: str):
        self.name = name
        self.steps: list[RobustWorkflowStep] = []
        self.executed_steps: list[RobustWorkflowStep] = []

    def add_step(self, step: RobustWorkflowStep):
        """Add step to workflow."""
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
        if not self.executed_steps:
            return

        print(f"\n{'='*60}")
        print(f"  Rolling Back {len(self.executed_steps)} Steps")
        print(f"{'='*60}\n")

        # Compensate in reverse order
        for step in reversed(self.executed_steps):
            await step.compensate(context)

        self.executed_steps.clear()


# Example workflow
async def example():
    """Demonstrate robust workflow."""

    # Create agents
    primary_agent = Agent(
        os.getenv("AI_MODEL", "openai:gpt-5.4-mini"),
        system_prompt="You are a helpful assistant.",
    )

    fallback_agent = Agent(
        "test",  # Use test model as fallback
        system_prompt="You are a simple fallback assistant.",
    )

    # Compensation actions
    async def compensate_step1(ctx):
        print("    Undoing step 1...")
        await asyncio.sleep(0.5)

    async def compensate_step2(ctx):
        print("    Undoing step 2...")
        await asyncio.sleep(0.5)

    # Build workflow
    workflow = RobustWorkflowOrchestrator("Data Processing Pipeline")

    workflow.add_step(
        RobustWorkflowStep(
            name="extract",
            agent=primary_agent,
            prompt_template="Extract key information from: {input_data}",
            fallback_agent=fallback_agent,
            compensation_action=compensate_step1,
            timeout_seconds=10.0,
            max_retries=2,
        )
    )

    workflow.add_step(
        RobustWorkflowStep(
            name="transform",
            agent=primary_agent,
            prompt_template="Transform this data: {extract_result}",
            fallback_agent=fallback_agent,
            compensation_action=compensate_step2,
            timeout_seconds=10.0,
            max_retries=2,
        )
    )

    workflow.add_step(
        RobustWorkflowStep(
            name="summarize",
            agent=primary_agent,
            prompt_template="Summarize: {transform_result}",
            fallback_agent=fallback_agent,
            timeout_seconds=10.0,
            max_retries=2,
        )
    )

    # Execute
    result = await workflow.execute({"input_data": "Sample customer feedback data..."})

    print("\nFinal Result:")
    print(f"Status: {result['_metadata']['status']}")
    print(f"Steps completed: {result['_metadata']['steps_completed']}")


if __name__ == "__main__":
    asyncio.run(example())
