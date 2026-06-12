# Robust Orchestration Q&A

## 1) What failures should trigger a fallback versus an immediate abort?

Use this decision rule:

- Trigger fallback for execution failures where an alternate implementation can still produce a safe, useful result.
- Abort immediately for failures that indicate unsafe continuation, invalid inputs, or policy/security violations.

Typical fallback candidates:
- Transient infrastructure faults: timeouts, temporary network failures, transport interruptions.
- Capacity failures: upstream 429/overload where a simpler model or cached path can still proceed.
- Partial dependency outage where degraded behavior is acceptable.

Typical immediate-abort candidates:
- Configuration/schema/validation errors (for example malformed payloads, missing required fields).
- Authentication/authorization failures (invalid credentials, permission denied).
- Deterministic business rule violations where retrying cannot change the outcome.

A useful heuristic is: if retry/fallback cannot change correctness, abort early.

## 2) How do circuit breakers prevent cascading incidents in a multi-step pipeline?

Circuit breakers stop repeated calls to an unhealthy dependency after a failure threshold is reached.

In a multi-step pipeline, this prevents:
- Resource exhaustion (threads, connections, tokens) from repeated failing calls.
- Queue buildup and latency amplification across downstream steps.
- Error fan-out where one failing service causes many later-stage failures.

Operationally:
- Closed: calls flow normally.
- Open: calls are rejected fast (fail-fast), protecting the pipeline.
- Half-open: limited probe calls test recovery; success closes, failure re-opens.

This isolation limits blast radius and gives dependent systems time to recover.

## 3) In what order should compensation actions run, and why?

Compensations should run in reverse order of successful execution (LIFO).

Why:
- Later steps usually depend on side effects from earlier steps.
- Undoing latest changes first removes dependencies safely before rolling back foundational changes.
- It mirrors transactional unwind semantics and minimizes inconsistent intermediate states.

Example:
- If steps execute A -> B -> C and C fails, compensate B then A.

## 4) When is exponential backoff preferable to a fixed retry delay?

Prefer exponential backoff when failures are likely transient and contention-sensitive.

Best cases:
- Shared services under load where immediate repeated retries worsen congestion.
- Rate-limited APIs where spacing requests increases success probability.
- Network instability where short recovery windows are common.

Benefits over fixed delay:
- Reduces retry storms and synchronized client spikes.
- Quickly retries once, then progressively gives systems time to recover.
- Typically improves overall success rate at lower system stress.

Use fixed delay mainly when the dependency has predictable recovery timing and low contention risk.
