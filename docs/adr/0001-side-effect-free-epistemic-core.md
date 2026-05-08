# ADR-0001: Keep epistemic rule evaluation side-effect free

- **Status:** Accepted
- **Date:** 2026-05-08

## Context

Belief updates require strict eligibility checks (new evidence, reliability
threshold, bounded confidence changes, and adversarial-pressure guards). If
these checks are mixed with persistence side effects, it becomes harder to test,
audit, and reason about update correctness.

## Decision

Keep `EpistemicCore` as a pure rule engine that evaluates and returns decisions
without writing to storage.

## Consequences

- Deterministic and testable update behavior.
- Separation of concerns between rule evaluation and persistence.
- Callers must always orchestrate writes through memory components after
  consulting the core.
