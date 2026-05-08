# ADR-0002: Use append-only revision logging in SQLite memory

- **Status:** Accepted
- **Date:** 2026-05-08

## Context

The system must preserve a complete epistemic trail showing how and why beliefs
changed over time. Overwriting beliefs without durable history would make
auditing and rollback reasoning difficult.

## Decision

Persist current belief state in SQLite while appending immutable
`BeliefRevision` records for each change.

## Consequences

- Full traceability for each belief transition.
- Auditable linkage from evidence to revision rationale.
- Additional storage and schema complexity compared to overwrite-only models.
