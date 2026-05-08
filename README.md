# swarm

EPISTEME is an epistemic foundation model architecture for representing beliefs,
evaluating evidence, and maintaining auditable revision history.

## Purpose

This repository provides a compact Python implementation of an epistemic system
where:

- beliefs are explicit typed objects with confidence and lifecycle state,
- evidence is validated before it can influence beliefs,
- update rules are enforced by a side-effect-free core,
- revisions are persisted with immutable history for auditability.

## Repository layout

- `episteme/models.py` – core data models
- `episteme/core.py` – epistemic rule engine
- `episteme/memory.py` – SQLite-backed memory layer
- `episteme/interface.py` – language/LLM adapter
- `episteme/experience.py` – observation-to-revision loop
- `tests/` – unit tests for all modules
- `docs/adr/` – architectural decision records

## Architectural overview

The system is organized as five collaborating layers:

1. **Models** define belief, evidence, and revision primitives.
2. **Core** applies deterministic eligibility and state-transition rules.
3. **Memory** persists current belief state plus full revision history.
4. **Language Interface** allows claim extraction/formatting but does not write beliefs.
5. **Experience Loop** transforms observations into evidence and requests revisions.

### High-level flow

1. A claim or observation is produced.
2. Evidence is generated (or deduplicated) from that input.
3. `EpistemicCore` validates whether an update is eligible.
4. `EpistemicMemory` applies the update and appends a revision record.
5. Consumers query current beliefs and/or revision history.

## Getting started

### Requirements

- Python 3.11+

### Install

```bash
cd <repository-root>
python -m pip install -e ".[dev]"
```

## Development workflow

Run tests:

```bash
cd <repository-root>
python -m pytest
```

Run lint checks:

```bash
cd <repository-root>
ruff check .
```

## Architectural decision records (ADRs)

Design decisions are tracked in markdown under:

- `docs/adr/README.md`
