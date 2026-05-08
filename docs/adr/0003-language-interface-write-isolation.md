# ADR-0003: Keep language interface write-isolated from belief storage

- **Status:** Accepted
- **Date:** 2026-05-08

## Context

LLM output can be useful for proposing claims and hypotheses, but allowing direct
writes from language output to belief storage would bypass epistemic safeguards
and increase hallucination risk.

## Decision

Constrain `LanguageInterface` to extraction and formatting responsibilities only.
Belief mutations must flow through `EpistemicCore` and `EpistemicMemory`.

## Consequences

- Reduced risk of language-only belief drift.
- Clear trust boundary around write operations.
- Slightly more orchestration code for callers integrating LLM output.
