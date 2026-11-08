# ADR-001: Hybrid Agent Architecture

## Status
Accepted

## Context

DataPilot AI requires autonomous research planning,
analytical execution, evidence verification and reporting.

A single monolithic agent would make responsibilities,
testing and execution control difficult to manage.

## Decision

Use a hybrid architecture consisting of:

- A central LangGraph orchestrator.
- Specialized analytical agents.
- Deterministic Python execution tools.
- Structured workflow state.
- Human approval checkpoints for risky actions.

## Consequences

Positive:
- Clear separation of responsibilities.
- Independent testing of agents.
- Better workflow observability.
- Controlled autonomy.
- Replaceable model providers.

Trade-offs:
- More architectural complexity.
- Additional workflow coordination.
- State management requires careful design.

## Implementation

Implement incrementally, beginning with a single
orchestrated workflow before adding specialized agents.
