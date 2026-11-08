# ADR-002: Local Open-Weight LLM Strategy

## Status
Proposed — Benchmarking Required

## Context

DataPilot AI should support local inference and avoid
mandatory dependence on paid hosted LLM APIs.

## Candidate

Qwen3.5-4B.

## Evaluation Criteria

- Structured output reliability.
- Tool-calling compatibility.
- Python generation quality.
- SQL generation quality.
- Statistical reasoning.
- Inference speed.
- Memory consumption.
- Model license.

## Decision

Use Qwen3.5-4B as the initial evaluation candidate.

Do not treat it as the final production model until
runtime compatibility and benchmark results are available.

## Consequences

- Local inference is the preferred deployment strategy.
- Model providers must remain replaceable.
- Model quality will be measured using a fixed benchmark.
