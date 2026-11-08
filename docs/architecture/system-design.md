# DataPilot AI — System Architecture

## 1. Overview

DataPilot AI is an autonomous, evidence-driven data research
and intelligence platform.

It accepts structured datasets and natural-language research
questions and produces reproducible analytical investigations.

The architecture follows a hybrid agent model:

- A central orchestrator manages the investigation.
- Specialized agents handle individual responsibilities.
- Deterministic Python tools perform analytical computations.
- The backend persists investigation state and evidence.
- The frontend visualizes the investigation as a Research Graph.

## 2. High-Level Architecture

The system contains five major layers.

### 2.1 Frontend Layer

Technology:
- React
- TypeScript
- Vite
- Tailwind CSS
- React Flow
- Plotly

Responsibilities:
- Dataset upload.
- Research question submission.
- Research Graph visualization.
- Execution status monitoring.
- Code and query inspection.
- Chart visualization.
- Findings and report inspection.

### 2.2 Backend Layer

Technology:
- Python
- FastAPI
- Pydantic
- SQLite

Responsibilities:
- API endpoints.
- Dataset metadata management.
- Request validation.
- Investigation persistence.
- Graph event delivery.
- Report management.

### 2.3 Agent Layer

Technology:
- LangGraph
- Local open-weight LLM
- Structured tool interfaces

Proposed specialized agents:

1. Planner Agent
   Converts research questions into structured plans.

2. Profiler Agent
   Interprets dataset profiles and identifies data quality issues.

3. SQL Analyst Agent
   Generates analytical SQL queries.

4. Python Analyst Agent
   Generates Python analysis code.

5. Statistics Agent
   Plans statistical procedures and interprets their results.

6. ML Analyst Agent
   Plans and evaluates machine learning experiments.

7. Critic Agent
   Reviews findings, evidence and analytical consistency.

8. Reporter Agent
   Produces structured research reports.

The central orchestrator controls routing, dependencies,
retries, state transitions and approval checkpoints.

### 2.4 Analytical Engine

Technology:
- Pandas
- NumPy
- DuckDB
- SciPy
- Statsmodels
- Scikit-learn
- XGBoost

Modules:

- ingestion/
- profiling/
- sql/
- statistics/
- machine_learning/
- visualization/
- verification/
- execution/

The analytical engine performs deterministic computations.
The LLM does not calculate or invent numerical results.

### 2.5 Evaluation Layer

Responsibilities:
- Model quality benchmarks.
- SQL correctness evaluation.
- Python execution evaluation.
- Statistical correctness tests.
- Evidence verification tests.
- Agent workflow reliability.
- Performance measurements.

## 3. Research Workflow

A typical investigation follows this sequence:

1. User uploads a dataset.
2. Backend validates the file.
3. Ingestion engine reads the dataset.
4. Profiler generates a dataset profile.
5. User submits a research question.
6. Planner generates an investigation plan.
7. Orchestrator creates workflow tasks.
8. Specialized agents select analytical operations.
9. Execution tools run approved SQL and Python.
10. Statistical and ML tools generate results.
11. Critic validates findings against execution evidence.
12. Backend persists findings and graph events.
13. Reporter generates the research report.
14. Frontend displays the completed Research Graph.

## 4. Research Graph

The Research Graph is the central workspace and primary
interface of DataPilot AI.

### Node Types

- Dataset
- Variable
- Question
- Hypothesis
- Operation
- Model
- Finding
- Evidence
- Report

### Node Lifecycle

Possible execution states:

- pending
- planned
- running
- awaiting_approval
- completed
- failed
- skipped

### Graph Relationships

Examples:

- Dataset contains Variable.
- Question produces Hypothesis.
- Hypothesis motivates Operation.
- Operation consumes Dataset.
- Operation produces Finding.
- Finding is supported by Evidence.
- Findings contribute to Report.

The graph is a visualization of persisted backend
investigation events and relationships.

React Flow is not the source of truth.

## 5. Evidence and Provenance

Every analytical finding should retain:

- Finding ID.
- Investigation ID.
- Input dataset identifiers.
- Source column identifiers.
- Executed SQL or Python.
- Execution parameters.
- Tool and software versions.
- Numerical outputs.
- Verification status.
- Creation timestamp.

Conclusions must be traceable to actual analytical outputs.

The system must distinguish correlation from causation
and must not present unsupported conclusions as facts.

## 6. Execution Security

Generated code must never have unrestricted host access.

The execution system will enforce:

- Isolated execution environments.
- Resource and time limits.
- Restricted filesystem access.
- Network access disabled by default.
- Approved dataset access.
- Read-only SQL operations for the MVP.
- Explicit approval for risky operations.

The LLM must not receive unrestricted Docker control,
shell access or host credentials.

## 7. Model Strategy

Initial candidate:
Qwen3.5-4B.

The model will be evaluated for:

- Structured output generation.
- Tool calling.
- SQL generation.
- Python generation.
- Statistical reasoning.
- Instruction following.
- Memory consumption.
- Inference speed.

The final model and inference runtime will be selected
after compatibility and quality benchmarking.

## 8. Persistence Strategy

Initial persistence:
SQLite.

The application will persist:

- Dataset metadata.
- Investigation records.
- Agent execution state.
- Graph nodes and edges.
- Findings and evidence references.
- Report metadata.

Large datasets and generated artifacts will be stored
separately from application metadata.

## 9. API Design

Initial API groups:

- /api/v1/datasets
- /api/v1/investigations
- /api/v1/graph
- /api/v1/executions
- /api/v1/reports
- /api/v1/health

API contracts will be defined using Pydantic schemas.

## 10. Development Principles

1. Deterministic computation over LLM guesswork.
2. Evidence-backed conclusions.
3. Reproducible investigations.
4. Modular and replaceable components.
5. Controlled autonomy.
6. Explicit execution permissions.
7. Testable agent workflows.
8. Observable execution traces.

## 11. Current Status

Architecture specification:
Initial draft.

Implementation:
Not yet started.

All proposed components must be validated through
incremental implementation and testing.
