# DataPilot AI

**An Autonomous, Evidence-Driven Data Research and Intelligence Platform.**

DataPilot AI transforms structured datasets and natural-language research questions into reproducible investigations.

## Vision

Build an intelligent research workspace that can:
- Explore and profile datasets.
- Plan research investigations.
- Generate and execute Python and SQL.
- Perform statistical analysis.
- Train and evaluate machine learning models.
- Verify analytical findings.
- Generate research reports.
- Visualize investigations through an interactive Research Graph.

## Core Features

- Multi-format dataset ingestion.
- Automated exploratory data analysis.
- Autonomous research planning.
- Python and SQL execution.
- Statistical hypothesis testing.
- Machine learning experiments.
- Evidence verification and provenance.
- Interactive Research Graph.
- Automated research reports.

## Technology Stack

| Layer | Technology |
|---|---|
| Frontend | React, TypeScript, Vite |
| UI | Tailwind CSS |
| Research Graph | React Flow |
| Backend | FastAPI |
| Agent Orchestration | LangGraph |
| LLM | Qwen3.5-4B candidate |
| Data Processing | Pandas, NumPy |
| Analytical SQL | DuckDB |
| Statistics | SciPy, Statsmodels |
| Machine Learning | Scikit-learn, XGBoost |
| Persistence | SQLite |
| Execution Isolation | Docker |

## Architecture

The system consists of five major layers:

1. Frontend — Research workspace and visualization.
2. Backend — APIs and application services.
3. Agent — Planning, orchestration and reasoning.
4. Engine — Deterministic analytical execution.
5. Evaluation — Benchmarks and quality measurement.

## Supported Dataset Formats

- CSV
- XLSX
- JSON
- Parquet

## Dataset Storage

Committed original dataset files count toward the configurable aggregate
`DATASET_STORAGE_MAX_BYTES` quota. Upload staging files are temporary and are
accounted separately by `DatasetStorageManager.get_staging_usage_bytes()`; they
do not count toward the committed-file quota, but they do consume real
filesystem space while an upload is in progress or abandoned.

Uploads use process-locked directories named
`.datapilot-staging-<uuid-hex>`. An interrupted upload releases its lock, so
`report_staging_directories()` can identify the remaining directory and report
its byte usage. Cleanup is explicit with
`remove_staging_directory(directory_name)`, refuses active uploads, and can be
retried if filesystem cleanup fails. This operation only accepts exact managed
directory names beneath the configured dataset storage root. It does not remove
committed dataset files.

Pre-recovery `.staging-<tempfile-suffix>` directories are reported with unknown
activity status because they have no process lock; they are not eligible for
cleanup based on age or naming alone.

## Dataset Read API

- `GET /api/v1/datasets?limit=50&offset=0` lists metadata-backed committed
  datasets in stable UUID order. `limit` is bounded to 100. Legacy files
  without sidecars are intentionally not listed or migrated. Invalid or
  unavailable sidecars are omitted and included in `unavailable_count`.
- `GET /api/v1/datasets/{dataset_id}` returns validated dataset metadata after
  checking the original file's integrity.
- `GET /api/v1/datasets/{dataset_id}/preview?limit=20&offset=0` returns bounded
  rows and indexed columns. The configurable `DATASET_PREVIEW_MAX_ROWS` has a
  default of 100 and an allowed range of 1–1,000. Duplicate source headers are
  preserved as separate indexed columns, with each row represented as a
  positional value array.

Metadata and preview requests validate the UUID, file presence and SHA-256
integrity. Listing also verifies file hashes before returning datasets; its
pagination bounds the response, but integrity checking scans all metadata-backed
datasets. Preview parses a temporary snapshot whose size and hash match the
metadata, then applies the existing format-specific resource checks,
row/column/cell limits, dataframe memory limit and processing deadline. API
responses never include server filesystem paths. Unknown datasets return 404;
invalid UUIDs return 422; invalid metadata or failed integrity checks return
409.

## Dataset Understanding Contracts

The canonical, versioned Pydantic contracts live in `engine/understanding/`.
`DatasetUnderstandingResult` groups column and relationship hypotheses,
aggregate evidence, and warnings without changing dataset upload or read APIs.
Evidence identifies its deterministic method, version, metric, measured
aggregate, scope, sample size or denominator, and reproducibility parameters.
Evidence references and positional column references are validated within each
result; raw cell values are not accepted as evidence measurements.

When available, `dataset_fingerprint` is the lowercase SHA-256 digest of the
original uploaded file, matching storage metadata. It is not a hash of a
DataFrame serialization. A missing original-file digest is represented as
`null`. Column IDs such as `col_0` are positional and scoped to the dataset
fingerprint and engine result; duplicate source column names therefore remain
distinct.

Hypotheses may carry an optional confidence estimate. The contract only
accepts calibrated probabilities with a calibration method/version and
evaluation reference; uncalibrated heuristic rankings are not probabilities
and should not be exposed as confidence. Inference and calibration algorithms
are outside these contracts. `understanding_schema_version`, `engine_version`,
and `profile_version` distinguish contract changes from engine and profiler
changes.

### Deterministic Evidence Adapter

`engine/understanding/evidence.py` converts existing
`profile_dataframe`, `classify_series`, `infer_series_types`, and
`detect_data_quality_issues` outputs into validated `EvidenceItem` records.
Stable method IDs identify those producers; `method_version="1"` versions the
adapter's mapping. Evidence IDs are deterministic for the dataset UUID,
original-file fingerprint when present, and observation key.
The source method IDs are `profile_dataframe`, `classify_series`,
`infer_series_types`, and `detect_data_quality_issues`.

The adapter maps dataset dimensions and duplicate-row/memory counts; column
row, missingness and uniqueness metrics; existing numeric summary statistics;
the profiler's bounded top-frequency counts; scalar type candidates,
parseability counts and ambiguity; profile classifications; and supported
quality findings for missing values, empty/constant/near-constant/identifier-
like columns, duplicate rows/headers, mixed or ambiguous types, potential
unparseable values and potential outliers. `col_N` is positional, matched to
profile entries by index.
Unique profile labels are used to map quality findings; findings with ambiguous
or invalid column references produce warnings rather than being attached to a
guessed column.

Categorical evidence contains frequency counts and rank only—the profiler's
raw top category values and quality descriptions are not copied. Unsupported
complex numeric statistics, unknown findings and unmappable columns are
reported as warnings. Pandas dtype labels and category identities are not
currently emitted as evidence. Unsupported future profile fields or quality
finding kinds also produce warnings. These records are observations of existing
engine behavior, not semantic types, roles, units, relationships, or confirmed
business meaning. No inference or LLM orchestration is implemented here.

### Semantic Type Hypotheses

`engine/understanding/semantic_types.py` adds conservative, evidence-linked
semantic candidates for numeric measurements, datetime values, Boolean values,
categorical values, and identifier-like columns. Numeric candidates require
existing numeric parseability plus a general profile classification and
numeric summary evidence, except ambiguous numeric/datetime columns, where the
existing date candidate is retained as an explicit alternative. Identifier
candidates reuse the profiler's `identifier_like` classification and are
clearly marked as heuristic, not confirmed keys.

The engine does not infer currency, percentage, ordinal, free-text meaning,
geography, duration, or nested-value semantics. It does not interpret column
names or raw categorical values. Type candidates that compete remain
ambiguous, confidence stays null, and no values are converted. Empty,
constant, near-constant, mixed, and unsupported complex columns do not receive
an invented semantic type.

### Field Roles and Relationship Discovery

`engine/understanding/field_roles.py` reuses semantic hypotheses and existing
profile evidence to propose these roles:

- `identifier_candidate` reuses the profiler's identifier-like semantic
  hypothesis and remains explicitly unconfirmed.
- `primary_key_candidate` requires that identifier signal plus a row count,
  zero missing values, and a distinct-value count matching the row count.
  Even when these checks pass, it remains a candidate—not a declared key.
- `measurement_field`, `temporal_field`, and `grouping_category` follow the
  corresponding numeric-measurement, datetime, categorical, or Boolean
  hypotheses. Ambiguous semantic candidates remain ambiguous role hypotheses.
- Target and feature roles are not inferred without an explicit analytical
  task or target designation. Free-text/descriptive and foreign-key roles are
  also withheld because the current evidence does not establish them.

All confidence values remain null. Role hypotheses retain their supporting
and contradictory evidence references and uncertainty explanations. Column
names are not used to invent roles. Multiple roles may coexist; for example,
an ambiguous numeric/date interpretation can produce both measurement and
temporal role candidates.

`engine/understanding/pairwise.py` adds an explicit adapter for an already
loaded DataFrame and its validated understanding result. It emits versioned
`PairwiseEvidence` records with two positional IDs, a deterministic ID,
method/version, scope, sample size, denominator, scalar result or bounded
numeric/boolean summary, and limitations. Evidence stores no source cell
values. IDs are derived from the dataset fingerprint (or dataset ID), ordered
positions, observation type, and method/version. It supports exact positional
equality, non-null distinct-value overlap and inclusion counts, plus
directional functional-dependency counts and violations on complete cases.
`discover_relationships_from_dataframe` is the explicit orchestration entry
point: it accepts the already-loaded positional frame, attaches validated
pairwise records and warnings, then calls the relationship consumer. It does
not load or reopen a dataset.

`engine/understanding/relationships.py` consumes only those validated
observations and can propose:

- `exact_duplicate_columns` when columns match across the full dataset, have
  at least three compared rows, and both contain more than one distinct value.
- `candidate_key_inclusion` when at least ten complete rows support inclusion
  of a multi-value domain in a complete, unique reference domain. It is not a
  declared foreign key or proof of common entity meaning.
- `candidate_functional_dependency` when at least ten complete rows support a
  non-constant, non-unique determinant mapping to a non-constant, non-unique
  dependent with no observed violations. The recorded violation count is the
  number of distinct determinant values associated with multiple dependent
  values, not the number of violating rows. This remains a candidate, not a
  business rule.

Each relationship references its pairwise evidence, keeps confidence null,
and states its limitations. The adapter may produce observations for sampled
rows, but relationship discovery does not promote sample-only observations to
full-dataset relationships. Names do not establish relationships. No numeric
correlation, categorical association, causal inference, temporal ordering,
cross-dataset join, or LLM interpretation is implemented.

Default resource limits are 4,096 column pairs, the first 5,000 rows,
10,000 distinct values per column, and 10,000,000 deterministic
cell-comparison work units. Pair order is positional; over-limit pairs are
omitted with explicit warnings. The row prefix is deterministic but not a
random sample and may be biased. The work-unit limit bounds planned
comparisons, not wall-clock time. Maximum categorical cardinality (50) and
contingency cells (2,500) are reserved limits; categorical association and
contingency evidence are not yet computed. High-cardinality structural pairs
are omitted with warnings. Missing values are excluded from overlap and
functional-dependency calculations and recorded as limitations; unsupported
non-scalar values are explicitly skipped.

These are conservative hypotheses, not declarations. Confidence calibration,
foreign-key declarations, numeric correlation, categorical association,
temporal ordering, and cross-dataset discovery remain unsupported.

### Value and Unit Interpretations

`engine/understanding/value_interpretations.py` adds deterministic
value-interpretation hypotheses from the validated evidence and semantic
hypotheses already present in a `DatasetUnderstandingResult`. It does not read
or parse source values, use column names as proof, change semantic-type
hypotheses, or transform data. Hypotheses carry positional column identity,
supporting and contradicting evidence references when available, method
version, analysis scope, sample size, uncertainty, limitations, and null
confidence.

Implemented, evidence-limited candidates are:

- `boolean_values` when existing semantic type evidence supports Boolean data;
  the particular encoding and domain meaning remain unknown.
- `numeric_string_values` when existing evidence supports string scalar values
  and both numeric and text type candidates. This remains ambiguous and does
  not convert or expose the strings.
- `mixed_scalar_values` when existing evidence reports heterogeneous scalar
  types. Mixed values are not automatically treated as malformed.
- `nonnegative_integer_count` when existing numeric-measurement, integer,
  numeric-candidate, and range evidence align. This is explicitly ambiguous:
  codes, indexes, years, and other encodings remain possible.
- `date_or_timestamp` when existing semantic evidence supports datetime. The
  engine does not repeat date parsing or distinguish date, timestamp,
  time-of-day, duration, or interval.

Observed numeric ranges can trigger warnings, but are not enough to infer a
proportion, percentage, score, or rating. Numeric summaries do not establish
currency or physical units. Existing missing counts cannot identify sentinel
values. Categorical evidence does not establish ordinal ordering. These
interpretations are unsupported until the evidence contract provides
appropriate observations. Integer counts versus continuous measurements,
duration versus ordinary numeric values, and numeric Boolean-like codes also
remain unresolved.

The engine reuses bounded profile, type-candidate, quality, and semantic-type
evidence; it adds no raw-value scan, category enumeration, or dependency.
Interpretations are hypotheses, not confirmed meanings, and do not imply a
value conversion. IDs are deterministic for the dataset identity, positional
column, and interpretation. Confidence remains null because no calibrated
evaluation process exists. Empty or insufficient evidence yields uncertainty
warnings or no interpretation rather than a fabricated result.

### Unified Understanding Pipeline and Synthetic Evaluation

`engine.understanding.run_understanding_pipeline` is the public orchestration
entry point. It accepts either:

- an already-loaded `pandas.DataFrame` plus a required `dataset_id`, with an
  optional original-content SHA-256 fingerprint, precomputed profiler output,
  quality findings, deterministic `generated_at`, and pairwise limits; or
- a validated `DatasetUnderstandingResult` to continue when no DataFrame is
  available.

The pipeline never opens a path or reloads an uploaded file. The caller owns
loading and must pass the same frame and dataset identity used for the evidence.
When a frame is supplied, the sequence is profile and quality analysis, evidence
adaptation, semantic types, field roles, bounded pairwise observations,
relationship discovery, and value interpretations. Existing modules perform
the inference; the pipeline only coordinates their inputs and outputs.

The result contains positional `col_N` columns, direct profile/quality evidence,
pairwise observations, semantic/role/value hypotheses, structural relationship
hypotheses, and warnings. Duplicate source headers remain separate positional
columns. Stage failures represented by ordinary `TypeError` or `ValueError`
validation failures discard that stage's partial output and add a stage-specific
warning; unexpected exception types propagate rather than being silently
suppressed. Missing profile, quality, or pairwise inputs are surfaced as
warnings and do not produce fabricated findings. Confidence remains null.
Output contains aggregates and source header names, but no raw cell values.
Supply `generated_at` to make the full serialized result repeatable across
runs; otherwise it is set to the current UTC time.

The deterministic synthetic integration suite exercises unfamiliar columns,
duplicate headers, numeric strings, Boolean and temporal values, missing and
constant values, duplicate/equivalent columns, candidate key inclusion,
resource limits, partial failures, invalid references, immutability, and
repeated execution. Its scorer (`score_labeled_outputs`) compares only
explicitly labeled positives and negatives. Intentionally withheld labels are
excluded from false negatives and are counted separately if emitted;
unlabeled hypotheses are not scored. A missing expected positive is a miss,
not a deliberate withholding.

For the current synthetic labels, the checked results are:

| Stage | True positives | False positives | False negatives | Withheld-label proposals | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Semantic types | 4 | 0 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| Field roles | 2 | 0 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| Pairwise equality | 1 | 0 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| Relationships | 2 | 0 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| Value interpretations | 3 | 0 | 0 | 0 | 1.00 | 1.00 | 1.00 |

These are micro-scores over the tests' small, explicitly labeled synthetic
subset—not estimates of generalization or real-world performance. Predictions
outside the explicitly labeled subset do not enter these scores. The benchmark
uses hand-constructed cases and is not an independent ground-truth corpus.
Passing it does not establish production readiness.

Future work includes a larger independently reviewed benchmark, additional
method-specific scoring of uncertainty and unsupported outputs, calibrated
confidence (only after calibration data exists), and optional numeric or
categorical association evidence. Correlation, causation, declared keys,
business meaning, transformations, and file-loading remain outside this
pipeline.

## Phase 1 Validation

Run the backend and deterministic engine test suite from the repository root:

```bash
.venv/bin/python -m pytest backend/tests engine/profiling engine/quality engine/understanding -q
```

The test directories are listed explicitly because pytest's default
`testpaths` setting currently includes only `backend/tests`. The four paths
are separate test roots and are each supplied once.

Run repository-wide linting and validate the current Git diff with:

```bash
.venv/bin/ruff check .
git diff --check
```

## MVP Roadmap

- [ ] Phase 0: Environment and architecture foundation
- [ ] Phase 1: Data ingestion and profiling
- [ ] Phase 2: LLM integration and research planning
- [ ] Phase 3: Agent orchestration and execution
- [ ] Phase 4: Statistical and ML intelligence
- [ ] Phase 5: Evidence verification
- [ ] Phase 6: Interactive Research Graph
- [ ] Phase 7: Reporting and integration
- [ ] Phase 8: Benchmarking and deployment

## Engineering Principles

- Evidence-backed conclusions.
- Reproducible analytical results.
- Modular and replaceable components.
- Controlled autonomous execution.
- Secure and isolated code execution.
- Transparent analytical provenance.

## Current Development Status

Phase 0 — Repository initialization.

The proposed technology stack and model choices will be validated through implementation and benchmarking.

## License

MIT
