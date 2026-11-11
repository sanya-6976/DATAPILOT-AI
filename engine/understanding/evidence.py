"""Adapt existing deterministic profile and quality outputs into evidence."""

import hashlib
import json
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from engine.understanding.models import (
    AnalysisScope,
    EvidenceItem,
    MeasuredValue,
    UnderstandingWarning,
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_UUID_HEX_PATTERN = re.compile(r"^[0-9a-fA-F]{32}$")
_UUID_DASHED_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

_PROFILE_CLASSIFICATIONS = {
    "empty",
    "constant",
    "near_constant",
    "identifier_like",
    "general",
}
_SCALAR_TYPES = {
    "boolean",
    "bytes",
    "categorical",
    "complex",
    "date",
    "datetime",
    "datetime64",
    "decimal",
    "empty",
    "floating",
    "integer",
    "integer_na",
    "mixed",
    "mixed_integer",
    "mixed_integer_float",
    "string",
    "time",
    "timedelta64",
    "period",
    "unknown_array",
    "unicode",
}
_CANDIDATE_TYPES = {
    "boolean",
    "complex",
    "datetime",
    "numeric",
    "text",
}
_PROFILE_METRICS = {
    "total_values": "column_total_values",
    "missing_values": "column_missing_count",
    "missing_percentage": "column_missing_percentage",
    "unique_values": "column_unique_count",
}
_CLASSIFICATION_METRICS = {
    "non_missing_values": "classification_non_missing_count",
    "unique_values": "classification_unique_count",
    "dominant_value_percentage": "classification_dominant_percentage",
    "threshold_percentage": "classification_threshold_percentage",
    "minimum_values": "classification_minimum_values",
    "uniqueness_ratio": "classification_uniqueness_ratio",
}
_TYPE_COUNT_METRICS = {
    "numeric_parseable_values": (
        "numeric_parseable_count",
        "numeric_parseable_values",
        "numeric_unparseable_values",
    ),
    "numeric_unparseable_values": (
        "numeric_unparseable_count",
        "numeric_parseable_values",
        "numeric_unparseable_values",
    ),
    "datetime_parseable_values": (
        "datetime_parseable_count",
        "datetime_parseable_values",
        "datetime_unparseable_values",
    ),
    "datetime_unparseable_values": (
        "datetime_unparseable_count",
        "datetime_parseable_values",
        "datetime_unparseable_values",
    ),
}
_QUALITY_FINDINGS = {
    "ambiguous_type": "quality_ambiguous_type",
    "constant_column": "quality_constant_column",
    "duplicate_column_names": "quality_duplicate_column_names",
    "duplicate_rows": "quality_duplicate_rows",
    "empty_column": "quality_empty_column",
    "empty_dataset": "quality_empty_dataset",
    "identifier_like_column": "quality_identifier_like_column",
    "mixed_types": "quality_mixed_types",
    "missing_values": "quality_missing_values",
    "near_constant_column": "quality_near_constant_column",
    "potential_outliers": "quality_potential_outliers",
    "potential_unparseable_values": "quality_potential_unparseable_values",
}
_COLUMN_QUALITY_FINDINGS = set(_QUALITY_FINDINGS) - {
    "duplicate_rows",
    "empty_dataset",
}
_MAX_CATEGORICAL_FREQUENCIES = 5
_ADAPTER_METHOD_VERSION = "1"
_OVERVIEW_FIELDS = {
    "total_rows",
    "total_columns",
    "duplicate_rows",
    "memory_usage_bytes",
}
_COLUMN_PROFILE_FIELDS = {
    "column_name",
    "data_type",
    "inferred_type",
    "type_inference",
    "column_classification",
    "classification_details",
    "total_values",
    "missing_values",
    "missing_percentage",
    "unique_values",
    "statistics",
    "statistics_status",
    "top_values",
    "source_column_name",
    "column_index",
}
_TYPE_INFERENCE_FIELDS = {
    "scalar_type",
    "candidate_types",
    "ambiguous",
    *_TYPE_COUNT_METRICS,
}
_QUALITY_FINDING_FIELDS = {
    "issue",
    "column",
    "column_index",
    "count",
    "percentage",
    "severity",
    "description",
}


@dataclass(frozen=True)
class ColumnIdentity:
    """Position-based identity plus names used by current engine outputs."""

    position: int
    source_name: str
    profile_name: str

    def __post_init__(self) -> None:
        if type(self.position) is not int or self.position < 0:
            raise ValueError("Column position must be a non-negative integer.")
        if not isinstance(self.source_name, str):
            raise TypeError("source_name must be a string.")
        if not isinstance(self.profile_name, str):
            raise TypeError("profile_name must be a string.")

    @property
    def column_id(self) -> str:
        return f"col_{self.position}"


@dataclass(frozen=True)
class EvidenceAdapterResult:
    """Validated evidence and warnings produced for one profile result."""

    evidence: tuple[EvidenceItem, ...]
    warnings: tuple[UnderstandingWarning, ...]
    analysis_scope: AnalysisScope


def adapt_profile_and_quality_evidence(
    *,
    dataset_id: str,
    dataset_fingerprint: str | None,
    profile: dict[str, Any],
    quality_findings: list[dict[str, Any]],
    columns: Sequence[ColumnIdentity],
) -> EvidenceAdapterResult:
    """Convert existing profiler and quality outputs into typed evidence.

    The adapter only records values already present in those outputs. It does
    not inspect source cells, infer semantic meanings, or expose category
    labels from the profiler's ``top_values`` list.
    """

    if not isinstance(profile, dict):
        raise TypeError("Profile output must be a mapping.")
    if not isinstance(quality_findings, list):
        raise TypeError("Quality findings must be a list.")
    normalized_dataset_id = _validate_dataset_identity(
        dataset_id,
        dataset_fingerprint,
    )
    column_identities = _validate_column_identities(columns)
    overview = profile.get("dataset_overview")
    profile_columns = profile.get("columns")
    if not isinstance(overview, dict) or not isinstance(profile_columns, list):
        raise TypeError("Profile output has an unsupported structure.")
    if len(profile_columns) != len(column_identities):
        raise ValueError("Profile columns do not match positional metadata.")

    rows = _non_negative_int(overview.get("total_rows"), "total_rows")
    declared_columns = _non_negative_int(
        overview.get("total_columns"),
        "total_columns",
    )
    if declared_columns != len(column_identities):
        raise ValueError("Profile column count does not match metadata.")

    scope = AnalysisScope(
        kind="full_dataset",
        rows_analyzed=rows,
        total_rows=rows,
    )
    evidence: list[EvidenceItem] = []
    warnings: list[UnderstandingWarning] = []
    evidence_keys: set[str] = set()

    def emit(
        *,
        key: str,
        kind: str,
        method_id: str,
        affected_column_ids: list[str],
        metric_name: str,
        measured_value: MeasuredValue = None,
        measured_category: str | None = None,
        sample_size: int | None = None,
        denominator: int | None = None,
        parameters: dict[str, int | float | bool | None] | None = None,
    ) -> None:
        evidence_id = _evidence_id(
            normalized_dataset_id,
            dataset_fingerprint,
            key,
        )
        if evidence_id in evidence_keys:
            raise ValueError("Deterministic evidence ID collision.")
        evidence_keys.add(evidence_id)
        evidence.append(
            EvidenceItem(
                evidence_id=evidence_id,
                dataset_id=normalized_dataset_id,
                dataset_fingerprint=dataset_fingerprint,
                kind=kind,
                method_id=method_id,
                method_version=_ADAPTER_METHOD_VERSION,
                affected_column_ids=affected_column_ids,
                metric_name=metric_name,
                measured_category=measured_category,
                measured_value=measured_value,
                sample_size=sample_size,
                denominator=denominator,
                scope=scope,
                parameters=parameters or {},
            )
        )

    def warn(
        code: str,
        explanation: str,
        affected_column_ids: list[str] | None = None,
    ) -> None:
        warnings.append(
            UnderstandingWarning(
                warning_code=code,
                explanation=explanation,
                affected_column_ids=affected_column_ids or [],
                severity="info",
            )
        )

    _warn_unrecognized_fields(
        overview,
        _OVERVIEW_FIELDS,
        warn,
        warning_code="unsupported_dataset_profile_field",
    )
    dataset_metrics = (
        ("total_rows", "dataset_row_count"),
        ("total_columns", "dataset_column_count"),
        ("duplicate_rows", "dataset_duplicate_row_count"),
        ("memory_usage_bytes", "dataset_memory_usage_bytes"),
    )
    for field_name, metric_name in dataset_metrics:
        if field_name not in overview:
            continue
        value = _non_negative_int(overview[field_name], field_name)
        emit(
            key=f"profile:dataset:{metric_name}",
            kind="profile_metric",
            method_id="profile_dataframe",
            affected_column_ids=[],
            metric_name=metric_name,
            measured_value=value,
        )

    for position, (column, column_profile) in enumerate(
        zip(column_identities, profile_columns, strict=True)
    ):
        if not isinstance(column_profile, dict):
            raise TypeError("Profile column entry must be an object.")
        _validate_profile_column(column, column_profile, position)
        column_id = column.column_id
        _warn_unrecognized_fields(
            column_profile,
            _COLUMN_PROFILE_FIELDS,
            warn,
            affected_column_ids=[column_id],
            warning_code="unsupported_profile_field",
        )

        for field_name, metric_name in _PROFILE_METRICS.items():
            if field_name not in column_profile:
                continue
            value = column_profile[field_name]
            if field_name == "missing_percentage":
                value = _finite_number(value, field_name)
            else:
                value = _non_negative_int(value, field_name)
            emit(
                key=f"profile:{column_id}:{metric_name}",
                kind="profile_metric",
                method_id="profile_dataframe",
                affected_column_ids=[column_id],
                metric_name=metric_name,
                measured_value=value,
                sample_size=rows,
                denominator=(
                    rows if field_name == "missing_percentage" else None
                ),
            )

        _emit_numeric_mapping(
            emit,
            profile_values=column_profile.get("statistics"),
            mapping={
                "mean": "numeric_mean",
                "median": "numeric_median",
                "std": "numeric_standard_deviation",
                "min": "numeric_minimum",
                "max": "numeric_maximum",
            },
            key_prefix=f"profile:{column_id}:statistics",
            method_id="profile_dataframe",
            column_id=column_id,
            sample_size=_optional_non_negative_int(
                column_profile.get("classification_details", {}).get(
                    "non_missing_values"
                )
                if isinstance(column_profile.get("classification_details"), dict)
                else None
            ),
        )
        statistics = column_profile.get("statistics")
        if statistics is not None and not isinstance(statistics, dict):
            warn(
                "unsupported_profile_statistics",
                "Numeric statistics were not provided as a mapping.",
                [column.column_id],
            )
        if isinstance(statistics, dict):
            unsupported_statistics = set(statistics) - {
                "mean",
                "median",
                "std",
                "min",
                "max",
            }
            if unsupported_statistics:
                warn(
                    "unsupported_profile_statistic",
                    "One or more profile statistics were not recognized by "
                    "the evidence adapter.",
                    [column.column_id],
                )

        details = column_profile.get("classification_details")
        classification = column_profile.get("column_classification")
        if classification is not None:
            if (
                not isinstance(classification, str)
                or classification not in _PROFILE_CLASSIFICATIONS
            ):
                warn(
                    "unsupported_profile_classification",
                    "A profile classification was not recognized by the "
                    "evidence adapter.",
                    [column_id],
                )
            else:
                emit(
                    key=f"profile:{column_id}:classification",
                    kind="deterministic_rule",
                    method_id="classify_series",
                    affected_column_ids=[column_id],
                    metric_name="profile_classification",
                    measured_category=classification,
                    sample_size=(
                        _optional_non_negative_int(
                            details.get("non_missing_values")
                        )
                        if isinstance(details, dict)
                        else None
                    ),
                )

        if details is not None and not isinstance(details, dict):
            warn(
                "unsupported_profile_details",
                "Classification details were not a supported mapping.",
                [column_id],
            )
        elif isinstance(details, dict):
            _warn_unrecognized_fields(
                details,
                set(_CLASSIFICATION_METRICS) | {
                    "identifier_evidence",
                },
                warn,
                affected_column_ids=[column_id],
                warning_code="unsupported_classification_detail",
            )
            _emit_numeric_mapping(
                emit,
                profile_values=details,
                mapping=_CLASSIFICATION_METRICS,
                key_prefix=f"profile:{column_id}:classification_details",
                method_id="classify_series",
                column_id=column_id,
                sample_size=_optional_non_negative_int(
                    details.get("non_missing_values")
                ),
            )

        type_inference = column_profile.get("type_inference")
        if type_inference is not None and not isinstance(type_inference, dict):
            warn(
                "unsupported_type_inference",
                "Type inference output was not a supported mapping.",
                [column_id],
            )
        elif isinstance(type_inference, dict):
            _warn_unrecognized_fields(
                type_inference,
                _TYPE_INFERENCE_FIELDS,
                warn,
                affected_column_ids=[column_id],
                warning_code="unsupported_type_inference_field",
            )
            scalar_type = type_inference.get("scalar_type")
            if scalar_type is not None:
                normalized_type = _normalize_scalar_type(scalar_type)
                if normalized_type is None:
                    warn(
                        "unsupported_scalar_type",
                        "A scalar type label was not recognized by the "
                        "evidence adapter.",
                        [column_id],
                    )
                else:
                    emit(
                        key=f"profile:{column_id}:scalar_type",
                        kind="deterministic_rule",
                        method_id="infer_series_types",
                        affected_column_ids=[column_id],
                        metric_name="inferred_scalar_type",
                        measured_category=normalized_type,
                        sample_size=max(
                            rows
                            - _non_negative_int(
                                column_profile.get("missing_values", 0),
                                "missing_values",
                            ),
                            0,
                        ),
                    )

            candidates = type_inference.get("candidate_types", [])
            if not isinstance(candidates, list):
                warn(
                    "unsupported_type_candidates",
                    "Type candidates were not provided as a list.",
                    [column_id],
                )
            else:
                for candidate in candidates:
                    if (
                        not isinstance(candidate, str)
                        or candidate not in _CANDIDATE_TYPES
                    ):
                        warn(
                            "unsupported_type_candidate",
                            "A type candidate was not recognized by the "
                            "evidence adapter.",
                            [column_id],
                        )
                        continue
                    emit(
                        key=f"profile:{column_id}:type_candidate:{candidate}",
                        kind="deterministic_rule",
                        method_id="infer_series_types",
                        affected_column_ids=[column_id],
                        metric_name="type_candidate_observed",
                        measured_category=candidate,
                        sample_size=max(
                            rows
                            - _non_negative_int(
                                column_profile.get("missing_values", 0),
                                "missing_values",
                            ),
                            0,
                        ),
                    )

            if "ambiguous" in type_inference:
                ambiguous = type_inference["ambiguous"]
                if type(ambiguous) is bool:
                    emit(
                        key=f"profile:{column_id}:type_ambiguity",
                        kind="deterministic_rule",
                        method_id="infer_series_types",
                        affected_column_ids=[column_id],
                        metric_name="type_ambiguity_observed",
                        measured_value=ambiguous,
                        sample_size=max(
                            rows
                            - _non_negative_int(
                                column_profile.get("missing_values", 0),
                                "missing_values",
                            ),
                            0,
                        ),
                    )
                else:
                    warn(
                        "invalid_type_ambiguity",
                        "Type ambiguity was not a boolean observation.",
                        [column_id],
                    )

            for field_name, (
                metric_name,
                parseable_field,
                unparseable_field,
            ) in _TYPE_COUNT_METRICS.items():
                if field_name not in type_inference:
                    continue
                parseable_count = _non_negative_int(
                    type_inference[parseable_field],
                    parseable_field,
                )
                unparseable_count = _non_negative_int(
                    type_inference[unparseable_field],
                    unparseable_field,
                )
                attempt_count = parseable_count + unparseable_count
                emit(
                    key=f"profile:{column_id}:type_inference:{metric_name}",
                    kind="profile_metric",
                    method_id="infer_series_types",
                    affected_column_ids=[column_id],
                    metric_name=metric_name,
                    measured_value=_non_negative_int(
                        type_inference[field_name],
                        field_name,
                    ),
                    sample_size=attempt_count,
                    denominator=attempt_count,
                )

        top_values = column_profile.get("top_values")
        if top_values is not None:
            if not isinstance(top_values, list):
                warn(
                    "unsupported_categorical_summary",
                    "Categorical frequency output was not a list.",
                    [column_id],
                )
            else:
                if len(top_values) > _MAX_CATEGORICAL_FREQUENCIES:
                    warn(
                        "categorical_summary_truncated",
                        "Categorical frequency evidence was limited to the "
                        "first five profiler entries.",
                        [column_id],
                    )
                for rank, entry in enumerate(
                    top_values[:_MAX_CATEGORICAL_FREQUENCIES],
                    start=1,
                ):
                    if (
                        not isinstance(entry, dict)
                        or "count" not in entry
                    ):
                        warn(
                            "unsupported_categorical_entry",
                            "A categorical frequency entry lacked a count.",
                            [column_id],
                        )
                        continue
                    count = _non_negative_int(
                        entry["count"],
                        "top_values.count",
                    )
                    frequency_denominator = max(
                        rows
                        - _non_negative_int(
                            column_profile.get("missing_values", 0),
                            "missing_values",
                        ),
                        0,
                    )
                    emit(
                        key=f"profile:{column_id}:top_frequency:{rank}",
                        kind="profile_metric",
                        method_id="profile_dataframe",
                        affected_column_ids=[column_id],
                        metric_name="categorical_top_value_count",
                        measured_value=count,
                        sample_size=frequency_denominator,
                        denominator=frequency_denominator,
                        parameters={"rank": rank},
                    )

        if column_profile.get("statistics_status") == "unsupported_complex":
            warn(
                "unsupported_complex_statistics",
                "The profiler does not provide numeric summary statistics "
                "for complex-valued columns.",
                [column_id],
            )

    _adapt_quality_findings(
        quality_findings=quality_findings,
        columns=column_identities,
        rows=rows,
        emit=emit,
        warn=warn,
    )
    return EvidenceAdapterResult(
        evidence=tuple(evidence),
        warnings=tuple(warnings),
        analysis_scope=scope,
    )


def adapt_quality_evidence(
    *,
    dataset_id: str,
    dataset_fingerprint: str | None,
    quality_findings: list[dict[str, Any]],
    columns: Sequence[ColumnIdentity],
    rows: int,
) -> EvidenceAdapterResult:
    """Adapt independently produced quality findings without profile output."""

    if not isinstance(quality_findings, list):
        raise TypeError("Quality findings must be a list.")
    normalized_dataset_id = _validate_dataset_identity(
        dataset_id,
        dataset_fingerprint,
    )
    column_identities = _validate_column_identities(columns)
    row_count = _non_negative_int(rows, "rows")
    scope = AnalysisScope(
        kind="full_dataset",
        rows_analyzed=row_count,
        total_rows=row_count,
    )
    evidence: list[EvidenceItem] = []
    warnings: list[UnderstandingWarning] = []
    evidence_keys: set[str] = set()

    def emit(
        *,
        key: str,
        kind: str,
        method_id: str,
        affected_column_ids: list[str],
        metric_name: str,
        measured_value: MeasuredValue = None,
        measured_category: str | None = None,
        sample_size: int | None = None,
        denominator: int | None = None,
        parameters: dict[str, int | float | bool | None] | None = None,
    ) -> None:
        evidence_id = _evidence_id(
            normalized_dataset_id,
            dataset_fingerprint,
            key,
        )
        if evidence_id in evidence_keys:
            raise ValueError("Deterministic evidence ID collision.")
        evidence_keys.add(evidence_id)
        evidence.append(
            EvidenceItem(
                evidence_id=evidence_id,
                dataset_id=normalized_dataset_id,
                dataset_fingerprint=dataset_fingerprint,
                kind=kind,
                method_id=method_id,
                method_version=_ADAPTER_METHOD_VERSION,
                affected_column_ids=affected_column_ids,
                metric_name=metric_name,
                measured_category=measured_category,
                measured_value=measured_value,
                sample_size=sample_size,
                denominator=denominator,
                scope=scope,
                parameters=parameters or {},
            )
        )

    def warn(
        code: str,
        explanation: str,
        affected_column_ids: list[str] | None = None,
    ) -> None:
        warnings.append(
            UnderstandingWarning(
                warning_code=code,
                explanation=explanation,
                affected_column_ids=affected_column_ids or [],
                severity="info",
            )
        )

    _adapt_quality_findings(
        quality_findings=quality_findings,
        columns=column_identities,
        rows=row_count,
        emit=emit,
        warn=warn,
    )
    return EvidenceAdapterResult(
        evidence=tuple(evidence),
        warnings=tuple(warnings),
        analysis_scope=scope,
    )


def _adapt_quality_findings(
    *,
    quality_findings: list[dict[str, Any]],
    columns: list[ColumnIdentity],
    rows: int,
    emit: Any,
    warn: Any,
) -> None:
    if not isinstance(quality_findings, list):
        raise TypeError("Quality findings must be a list.")

    for finding_index, finding in enumerate(quality_findings):
        if not isinstance(finding, dict):
            warn(
                "unsupported_quality_finding",
                "A quality finding was not a supported mapping.",
            )
            continue
        _warn_unrecognized_fields(
            finding,
            _QUALITY_FINDING_FIELDS,
            warn,
            warning_code="unsupported_quality_finding_field",
        )
        issue = finding.get("issue")
        if not isinstance(issue, str):
            warn(
                "unsupported_quality_finding",
                "A quality finding kind was not recognized by the adapter.",
            )
            continue
        metric_name = _QUALITY_FINDINGS.get(issue)
        if metric_name is None:
            warn(
                "unsupported_quality_finding",
                "A quality finding kind was not recognized by the adapter.",
            )
            continue

        affected_columns: list[str] = []
        if issue in _COLUMN_QUALITY_FINDINGS:
            affected_columns = _resolve_quality_column(finding, columns)
            if not affected_columns:
                warn(
                    "unmappable_quality_finding",
                    "A column quality finding could not be linked to a "
                    "unique positional column.",
                )
                continue
        elif finding.get("column") is not None:
            affected_columns = _resolve_quality_column(finding, columns)
            if not affected_columns:
                warn(
                    "unmappable_quality_finding",
                    "A dataset quality finding contained an invalid column "
                    "reference.",
                )
                continue

        base_key = (
            f"quality:{finding_index}:{issue}:"
            f"{','.join(affected_columns)}"
        )
        severity = finding.get("severity")
        if severity is not None:
            if not isinstance(severity, str) or severity not in {
                "info",
                "low",
                "medium",
                "high",
            }:
                warn(
                    "unsupported_quality_severity",
                    "A quality finding severity was not recognized.",
                    affected_columns,
                )
            else:
                emit(
                    key=f"{base_key}:severity",
                    kind="deterministic_rule",
                    method_id="detect_data_quality_issues",
                    affected_column_ids=affected_columns,
                    metric_name="quality_severity",
                    measured_category=severity,
                )

        if "count" in finding:
            count = _non_negative_int(finding["count"], "quality.count")
            population_size = (
                len(columns)
                if issue == "duplicate_column_names"
                else rows
            )
            emit(
                key=f"{base_key}:count",
                kind="observed_fact",
                method_id="detect_data_quality_issues",
                affected_column_ids=affected_columns,
                metric_name=f"{metric_name}_count",
                measured_value=count,
                sample_size=population_size,
                denominator=population_size,
            )
        else:
            emit(
                key=f"{base_key}:presence",
                kind="observed_fact",
                method_id="detect_data_quality_issues",
                affected_column_ids=affected_columns,
                metric_name=f"{metric_name}_present",
                measured_value=True,
                sample_size=rows,
            )

        if "percentage" in finding:
            percentage = _finite_number(
                finding["percentage"],
                "quality.percentage",
            )
            percentage_denominator = (
                rows
                if issue in {"missing_values", "empty_column"}
                else _non_negative_int(
                    finding["count"],
                    "quality.count",
                )
            )
            emit(
                key=f"{base_key}:percentage",
                kind="observed_fact",
                method_id="detect_data_quality_issues",
                affected_column_ids=affected_columns,
                metric_name=f"{metric_name}_percentage",
                measured_value=percentage,
                sample_size=percentage_denominator,
                denominator=percentage_denominator,
            )


def _resolve_quality_column(
    finding: dict[str, Any],
    columns: list[ColumnIdentity],
) -> list[str]:
    explicit_position = finding.get("column_index")
    if explicit_position is not None:
        if (
            type(explicit_position) is not int
            or explicit_position < 0
            or explicit_position >= len(columns)
        ):
            raise ValueError("Quality finding has an invalid column position.")
        return [columns[explicit_position].column_id]

    name = finding.get("column")
    if not isinstance(name, str):
        return []
    source_matches = [
        column for column in columns if column.source_name == name
    ]
    if issue_is_duplicate_column_name(finding):
        return [column.column_id for column in source_matches]

    profile_matches = [
        column for column in columns if column.profile_name == name
    ]
    if len(profile_matches) == 1:
        return [profile_matches[0].column_id]
    if len(profile_matches) > 1:
        return []

    if len(source_matches) == 1:
        return [source_matches[0].column_id]
    return []


def issue_is_duplicate_column_name(finding: dict[str, Any]) -> bool:
    return finding.get("issue") == "duplicate_column_names"


def _warn_unrecognized_fields(
    values: dict[str, Any],
    recognized_fields: set[str],
    warn: Any,
    *,
    warning_code: str,
    affected_column_ids: list[str] | None = None,
) -> None:
    if set(values) - recognized_fields:
        warn(
            warning_code,
            "One or more output fields were not recognized by the evidence "
            "adapter.",
            affected_column_ids,
        )


def _validate_profile_column(
    identity: ColumnIdentity,
    profile: dict[str, Any],
    expected_position: int,
) -> None:
    if identity.position != expected_position:
        raise ValueError("Column metadata must use contiguous positions.")
    profile_position = profile.get("column_index")
    if profile_position is not None and (
        type(profile_position) is not int
        or profile_position != expected_position
    ):
        raise ValueError("Profile column index does not match metadata.")
    profile_name = profile.get("column_name")
    if profile_name is not None and profile_name != identity.profile_name:
        raise ValueError("Profile column name does not match metadata.")
    source_name = profile.get("source_column_name")
    if source_name is not None and source_name != identity.source_name:
        raise ValueError("Profile source name does not match metadata.")


def _validate_column_identities(
    columns: Sequence[ColumnIdentity],
) -> list[ColumnIdentity]:
    identities = list(columns)
    if any(not isinstance(item, ColumnIdentity) for item in identities):
        raise ValueError("columns must contain ColumnIdentity records.")
    if any(item.position != index for index, item in enumerate(identities)):
        raise ValueError("Column metadata must use contiguous positions.")
    return identities


def _validate_dataset_identity(
    dataset_id: str,
    fingerprint: str | None,
) -> str:
    if not isinstance(dataset_id, str) or not (
        _UUID_HEX_PATTERN.fullmatch(dataset_id)
        or _UUID_DASHED_PATTERN.fullmatch(dataset_id)
    ):
        raise ValueError("dataset_id must be a valid UUID.")
    try:
        normalized_id = uuid.UUID(dataset_id).hex
    except ValueError as exc:
        raise ValueError("dataset_id must be a valid UUID.") from exc
    if fingerprint is not None and (
        not isinstance(fingerprint, str)
        or _SHA256_PATTERN.fullmatch(fingerprint) is None
    ):
        raise ValueError("dataset_fingerprint must be lowercase SHA-256.")
    return normalized_id


def _evidence_id(
    dataset_id: str,
    fingerprint: str | None,
    key: str,
) -> str:
    identity = json.dumps(
        [dataset_id, fingerprint, key],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return f"ev_{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:40]}"


def _emit_numeric_mapping(
    emit: Any,
    *,
    profile_values: Any,
    mapping: dict[str, str],
    key_prefix: str,
    method_id: str,
    column_id: str,
    sample_size: int | None,
) -> None:
    if profile_values is None:
        return
    if not isinstance(profile_values, dict):
        return
    for source_name, metric_name in mapping.items():
        if source_name not in profile_values:
            continue
        value = profile_values[source_name]
        if value is None:
            continue
        if source_name in {
            "non_missing_values",
            "unique_values",
            "minimum_values",
            "numeric_parseable_values",
            "numeric_unparseable_values",
            "datetime_parseable_values",
            "datetime_unparseable_values",
        }:
            measured_value: MeasuredValue = _non_negative_int(value, source_name)
        else:
            measured_value = _finite_number(value, source_name)
        emit(
            key=f"{key_prefix}:{metric_name}",
            kind="profile_metric",
            method_id=method_id,
            affected_column_ids=[column_id],
            metric_name=metric_name,
            measured_value=measured_value,
            sample_size=sample_size,
        )


def _normalize_scalar_type(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.replace("-", "_")
    return normalized if normalized in _SCALAR_TYPES else None


def _non_negative_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return value


def _optional_non_negative_int(value: Any) -> int | None:
    if value is None:
        return None
    return _non_negative_int(value, "sample_size")


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric.")
    if not (float("-inf") < float(value) < float("inf")):
        raise ValueError(f"{field_name} must be finite.")
    return float(value)
