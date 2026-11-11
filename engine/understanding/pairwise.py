"""Bounded deterministic structural evidence for pairs of dataset columns."""

import hashlib
import json
import numbers
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from itertools import combinations
from typing import TypeAlias

import pandas as pd

from engine.understanding.models import (
    AnalysisScope,
    DatasetUnderstandingResult,
    PairwiseEvidence,
    UnderstandingWarning,
)

_METHOD_ID = "pairwise_structural_comparison"
_METHOD_VERSION = "1"
_MISSING = object()
_ScalarKey: TypeAlias = tuple[object, ...]
# Estimates normalization, comparison, set, and both dependency passes.
_ESTIMATED_CELL_OPERATIONS_PER_ROW_PAIR = 10


@dataclass(frozen=True)
class PairwiseEvidenceLimits:
    """Deterministic workload and future categorical-summary limits."""

    max_column_pairs: int = 4096
    max_sample_rows: int = 5000
    max_distinct_values: int = 10_000
    max_category_cardinality: int = 50
    max_contingency_cells: int = 2500
    max_cell_comparisons: int = 10_000_000

    def __post_init__(self) -> None:
        for field_name in (
            "max_column_pairs",
            "max_sample_rows",
            "max_distinct_values",
            "max_category_cardinality",
            "max_contingency_cells",
            "max_cell_comparisons",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{field_name} must be a positive integer.")


@dataclass(frozen=True)
class PairwiseEvidenceResult:
    pairwise_evidence: tuple[PairwiseEvidence, ...]
    warnings: tuple[UnderstandingWarning, ...]


DEFAULT_PAIRWISE_EVIDENCE_LIMITS = PairwiseEvidenceLimits()


def build_pairwise_evidence(
    dataframe: pd.DataFrame,
    result: DatasetUnderstandingResult,
    *,
    limits: PairwiseEvidenceLimits = DEFAULT_PAIRWISE_EVIDENCE_LIMITS,
) -> PairwiseEvidenceResult:
    """Build bounded pairwise observations from an already-loaded DataFrame.

    Values are used only transiently to compute equality, set overlap, and
    functional-dependency aggregates. No source values are returned or stored.
    Rows are sampled as a deterministic prefix when the row cap is exceeded.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")
    result = DatasetUnderstandingResult.model_validate(
        result.model_dump(mode="python")
    )
    if dataframe.shape[1] != len(result.columns):
        raise ValueError(
            "DataFrame columns do not match positional understanding columns."
        )
    if not isinstance(limits, PairwiseEvidenceLimits):
        raise TypeError("limits must be PairwiseEvidenceLimits.")

    if (
        result.analysis_scope.rows_analyzed is not None
        and result.analysis_scope.rows_analyzed != len(dataframe)
    ):
        raise ValueError(
            "DataFrame row count does not match the understanding scope."
        )
    available_rows = len(dataframe)
    total_rows = (
        result.analysis_scope.total_rows
        if result.analysis_scope.total_rows is not None
        else available_rows
    )
    if total_rows < available_rows:
        raise ValueError(
            "DataFrame contains more rows than the declared dataset scope."
        )
    rows_analyzed = min(available_rows, limits.max_sample_rows)
    scope_kind = (
        "full_dataset"
        if (
            result.analysis_scope.kind == "full_dataset"
            and rows_analyzed == total_rows
        )
        else "sample"
    )
    scope = AnalysisScope(
        kind=scope_kind,
        rows_analyzed=rows_analyzed,
        total_rows=total_rows,
    )
    if rows_analyzed == 0:
        warning_code = (
            "pairwise_empty_dataset"
            if total_rows == 0
            else "pairwise_empty_sample"
        )
        explanation = (
            "Pairwise evidence requires at least one row; no "
            "column comparisons were performed."
            if total_rows == 0
            else (
                "The provided data sample has no rows; no pairwise "
                "comparisons were performed."
            )
        )
        return PairwiseEvidenceResult(
            pairwise_evidence=(),
            warnings=(
                UnderstandingWarning(
                    warning_code=warning_code,
                    explanation=explanation,
                    affected_column_ids=[],
                    severity="info",
                ),
            ),
        )

    possible_pairs = len(result.columns) * (len(result.columns) - 1) // 2
    estimated_pair_work = max(
        1,
        _ESTIMATED_CELL_OPERATIONS_PER_ROW_PAIR * rows_analyzed,
    )
    budget_pairs = limits.max_cell_comparisons // estimated_pair_work
    pair_limit = min(limits.max_column_pairs, budget_pairs)
    pairs_to_process = min(possible_pairs, pair_limit)
    warnings: list[UnderstandingWarning] = []
    if rows_analyzed < available_rows:
        warnings.append(
            UnderstandingWarning(
                warning_code="pairwise_row_sampled",
                explanation=(
                    f"Pairwise calculations use the first {rows_analyzed} "
                    f"of {available_rows} available rows. This deterministic "
                    "prefix is not a random sample and may be biased."
                ),
                affected_column_ids=[
                    column.column_id for column in result.columns
                ],
                severity="low",
            )
        )
    if result.analysis_scope.kind == "sample":
        warnings.append(
            UnderstandingWarning(
                warning_code="pairwise_input_scope_sampled",
                explanation=(
                    "The input understanding result already covers a sample; "
                    "pairwise observations cannot represent unobserved rows."
                ),
                affected_column_ids=[
                    column.column_id for column in result.columns
                ],
                severity="info",
            )
        )
    if pairs_to_process < possible_pairs:
        warnings.append(
            UnderstandingWarning(
                warning_code="pairwise_comparison_limit_exceeded",
                explanation=(
                    f"Only the first {pairs_to_process} of {possible_pairs} "
                    "positional column pairs were processed because a pair "
                    "or deterministic work limit was reached."
                ),
                affected_column_ids=[
                    column.column_id for column in result.columns
                ],
                severity="info",
            )
        )
    if pairs_to_process == 0:
        return PairwiseEvidenceResult((), tuple(warnings))

    identity = result.dataset_fingerprint or result.dataset_id
    output: list[PairwiseEvidence] = []
    unsupported_pairs = 0
    high_cardinality_pairs = 0
    for pair_index, (left_position, right_position) in enumerate(
        combinations(range(len(result.columns)), 2)
    ):
        if pair_index >= pairs_to_process:
            break
        left_values, left_supported = _column_keys(
            dataframe.iloc[:rows_analyzed, left_position]
        )
        right_values, right_supported = _column_keys(
            dataframe.iloc[:rows_analyzed, right_position]
        )
        if not left_supported or not right_supported:
            unsupported_pairs += 1
            continue
        left_set = {value for value in left_values if value is not _MISSING}
        right_set = {value for value in right_values if value is not _MISSING}
        if (
            len(left_set) > limits.max_distinct_values
            or len(right_set) > limits.max_distinct_values
        ):
            high_cardinality_pairs += 1
            continue

        left_id = f"col_{left_position}"
        right_id = f"col_{right_position}"
        pair_ids = [left_id, right_id]
        non_missing_pairs = sum(
            left is not _MISSING and right is not _MISSING
            for left, right in zip(left_values, right_values, strict=True)
        )
        equality = all(
            left == right
            for left, right in zip(left_values, right_values, strict=True)
        )
        overlap = left_set & right_set
        limitations = []
        if non_missing_pairs < 10:
            limitations.append(
                "Fewer than ten complete-case rows are available for "
                "key-inclusion or functional-dependency hypotheses."
            )
        if scope_kind == "sample":
            limitations.append(
                "Observation covers only the deterministic prefix sample."
            )
        if non_missing_pairs < rows_analyzed:
            limitations.append(
                "Value overlap and functional-dependency summaries exclude "
                "rows missing either participating value."
            )
        if non_missing_pairs == 0:
            limitations.append(
                "No complete-case values are available for overlap or "
                "functional-dependency analysis."
            )

        output.append(
            _make_evidence(
                identity,
                pair_ids,
                "exact_column_equality",
                scope,
                dataset_id=result.dataset_id,
                dataset_fingerprint=result.dataset_fingerprint,
                sample_size=rows_analyzed,
                denominator=rows_analyzed,
                measured_value=equality,
                limitations=limitations,
            )
        )
        output.append(
            _make_evidence(
                identity,
                pair_ids,
                "value_overlap",
                scope,
                dataset_id=result.dataset_id,
                dataset_fingerprint=result.dataset_fingerprint,
                sample_size=non_missing_pairs,
                denominator=rows_analyzed,
                measured_summary={
                    "left_non_null_count": sum(
                        value is not _MISSING for value in left_values
                    ),
                    "left_unique_count": len(left_set),
                    "right_unique_count": len(right_set),
                    "right_non_null_count": sum(
                        value is not _MISSING for value in right_values
                    ),
                    "shared_unique_count": len(overlap),
                    "left_values_in_right": left_set <= right_set,
                    "right_values_in_left": right_set <= left_set,
                    "complete_case_rows": non_missing_pairs,
                },
                limitations=limitations,
            )
        )
        fd_summary = _functional_dependency_summary(
            left_values,
            right_values,
        )
        output.append(
            _make_evidence(
                identity,
                pair_ids,
                "functional_dependency",
                scope,
                dataset_id=result.dataset_id,
                dataset_fingerprint=result.dataset_fingerprint,
                sample_size=non_missing_pairs,
                denominator=rows_analyzed,
                measured_summary=fd_summary,
                limitations=limitations,
            )
        )
    if unsupported_pairs:
        warnings.append(
            UnderstandingWarning(
                warning_code="pairwise_unsupported_values",
                explanation=(
                    f"Pairwise evidence was omitted for {unsupported_pairs} "
                    "pair(s) containing non-scalar or unsupported values."
                ),
                affected_column_ids=[
                    column.column_id for column in result.columns
                ],
                severity="low",
            )
        )
    if high_cardinality_pairs:
        warnings.append(
            UnderstandingWarning(
                warning_code="pairwise_cardinality_limit_exceeded",
                explanation=(
                    f"Pairwise evidence was omitted for {high_cardinality_pairs} "
                    "pair(s) exceeding the distinct-value cardinality limit."
                ),
                affected_column_ids=[
                    column.column_id for column in result.columns
                ],
                severity="low",
            )
        )
    return PairwiseEvidenceResult(tuple(output), tuple(warnings))


def _column_keys(series: pd.Series) -> tuple[list[object], bool]:
    output: list[object] = []
    for value in series.tolist():
        if _is_missing(value):
            output.append(_MISSING)
            continue
        key = _scalar_key(value)
        if key is None:
            return [], False
        output.append(key)
    return output, True


def _is_missing(value: object) -> bool:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return pd.api.types.is_scalar(missing) and bool(missing)


def _scalar_key(value: object) -> _ScalarKey | None:
    if not pd.api.types.is_scalar(value):
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (AttributeError, ValueError):
            pass
    if isinstance(value, bool):
        return ("boolean", value)
    if isinstance(value, numbers.Number):
        if isinstance(value, complex):
            return ("complex", value.real, value.imag)
        try:
            return ("number", Decimal(str(value)))
        except (ArithmeticError, ValueError):
            return None
    if isinstance(value, str):
        return ("string", value)
    if isinstance(value, bytes):
        return ("bytes", value)
    if isinstance(value, (datetime, date, time)):
        return ("temporal", type(value).__name__, value.isoformat())
    return None


def _functional_dependency_summary(
    left_values: list[object],
    right_values: list[object],
) -> dict[str, int]:
    output: dict[str, int] = {}
    for determinant_name, determinant, dependent in (
        ("left_to_right", left_values, right_values),
        ("right_to_left", right_values, left_values),
    ):
        groups: dict[object, set[object]] = {}
        for determinant_value, dependent_value in zip(
            determinant,
            dependent,
            strict=True,
        ):
            if (
                determinant_value is _MISSING
                or dependent_value is _MISSING
            ):
                continue
            groups.setdefault(determinant_value, set()).add(dependent_value)
        output[f"{determinant_name}_determinant_count"] = len(groups)
        output[f"{determinant_name}_violation_count"] = sum(
            len(values) > 1 for values in groups.values()
        )
    return output


def _make_evidence(
    identity: str,
    column_ids: list[str],
    observation_type: str,
    scope: AnalysisScope,
    sample_size: int,
    denominator: int,
    *,
    dataset_id: str,
    dataset_fingerprint: str | None,
    measured_value: bool | None = None,
    measured_summary: dict[str, int | bool] | None = None,
    limitations: list[str],
) -> PairwiseEvidence:
    payload = json.dumps(
        [
            identity,
            column_ids,
            observation_type,
            _METHOD_ID,
            _METHOD_VERSION,
        ],
        separators=(",", ":"),
    )
    evidence_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:40]
    return PairwiseEvidence(
        evidence_id=f"pair_{evidence_id}",
        dataset_id=dataset_id,
        dataset_fingerprint=dataset_fingerprint,
        observation_type=observation_type,
        column_ids=column_ids,
        method_id=_METHOD_ID,
        method_version=_METHOD_VERSION,
        sample_size=sample_size,
        denominator=denominator,
        scope=scope,
        measured_value=measured_value,
        measured_summary=measured_summary or {},
        limitations=limitations,
    )
