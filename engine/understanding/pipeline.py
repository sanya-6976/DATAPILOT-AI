"""Controlled orchestration for the dataset understanding stages."""

from collections.abc import Callable
from datetime import UTC, datetime

import pandas as pd

from engine.profiling.profiler import profile_dataframe
from engine.quality.data_quality import detect_data_quality_issues
from engine.understanding.evidence import (
    ColumnIdentity,
    adapt_profile_and_quality_evidence,
    adapt_quality_evidence,
)
from engine.understanding.field_roles import infer_field_roles
from engine.understanding.models import (
    AnalysisScope,
    ColumnUnderstanding,
    DatasetUnderstandingResult,
    UnderstandingWarning,
)
from engine.understanding.pairwise import (
    DEFAULT_PAIRWISE_EVIDENCE_LIMITS,
    PairwiseEvidenceLimits,
    build_pairwise_evidence,
)
from engine.understanding.relationships import discover_relationships
from engine.understanding.semantic_types import infer_semantic_types
from engine.understanding.value_interpretations import (
    infer_value_interpretations,
)

_SCHEMA_VERSION = "1.0"
_ENGINE_VERSION = "0.1.0"
_PROFILE_VERSION = "1"
_EXPECTED_STAGE_ERRORS = (TypeError, ValueError, RuntimeError, ArithmeticError)


def run_understanding_pipeline(
    dataframe: pd.DataFrame | None = None,
    *,
    dataset_id: str | None = None,
    dataset_fingerprint: str | None = None,
    existing_result: DatasetUnderstandingResult | None = None,
    generated_at: datetime | None = None,
    pairwise_limits: PairwiseEvidenceLimits = (
        DEFAULT_PAIRWISE_EVIDENCE_LIMITS
    ),
) -> DatasetUnderstandingResult:
    """Run supported stages on an in-memory frame or an existing result.

    The caller owns all file loading. If a frame is supplied, profile and
    quality evidence are computed unless precomputed outputs are supplied.
    An existing result may be continued without a frame; pairwise evidence
    then remains unavailable unless it is already present.
    """

    if dataframe is not None and not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame or None.")
    if not isinstance(pairwise_limits, PairwiseEvidenceLimits):
        raise TypeError("pairwise_limits must be PairwiseEvidenceLimits.")
    if existing_result is not None:
        if dataframe is not None:
            raise ValueError(
                "Provide either a DataFrame or existing_result, not both."
            )
        if dataset_id is not None or dataset_fingerprint is not None:
            raise ValueError(
                "Dataset identity must come from existing_result when it "
                "is provided."
            )
        result = DatasetUnderstandingResult.model_validate(
            existing_result.model_dump(mode="python")
        )
    else:
        if dataframe is None:
            raise ValueError(
                "Provide an in-memory dataframe or existing_result."
            )
        if dataset_id is None:
            raise ValueError("dataset_id is required for a new DataFrame.")
        result = _build_initial_result(
            dataframe,
            dataset_id=dataset_id,
            dataset_fingerprint=dataset_fingerprint,
            generated_at=generated_at or datetime.now(UTC),
        )

    if dataframe is None:
        if not result.evidence:
            result = _append_warning(
                result,
                UnderstandingWarning(
                    warning_code="profile_quality_evidence_unavailable",
                    explanation=(
                        "No profile or quality evidence is attached; "
                        "evidence-dependent hypotheses are withheld."
                    ),
                    severity="info",
                ),
            )
        if not result.pairwise_evidence and len(result.columns) > 1:
            result = _append_warning(
                result,
                UnderstandingWarning(
                    warning_code="pairwise_input_unavailable",
                    explanation=(
                        "No in-memory DataFrame was supplied and no pairwise "
                        "evidence is attached; pairwise comparisons were "
                        "withheld."
                    ),
                    affected_column_ids=[
                        column.column_id for column in result.columns
                    ],
                    severity="info",
                ),
            )

    result = _run_result_stage(
        result,
        "semantic_type_stage_failed",
        "Semantic-type inference could not be completed; its partial "
        "hypotheses were discarded.",
        infer_semantic_types,
    )
    result = _run_result_stage(
        result,
        "field_role_stage_failed",
        "Field-role inference could not be completed; its partial "
        "hypotheses were discarded.",
        infer_field_roles,
    )

    if dataframe is not None:
        result = _run_result_stage(
            result,
            "pairwise_evidence_stage_failed",
            "Pairwise evidence generation could not be completed; no "
            "partial pairwise observations were attached.",
            lambda current: _add_pairwise_evidence(
                current,
                dataframe,
                pairwise_limits,
            ),
        )
    elif not result.pairwise_evidence:
        result = _append_warning(
            result,
            UnderstandingWarning(
                warning_code="pairwise_evidence_unavailable",
                explanation=(
                    "Pairwise observations were not generated because no "
                    "loaded DataFrame was supplied."
                ),
                affected_column_ids=[
                    column.column_id for column in result.columns
                ],
                severity="info",
            ),
        )

    result = _run_result_stage(
        result,
        "relationship_stage_failed",
        "Relationship discovery could not be completed; its partial "
        "hypotheses were discarded.",
        discover_relationships,
    )
    result = _run_result_stage(
        result,
        "value_interpretation_stage_failed",
        "Value interpretation could not be completed; its partial "
        "hypotheses were discarded.",
        infer_value_interpretations,
    )
    return DatasetUnderstandingResult.model_validate(
        result.model_dump(mode="python")
    )


def _build_initial_result(
    dataframe: pd.DataFrame,
    *,
    dataset_id: str,
    dataset_fingerprint: str | None,
    generated_at: datetime,
) -> DatasetUnderstandingResult:
    names = [str(column) for column in dataframe.columns]
    columns = [
        ColumnIdentity(
            position=position,
            source_name=name,
            profile_name=name,
        )
        for position, name in enumerate(names)
    ]
    warnings: list[UnderstandingWarning] = []
    current_profile = None
    try:
        current_profile = profile_dataframe(dataframe)
    except _EXPECTED_STAGE_ERRORS as exc:
        warnings.append(
            UnderstandingWarning(
                warning_code="profile_stage_failed",
                explanation=(
                    "Profiling failed with "
                    f"{type(exc).__name__}; profile-derived evidence was "
                    "withheld."
                ),
                affected_column_ids=[
                    f"col_{position}"
                    for position in range(dataframe.shape[1])
                ],
                severity="high",
            )
        )

    current_quality = None
    try:
        current_quality = detect_data_quality_issues(dataframe)
    except _EXPECTED_STAGE_ERRORS as exc:
        warnings.append(
            UnderstandingWarning(
                warning_code="quality_stage_failed",
                explanation=(
                    "Quality analysis failed with "
                    f"{type(exc).__name__}; quality findings were withheld."
                ),
                affected_column_ids=[
                    f"col_{position}"
                    for position in range(dataframe.shape[1])
                ],
                severity="medium",
            )
        )

    adapted = None
    if current_profile is not None:
        try:
            adapted = adapt_profile_and_quality_evidence(
                dataset_id=dataset_id,
                dataset_fingerprint=dataset_fingerprint,
                profile=current_profile,
                quality_findings=current_quality or [],
                columns=columns,
            )
        except _EXPECTED_STAGE_ERRORS as exc:
            warnings.append(
                UnderstandingWarning(
                    warning_code="profile_quality_evidence_stage_failed",
                    explanation=(
                        "Profile/quality adaptation failed with "
                        f"{type(exc).__name__}; combined evidence was "
                        "withheld."
                    ),
                    affected_column_ids=[
                        f"col_{position}"
                        for position in range(dataframe.shape[1])
                    ],
                    severity="high",
                )
            )
    if adapted is None and current_profile is None and current_quality is not None:
        try:
            adapted = adapt_quality_evidence(
                dataset_id=dataset_id,
                dataset_fingerprint=dataset_fingerprint,
                quality_findings=current_quality,
                columns=columns,
                rows=len(dataframe),
            )
        except _EXPECTED_STAGE_ERRORS as exc:
            warnings.append(
                UnderstandingWarning(
                    warning_code="quality_evidence_stage_failed",
                    explanation=(
                        "Quality evidence adaptation failed with "
                        f"{type(exc).__name__}; quality findings were "
                        "withheld."
                    ),
                    affected_column_ids=[
                        f"col_{position}"
                        for position in range(dataframe.shape[1])
                    ],
                    severity="medium",
                )
            )
    if adapted is None and current_profile is None:
        warnings.append(
            UnderstandingWarning(
                warning_code="profile_quality_evidence_unavailable",
                explanation=(
                    "No profile-derived evidence is available; semantic "
                    "inferences requiring profile evidence are withheld."
                ),
                affected_column_ids=[
                    f"col_{position}"
                    for position in range(dataframe.shape[1])
                ],
                severity="info",
            )
        )

    if adapted is None:
        scope = AnalysisScope(
            kind="full_dataset",
            rows_analyzed=len(dataframe),
            total_rows=len(dataframe),
        )
        evidence = []
    else:
        scope = adapted.analysis_scope
        evidence = list(adapted.evidence)
        warnings.extend(adapted.warnings)

    evidence_by_column: dict[str, list[str]] = {
        f"col_{position}": [] for position in range(len(names))
    }
    for item in evidence:
        for column_id in item.affected_column_ids:
            evidence_by_column[column_id].append(item.evidence_id)
    base_columns = [
        ColumnUnderstanding(
            column_id=f"col_{position}",
            source_name=name,
            position=position,
            relevant_evidence_ids=evidence_by_column[f"col_{position}"],
        )
        for position, name in enumerate(names)
    ]
    if current_quality is None:
        warnings.append(
            UnderstandingWarning(
                warning_code="quality_evidence_unavailable",
                explanation=(
                    "The quality stage failed, so no quality observations "
                    "are present in the evidence packet."
                ),
                severity="info",
            )
        )
    return DatasetUnderstandingResult(
        dataset_id=dataset_id,
        dataset_fingerprint=dataset_fingerprint,
        generated_at=generated_at,
        understanding_schema_version=_SCHEMA_VERSION,
        engine_version=_ENGINE_VERSION,
        profile_version=_PROFILE_VERSION,
        analysis_scope=scope,
        columns=base_columns,
        evidence=evidence,
        warnings=_unique_warnings(warnings),
    )


def _add_pairwise_evidence(
    result: DatasetUnderstandingResult,
    dataframe: pd.DataFrame,
    limits: PairwiseEvidenceLimits,
) -> DatasetUnderstandingResult:
    adapted = build_pairwise_evidence(
        dataframe,
        result,
        limits=limits,
    )
    pairwise_by_id = {
        item.evidence_id: item for item in result.pairwise_evidence
    }
    for item in adapted.pairwise_evidence:
        previous = pairwise_by_id.get(item.evidence_id)
        if previous is not None and previous != item:
            raise ValueError("Pairwise evidence ID collision.")
        pairwise_by_id[item.evidence_id] = item
    updated = result.model_dump(mode="python")
    updated["pairwise_evidence"] = list(pairwise_by_id.values())
    updated["warnings"] = _unique_warnings([
        *result.warnings,
        *adapted.warnings,
    ])
    return DatasetUnderstandingResult.model_validate(updated)


def _run_result_stage(
    result: DatasetUnderstandingResult,
    warning_code: str,
    explanation: str,
    operation: Callable[
        [DatasetUnderstandingResult],
        DatasetUnderstandingResult,
    ],
) -> DatasetUnderstandingResult:
    try:
        updated = operation(result)
        return DatasetUnderstandingResult.model_validate(
            updated.model_dump(mode="python")
        )
    except _EXPECTED_STAGE_ERRORS as exc:
        return _append_warning(
            result,
            UnderstandingWarning(
                warning_code=warning_code,
                explanation=(
                    f"{explanation} Failure type: {type(exc).__name__}."
                ),
                affected_column_ids=[
                    column.column_id for column in result.columns
                ],
                severity="medium",
            ),
        )


def _append_warning(
    result: DatasetUnderstandingResult,
    warning: UnderstandingWarning,
) -> DatasetUnderstandingResult:
    return DatasetUnderstandingResult.model_validate({
        **result.model_dump(mode="python"),
        "warnings": _unique_warnings([*result.warnings, warning]),
    })


def _unique_warnings(
    warnings: list[UnderstandingWarning],
) -> list[UnderstandingWarning]:
    unique: list[UnderstandingWarning] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    for warning in warnings:
        key = (
            warning.warning_code,
            warning.explanation,
            tuple(warning.affected_column_ids),
        )
        if key not in seen:
            unique.append(warning)
            seen.add(key)
    return unique
