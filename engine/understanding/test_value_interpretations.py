import uuid
from datetime import UTC, datetime

import pandas as pd
import pytest
from pydantic import ValidationError

from engine.profiling.profiler import profile_dataframe
from engine.quality.data_quality import detect_data_quality_issues
from engine.understanding.evidence import (
    ColumnIdentity,
    adapt_profile_and_quality_evidence,
)
from engine.understanding.models import (
    ColumnUnderstanding,
    DatasetUnderstandingResult,
    Hypothesis,
)
from engine.understanding.semantic_types import infer_semantic_types
from engine.understanding.value_interpretations import (
    infer_value_interpretations,
)


def _result(dataframe: pd.DataFrame) -> DatasetUnderstandingResult:
    dataset_id = uuid.uuid5(
        uuid.NAMESPACE_URL,
        "task-3.5-synthetic-dataset",
    ).hex
    names = [str(column) for column in dataframe.columns]
    adapted = adapt_profile_and_quality_evidence(
        dataset_id=dataset_id,
        dataset_fingerprint=None,
        profile=profile_dataframe(dataframe),
        quality_findings=detect_data_quality_issues(dataframe),
        columns=[
            ColumnIdentity(position, name, name)
            for position, name in enumerate(names)
        ],
    )
    result = DatasetUnderstandingResult(
        dataset_id=dataset_id,
        generated_at=datetime(2026, 11, 10, 12, 0, tzinfo=UTC),
        understanding_schema_version="1.0",
        engine_version="0.1.0",
        profile_version="1",
        analysis_scope=adapted.analysis_scope,
        columns=[
            ColumnUnderstanding(
                column_id=f"col_{position}",
                source_name=name,
                position=position,
            )
            for position, name in enumerate(names)
        ],
        evidence=list(adapted.evidence),
        warnings=list(adapted.warnings),
    )
    return infer_semantic_types(result)


def _interpretations(
    result: DatasetUnderstandingResult,
    position: int,
) -> set[str]:
    return {
        hypothesis.proposed_value
        for hypothesis in (
            result.columns[position].value_interpretation_hypotheses
        )
    }


def _warnings(
    result: DatasetUnderstandingResult,
    position: int,
) -> set[str]:
    column_id = result.columns[position].column_id
    return {
        warning.warning_code
        for warning in result.warnings
        if column_id in warning.affected_column_ids
    }


def test_nonnegative_integer_candidate_is_ambiguous_not_confirmed_count():
    dataframe = pd.DataFrame({
        "first": [0, 1, 2, 3, 5],
        "second": [0.25, 0.5, 0.75, 1.0, 1.25],
    })
    result = infer_value_interpretations(
        _result(dataframe)
    )

    integer_candidate = next(
        hypothesis
        for hypothesis in result.columns[0].value_interpretation_hypotheses
        if hypothesis.proposed_value == "nonnegative_integer_count"
    )
    assert integer_candidate.status == "ambiguous"
    assert integer_candidate.confidence is None
    assert integer_candidate.method_id == "evidence_value_interpretation"
    assert integer_candidate.method_version == "1"
    assert integer_candidate.analysis_scope == result.analysis_scope
    assert integer_candidate.sample_size == len(dataframe)
    assert "count" not in _interpretations(result, 1)
    assert "numeric_scale_not_interpreted" in _warnings(result, 1)


def test_proportion_percentage_and_bounded_score_are_not_guessed_from_ranges():
    dataframe = pd.DataFrame({
        "first": [0.1, 0.25, 0.75, 1.0],
        "second": [0, 10, 75, 100],
        "third": [-5, 0, 5, 10],
    })
    result = infer_value_interpretations(_result(dataframe))

    assert not any(
        hypothesis.proposed_value in {
            "proportion",
            "percentage",
            "bounded_score",
            "rating",
        }
        for column in result.columns
        for hypothesis in column.value_interpretation_hypotheses
    )
    assert "proportion_percentage_ambiguous" in _warnings(result, 0)
    assert "bounded_numeric_scale_ambiguous" in _warnings(result, 1)
    assert "measurement_unit_unavailable" in _warnings(result, 0)
    assert "measurement_unit_unavailable" in _warnings(result, 2)


def test_currency_physical_units_duration_and_temporal_intervals_unsupported():
    dataframe = pd.DataFrame({
        "amount_like": [10.5, 20.25, 30.75],
        "temporal": ["2025-01-01", "2025-02-01", "2025-03-01"],
    })
    result = infer_value_interpretations(_result(dataframe))

    assert not any(
        hypothesis.proposed_value in {
            "currency",
            "length",
            "weight",
            "temperature",
            "duration",
            "temporal_interval",
        }
        for column in result.columns
        for hypothesis in column.unit_hypotheses
        + column.value_interpretation_hypotheses
    )
    assert "measurement_unit_unavailable" in _warnings(result, 0)
    temporal = next(
        hypothesis
        for hypothesis in result.columns[1].value_interpretation_hypotheses
        if hypothesis.proposed_value == "date_or_timestamp"
    )
    assert temporal.status == "ambiguous"
    assert "temporal_granularity_unresolved" in _warnings(result, 1)


def test_boolean_and_numeric_string_formats_are_hypotheses_not_transforms():
    dataframe = pd.DataFrame({
        "flag": pd.Series([True, False, pd.NA], dtype="boolean"),
        "numeric_text": ["1.25", "2.5", "not-a-number"],
    })
    before = dataframe.copy(deep=True)
    result = infer_value_interpretations(_result(dataframe))

    boolean = next(
        hypothesis
        for hypothesis in result.columns[0].value_interpretation_hypotheses
        if hypothesis.proposed_value == "boolean_values"
    )
    assert boolean.status == "candidate"
    assert boolean.confidence is None
    numeric_string = next(
        hypothesis
        for hypothesis in result.columns[1].value_interpretation_hypotheses
        if hypothesis.proposed_value == "numeric_string_values"
    )
    assert numeric_string.status == "ambiguous"
    assert numeric_string.contradicting_evidence_ids
    assert "not-a-number" not in result.model_dump_json()
    pd.testing.assert_frame_equal(dataframe, before)


def test_numeric_boolean_codes_and_ordinal_categories_remain_uninterpreted():
    dataframe = pd.DataFrame({
        "binary_code": [0, 1, 0, 1],
        "ordered_looking": ["low", "medium", "high", "low"],
    })
    result = infer_value_interpretations(_result(dataframe))

    assert "boolean_values" not in _interpretations(result, 0)
    assert not any(
        hypothesis.proposed_value in {"ordinal", "rating", "bounded_score"}
        for column in result.columns
        for hypothesis in column.value_interpretation_hypotheses
    )
    assert "ordinal_interpretation_unavailable" in _warnings(result, 1)


def test_mixed_formats_and_missing_sentinel_unknowns_are_preserved():
    dataframe = pd.DataFrame({
        "mixed": pd.Series([1, "two", 3, None], dtype=object),
        "nullable": [1.0, None, 2.0, 3.0],
    })
    result = infer_value_interpretations(_result(dataframe))

    mixed = result.columns[0].value_interpretation_hypotheses
    assert any(item.proposed_value == "mixed_scalar_values" for item in mixed)
    assert any(
        "may be intentional" in reason
        for hypothesis in mixed
        for reason in hypothesis.uncertainty_reasons
    )
    assert "missing_sentinel_unavailable" in _warnings(result, 1)
    assert not any(
        "sentinel" in hypothesis.proposed_value
        for column in result.columns
        for hypothesis in column.value_interpretation_hypotheses
    )


def test_duplicate_headers_keep_independent_positional_interpretations():
    dataframe = pd.DataFrame(
        [[1, "2"], [2, "3"], [3, "4"]],
        columns=["shared", "shared"],
    )
    result = infer_value_interpretations(_result(dataframe))

    assert [column.column_id for column in result.columns] == [
        "col_0",
        "col_1",
    ]
    assert any(
        hypothesis.proposed_value == "nonnegative_integer_count"
        for hypothesis in result.columns[0].value_interpretation_hypotheses
    )
    assert any(
        hypothesis.proposed_value == "numeric_string_values"
        for hypothesis in result.columns[1].value_interpretation_hypotheses
    )


def test_empty_constant_and_insufficient_evidence_do_not_invent_units():
    dataframe = pd.DataFrame({
        "empty": [None, None, None],
        "constant": [5, 5, 5],
        "small": [1, None, None],
    })
    result = infer_value_interpretations(_result(dataframe))

    assert all(
        not column.value_interpretation_hypotheses
        for column in result.columns
    )
    assert not any(
        hypothesis.category == "unit"
        for column in result.columns
        for hypothesis in column.unit_hypotheses
    )


def test_repeated_runs_are_deterministic_and_do_not_mutate_input_result():
    result = _result(pd.DataFrame({
        "number": [1, 2, 3],
        "date": ["2025-01-01", "2025-01-02", "2025-01-03"],
    }))
    before = result.model_dump(mode="json")

    first = infer_value_interpretations(result)
    second = infer_value_interpretations(result)
    repeated = infer_value_interpretations(first)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.model_dump(mode="json") == repeated.model_dump(mode="json")
    assert result.model_dump(mode="json") == before


def test_existing_semantic_and_unit_hypotheses_are_preserved_unchanged():
    base = _result(pd.DataFrame({"reading": [1, 2, 4, 8]}))
    evidence_id = next(
        item.evidence_id
        for item in base.evidence
        if item.affected_column_ids == ["col_0"]
    )
    existing_semantic = Hypothesis(
        hypothesis_id="existing-semantic",
        category="semantic_type",
        proposed_value="numeric_measurement",
        status="candidate",
        supporting_evidence_ids=[evidence_id],
    )
    existing_unit = Hypothesis(
        hypothesis_id="existing-unit",
        category="unit",
        proposed_value="unit_unspecified",
        status="ambiguous",
        supporting_evidence_ids=[evidence_id],
    )
    column = base.columns[0].model_copy(update={
        "semantic_hypotheses": [existing_semantic],
        "unit_hypotheses": [existing_unit],
    })
    data = base.model_dump(mode="python")
    data["columns"] = [column]
    with_existing = DatasetUnderstandingResult.model_validate(data)

    interpreted = infer_value_interpretations(with_existing)

    assert interpreted.columns[0].semantic_hypotheses == [existing_semantic]
    assert interpreted.columns[0].unit_hypotheses == [existing_unit]
    assert interpreted.columns[0].value_interpretation_hypotheses


def test_invalid_or_unresolved_evidence_references_are_rejected():
    result = infer_value_interpretations(
        _result(pd.DataFrame({"number": [1, 2, 3]}))
    )
    with pytest.raises(ValidationError):
        result.evidence = []
