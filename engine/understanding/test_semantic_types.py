import hashlib
import uuid
from datetime import UTC, datetime
from pathlib import Path

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


def _understanding_input(
    dataframe: pd.DataFrame,
    *,
    source_names: list[str] | None = None,
    dataset_id: str | None = None,
    fingerprint: str | None = None,
) -> DatasetUnderstandingResult:
    dataset_id = dataset_id or uuid.uuid4().hex
    names = [str(column) for column in dataframe.columns]
    source_names = source_names or names
    profile = profile_dataframe(dataframe)
    if fingerprint is not None and not (
        len(fingerprint) == 64
        and all(character in "0123456789abcdef" for character in fingerprint)
    ):
        raise ValueError("fingerprint must be an original-file SHA-256.")
    adapted = adapt_profile_and_quality_evidence(
        dataset_id=dataset_id,
        dataset_fingerprint=fingerprint,
        profile=profile,
        quality_findings=detect_data_quality_issues(dataframe),
        columns=[
            ColumnIdentity(index, source_names[index], names[index])
            for index in range(len(names))
        ],
    )
    return DatasetUnderstandingResult(
        dataset_id=dataset_id,
        dataset_fingerprint=fingerprint,
        generated_at=datetime(2026, 11, 10, 12, 0, tzinfo=UTC),
        understanding_schema_version="1.0",
        engine_version="0.1.0",
        profile_version="1",
        analysis_scope=adapted.analysis_scope,
        columns=[
            ColumnUnderstanding(
                column_id=f"col_{index}",
                source_name=source_names[index],
                position=index,
            )
            for index in range(len(names))
        ],
        evidence=list(adapted.evidence),
        warnings=list(adapted.warnings),
    )


def _semantic_values(result: DatasetUnderstandingResult, position: int) -> set[str]:
    return {
        hypothesis.proposed_value
        for hypothesis in result.columns[position].semantic_hypotheses
    }


def _hypothesis(result: DatasetUnderstandingResult, position: int, value: str):
    return next(
        hypothesis
        for hypothesis in result.columns[position].semantic_hypotheses
        if hypothesis.proposed_value == value
    )


def _evidence_by_id(result: DatasetUnderstandingResult):
    return {item.evidence_id: item for item in result.evidence}


def test_numeric_measurement_requires_statistics_and_general_classification():
    dataframe = pd.DataFrame({
        "first": [2, 4, 8, 16],
        "second": [3.5, 4.5, 8.5, 9.5],
    })
    result = infer_semantic_types(_understanding_input(dataframe))

    for position in range(dataframe.shape[1]):
        hypothesis = _hypothesis(result, position, "numeric_measurement")
        evidence = _evidence_by_id(result)
        assert hypothesis.confidence is None
        assert {
            evidence[evidence_id].metric_name
            for evidence_id in hypothesis.supporting_evidence_ids
        } >= {"type_candidate_observed", "profile_classification"}
        assert any(
            evidence[evidence_id].metric_name.startswith("numeric_")
            for evidence_id in hypothesis.supporting_evidence_ids
        )


def test_regular_numeric_sequence_is_not_automatically_an_identifier():
    dataframe = pd.DataFrame({"arbitrary_label": [101, 102, 103, 104]})
    result = infer_semantic_types(_understanding_input(dataframe))

    assert _semantic_values(result, 0) == {"numeric_measurement"}


def test_identifier_candidate_uses_existing_identifier_like_classification():
    dataframe = pd.DataFrame({
        "unrelated_name": ["row-10", "row-11", "row-12", "row-13"],
    })
    result = infer_semantic_types(_understanding_input(dataframe))

    hypothesis = _hypothesis(result, 0, "identifier")
    evidence = _evidence_by_id(result)
    assert any(
        evidence[evidence_id].metric_name == "profile_classification"
        and evidence[evidence_id].measured_category == "identifier_like"
        for evidence_id in hypothesis.supporting_evidence_ids
    )
    assert hypothesis.confidence is None
    assert "not a confirmed key" in hypothesis.uncertainty_reasons[0]


@pytest.mark.parametrize(
    "values",
    [
        pd.to_datetime(["2025-01-01", "2025-02-01", "2025-03-01"]),
        ["2025-01-01", "2025-02-01", "2025-03-01"],
    ],
    ids=["datetime-dtype", "parseable-date-strings"],
)
def test_date_and_datetime_candidates_reuse_existing_datetime_evidence(values):
    result = infer_semantic_types(
        _understanding_input(pd.DataFrame({"unrelated": values}))
    )

    hypothesis = _hypothesis(result, 0, "datetime")
    evidence = _evidence_by_id(result)
    assert any(
        evidence[evidence_id].metric_name == "type_candidate_observed"
        and evidence[evidence_id].measured_category == "datetime"
        for evidence_id in hypothesis.supporting_evidence_ids
    )
    assert hypothesis.confidence is None


def test_numeric_date_ambiguity_generates_alternative_hypotheses():
    dataframe = pd.DataFrame({"period_like": [20240101, 20240201, 20240301]})
    result = infer_semantic_types(_understanding_input(dataframe))

    numeric = _hypothesis(result, 0, "numeric_measurement")
    datetime = _hypothesis(result, 0, "datetime")

    assert numeric.status == datetime.status == "ambiguous"
    assert datetime.hypothesis_id in numeric.alternative_hypothesis_ids
    assert numeric.hypothesis_id in datetime.alternative_hypothesis_ids
    assert any(
        "datetime candidate" in reason
        for reason in numeric.uncertainty_reasons
    )
    assert any(
        "numeric interpretation" in reason
        for reason in datetime.uncertainty_reasons
    )
    assert numeric.confidence is datetime.confidence is None


def test_datetime_parse_failures_are_kept_as_contradictory_evidence():
    dataframe = pd.DataFrame({
        "period": ["2025-01-01", "2025-02-01", "not-a-date"],
    })
    result = infer_semantic_types(_understanding_input(dataframe))

    hypothesis = _hypothesis(result, 0, "datetime")
    evidence = _evidence_by_id(result)

    assert any(
        evidence[evidence_id].metric_name == "datetime_unparseable_count"
        and evidence[evidence_id].measured_value > 0
        for evidence_id in hypothesis.contradicting_evidence_ids
    )


def test_boolean_and_nullable_boolean_are_not_numeric_measurements():
    dataframe = pd.DataFrame({
        "first_flag": pd.Series([True, False, True], dtype=bool),
        "second_flag": pd.Series(
            [True, False, pd.NA],
            dtype="boolean",
        ),
    })
    result = infer_semantic_types(_understanding_input(dataframe))

    for position in range(dataframe.shape[1]):
        assert _semantic_values(result, position) == {"boolean"}


def test_string_categories_are_candidates_not_free_text_or_ordinal():
    dataframe = pd.DataFrame({
        "group_a": ["east", "west", "east", "north"],
        "group_b": ["low", "medium", "high", "low"],
    })
    result = infer_semantic_types(_understanding_input(dataframe))

    for position in range(dataframe.shape[1]):
        assert _semantic_values(result, position) == {"categorical"}
        hypothesis = _hypothesis(result, position, "categorical")
        assert any(
            "do not distinguish categories from free text" in reason
            for reason in hypothesis.uncertainty_reasons
        )
    assert not any(
        hypothesis.proposed_value in {"free_text", "ordinal"}
        for column in result.columns
        for hypothesis in column.semantic_hypotheses
    )


def test_empty_constant_near_constant_and_mixed_columns_stay_unclassified():
    dataframe = pd.DataFrame({
        "blank": [None] * 20,
        "fixed": [5] * 20,
        "mostly_fixed": ["common"] * 19 + ["rare"],
        "mixed": [1, "other"] * 10,
    })
    result = infer_semantic_types(_understanding_input(dataframe))

    for position in range(dataframe.shape[1]):
        assert result.columns[position].semantic_hypotheses == []
        assert any(
            warning.warning_code == "insufficient_semantic_evidence"
            and warning.affected_column_ids == [f"col_{position}"]
            for warning in result.warnings
        )


def test_small_numeric_dataset_keeps_candidate_and_exposes_uncertainty():
    dataframe = pd.DataFrame({"arbitrary": [1, 2]})
    result = infer_semantic_types(_understanding_input(dataframe))

    hypothesis = _hypothesis(result, 0, "numeric_measurement")
    assert hypothesis.status == "candidate"
    assert any(
        "Fewer than three" in reason
        for reason in hypothesis.uncertainty_reasons
    )


def test_duplicate_names_use_distinct_positional_hypotheses():
    dataframe = pd.DataFrame(
        [[1, "north"], [2, "south"], [3, "north"]],
        columns=["measure", "measure"],
    )
    result = infer_semantic_types(_understanding_input(dataframe))

    assert [column.column_id for column in result.columns] == [
        "col_0",
        "col_1",
    ]
    assert [column.source_name for column in result.columns] == [
        "measure",
        "measure",
    ]
    assert _semantic_values(result, 0) == {"numeric_measurement"}
    assert _semantic_values(result, 1) == {"categorical"}


def test_evidence_and_dataframe_are_not_modified_and_output_is_idempotent():
    dataframe = pd.DataFrame({
        "number": [1, 2, 4, 8],
        "category": ["private-a", "private-b", "private-a", "private-c"],
    })
    dataframe_before = dataframe.copy(deep=True)
    input_result = _understanding_input(dataframe)
    evidence_before = [
        item.model_dump(mode="json") for item in input_result.evidence
    ]

    first = infer_semantic_types(input_result)
    second = infer_semantic_types(input_result)
    repeated = infer_semantic_types(first)

    pd.testing.assert_frame_equal(dataframe, dataframe_before)
    assert [
        item.model_dump(mode="json") for item in input_result.evidence
    ] == evidence_before
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.model_dump(mode="json") == repeated.model_dump(mode="json")
    assert "private-a" not in first.model_dump_json()


def test_inference_requires_evidence_references_to_resolve():
    dataframe = pd.DataFrame({"number": [1, 2, 3]})
    input_result = _understanding_input(dataframe)
    with pytest.raises(ValidationError):
        input_result.evidence = []


def test_no_candidate_for_complex_values_or_arbitrary_coordinate_pairs():
    dataframe = pd.DataFrame({
        "first": [1 + 2j, 2 + 3j, 3 + 4j],
        "second": [10, 20, 30],
        "third": [40, 50, 60],
    })
    result = infer_semantic_types(_understanding_input(dataframe))

    assert result.columns[0].semantic_hypotheses == []
    assert _semantic_values(result, 1) == {"numeric_measurement"}
    assert _semantic_values(result, 2) == {"numeric_measurement"}
    assert not any(
        hypothesis.proposed_value == "geographic"
        for column in result.columns
        for hypothesis in column.semantic_hypotheses
    )


def test_real_world_csv_profiles_produce_only_resolvable_hypotheses():
    paths = sorted(
        (
            Path(__file__).resolve().parents[2]
            / "sample_data"
            / "real_world"
        ).glob("*.csv")
    )
    assert len(paths) >= 2

    for path in paths:
        dataframe = pd.read_csv(path)
        result = infer_semantic_types(
            _understanding_input(
                dataframe,
                dataset_id=uuid.uuid5(uuid.NAMESPACE_URL, path.name).hex,
                fingerprint=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
        evidence_ids = {item.evidence_id for item in result.evidence}
        column_ids = {column.column_id for column in result.columns}
        for column in result.columns:
            for hypothesis in column.semantic_hypotheses:
                assert set(hypothesis.supporting_evidence_ids) <= evidence_ids
                assert set(hypothesis.contradicting_evidence_ids) <= evidence_ids
                assert set(hypothesis.alternative_hypothesis_ids) <= {
                    item.hypothesis_id
                    for item in column.semantic_hypotheses
                }
        assert len(column_ids) == dataframe.shape[1]


def _hypothesis_with_evidence(result: DatasetUnderstandingResult):
    evidence_id = next(
        item.evidence_id
        for item in result.evidence
        if item.affected_column_ids == ["col_0"]
    )
    return Hypothesis(
        hypothesis_id="preexisting-semantic",
        category="semantic_type",
        proposed_value="numeric_measurement",
        status="candidate",
        supporting_evidence_ids=[evidence_id],
    )
