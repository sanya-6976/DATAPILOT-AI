import copy
import hashlib
import json
import uuid
from pathlib import Path

import pandas as pd
import pytest

from engine.profiling.profiler import profile_dataframe
from engine.quality.data_quality import detect_data_quality_issues
from engine.understanding.evidence import (
    ColumnIdentity,
    adapt_profile_and_quality_evidence,
)

DATASET_ID = "12345678123456781234567812345678"


def _column_identities(
    dataframe: pd.DataFrame,
    *,
    source_names: list[str] | None = None,
) -> list[ColumnIdentity]:
    names = [str(column) for column in dataframe.columns]
    original_names = source_names or names
    return [
        ColumnIdentity(
            position=index,
            source_name=original_names[index],
            profile_name=name,
        )
        for index, name in enumerate(names)
    ]


def _adapt(
    dataframe: pd.DataFrame,
    *,
    dataset_id: str = DATASET_ID,
    dataset_fingerprint: str | None = "a" * 64,
    source_names: list[str] | None = None,
    profile: dict | None = None,
    quality_findings: list[dict] | None = None,
):
    profile_result = profile if profile is not None else profile_dataframe(
        dataframe
    )
    findings = (
        quality_findings
        if quality_findings is not None
        else detect_data_quality_issues(dataframe)
    )
    return adapt_profile_and_quality_evidence(
        dataset_id=dataset_id,
        dataset_fingerprint=dataset_fingerprint,
        profile=profile_result,
        quality_findings=findings,
        columns=_column_identities(dataframe, source_names=source_names),
    )


def _by_metric(result, metric_name: str):
    return [
        item for item in result.evidence
        if item.metric_name == metric_name
    ]


def test_profile_mapping_emits_dataset_and_positional_column_evidence():
    dataframe = pd.DataFrame({
        "first_measure": [1.0, 3.0, None],
        "other_group": ["red", "blue", "red"],
    })

    result = _adapt(dataframe)

    assert _by_metric(result, "dataset_row_count")[0].measured_value == 3
    assert _by_metric(result, "dataset_column_count")[0].measured_value == 2
    assert {
        evidence.affected_column_ids[0]
        for evidence in result.evidence
        if evidence.affected_column_ids
    } == {"col_0", "col_1"}
    assert _by_metric(result, "column_missing_count")[0].affected_column_ids == [
        "col_0"
    ]
    missing_ratio = _by_metric(result, "column_missing_percentage")[0]
    assert missing_ratio.measured_value == 33.33
    assert missing_ratio.denominator == len(dataframe)
    assert _by_metric(result, "numeric_mean")[0].measured_value == 2.0
    assert all(
        item.method_version == "1"
        for item in result.evidence
    )
    assert result.analysis_scope.rows_analyzed == len(dataframe)


def test_quality_findings_are_preserved_as_observations_not_hypotheses():
    dataframe = pd.DataFrame({
        "measure": [1, 2, 3, 1, None],
        "group": ["same", "same", "same", "same", "same"],
    })

    result = _adapt(dataframe)
    quality_evidence = [
        item for item in result.evidence
        if item.method_id == "detect_data_quality_issues"
    ]

    assert quality_evidence
    assert not any(
        item.metric_name in {"semantic_type", "field_role", "unit"}
        for item in result.evidence
    )
    assert any(
        item.metric_name == "quality_missing_values_count"
        and item.measured_value == 1
        and item.affected_column_ids == ["col_0"]
        and item.denominator == len(dataframe)
        for item in quality_evidence
    )
    assert any(
        item.metric_name == "quality_constant_column_count"
        and item.affected_column_ids == ["col_1"]
        for item in quality_evidence
    )
    assert any(
        item.metric_name == "quality_duplicate_rows_count"
        and item.affected_column_ids == []
        for item in quality_evidence
    )
    assert _by_metric(result, "dataset_duplicate_row_count")


def test_duplicate_source_names_keep_distinct_positions_and_warn_if_ambiguous():
    dataframe = pd.DataFrame(
        [[1, 10], [2, 20], [3, 30]],
        columns=["measure", "measure"],
    )
    profile = profile_dataframe(dataframe)
    findings = detect_data_quality_issues(dataframe)

    result = _adapt(
        dataframe,
        profile=profile,
        quality_findings=[
            *findings,
            {
                "issue": "mixed_types",
                "column": "measure",
                "severity": "low",
                "description": "ignored",
            },
        ],
    )

    assert {
        item.affected_column_ids[0]
        for item in result.evidence
        if item.metric_name == "column_unique_count"
    } == {"col_0", "col_1"}
    assert any(
        warning.warning_code == "unmappable_quality_finding"
        for warning in result.warnings
    )


def test_quality_findings_with_duplicate_headers_map_by_position():
    dataframe = pd.DataFrame(
        [[1, None], [None, 20], [3, 30]],
        columns=["measure", "measure"],
    )
    result = _adapt(dataframe)
    missing_by_column = {
        item.affected_column_ids[0]: item.measured_value
        for item in result.evidence
        if item.metric_name == "quality_missing_values_count"
    }

    assert missing_by_column == {"col_0": 1, "col_1": 1}
    assert not any(
        warning.warning_code == "unmappable_quality_finding"
        for warning in result.warnings
    )


def test_duplicate_header_quality_finding_maps_all_matching_positions():
    dataframe = pd.DataFrame(
        [[1, 10], [2, 20]],
        columns=["duplicate", "duplicate"],
    )
    profile = profile_dataframe(dataframe)
    findings = [
        {
            "issue": "duplicate_column_names",
            "column": "duplicate",
            "count": 1,
            "severity": "low",
            "description": "ignored",
        }
    ]

    result = _adapt(
        dataframe,
        profile=profile,
        quality_findings=findings,
    )

    duplicate_evidence = _by_metric(
        result,
        "quality_duplicate_column_names_count",
    )
    assert len(duplicate_evidence) == 1
    assert duplicate_evidence[0].affected_column_ids == ["col_0", "col_1"]


def test_duplicate_header_finding_maps_source_names_after_profile_renaming():
    dataframe = pd.DataFrame(
        [[1, 10], [2, 20]],
        columns=["measure", "measure"],
    )
    renamed = dataframe.copy(deep=True)
    renamed.columns = ["measure", "measure__3"]
    profile = profile_dataframe(renamed)
    for position, item in enumerate(profile["columns"]):
        item["source_column_name"] = "measure"
        item["column_index"] = position
    columns = [
        ColumnIdentity(position, "measure", name)
        for position, name in enumerate(renamed.columns)
    ]

    result = adapt_profile_and_quality_evidence(
        dataset_id=DATASET_ID,
        dataset_fingerprint="a" * 64,
        profile=profile,
        quality_findings=[
            {
                "issue": "duplicate_column_names",
                "column": "measure",
                "count": 1,
                "severity": "low",
                "description": "ignored",
            }
        ],
        columns=columns,
    )

    duplicate_evidence = _by_metric(
        result,
        "quality_duplicate_column_names_count",
    )
    assert duplicate_evidence[0].affected_column_ids == ["col_0", "col_1"]


@pytest.mark.parametrize(
    "dataframe",
    [
        pd.DataFrame(columns=["empty_header"]),
        pd.DataFrame(),
    ],
    ids=["header-only", "no-columns"],
)
def test_empty_and_header_only_datasets_have_valid_scopes(dataframe):
    result = _adapt(dataframe)

    assert result.analysis_scope.kind == "full_dataset"
    assert result.analysis_scope.rows_analyzed == 0
    assert _by_metric(result, "dataset_row_count")[0].measured_value == 0


def test_boolean_numeric_ambiguous_identifier_and_complex_evidence():
    dataframe = pd.DataFrame({
        "flag": pd.Series([True, False, pd.NA], dtype="boolean"),
        "reading": pd.Series([1, 3, pd.NA], dtype="Int64"),
        "period": [20240101, 20240201, 20240301],
        "serial": ["rec-1", "rec-2", "rec-3"],
        "complex_value": [1 + 2j, 2 + 3j, 3 + 4j],
    })

    result = _adapt(dataframe)

    assert any(
        item.metric_name == "type_candidate_observed"
        and item.measured_category == "boolean"
        and item.affected_column_ids == ["col_0"]
        for item in result.evidence
    )
    assert any(
        item.metric_name == "numeric_mean"
        and item.affected_column_ids == ["col_1"]
        for item in result.evidence
    )
    assert any(
        item.metric_name == "type_ambiguity_observed"
        and item.measured_value is True
        and item.affected_column_ids == ["col_2"]
        for item in result.evidence
    )
    assert any(
        item.metric_name == "profile_classification"
        and item.measured_category == "identifier_like"
        and item.affected_column_ids == ["col_3"]
        for item in result.evidence
    )
    assert any(
        warning.warning_code == "unsupported_complex_statistics"
        and warning.affected_column_ids == ["col_4"]
        for warning in result.warnings
    )


def test_unmappable_and_unsupported_quality_findings_emit_warnings():
    dataframe = pd.DataFrame({"value": [1, 2, 3]})
    findings = [
        {
            "issue": "missing_values",
            "column": "not-a-column",
            "count": 1,
        },
        {
            "issue": "future_quality_finding",
            "description": "raw description should not pass through",
        },
    ]

    result = _adapt(dataframe, quality_findings=findings)

    assert [
        warning.warning_code for warning in result.warnings
    ] == [
        "unmappable_quality_finding",
        "unsupported_quality_finding",
    ]
    assert "raw description" not in json.dumps(
        [warning.model_dump() for warning in result.warnings]
    )


def test_invalid_column_positions_are_rejected():
    dataframe = pd.DataFrame({"value": [1]})
    columns = [ColumnIdentity(1, "value", "value")]

    with pytest.raises(ValueError, match="contiguous positions"):
        adapt_profile_and_quality_evidence(
            dataset_id=DATASET_ID,
            dataset_fingerprint=None,
            profile=profile_dataframe(dataframe),
            quality_findings=[],
            columns=columns,
        )

    with pytest.raises(ValueError, match="invalid column position"):
        _adapt(
            dataframe,
            quality_findings=[
                {
                    "issue": "missing_values",
                    "column": "value",
                    "column_index": 4,
                    "count": 1,
                }
            ],
        )

    profile = profile_dataframe(dataframe)
    profile["columns"][0]["column_index"] = 2
    with pytest.raises(ValueError, match="Profile column index"):
        _adapt(dataframe, profile=profile, quality_findings=[])


def test_evidence_ids_are_unique_and_output_is_deterministic():
    dataframe = pd.DataFrame({
        "number": [1, 2, 3, None],
        "label": ["x", "y", "x", "z"],
    })

    first = _adapt(dataframe)
    second = _adapt(dataframe)

    first_dump = [item.model_dump(mode="json") for item in first.evidence]
    second_dump = [item.model_dump(mode="json") for item in second.evidence]
    ids = [item.evidence_id for item in first.evidence]

    assert first_dump == second_dump
    assert len(ids) == len(set(ids))


def test_adapter_does_not_mutate_dataframe_profile_or_quality_findings():
    dataframe = pd.DataFrame({
        "measure": [1, None, 3],
        "label": ["secret-category", "other-secret", "secret-category"],
    })
    dataframe_before = dataframe.copy(deep=True)
    profile = profile_dataframe(dataframe)
    profile_before = copy.deepcopy(profile)
    findings = detect_data_quality_issues(dataframe)
    findings_before = copy.deepcopy(findings)

    _adapt(
        dataframe,
        profile=profile,
        quality_findings=findings,
    )

    pd.testing.assert_frame_equal(dataframe, dataframe_before)
    assert profile == profile_before
    assert findings == findings_before


def test_evidence_never_contains_raw_category_values_and_summary_is_bounded():
    dataframe = pd.DataFrame({
        "labels": [f"private-label-{index}" for index in range(10)],
    })
    profile = profile_dataframe(dataframe)
    profile["columns"][0]["top_values"] = [
        {"value": f"private-label-{index}", "count": 10 - index}
        for index in range(12)
    ]

    result = _adapt(dataframe, profile=profile, quality_findings=[])
    serialized = json.dumps(
        [item.model_dump(mode="json") for item in result.evidence]
    )

    assert len(_by_metric(result, "categorical_top_value_count")) == 5
    assert "private-label" not in serialized
    assert any(
        warning.warning_code == "categorical_summary_truncated"
        for warning in result.warnings
    )


def test_unrecognized_profile_fields_are_reported_without_echoing_values():
    dataframe = pd.DataFrame({"value": [1, 2, 3]})
    profile = profile_dataframe(dataframe)
    profile["columns"][0]["future_private_stat"] = "sensitive"
    profile["columns"][0]["statistics"] = "unsupported"

    result = _adapt(dataframe, profile=profile, quality_findings=[])

    warning_codes = {warning.warning_code for warning in result.warnings}
    warning_text = json.dumps(
        [warning.model_dump() for warning in result.warnings]
    )
    assert "unsupported_profile_field" in warning_codes
    assert "unsupported_profile_statistics" in warning_codes
    assert "sensitive" not in warning_text


def test_unrecognized_quality_fields_are_warned_without_echoing_values():
    dataframe = pd.DataFrame({"value": [1, 2, 3]})
    result = _adapt(
        dataframe,
        quality_findings=[
            {
                "issue": "duplicate_rows",
                "count": 1,
                "private_payload": "do-not-copy",
            }
        ],
    )

    assert any(
        warning.warning_code == "unsupported_quality_finding_field"
        for warning in result.warnings
    )
    assert "do-not-copy" not in json.dumps(
        [warning.model_dump() for warning in result.warnings]
    )


@pytest.mark.parametrize(
    ("dataset_id", "fingerprint"),
    [
        ("not-a-uuid", "a" * 64),
        (DATASET_ID, "A" * 64),
    ],
)
def test_invalid_dataset_identity_is_rejected(dataset_id, fingerprint):
    with pytest.raises(ValueError):
        _adapt(
            pd.DataFrame({"value": [1]}),
            dataset_id=dataset_id,
            dataset_fingerprint=fingerprint,
        )


def test_real_world_csv_profiles_are_adapted_without_schema_assumptions():
    fixture_paths = sorted(
        (
            Path(__file__).resolve().parents[2]
            / "sample_data"
            / "real_world"
        ).glob("*.csv")
    )
    assert len(fixture_paths) >= 2

    for path in fixture_paths:
        dataframe = pd.read_csv(path)
        fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
        dataset_id = uuid.uuid5(uuid.NAMESPACE_URL, path.name).hex

        result = _adapt(
            dataframe,
            dataset_id=dataset_id,
            dataset_fingerprint=fingerprint,
        )

        assert _by_metric(result, "dataset_row_count")[0].measured_value == (
            len(dataframe)
        )
        assert {
            column_id
            for item in result.evidence
            for column_id in item.affected_column_ids
        } <= {f"col_{index}" for index in range(dataframe.shape[1])}
        assert len(
            {item.evidence_id for item in result.evidence}
        ) == len(result.evidence)
