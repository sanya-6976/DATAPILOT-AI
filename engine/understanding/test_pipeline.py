import uuid
from datetime import UTC, datetime

import pandas as pd
import pytest
from pydantic import ValidationError

from engine.understanding import (
    AnalysisScope,
    ColumnUnderstanding,
    DatasetUnderstandingResult,
    PairwiseEvidenceLimits,
    run_understanding_pipeline,
    score_labeled_outputs,
)

_DATASET_ID = uuid.UUID("32c64a00-9e4c-44ae-bf22-b59672b49806").hex
_GENERATED_AT = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)


def _frame() -> pd.DataFrame:
    rows = list(range(1, 13))
    frame = pd.DataFrame({
        "measure_unnamed": [value * 1.5 for value in rows],
        "code_unnamed": [
            str(value) if value != 5 else "five" for value in rows
        ],
        "flag_unnamed": [value % 2 == 0 for value in rows],
        "time_unnamed": pd.date_range("2020-01-01", periods=12),
        "group_unnamed": ["east", "west"] * 6,
        "constant_unnamed": ["fixed"] * 12,
        "incomplete_unnamed": [*rows[:11], None],
    })
    columns = list(frame.columns)
    values = [frame.iloc[:, index] for index in range(len(columns))]
    columns[5:5] = ["duplicate_header", "duplicate_header"]
    values[5:5] = [pd.Series(rows), pd.Series(rows)]
    frame = pd.concat(values, axis=1).set_axis(columns, axis=1)
    frame["repeated_domain"] = [*range(1, 7), *range(1, 7)]
    frame["reference_domain"] = rows
    return frame


def _run(frame: pd.DataFrame, **kwargs) -> DatasetUnderstandingResult:
    return run_understanding_pipeline(
        frame,
        dataset_id=_DATASET_ID,
        generated_at=_GENERATED_AT,
        **kwargs,
    )


def _hypothesis_labels(result: DatasetUnderstandingResult, field: str):
    return {
        (column.column_id, hypothesis.proposed_value)
        for column in result.columns
        for hypothesis in getattr(column, field)
    }


def test_public_pipeline_runs_supported_stages_on_unfamiliar_schema():
    frame = _frame()
    before = frame.copy(deep=True)

    result = _run(frame)

    pd.testing.assert_frame_equal(frame, before)
    assert len(result.columns) == frame.shape[1]
    assert [column.column_id for column in result.columns] == [
        f"col_{index}" for index in range(frame.shape[1])
    ]
    assert [column.source_name for column in result.columns].count(
        "duplicate_header"
    ) == 2
    assert result.evidence
    assert result.pairwise_evidence
    assert result.relationships
    assert any(column.semantic_hypotheses for column in result.columns)
    assert any(column.role_hypotheses for column in result.columns)
    assert any(
        column.value_interpretation_hypotheses for column in result.columns
    )
    assert all(
        hypothesis.confidence is None
        for column in result.columns
        for hypothesis in (
            column.semantic_hypotheses
            + column.role_hypotheses
            + column.unit_hypotheses
            + column.value_interpretation_hypotheses
        )
    )
    assert all(relationship.confidence is None for relationship in result.relationships)
    evidence_ids = {
        item.evidence_id for item in result.evidence + result.pairwise_evidence
    }
    assert all(
        referenced_id in evidence_ids
        for column in result.columns
        for hypothesis in (
            column.semantic_hypotheses
            + column.role_hypotheses
            + column.value_interpretation_hypotheses
        )
        for referenced_id in (
            hypothesis.supporting_evidence_ids
            + hypothesis.contradicting_evidence_ids
        )
    )
    assert all(
        referenced_id in evidence_ids
        for relationship in result.relationships
        for referenced_id in (
            relationship.supporting_evidence_ids
            + relationship.contradicting_evidence_ids
        )
    )
    assert all(
        hypothesis.method_id is not None
        and hypothesis.method_version is not None
        and hypothesis.analysis_scope is not None
        and hypothesis.sample_size is not None
        for column in result.columns
        for hypothesis in (
            column.semantic_hypotheses
            + column.role_hypotheses
            + column.value_interpretation_hypotheses
        )
    )


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame(columns=["empty_a", "empty_b"]),
        pd.DataFrame({"minimal": [1]}),
        pd.DataFrame({"ambiguous_a": [1, 2, 3], "ambiguous_b": ["x", "x", "y"]}),
    ],
)
def test_public_pipeline_handles_empty_minimal_and_ambiguous_frames(frame):
    result = _run(frame)
    assert len(result.columns) == frame.shape[1]
    assert result.analysis_scope.total_rows == len(frame)
    assert result.model_validate(result.model_dump(mode="python")) == result


def test_pipeline_is_deterministic_except_for_explicit_generation_time():
    frame = _frame()
    first = _run(frame)
    second = _run(frame)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_missing_evidence_and_unavailable_pairwise_data_are_explicit():
    base = DatasetUnderstandingResult(
        dataset_id=_DATASET_ID,
        generated_at=_GENERATED_AT,
        understanding_schema_version="1.0",
        engine_version="0.1.0",
        profile_version="1",
        analysis_scope=AnalysisScope(
            kind="full_dataset",
            rows_analyzed=0,
            total_rows=0,
        ),
        columns=[
            ColumnUnderstanding(
                column_id="col_0",
                source_name="no_evidence",
                position=0,
            ),
            ColumnUnderstanding(
                column_id="col_1",
                source_name="no_evidence",
                position=1,
            ),
        ],
    )

    result = run_understanding_pipeline(existing_result=base)

    assert not result.evidence
    assert not result.pairwise_evidence
    warning_codes = {warning.warning_code for warning in result.warnings}
    assert "profile_quality_evidence_unavailable" in warning_codes
    assert "pairwise_evidence_unavailable" in warning_codes


def test_pipeline_rejects_ambiguous_dataframe_and_existing_result_inputs():
    base = DatasetUnderstandingResult(
        dataset_id=_DATASET_ID,
        generated_at=_GENERATED_AT,
        understanding_schema_version="1.0",
        engine_version="0.1.0",
        profile_version="1",
        analysis_scope=AnalysisScope(
            kind="full_dataset",
            rows_analyzed=0,
            total_rows=0,
        ),
    )
    with pytest.raises(ValueError, match="either a DataFrame or existing_result"):
        run_understanding_pipeline(
            pd.DataFrame(),
            existing_result=base,
        )


def test_profile_failure_preserves_independent_quality_evidence(monkeypatch):
    import engine.understanding.pipeline as pipeline_module

    frame = _frame()
    def fail_profile(_frame):
        raise RuntimeError("controlled profile failure")

    monkeypatch.setattr(pipeline_module, "profile_dataframe", fail_profile)
    result = _run(frame)

    assert result.evidence
    assert result.pairwise_evidence
    assert "profile_stage_failed" in {
        warning.warning_code for warning in result.warnings
    }
    assert all(not column.semantic_hypotheses for column in result.columns)
    assert any(
        item.method_id == "detect_data_quality_issues"
        for item in result.evidence
    )


def test_quality_failure_preserves_independent_profile_evidence(monkeypatch):
    import engine.understanding.pipeline as pipeline_module

    def fail_quality(_frame):
        raise RuntimeError("controlled quality failure")

    monkeypatch.setattr(
        pipeline_module,
        "detect_data_quality_issues",
        fail_quality,
    )
    result = _run(_frame())

    assert any(
        item.method_id == "profile_dataframe"
        for item in result.evidence
    )
    assert not any(
        item.method_id == "detect_data_quality_issues"
        for item in result.evidence
    )
    assert "quality_stage_failed" in {
        warning.warning_code for warning in result.warnings
    }


def test_relationship_failure_is_explicit_and_later_stages_continue(monkeypatch):
    import engine.understanding.pipeline as pipeline_module

    def fail_relationships(result):
        raise ValueError("synthetic relationship-stage failure")

    monkeypatch.setattr(pipeline_module, "discover_relationships", fail_relationships)
    result = _run(_frame())

    assert "relationship_stage_failed" in {
        warning.warning_code for warning in result.warnings
    }
    assert any(
        column.value_interpretation_hypotheses for column in result.columns
    )


def test_resource_limits_are_visible_and_relationships_remain_bounded():
    frame = _frame()
    limits = PairwiseEvidenceLimits(
        max_column_pairs=1,
        max_sample_rows=5,
        max_distinct_values=100,
        max_category_cardinality=10,
        max_contingency_cells=100,
        max_cell_comparisons=1000,
    )

    result = _run(frame, pairwise_limits=limits)

    warning_codes = {warning.warning_code for warning in result.warnings}
    assert "pairwise_row_sampled" in warning_codes
    assert "pairwise_comparison_limit_exceeded" in warning_codes
    assert all(
        item.scope.rows_analyzed <= limits.max_sample_rows
        for item in result.pairwise_evidence
    )
    assert all(item.column_ids == ["col_0", "col_1"] for item in result.pairwise_evidence)
    assert not result.relationships


def test_invalid_existing_evidence_references_are_rejected():
    valid = DatasetUnderstandingResult(
        dataset_id=_DATASET_ID,
        generated_at=_GENERATED_AT,
        understanding_schema_version="1.0",
        engine_version="0.1.0",
        profile_version="1",
        analysis_scope=AnalysisScope(
            kind="full_dataset",
            rows_analyzed=0,
            total_rows=0,
        ),
        columns=[
            ColumnUnderstanding(
                column_id="col_0",
                source_name="invalid",
                position=0,
            )
        ],
    )
    base = valid.model_copy(update={
        "columns": [
            valid.columns[0].model_copy(update={
                "relevant_evidence_ids": ["missing_evidence"],
            })
        ],
    })
    with pytest.raises(ValidationError, match="Unknown column evidence"):
        run_understanding_pipeline(existing_result=base)


def test_synthetic_evaluation_scores_positive_negative_and_withheld_labels():
    frame = _frame()
    result = _run(frame)

    semantic = _hypothesis_labels(result, "semantic_hypotheses")
    roles = _hypothesis_labels(result, "role_hypotheses")
    interpretations = _hypothesis_labels(
        result,
        "value_interpretation_hypotheses",
    )
    temporal_semantics = _hypothesis_labels(result, "semantic_hypotheses")
    pairwise = {
        (
            tuple(item.column_ids),
            item.observation_type,
        )
        for item in result.pairwise_evidence
        if (
            item.observation_type == "exact_column_equality"
            and item.measured_value is True
        )
    }
    relationships = {
        (tuple(item.participating_column_ids), item.relationship_kind)
        for item in result.relationships
    }
    semantic_metrics = score_labeled_outputs(
        semantic,
        expected_positive={
            ("col_0", "numeric_measurement"),
            ("col_2", "boolean"),
            ("col_4", "categorical"),
            ("col_3", "datetime"),
        },
        explicitly_negative={
            ("col_2", "numeric_measurement"),
            ("col_4", "numeric_measurement"),
            ("col_3", "numeric_measurement"),
        },
        intentionally_withheld={
            ("col_0", "currency"),
            ("col_4", "ordinal"),
        },
    )
    role_metrics = score_labeled_outputs(
        roles,
        expected_positive={
            ("col_0", "measurement_field"),
            ("col_2", "grouping_category"),
        },
        explicitly_negative={
            ("col_2", "measurement_field"),
            ("col_4", "measurement_field"),
        },
        intentionally_withheld={
            ("col_0", "potential_target"),
            ("col_0", "feature_predictor"),
        },
    )
    pairwise_metrics = score_labeled_outputs(
        pairwise,
        expected_positive={
            (("col_5", "col_6"), "exact_column_equality"),
        },
        explicitly_negative={
            (("col_0", "col_2"), "exact_column_equality"),
        },
        intentionally_withheld=set(),
    )
    relationship_metrics = score_labeled_outputs(
        relationships,
        expected_positive={
            (("col_5", "col_6"), "exact_duplicate_columns"),
            (("col_9", "col_10"), "candidate_key_inclusion"),
        },
        explicitly_negative={
            (("col_0", "col_2"), "candidate_key_inclusion"),
        },
        intentionally_withheld={
            (("col_9", "col_10"), "declared_foreign_key"),
        },
    )
    interpretation_metrics = score_labeled_outputs(
        interpretations,
        expected_positive={
            ("col_2", "boolean_values"),
            ("col_1", "numeric_string_values"),
            ("col_3", "date_or_timestamp"),
        },
        explicitly_negative={
            ("col_0", "currency"),
        },
        intentionally_withheld={
            ("col_0", "percentage"),
            ("col_4", "ordinal"),
        },
    )

    assert semantic_metrics.true_positive_count == 4
    assert semantic_metrics.false_positive_count == 0
    assert semantic_metrics.false_negative_count == 0
    assert semantic_metrics.unsupported_hypothesis_count == 0
    assert semantic_metrics.precision == semantic_metrics.recall == semantic_metrics.f1 == 1
    assert ("col_3", "temporal_field") in roles
    assert ("col_3", "datetime") in temporal_semantics
    assert role_metrics.true_positive_count == 2
    assert role_metrics.false_positive_count == 0
    assert role_metrics.false_negative_count == 0
    assert role_metrics.unsupported_hypothesis_count == 0
    for metrics, expected_true_positives in (
        (pairwise_metrics, 1),
        (relationship_metrics, 2),
        (interpretation_metrics, 3),
    ):
        assert metrics.true_positive_count == expected_true_positives
        assert metrics.false_positive_count == 0
        assert metrics.false_negative_count == 0
        assert metrics.unsupported_hypothesis_count == 0
        assert metrics.precision == metrics.recall == metrics.f1 == 1


def test_evaluation_distinguishes_withheld_from_missed_hypotheses():
    metrics = score_labeled_outputs(
        predicted={"positive"},
        expected_positive={"positive", "missed"},
        explicitly_negative={"incorrect"},
        intentionally_withheld={"withheld"},
    )
    assert metrics.true_positive_count == 1
    assert metrics.false_positive_count == 0
    assert metrics.false_negative_count == 1
    assert metrics.unsupported_hypothesis_count == 0
    assert metrics.precision == 1
    assert metrics.recall == 0.5
    assert metrics.f1 == pytest.approx(2 / 3)
