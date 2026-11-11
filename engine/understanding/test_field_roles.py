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
from engine.understanding.field_roles import infer_field_roles
from engine.understanding.models import (
    ColumnUnderstanding,
    DatasetUnderstandingResult,
    Hypothesis,
)
from engine.understanding.relationships import discover_relationships
from engine.understanding.semantic_types import infer_semantic_types


def _result(dataframe: pd.DataFrame) -> DatasetUnderstandingResult:
    dataset_id = uuid.uuid4().hex
    names = [str(column) for column in dataframe.columns]
    adapted = adapt_profile_and_quality_evidence(
        dataset_id=dataset_id,
        dataset_fingerprint=None,
        profile=profile_dataframe(dataframe),
        quality_findings=detect_data_quality_issues(dataframe),
        columns=[
            ColumnIdentity(index, names[index], names[index])
            for index in range(len(names))
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
                column_id=f"col_{index}",
                source_name=names[index],
                position=index,
            )
            for index in range(len(names))
        ],
        evidence=list(adapted.evidence),
        warnings=list(adapted.warnings),
    )
    return infer_semantic_types(result)


def _role_values(result: DatasetUnderstandingResult, position: int) -> set[str]:
    return {
        hypothesis.proposed_value
        for hypothesis in result.columns[position].role_hypotheses
    }


def _role(
    result: DatasetUnderstandingResult,
    position: int,
    value: str,
) -> Hypothesis:
    return next(
        hypothesis
        for hypothesis in result.columns[position].role_hypotheses
        if hypothesis.proposed_value == value
    )


def _add_identifier_candidate(
    result: DatasetUnderstandingResult,
    position: int,
) -> DatasetUnderstandingResult:
    column = result.columns[position]
    evidence_id = next(
        item.evidence_id
        for item in result.evidence
        if item.metric_name == "column_unique_count"
        and item.affected_column_ids == [column.column_id]
    )
    columns = list(result.columns)
    columns[position] = column.model_copy(
        update={
            "semantic_hypotheses": [
                *column.semantic_hypotheses,
                Hypothesis(
                    hypothesis_id=f"test_identifier_{position}",
                    category="semantic_type",
                    proposed_value="identifier",
                    status="candidate",
                    supporting_evidence_ids=[evidence_id],
                    uncertainty_reasons=[
                        "Synthetic test candidate; identity is unconfirmed."
                    ],
                ),
            ]
        }
    )
    updated = result.model_dump(mode="python")
    updated["columns"] = columns
    return DatasetUnderstandingResult.model_validate(updated)


def test_identifier_and_measurement_roles_do_not_promote_unique_numbers():
    result = _result(pd.DataFrame({
        "unrelated_measure": [10, 20, 30, 40],
        "arbitrary_string": ["rec-1", "rec-2", "rec-3", "rec-4"],
    }))
    with_roles = infer_field_roles(result)

    assert "measurement_field" in _role_values(with_roles, 0)
    assert "identifier_candidate" not in _role_values(with_roles, 0)
    assert "primary_key_candidate" not in _role_values(with_roles, 0)
    assert "identifier_candidate" in _role_values(with_roles, 1)
    assert _role(with_roles, 1, "identifier_candidate").confidence is None


def test_primary_key_requires_identifier_signal_and_complete_uniqueness():
    complete = _add_identifier_candidate(
        _result(pd.DataFrame({"first": ["a", "b", "c"]})),
        0,
    )
    incomplete = _add_identifier_candidate(
        _result(pd.DataFrame({"second": ["a", "b", "b", None]})),
        0,
    )

    complete_role = _role(infer_field_roles(complete), 0, "primary_key_candidate")
    incomplete_role = _role(
        infer_field_roles(incomplete),
        0,
        "primary_key_candidate",
    )
    assert complete_role.status == "candidate"
    assert complete_role.confidence is None
    assert incomplete_role.status == "unsupported"
    assert incomplete_role.contradicting_evidence_ids
    assert any(
        "Missing values" in reason or "Repeated non-missing" in reason
        for reason in incomplete_role.uncertainty_reasons
    )


def test_temporal_grouping_and_ambiguous_measurement_roles_reuse_semantics():
    result = _result(pd.DataFrame({
        "time": ["2025-01-01", "2025-02-01", "2025-03-01"],
        "period_like": [20240101, 20240201, 20240301],
        "group": ["north", "south", "north"],
    }))
    with_roles = infer_field_roles(result)

    assert "temporal_field" in _role_values(with_roles, 0)
    assert "measurement_field" in _role_values(with_roles, 1)
    assert "temporal_field" in _role_values(with_roles, 1)
    measurement = _role(with_roles, 1, "measurement_field")
    temporal = _role(with_roles, 1, "temporal_field")
    assert measurement.status == temporal.status == "ambiguous"
    assert temporal.hypothesis_id in measurement.alternative_hypothesis_ids
    assert measurement.hypothesis_id in temporal.alternative_hypothesis_ids
    assert "grouping_category" in _role_values(with_roles, 2)


def test_target_and_feature_roles_remain_unresolved_without_task_context():
    result = infer_field_roles(
        _result(pd.DataFrame({
            "a": [1, 2, 3],
            "b": ["x", "y", "x"],
        }))
    )

    assert not any(
        role.proposed_value in {"potential_target", "potential_feature"}
        for column in result.columns
        for role in column.role_hypotheses
    )
    assert any(
        warning.warning_code == "analytical_role_context_unavailable"
        for warning in result.warnings
    )


def test_duplicate_headers_keep_distinct_role_positions():
    result = infer_field_roles(
        _result(pd.DataFrame(
            [[1, "east"], [2, "west"], [3, "east"]],
            columns=["shared", "shared"],
        ))
    )

    assert [column.column_id for column in result.columns] == [
        "col_0",
        "col_1",
    ]
    assert "measurement_field" in _role_values(result, 0)
    assert "grouping_category" in _role_values(result, 1)


def test_foreign_key_and_relationships_are_not_inferred_without_pairwise_data():
    result = _result(pd.DataFrame({
        "same_label": ["a", "b", "c"],
        "same_label_copy": ["a", "b", "c"],
    }))
    with_roles = infer_field_roles(result)
    discovered = discover_relationships(with_roles)

    assert discovered.relationships == []
    assert not any(
        "foreign_key" in role.proposed_value
        for column in with_roles.columns
        for role in column.role_hypotheses
    )
    warning = next(
        item for item in discovered.warnings
        if item.warning_code == "pairwise_relationship_evidence_unavailable"
    )
    assert "correlation" in warning.explanation.lower()
    assert "column names" in warning.explanation
    assert "rather than causation" in warning.explanation
    assert not any(
        "causal" in role.proposed_value
        for column in discovered.columns
        for role in column.role_hypotheses
    )


def test_relationship_absence_is_explicit_for_minimal_evidence():
    result = DatasetUnderstandingResult(
        dataset_id=uuid.uuid4().hex,
        generated_at=datetime(2026, 11, 10, 12, 0, tzinfo=UTC),
        understanding_schema_version="1.0",
        engine_version="0.1.0",
        profile_version="1",
        analysis_scope={"kind": "full_dataset", "rows_analyzed": 0, "total_rows": 0},
        columns=[
            ColumnUnderstanding(
                column_id=f"col_{position}",
                source_name=f"field_{position}",
                position=position,
            )
            for position in range(2)
        ],
    )

    discovered = discover_relationships(result)

    assert discovered.relationships == []
    assert any(
        warning.warning_code == "pairwise_relationship_evidence_unavailable"
        for warning in discovered.warnings
    )


def test_pairwise_comparison_limit_is_reported_without_pair_enumeration():
    column_count = 92
    result = DatasetUnderstandingResult(
        dataset_id=uuid.uuid4().hex,
        generated_at=datetime(2026, 11, 10, 12, 0, tzinfo=UTC),
        understanding_schema_version="1.0",
        engine_version="0.1.0",
        profile_version="1",
        analysis_scope={"kind": "full_dataset", "rows_analyzed": 0, "total_rows": 0},
        columns=[
            ColumnUnderstanding(
                column_id=f"col_{position}",
                source_name=f"field_{position}",
                position=position,
            )
            for position in range(column_count)
        ],
    )

    discovered = discover_relationships(result)

    warning = next(
        item for item in discovered.warnings
        if item.warning_code == "pairwise_comparison_limit_exceeded"
    )
    assert "4186 possible column pairs" in warning.explanation
    assert discovered.relationships == []


def test_roles_and_relationship_results_are_deterministic_and_non_mutating():
    dataframe = pd.DataFrame({
        "measure": [1, 2, 3],
        "category": ["private-a", "private-b", "private-a"],
    })
    dataframe_before = dataframe.copy(deep=True)
    result = _result(dataframe)
    evidence_before = [
        item.model_dump(mode="json") for item in result.evidence
    ]

    first = discover_relationships(infer_field_roles(result))
    second = discover_relationships(infer_field_roles(result))
    repeated = discover_relationships(
        infer_field_roles(infer_field_roles(result))
    )

    pd.testing.assert_frame_equal(dataframe, dataframe_before)
    assert [
        item.model_dump(mode="json") for item in result.evidence
    ] == evidence_before
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.model_dump(mode="json") == repeated.model_dump(mode="json")
    assert "private-a" not in first.model_dump_json()


def test_invalid_or_unresolved_evidence_is_rejected_before_role_inference():
    result = _result(pd.DataFrame({"measure": [1, 2, 3]}))
    with_roles = infer_field_roles(result)
    with pytest.raises(ValidationError):
        with_roles.evidence = []


def test_empty_and_single_column_results_remain_valid():
    empty = DatasetUnderstandingResult(
        dataset_id=uuid.uuid4().hex,
        generated_at=datetime(2026, 11, 10, 12, 0, tzinfo=UTC),
        understanding_schema_version="1.0",
        engine_version="0.1.0",
        profile_version="1",
        analysis_scope={"kind": "full_dataset", "rows_analyzed": 0, "total_rows": 0},
    )
    single = _result(pd.DataFrame({"only": [1]}))

    assert discover_relationships(infer_field_roles(empty)).relationships == []
    assert discover_relationships(infer_field_roles(single)).relationships == []
