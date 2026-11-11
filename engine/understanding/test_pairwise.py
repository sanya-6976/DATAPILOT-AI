import uuid
from datetime import UTC, datetime

import pandas as pd
import pytest
from pydantic import ValidationError

from engine.understanding.models import (
    AnalysisScope,
    ColumnUnderstanding,
    DatasetUnderstandingResult,
    EvidenceItem,
    Hypothesis,
    PairwiseEvidence,
    RelationshipHypothesis,
)
from engine.understanding.pairwise import (
    PairwiseEvidenceLimits,
    build_pairwise_evidence,
)
from engine.understanding.relationships import (
    discover_relationships,
    discover_relationships_from_dataframe,
)


def _result(dataframe: pd.DataFrame) -> DatasetUnderstandingResult:
    dataset_id = uuid.UUID("12345678-1234-5678-1234-567812345678").hex
    columns = [
        ColumnUnderstanding(
            column_id=f"col_{position}",
            source_name=str(name),
            position=position,
        )
        for position, name in enumerate(dataframe.columns)
    ]
    return DatasetUnderstandingResult(
        dataset_id=dataset_id,
        dataset_fingerprint=None,
        generated_at=datetime(2026, 11, 10, 12, 0, tzinfo=UTC),
        understanding_schema_version="1.0",
        engine_version="0.1.0",
        profile_version="1",
        analysis_scope=AnalysisScope(
            kind="full_dataset",
            rows_analyzed=len(dataframe),
            total_rows=len(dataframe),
        ),
        columns=columns,
    )


def _build(
    dataframe: pd.DataFrame,
    *,
    limits: PairwiseEvidenceLimits | None = None,
) -> tuple[DatasetUnderstandingResult, list[PairwiseEvidence]]:
    result = _result(dataframe)
    adapted = build_pairwise_evidence(
        dataframe,
        result,
        limits=limits or PairwiseEvidenceLimits(),
    )
    updated = result.model_dump(mode="python")
    updated["pairwise_evidence"] = list(adapted.pairwise_evidence)
    updated["warnings"] = list(adapted.warnings)
    return DatasetUnderstandingResult.model_validate(updated), list(
        adapted.pairwise_evidence
    )


def _observation(
    result: DatasetUnderstandingResult,
    observation_type: str,
    column_ids: tuple[str, str] = ("col_0", "col_1"),
) -> PairwiseEvidence:
    return next(
        item
        for item in result.pairwise_evidence
        if item.observation_type == observation_type
        and tuple(item.column_ids) == column_ids
    )


def test_pairwise_ids_outputs_and_input_are_deterministic_and_non_mutating():
    dataframe = pd.DataFrame(
        [["a", 1], ["b", 2], ["a", 3], ["c", 4]],
        columns=["shared", "shared"],
    )
    before = dataframe.copy(deep=True)

    first, first_evidence = _build(dataframe)
    _, second_evidence = _build(dataframe)

    pd.testing.assert_frame_equal(dataframe, before)
    assert [item.model_dump(mode="json") for item in first_evidence] == [
        item.model_dump(mode="json") for item in second_evidence
    ]
    assert all(
        item.method_id == "pairwise_structural_comparison"
        and item.method_version == "1"
        for item in first_evidence
    )
    assert all(
        item.column_ids == ["col_0", "col_1"]
        for item in first_evidence
    )
    assert [column.source_name for column in first.columns] == [
        "shared",
        "shared",
    ]
    assert len({item.evidence_id for item in first_evidence}) == 3


def test_exact_equality_is_separate_from_partial_value_overlap_and_missing():
    dataframe = pd.DataFrame({
        "left": ["a", "b", "a", None],
        "right": ["a", "c", "a", None],
    })
    result, _ = _build(dataframe)

    equality = _observation(result, "exact_column_equality")
    overlap = _observation(result, "value_overlap")
    assert equality.measured_value is False
    assert equality.sample_size == equality.denominator == len(dataframe)
    assert overlap.sample_size == 3
    assert overlap.denominator == 4
    assert overlap.measured_summary == {
        "complete_case_rows": 3,
        "left_non_null_count": 3,
        "left_unique_count": 2,
        "left_values_in_right": False,
        "right_non_null_count": 3,
        "right_unique_count": 2,
        "right_values_in_left": False,
        "shared_unique_count": 1,
    }
    assert overlap.limitations


def test_exact_duplicate_columns_create_structural_relationship_not_correlation():
    dataframe = pd.DataFrame({
        "unrelated": ["a", "b", "a", "c"],
        "duplicate": ["a", "b", "a", "c"],
        "number": [1.0, 2.0, 3.0, 4.0],
    })
    result, _ = _build(dataframe)

    discovered = discover_relationships(result)

    exact = [
        relation
        for relation in discovered.relationships
        if relation.relationship_kind == "exact_duplicate_columns"
    ]
    assert len(exact) == 1
    assert exact[0].participating_column_ids == ["col_0", "col_1"]
    assert exact[0].confidence is None
    assert exact[0].supporting_evidence_ids == [
        _observation(
            result,
            "exact_column_equality",
            ("col_0", "col_1"),
        ).evidence_id
    ] + [
        _observation(
            result,
            "value_overlap",
            ("col_0", "col_1"),
        ).evidence_id
    ]


def test_dataframe_orchestration_reuses_loaded_frame_and_preserves_input():
    dataframe = pd.DataFrame({
        "first": ["a", "b", "a", "c"],
        "second": ["a", "b", "a", "c"],
    })
    result = _result(dataframe)
    result_before = result.model_dump(mode="json")
    frame_before = dataframe.copy(deep=True)

    discovered = discover_relationships_from_dataframe(result, dataframe)

    pd.testing.assert_frame_equal(dataframe, frame_before)
    assert result.model_dump(mode="json") == result_before
    assert discovered.pairwise_evidence
    assert any(
        relation.relationship_kind == "exact_duplicate_columns"
        for relation in discovered.relationships
    )


def test_complete_value_inclusion_and_unique_reference_is_only_a_candidate():
    dataframe = pd.DataFrame({
        "observed_key": [1, 2] * 6,
        "reference_values": list(range(1, 13)),
    })
    result, _ = _build(dataframe)

    discovered = discover_relationships(result)

    candidate = next(
        relation
        for relation in discovered.relationships
        if relation.relationship_kind == "candidate_key_inclusion"
    )
    assert candidate.participating_column_ids == ["col_0", "col_1"]
    assert candidate.confidence is None
    assert "do not establish" in candidate.uncertainty_reasons[0]


def test_missing_or_nonunique_reference_values_do_not_form_key_inclusion():
    missing_key = pd.DataFrame({
        "observed_key": [1, 2] * 6,
        "reference_values": [*range(1, 12), None],
    })
    missing_result, _ = _build(missing_key)
    assert not any(
        relation.relationship_kind == "candidate_key_inclusion"
        for relation in discover_relationships(missing_result).relationships
    )

    nonunique_key = pd.DataFrame({
        "observed_key": [1, 2] * 6,
        "reference_values": [1, 2, 3, 4, 5, 6] * 2,
    })
    nonunique_result, _ = _build(nonunique_key)
    assert not any(
        relation.relationship_kind == "candidate_key_inclusion"
        for relation in discover_relationships(nonunique_result).relationships
    )


def test_functional_dependency_evidence_records_violations_and_candidates():
    dataframe = pd.DataFrame({
        "determinant": [row % 4 for row in range(12)],
        "dependent": [row % 2 for row in range(12)],
        "violating_value": [row for row in range(12)],
    })
    result, _ = _build(dataframe)
    fd = _observation(result, "functional_dependency")

    assert fd.measured_summary["left_to_right_determinant_count"] == 4
    assert fd.measured_summary["left_to_right_violation_count"] == 0
    assert fd.measured_summary["right_to_left_violation_count"] == 2
    discovered = discover_relationships(result)
    assert any(
        relation.relationship_kind == "candidate_functional_dependency"
        and relation.participating_column_ids == ["col_0", "col_1"]
        for relation in discovered.relationships
    )
    assert not any(
        relation.relationship_kind == "candidate_functional_dependency"
        and relation.participating_column_ids == ["col_0", "col_2"]
        for relation in discovered.relationships
    )


def test_small_samples_and_sampled_prefix_do_not_produce_relations():
    small = pd.DataFrame({
        "left": [1, 2, 1, 3],
        "right": [1, 2, 1, 3],
    })
    small_result, _ = _build(small)
    assert any(
        relation.relationship_kind == "exact_duplicate_columns"
        for relation in discover_relationships(small_result).relationships
    )
    inclusion = pd.DataFrame({
        "left": [1, 2, 1, 2],
        "right": [1, 2, 3, 4],
    })
    small_inclusion, _ = _build(inclusion)
    assert not discover_relationships(small_inclusion).relationships

    sampled, _ = _build(
        pd.DataFrame({"first": range(6), "second": range(6)}),
        limits=PairwiseEvidenceLimits(max_sample_rows=3),
    )
    assert all(
        item.scope.kind == "sample"
        for item in sampled.pairwise_evidence
    )
    assert not discover_relationships(sampled).relationships
    assert any(
        warning.warning_code == "pairwise_row_sampled"
        for warning in sampled.warnings
    )


def test_pair_count_cardinality_and_work_limits_report_omitted_evidence():
    wide = pd.DataFrame({
        f"field_{index}": range(4)
        for index in range(4)
    })
    pair_limited, _ = _build(
        wide,
        limits=PairwiseEvidenceLimits(max_column_pairs=1),
    )
    assert len(pair_limited.pairwise_evidence) == 3
    assert any(
        warning.warning_code == "pairwise_comparison_limit_exceeded"
        for warning in pair_limited.warnings
    )

    cardinality_limited, _ = _build(
        pd.DataFrame({
            "left_category": ["north", "south", "west"],
            "right_category": ["north", "south", "west"],
        }),
        limits=PairwiseEvidenceLimits(max_distinct_values=2),
    )
    assert cardinality_limited.pairwise_evidence == []
    assert any(
        warning.warning_code == "pairwise_cardinality_limit_exceeded"
        for warning in cardinality_limited.warnings
    )

    work_limited, _ = _build(
        wide,
        limits=PairwiseEvidenceLimits(max_cell_comparisons=40),
    )
    assert len(work_limited.pairwise_evidence) == 3
    assert any(
        warning.warning_code == "pairwise_comparison_limit_exceeded"
        for warning in work_limited.warnings
    )


def test_constant_columns_and_zero_variance_do_not_make_key_relations():
    dataframe = pd.DataFrame({
        "constant_a": [1] * 12,
        "constant_b": [1] * 12,
        "varying": list(range(12)),
    })
    result, _ = _build(dataframe)

    assert not discover_relationships(result).relationships
    assert _observation(
        result,
        "exact_column_equality",
        ("col_0", "col_1"),
    ).measured_value is True


def test_empty_dataset_and_unhashable_values_report_without_raw_values():
    empty = pd.DataFrame(columns=["first", "second"])
    empty_result, _ = _build(empty)
    assert empty_result.pairwise_evidence == []
    assert any(
        warning.warning_code == "pairwise_empty_dataset"
        for warning in empty_result.warnings
    )

    no_rows_available = _result(pd.DataFrame({"first": [], "second": []}))
    no_rows_data = no_rows_available.model_dump(mode="python")
    no_rows_data["analysis_scope"] = AnalysisScope(
        kind="sample",
        rows_analyzed=0,
        total_rows=4,
    )
    no_rows_available = DatasetUnderstandingResult.model_validate(no_rows_data)
    no_rows_result = build_pairwise_evidence(
        pd.DataFrame({"first": [], "second": []}),
        no_rows_available,
    )
    assert no_rows_result.pairwise_evidence == ()
    assert no_rows_result.warnings[0].warning_code == "pairwise_empty_sample"

    nested = pd.DataFrame({
        "nested": [[1], [2], [3]],
        "other": ["secret-a", "secret-b", "secret-c"],
    })
    nested_result, _ = _build(nested)
    assert nested_result.pairwise_evidence == []
    assert any(
        warning.warning_code == "pairwise_unsupported_values"
        for warning in nested_result.warnings
    )
    assert "secret-a" not in nested_result.model_dump_json()


def test_no_complete_case_values_are_explicit_and_do_not_create_relations():
    dataframe = pd.DataFrame({
        "nullable_a": [None, None, None],
        "nullable_b": [1, 2, 3],
    })
    result, _ = _build(dataframe)

    overlap = _observation(result, "value_overlap")
    dependency = _observation(result, "functional_dependency")
    assert overlap.sample_size == dependency.sample_size == 0
    assert overlap.measured_summary["complete_case_rows"] == 0
    assert any("No complete-case values" in item for item in overlap.limitations)
    assert not discover_relationships(result).relationships


def test_ambiguous_types_do_not_change_pairwise_structural_provenance():
    dataframe = pd.DataFrame({
        "numeric_or_date": [20240101, 20240201, 20240301],
        "other": [20240101, 20240201, 20240301],
    })
    result, _ = _build(dataframe)
    evidence = [
        EvidenceItem(
            evidence_id="numeric-candidate-evidence",
            kind="deterministic_rule",
            method_id="infer_series_types",
            method_version="1",
            affected_column_ids=["col_0"],
            metric_name="type_candidate_observed",
            measured_category="numeric",
            measured_value=None,
            sample_size=3,
            scope=result.analysis_scope,
        ),
        EvidenceItem(
            evidence_id="datetime-candidate-evidence",
            kind="deterministic_rule",
            method_id="infer_series_types",
            method_version="1",
            affected_column_ids=["col_0"],
            metric_name="type_candidate_observed",
            measured_category="datetime",
            measured_value=None,
            sample_size=3,
            scope=result.analysis_scope,
        ),
    ]
    numeric = Hypothesis(
        hypothesis_id="numeric-semantic-candidate",
        category="semantic_type",
        proposed_value="numeric_measurement",
        status="ambiguous",
        supporting_evidence_ids=[evidence[0].evidence_id],
        alternative_hypothesis_ids=["datetime-semantic-candidate"],
    )
    temporal = Hypothesis(
        hypothesis_id="datetime-semantic-candidate",
        category="semantic_type",
        proposed_value="datetime",
        status="ambiguous",
        supporting_evidence_ids=[evidence[1].evidence_id],
        alternative_hypothesis_ids=["numeric-semantic-candidate"],
    )
    augmented = result.model_dump(mode="python")
    augmented["evidence"].extend(evidence)
    augmented["columns"][0]["semantic_hypotheses"] = [numeric, temporal]
    result = DatasetUnderstandingResult.model_validate(augmented)

    assert all(
        item.method_id == "pairwise_structural_comparison"
        for item in result.pairwise_evidence
    )
    assert any(
        item.observation_type == "exact_column_equality"
        and item.measured_value is True
        for item in result.pairwise_evidence
    )
    assert not any(
        "correlation" in relation.relationship_kind
        for relation in discover_relationships(result).relationships
    )


def test_pairwise_contracts_reject_invalid_columns_summaries_and_references():
    base = _result(pd.DataFrame({"a": [1, 2], "b": [1, 2]}))
    with pytest.raises(ValidationError):
        PairwiseEvidence(
            evidence_id="pair-invalid",
            observation_type="value_overlap",
            column_ids=["col_0", "col_0"],
            method_id="pairwise_method",
            method_version="1",
            sample_size=2,
            denominator=2,
            scope=base.analysis_scope,
            measured_summary={"left_unique_count": 2},
        )

    built, evidence = _build(pd.DataFrame({"a": [1, 2], "b": [1, 2]}))
    invalid_column = built.model_dump(mode="python")
    invalid_column["pairwise_evidence"] = [
        evidence[0].model_copy(update={"column_ids": ["col_0", "col_9"]})
    ]
    with pytest.raises(ValidationError):
        DatasetUnderstandingResult.model_validate(invalid_column)

    duplicate_observation = built.model_dump(mode="python")
    duplicate_observation["pairwise_evidence"] = [
        evidence[0],
        evidence[0].model_copy(update={"evidence_id": "pair-another-id"}),
    ]
    with pytest.raises(ValidationError):
        DatasetUnderstandingResult.model_validate(duplicate_observation)

    invalid_reference = built.model_dump(mode="python")
    invalid_reference["relationships"] = [
        RelationshipHypothesis(
            relationship_id="rel-invalid",
            relationship_kind="candidate_key_inclusion",
            participating_column_ids=["col_0", "col_1"],
            supporting_evidence_ids=["unknown_pair_evidence"],
        )
    ]
    with pytest.raises(ValidationError):
        DatasetUnderstandingResult.model_validate(invalid_reference)


def test_limits_reject_nonpositive_values():
    with pytest.raises(ValueError):
        PairwiseEvidenceLimits(max_column_pairs=0)
