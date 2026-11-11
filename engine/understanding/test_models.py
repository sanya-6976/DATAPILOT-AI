from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from engine.understanding.models import (
    AnalysisScope,
    ColumnUnderstanding,
    ConfidenceEstimate,
    DatasetUnderstandingResult,
    EvidenceItem,
    Hypothesis,
    RelationshipHypothesis,
    UnderstandingWarning,
)

DATASET_ID = "12345678123456781234567812345678"
FINGERPRINT = "a" * 64
GENERATED_AT = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)


def _scope() -> AnalysisScope:
    return AnalysisScope(
        kind="full_dataset",
        rows_analyzed=3,
        total_rows=3,
    )


def _evidence(
    evidence_id: str = "evidence-1",
    *,
    affected_column_ids: list[str] | None = None,
    measured_value: float | bool | None = 3,
    parameters: dict | None = None,
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        kind="profile_metric",
        method_id="missing_value_count",
        method_version="1",
        affected_column_ids=affected_column_ids or ["col_0"],
        metric_name="missing_count",
        measured_value=measured_value,
        sample_size=3,
        denominator=3,
        scope=_scope(),
        parameters=parameters or {},
    )


def _hypothesis(
    hypothesis_id: str = "hypothesis-1",
    *,
    category: str = "semantic_type",
    proposed_value: str = "numeric_measurement",
    status: str = "candidate",
    supporting: list[str] | None = None,
    contradicting: list[str] | None = None,
    alternatives: list[str] | None = None,
    confidence: ConfidenceEstimate | None = None,
) -> Hypothesis:
    return Hypothesis(
        hypothesis_id=hypothesis_id,
        category=category,
        proposed_value=proposed_value,
        status=status,
        confidence=confidence,
        supporting_evidence_ids=supporting or [],
        contradicting_evidence_ids=contradicting or [],
        alternative_hypothesis_ids=alternatives or [],
        uncertainty_reasons=[],
    )


def _column(
    position: int,
    *,
    name: str = "value",
    semantic: list[Hypothesis] | None = None,
    roles: list[Hypothesis] | None = None,
    units: list[Hypothesis] | None = None,
    evidence_ids: list[str] | None = None,
) -> ColumnUnderstanding:
    return ColumnUnderstanding(
        column_id=f"col_{position}",
        source_name=name,
        position=position,
        semantic_hypotheses=semantic or [],
        role_hypotheses=roles or [],
        unit_hypotheses=units or [],
        relevant_evidence_ids=evidence_ids or [],
    )


def _result(**overrides) -> DatasetUnderstandingResult:
    values = {
        "dataset_id": DATASET_ID,
        "dataset_fingerprint": FINGERPRINT,
        "generated_at": GENERATED_AT,
        "understanding_schema_version": "1.0",
        "engine_version": "0.1.0",
        "profile_version": "1",
        "analysis_scope": _scope(),
    }
    values.update(overrides)
    return DatasetUnderstandingResult(**values)


def test_valid_empty_result_can_represent_unknown_understanding():
    result = _result(dataset_fingerprint=None)

    assert result.columns == []
    assert result.relationships == []
    assert result.evidence == []
    assert result.warnings == []
    assert result.dataset_fingerprint is None


def test_unsupported_hypothesis_remains_explicitly_uncertain():
    result = _result(
        columns=[
            _column(
                0,
                semantic=[
                    _hypothesis(
                        "unknown-type",
                        proposed_value="unknown",
                        status="unsupported",
                        supporting=["evidence-1"],
                    )
                ],
            )
        ],
        evidence=[_evidence()],
    )

    hypothesis = result.columns[0].semantic_hypotheses[0]
    assert hypothesis.status == "unsupported"
    assert hypothesis.confidence is None


def test_valid_semantic_role_and_multiple_hypotheses():
    evidence = _evidence(affected_column_ids=["col_0"])
    column = _column(
        0,
        semantic=[
            _hypothesis(
                "semantic-numeric",
                supporting=["evidence-1"],
            ),
            _hypothesis(
                "semantic-categorical",
                proposed_value="categorical",
                supporting=["evidence-1"],
            ),
        ],
        roles=[
            _hypothesis(
                "role-feature",
                category="field_role",
                proposed_value="feature",
                supporting=["evidence-1"],
            ),
        ],
        units=[
            _hypothesis(
                "unit-percentage",
                category="unit",
                proposed_value="percentage",
                supporting=["evidence-1"],
            ),
        ],
        evidence_ids=["evidence-1"],
    )

    result = _result(columns=[column], evidence=[evidence])

    assert len(result.columns[0].semantic_hypotheses) == 2
    assert result.columns[0].role_hypotheses[0].proposed_value == "feature"
    assert result.columns[0].unit_hypotheses[0].proposed_value == "percentage"


def test_duplicate_source_names_have_distinct_positional_ids():
    result = _result(
        columns=[
            _column(0, name="measure"),
            _column(1, name="measure"),
        ],
    )

    assert [column.source_name for column in result.columns] == [
        "measure",
        "measure",
    ]
    assert [column.column_id for column in result.columns] == [
        "col_0",
        "col_1",
    ]


def test_confidence_is_nullable_and_only_accepts_calibrated_probability():
    assert _hypothesis(supporting=["evidence-1"]).confidence is None
    confidence = ConfidenceEstimate(
        probability=0.75,
        calibration_method="isotonic",
        calibration_version="cal-1",
        evaluation_reference="heldout-v1",
    )

    assert (
        _hypothesis(
            confidence=confidence,
            supporting=["evidence-1"],
        ).confidence
        == confidence
    )


@pytest.mark.parametrize("probability", [-0.01, 1.01])
def test_confidence_rejects_values_outside_probability_range(probability):
    with pytest.raises(ValidationError):
        ConfidenceEstimate(
            probability=probability,
            calibration_method="isotonic",
            calibration_version="cal-1",
            evaluation_reference="heldout-v1",
        )


def test_duplicate_evidence_ids_are_rejected():
    with pytest.raises(ValidationError, match="Evidence IDs must be unique"):
        _result(evidence=[_evidence(), _evidence()])


def test_duplicate_hypothesis_ids_are_rejected():
    hypothesis = _hypothesis(supporting=["evidence-1"])
    with pytest.raises(ValidationError, match="Hypothesis IDs must be unique"):
        _result(
            columns=[
                _column(0, semantic=[hypothesis]),
                _column(1, semantic=[hypothesis.model_copy()]),
            ],
            evidence=[_evidence()],
        )


def test_missing_evidence_references_are_rejected():
    with pytest.raises(ValidationError, match="Unknown supporting evidence"):
        _result(
            columns=[
                _column(
                    0,
                    semantic=[
                        _hypothesis(supporting=["missing-evidence"]),
                    ],
                ),
            ],
        )


def test_hypothesis_references_must_belong_to_its_column():
    with pytest.raises(
        ValidationError,
        match="Hypothesis evidence must refer to its column",
    ):
        _result(
            columns=[
                _column(
                    0,
                    semantic=[_hypothesis(supporting=["other-column"])],
                ),
                _column(1),
            ],
            evidence=[
                _evidence(
                    "other-column",
                    affected_column_ids=["col_1"],
                )
            ],
        )


def test_relationship_evidence_must_cover_all_participating_columns():
    with pytest.raises(
        ValidationError,
        match="Relationship evidence must refer to all participating columns",
    ):
        _result(
            columns=[_column(0), _column(1)],
            evidence=[_evidence(affected_column_ids=["col_0"])],
            relationships=[
                RelationshipHypothesis(
                    relationship_id="relationship-1",
                    relationship_kind="candidate_relationship",
                    participating_column_ids=["col_0", "col_1"],
                    supporting_evidence_ids=["evidence-1"],
                )
            ],
        )


def test_dataset_evidence_identity_is_bound_to_result_identity():
    evidence_data = _evidence().model_dump()
    evidence_data["dataset_id"] = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    with pytest.raises(ValidationError, match="Evidence dataset_id"):
        _result(
            columns=[_column(0)],
            evidence=[EvidenceItem(**evidence_data)],
        )


def test_validated_contract_collections_and_fields_are_immutable():
    result = _result(
        columns=[_column(0, evidence_ids=["evidence-1"])],
        evidence=[_evidence(parameters={"threshold": 0.9})],
    )

    with pytest.raises(ValidationError):
        result.dataset_id = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    with pytest.raises(TypeError, match="immutable"):
        result.evidence.append(_evidence("evidence-2"))
    with pytest.raises(TypeError, match="immutable"):
        result.evidence[0].parameters["threshold"] = 0.1
    with pytest.raises(TypeError, match="immutable"):
        result.columns[0].semantic_hypotheses.append(
            _hypothesis(supporting=["evidence-1"])
        )


def test_hypotheses_and_relationships_require_evidence():
    with pytest.raises(
        ValidationError,
        match="requires supporting or contradicting evidence",
    ):
        _hypothesis()

    with pytest.raises(
        ValidationError,
        match="relationship hypothesis requires evidence",
    ):
        RelationshipHypothesis(
            relationship_id="relationship-1",
            relationship_kind="correlated",
            participating_column_ids=["col_0", "col_1"],
        )


def test_invalid_column_references_are_rejected():
    with pytest.raises(ValidationError, match="Unknown relationship column"):
        _result(
            columns=[_column(0), _column(1)],
            evidence=[
                _evidence(
                    affected_column_ids=["col_0", "col_1"],
                )
            ],
            relationships=[
                RelationshipHypothesis(
                    relationship_id="relationship-1",
                    relationship_kind="correlated",
                    participating_column_ids=["col_0", "col_9"],
                    supporting_evidence_ids=["evidence-1"],
                )
            ],
        )

    with pytest.raises(ValidationError, match="Unknown evidence column"):
        _result(
            evidence=[_evidence(affected_column_ids=["col_9"])],
        )

    with pytest.raises(ValidationError, match="Unknown warning column"):
        _result(
            warnings=[
                UnderstandingWarning(
                    warning_code="insufficient_data",
                    explanation="Not enough observations.",
                    affected_column_ids=["col_4"],
                    severity="info",
                )
            ],
        )


@pytest.mark.parametrize(
    "fingerprint",
    ["not-a-hash", "A" * 64, "a" * 63, "g" * 64],
)
def test_invalid_dataset_fingerprints_are_rejected(fingerprint):
    with pytest.raises(ValidationError, match="dataset_fingerprint"):
        _result(dataset_fingerprint=fingerprint)


@pytest.mark.parametrize(
    ("column_id", "position"),
    [("col_1", 0), ("col_00", 0), ("column_0", 0), ("col_0", -1)],
)
def test_invalid_column_ids_and_positions_are_rejected(column_id, position):
    with pytest.raises(ValidationError):
        ColumnUnderstanding(
            column_id=column_id,
            source_name="value",
            position=position,
        )


def test_result_rejects_noncontiguous_or_unordered_column_positions():
    with pytest.raises(ValidationError, match="contiguous zero-based position"):
        _result(columns=[_column(1)])

    with pytest.raises(ValidationError, match="contiguous zero-based position"):
        _result(columns=[_column(1), _column(0)])


def test_alternative_hypothesis_references_must_resolve_compatibly():
    first = _hypothesis(
        "type-one",
        supporting=["evidence-1"],
        alternatives=["type-two"],
    )
    second = _hypothesis(
        "type-two",
        proposed_value="categorical",
        supporting=["evidence-1"],
    )

    result = _result(
        columns=[_column(0, semantic=[first, second])],
        evidence=[_evidence()],
    )

    assert result.columns[0].semantic_hypotheses[0].alternative_hypothesis_ids == [
        "type-two"
    ]

    with pytest.raises(
        ValidationError,
        match="Unknown alternative hypothesis",
    ):
        _result(
            columns=[
                _column(
                    0,
                    semantic=[
                        _hypothesis(
                            supporting=["evidence-1"],
                            alternatives=["not-present"],
                        ),
                    ],
                )
            ],
            evidence=[_evidence()],
        )


def test_contradictory_evidence_is_preserved():
    hypothesis = _hypothesis(
        supporting=["supports"],
        contradicting=["contradicts"],
    )
    result = _result(
        columns=[
            _column(
                0,
                semantic=[hypothesis],
                evidence_ids=["supports", "contradicts"],
            )
        ],
        evidence=[
            _evidence("supports"),
            _evidence("contradicts"),
        ],
    )

    stored = result.columns[0].semantic_hypotheses[0]
    assert stored.supporting_evidence_ids == ["supports"]
    assert stored.contradicting_evidence_ids == ["contradicts"]


def test_evidence_contract_does_not_accept_raw_string_measurements():
    with pytest.raises(ValidationError):
        _evidence(measured_value="raw cell value")


def test_json_round_trip_and_parameter_order_are_deterministic():
    first = _result(
        columns=[_column(0)],
        evidence=[
            _evidence(
                parameters={"threshold": 0.9, "minimum_count": 5},
            )
        ],
    )
    second = _result(
        columns=[_column(0)],
        evidence=[
            _evidence(
                parameters={"minimum_count": 5, "threshold": 0.9},
            )
        ],
    )

    first_json = first.model_dump_json()
    second_json = second.model_dump_json()
    assert first_json == second_json

    restored = DatasetUnderstandingResult.model_validate_json(first_json)
    assert restored == first


def test_dataset_uuid_is_normalized_to_existing_storage_identity_format():
    result = _result(
        dataset_id="12345678-1234-5678-1234-567812345678",
    )

    assert result.dataset_id == DATASET_ID
