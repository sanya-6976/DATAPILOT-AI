"""Conservative field-role hypotheses derived from existing evidence."""

import hashlib
import json

from engine.understanding.models import (
    DatasetUnderstandingResult,
    EvidenceItem,
    Hypothesis,
    UnderstandingWarning,
)

_SEMANTIC_ROLE_MAP = {
    "numeric_measurement": "measurement_field",
    "datetime": "temporal_field",
    "categorical": "grouping_category",
    "boolean": "grouping_category",
    "identifier": "identifier_candidate",
}
_METHOD_ID = "field_role_inference"
_METHOD_VERSION = "1"


def infer_field_roles(
    result: DatasetUnderstandingResult,
) -> DatasetUnderstandingResult:
    """Add evidence-backed structural and analytical role hypotheses.

    Semantic hypotheses must already be present. No column names or source
    values are inspected, and target/feature roles are not guessed without an
    analytical task context.
    """

    result = DatasetUnderstandingResult.model_validate(
        result.model_dump(mode="python")
    )
    dataset_rows = _single_value(
        result.evidence,
        "dataset_row_count",
        affected_column_id=None,
    )
    updated_columns = []
    for column in result.columns:
        column_evidence = [
            item
            for item in result.evidence
            if column.column_id in item.affected_column_ids
        ]
        specs: dict[str, dict[str, object]] = {}
        semantic_role_hypotheses = {
            hypothesis.proposed_value: hypothesis
            for hypothesis in column.semantic_hypotheses
            if hypothesis.proposed_value in _SEMANTIC_ROLE_MAP
        }

        for semantic_value, semantic_hypothesis in (
            semantic_role_hypotheses.items()
        ):
            role = _SEMANTIC_ROLE_MAP[semantic_value]
            spec = specs.setdefault(
                role,
                {
                    "supporting": [],
                    "contradicting": [],
                    "uncertainty": [],
                    "status": "candidate",
                    "semantic_values": [],
                },
            )
            spec["supporting"].extend(
                semantic_hypothesis.supporting_evidence_ids
            )
            spec["contradicting"].extend(
                semantic_hypothesis.contradicting_evidence_ids
            )
            spec["uncertainty"].extend(
                semantic_hypothesis.uncertainty_reasons
            )
            spec["semantic_values"].append(semantic_hypothesis)
            if semantic_hypothesis.status == "ambiguous":
                spec["status"] = "ambiguous"
            elif semantic_hypothesis.status == "unsupported":
                spec["status"] = "unsupported"
            if semantic_value == "identifier":
                spec["uncertainty"].append(
                    "Identifier-like evidence does not establish a key."
                )
            elif semantic_value in {"categorical", "boolean"}:
                spec["uncertainty"].append(
                    "The analytical task does not establish whether this "
                    "field is used for grouping."
                )
            elif semantic_value == "numeric_measurement":
                spec["uncertainty"].append(
                    "Measurement evidence does not establish whether this "
                    "field is selected as a predictor."
                )
            elif semantic_value == "datetime":
                spec["uncertainty"].append(
                    "Datetime evidence does not establish how this field is "
                    "used in an analysis."
                )

        identifier_hypothesis = semantic_role_hypotheses.get("identifier")
        if identifier_hypothesis is not None:
            row_count = dataset_rows
            missing = _single_value(
                column_evidence,
                "column_missing_count",
                affected_column_id=column.column_id,
            )
            unique = _single_value(
                column_evidence,
                "column_unique_count",
                affected_column_id=column.column_id,
            )
            if (
                row_count is not None
                and missing is not None
                and unique is not None
            ):
                supporting = [
                    *identifier_hypothesis.supporting_evidence_ids,
                    row_count.evidence_id,
                ]
                contradicting: list[str] = []
                uncertainty = [
                    (
                        "Uniqueness is only a structural signal; it does not "
                        "establish entity identity or a declared primary key."
                    )
                ]
                if missing.measured_value == 0 and (
                    unique.measured_value == row_count.measured_value
                ):
                    supporting.extend([missing.evidence_id, unique.evidence_id])
                    status = "candidate"
                else:
                    status = "unsupported"
                    if missing.measured_value != 0:
                        contradicting.append(missing.evidence_id)
                        uncertainty.append(
                            "Missing values contradict a complete primary-key "
                            "candidate."
                        )
                    if unique.measured_value != (
                        row_count.measured_value
                        - missing.measured_value
                    ):
                        contradicting.append(unique.evidence_id)
                        uncertainty.append(
                            "Repeated non-missing values contradict a "
                            "single-column primary-key candidate."
                        )
                specs["primary_key_candidate"] = {
                    "supporting": supporting,
                    "contradicting": contradicting,
                    "uncertainty": uncertainty,
                    "status": status,
                    "semantic_values": [],
                }

        semantic_to_role_id = {
            semantic_value: _role_hypothesis_id(
                result,
                column.column_id,
                _SEMANTIC_ROLE_MAP[semantic_value],
            )
            for semantic_value in semantic_role_hypotheses
        }
        for role, spec in specs.items():
            spec["supporting"] = _unique(spec["supporting"])
            spec["contradicting"] = _unique(spec["contradicting"])
            spec["uncertainty"] = _unique(spec["uncertainty"])

        role_hypotheses = list(column.role_hypotheses)
        existing_ids = {
            hypothesis.hypothesis_id for hypothesis in role_hypotheses
        }
        existing_values = {
            hypothesis.proposed_value for hypothesis in role_hypotheses
        }
        for role, spec in specs.items():
            hypothesis_id = _role_hypothesis_id(
                result,
                column.column_id,
                role,
            )
            if hypothesis_id in existing_ids or role in existing_values:
                continue
            semantic_values = spec["semantic_values"]
            alternatives = []
            for semantic_hypothesis in semantic_values:
                alternatives.extend(
                    semantic_to_role_id[semantic_value]
                    for semantic_value, other in semantic_role_hypotheses.items()
                    if (
                        other.hypothesis_id
                        in semantic_hypothesis.alternative_hypothesis_ids
                        and _SEMANTIC_ROLE_MAP[semantic_value] != role
                    )
                )
            role_hypotheses.append(
                Hypothesis(
                    hypothesis_id=hypothesis_id,
                    category="field_role",
                    proposed_value=role,
                    status=spec["status"],
                    confidence=None,
                    supporting_evidence_ids=spec["supporting"],
                    contradicting_evidence_ids=spec["contradicting"],
                    alternative_hypothesis_ids=_unique(alternatives),
                    uncertainty_reasons=spec["uncertainty"],
                    method_id=(
                        _METHOD_ID
                        if result.analysis_scope.rows_analyzed is not None
                        else None
                    ),
                    method_version=(
                        _METHOD_VERSION
                        if result.analysis_scope.rows_analyzed is not None
                        else None
                    ),
                    analysis_scope=(
                        result.analysis_scope
                        if result.analysis_scope.rows_analyzed is not None
                        else None
                    ),
                    sample_size=result.analysis_scope.rows_analyzed,
                )
            )
            existing_ids.add(hypothesis_id)
            existing_values.add(role)
        updated_columns.append(
            column.model_copy(update={"role_hypotheses": role_hypotheses})
        )

    warnings = list(result.warnings)
    if result.columns:
        warnings.extend([
            UnderstandingWarning(
                warning_code="analytical_role_context_unavailable",
                explanation=(
                    "Potential target and feature roles are not inferred "
                    "without an explicit analytical task or target "
                    "designation; column names are not used as proof."
                ),
                affected_column_ids=[column.column_id for column in result.columns],
                severity="info",
            ),
            UnderstandingWarning(
                warning_code="foreign_key_role_evidence_unavailable",
                explanation=(
                    "Field-role inference does not assign foreign-key roles. "
                    "Review separately generated pairwise relationship "
                    "hypotheses; column names alone are not proof."
                ),
                affected_column_ids=[column.column_id for column in result.columns],
                severity="info",
            ),
        ])
    updated = result.model_dump(mode="python")
    updated["columns"] = updated_columns
    updated["warnings"] = _unique_warnings(warnings)
    return DatasetUnderstandingResult.model_validate(updated)


def _single_value(
    evidence: list[EvidenceItem],
    metric_name: str,
    *,
    affected_column_id: str | None,
) -> EvidenceItem | None:
    matches = [
        item
        for item in evidence
        if item.metric_name == metric_name
        and (
            (
                affected_column_id is None
                and not item.affected_column_ids
            )
            or (
                affected_column_id is not None
                and item.affected_column_ids == [affected_column_id]
            )
        )
        and item.measured_value is not None
    ]
    return matches[0] if len(matches) == 1 else None


def _role_hypothesis_id(
    result: DatasetUnderstandingResult,
    column_id: str,
    role: str,
) -> str:
    identity = result.dataset_fingerprint or result.dataset_id
    payload = json.dumps(
        [identity, column_id, "field_role", role],
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:40]
    return f"role_{digest}"


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


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
