"""Versioned contracts for dataset understanding and evidence provenance."""

import re
import uuid
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StringConstraints,
    field_validator,
    model_validator,
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_COLUMN_ID_PATTERN = re.compile(r"^col_(0|[1-9][0-9]*)$")
_UUID_HEX_PATTERN = re.compile(r"^[0-9a-fA-F]{32}$")
_UUID_DASHED_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


class _FrozenList(list):
    def _immutable(self, *_args, **_kwargs):
        raise TypeError("Validated contract collections are immutable.")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __iadd__ = _immutable
    __imul__ = _immutable
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable


class _FrozenDict(dict):
    def _immutable(self, *_args, **_kwargs):
        raise TypeError("Validated contract mappings are immutable.")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __ior__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


def _freeze_collection(value):
    if isinstance(value, _FrozenList | _FrozenDict):
        return value
    if isinstance(value, list):
        return _FrozenList(_freeze_collection(item) for item in value)
    if isinstance(value, dict):
        return _FrozenDict(
            (key, _freeze_collection(item))
            for key, item in value.items()
        )
    return value


def _normalize_dataset_id(value: str) -> str:
    if not (
        _UUID_HEX_PATTERN.fullmatch(value)
        or _UUID_DASHED_PATTERN.fullmatch(value)
    ):
        raise ValueError("dataset_id must be a UUID.")
    try:
        return uuid.UUID(value).hex
    except ValueError as exc:
        raise ValueError("dataset_id must be a valid UUID.") from exc


def _validate_fingerprint(value: str | None) -> str | None:
    if value is not None and _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(
            "dataset_fingerprint must be a lowercase SHA-256 digest."
        )
    return value

Identifier = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=128,
        pattern=_IDENTIFIER_PATTERN.pattern,
    ),
]
EntityId = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    ),
]
MeasuredValue = StrictInt | StrictFloat | StrictBool | None
ParameterValue = StrictInt | StrictFloat | StrictBool | None
PairwiseSummaryValue = StrictInt | StrictFloat | StrictBool


class ContractModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        frozen=True,
    )

    def model_post_init(self, _context: Any) -> None:
        for field_name, value in self.__dict__.items():
            if isinstance(value, list | dict):
                object.__setattr__(
                    self,
                    field_name,
                    _freeze_collection(value),
                )


class AnalysisScope(ContractModel):
    """Describes which rows a deterministic observation covers."""

    kind: Literal["full_dataset", "sample"]
    rows_analyzed: int | None = Field(default=None, ge=0)
    total_rows: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_row_counts(self) -> "AnalysisScope":
        if (
            self.rows_analyzed is not None
            and self.total_rows is not None
            and self.rows_analyzed > self.total_rows
        ):
            raise ValueError("rows_analyzed cannot exceed total_rows.")
        if (
            self.kind == "full_dataset"
            and self.rows_analyzed is not None
            and self.total_rows is not None
            and self.rows_analyzed != self.total_rows
        ):
            raise ValueError(
                "Full-dataset scope must analyze all declared rows."
            )
        return self


class ConfidenceEstimate(ContractModel):
    """A calibrated probability; uncalibrated ranking scores are not allowed."""

    probability: float = Field(ge=0.0, le=1.0)
    calibration_method: Identifier
    calibration_version: EntityId
    evaluation_reference: EntityId


class EvidenceItem(ContractModel):
    """A reproducible aggregate observation, not a raw cell value."""

    evidence_id: EntityId
    dataset_id: str | None = None
    dataset_fingerprint: str | None = None
    kind: Literal[
        "observed_fact",
        "profile_metric",
        "deterministic_rule",
    ]
    method_id: Identifier
    method_version: EntityId | None = None
    affected_column_ids: list[str] = Field(default_factory=list)
    metric_name: Identifier
    measured_category: Identifier | None = None
    measured_value: MeasuredValue = Field(
        description=(
            "Aggregate or deterministic metric result; never a raw cell value."
        )
    )
    sample_size: int | None = Field(default=None, ge=0)
    denominator: int | None = Field(default=None, ge=0)
    scope: AnalysisScope
    parameters: dict[Identifier, ParameterValue] = Field(default_factory=dict)

    @field_validator("evidence_id")
    @classmethod
    def validate_evidence_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("evidence_id must not be blank.")
        return value

    @field_validator("dataset_id")
    @classmethod
    def normalize_dataset_id(cls, value: str | None) -> str | None:
        return None if value is None else _normalize_dataset_id(value)

    @field_validator("dataset_fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str | None) -> str | None:
        return _validate_fingerprint(value)

    @field_validator("affected_column_ids")
    @classmethod
    def validate_affected_column_ids(
        cls,
        value: list[str],
    ) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("affected_column_ids must be unique.")
        if any(_COLUMN_ID_PATTERN.fullmatch(item) is None for item in value):
            raise ValueError("Invalid positional column ID.")
        return value

    @field_validator("parameters")
    @classmethod
    def sort_parameters(
        cls,
        value: dict[str, ParameterValue],
    ) -> dict[str, ParameterValue]:
        return dict(sorted(value.items()))

    @model_validator(mode="after")
    def validate_evidence_counts(self) -> "EvidenceItem":
        if (self.measured_category is None) == (self.measured_value is None):
            raise ValueError(
                "Provide exactly one of measured_category or measured_value."
            )
        if (
            self.sample_size is not None
            and self.denominator is not None
            and self.sample_size > self.denominator
        ):
            raise ValueError("sample_size cannot exceed denominator.")
        if (
            self.scope.rows_analyzed is not None
            and self.sample_size is not None
            and self.sample_size > self.scope.rows_analyzed
        ):
            raise ValueError(
                "sample_size cannot exceed rows_analyzed."
            )
        return self


class PairwiseEvidence(ContractModel):
    """A bounded aggregate observation involving exactly two columns."""

    evidence_id: EntityId
    dataset_id: str | None = None
    dataset_fingerprint: str | None = None
    observation_type: Literal[
        "exact_column_equality",
        "value_overlap",
        "functional_dependency",
    ]
    column_ids: list[str] = Field(min_length=2, max_length=2)
    method_id: Identifier
    method_version: EntityId
    sample_size: int = Field(ge=0)
    denominator: int = Field(ge=0)
    scope: AnalysisScope
    measured_value: StrictInt | StrictFloat | StrictBool | None = None
    measured_summary: dict[Identifier, PairwiseSummaryValue] = Field(
        default_factory=dict
    )
    limitations: list[str] = Field(default_factory=list, max_length=8)

    @field_validator("evidence_id")
    @classmethod
    def validate_evidence_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("evidence_id must not be blank.")
        return value

    @field_validator("dataset_id")
    @classmethod
    def normalize_dataset_id(cls, value: str | None) -> str | None:
        return None if value is None else _normalize_dataset_id(value)

    @field_validator("dataset_fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str | None) -> str | None:
        return _validate_fingerprint(value)

    @field_validator("column_ids")
    @classmethod
    def validate_column_ids(cls, value: list[str]) -> list[str]:
        if len(set(value)) != 2:
            raise ValueError("Pairwise evidence requires two distinct columns.")
        if any(_COLUMN_ID_PATTERN.fullmatch(item) is None for item in value):
            raise ValueError("Invalid positional column ID.")
        positions = [int(item[4:]) for item in value]
        if positions != sorted(positions):
            raise ValueError("Pairwise columns must be in positional order.")
        return value

    @field_validator("measured_summary")
    @classmethod
    def sort_summary(
        cls,
        value: dict[str, PairwiseSummaryValue],
    ) -> dict[str, PairwiseSummaryValue]:
        return dict(sorted(value.items()))

    @field_validator("limitations")
    @classmethod
    def validate_limitations(cls, value: list[str]) -> list[str]:
        if any(
            not limitation.strip() or len(limitation) > 500
            for limitation in value
        ):
            raise ValueError(
                "Limitations must be non-blank and at most 500 characters."
            )
        return value

    @model_validator(mode="after")
    def validate_measurement(self) -> "PairwiseEvidence":
        if (self.measured_value is None) == (not self.measured_summary):
            raise ValueError(
                "Provide exactly one of measured_value or measured_summary."
            )
        expected_summary_keys = {
            "exact_column_equality": set(),
            "value_overlap": {
                "left_non_null_count",
                "left_unique_count",
                "right_unique_count",
                "right_non_null_count",
                "shared_unique_count",
                "left_values_in_right",
                "right_values_in_left",
                "complete_case_rows",
            },
            "functional_dependency": {
                "left_to_right_determinant_count",
                "left_to_right_violation_count",
                "right_to_left_determinant_count",
                "right_to_left_violation_count",
            },
        }[self.observation_type]
        if self.observation_type == "exact_column_equality":
            if type(self.measured_value) is not bool:
                raise ValueError(
                    "Exact-column equality must have a boolean result."
                )
            if self.sample_size != self.denominator:
                raise ValueError(
                    "Exact-column equality must compare the declared sample."
                )
        else:
            if self.measured_value is not None:
                raise ValueError(
                    "This pairwise observation requires a bounded summary."
                )
            if set(self.measured_summary) != expected_summary_keys:
                raise ValueError(
                    "Pairwise summary fields do not match observation type."
                )
            boolean_keys = {
                "left_values_in_right",
                "right_values_in_left",
            }
            for key, value in self.measured_summary.items():
                if key in boolean_keys:
                    if type(value) is not bool:
                        raise ValueError(
                            "Value inclusion summaries must be booleans."
                        )
                elif type(value) is not int or value < 0:
                    raise ValueError(
                        "Pairwise count summaries must be non-negative integers."
                    )
            if self.observation_type == "value_overlap":
                summary = self.measured_summary
                left_unique = summary["left_unique_count"]
                right_unique = summary["right_unique_count"]
                shared = summary["shared_unique_count"]
                complete_rows = summary["complete_case_rows"]
                left_non_null = summary["left_non_null_count"]
                right_non_null = summary["right_non_null_count"]
                if (
                    shared > min(left_unique, right_unique)
                    or left_unique > left_non_null
                    or right_unique > right_non_null
                    or complete_rows > min(left_non_null, right_non_null)
                    or complete_rows != self.sample_size
                    or left_non_null > self.denominator
                    or right_non_null > self.denominator
                    or summary["left_values_in_right"]
                    != (shared == left_unique)
                    or summary["right_values_in_left"]
                    != (shared == right_unique)
                ):
                    raise ValueError(
                        "Value-overlap summary counts are inconsistent."
                    )
            elif self.observation_type == "functional_dependency":
                summary = self.measured_summary
                for direction in ("left_to_right", "right_to_left"):
                    determinant_count = summary[
                        f"{direction}_determinant_count"
                    ]
                    violation_count = summary[
                        f"{direction}_violation_count"
                    ]
                    if (
                        determinant_count > self.sample_size
                        or violation_count > determinant_count
                    ):
                        raise ValueError(
                            "Functional-dependency counts are inconsistent."
                        )
        if self.sample_size > self.denominator:
            raise ValueError("sample_size cannot exceed denominator.")
        if (
            self.scope.rows_analyzed is None
            or self.denominator != self.scope.rows_analyzed
        ):
            raise ValueError(
                "Pairwise denominator must match rows_analyzed."
            )
        if (
            self.scope.rows_analyzed is not None
            and self.sample_size > self.scope.rows_analyzed
        ):
            raise ValueError("sample_size cannot exceed rows_analyzed.")
        if (
            self.scope.rows_analyzed is not None
            and self.denominator > self.scope.rows_analyzed
        ):
            raise ValueError("denominator cannot exceed rows_analyzed.")
        return self


class Hypothesis(ContractModel):
    """A candidate interpretation with explicit provenance and uncertainty."""

    hypothesis_id: EntityId
    category: Literal[
        "semantic_type",
        "field_role",
        "unit",
        "value_interpretation",
    ]
    proposed_value: Identifier
    status: Literal["candidate", "ambiguous", "unsupported"]
    confidence: ConfidenceEstimate | None = None
    supporting_evidence_ids: list[EntityId] = Field(default_factory=list)
    contradicting_evidence_ids: list[EntityId] = Field(default_factory=list)
    alternative_hypothesis_ids: list[EntityId] = Field(default_factory=list)
    uncertainty_reasons: list[str] = Field(default_factory=list)
    method_id: Identifier | None = None
    method_version: EntityId | None = None
    analysis_scope: AnalysisScope | None = None
    sample_size: int | None = Field(default=None, ge=0)
    limitations: list[str] = Field(default_factory=list, max_length=8)

    @field_validator(
        "supporting_evidence_ids",
        "contradicting_evidence_ids",
        "alternative_hypothesis_ids",
    )
    @classmethod
    def validate_unique_references(
        cls,
        value: list[str],
    ) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("Reference IDs must be unique within a list.")
        return value

    @field_validator("uncertainty_reasons")
    @classmethod
    def validate_uncertainty_reasons(
        cls,
        value: list[str],
    ) -> list[str]:
        if any(not reason.strip() for reason in value):
            raise ValueError("Uncertainty reasons must not be blank.")
        return value

    @field_validator("limitations")
    @classmethod
    def validate_limitations(cls, value: list[str]) -> list[str]:
        if any(
            not limitation.strip() or len(limitation) > 500
            for limitation in value
        ):
            raise ValueError(
                "Limitations must be non-blank and at most 500 characters."
            )
        return value

    @model_validator(mode="after")
    def require_evidence(self) -> "Hypothesis":
        if not (
            self.supporting_evidence_ids
            or self.contradicting_evidence_ids
        ):
            raise ValueError(
                "A hypothesis requires supporting or contradicting evidence."
            )
        provenance = (
            self.method_id,
            self.method_version,
            self.analysis_scope,
            self.sample_size,
        )
        if self.method_id is not None and any(
            item is None for item in provenance
        ):
            raise ValueError(
                "Method provenance requires version, scope, and sample size."
            )
        return self

class ColumnUnderstanding(ContractModel):
    """Hypotheses and evidence for one position in a dataset schema."""

    column_id: str
    source_name: str
    position: int = Field(ge=0)
    semantic_hypotheses: list[Hypothesis] = Field(default_factory=list)
    role_hypotheses: list[Hypothesis] = Field(default_factory=list)
    unit_hypotheses: list[Hypothesis] = Field(default_factory=list)
    value_interpretation_hypotheses: list[Hypothesis] = Field(
        default_factory=list
    )
    relevant_evidence_ids: list[EntityId] = Field(default_factory=list)

    @field_validator("column_id")
    @classmethod
    def validate_column_id(cls, value: str) -> str:
        if _COLUMN_ID_PATTERN.fullmatch(value) is None:
            raise ValueError("column_id must use the positional col_N form.")
        return value

    @field_validator("relevant_evidence_ids")
    @classmethod
    def validate_unique_evidence_ids(
        cls,
        value: list[str],
    ) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("relevant_evidence_ids must be unique.")
        return value

    @model_validator(mode="after")
    def validate_position_and_categories(self) -> "ColumnUnderstanding":
        if self.column_id != f"col_{self.position}":
            raise ValueError("column_id must match the column position.")
        expected_categories = (
            ("semantic_hypotheses", "semantic_type"),
            ("role_hypotheses", "field_role"),
            ("unit_hypotheses", "unit"),
            ("value_interpretation_hypotheses", "value_interpretation"),
        )
        for field_name, expected_category in expected_categories:
            if any(
                hypothesis.category != expected_category
                for hypothesis in getattr(self, field_name)
            ):
                raise ValueError(
                    f"{field_name} must contain {expected_category} "
                    "hypotheses."
                )
        return self


class RelationshipHypothesis(ContractModel):
    """A candidate relationship across two or more positional columns."""

    relationship_id: EntityId
    relationship_kind: Identifier
    participating_column_ids: list[str] = Field(min_length=2)
    supporting_evidence_ids: list[EntityId] = Field(default_factory=list)
    contradicting_evidence_ids: list[EntityId] = Field(default_factory=list)
    confidence: ConfidenceEstimate | None = None
    uncertainty_reasons: list[str] = Field(default_factory=list)
    method_id: Identifier | None = None
    method_version: EntityId | None = None
    analysis_scope: AnalysisScope | None = None
    sample_size: int | None = Field(default=None, ge=0)
    limitations: list[str] = Field(default_factory=list, max_length=8)

    @field_validator("participating_column_ids")
    @classmethod
    def validate_participating_columns(
        cls,
        value: list[str],
    ) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("Participating column IDs must be unique.")
        if any(_COLUMN_ID_PATTERN.fullmatch(item) is None for item in value):
            raise ValueError("Invalid positional column ID.")
        return value

    @field_validator(
        "supporting_evidence_ids",
        "contradicting_evidence_ids",
    )
    @classmethod
    def validate_unique_evidence_ids(
        cls,
        value: list[str],
    ) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("Evidence references must be unique.")
        return value

    @model_validator(mode="after")
    def require_evidence(self) -> "RelationshipHypothesis":
        if not (
            self.supporting_evidence_ids
            or self.contradicting_evidence_ids
        ):
            raise ValueError(
                "A relationship hypothesis requires evidence."
            )
        provenance = (
            self.method_id,
            self.method_version,
            self.analysis_scope,
            self.sample_size,
        )
        if self.method_id is not None and any(
            item is None for item in provenance
        ):
            raise ValueError(
                "Method provenance requires version, scope, and sample size."
            )
        if (
            self.sample_size is not None
            and self.analysis_scope is not None
            and self.analysis_scope.rows_analyzed is not None
            and self.sample_size > self.analysis_scope.rows_analyzed
        ):
            raise ValueError("sample_size cannot exceed rows_analyzed.")
        if any(
            not limitation.strip() or len(limitation) > 500
            for limitation in self.limitations
        ):
            raise ValueError(
                "Limitations must be non-blank and at most 500 characters."
            )
        return self


class UnderstandingWarning(ContractModel):
    """A limitation or unsupported interpretation surfaced to consumers."""

    warning_code: Identifier
    explanation: str = Field(min_length=1)
    affected_column_ids: list[str] = Field(default_factory=list)
    severity: Literal["info", "low", "medium", "high"]

    @field_validator("affected_column_ids")
    @classmethod
    def validate_affected_columns(
        cls,
        value: list[str],
    ) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("affected_column_ids must be unique.")
        if any(_COLUMN_ID_PATTERN.fullmatch(item) is None for item in value):
            raise ValueError("Invalid positional column ID.")
        return value


class DatasetUnderstandingResult(ContractModel):
    """Versioned understanding output tied to a dataset content fingerprint."""

    dataset_id: str
    dataset_fingerprint: str | None = None
    generated_at: datetime
    understanding_schema_version: EntityId
    engine_version: EntityId
    profile_version: EntityId
    analysis_scope: AnalysisScope
    columns: list[ColumnUnderstanding] = Field(default_factory=list)
    relationships: list[RelationshipHypothesis] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    pairwise_evidence: list[PairwiseEvidence] = Field(default_factory=list)
    warnings: list[UnderstandingWarning] = Field(default_factory=list)

    @field_validator("dataset_id")
    @classmethod
    def normalize_dataset_id(cls, value: str) -> str:
        return _normalize_dataset_id(value)

    @field_validator("dataset_fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str | None) -> str | None:
        return _validate_fingerprint(value)

    @field_validator("generated_at")
    @classmethod
    def require_utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(None):
            raise ValueError("generated_at must be timezone-aware UTC.")
        return value

    @model_validator(mode="after")
    def validate_references(self) -> "DatasetUnderstandingResult":
        column_ids = [column.column_id for column in self.columns]
        if len(column_ids) != len(set(column_ids)):
            raise ValueError("Column IDs must be unique within the result.")
        if any(
            column.column_id != f"col_{position}"
            or column.position != position
            for position, column in enumerate(self.columns)
        ):
            raise ValueError(
                "Columns must be ordered by contiguous zero-based position."
            )
        known_columns = set(column_ids)

        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("Evidence IDs must be unique within the result.")
        pairwise_evidence_ids = [
            item.evidence_id for item in self.pairwise_evidence
        ]
        if len(pairwise_evidence_ids) != len(set(pairwise_evidence_ids)):
            raise ValueError(
                "Pairwise evidence IDs must be unique within the result."
            )
        all_evidence_ids = evidence_ids + pairwise_evidence_ids
        if len(all_evidence_ids) != len(set(all_evidence_ids)):
            raise ValueError(
                "Evidence IDs must be unique across evidence collections."
            )
        pairwise_observations = [
            (tuple(item.column_ids), item.observation_type)
            for item in self.pairwise_evidence
        ]
        if len(pairwise_observations) != len(set(pairwise_observations)):
            raise ValueError(
                "Pairwise observation types must be unique per column pair."
            )
        known_evidence = set(all_evidence_ids)
        evidence_by_id = {item.evidence_id: item for item in self.evidence}
        pairwise_by_id = {
            item.evidence_id: item for item in self.pairwise_evidence
        }

        all_hypotheses = [
            (column.column_id, hypothesis)
            for column in self.columns
            for hypothesis in (
                column.semantic_hypotheses
                + column.role_hypotheses
                + column.unit_hypotheses
                + column.value_interpretation_hypotheses
            )
        ]
        hypothesis_ids = [
            hypothesis.hypothesis_id
            for _, hypothesis in all_hypotheses
        ]
        if len(hypothesis_ids) != len(set(hypothesis_ids)):
            raise ValueError(
                "Hypothesis IDs must be unique within the result."
            )
        hypothesis_by_id = {
            hypothesis.hypothesis_id: (column_id, hypothesis)
            for column_id, hypothesis in all_hypotheses
        }

        relationship_ids = [
            relationship.relationship_id
            for relationship in self.relationships
        ]
        if len(relationship_ids) != len(set(relationship_ids)):
            raise ValueError(
                "Relationship IDs must be unique within the result."
            )

        for column in self.columns:
            self._require_known_ids(
                column.relevant_evidence_ids,
                known_evidence,
                "column evidence",
            )
            for evidence_id in column.relevant_evidence_ids:
                evidence_item = evidence_by_id.get(evidence_id)
                pairwise_item = pairwise_by_id.get(evidence_id)
                related_columns = (
                    set(evidence_item.affected_column_ids)
                    if evidence_item is not None
                    else set(pairwise_item.column_ids)
                    if pairwise_item is not None
                    else set()
                )
                if related_columns and column.column_id not in related_columns:
                    raise ValueError(
                        "Column evidence must refer to the column."
                    )

        for item in self.evidence:
            self._require_known_ids(
                item.affected_column_ids,
                known_columns,
                "evidence column",
            )
            if item.dataset_id is not None and item.dataset_id != self.dataset_id:
                raise ValueError("Evidence dataset_id does not match result.")
            if (
                item.dataset_fingerprint is not None
                and item.dataset_fingerprint != self.dataset_fingerprint
            ):
                raise ValueError(
                    "Evidence fingerprint does not match result."
                )
        for item in self.pairwise_evidence:
            self._require_known_ids(
                item.column_ids,
                known_columns,
                "pairwise evidence column",
            )
            if item.dataset_id is not None and item.dataset_id != self.dataset_id:
                raise ValueError(
                    "Pairwise evidence dataset_id does not match result."
                )
            if (
                item.dataset_fingerprint is not None
                and item.dataset_fingerprint != self.dataset_fingerprint
            ):
                raise ValueError(
                    "Pairwise evidence fingerprint does not match result."
                )
            if (
                self.analysis_scope.total_rows is not None
                and item.scope.total_rows is not None
                and (
                    item.scope.total_rows != self.analysis_scope.total_rows
                )
            ):
                raise ValueError(
                    "Pairwise evidence scope must match the dataset row count."
                )

        for column_id, hypothesis in all_hypotheses:
            self._require_known_ids(
                hypothesis.supporting_evidence_ids,
                known_evidence,
                "supporting evidence",
            )
            self._require_known_ids(
                hypothesis.contradicting_evidence_ids,
                known_evidence,
                "contradicting evidence",
            )
            for evidence_id in (
                hypothesis.supporting_evidence_ids
                + hypothesis.contradicting_evidence_ids
            ):
                evidence_item = evidence_by_id.get(evidence_id)
                pairwise_item = pairwise_by_id.get(evidence_id)
                related_columns = (
                    set(evidence_item.affected_column_ids)
                    if evidence_item is not None
                    else set(pairwise_item.column_ids)
                    if pairwise_item is not None
                    else set()
                )
                if related_columns and column_id not in related_columns:
                    raise ValueError(
                        "Hypothesis evidence must refer to its column."
                    )
            self._require_known_ids(
                hypothesis.alternative_hypothesis_ids,
                set(hypothesis_by_id),
                "alternative hypothesis",
            )
            for alternative_id in hypothesis.alternative_hypothesis_ids:
                alternative_column_id, alternative = hypothesis_by_id[
                    alternative_id
                ]
                if (
                    alternative_id == hypothesis.hypothesis_id
                    or alternative_column_id != column_id
                    or alternative.category != hypothesis.category
                ):
                    raise ValueError(
                        "Alternative hypotheses must be distinct hypotheses "
                        "for the same column and category."
                    )

        for relationship in self.relationships:
            self._require_known_ids(
                relationship.participating_column_ids,
                known_columns,
                "relationship column",
            )
            self._require_known_ids(
                relationship.supporting_evidence_ids,
                known_evidence,
                "relationship supporting evidence",
            )
            self._require_known_ids(
                relationship.contradicting_evidence_ids,
                known_evidence,
                "relationship contradicting evidence",
            )
            participating = set(relationship.participating_column_ids)
            for evidence_id in (
                relationship.supporting_evidence_ids
                + relationship.contradicting_evidence_ids
            ):
                pairwise_item = pairwise_by_id.get(evidence_id)
                evidence_item = evidence_by_id.get(evidence_id)
                related_columns = (
                    set(pairwise_item.column_ids)
                    if pairwise_item is not None
                    else set(evidence_item.affected_column_ids)
                    if evidence_item is not None
                    else set()
                )
                if not participating.issubset(related_columns):
                    raise ValueError(
                        "Relationship evidence must refer to all "
                        "participating columns."
                    )

        for warning in self.warnings:
            self._require_known_ids(
                warning.affected_column_ids,
                known_columns,
                "warning column",
            )

        entity_ids = (
            set(all_evidence_ids)
            | set(hypothesis_ids)
            | set(relationship_ids)
        )
        if len(entity_ids) != (
            len(all_evidence_ids)
            + len(hypothesis_ids)
            + len(relationship_ids)
        ):
            raise ValueError("Entity IDs must be unique within the result.")
        return self

    @staticmethod
    def _require_known_ids(
        referenced_ids: list[str],
        known_ids: set[str],
        reference_kind: str,
    ) -> None:
        missing_ids = set(referenced_ids) - known_ids
        if missing_ids:
            raise ValueError(
                f"Unknown {reference_kind} reference(s): "
                f"{', '.join(sorted(missing_ids))}."
            )
