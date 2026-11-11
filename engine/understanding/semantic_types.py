"""Conservative semantic-type hypotheses derived from validated evidence."""

import hashlib
import json

from engine.understanding.models import (
    AnalysisScope,
    DatasetUnderstandingResult,
    EvidenceItem,
    Hypothesis,
    UnderstandingWarning,
)

_NUMERIC_STATISTICS = {
    "numeric_mean",
    "numeric_median",
    "numeric_standard_deviation",
    "numeric_minimum",
    "numeric_maximum",
}
_METHOD_ID = "semantic_type_hypothesis"
_METHOD_VERSION = "1"


def infer_semantic_types(
    result: DatasetUnderstandingResult,
) -> DatasetUnderstandingResult:
    """Add evidence-backed semantic hypotheses to an understanding result.

    This function consumes only validated profile and quality evidence. It
    does not inspect or transform source values and does not infer roles,
    relationships, units, or business meaning.
    """

    result = DatasetUnderstandingResult.model_validate(
        result.model_dump(mode="python")
    )
    evidence_by_column: dict[str, list[EvidenceItem]] = {
        column.column_id: [] for column in result.columns
    }
    for item in result.evidence:
        for column_id in item.affected_column_ids:
            if column_id not in evidence_by_column:
                raise ValueError(
                    f"Evidence references unknown column {column_id!r}."
                )
            evidence_by_column[column_id].append(item)

    columns = []
    new_warnings: list[UnderstandingWarning] = []
    for column in result.columns:
        evidence = evidence_by_column[column.column_id]
        hypotheses, warnings = _infer_column_hypotheses(
            result.dataset_fingerprint or result.dataset_id,
            column.column_id,
            evidence,
            result.analysis_scope,
        )
        existing_ids = {
            hypothesis.hypothesis_id
            for hypothesis in column.semantic_hypotheses
        }
        semantic_hypotheses = list(column.semantic_hypotheses)
        semantic_hypotheses.extend(
            hypothesis
            for hypothesis in hypotheses
            if hypothesis.hypothesis_id not in existing_ids
        )
        columns.append(
            column.model_copy(
                update={"semantic_hypotheses": semantic_hypotheses}
            )
        )
        new_warnings.extend(warnings)

    existing_warning_keys = {
        (
            warning.warning_code,
            warning.explanation,
            tuple(warning.affected_column_ids),
        )
        for warning in result.warnings
    }
    warnings = list(result.warnings)
    for warning in new_warnings:
        key = (
            warning.warning_code,
            warning.explanation,
            tuple(warning.affected_column_ids),
        )
        if key not in existing_warning_keys:
            warnings.append(warning)
            existing_warning_keys.add(key)
    updated = result.model_dump(mode="python")
    updated["columns"] = columns
    updated["warnings"] = warnings
    return DatasetUnderstandingResult.model_validate(updated)


def _infer_column_hypotheses(
    dataset_identity: str,
    column_id: str,
    evidence: list[EvidenceItem],
    scope: AnalysisScope,
) -> tuple[list[Hypothesis], list[UnderstandingWarning]]:
    by_metric: dict[str, list[EvidenceItem]] = {}
    for item in evidence:
        by_metric.setdefault(item.metric_name, []).append(item)

    def evidence_ids(metric_name: str) -> list[str]:
        return [
            item.evidence_id
            for item in by_metric.get(metric_name, [])
        ]

    def category_evidence(metric_name: str, value: str) -> list[str]:
        return [
            item.evidence_id
            for item in by_metric.get(metric_name, [])
            if item.measured_category == value
        ]

    numeric_ids = category_evidence("type_candidate_observed", "numeric")
    datetime_ids = category_evidence("type_candidate_observed", "datetime")
    boolean_ids = category_evidence("type_candidate_observed", "boolean")
    text_ids = category_evidence("type_candidate_observed", "text")
    scalar_string_ids = [
        item.evidence_id
        for item in by_metric.get("inferred_scalar_type", [])
        if item.measured_category in {"string", "unicode"}
    ]
    numeric_stat_ids = [
        item.evidence_id
        for metric in _NUMERIC_STATISTICS
        for item in by_metric.get(metric, [])
    ]
    classification_ids = {
        item.measured_category: item.evidence_id
        for item in by_metric.get("profile_classification", [])
        if item.measured_category is not None
    }
    ambiguity_ids = [
        item.evidence_id
        for item in by_metric.get("type_ambiguity_observed", [])
        if item.measured_value is True
    ]
    unique_ids = evidence_ids("column_unique_count")
    missing_ids = evidence_ids("column_missing_count")
    top_frequency_ids = evidence_ids("categorical_top_value_count")
    candidate_ids = {
        "numeric": numeric_ids,
        "datetime": datetime_ids,
        "boolean": boolean_ids,
        "text": text_ids,
    }

    specs: list[tuple[str, list[str], list[str], list[str]]] = []
    warnings: list[UnderstandingWarning] = []

    profile_classification_id = classification_ids.get("identifier_like")
    identifier_quality_ids = (
        evidence_ids("quality_identifier_like_column_present")
        + evidence_ids("quality_identifier_like_column_count")
    )
    if profile_classification_id is not None:
        specs.append((
            "identifier",
            [profile_classification_id, *identifier_quality_ids],
            [],
            [
                (
                "The existing classifier calls this identifier-like; "
                    "this is a heuristic candidate, not a confirmed key."
                )
            ],
        ))

    is_ambiguous = bool(ambiguity_ids or (numeric_ids and datetime_ids))
    can_propose_numeric = bool(
        numeric_ids
        and classification_ids.get("general") is not None
        and (numeric_stat_ids or datetime_ids)
    )
    can_propose_datetime = bool(datetime_ids)
    if can_propose_numeric:
        uncertainty = [
            "Numeric statistics do not establish units or business meaning."
        ]
        if len(numeric_ids) and len(datetime_ids):
            uncertainty.append(
                "Existing type inference also reports a datetime candidate."
            )
        if not numeric_stat_ids:
            uncertainty.append(
                "Numeric summary statistics are unavailable for this "
                "ambiguous column."
            )
        if _evidence_sample_size(numeric_stat_ids, evidence) < 3:
            uncertainty.append(
                "Fewer than three non-missing values support this candidate."
            )
        numeric_contradictions = evidence_ids_for_positive_metric(
            evidence,
            "numeric_unparseable_count",
        )
        specs.append((
            "numeric_measurement",
            [
                *numeric_ids,
                *numeric_stat_ids,
                classification_ids["general"],
                *unique_ids,
                *missing_ids,
                *ambiguity_ids,
            ],
            numeric_contradictions,
            uncertainty,
        ))

    if can_propose_datetime:
        uncertainty = [
            (
                "The existing type inference provides a parseability "
                "candidate, not a confirmed date interpretation."
            )
        ]
        if numeric_ids:
            uncertainty.append(
                "The same values also have a numeric interpretation."
            )
        if is_ambiguous:
            uncertainty.append(
                "Existing type inference marks the column ambiguous."
            )
        datetime_contradictions = evidence_ids_for_positive_metric(
            evidence,
            "datetime_unparseable_count",
        )
        specs.append((
            "datetime",
            [*datetime_ids, *ambiguity_ids],
            datetime_contradictions,
            uncertainty,
        ))

    if boolean_ids:
        specs.append((
            "boolean",
            boolean_ids,
            [],
            [
                (
                "Boolean is the observed scalar candidate; it does not "
                    "establish a business-specific meaning."
                )
            ],
        ))

    if (
        text_ids
        and scalar_string_ids
        and top_frequency_ids
        and classification_ids.get("general") is not None
        and not numeric_ids
        and not datetime_ids
    ):
        specs.append((
            "categorical",
            [
                *text_ids,
                *scalar_string_ids,
                *top_frequency_ids,
                classification_ids["general"],
            ],
            [],
            [
                (
                    "String values and frequency summaries are compatible "
                    "with categories but do not distinguish categories from "
                    "free text."
                ),
                "No raw category values were inspected or included.",
            ],
        ))

    hypothesis_ids = {
        value: _hypothesis_id(dataset_identity, column_id, value)
        for value, _, _, _ in specs
    }
    hypotheses = []
    for value, supporting_ids, contradicting_ids, uncertainty in specs:
        alternatives = []
        if value in {"numeric_measurement", "datetime"} and is_ambiguous:
            alternatives = [
                hypothesis_id
                for other_value, hypothesis_id in hypothesis_ids.items()
                if other_value in {"numeric_measurement", "datetime"}
                and other_value != value
            ]
        hypotheses.append(
            Hypothesis(
                hypothesis_id=hypothesis_ids[value],
                category="semantic_type",
                proposed_value=value,
                status=(
                    "ambiguous"
                    if value in {"numeric_measurement", "datetime"}
                    and is_ambiguous
                    else "candidate"
                ),
                confidence=None,
                supporting_evidence_ids=_unique(supporting_ids),
                contradicting_evidence_ids=_unique(contradicting_ids),
                alternative_hypothesis_ids=alternatives,
                uncertainty_reasons=uncertainty,
                method_id=(
                    _METHOD_ID
                    if scope.rows_analyzed is not None
                    else None
                ),
                method_version=(
                    _METHOD_VERSION
                    if scope.rows_analyzed is not None
                    else None
                ),
                analysis_scope=(
                    scope if scope.rows_analyzed is not None else None
                ),
                sample_size=scope.rows_analyzed,
            )
        )

    if not hypotheses:
        reason = _no_hypothesis_reason(
            evidence,
            classification_ids,
            candidate_ids,
        )
        if reason is not None:
            warnings.append(
                UnderstandingWarning(
                    warning_code="insufficient_semantic_evidence",
                    explanation=reason,
                    affected_column_ids=[column_id],
                    severity="info",
                )
            )
    return hypotheses, warnings


def _no_hypothesis_reason(
    evidence: list[EvidenceItem],
    classifications: dict[str, str],
    candidates: dict[str, list[str]],
) -> str | None:
    if classifications.get("empty") is not None:
        return "The column has no non-missing values; no semantic type is proposed."
    if candidates.get("complex"):
        return (
            "Complex values are observed, but the current profile does not "
            "support a semantic interpretation."
        )
    if (
        evidence_ids_for_metric(evidence, "quality_mixed_types_present")
        or evidence_ids_for_metric(evidence, "quality_mixed_types_count")
    ):
        return (
            "Mixed scalar types are observed without enough evidence for one "
            "semantic type."
        )
    if (
        classifications.get("constant") is not None
        or classifications.get("near_constant") is not None
    ):
        return (
            "Constant or near-constant values do not provide enough "
            "variation for a semantic type hypothesis."
        )
    if candidates.get("text") and (
        candidates.get("numeric") or candidates.get("datetime")
    ):
        return (
            "Text values have competing parseability candidates; no semantic "
            "type is selected."
        )
    return None


def _evidence_sample_size(
    evidence_ids: list[str],
    evidence: list[EvidenceItem],
) -> int:
    id_set = set(evidence_ids)
    return max(
        (
            item.sample_size or 0
            for item in evidence
            if item.evidence_id in id_set
        ),
        default=0,
    )


def evidence_ids_for_metric(
    evidence: list[EvidenceItem],
    metric_name: str,
) -> list[str]:
    return [
        item.evidence_id for item in evidence
        if item.metric_name == metric_name
    ]


def evidence_ids_for_positive_metric(
    evidence: list[EvidenceItem],
    metric_name: str,
) -> list[str]:
    return [
        item.evidence_id
        for item in evidence
        if item.metric_name == metric_name
        and isinstance(item.measured_value, (int, float))
        and not isinstance(item.measured_value, bool)
        and item.measured_value > 0
    ]


def _hypothesis_id(
    dataset_identity: str,
    column_id: str,
    semantic_type: str,
) -> str:
    payload = json.dumps(
        [dataset_identity, column_id, semantic_type],
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:40]
    return f"sem_{digest}"


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
