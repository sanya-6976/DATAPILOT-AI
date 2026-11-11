"""Conservative value and unit hypotheses derived from existing evidence."""

import hashlib
import json

from engine.understanding.models import (
    DatasetUnderstandingResult,
    EvidenceItem,
    Hypothesis,
    UnderstandingWarning,
)

_METHOD_ID = "evidence_value_interpretation"
_METHOD_VERSION = "1"
_NUMERIC_TYPES = {"numeric_measurement"}
_TEMPORAL_TYPES = {"datetime"}


def infer_value_interpretations(
    result: DatasetUnderstandingResult,
) -> DatasetUnderstandingResult:
    """Add interpretations only when existing evidence supports the candidate.

    The inference uses validated evidence and semantic hypotheses only. It
    does not read cells, parse strings, infer units from names, or convert data.
    """

    result = DatasetUnderstandingResult.model_validate(
        result.model_dump(mode="python")
    )
    evidence_by_column: dict[str, list[EvidenceItem]] = {
        column.column_id: []
        for column in result.columns
    }
    for item in result.evidence:
        for column_id in item.affected_column_ids:
            evidence_by_column[column_id].append(item)

    columns = []
    generated_warnings: list[UnderstandingWarning] = []
    for column in result.columns:
        evidence = evidence_by_column[column.column_id]
        evidence_by_metric: dict[str, list[EvidenceItem]] = {}
        for item in evidence:
            evidence_by_metric.setdefault(item.metric_name, []).append(item)

        semantic = {
            hypothesis.proposed_value: hypothesis
            for hypothesis in column.semantic_hypotheses
        }
        specs: list[
            tuple[str, str, list[str], list[str], list[str], list[str], str]
        ] = []
        warning_codes: set[str] = set()

        boolean_hypothesis = semantic.get("boolean")
        if boolean_hypothesis is not None:
            specs.append((
                "value_interpretation",
                "boolean_values",
                boolean_hypothesis.supporting_evidence_ids,
                boolean_hypothesis.contradicting_evidence_ids,
                [
                    (
                        "The existing type inference supports Boolean values, "
                        "but does not establish a domain-specific encoding."
                    )
                ],
                [
                    "No values or encodings are transformed.",
                    "Text/numeric Boolean-like codes are not inferred.",
                ],
                "candidate",
            ))

        scalar_type = _single_category(
            evidence_by_metric.get("inferred_scalar_type", [])
        )
        scalar_type_ids = [
            item.evidence_id
            for item in evidence_by_metric.get("inferred_scalar_type", [])
        ]
        numeric_candidate_ids = _category_ids(
            evidence_by_metric.get("type_candidate_observed", []),
            "numeric",
        )
        text_candidate_ids = _category_ids(
            evidence_by_metric.get("type_candidate_observed", []),
            "text",
        )
        if (
            scalar_type == "string"
            and numeric_candidate_ids
            and text_candidate_ids
        ):
            specs.append((
                "value_interpretation",
                "numeric_string_values",
                [
                    *scalar_type_ids,
                    *numeric_candidate_ids,
                    *text_candidate_ids,
                ],
                _positive_metric_ids(
                    evidence_by_metric.get("numeric_unparseable_count", [])
                ),
                [
                    (
                        "Numeric parsing is a candidate for string values, "
                        "not a guarantee that every value has numeric meaning."
                    )
                ],
                [
                    "No string values are exposed or converted.",
                    "The result does not specify locale, separators, or units.",
                ],
                "ambiguous",
            ))

        if scalar_type in {"mixed", "mixed_integer"}:
            mixed_ids = [
                item.evidence_id
                for item in evidence_by_metric.get("inferred_scalar_type", [])
            ]
            mixed_quality_ids = [
                item.evidence_id
                for metric in (
                    "quality_mixed_types_present",
                    "quality_mixed_types_count",
                )
                for item in evidence_by_metric.get(metric, [])
            ]
            specs.append((
                "value_interpretation",
                "mixed_scalar_values",
                [*mixed_ids, *mixed_quality_ids],
                [],
                [
                    (
                        "Heterogeneous scalar types may be intentional and "
                        "do not by themselves establish malformed data."
                    )
                ],
                ["Original values are not normalized or changed."],
                "ambiguous",
            ))

        if semantic.keys() & _NUMERIC_TYPES:
            minimum = _single_numeric(
                evidence_by_metric.get("numeric_minimum", [])
            )
            maximum = _single_numeric(
                evidence_by_metric.get("numeric_maximum", [])
            )
            if minimum is not None and maximum is not None:
                integer_type_ids = [
                    item.evidence_id
                    for item in evidence_by_metric.get(
                        "inferred_scalar_type",
                        [],
                    )
                    if item.measured_category == "integer"
                ]
                numeric_ids = _category_ids(
                    evidence_by_metric.get("type_candidate_observed", []),
                    "numeric",
                )
                if (
                    minimum >= 0
                    and integer_type_ids
                    and numeric_ids
                    and "identifier" not in semantic
                ):
                    specs.append((
                        "value_interpretation",
                        "nonnegative_integer_count",
                        [
                            *integer_type_ids,
                            *numeric_ids,
                            *[
                                item.evidence_id
                                for item in evidence_by_metric.get(
                                    "numeric_minimum",
                                    [],
                                )
                            ],
                            *[
                                item.evidence_id
                                for item in evidence_by_metric.get(
                                    "numeric_maximum",
                                    [],
                                )
                            ],
                        ],
                        [],
                        [
                            (
                                "Non-negative integer-valued data may still "
                                "be a code, index, year, or other encoded "
                                "value."
                            )
                        ],
                        [
                            "This is not a confirmed count or measurement.",
                            "No values or units are transformed.",
                        ],
                        "ambiguous",
                    ))
                # Range evidence is retained as context, never treated as proof
                # of a unit or scale convention.
                if 0 <= minimum <= maximum <= 1:
                    warning_codes.add("proportion_percentage_ambiguous")
                elif 0 <= minimum <= maximum <= 100:
                    warning_codes.add("bounded_numeric_scale_ambiguous")
                warning_codes.add("measurement_unit_unavailable")
            else:
                warning_codes.add("numeric_range_incomplete")
        if "categorical" in semantic:
            warning_codes.add("ordinal_interpretation_unavailable")

        if semantic.keys() & _TEMPORAL_TYPES:
            specs.append((
                "value_interpretation",
                "date_or_timestamp",
                semantic["datetime"].supporting_evidence_ids,
                semantic["datetime"].contradicting_evidence_ids,
                [
                    (
                        "Existing datetime evidence does not distinguish "
                        "calendar dates, timestamps, or other temporal "
                        "encodings."
                    )
                ],
                [
                    "No additional parsing is performed.",
                    (
                        "Duration, time-of-day, and temporal-interval "
                        "meanings remain unresolved."
                    ),
                ],
                "ambiguous",
            ))

        hypotheses = list(column.value_interpretation_hypotheses)
        existing_ids = {
            hypothesis.hypothesis_id for hypothesis in hypotheses
        }
        for (
            category,
            value,
            supporting,
            contradicting,
            uncertainty,
            limitations,
            status,
        ) in specs:
            if not supporting:
                continue
            hypothesis_id = _hypothesis_id(
                result.dataset_fingerprint or result.dataset_id,
                column.column_id,
                category,
                value,
            )
            if hypothesis_id in existing_ids:
                continue
            source_evidence = [
                evidence_by_id
                for evidence_by_id in evidence
                if evidence_by_id.evidence_id in supporting
            ]
            hypotheses.append(
                Hypothesis(
                    hypothesis_id=hypothesis_id,
                    category=category,
                    proposed_value=value,
                    status=status,
                    confidence=None,
                    supporting_evidence_ids=_unique(supporting),
                    contradicting_evidence_ids=_unique(contradicting),
                    uncertainty_reasons=uncertainty,
                    method_id=_METHOD_ID,
                    method_version=_METHOD_VERSION,
                    analysis_scope=result.analysis_scope,
                    sample_size=max(
                        (
                            item.sample_size
                            for item in source_evidence
                            if item.sample_size is not None
                        ),
                        default=None,
                    ),
                    limitations=limitations,
                )
            )
            existing_ids.add(hypothesis_id)

        columns.append(
            column.model_copy(
                update={"value_interpretation_hypotheses": hypotheses}
            )
        )
        if warning_codes:
            generated_warnings.extend(
                _warnings_for_column(column.column_id, warning_codes)
            )
        if (
            semantic.keys() & _NUMERIC_TYPES
            and scalar_type in {"integer", "floating", "decimal"}
        ):
            generated_warnings.append(
                UnderstandingWarning(
                    warning_code="numeric_scale_not_interpreted",
                    explanation=(
                        "Existing statistics do not distinguish counts, "
                        "continuous measurements, proportions, percentages, "
                        "scores, or encoded values."
                    ),
                    affected_column_ids=[column.column_id],
                    severity="info",
                )
            )
        if semantic.keys() & _TEMPORAL_TYPES:
            generated_warnings.append(
                UnderstandingWarning(
                    warning_code="temporal_granularity_unresolved",
                    explanation=(
                        "Existing datetime evidence does not distinguish "
                        "date, timestamp, time-of-day, duration, or interval."
                    ),
                    affected_column_ids=[column.column_id],
                    severity="info",
                )
            )
        if any(
            item.metric_name == "column_missing_count"
            and isinstance(item.measured_value, (int, float))
            and item.measured_value > 0
            for item in evidence
        ):
            generated_warnings.append(
                UnderstandingWarning(
                    warning_code="missing_sentinel_unavailable",
                    explanation=(
                        "Available missing counts do not identify possible "
                        "sentinel values. No source values were inspected or "
                        "reclassified."
                    ),
                    affected_column_ids=[column.column_id],
                    severity="info",
                )
            )

    updated = result.model_dump(mode="python")
    updated["columns"] = columns
    updated["warnings"] = _merge_warnings(
        result.warnings,
        generated_warnings,
    )
    return DatasetUnderstandingResult.model_validate(updated)


def _single_category(evidence: list[EvidenceItem]) -> str | None:
    categories = {
        item.measured_category
        for item in evidence
        if item.measured_category is not None
    }
    return next(iter(categories)) if len(categories) == 1 else None


def _category_ids(
    evidence: list[EvidenceItem],
    category: str,
) -> list[str]:
    return [
        item.evidence_id
        for item in evidence
        if item.measured_category == category
    ]


def _single_numeric(
    evidence: list[EvidenceItem],
) -> float | int | None:
    values = [
        item.measured_value
        for item in evidence
        if isinstance(item.measured_value, (int, float))
        and not isinstance(item.measured_value, bool)
    ]
    return values[0] if len(values) == 1 else None


def _positive_metric_ids(evidence: list[EvidenceItem]) -> list[str]:
    return [
        item.evidence_id
        for item in evidence
        if isinstance(item.measured_value, (int, float))
        and not isinstance(item.measured_value, bool)
        and item.measured_value > 0
    ]


def _warnings_for_column(
    column_id: str,
    warning_codes: set[str],
) -> list[UnderstandingWarning]:
    messages = {
        "proportion_percentage_ambiguous": (
            "A range within zero and one alone cannot distinguish proportions "
            "from percentages or other bounded measurements."
        ),
        "bounded_numeric_scale_ambiguous": (
            "A range within zero and one hundred does not establish a score, "
            "rating, percentage, or measurement scale."
        ),
        "measurement_unit_unavailable": (
            "Currency and physical units are not inferred without explicit "
            "unit evidence."
        ),
        "numeric_range_incomplete": (
            "Numeric range evidence is incomplete; scale interpretations "
            "remain unavailable."
        ),
        "ordinal_interpretation_unavailable": (
            "Categorical evidence does not establish an ordinal ranking; "
            "category ordering is not inferred."
        ),
    }
    return [
        UnderstandingWarning(
            warning_code=code,
            explanation=messages[code],
            affected_column_ids=[column_id],
            severity="info",
        )
        for code in sorted(warning_codes)
    ]


def _merge_warnings(
    existing: list[UnderstandingWarning],
    generated: list[UnderstandingWarning],
) -> list[UnderstandingWarning]:
    merged: list[UnderstandingWarning] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    for warning in [*existing, *generated]:
        key = (
            warning.warning_code,
            warning.explanation,
            tuple(warning.affected_column_ids),
        )
        if key not in seen:
            merged.append(warning)
            seen.add(key)
    return merged


def _hypothesis_id(
    dataset_identity: str,
    column_id: str,
    category: str,
    interpretation: str,
) -> str:
    payload = json.dumps(
        [dataset_identity, column_id, category, interpretation],
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:40]
    return f"val_{digest}"


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
