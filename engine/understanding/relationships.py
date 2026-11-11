"""Relationship discovery bounded by currently available evidence."""

import hashlib
import json

import pandas as pd

from engine.understanding.models import (
    DatasetUnderstandingResult,
    PairwiseEvidence,
    RelationshipHypothesis,
    UnderstandingWarning,
)
from engine.understanding.pairwise import (
    DEFAULT_PAIRWISE_EVIDENCE_LIMITS,
    PairwiseEvidenceLimits,
    build_pairwise_evidence,
)


def discover_relationships_from_dataframe(
    result: DatasetUnderstandingResult,
    dataframe: pd.DataFrame,
    *,
    limits: PairwiseEvidenceLimits = DEFAULT_PAIRWISE_EVIDENCE_LIMITS,
) -> DatasetUnderstandingResult:
    """Build bounded evidence from an already-loaded frame, then discover.

    The caller owns data loading and must pass the same positional frame used
    to produce the understanding result. This function never reads files.
    """

    result = DatasetUnderstandingResult.model_validate(
        result.model_dump(mode="python")
    )
    adapted = build_pairwise_evidence(
        dataframe,
        result,
        limits=limits,
    )
    evidence_by_id = {
        item.evidence_id: item
        for item in result.pairwise_evidence
    }
    for item in adapted.pairwise_evidence:
        existing = evidence_by_id.get(item.evidence_id)
        if existing is not None and existing != item:
            raise ValueError("Pairwise evidence ID collision.")
        evidence_by_id[item.evidence_id] = item
    updated = result.model_dump(mode="python")
    updated["pairwise_evidence"] = list(evidence_by_id.values())
    updated["warnings"] = _unique_warnings([
        *result.warnings,
        *adapted.warnings,
    ])
    validated = DatasetUnderstandingResult.model_validate(updated)
    return discover_relationships(validated)


def discover_relationships(
    result: DatasetUnderstandingResult,
) -> DatasetUnderstandingResult:
    """Infer conservative structural relationships from pairwise evidence.

    This function consumes only validated pairwise observations already
    attached to the result. It never rereads source data or treats statistical
    association as causation.
    """

    result = DatasetUnderstandingResult.model_validate(
        result.model_dump(mode="python")
    )
    warnings = list(result.warnings)
    relationships = list(result.relationships)
    possible_pairs = len(result.columns) * (len(result.columns) - 1) // 2
    if not result.pairwise_evidence and possible_pairs:
        warnings.append(
            UnderstandingWarning(
                warning_code="pairwise_relationship_evidence_unavailable",
                explanation=(
                    "No pairwise observations are present, so no relationship "
                    "is inferred from column names or marginal profile "
                    "statistics. Correlation, if measured in future, would "
                    "indicate association rather than causation."
                ),
                affected_column_ids=[
                    column.column_id for column in result.columns
                ],
                severity="info",
            )
        )
    elif result.pairwise_evidence and not any(
        item.scope.kind == "full_dataset"
        for item in result.pairwise_evidence
    ):
        warnings.append(
            UnderstandingWarning(
                warning_code="pairwise_sample_relationships_withheld",
                explanation=(
                    "Pairwise evidence covers only a sample; structural "
                    "relationships are not promoted to full-dataset findings."
                ),
                affected_column_ids=[
                    column.column_id for column in result.columns
                ],
                severity="info",
            )
        )
    if (
        possible_pairs
        > DEFAULT_PAIRWISE_EVIDENCE_LIMITS.max_column_pairs
        and not any(
            warning.warning_code == "pairwise_comparison_limit_exceeded"
            for warning in warnings
        )
    ):
        warnings.append(
            UnderstandingWarning(
                warning_code="pairwise_comparison_limit_exceeded",
                explanation=(
                    f"The dataset has {possible_pairs} possible column pairs, "
                    "which exceeds the default pair-comparison limit."
                ),
                affected_column_ids=[
                    column.column_id for column in result.columns
                ],
                severity="info",
            )
        )
    evidence_by_pair_and_type = {
        (tuple(item.column_ids), item.observation_type): item
        for item in result.pairwise_evidence
    }
    identity = result.dataset_fingerprint or result.dataset_id
    generated: dict[str, RelationshipHypothesis] = {}
    for (column_ids, observation_type), item in evidence_by_pair_and_type.items():
        if item.scope.kind != "full_dataset":
            continue
        if observation_type == "exact_column_equality":
            overlap = evidence_by_pair_and_type.get(
                (column_ids, "value_overlap")
            )
            if (
                item.measured_value is True
                and overlap is not None
                and overlap.measured_summary["left_unique_count"] > 1
                and overlap.measured_summary["right_unique_count"] > 1
                and item.sample_size >= 3
                and item.scope.total_rows is not None
                and item.scope.total_rows >= 3
            ):
                relationship = _relationship(
                    identity,
                    kind="exact_duplicate_columns",
                    column_ids=list(column_ids),
                    evidence=[item, overlap],
                    uncertainty=[
                        (
                            "Exact equality describes observed values, not "
                            "whether either field is redundant for downstream "
                            "use."
                        )
                    ],
                    prefix="rel",
                )
                generated[relationship.relationship_id] = relationship
        elif observation_type == "value_overlap":
            if item.sample_size < 10:
                continue
            left_unique = item.measured_summary["left_unique_count"]
            right_unique = item.measured_summary["right_unique_count"]
            left_non_null = item.measured_summary["left_non_null_count"]
            right_non_null = item.measured_summary["right_non_null_count"]
            left_in_right = item.measured_summary["left_values_in_right"]
            right_in_left = item.measured_summary["right_values_in_left"]
            if (
                left_in_right is True
                and left_unique > 1
                and right_unique == right_non_null == item.denominator
                and right_unique > 1
                and not (right_in_left is True and left_unique == right_unique)
            ):
                relationship = _relationship(
                    identity,
                    kind="candidate_key_inclusion",
                    column_ids=[column_ids[0], column_ids[1]],
                    evidence=[item],
                    uncertainty=[
                        (
                            "Complete-value inclusion and uniqueness do not "
                            "establish a declared foreign-key constraint or "
                            "shared entity meaning."
                        )
                    ],
                    prefix="rel",
                )
                generated[relationship.relationship_id] = relationship
            if (
                right_in_left is True
                and right_unique > 1
                and left_unique == left_non_null == item.denominator
                and left_unique > 1
                and not (left_in_right is True and left_unique == right_unique)
            ):
                relationship = _relationship(
                    identity,
                    kind="candidate_key_inclusion",
                    column_ids=[column_ids[1], column_ids[0]],
                    evidence=[item],
                    uncertainty=[
                        (
                            "Complete-value inclusion and uniqueness do not "
                            "establish a declared foreign-key constraint or "
                            "shared entity meaning."
                        )
                    ],
                    prefix="rel",
                )
                generated[relationship.relationship_id] = relationship
        elif observation_type == "functional_dependency":
            if item.sample_size < 10:
                continue
            for determinant, dependent, direction in (
                (column_ids[0], column_ids[1], "left_to_right"),
                (column_ids[1], column_ids[0], "right_to_left"),
            ):
                determinant_count = item.measured_summary[
                    f"{direction}_determinant_count"
                ]
                dependent_count = item.measured_summary[
                    (
                        "right_to_left_determinant_count"
                        if direction == "left_to_right"
                        else "left_to_right_determinant_count"
                    )
                ]
                violation_count = item.measured_summary[
                    f"{direction}_violation_count"
                ]
                if (
                    violation_count == 0
                    and 1 < determinant_count < item.sample_size
                    and 1 < dependent_count < item.sample_size
                ):
                    relationship = _relationship(
                        identity,
                        kind="candidate_functional_dependency",
                        column_ids=[determinant, dependent],
                        evidence=[item],
                        uncertainty=[
                            (
                                "Dependency is based on complete-case rows "
                                "and does not establish a causal or business "
                                "rule."
                            )
                        ],
                        prefix="rel",
                    )
                    generated[relationship.relationship_id] = relationship

    existing_ids = {item.relationship_id for item in relationships}
    relationships.extend(
        relationship
        for relationship_id, relationship in generated.items()
        if relationship_id not in existing_ids
    )
    updated = result.model_dump(mode="python")
    updated["relationships"] = relationships
    updated["warnings"] = _unique_warnings(warnings)
    return DatasetUnderstandingResult.model_validate(updated)


def _relationship(
    identity: str,
    *,
    kind: str,
    column_ids: list[str],
    evidence: list[PairwiseEvidence],
    uncertainty: list[str],
    prefix: str,
) -> RelationshipHypothesis:
    payload = json.dumps(
        [identity, kind, column_ids],
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:40]
    return RelationshipHypothesis(
        relationship_id=f"{prefix}_{digest}",
        relationship_kind=kind,
        participating_column_ids=column_ids,
        supporting_evidence_ids=[item.evidence_id for item in evidence],
        confidence=None,
        uncertainty_reasons=uncertainty,
        method_id="structural_relationship_discovery",
        method_version="1",
        analysis_scope=evidence[0].scope,
        sample_size=evidence[0].sample_size,
        limitations=list(dict.fromkeys(
            limitation
            for item in evidence
            for limitation in item.limitations
        ))[:8],
    )


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
