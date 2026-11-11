"""Scoring helpers for explicitly labeled synthetic understanding cases."""

from collections.abc import Hashable
from dataclasses import dataclass
from typing import TypeVar

Label = TypeVar("Label", bound=Hashable)


@dataclass(frozen=True)
class EvaluationMetrics:
    true_positive_count: int
    false_positive_count: int
    false_negative_count: int
    unsupported_hypothesis_count: int
    precision: float | None
    recall: float | None
    f1: float | None


def score_labeled_outputs(
    predicted: set[Label],
    expected_positive: set[Label],
    explicitly_negative: set[Label],
    intentionally_withheld: set[Label] | None = None,
) -> EvaluationMetrics:
    """Score only labels with an explicit positive or negative annotation.

    Intentionally withheld labels are reported separately and excluded from
    false negatives. Unlabeled outputs are not scored.
    """

    withheld = intentionally_withheld or set()
    if expected_positive & explicitly_negative:
        raise ValueError("Positive and negative evaluation labels overlap.")
    if expected_positive & withheld or explicitly_negative & withheld:
        raise ValueError("Evaluation label sets must be disjoint.")

    true_positive = len(predicted & expected_positive)
    false_positive = len(predicted & explicitly_negative)
    false_negative = len(expected_positive - predicted)
    unsupported = len(predicted & withheld)
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else None
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else None
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None
        and recall is not None
        and precision + recall
        else None
    )
    return EvaluationMetrics(
        true_positive_count=true_positive,
        false_positive_count=false_positive,
        false_negative_count=false_negative,
        unsupported_hypothesis_count=unsupported,
        precision=precision,
        recall=recall,
        f1=f1,
    )
