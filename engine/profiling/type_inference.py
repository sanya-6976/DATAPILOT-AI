import re
import uuid
from typing import Any

import pandas as pd
from pandas.api.types import infer_dtype

NEAR_CONSTANT_MIN_VALUES = 20
NEAR_CONSTANT_THRESHOLD = 0.95
_PREFIXED_IDENTIFIER_PATTERN = re.compile(r"^(.*?)[-_]?(\d+)$")


def classify_series(
    series: pd.Series,
    column_name: str | None = None,
) -> dict[str, Any]:
    """Classify empty, low-variation, and structurally ID-like columns."""

    non_missing = series.dropna()
    value_count = len(non_missing)
    unique_count = int(non_missing.nunique())

    if value_count == 0:
        return {
            "column_classification": "empty",
            "classification_details": {
                "non_missing_values": 0,
                "unique_values": 0,
            },
        }

    if unique_count == 1:
        return {
            "column_classification": "constant",
            "classification_details": {
                "non_missing_values": value_count,
                "unique_values": unique_count,
                "dominant_value_percentage": 100.0,
            },
        }

    value_counts = non_missing.value_counts(dropna=True)
    dominant_percentage = float(value_counts.iloc[0] / value_count * 100)
    if (
        value_count >= NEAR_CONSTANT_MIN_VALUES
        and dominant_percentage >= NEAR_CONSTANT_THRESHOLD * 100
    ):
        return {
            "column_classification": "near_constant",
            "classification_details": {
                "non_missing_values": value_count,
                "unique_values": unique_count,
                "dominant_value_percentage": round(
                    dominant_percentage,
                    2,
                ),
                "threshold_percentage": NEAR_CONSTANT_THRESHOLD * 100,
                "minimum_values": NEAR_CONSTANT_MIN_VALUES,
            },
        }

    if _is_identifier_like(non_missing, unique_count, column_name):
        return {
            "column_classification": "identifier_like",
            "classification_details": {
                "non_missing_values": value_count,
                "unique_values": unique_count,
                "uniqueness_ratio": 1.0,
                "identifier_evidence": _identifier_evidence(non_missing),
            },
        }

    return {
        "column_classification": "general",
        "classification_details": {
            "non_missing_values": value_count,
            "unique_values": unique_count,
            "uniqueness_ratio": round(unique_count / value_count, 4),
        },
    }


def _is_identifier_like(
    series: pd.Series,
    unique_count: int,
    column_name: str | None,
) -> bool:
    if series.isna().any() or unique_count != len(series) or len(series) < 3:
        return False

    if pd.api.types.is_integer_dtype(series.dtype):
        if not column_name or not re.search(
            r"(?:^|[_\W])(id|identifier|key|code|uuid|serial)(?:$|[_\W])",
            column_name,
            flags=re.IGNORECASE,
        ):
            return False
        differences = series.sort_values().diff().dropna()
        return bool(
            not differences.empty
            and differences.iloc[0] != 0
            and differences.eq(differences.iloc[0]).all()
        )

    values = series.astype(str).tolist()
    if all(_is_uuid(value) for value in values):
        return True

    matches = [
        _PREFIXED_IDENTIFIER_PATTERN.fullmatch(value)
        for value in values
    ]
    return bool(
        all(matches)
        and matches[0] is not None
        and matches[0].group(1)
        and len({match.group(1) for match in matches if match}) == 1
    )


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
    except ValueError:
        return False
    return True


def _identifier_evidence(series: pd.Series) -> str:
    if pd.api.types.is_integer_dtype(series.dtype):
        return "unique regular-step integer sequence"
    values = series.astype(str).tolist()
    if all(_is_uuid(value) for value in values):
        return "unique UUID-formatted values"
    return "unique values with a shared prefix and numeric suffix"

def infer_series_types(series: pd.Series) -> dict[str, Any]:
    """Describe plausible scalar interpretations without changing values."""

    non_missing = series.dropna()
    scalar_type = infer_dtype(non_missing, skipna=True)
    is_boolean_dtype = pd.api.types.is_bool_dtype(series.dtype)
    is_datetime_dtype = pd.api.types.is_datetime64_any_dtype(series.dtype)
    is_complex_dtype = (
        pd.api.types.is_complex_dtype(series.dtype)
        or scalar_type == "complex"
    )
    is_numeric_dtype = (
        pd.api.types.is_numeric_dtype(series.dtype)
        and not is_boolean_dtype
        and not is_complex_dtype
    )

    non_boolean = non_missing.loc[
        ~non_missing.map(lambda value: isinstance(value, bool))
    ]
    if is_complex_dtype:
        numeric_parseable = 0
        numeric_unparseable = 0
    else:
        numeric_values = pd.to_numeric(non_boolean, errors="coerce")
        numeric_parseable = int(numeric_values.notna().sum())
        numeric_unparseable = len(non_boolean) - numeric_parseable

    datetime_parseable = 0
    can_check_datetime = (
        not is_boolean_dtype
        and (
            pd.api.types.is_object_dtype(series.dtype)
            or pd.api.types.is_string_dtype(series.dtype)
            or pd.api.types.is_integer_dtype(series.dtype)
        )
    )
    if can_check_datetime and len(non_missing):
        datetime_values = pd.to_datetime(
            non_missing.astype("string"),
            errors="coerce",
            format="mixed",
        )
        datetime_parseable = int(datetime_values.notna().sum())

    candidate_types = []

    if is_boolean_dtype:
        candidate_types.append("boolean")
    elif is_datetime_dtype:
        candidate_types.append("datetime")
    elif is_complex_dtype:
        candidate_types.append("complex")
    else:
        if non_missing.map(lambda value: isinstance(value, str)).any():
            candidate_types.append("text")

        if is_numeric_dtype or numeric_parseable:
            candidate_types.append("numeric")

        if datetime_parseable >= 2:
            candidate_types.append("datetime")

    numeric_candidate = "numeric" in candidate_types
    datetime_candidate = "datetime" in candidate_types

    return {
        "scalar_type": scalar_type,
        "candidate_types": candidate_types,
        "ambiguous": len(candidate_types) > 1,
        "numeric_parseable_values": numeric_parseable,
        "numeric_unparseable_values": (
            numeric_unparseable if numeric_candidate else 0
        ),
        "datetime_parseable_values": (
            datetime_parseable if datetime_candidate else 0
        ),
        "datetime_unparseable_values": (
            len(non_missing) - datetime_parseable
            if datetime_candidate else 0
        ),
    }
