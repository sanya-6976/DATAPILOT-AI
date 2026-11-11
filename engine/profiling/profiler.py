
from typing import Any

import pandas as pd

from engine.profiling.type_inference import (
    classify_series,
    infer_series_types,
)


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Generate a deterministic statistical profile of a DataFrame."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    rows, columns = df.shape

    column_profiles = []

    for column_index, column in enumerate(df.columns):
        series = df.iloc[:, column_index]
        non_null = series.dropna()
        type_inference = infer_series_types(series)
        classification = classify_series(series, str(column))

        profile = {
            "column_name": str(column),
            "data_type": str(series.dtype),
            "inferred_type": type_inference["scalar_type"],
            "type_inference": type_inference,
            **classification,
            "total_values": int(rows),
            "missing_values": int(series.isna().sum()),
            "missing_percentage": (
                round(float(series.isna().mean() * 100), 2)
                if rows else 0.0
            ),
            "unique_values": int(series.nunique()),
        }

        if (
            pd.api.types.is_numeric_dtype(series)
            and not pd.api.types.is_bool_dtype(series.dtype)
            and not pd.api.types.is_complex_dtype(series.dtype)
            and not type_inference["ambiguous"]
            and classification["column_classification"] not in {
                "empty",
                "constant",
                "near_constant",
                "identifier_like",
            }
        ):
            profile["statistics"] = {
                "mean": _safe_number(series.mean()),
                "median": _safe_number(series.median()),
                "std": _safe_number(series.std()),
                "min": _safe_number(series.min()),
                "max": _safe_number(series.max()),
            }

        elif (
            pd.api.types.is_complex_dtype(series.dtype)
            or "complex" in type_inference["candidate_types"]
        ):
            profile["statistics_status"] = "unsupported_complex"

        elif (
            classification["column_classification"] != "empty"
            and classification["column_classification"] != "identifier_like"
            and (
                pd.api.types.is_bool_dtype(series.dtype)
                or pd.api.types.is_object_dtype(series)
                or isinstance(series.dtype, pd.CategoricalDtype)
                or pd.api.types.is_string_dtype(series)
            )
        ):
            profile["top_values"] = [
                {
                    "value": str(value),
                    "value_type": type(value).__name__,
                    "count": int(count),
                }
                for value, count in non_null.value_counts().head(5).items()
            ]

        column_profiles.append(profile)

    return {
        "dataset_overview": {
            "total_rows": int(rows),
            "total_columns": int(columns),
            "duplicate_rows": int(df.duplicated().sum()),
            "memory_usage_bytes": int(
                df.memory_usage(deep=True).sum()
            ),
        },
        "columns": column_profiles,
    }


def _safe_number(value: Any) -> float | None:
    """Convert numeric statistics to JSON-safe Python values."""
    if pd.isna(value):
        return None
    return float(value)
