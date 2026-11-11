
from typing import Any

import pandas as pd
from pandas.api.types import infer_dtype

from engine.profiling.type_inference import (
    classify_series,
    infer_series_types,
)


def detect_data_quality_issues(
    df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Detect common data quality issues without modifying the dataset."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    issues = []

    if df.empty:
        return [{
            "issue": "empty_dataset",
            "severity": "high",
            "description": "The dataset contains no rows.",
        }]

    # 1. Detect missing values
    for column_index, column in enumerate(df.columns):
        series = df.iloc[:, column_index]
        missing_count = int(series.isna().sum())

        if missing_count == len(series) and len(series):
            issues.append({
                "issue": "empty_column",
                "column": str(column),
                "column_index": column_index,
                "severity": "medium",
                "count": missing_count,
                "percentage": 100.0,
                "description": (
                    f"Column '{column}' contains no non-missing values."
                ),
            })

        if missing_count > 0:
            missing_percentage = round(
                float(series.isna().mean() * 100), 2
            )

            severity = (
                "high" if missing_percentage >= 20 else "medium"
            )

            issues.append({
                "issue": "missing_values",
                "column": str(column),
                "column_index": column_index,
                "severity": severity,
                "count": missing_count,
                "percentage": missing_percentage,
                "description": (
                    f"Column '{column}' contains "
                    f"{missing_count} missing values."
                ),
            })

    # 2. Detect duplicate rows
    duplicate_count = int(df.duplicated().sum())

    if duplicate_count > 0:
        issues.append({
            "issue": "duplicate_rows",
            "severity": "medium",
            "count": duplicate_count,
            "description": (
                f"Dataset contains {duplicate_count} duplicate rows."
            ),
        })

    # 3. Detect constant and near-constant columns, excluding empty columns.
    for column_index, column in enumerate(df.columns):
        series = df.iloc[:, column_index]
        classification = classify_series(series, str(column))

        if classification["column_classification"] == "constant":
            details = classification["classification_details"]
            issues.append({
                "issue": "constant_column",
                "column": str(column),
                "column_index": column_index,
                "severity": "low",
                "count": details["non_missing_values"],
                "description": (
                    f"Column '{column}' has one distinct non-missing value "
                    f"across {details['non_missing_values']} rows."
                ),
            })
        elif classification["column_classification"] == "near_constant":
            details = classification["classification_details"]
            issues.append({
                "issue": "near_constant_column",
                "column": str(column),
                "column_index": column_index,
                "severity": "low",
                "count": details["non_missing_values"],
                "percentage": details["dominant_value_percentage"],
                "description": (
                    f"One value occurs in "
                    f"{details['dominant_value_percentage']}% of "
                    f"non-missing values in column '{column}' "
                    f"(threshold: {details['threshold_percentage']}%)."
                ),
            })
        elif (
            classification["column_classification"] == "identifier_like"
        ):
            details = classification["classification_details"]
            issues.append({
                "issue": "identifier_like_column",
                "column": str(column),
                "column_index": column_index,
                "severity": "low",
                "count": details["non_missing_values"],
                "percentage": 100.0,
                "description": (
                    f"Column '{column}' has unique values matching "
                    f"{details['identifier_evidence']}; this is a "
                    "potential identifier, not a confirmed one."
                ),
            })

    # 4. Flag columns with genuinely heterogeneous inferred scalar types.
    for column_index, column in enumerate(df.columns):
        series = df.iloc[:, column_index]

        if not pd.api.types.is_object_dtype(series.dtype):
            continue

        inferred_type = infer_dtype(series.dropna(), skipna=True)

        if inferred_type in {"mixed", "mixed-integer"}:
            issues.append({
                "issue": "mixed_types",
                "column": str(column),
                "column_index": column_index,
                "severity": "low",
                "description": (
                    f"Column '{column}' contains heterogeneous scalar "
                    "types. This is a potential issue; mixed values may "
                    "also be valid for the dataset."
                ),
            })

    # 5. Report failed parses only when the column also has values supporting
    # that interpretation; these findings are advisory and never convert data.
    for column_index, column in enumerate(df.columns):
        type_inference = infer_series_types(df.iloc[:, column_index])
        for candidate_type, unparseable_count in (
            (
                "numeric",
                type_inference["numeric_unparseable_values"],
            ),
            (
                "datetime",
                type_inference["datetime_unparseable_values"],
            ),
        ):
            if unparseable_count:
                issues.append({
                    "issue": "potential_unparseable_values",
                    "column": str(column),
                    "column_index": column_index,
                    "severity": "low",
                    "count": unparseable_count,
                    "description": (
                        f"{unparseable_count} non-missing values in "
                        f"column '{column}' do not match the candidate "
                        f"{candidate_type} interpretation. They may be "
                        "valid under another interpretation; no values "
                        "were converted."
                    ),
                })

        if (
            type_inference["ambiguous"]
            and "numeric" in type_inference["candidate_types"]
            and "datetime" in type_inference["candidate_types"]
        ):
            issues.append({
                "issue": "ambiguous_type",
                "column": str(column),
                "column_index": column_index,
                "severity": "low",
                "count": int(df.iloc[:, column_index].notna().sum()),
                "description": (
                    f"Column '{column}' has both numeric and datetime "
                    "interpretations. Review the intended meaning; no "
                    "values were converted."
                ),
            })

    # 6. Detect numerical outliers using the IQR method
    numeric_column_indexes = [
        index
        for index in range(len(df.columns))
        if (
            pd.api.types.is_numeric_dtype(df.iloc[:, index].dtype)
            and not pd.api.types.is_bool_dtype(df.iloc[:, index].dtype)
            and not pd.api.types.is_complex_dtype(df.iloc[:, index].dtype)
            and not infer_series_types(
                df.iloc[:, index]
            )["ambiguous"]
            and classify_series(
                df.iloc[:, index],
                str(df.columns[index]),
            )["column_classification"] == "general"
        )
    ]

    for column_index in numeric_column_indexes:
        column = df.columns[column_index]
        series = df.iloc[:, column_index].dropna()

        if series.nunique() < 2:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue

        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        outlier_count = int(
            ((series < lower_bound) | (series > upper_bound)).sum()
        )

        if outlier_count > 0:
            issues.append({
                "issue": "potential_outliers",
                "column": str(column),
                "column_index": column_index,
                "severity": "medium",
                "count": outlier_count,
                "description": (
                    f"Column '{column}' contains "
                    f"{outlier_count} potential outliers."
                ),
            })

    return issues
