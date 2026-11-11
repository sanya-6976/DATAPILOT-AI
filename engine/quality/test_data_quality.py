
import pandas as pd
import pytest

from engine.quality.data_quality import detect_data_quality_issues


def test_missing_values_detection():
    df = pd.DataFrame({
        "marks": [80, None, 90, None, 70],
    })

    issues = detect_data_quality_issues(df)

    missing_issue = next(
        issue for issue in issues
        if issue["issue"] == "missing_values"
    )

    assert missing_issue["count"] == 2
    assert missing_issue["percentage"] == 40.0


def test_duplicate_rows_detection():
    df = pd.DataFrame({
        "name": ["Aman", "Riya", "Aman"],
        "marks": [80, 90, 80],
    })

    issues = detect_data_quality_issues(df)

    duplicate_issue = next(
        issue for issue in issues
        if issue["issue"] == "duplicate_rows"
    )

    assert duplicate_issue["count"] == 1


def test_constant_column_detection():
    df = pd.DataFrame({
        "city": ["Delhi", "Delhi", "Delhi"],
    })

    issues = detect_data_quality_issues(df)

    constant_issue = next(
        issue for issue in issues
        if issue["issue"] == "constant_column"
    )

    assert constant_issue["column"] == "city"


def test_outlier_detection():
    df = pd.DataFrame({
        "salary": [10, 11, 12, 13, 14, 100],
    })

    issues = detect_data_quality_issues(df)

    outlier_issue = next(
        issue for issue in issues
        if issue["issue"] == "potential_outliers"
    )

    assert outlier_issue["column"] == "salary"
    assert outlier_issue["count"] == 1


def test_boolean_columns_use_categorical_checks_not_iqr():
    df = pd.DataFrame({
        "flag": [True, False, True, None],
        "constant_flag": [True, True, True, True],
    })

    issues = detect_data_quality_issues(df)

    assert any(
        issue["issue"] == "missing_values"
        and issue["column"] == "flag"
        and issue["count"] == 1
        for issue in issues
    )
    assert any(
        issue["issue"] == "constant_column"
        and issue["column"] == "constant_flag"
        for issue in issues
    )
    assert not any(
        issue["issue"] == "potential_outliers"
        and issue["column"] in {"flag", "constant_flag"}
        for issue in issues
    )


def test_nullable_boolean_columns_use_categorical_checks_not_iqr():
    df = pd.DataFrame({
        "flag": pd.Series([True, False, pd.NA, True], dtype="boolean"),
    })

    issues = detect_data_quality_issues(df)

    assert any(
        issue["issue"] == "missing_values"
        and issue["column"] == "flag"
        and issue["count"] == 1
        for issue in issues
    )
    assert not any(
        issue["issue"] == "potential_outliers"
        and issue["column"] == "flag"
        for issue in issues
    )


@pytest.mark.parametrize(
    "values",
    [
        [10, 11, 12, 13, 14, 100],
        [10.0, 11.0, 12.0, 13.0, 14.0, 100.0],
        pd.array([10, 11, 12, 13, 14, 100], dtype="Int64"),
        pd.array([10.0, 11.0, 12.0, 13.0, 14.0, 100.0], dtype="Float64"),
    ],
    ids=["integer", "float", "nullable-integer", "nullable-float"],
)
def test_real_numeric_columns_still_receive_iqr_outlier_checks(values):
    df = pd.DataFrame({"measurement": values})

    issues = detect_data_quality_issues(df)

    outlier_issue = next(
        issue for issue in issues
        if issue["issue"] == "potential_outliers"
    )

    assert outlier_issue["column"] == "measurement"
    assert outlier_issue["count"] == 1


def test_mixed_scalar_types_are_reported_as_potential_quality_issue():
    df = pd.DataFrame({
        "measurement": [1, "unknown", None],
    })

    issues = detect_data_quality_issues(df)

    mixed_issue = next(
        issue for issue in issues
        if issue["issue"] == "mixed_types"
    )

    assert mixed_issue["column"] == "measurement"
    assert mixed_issue["severity"] == "low"
    assert "potential issue" in mixed_issue["description"]


def test_duplicate_column_names_are_processed_by_position():
    df = pd.DataFrame(
        [[1, None], [None, 20], [3, 30]],
        columns=["measure", "measure"],
    )

    issues = detect_data_quality_issues(df)

    assert isinstance(issues, list)
    findings = [
        issue for issue in issues if issue["issue"] == "missing_values"
    ]
    assert [issue["column_index"] for issue in findings] == [0, 1]


def test_mixed_numeric_text_reports_unparseable_values_separately():
    df = pd.DataFrame({
        "measurement": [10, "unknown", None, 30],
    })

    issues = detect_data_quality_issues(df)

    unparseable_issue = next(
        issue for issue in issues
        if issue["issue"] == "potential_unparseable_values"
    )
    missing_issue = next(
        issue for issue in issues
        if issue["issue"] == "missing_values"
    )

    assert unparseable_issue["count"] == 1
    assert missing_issue["count"] == 1
    assert unparseable_issue["issue"] != missing_issue["issue"]


def test_mixed_and_complex_columns_do_not_crash_numerical_checks():
    df = pd.DataFrame({
        "mixed": pd.Series([10, "unknown", None], dtype="object"),
        "complex": [1 + 0j, 2 + 0j, 100 + 0j],
    })

    issues = detect_data_quality_issues(df)

    assert any(
        issue["issue"] == "mixed_types"
        and issue["column"] == "mixed"
        for issue in issues
    )
    assert any(
        issue["issue"] == "potential_unparseable_values"
        and issue["column"] == "mixed"
        for issue in issues
    )
    assert not any(
        issue["issue"] == "potential_outliers"
        and issue["column"] in {"mixed", "complex"}
        for issue in issues
    )


def test_boolean_like_categories_are_not_treated_as_numeric():
    df = pd.DataFrame({
        "flag": [True, False, "yes", "no", None],
    })

    issues = detect_data_quality_issues(df)

    assert not any(
        issue["issue"] == "potential_unparseable_values"
        for issue in issues
    )


def test_ambiguous_numeric_date_column_is_reported_without_cleaning():
    df = pd.DataFrame({
        "period": [20240101, 20240201],
    })

    issues = detect_data_quality_issues(df)

    ambiguity_issue = next(
        issue for issue in issues
        if issue["issue"] == "ambiguous_type"
    )

    assert ambiguity_issue["column"] == "period"
    assert ambiguity_issue["count"] == 2
    assert df["period"].tolist() == [20240101, 20240201]


def test_empty_column_is_not_reported_as_constant():
    df = pd.DataFrame({
        "blank": [None, None, None],
        "fixed": [7, 7, 7],
    })

    issues = detect_data_quality_issues(df)
    column_issues = {
        (issue.get("column"), issue["issue"])
        for issue in issues
    }

    assert ("blank", "empty_column") in column_issues
    assert ("blank", "constant_column") not in column_issues
    assert ("fixed", "constant_column") in column_issues


def test_single_value_with_missing_rows_is_constant_and_missing():
    df = pd.DataFrame({
        "mostly_blank": [None, "fixed", None, None],
    })

    issues = detect_data_quality_issues(df)

    assert any(
        issue["issue"] == "missing_values"
        and issue["column"] == "mostly_blank"
        and issue["count"] == 3
        for issue in issues
    )
    assert any(
        issue["issue"] == "constant_column"
        and issue["column"] == "mostly_blank"
        and issue["count"] == 1
        for issue in issues
    )


def test_near_constant_and_identifier_like_findings_are_conservative():
    df = pd.DataFrame({
        "near_constant": ["common"] * 99 + ["rare"],
        "serial": list(range(12001, 12101)),
        "unique_category": ["amber", "birch", "cobalt", "dune"] * 25,
    })

    issues = detect_data_quality_issues(df)

    assert any(
        issue["issue"] == "near_constant_column"
        and issue["column"] == "near_constant"
        and issue["percentage"] == 99.0
        for issue in issues
    )
    assert any(
        issue["issue"] == "identifier_like_column"
        and issue["column"] == "serial"
        for issue in issues
    )
    assert not any(
        issue["issue"] == "identifier_like_column"
        and issue["column"] == "unique_category"
        for issue in issues
    )


def test_unique_numeric_measurements_are_not_called_identifiers():
    df = pd.DataFrame({
        "measurement": [70, 80, 90, 100],
    })

    issues = detect_data_quality_issues(df)

    assert not any(
        issue["issue"] == "identifier_like_column"
        for issue in issues
    )


def test_empty_dataframe():
    df = pd.DataFrame()

    issues = detect_data_quality_issues(df)

    assert issues[0]["issue"] == "empty_dataset"


def test_invalid_input():
    with pytest.raises(TypeError):
        detect_data_quality_issues([1, 2, 3])
