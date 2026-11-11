
import warnings

import pandas as pd
import pytest

from engine.profiling.profiler import profile_dataframe


def test_dataset_overview():
    """Check total rows, columns and duplicate records."""

    df = pd.DataFrame({
        "name": ["Aman", "Riya", "Aman"],
        "marks": [80, 90, 80],
    })

    result = profile_dataframe(df)
    overview = result["dataset_overview"]

    assert overview["total_rows"] == 3
    assert overview["total_columns"] == 2
    assert overview["duplicate_rows"] == 1


def test_numeric_column_statistics():
    """Check numerical statistics."""

    df = pd.DataFrame({
        "marks": [70, 80, 90],
    })

    result = profile_dataframe(df)
    stats = result["columns"][0]["statistics"]

    assert stats["mean"] == 80.0
    assert stats["median"] == 80.0
    assert stats["min"] == 70.0
    assert stats["max"] == 90.0


@pytest.mark.parametrize(
    "values",
    [
        [1, 2, 100],
        [1.0, 2.0, 100.0],
        pd.array([1, 2, pd.NA, 100], dtype="Int64"),
        pd.array([1.0, 2.0, pd.NA, 100.0], dtype="Float64"),
    ],
    ids=["integer", "float", "nullable-integer", "nullable-float"],
)
def test_real_numeric_dtypes_keep_measurement_statistics(values):
    profile = profile_dataframe(
        pd.DataFrame({"measurement": values})
    )["columns"][0]

    assert profile["statistics"]["mean"] == pytest.approx(103 / 3)
    assert profile["statistics"]["min"] == 1.0
    assert profile["statistics"]["max"] == 100.0
    assert profile["missing_values"] == int(pd.isna(values).sum())


@pytest.mark.parametrize(
    "series",
    [
        pd.Series([True, False, True], dtype=bool),
        pd.Series([True, False, pd.NA], dtype="boolean"),
    ],
    ids=["boolean", "nullable-boolean"],
)
def test_boolean_columns_are_categorical_not_numeric_measurements(series):
    original = series.copy(deep=True)

    profile = profile_dataframe(
        pd.DataFrame({"flag": series})
    )["columns"][0]

    assert profile["type_inference"]["candidate_types"] == ["boolean"]
    assert "statistics" not in profile
    assert profile["top_values"]
    assert {item["value"] for item in profile["top_values"]} <= {
        "True",
        "False",
    }
    pd.testing.assert_series_equal(series, original)


def test_complex_columns_are_marked_unsupported_without_losing_values():
    for dtype in ("complex128", "object"):
        dataframe = pd.DataFrame({
            "measurement": pd.Series(
                [1 + 2j, 2 + 3j, 100 + 4j],
                dtype=dtype,
            ),
        })
        original = dataframe.copy(deep=True)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            profile = profile_dataframe(dataframe)["columns"][0]

        assert not caught
        assert profile["type_inference"]["candidate_types"] == ["complex"]
        assert profile["type_inference"]["numeric_parseable_values"] == 0
        assert "statistics" not in profile
        assert profile["statistics_status"] == "unsupported_complex"
        pd.testing.assert_frame_equal(dataframe, original)


def test_missing_values():
    """Check missing value detection."""

    df = pd.DataFrame({
        "marks": [80, None, 100, None],
    })

    result = profile_dataframe(df)
    column = result["columns"][0]

    assert column["missing_values"] == 2
    assert column["missing_percentage"] == 50.0


def test_categorical_column():
    """Check categorical value frequencies."""

    df = pd.DataFrame({
        "city": ["Delhi", "Meerut", "Delhi", "Noida"],
    })

    result = profile_dataframe(df)
    top_values = result["columns"][0]["top_values"]

    assert top_values[0]["value"] == "Delhi"
    assert top_values[0]["count"] == 2


def test_duplicate_column_names_are_profiled_by_position():
    df = pd.DataFrame(
        [[1, 10], [2, 20], [3, 30]],
        columns=["measure", "measure"],
    )

    result = profile_dataframe(df)

    assert [column["column_name"] for column in result["columns"]] == [
        "measure",
        "measure",
    ]
    assert [
        column["statistics"]["mean"]
        for column in result["columns"]
    ] == [2.0, 20.0]


def test_mixed_numeric_and_text_values_keep_scalar_type_metadata():
    df = pd.DataFrame({
        "measurement": [10, "unknown", 30, None],
    })
    original_values = df["measurement"].tolist()

    column = profile_dataframe(df)["columns"][0]

    assert column["inferred_type"] == "mixed-integer"
    assert column["type_inference"]["numeric_parseable_values"] == 2
    assert column["type_inference"]["numeric_unparseable_values"] == 1
    assert "statistics" not in column
    assert df["measurement"].tolist() == original_values


def test_numeric_strings_are_candidates_without_being_converted():
    df = pd.DataFrame({
        "measurement": pd.Series(["10", "20", None], dtype="object"),
    })

    column = profile_dataframe(df)["columns"][0]

    assert column["data_type"] == "object"
    assert column["type_inference"]["candidate_types"] == ["text", "numeric"]
    assert "statistics" not in column
    assert df["measurement"].dropna().tolist() == ["10", "20"]


def test_compact_date_values_are_reported_as_ambiguous_with_numbers():
    df = pd.DataFrame({
        "period": [20240101, 20240201],
    })

    column = profile_dataframe(df)["columns"][0]

    assert column["type_inference"]["candidate_types"] == [
        "numeric",
        "datetime",
    ]
    assert column["type_inference"]["ambiguous"] is True
    assert "statistics" not in column


def test_missing_values_do_not_count_as_unparseable():
    df = pd.DataFrame({
        "measurement": [10, 20, None],
    })

    column = profile_dataframe(df)["columns"][0]

    assert column["missing_values"] == 1
    assert column["type_inference"]["numeric_unparseable_values"] == 0


def test_date_like_numeric_strings_remain_ambiguous():
    df = pd.DataFrame({
        "period": pd.Series(
            ["20240101", "20240201", "100"],
            dtype="object",
        ),
    })

    column = profile_dataframe(df)["columns"][0]

    assert column["type_inference"]["candidate_types"] == [
        "text",
        "numeric",
        "datetime",
    ]
    assert column["type_inference"]["datetime_parseable_values"] == 2
    assert column["type_inference"]["datetime_unparseable_values"] == 1
    assert "statistics" not in column


def test_top_values_keep_scalar_type_when_display_strings_match():
    df = pd.DataFrame({
        "value": pd.Series([1, "1"], dtype="object"),
    })

    top_values = profile_dataframe(df)["columns"][0]["top_values"]

    assert {(item["value"], item["value_type"]) for item in top_values} == {
        ("1", "int"),
        ("1", "str"),
    }


def test_empty_column_is_distinguished_from_constant_column():
    df = pd.DataFrame({
        "blank": [None, None, None],
        "fixed": [7, 7, 7],
    })

    columns = profile_dataframe(df)["columns"]

    assert columns[0]["column_classification"] == "empty"
    assert columns[0]["missing_values"] == 3
    assert "statistics" not in columns[0]
    assert columns[1]["column_classification"] == "constant"
    assert columns[1]["unique_values"] == 1
    assert "statistics" not in columns[1]


def test_constant_with_missing_values_is_classified_as_constant():
    df = pd.DataFrame({
        "mostly_blank": [None, "fixed", None, None],
    })

    column = profile_dataframe(df)["columns"][0]

    assert column["column_classification"] == "constant"
    assert column["missing_values"] == 3
    assert column["unique_values"] == 1
    assert column["classification_details"]["non_missing_values"] == 1


def test_near_constant_column_is_reported_but_not_removed():
    values = ["common"] * 99 + ["rare"]
    df = pd.DataFrame({"category": values})

    column = profile_dataframe(df)["columns"][0]

    assert column["column_classification"] == "near_constant"
    assert column["classification_details"]["dominant_value_percentage"] == 99.0
    assert "statistics" not in column
    assert df["category"].tolist() == values


def test_identifier_like_values_are_distinguished_from_unique_categories():
    df = pd.DataFrame({
        "serial": [12001, 12002, 12003, 12004],
        "category": ["amber", "birch", "cobalt", "dune"],
    })

    columns = profile_dataframe(df)["columns"]

    assert columns[0]["column_classification"] == "identifier_like"
    assert "statistics" not in columns[0]
    assert columns[1]["column_classification"] == "general"
    assert columns[1]["unique_values"] == 4
    assert "statistics" not in columns[1]
    assert "top_values" in columns[1]


def test_mixed_column_profile_preserves_identity_and_classification():
    df = pd.DataFrame({
        "blank": [None] * 25,
        "fixed": [5] * 25,
        "near_fixed": ["same"] * 24 + ["other"],
        "record_code": [f"record-{index:03}" for index in range(25)],
        "unique_category": [
            "amber", "birch", "cobalt", "dune", "elm",
            "fern", "grove", "heather", "indigo", "jade",
            "khaki", "lilac", "maple", "navy", "ochre",
            "plum", "quartz", "russet", "sage", "teal",
            "umber", "violet", "willow", "xanthic", "yellow",
        ],
    })
    original = df.copy(deep=True)

    columns = profile_dataframe(df)["columns"]

    assert [column["column_name"] for column in columns] == list(df.columns)
    assert [column["column_classification"] for column in columns] == [
        "empty",
        "constant",
        "near_constant",
        "identifier_like",
        "general",
    ]
    pd.testing.assert_frame_equal(df, original)


def test_invalid_input():
    """Ensure non-DataFrame input is rejected."""

    with pytest.raises(TypeError):
        profile_dataframe([1, 2, 3])
