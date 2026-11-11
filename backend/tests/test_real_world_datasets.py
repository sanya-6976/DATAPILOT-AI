from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services import dataset_service

client = TestClient(app)
REAL_DATASET_DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "sample_data"
    / "real_world"
)
REAL_DATASET_PATHS = sorted(REAL_DATASET_DIRECTORY.glob("*.csv"))


@pytest.mark.parametrize(
    "dataset_path",
    REAL_DATASET_PATHS,
    ids=lambda path: path.stem,
)
def test_real_world_csv_upload_matches_independent_pandas_profile(
    dataset_path,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        dataset_service,
        "DATASET_STORAGE_DIR",
        tmp_path,
    )

    independent_data = pd.read_csv(dataset_path)

    with dataset_path.open("rb") as dataset_file:
        response = client.post(
            "/api/v1/datasets/upload",
            files={
                "file": (
                    dataset_path.name,
                    dataset_file,
                    "text/csv",
                )
            },
        )

    assert response.status_code == 201, response.text
    result = response.json()

    assert result["row_count"] == len(independent_data)
    assert result["column_count"] == len(independent_data.columns)
    assert result["profile"]["dataset_overview"]["total_rows"] == (
        len(independent_data)
    )
    assert result["profile"]["dataset_overview"]["total_columns"] == (
        len(independent_data.columns)
    )

    expected_names = [str(column) for column in independent_data.columns]
    profile_columns = result["profile"]["columns"]
    response_columns = result["columns"]
    assert [column["column_name"] for column in profile_columns] == (
        expected_names
    )
    assert [column["name"] for column in response_columns] == expected_names

    for column_index, profile in enumerate(profile_columns):
        independent_series = independent_data.iloc[:, column_index]
        assert profile["missing_values"] == int(independent_series.isna().sum())
        if "statistics" in profile:
            assert pd.api.types.is_numeric_dtype(independent_series)
            assert profile["statistics"]["mean"] == pytest.approx(
                independent_series.mean()
            )

    expected_names_set = set(expected_names)
    finding_columns = {
        finding["column"]
        for finding in result["quality_issues"]
        if finding.get("column") is not None
    }
    assert finding_columns <= expected_names_set

    for preview_row in result["preview"]:
        assert set(preview_row) <= expected_names_set


def test_real_world_fixture_set_covers_multiple_datasets():
    assert len(REAL_DATASET_PATHS) >= 2
