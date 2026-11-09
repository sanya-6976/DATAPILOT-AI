from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services import dataset_service

client = TestClient(app)


def test_csv_upload(tmp_path, monkeypatch):
    monkeypatch.setattr(
        dataset_service,
        "DATASET_STORAGE_DIR",
        tmp_path,
    )

    csv_content = (
        b"name,age,salary\n"
        b"Alice,25,50000\n"
        b"Bob,30,60000\n"
    )

    response = client.post(
        "/api/v1/datasets/upload",
        files={
            "file": (
                "employees.csv",
                csv_content,
                "text/csv",
            )
        },
    )

    assert response.status_code == 201

    data = response.json()

    assert data["filename"] == "employees.csv"
    assert data["file_format"] == "csv"
    assert data["row_count"] == 2
    assert data["column_count"] == 3
    assert data["preview"][0]["name"] == "Alice"
    assert len(data["sha256"]) == 64


def test_unsupported_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        dataset_service,
        "DATASET_STORAGE_DIR",
        tmp_path,
    )

    response = client.post(
        "/api/v1/datasets/upload",
        files={
            "file": (
                "sample.exe",
                b"not a dataset",
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 415


def test_empty_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        dataset_service,
        "DATASET_STORAGE_DIR",
        tmp_path,
    )

    response = client.post(
        "/api/v1/datasets/upload",
        files={
            "file": (
                "empty.csv",
                b"",
                "text/csv",
            )
        },
    )

    assert response.status_code == 400
