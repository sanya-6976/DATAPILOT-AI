import csv
import hashlib
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO, StringIO
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services import dataset_service
from backend.app.services.dataset_storage import (
    DatasetStorageManager,
    StorageReservation,
)

client = TestClient(app)


def _assert_upload_artifacts_cleaned(storage_dir: Path) -> None:
    if not storage_dir.exists():
        return
    assert {
        path.name
        for path in storage_dir.iterdir()
        if path.name != ".storage.lock"
    } == set()
    manager = DatasetStorageManager(storage_dir, max_bytes=1_073_741_824)
    assert manager.get_storage_usage_bytes() == 0
    assert manager.get_reserved_bytes() == 0


def test_titanic_upload_returns_profile_and_quality_report(
    tmp_path,
    monkeypatch,
):
    """Verify both engines run during a real dataset upload."""

    monkeypatch.setattr(
        dataset_service,
        "DATASET_STORAGE_DIR",
        tmp_path,
    )

    dataset_path = (
        Path(__file__).resolve().parents[2]
        / "sample_data"
        / "titanic.csv"
    )

    assert dataset_path.exists(), "Download Titanic dataset first."

    with dataset_path.open("rb") as dataset:
        response = client.post(
            "/api/v1/datasets/upload",
            files={
                "file": (
                    "titanic.csv",
                    dataset,
                    "text/csv",
                )
            },
        )

    assert response.status_code == 201

    result = response.json()

    assert result["row_count"] == 891
    assert result["column_count"] == 15

    assert "profile" in result
    assert "quality_issues" in result

    assert result["profile"]["dataset_overview"]["total_rows"] == 891

    missing_issues = [
        issue
        for issue in result["quality_issues"]
        if issue["issue"] == "missing_values"
    ]

    age_issue = next(
        issue for issue in missing_issues
        if issue["column"] == "age"
    )

    assert age_issue["count"] == 177


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

    assert set(data) == {
        "dataset_id",
        "filename",
        "file_format",
        "size_bytes",
        "row_count",
        "column_count",
        "columns",
        "missing_values",
        "preview",
        "sha256",
        "profile",
        "quality_issues",
    }
    assert data["filename"] == "employees.csv"
    assert data["file_format"] == "csv"
    assert data["row_count"] == 2
    assert data["column_count"] == 3
    assert data["preview"][0]["name"] == "Alice"
    assert len(data["sha256"]) == 64
    manager = DatasetStorageManager(tmp_path)
    metadata = manager.read_metadata(data["dataset_id"])
    stored_file = manager.find_dataset_file(data["dataset_id"])
    assert metadata.original_filename == "employees.csv"
    assert metadata.row_count == data["row_count"]
    assert metadata.column_count == data["column_count"]
    assert stored_file.read_bytes() == csv_content
    assert hashlib.sha256(stored_file.read_bytes()).hexdigest() == data["sha256"]
    assert manager.verify_dataset_integrity(data["dataset_id"]).sha256 == (
        data["sha256"]
    )
    assert manager.report_staging_directories() == ()
    metadata_response = client.get(
        f"/api/v1/datasets/{data['dataset_id']}"
    )
    assert metadata_response.status_code == 200
    assert metadata_response.json()["filename"] == "employees.csv"
    assert metadata_response.json()["sha256"] == data["sha256"]
    assert str(tmp_path) not in metadata_response.text


def test_dataset_listing_is_stable_paginated_and_excludes_uncommitted_files(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(dataset_service, "DATASET_STORAGE_DIR", tmp_path)
    dataset_ids = []
    for filename, content in (
        ("first.csv", b"alpha\n1\n"),
        ("second.csv", b"beta,gamma\nx,2\n"),
        ("third.csv", b"delta\ntrue\n"),
    ):
        response = client.post(
            "/api/v1/datasets/upload",
            files={"file": (filename, content, "text/csv")},
        )
        assert response.status_code == 201
        dataset_ids.append(response.json()["dataset_id"])

    legacy_id = "12345678123456781234567812345678"
    (tmp_path / f"{legacy_id}.csv").write_bytes(b"legacy\n1\n")
    (tmp_path / ".unrelated.tmp").write_bytes(b"temporary")
    staging_dir = tmp_path / (
        ".datapilot-staging-87654321876543218765432187654321"
    )
    staging_dir.mkdir()
    (staging_dir / ".active.lock").write_bytes(b"")
    (staging_dir / "partial.csv").write_bytes(b"partial")

    first_page = client.get(
        "/api/v1/datasets",
        params={"limit": 2, "offset": 0},
    )
    second_page = client.get(
        "/api/v1/datasets",
        params={"limit": 2, "offset": 2},
    )

    assert first_page.status_code == second_page.status_code == 200
    first_page_data = first_page.json()
    second_page_data = second_page.json()
    ordered_ids = sorted(dataset_ids)
    assert [item["dataset_id"] for item in first_page_data["datasets"]] == (
        ordered_ids[:2]
    )
    assert [item["dataset_id"] for item in second_page_data["datasets"]] == (
        ordered_ids[2:]
    )
    assert first_page_data["total"] == 3
    assert first_page_data["unavailable_count"] == 0
    assert str(tmp_path) not in first_page.text
    assert ".staging-" not in first_page.text
    assert ".tmp" not in first_page.text


def test_dataset_metadata_retrieval_validates_uuid_and_reports_not_found(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(dataset_service, "DATASET_STORAGE_DIR", tmp_path)

    invalid_response = client.get("/api/v1/datasets/../outside")
    invalid_uuid_response = client.get(
        "/api/v1/datasets/not-a-dataset-uuid"
    )
    unknown_response = client.get(
        "/api/v1/datasets/12345678123456781234567812345678"
    )

    assert invalid_response.status_code in {404, 422}
    assert invalid_uuid_response.status_code == 422
    assert unknown_response.status_code == 404
    assert "storage" not in unknown_response.text.lower()
    assert str(tmp_path) not in unknown_response.text


def test_metadata_retrieval_reports_missing_original_and_corrupt_metadata(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(dataset_service, "DATASET_STORAGE_DIR", tmp_path)
    upload_response = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("sample.csv", b"value\n1\n", "text/csv")},
    )
    dataset_id = upload_response.json()["dataset_id"]
    stored_path = tmp_path / f"{dataset_id}.csv"
    stored_path.unlink()

    missing_file = client.get(f"/api/v1/datasets/{dataset_id}")
    assert missing_file.status_code == 404

    replacement_id = "87654321876543218765432187654321"
    metadata_path = tmp_path / f"{replacement_id}.metadata.json"
    metadata_path.write_text("{bad json", encoding="utf-8")
    corrupted = client.get(f"/api/v1/datasets/{replacement_id}")
    assert corrupted.status_code == 409
    assert str(tmp_path) not in corrupted.text


def test_metadata_and_preview_reject_integrity_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset_service, "DATASET_STORAGE_DIR", tmp_path)
    content = b"value\n1\n"
    upload_response = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("tamper-check.csv", content, "text/csv")},
    )
    dataset_id = upload_response.json()["dataset_id"]
    stored_path = tmp_path / f"{dataset_id}.csv"
    stored_path.write_bytes(b"value\n2\n")

    metadata_response = client.get(f"/api/v1/datasets/{dataset_id}")
    preview_response = client.get(
        f"/api/v1/datasets/{dataset_id}/preview"
    )

    assert metadata_response.status_code == 409
    assert preview_response.status_code == 409
    assert "sha" not in metadata_response.text.lower()
    assert str(tmp_path) not in preview_response.text


def test_preview_parses_hash_verified_snapshot_if_source_changes(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(dataset_service, "DATASET_STORAGE_DIR", tmp_path)
    content = b"value\n1\n"
    upload_response = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("snapshot.csv", content, "text/csv")},
    )
    assert upload_response.status_code == 201
    dataset_id = upload_response.json()["dataset_id"]
    stored_path = tmp_path / f"{dataset_id}.csv"
    original_loader = dataset_service.load_dataframe

    def replace_source_during_preview(file_path, file_format):
        assert file_path != stored_path
        stored_path.write_bytes(b"value\n2\n")
        return original_loader(file_path, file_format)

    monkeypatch.setattr(
        dataset_service,
        "load_dataframe",
        replace_source_during_preview,
    )

    response = client.get(f"/api/v1/datasets/{dataset_id}/preview")

    assert response.status_code == 200
    assert response.json()["rows"] == [[1]]
    assert client.get(
        f"/api/v1/datasets/{dataset_id}"
    ).status_code == 409


def test_metadata_and_preview_reject_symlinked_dataset_file(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(dataset_service, "DATASET_STORAGE_DIR", tmp_path)
    upload_response = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("linked.csv", b"value\n1\n", "text/csv")},
    )
    assert upload_response.status_code == 201
    dataset_id = upload_response.json()["dataset_id"]
    stored_path = tmp_path / f"{dataset_id}.csv"
    target_path = tmp_path / "outside.csv"
    target_path.write_bytes(b"value\n1\n")
    stored_path.unlink()
    stored_path.symlink_to(target_path)

    metadata_response = client.get(f"/api/v1/datasets/{dataset_id}")
    preview_response = client.get(
        f"/api/v1/datasets/{dataset_id}/preview"
    )

    assert metadata_response.status_code == 409
    assert preview_response.status_code == 409
    assert str(tmp_path) not in metadata_response.text
    assert str(tmp_path) not in preview_response.text


def test_dataset_preview_preserves_bytes_and_duplicate_column_names(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(dataset_service, "DATASET_STORAGE_DIR", tmp_path)
    content = b"measure,measure,group\n1,10,a\n2,20,b\n3,30,c\n"
    upload_response = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("duplicate.csv", content, "text/csv")},
    )
    assert upload_response.status_code == 201
    dataset_id = upload_response.json()["dataset_id"]
    original_path = tmp_path / f"{dataset_id}.csv"
    original_bytes = original_path.read_bytes()

    response = client.get(
        f"/api/v1/datasets/{dataset_id}/preview",
        params={"limit": 2, "offset": 1},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["columns"] == [
        {"name": "measure", "index": 0},
        {"name": "measure", "index": 1},
        {"name": "group", "index": 2},
    ]
    assert result["rows"] == [[2, 20, "b"], [3, 30, "c"]]
    assert result["total_rows"] == 3
    assert result["offset"] == 1
    assert result["limit"] == 2
    assert str(tmp_path) not in response.text
    assert original_path.read_bytes() == original_bytes == content


@pytest.mark.parametrize("file_format", ["csv", "xlsx", "json", "parquet"])
def test_preview_supports_all_dataset_formats(tmp_path, monkeypatch, file_format):
    monkeypatch.setattr(dataset_service, "DATASET_STORAGE_DIR", tmp_path)
    dataframe = pd.DataFrame({
        "unrelated_label": ["left", "right"],
        "value_17": [17, 29],
    })
    if file_format == "csv":
        source = dataframe.to_csv(index=False).encode("utf-8")
        content_type = "text/csv"
    elif file_format == "json":
        source = dataframe.to_json(orient="records").encode("utf-8")
        content_type = "application/json"
    else:
        buffer = BytesIO()
        if file_format == "xlsx":
            dataframe.to_excel(buffer, index=False)
            content_type = (
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            )
        else:
            dataframe.to_parquet(buffer, index=False)
            content_type = "application/octet-stream"
        source = buffer.getvalue()

    upload_response = client.post(
        "/api/v1/datasets/upload",
        files={
            "file": (
                f"format-sample.{file_format}",
                source,
                content_type,
            )
        },
    )
    assert upload_response.status_code == 201
    dataset_id = upload_response.json()["dataset_id"]
    original_path = tmp_path / f"{dataset_id}.{file_format}"
    original_bytes = original_path.read_bytes()

    preview_response = client.get(
        f"/api/v1/datasets/{dataset_id}/preview",
        params={"limit": 1},
    )

    assert preview_response.status_code == 200
    result = preview_response.json()
    assert [column["name"] for column in result["columns"]] == [
        "unrelated_label",
        "value_17",
    ]
    assert len(result["rows"]) == 1
    assert result["rows"][0][0] == "left"
    assert result["rows"][0][1] == 17
    assert original_path.read_bytes() == original_bytes == source


def test_header_only_dataset_has_columns_and_no_preview_rows(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(dataset_service, "DATASET_STORAGE_DIR", tmp_path)
    content = b"first_field,second_field\n"
    upload_response = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("empty-rows.csv", content, "text/csv")},
    )
    assert upload_response.status_code == 201
    dataset_id = upload_response.json()["dataset_id"]

    response = client.get(f"/api/v1/datasets/{dataset_id}/preview")

    assert response.status_code == 200
    assert response.json()["columns"] == [
        {"name": "first_field", "index": 0},
        {"name": "second_field", "index": 1},
    ]
    assert response.json()["rows"] == []
    assert response.json()["total_rows"] == 0


def test_preview_limit_is_bounded_and_invalid_dataset_is_safe(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(dataset_service, "DATASET_STORAGE_DIR", tmp_path)
    content = b"value\n" + b"1\n" * 25
    upload_response = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("many-rows.csv", content, "text/csv")},
    )
    dataset_id = upload_response.json()["dataset_id"]

    too_many = client.get(
        f"/api/v1/datasets/{dataset_id}/preview",
        params={"limit": dataset_service.MAX_PREVIEW_ROWS + 1},
    )
    invalid_id = client.get(
        "/api/v1/datasets/not-a-dataset-uuid/preview"
    )

    assert too_many.status_code == 422
    assert invalid_id.status_code == 422
    assert str(tmp_path) not in invalid_id.text


def test_preview_row_limit_is_configurable_through_settings(monkeypatch):
    from backend.app.core.config import Settings

    monkeypatch.setenv("DATASET_PREVIEW_MAX_ROWS", "23")
    settings = Settings(_env_file=None)

    assert settings.dataset_preview_max_rows == 23

    monkeypatch.setenv("DATASET_PREVIEW_MAX_ROWS", "1001")
    with pytest.raises(ValueError):
        Settings(_env_file=None)


def test_listing_omits_corrupt_and_missing_datasets_with_count(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(dataset_service, "DATASET_STORAGE_DIR", tmp_path)
    valid_response = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("valid.csv", b"value\n1\n", "text/csv")},
    )
    valid_id = valid_response.json()["dataset_id"]
    corrupt_id = "12345678123456781234567812345678"
    missing_id = "87654321876543218765432187654321"
    (tmp_path / f"{corrupt_id}.metadata.json").write_text(
        "{bad json",
        encoding="utf-8",
    )
    (tmp_path / f"{missing_id}.metadata.json").write_text(
        (tmp_path / f"{valid_id}.metadata.json").read_text(encoding="utf-8")
        .replace(valid_id, missing_id),
        encoding="utf-8",
    )

    response = client.get("/api/v1/datasets")

    assert response.status_code == 200
    result = response.json()
    assert [item["dataset_id"] for item in result["datasets"]] == [valid_id]
    assert result["total"] == 1
    assert result["unavailable_count"] == 2
    assert str(tmp_path) not in response.text


def test_listing_omits_same_size_dataset_content_mismatch(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(dataset_service, "DATASET_STORAGE_DIR", tmp_path)
    content = b"value\n1\n"
    upload_response = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("dataset.csv", content, "text/csv")},
    )
    dataset_id = upload_response.json()["dataset_id"]
    stored_path = next(tmp_path.glob(f"{dataset_id}.csv"))
    stored_path.write_bytes(b"value\n2\n")
    assert stored_path.stat().st_size == len(content)

    response = client.get("/api/v1/datasets")

    assert response.status_code == 200
    assert response.json()["datasets"] == []
    assert response.json()["total"] == 0
    assert response.json()["unavailable_count"] == 1
    assert str(tmp_path) not in response.text


def test_upload_rejects_quota_exhaustion_without_changing_legacy_files(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(dataset_service, "DATASET_STORAGE_DIR", tmp_path)
    content = b"value\n1\n"
    legacy_content = b"legacy contents"
    (tmp_path / "legacy.csv").write_bytes(legacy_content)
    monkeypatch.setattr(
        dataset_service,
        "MAX_STORAGE_BYTES",
        len(content) + len(legacy_content) - 1,
    )

    response = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("quota.csv", content, "text/csv")},
    )

    assert response.status_code == 507
    assert response.json()["detail"] == (
        "Dataset storage capacity would be exceeded."
    )
    assert (tmp_path / "legacy.csv").read_bytes() == legacy_content
    assert list(tmp_path.glob("*.csv")) == [tmp_path / "legacy.csv"]
    manager = DatasetStorageManager(
        tmp_path,
        max_bytes=dataset_service.MAX_STORAGE_BYTES,
    )
    assert manager.get_storage_usage_bytes() == len(legacy_content)
    assert manager.get_reserved_bytes() == 0


def test_identical_uploads_receive_independent_dataset_ids(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(dataset_service, "DATASET_STORAGE_DIR", tmp_path)
    content = b"value\n1\n"
    responses = [
        client.post(
            "/api/v1/datasets/upload",
            files={"file": ("same.csv", content, "text/csv")},
        )
        for _ in range(2)
    ]

    assert all(response.status_code == 201 for response in responses)
    results = [response.json() for response in responses]
    assert results[0]["dataset_id"] != results[1]["dataset_id"]
    assert results[0]["sha256"] == results[1]["sha256"]
    assert len(list(tmp_path.glob("*.csv"))) == 2
    assert len(list(tmp_path.glob("*.metadata.json"))) == 2
    assert all(path.read_bytes() == content for path in tmp_path.glob("*.csv"))


def test_concurrent_uploads_reserve_capacity_before_profiling(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(dataset_service, "DATASET_STORAGE_DIR", tmp_path)
    content = b"value\n1\n"
    monkeypatch.setattr(
        dataset_service,
        "MAX_STORAGE_BYTES",
        len(content) * 2 - 1,
    )
    profile_started = threading.Event()
    continue_profile = threading.Event()
    real_profile_dataframe = dataset_service.profile_dataframe

    def blocked_profile(dataframe):
        profile_started.set()
        assert continue_profile.wait(timeout=5)
        return real_profile_dataframe(dataframe)

    monkeypatch.setattr(dataset_service, "profile_dataframe", blocked_profile)
    with ThreadPoolExecutor(max_workers=1) as executor:
        first_request = executor.submit(
            client.post,
            "/api/v1/datasets/upload",
            files={"file": ("first.csv", content, "text/csv")},
        )
        assert profile_started.wait(timeout=5)
        second_response = client.post(
            "/api/v1/datasets/upload",
            files={"file": ("second.csv", content, "text/csv")},
        )
        continue_profile.set()
        first_response = first_request.result(timeout=10)

    assert first_response.status_code == 201
    assert second_response.status_code == 507
    assert len(list(tmp_path.glob("*.csv"))) == 1
    manager = DatasetStorageManager(
        tmp_path,
        max_bytes=dataset_service.MAX_STORAGE_BYTES,
    )
    assert manager.get_storage_usage_bytes() == len(content)
    assert manager.get_reserved_bytes() == 0


@pytest.mark.parametrize(
    "failing_stage",
    ["profiling", "quality"],
)
def test_analysis_failures_release_reservation_and_leave_no_dataset(
    tmp_path,
    monkeypatch,
    failing_stage,
):
    monkeypatch.setattr(dataset_service, "DATASET_STORAGE_DIR", tmp_path)

    def fail_analysis(*_args, **_kwargs):
        raise RuntimeError(f"simulated {failing_stage} failure")

    function_name = (
        "profile_dataframe"
        if failing_stage == "profiling"
        else "detect_data_quality_issues"
    )
    monkeypatch.setattr(dataset_service, function_name, fail_analysis)
    response = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("failure.csv", b"value\n1\n", "text/csv")},
    )

    assert response.status_code == 422
    assert response.json()["detail"].startswith("Unable to parse uploaded")
    _assert_upload_artifacts_cleaned(tmp_path)


def test_promotion_failure_releases_reservation_without_touching_existing(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(dataset_service, "DATASET_STORAGE_DIR", tmp_path)
    legacy_path = tmp_path / "legacy.csv"
    legacy_path.write_bytes(b"keep")

    def fail_link(*_args, **_kwargs):
        raise OSError("simulated promotion failure")

    monkeypatch.setattr(dataset_service.os, "link", fail_link)
    response = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("new.csv", b"value\n1\n", "text/csv")},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Unable to commit uploaded dataset."
    assert legacy_path.read_bytes() == b"keep"
    assert list(tmp_path.glob("*.csv")) == [legacy_path]
    assert list(tmp_path.glob("*.metadata.json")) == []
    assert DatasetStorageManager(tmp_path).get_reserved_bytes() == 0


@pytest.mark.parametrize("existing_kind", ["dataset", "metadata"])
def test_upload_never_overwrites_a_preexisting_uuid_path(
    tmp_path,
    monkeypatch,
    existing_kind,
):
    monkeypatch.setattr(dataset_service, "DATASET_STORAGE_DIR", tmp_path)
    dataset_id = "12345678123456781234567812345678"
    monkeypatch.setattr(
        dataset_service.uuid,
        "uuid4",
        lambda: uuid.UUID(dataset_id),
    )
    existing_path = (
        tmp_path / f"{dataset_id}.csv"
        if existing_kind == "dataset"
        else tmp_path / f"{dataset_id}.metadata.json"
    )
    original_bytes = b"pre-existing contents"
    existing_path.write_bytes(original_bytes)

    response = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("collision.csv", b"value\n1\n", "text/csv")},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Unable to commit uploaded dataset."
    assert existing_path.read_bytes() == original_bytes
    assert DatasetStorageManager(tmp_path).get_reserved_bytes() == 0


def test_metadata_failure_preserves_promoted_file_for_reconciliation(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(dataset_service, "DATASET_STORAGE_DIR", tmp_path)

    def fail_metadata_write(*_args, **_kwargs):
        raise OSError("simulated metadata persistence failure")

    monkeypatch.setattr(
        DatasetStorageManager,
        "write_metadata_atomic",
        fail_metadata_write,
    )
    content = b"value\n1\n"
    response = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("orphan.csv", content, "text/csv")},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Dataset file was committed, but metadata persistence failed; "
        "reconciliation is required."
    )
    stored_files = list(tmp_path.glob("*.csv"))
    assert len(stored_files) == 1
    assert stored_files[0].read_bytes() == content
    manager = DatasetStorageManager(tmp_path)
    assert manager.get_storage_usage_bytes() == len(content)
    unreconciled = manager.report_files_without_metadata()
    assert len(unreconciled) == 1
    assert unreconciled[0].file_size_bytes == len(content)
    assert unreconciled[0].file_extension == ".csv"
    assert unreconciled[0].dataset_id == stored_files[0].stem


def test_reservation_release_failure_reports_committed_dataset(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(dataset_service, "DATASET_STORAGE_DIR", tmp_path)

    def fail_release(_self):
        raise PermissionError("simulated reservation release failure")

    monkeypatch.setattr(StorageReservation, "release", fail_release)
    content = b"value\n1\n"
    response = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("committed.csv", content, "text/csv")},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Dataset was committed, but its storage reservation could not be "
        "released."
    )
    stored_files = list(tmp_path.glob("*.csv"))
    assert len(stored_files) == 1
    assert stored_files[0].read_bytes() == content
    metadata_files = list(tmp_path.glob("*.metadata.json"))
    assert len(metadata_files) == 1
    manager = DatasetStorageManager(tmp_path)
    assert manager.get_storage_usage_bytes() == len(content)
    assert manager.read_metadata(stored_files[0].stem).file_size_bytes == (
        len(content)
    )


def test_generic_dataset_upload_profiles_actual_structure(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        dataset_service,
        "DATASET_STORAGE_DIR",
        tmp_path,
    )

    for dataset_index in range(2):
        column_names = [
            f"field_{dataset_index}_{column_index}"
            for column_index in range(2 + dataset_index)
        ]
        rows = []

        for row_index in range(3 + dataset_index):
            row = {}

            for column_index, column_name in enumerate(column_names):
                if column_index == 0:
                    value = f"value-{row_index}"
                elif column_index == 1:
                    value = row_index * 10
                else:
                    value = f"group-{row_index % 2}"

                row[column_name] = value

            rows.append(row)

        rows[-1][column_names[-1]] = None

        csv_file = StringIO()
        writer = csv.DictWriter(csv_file, fieldnames=column_names)
        writer.writeheader()
        writer.writerows(rows)

        response = client.post(
            "/api/v1/datasets/upload",
            files={
                "file": (
                    f"dataset-{dataset_index}.csv",
                    csv_file.getvalue().encode("utf-8"),
                    "text/csv",
                )
            },
        )

        assert response.status_code == 201

        result = response.json()
        expected_columns = set(column_names)
        profiled_columns = {
            column["column_name"]
            for column in result["profile"]["columns"]
        }
        quality_issue_columns = {
            issue["column"]
            for issue in result["quality_issues"]
            if issue.get("column") is not None
        }

        assert result["row_count"] == len(rows)
        assert result["column_count"] == len(column_names)
        assert profiled_columns == expected_columns
        assert quality_issue_columns <= expected_columns


def test_json_upload(tmp_path, monkeypatch):
    monkeypatch.setattr(
        dataset_service,
        "DATASET_STORAGE_DIR",
        tmp_path,
    )

    json_content = b'[{"name":"Alice","age":25},{"name":"Bob","age":30}]'

    response = client.post(
        "/api/v1/datasets/upload",
        files={
            "file": (
                "people.json",
                json_content,
                "application/json",
            )
        },
    )

    assert response.status_code == 201

    data = response.json()

    assert data["file_format"] == "json"
    assert data["row_count"] == 2
    assert data["column_count"] == 2
    assert data["preview"][0]["name"] == "Alice"


def test_json_upload_preserves_mixed_values_and_reports_ambiguity(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        dataset_service,
        "DATASET_STORAGE_DIR",
        tmp_path,
    )

    json_content = (
        b'[{"measurement":10,"numeric_text":"10",'
        b'"partly_missing":10,"flag":true,"period":20240101},'
        b'{"measurement":"unknown","numeric_text":"20",'
        b'"partly_missing":null,"flag":"yes","period":20240201},'
        b'{"measurement":30,"numeric_text":"30",'
        b'"partly_missing":30,"flag":false,"period":20240301}]'
    )

    response = client.post(
        "/api/v1/datasets/upload",
        files={
            "file": (
                "mixed-values.json",
                json_content,
                "application/json",
            )
        },
    )

    assert response.status_code == 201
    result = response.json()
    profiled_columns = {
        column["column_name"]: column
        for column in result["profile"]["columns"]
    }

    assert result["preview"][1]["measurement"] == "unknown"
    assert result["preview"][1]["numeric_text"] == "20"
    assert result["missing_values"]["partly_missing"] == 1
    assert profiled_columns["measurement"]["missing_values"] == 0
    assert profiled_columns["measurement"]["type_inference"][
        "numeric_unparseable_values"
    ] == 1
    assert "statistics" not in profiled_columns["measurement"]
    assert profiled_columns["numeric_text"]["type_inference"][
        "candidate_types"
    ] == ["text", "numeric"]
    assert "statistics" not in profiled_columns["numeric_text"]
    assert profiled_columns["period"]["type_inference"][
        "candidate_types"
    ] == ["numeric", "datetime"]
    assert "statistics" not in profiled_columns["period"]

    issue_types_by_column = {
        (issue["column"], issue["issue"])
        for issue in result["quality_issues"]
        if issue.get("column") is not None
    }
    assert ("measurement", "missing_values") not in issue_types_by_column
    assert (
        "measurement",
        "potential_unparseable_values",
    ) in issue_types_by_column
    assert ("partly_missing", "missing_values") in issue_types_by_column
    assert ("period", "ambiguous_type") in issue_types_by_column
    assert ("flag", "potential_unparseable_values") not in issue_types_by_column
    assert next(
        item for item in result["quality_issues"]
        if item["issue"] == "potential_unparseable_values"
        and item["column"] == "measurement"
    )["count"] == 1
    assert next(
        item for item in result["quality_issues"]
        if item["issue"] == "missing_values"
        and item["column"] == "partly_missing"
    )["count"] == 1
    stored_file = tmp_path / f"{result['dataset_id']}.json"
    assert stored_file.read_bytes() == json_content


def test_upload_profiles_empty_constant_near_constant_and_identifier_columns(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        dataset_service,
        "DATASET_STORAGE_DIR",
        tmp_path,
    )

    dataframe = pd.DataFrame({
        "blank": [None] * 100,
        "fixed_number": [7] * 100,
        "fixed_category": ["stable"] * 100,
        "mostly_fixed": ["common"] * 99 + ["rare"],
        "one_value": [None] * 99 + ["present"],
        "serial": list(range(5001, 5101)),
        "unique_category": [
            f"category-{chr(65 + index // 26)}{chr(65 + index % 26)}"
            for index in range(100)
        ],
    })
    source = StringIO()
    dataframe.to_csv(source, index=False)
    content = source.getvalue().encode("utf-8")

    response = client.post(
        "/api/v1/datasets/upload",
        files={
            "file": ("column-classes.csv", content, "text/csv"),
        },
    )

    assert response.status_code == 201
    result = response.json()
    columns = {
        column["column_name"]: column
        for column in result["profile"]["columns"]
    }

    assert {
        name: column["column_classification"]
        for name, column in columns.items()
    } == {
        "blank": "empty",
        "fixed_number": "constant",
        "fixed_category": "constant",
        "mostly_fixed": "near_constant",
        "one_value": "constant",
        "serial": "identifier_like",
        "unique_category": "general",
    }
    assert columns["blank"]["missing_values"] == 100
    assert columns["one_value"]["missing_values"] == 99
    assert columns["mostly_fixed"]["classification_details"][
        "dominant_value_percentage"
    ] == 99.0
    assert "statistics" not in columns["fixed_number"]
    assert "statistics" not in columns["serial"]

    quality_findings = {
        (issue.get("column"), issue["issue"])
        for issue in result["quality_issues"]
    }
    assert ("blank", "empty_column") in quality_findings
    assert ("blank", "constant_column") not in quality_findings
    assert ("one_value", "constant_column") in quality_findings
    assert ("mostly_fixed", "near_constant_column") in quality_findings
    assert ("serial", "identifier_like_column") in quality_findings
    assert ("unique_category", "identifier_like_column") not in quality_findings

    stored_file = tmp_path / f"{result['dataset_id']}.csv"
    assert stored_file.read_bytes() == content


@pytest.mark.parametrize(
    ("file_format", "content_type"),
    [
        ("xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        ("parquet", "application/octet-stream"),
    ],
)
def test_columnar_and_spreadsheet_uploads(
    tmp_path,
    monkeypatch,
    file_format,
    content_type,
):
    monkeypatch.setattr(
        dataset_service,
        "DATASET_STORAGE_DIR",
        tmp_path,
    )

    dataframe = pd.DataFrame({
        "first_field": [1, 2],
        "second_field": ["alpha", "beta"],
    })
    dataset_file = BytesIO()

    if file_format == "xlsx":
        dataframe.to_excel(dataset_file, index=False)
    else:
        dataframe.to_parquet(dataset_file, index=False)

    response = client.post(
        "/api/v1/datasets/upload",
        files={
            "file": (
                f"sample.{file_format}",
                dataset_file.getvalue(),
                content_type,
            )
        },
    )

    assert response.status_code == 201
    result = response.json()
    assert result["file_format"] == file_format
    assert result["row_count"] == len(dataframe)
    assert result["column_count"] == len(dataframe.columns)


@pytest.mark.parametrize("file_format", ["csv", "xlsx"])
def test_duplicate_column_names_are_preserved_and_reported(
    tmp_path,
    monkeypatch,
    file_format,
):
    monkeypatch.setattr(
        dataset_service,
        "DATASET_STORAGE_DIR",
        tmp_path,
    )

    source_columns = ["measure", "measure", "measure__2"]
    dataframe = pd.DataFrame(
        [
            ["first", None, 10],
            ["second", "present", 20],
        ],
        columns=source_columns,
    )

    if file_format == "csv":
        dataset_file = StringIO()
        dataframe.to_csv(dataset_file, index=False)
        content = dataset_file.getvalue().encode("utf-8")
        content_type = "text/csv"
    else:
        dataset_file = BytesIO()
        dataframe.to_excel(dataset_file, index=False)
        content = dataset_file.getvalue()
        content_type = (
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        )

    response = client.post(
        "/api/v1/datasets/upload",
        files={
            "file": (
                f"duplicate-columns.{file_format}",
                content,
                content_type,
            )
        },
    )

    assert response.status_code == 201
    result = response.json()
    result_columns = [column["name"] for column in result["columns"]]
    profile_columns = result["profile"]["columns"]
    valid_columns = set(result_columns)

    assert result_columns == ["measure", "measure__3", "measure__2"]
    assert [column["source_name"] for column in result["columns"]] == (
        source_columns
    )
    assert [
        column["source_column_name"]
        for column in profile_columns
    ] == source_columns
    assert [column["column_name"] for column in profile_columns] == (
        result_columns
    )
    assert result["missing_values"] == {
        "measure": 0,
        "measure__3": 1,
        "measure__2": 0,
    }
    assert [
        column["missing_values"]
        for column in profile_columns
    ] == [0, 1, 0]
    assert result["preview"][0] == {
        "measure": "first",
        "measure__3": None,
        "measure__2": 10,
    }
    assert all(
        issue.get("column") is None
        or issue["column"] in valid_columns
        for issue in result["quality_issues"]
    )
    assert any(
        issue["issue"] == "missing_values"
        and issue["column"] == "measure__3"
        and issue["count"] == 1
        for issue in result["quality_issues"]
    )
    assert any(
        issue["issue"] == "duplicate_column_names"
        and issue["column"] == "measure"
        and issue["count"] == 1
        for issue in result["quality_issues"]
    )
    stored_files = list(tmp_path.glob(f"*.{file_format}"))
    assert len(stored_files) == 1
    assert stored_files[0].read_bytes() == content


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("malformed.json", b'{"missing":'),
        ("scalar.json", b"42"),
        ("object.json", b'{"alpha":1,"beta":2}'),
    ],
)
def test_invalid_json_structures_are_rejected_and_cleaned_up(
    tmp_path,
    monkeypatch,
    filename,
    content,
):
    monkeypatch.setattr(
        dataset_service,
        "DATASET_STORAGE_DIR",
        tmp_path,
    )

    response = client.post(
        "/api/v1/datasets/upload",
        files={
            "file": (
                filename,
                content,
                "application/json",
            )
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "Unable to parse uploaded dataset. Check that the file contents "
        "match its extension and contain tabular data."
    )
    assert str(tmp_path) not in response.text
    _assert_upload_artifacts_cleaned(tmp_path)


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("random.csv", b"\x00\xff\xfe\x80"),
        ("random.xlsx", b"not a workbook"),
        ("random.json", b"\x00\xff\xfe\x80"),
        ("random.parquet", b"not parquet"),
        ("corrupt.xlsx", b"PK\x03\x04 truncated zip content"),
        ("misleading.csv", b'{"alpha":1,"beta":2}'),
        ("misleading.csv", b'[{"alpha":1,"beta":2}]'),
        ("misleading.json", b"alpha,beta\n1,2\n"),
        ("misleading.xlsx", b"alpha,beta\n1,2\n"),
        ("misleading.parquet", b"alpha,beta\n1,2\n"),
    ],
)
def test_invalid_file_content_is_rejected_without_leaving_files(
    tmp_path,
    monkeypatch,
    filename,
    content,
):
    monkeypatch.setattr(
        dataset_service,
        "DATASET_STORAGE_DIR",
        tmp_path,
    )

    response = client.post(
        "/api/v1/datasets/upload",
        files={
            "file": (
                filename,
                content,
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "Unable to parse uploaded dataset. Check that the file contents "
        "match its extension and contain tabular data."
    )
    assert str(tmp_path) not in response.text
    _assert_upload_artifacts_cleaned(tmp_path)


@pytest.mark.parametrize(
    "filename",
    ["empty.csv", "empty.xlsx", "empty.json", "empty.parquet"],
)
def test_empty_supported_files_are_rejected_without_storage(
    tmp_path,
    monkeypatch,
    filename,
):
    monkeypatch.setattr(
        dataset_service,
        "DATASET_STORAGE_DIR",
        tmp_path,
    )

    response = client.post(
        "/api/v1/datasets/upload",
        files={
            "file": (
                filename,
                b"",
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is empty."
    _assert_upload_artifacts_cleaned(tmp_path)


@pytest.mark.parametrize(
    ("filename", "content_type", "content"),
    [
        ("headers.csv", "text/csv", b"first,second\n"),
        (
            "headers.xlsx",
            (
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            b"",
        ),
    ],
)
def test_header_only_files_are_accepted_as_zero_row_datasets(
    tmp_path,
    monkeypatch,
    filename,
    content_type,
    content,
):
    monkeypatch.setattr(
        dataset_service,
        "DATASET_STORAGE_DIR",
        tmp_path,
    )

    if filename.endswith(".xlsx"):
        workbook = BytesIO()
        pd.DataFrame(columns=["first", "second"]).to_excel(
            workbook,
            index=False,
        )
        content = workbook.getvalue()

    response = client.post(
        "/api/v1/datasets/upload",
        files={"file": (filename, content, content_type)},
    )

    assert response.status_code == 201
    result = response.json()
    assert result["row_count"] == 0
    assert result["column_count"] == 2
    assert [column["name"] for column in result["columns"]] == [
        "first",
        "second",
    ]
    assert len(list(tmp_path.glob(f"*{Path(filename).suffix}"))) == 1


def test_upload_size_limit_cleans_up_partial_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        dataset_service,
        "DATASET_STORAGE_DIR",
        tmp_path,
    )
    monkeypatch.setattr(dataset_service, "MAX_UPLOAD_SIZE", 3)

    response = client.post(
        "/api/v1/datasets/upload",
        files={
            "file": (
                "large.csv",
                b"more than three bytes",
                "text/csv",
            )
        },
    )

    assert response.status_code == 413
    _assert_upload_artifacts_cleaned(tmp_path)


@pytest.mark.parametrize(
    ("limit_name", "limit", "content", "message"),
    [
        (
            "MAX_DATASET_ROWS",
            2,
            b"value\n1\n2\n3\n",
            "Dataset exceeds the configured row limit.",
        ),
        (
            "MAX_DATASET_COLUMNS",
            2,
            b"first,second,third\n1,2,3\n",
            "Dataset exceeds the configured column limit.",
        ),
        (
            "MAX_DATASET_CELLS",
            4,
            b"first,second\n1,2\n3,4\n5,6\n",
            "Dataset exceeds the configured cell limit.",
        ),
    ],
)
def test_csv_dimension_limits_reject_without_truncating(
    tmp_path,
    monkeypatch,
    limit_name,
    limit,
    content,
    message,
):
    monkeypatch.setattr(
        dataset_service,
        "DATASET_STORAGE_DIR",
        tmp_path,
    )
    monkeypatch.setattr(dataset_service, limit_name, limit)

    response = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("large.csv", content, "text/csv")},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == message
    _assert_upload_artifacts_cleaned(tmp_path)


def test_parsed_dataframe_memory_limit_cleans_up(tmp_path, monkeypatch):
    monkeypatch.setattr(
        dataset_service,
        "DATASET_STORAGE_DIR",
        tmp_path,
    )
    monkeypatch.setattr(
        dataset_service,
        "MAX_DATAFRAME_MEMORY_BYTES",
        1,
    )

    response = client.post(
        "/api/v1/datasets/upload",
        files={
            "file": ("memory.csv", b"value\nsome text\n", "text/csv"),
        },
    )

    assert response.status_code == 413
    assert response.json()["detail"] == (
        "Parsed dataset exceeds the configured memory limit."
    )
    _assert_upload_artifacts_cleaned(tmp_path)


def test_processing_deadline_rejects_and_cleans_up(tmp_path, monkeypatch):
    monkeypatch.setattr(
        dataset_service,
        "DATASET_STORAGE_DIR",
        tmp_path,
    )
    monkeypatch.setattr(dataset_service, "MAX_PROCESSING_SECONDS", 0)

    response = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("deadline.csv", b"value\n1\n", "text/csv")},
    )

    assert response.status_code == 408
    assert response.json()["detail"] == (
        "Dataset processing exceeded the configured time limit."
    )
    _assert_upload_artifacts_cleaned(tmp_path)


def test_xlsx_expanded_size_limit_cleans_up(tmp_path, monkeypatch):
    monkeypatch.setattr(
        dataset_service,
        "DATASET_STORAGE_DIR",
        tmp_path,
    )
    monkeypatch.setattr(
        dataset_service,
        "MAX_XLSX_UNCOMPRESSED_BYTES",
        1,
    )

    workbook = BytesIO()
    pd.DataFrame({"value": ["data"]}).to_excel(workbook, index=False)
    response = client.post(
        "/api/v1/datasets/upload",
        files={
            "file": (
                "expanded.xlsx",
                workbook.getvalue(),
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 413
    assert response.json()["detail"] == (
        "Spreadsheet exceeds the configured expanded-size limit."
    )
    _assert_upload_artifacts_cleaned(tmp_path)


def test_xlsx_row_limit_is_checked_before_dataframe_load(tmp_path, monkeypatch):
    monkeypatch.setattr(
        dataset_service,
        "DATASET_STORAGE_DIR",
        tmp_path,
    )
    monkeypatch.setattr(dataset_service, "MAX_DATASET_ROWS", 1)

    workbook = BytesIO()
    pd.DataFrame({"value": [1, 2, 3]}).to_excel(workbook, index=False)
    monkeypatch.setattr(
        dataset_service,
        "load_dataframe",
        lambda *_: pytest.fail("Oversized workbook must fail preflight"),
    )
    response = client.post(
        "/api/v1/datasets/upload",
        files={
            "file": (
                "too-many-rows.xlsx",
                workbook.getvalue(),
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 413
    assert response.json()["detail"] == (
        "Dataset exceeds the configured row limit."
    )
    _assert_upload_artifacts_cleaned(tmp_path)


def test_json_nesting_depth_limit_cleans_up(tmp_path, monkeypatch):
    monkeypatch.setattr(
        dataset_service,
        "DATASET_STORAGE_DIR",
        tmp_path,
    )
    monkeypatch.setattr(dataset_service, "MAX_JSON_NESTING_DEPTH", 3)

    content = b'{"a":{"b":{"c":{"d":1}}}}'
    response = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("nested.json", content, "application/json")},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == (
        "JSON exceeds the configured nesting-depth limit."
    )
    _assert_upload_artifacts_cleaned(tmp_path)


def test_json_row_limit_rejects_without_truncating(tmp_path, monkeypatch):
    monkeypatch.setattr(
        dataset_service,
        "DATASET_STORAGE_DIR",
        tmp_path,
    )
    monkeypatch.setattr(dataset_service, "MAX_DATASET_ROWS", 2)
    monkeypatch.setattr(
        dataset_service,
        "load_dataframe",
        lambda *_: pytest.fail("Oversized JSON must fail preflight"),
    )

    content = b'[{"value":1},{"value":2},{"value":3}]'
    response = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("too-many-rows.json", content, "application/json")},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == (
        "Dataset exceeds the configured row limit."
    )
    _assert_upload_artifacts_cleaned(tmp_path)


def test_json_column_limit_is_checked_before_dataframe_load(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        dataset_service,
        "DATASET_STORAGE_DIR",
        tmp_path,
    )
    monkeypatch.setattr(dataset_service, "MAX_DATASET_COLUMNS", 1)
    monkeypatch.setattr(
        dataset_service,
        "load_dataframe",
        lambda *_: pytest.fail("Oversized JSON must fail preflight"),
    )

    content = b'{"first":{"0":1},"second":{"0":2}}'
    response = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("too-many-columns.json", content, "application/json")},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == (
        "Dataset exceeds the configured column limit."
    )
    _assert_upload_artifacts_cleaned(tmp_path)


def test_parquet_uncompressed_payload_limit_cleans_up(tmp_path, monkeypatch):
    monkeypatch.setattr(
        dataset_service,
        "DATASET_STORAGE_DIR",
        tmp_path,
    )
    monkeypatch.setattr(
        dataset_service,
        "MAX_PARQUET_UNCOMPRESSED_BYTES",
        1,
    )

    parquet_file = BytesIO()
    pd.DataFrame({"value": ["a repeated value"] * 20}).to_parquet(
        parquet_file,
        index=False,
    )
    response = client.post(
        "/api/v1/datasets/upload",
        files={
            "file": (
                "expanded.parquet",
                parquet_file.getvalue(),
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 413
    assert response.json()["detail"] == (
        "Parquet dataset exceeds the configured expanded-size limit."
    )
    _assert_upload_artifacts_cleaned(tmp_path)


def test_parquet_row_limit_is_checked_before_dataframe_load(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        dataset_service,
        "DATASET_STORAGE_DIR",
        tmp_path,
    )
    monkeypatch.setattr(dataset_service, "MAX_DATASET_ROWS", 1)

    parquet_file = BytesIO()
    pd.DataFrame({"value": [1, 2, 3]}).to_parquet(
        parquet_file,
        index=False,
    )
    monkeypatch.setattr(
        dataset_service,
        "load_dataframe",
        lambda *_: pytest.fail("Oversized Parquet must fail preflight"),
    )
    response = client.post(
        "/api/v1/datasets/upload",
        files={
            "file": (
                "too-many-rows.parquet",
                parquet_file.getvalue(),
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 413
    assert response.json()["detail"] == (
        "Dataset exceeds the configured row limit."
    )
    _assert_upload_artifacts_cleaned(tmp_path)


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
