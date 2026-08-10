#!/usr/bin/env python3
"""Tests for scitex_cv._document._sidecar (on-disk result + resume policy).

No mocks, no monkeypatch (PA-306): sidecars are written to and read from real
files in tmp_path, and the corrupt/newer-version cases are real files with
real contents rather than simulated failures.
"""

import json
import os

import pytest

from scitex_cv._document import (
    SIDECAR_SUFFIX,
    SIDECAR_VERSION,
    DocumentReading,
    PageReading,
    ladder_fingerprint,
    read_sidecar,
    sidecar_path_for,
    write_sidecar,
)
from scitex_cv._document._sidecar import page_from_payload, reusable


@pytest.fixture
def a_reading():
    return DocumentReading(
        path="/tmp/doc.pdf",
        page_count=2,
        pages=(
            PageReading(
                index=0,
                status="text",
                text="hello",
                source="ocr",
                rendering=(1770, "cubic"),
                image_tokens=2201,
                attempts=1,
            ),
            PageReading(
                index=1,
                status="unreadable",
                text=None,
                source="ocr",
                rendering=(1740, "area"),
                image_tokens=2201,
                attempts=6,
                detail="all 6 rung(s) returned layout-only",
            ),
        ),
        ladder=((1770, "cubic"), (1740, "area")),
    )


class TestSidecarLocation:
    def test_sidecar_path_is_adjacent_and_stable(self, tmp_path):
        # Arrange
        document = tmp_path / "scan.pdf"
        # Act
        sidecar = sidecar_path_for(document)
        # Assert
        assert sidecar == tmp_path / ("scan.pdf" + SIDECAR_SUFFIX)


class TestSidecarRoundTrip:
    def test_written_sidecar_is_valid_json_without_importing_us(
        self, tmp_path, a_reading
    ):
        # Arrange — the consumer must read a result without calling Python.
        sidecar = tmp_path / "doc.json"
        write_sidecar(sidecar, a_reading)
        # Act
        with open(sidecar, encoding="utf-8") as handle:
            payload = json.load(handle)
        # Assert
        assert payload["pages"][0]["text"] == "hello"

    def test_sidecar_records_the_ladder_it_was_produced_under(
        self, tmp_path, a_reading
    ):
        # Arrange
        sidecar = tmp_path / "doc.json"
        write_sidecar(sidecar, a_reading)
        # Act
        payload = read_sidecar(sidecar)
        # Assert
        assert payload["ladder_fingerprint"] == ladder_fingerprint(a_reading.ladder)

    def test_no_partial_file_survives_a_completed_write(self, tmp_path, a_reading):
        # Arrange
        sidecar = tmp_path / "doc.json"
        # Act
        write_sidecar(sidecar, a_reading)
        # Assert — the atomic rename must leave nothing behind.
        assert not (tmp_path / "doc.json.partial").exists()

    def test_absent_sidecar_reads_as_no_prior_result(self, tmp_path):
        # Arrange
        missing = tmp_path / "nothing.json"
        # Act
        payload = read_sidecar(missing)
        # Assert
        assert payload is None

    def test_corrupt_sidecar_reads_as_no_prior_result(self, tmp_path):
        # Arrange — truncation is a reason to redo work, not to crash.
        sidecar = tmp_path / "bad.json"
        sidecar.write_text('{"sidecar_version": 1, "pages": [', encoding="utf-8")
        # Act
        payload = read_sidecar(sidecar)
        # Assert
        assert payload is None

    def test_sidecar_from_a_newer_writer_is_refused_not_guessed_at(self, tmp_path):
        # Arrange
        sidecar = tmp_path / "future.json"
        sidecar.write_text(
            json.dumps({"sidecar_version": SIDECAR_VERSION + 1}), encoding="utf-8"
        )
        # Act
        ctx = pytest.raises(ValueError, match="sidecar_version")
        # Assert
        with ctx:
            read_sidecar(sidecar)

    def test_malformed_page_entry_yields_none_rather_than_raising(self):
        # Arrange — one bad entry must invalidate that page only.
        entry = {"index": 0, "status": "text", "text": None}
        # Act
        page = page_from_payload(entry)
        # Assert
        assert page is None


class TestResumePolicy:
    def test_text_page_is_carried_forward(self, a_reading):
        # Arrange
        page = a_reading.pages[0]
        # Act
        carried = reusable(page, "same", "same")
        # Assert
        assert carried is True

    def test_error_page_is_always_retried(self):
        # Arrange — the failure described the run, not the page.
        page = PageReading(
            index=0,
            status="error",
            text=None,
            source=None,
            rendering=None,
            image_tokens=None,
            attempts=0,
            detail="endpoint unreachable",
        )
        # Act
        carried = reusable(page, "same", "same")
        # Assert
        assert carried is False

    def test_unreadable_is_kept_under_the_same_ladder(self, a_reading):
        # Arrange
        page = a_reading.pages[1]
        # Act
        carried = reusable(page, "same", "same")
        # Assert
        assert carried is True

    def test_unreadable_is_retried_when_the_ladder_changed(self, a_reading):
        # Arrange — a stale negative must not outlive its evidence.
        page = a_reading.pages[1]
        # Act
        carried = reusable(page, "old-ladder", "new-ladder")
        # Assert
        assert carried is False


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
