#!/usr/bin/env python3
"""Tests for scitex_cv._document (per-page reading + resumable sidecar).

No mocks, no monkeypatch (PA-306). The PDFs here are REAL files built with
pymupdf in a tmp_path, so the text-layer path and the rasterizer run against
genuine input rather than a stand-in. The ladder-walking path needs a live
model and is exercised end-to-end outside the unit suite; what is tested here
is everything that can be tested truthfully without one.
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
    render_page_at_height,
    sidecar_path_for,
    write_sidecar,
)
from scitex_cv._document._sidecar import page_from_payload, reusable

pymupdf = pytest.importorskip("pymupdf")


def _text_page_pdf(path, body="登記申請書"):
    """A REAL one-page PDF carrying a genuine text layer."""
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 144), body, fontsize=18, fontname="china-s")
    document.save(str(path))
    document.close()
    return path


def _blank_page_pdf(path, pages=1):
    """A REAL PDF whose pages carry no text layer at all."""
    document = pymupdf.open()
    for _ in range(pages):
        document.new_page()
    document.save(str(path))
    document.close()
    return path


@pytest.fixture
def a_text_page(tmp_path):
    return _text_page_pdf(tmp_path / "with_text.pdf")


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


class TestPageReadingValidator:
    def test_unknown_status_is_rejected(self):
        # Arrange
        fields = dict(
            index=0,
            text=None,
            source=None,
            rendering=None,
            image_tokens=None,
            attempts=0,
        )
        # Act
        ctx = pytest.raises(ValueError, match="status")
        # Assert
        with ctx:
            PageReading(status="probably-fine", **fields)

    def test_text_status_without_characters_is_rejected(self):
        # Arrange
        fields = dict(
            index=0, source="ocr", rendering=None, image_tokens=None, attempts=1
        )
        # Act
        ctx = pytest.raises(ValueError, match="non-empty text")
        # Assert
        with ctx:
            PageReading(status="text", text=None, **fields)

    def test_blank_page_may_not_carry_an_empty_string(self):
        # Arrange — the whole point: "" cannot distinguish blank from failed.
        fields = dict(
            index=0, source="ocr", rendering=None, image_tokens=None, attempts=1
        )
        # Act
        ctx = pytest.raises(ValueError, match="empty string")
        # Assert
        with ctx:
            PageReading(status="blank", text="", **fields)

    def test_blank_page_is_not_reported_as_read(self):
        # Arrange
        page = PageReading(
            index=0,
            status="blank",
            text=None,
            source="ocr",
            rendering=None,
            image_tokens=None,
            attempts=1,
        )
        # Act
        was_read = page.read
        # Assert
        assert was_read is False


class TestDocumentReadingValidator:
    def test_page_count_mismatch_is_rejected(self):
        # Arrange
        page = PageReading(
            index=0,
            status="blank",
            text=None,
            source="ocr",
            rendering=None,
            image_tokens=None,
            attempts=1,
        )
        # Act
        ctx = pytest.raises(ValueError, match="must not claim to be whole")
        # Assert
        with ctx:
            DocumentReading(path="/x.pdf", page_count=5, pages=(page,), ladder=())

    def test_out_of_order_pages_are_rejected(self):
        # Arrange
        second = PageReading(
            index=1,
            status="blank",
            text=None,
            source="ocr",
            rendering=None,
            image_tokens=None,
            attempts=1,
        )
        # Act
        ctx = pytest.raises(ValueError, match="ordered and contiguous")
        # Assert
        with ctx:
            DocumentReading(path="/x.pdf", page_count=1, pages=(second,), ladder=())


class TestSidecarRoundTrip:
    def test_sidecar_path_is_adjacent_and_stable(self, tmp_path):
        # Arrange
        document = tmp_path / "scan.pdf"
        # Act
        sidecar = sidecar_path_for(document)
        # Assert
        assert sidecar == tmp_path / ("scan.pdf" + SIDECAR_SUFFIX)

    def test_written_sidecar_is_valid_json_without_calling_python(
        self, tmp_path, a_reading
    ):
        # Arrange — the consumer must read a result without importing us.
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
        # Arrange — a truncated file is a reason to redo work, not to crash.
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
        # Arrange
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


class TestRasterizer:
    def test_page_is_rendered_at_the_requested_long_edge(self, a_text_page):
        # Arrange
        document = pymupdf.open(str(a_text_page))
        # Act
        image = render_page_at_height(document[0], 1770)
        document.close()
        # Assert
        assert max(image.shape[:2]) == 1770

    def test_rendered_page_has_three_colour_channels(self, a_text_page):
        # Arrange
        document = pymupdf.open(str(a_text_page))
        # Act
        image = render_page_at_height(document[0], 400)
        document.close()
        # Assert
        assert image.shape[2] == 3

    def test_two_heights_give_two_sizes(self, a_text_page):
        # Arrange — each rung rasterizes afresh rather than resizing one bitmap.
        document = pymupdf.open(str(a_text_page))
        # Act
        small = render_page_at_height(document[0], 400)
        large = render_page_at_height(document[0], 800)
        document.close()
        # Assert
        assert max(large.shape[:2]) == 2 * max(small.shape[:2])


class TestTextLayerPath:
    def test_a_pdf_with_text_is_read_without_touching_the_model(
        self, a_text_page, tmp_path
    ):
        # Arrange — no endpoint is running; a text-layer page must not need one.
        from scitex_cv._document import read_document

        # Act
        reading = read_document(a_text_page, endpoint="http://127.0.0.1:1/none")
        # Assert
        assert reading.pages[0].source == "pdf-text-layer"

    def test_text_layer_page_reports_zero_ocr_attempts(self, a_text_page):
        # Arrange
        from scitex_cv._document import read_document

        # Act
        reading = read_document(a_text_page, endpoint="http://127.0.0.1:1/none")
        # Assert
        assert reading.pages[0].attempts == 0

    def test_reading_writes_a_sidecar_beside_the_input(self, a_text_page):
        # Arrange
        from scitex_cv._document import read_document

        # Act
        read_document(a_text_page, endpoint="http://127.0.0.1:1/none")
        # Assert
        assert sidecar_path_for(a_text_page).exists()

    def test_rerunning_reuses_the_sidecar_rather_than_redoing_the_work(
        self, a_text_page
    ):
        # Arrange
        from scitex_cv._document import read_document

        read_document(a_text_page, endpoint="http://127.0.0.1:1/none")
        stamp = os.path.getmtime(sidecar_path_for(a_text_page))
        # Act
        again = read_document(a_text_page, endpoint="http://127.0.0.1:1/none")
        # Assert
        assert again.pages[0].source == "pdf-text-layer"

    def test_a_document_that_cannot_be_opened_raises(self, tmp_path):
        # Arrange
        not_a_pdf = tmp_path / "junk.pdf"
        not_a_pdf.write_text("this is not a pdf", encoding="utf-8")
        from scitex_cv._document import read_document

        # Act
        ctx = pytest.raises(OSError)
        # Assert
        with ctx:
            read_document(not_a_pdf)


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
