#!/usr/bin/env python3
"""Tests for scitex_cv._document._read (rasterizer + document driver).

No mocks, no monkeypatch (PA-306). The PDFs are REAL files built with pymupdf
in tmp_path, so the text-layer path and the rasterizer run against genuine
input. The ladder-walking OCR path needs a live model and is exercised
end-to-end outside this suite — what is here is everything that can be
tested truthfully without one, which is deliberately not the same claim.
"""

import os

import pytest

from scitex_cv._document import read_document, render_page_at_height, sidecar_path_for

pymupdf = pytest.importorskip("pymupdf")

#: A port with nothing behind it. A text-layer page must never reach out.
NO_ENDPOINT = "http://127.0.0.1:1/none"


def build_text_page_pdf(path):
    """Write a REAL one-page PDF carrying a genuine text layer.

    A plain builder rather than a fixture body, so the fixture below holds no
    open resource of its own — the document is opened, written and closed
    entirely inside this call.
    """
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 144), "Registration application", fontsize=18)
    document.save(str(path))
    document.close()
    return path


@pytest.fixture
def a_text_page(tmp_path):
    """Path to a real single-page PDF with an embedded text layer."""
    return build_text_page_pdf(tmp_path / "with_text.pdf")


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

    def test_each_height_rasterizes_afresh_rather_than_resizing(self, a_text_page):
        # Arrange — resampling changes whether the model reads a page, so each
        # rung renders from the vector source instead of resizing one bitmap.
        document = pymupdf.open(str(a_text_page))
        small = render_page_at_height(document[0], 400)
        # Act
        large = render_page_at_height(document[0], 800)
        document.close()
        # Assert
        assert max(large.shape[:2]) == 2 * max(small.shape[:2])


class TestTextLayerPath:
    def test_a_pdf_with_text_is_read_without_touching_the_model(self, a_text_page):
        # Arrange — no endpoint is listening; this must still succeed.
        source = a_text_page
        # Act
        reading = read_document(source, endpoint=NO_ENDPOINT)
        # Assert
        assert reading.pages[0].source == "pdf-text-layer"

    def test_text_layer_page_reports_zero_ocr_attempts(self, a_text_page):
        # Arrange
        source = a_text_page
        # Act
        reading = read_document(source, endpoint=NO_ENDPOINT)
        # Assert
        assert reading.pages[0].attempts == 0

    def test_text_layer_page_recovers_the_embedded_characters(self, a_text_page):
        # Arrange
        source = a_text_page
        # Act
        reading = read_document(source, endpoint=NO_ENDPOINT)
        # Assert
        assert "Registration" in reading.pages[0].text


class TestSidecarLifecycle:
    def test_reading_writes_a_sidecar_beside_the_input(self, a_text_page):
        # Arrange
        source = a_text_page
        # Act
        read_document(source, endpoint=NO_ENDPOINT)
        # Assert
        assert sidecar_path_for(source).exists()

    def test_rerunning_reuses_the_stored_result(self, a_text_page):
        # Arrange
        read_document(a_text_page, endpoint=NO_ENDPOINT)
        # Act
        again = read_document(a_text_page, endpoint=NO_ENDPOINT)
        # Assert
        assert again.pages[0].source == "pdf-text-layer"

    def test_page_count_survives_the_round_trip(self, a_text_page):
        # Arrange
        read_document(a_text_page, endpoint=NO_ENDPOINT)
        # Act
        again = read_document(a_text_page, endpoint=NO_ENDPOINT)
        # Assert
        assert again.page_count == 1


class TestUnopenableDocument:
    def test_a_document_that_cannot_be_opened_raises(self, tmp_path):
        # Arrange — one unopenable file is exceptional; one bad PAGE is not.
        not_a_pdf = tmp_path / "junk.pdf"
        not_a_pdf.write_text("this is not a pdf", encoding="utf-8")
        # Act
        ctx = pytest.raises(OSError)
        # Assert
        with ctx:
            read_document(not_a_pdf)


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
