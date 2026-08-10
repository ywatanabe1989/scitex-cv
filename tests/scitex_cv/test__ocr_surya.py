#!/usr/bin/env python3
"""Tests for scitex_cv._ocr_surya (Surya-2 OCR via llama-server).

Every test here runs against real behaviour — no mocks, no monkeypatch
(PA-306). The normalization, the layout-only detector and the reading's
validator are pure and need no server. The transport-failure test points at a
genuinely closed port, which is a real unreachable endpoint rather than a
simulated one. The end-to-end path is gated behind a server actually running.
"""

import json
import os
import socket

import numpy as np
import pytest

from scitex_cv._ocr_surya import (
    SURYA_PAGE_HEIGHT,
    SuryaReading,
    is_layout_only,
    normalize_page,
    ocr_surya,
)


def _closed_port() -> int:
    """Bind a port, learn its number, release it — so nothing listens there."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


@pytest.fixture
def a4_page():
    """A page already at the working resolution (1241x1755, A4-shaped)."""
    return np.full((1755, 1241, 3), 255, dtype=np.uint8)


@pytest.fixture
def soft_scan():
    """A page at the size the real ScanSnap output arrived in (753x1073)."""
    return np.full((1073, 753, 3), 255, dtype=np.uint8)


@pytest.fixture
def layout_only_body():
    """The exact shape the model returns when it does not read a page."""
    return json.dumps(
        [
            {"label": "Section-Header", "bbox": "342 103 610 136", "count": 10},
            {"label": "Text", "bbox": "322 189 664 218", "count": 20},
        ]
    )


class TestNormalizePage:
    def test_undersized_page_is_scaled_up_to_the_working_height(self, soft_scan):
        # Arrange
        source = soft_scan
        # Act
        normalized = normalize_page(source)
        # Assert
        assert normalized.shape[0] == SURYA_PAGE_HEIGHT

    def test_oversized_page_is_scaled_down_not_left_large(self):
        # Arrange — twice the working resolution; a soft scan measurably broke
        # when upscaled past it, so this must shrink rather than pass through.
        source = np.full((3510, 2482, 3), 255, dtype=np.uint8)
        # Act
        normalized = normalize_page(source)
        # Assert
        assert normalized.shape[0] == SURYA_PAGE_HEIGHT

    def test_page_already_at_the_working_height_is_returned_untouched(self, a4_page):
        # Arrange
        source = a4_page
        # Act
        normalized = normalize_page(source)
        # Assert
        assert normalized is source

    def test_aspect_ratio_is_preserved(self, soft_scan):
        # Arrange
        source_ratio = soft_scan.shape[1] / soft_scan.shape[0]
        # Act
        normalized = normalize_page(soft_scan)
        # Assert
        assert normalized.shape[1] / normalized.shape[0] == pytest.approx(
            source_ratio, abs=0.01
        )

    def test_degenerate_image_raises_value_error(self):
        # Arrange
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        # Act
        ctx = pytest.raises(ValueError)
        # Assert
        with ctx:
            normalize_page(empty)


class TestIsLayoutOnly:
    def test_block_geometry_response_is_detected(self, layout_only_body):
        # Arrange
        body = layout_only_body
        # Act
        detected = is_layout_only(body)
        # Assert
        assert detected is True

    def test_html_text_response_is_not_layout_only(self):
        # Arrange
        body = '<div data-bbox="106 625 431 644"><p>登記の申請をします</p></div>'
        # Act
        detected = is_layout_only(body)
        # Assert
        assert detected is False

    def test_malformed_json_array_is_not_layout_only(self):
        # Arrange
        body = "[this is not json"
        # Act
        detected = is_layout_only(body)
        # Assert
        assert detected is False

    def test_empty_json_array_is_not_layout_only(self):
        # Arrange
        body = "[]"
        # Act
        detected = is_layout_only(body)
        # Assert
        assert detected is False


class TestSuryaReadingValidator:
    def test_unknown_status_is_rejected_where_it_is_built(self):
        # Arrange
        fields = dict(
            text=None,
            body="",
            source_size=(1, 1),
            sent_size=(1, 1),
            image_tokens=None,
        )
        # Act
        ctx = pytest.raises(ValueError, match="status")
        # Assert
        with ctx:
            SuryaReading(status="probably-fine", **fields)

    def test_text_status_without_characters_is_rejected(self):
        # Arrange
        fields = dict(
            body="", source_size=(1, 1), sent_size=(1, 1), image_tokens=None
        )
        # Act
        ctx = pytest.raises(ValueError, match="non-empty text")
        # Assert
        with ctx:
            SuryaReading(status="text", text=None, **fields)

    def test_layout_only_reading_reports_read_false(self, layout_only_body):
        # Arrange
        reading = SuryaReading(
            status="layout-only",
            text=None,
            body=layout_only_body,
            source_size=(753, 1073),
            sent_size=(1232, 1755),
            image_tokens=2201,
        )
        # Act
        was_read = reading.read
        # Assert
        assert was_read is False

    def test_layout_only_text_is_none_not_empty_string(self, layout_only_body):
        # Arrange
        reading = SuryaReading(
            status="layout-only",
            text=None,
            body=layout_only_body,
            source_size=(753, 1073),
            sent_size=(1232, 1755),
            image_tokens=2201,
        )
        # Act
        recovered = reading.text
        # Assert
        assert recovered is None


class TestDispatch:
    def test_integer_input_raises_type_error(self):
        # Arrange
        bad_input = 12345
        # Act
        ctx = pytest.raises(TypeError, match="int")
        # Assert
        with ctx:
            ocr_surya(bad_input)


class TestTransportFailure:
    def test_unreachable_endpoint_raises_connection_error(self, a4_page):
        # Arrange — a real port with nothing listening on it.
        endpoint = f"http://127.0.0.1:{_closed_port()}/v1/chat/completions"
        # Act
        ctx = pytest.raises(ConnectionError)
        # Assert
        with ctx:
            ocr_surya(a4_page, endpoint=endpoint, timeout=5.0)

    def test_connection_error_names_the_relaunch_command(self, a4_page):
        # Arrange
        endpoint = f"http://127.0.0.1:{_closed_port()}/v1/chat/completions"
        # Act
        ctx = pytest.raises(ConnectionError, match="llama-server")
        # Assert
        with ctx:
            ocr_surya(a4_page, endpoint=endpoint, timeout=5.0)


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__), "-v"])

# EOF
