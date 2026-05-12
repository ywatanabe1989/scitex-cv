"""Smoke tests for scitex_cv._dev (compose + title-card helpers).

Originally ported from the umbrella `scitex_dev.cv` test; the
module now lives at `scitex_cv._dev` (this package's private
`_dev/` subtree). `matplotlib` is a `[dev]` extra here, so guard
the import per PA-303.
"""

import pytest

pytest.importorskip("numpy")
pytest.importorskip("matplotlib")

import scitex_cv._dev as cv_mod  # noqa: E402


def test_cv_module_loads():
    assert hasattr(cv_mod, "compose")


def test_cv_title_card_importable():
    from scitex_cv._dev import _title_card

    assert _title_card is not None
