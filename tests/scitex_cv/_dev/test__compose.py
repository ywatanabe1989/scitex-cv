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


def test_cv_dev_module_exposes_compose_attribute():
    # Arrange
    module = cv_mod
    # Act
    has_compose = hasattr(module, "compose")
    # Assert
    assert has_compose


def test_cv_dev_title_card_submodule_is_importable():
    # Arrange
    import importlib

    target_name = "scitex_cv._dev._title_card"
    # Act
    submodule = importlib.import_module(target_name)
    # Assert
    assert submodule is not None
