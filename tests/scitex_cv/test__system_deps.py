#!/usr/bin/env python3
"""Tests for scitex_cv._system_deps (ecosystem system-deps provider).

scitex-cv uses opencv-python-headless (self-contained wheel), so it
declares no OS-level apt dependencies; the provider returns an empty list.
"""

from scitex_cv import _system_deps


class TestSystemDepsProvider:
    def test_apt_packages_is_empty_with_headless(self):
        # Arrange
        expected = []
        # Act
        actual = _system_deps.apt_packages()
        # Assert
        assert actual == expected

    def test_provide_returns_no_specs(self):
        # Arrange
        provide = _system_deps.provide
        # Act
        specs = provide()
        # Assert
        assert specs == []

    def test_provide_result_is_a_list(self):
        # Arrange
        provide = _system_deps.provide
        # Act
        result = provide()
        # Assert
        assert isinstance(result, list)

    def test_provider_name_is_the_owning_leaf(self):
        # Arrange
        expected = "scitex-cv"
        # Act
        actual = _system_deps.PROVIDER
        # Assert
        assert actual == expected

# EOF
