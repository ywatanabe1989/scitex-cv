#!/usr/bin/env python3
"""Compile-only smoke test for examples/quickstart.py."""

import py_compile
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "quickstart.py"


def test_quickstart_example_file_exists_on_disk():
    # Arrange
    expected_path = EXAMPLE
    # Act
    is_file = expected_path.is_file()
    # Assert
    assert is_file, f"missing example: {expected_path}"


def test_quickstart_example_compiles_without_syntax_error():
    # Arrange
    source_path = str(EXAMPLE)
    # Act
    compiled = py_compile.compile(source_path, doraise=True)
    # Assert
    assert compiled is not None


# EOF
