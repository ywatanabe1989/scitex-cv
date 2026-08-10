#!/usr/bin/env python3
"""Importing scitex-cv must not eagerly import cv2.

The ecosystem aggregator discovers scitex-cv's `scitex_dev.system_deps`
provider by importing `scitex_cv._system_deps` — which imports the
`scitex_cv` package. If that pulled in cv2 (and its OS shared libs
libxcb/libgl/libglib), discovery would fail in the very build env that
hasn't installed those libs yet. A fresh subprocess proves the package
imports without cv2 landing in sys.modules.
"""

import subprocess
import sys


def test_importing_package_does_not_import_cv2():
    # Arrange
    code = "import scitex_cv, sys; sys.exit(1 if 'cv2' in sys.modules else 0)"
    # Act
    result = subprocess.run([sys.executable, "-c", code])
    # Assert
    assert result.returncode == 0


def test_importing_system_deps_provider_does_not_import_cv2():
    # Arrange
    code = (
        "import scitex_cv._system_deps as m, sys; "
        "sys.exit(1 if 'cv2' in sys.modules else 0)"
    )
    # Act
    result = subprocess.run([sys.executable, "-c", code])
    # Assert
    assert result.returncode == 0

# EOF
