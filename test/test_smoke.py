# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Smoke test: the package imports and its engine dependency resolves.

This is the trivial passing test — proof that ``pip install -e .`` produced an
importable package and that Jaxonomy (the engine Jaxterity is built on) is
available. Real tests arrive as the suite grows.
"""

import jaxterity


def test_package_imports_and_has_version():
    """Jaxterity imports and exposes a non-empty version string."""
    assert isinstance(jaxterity.__version__, str)
    assert jaxterity.__version__


def test_jaxonomy_is_importable():
    """The engine dependency is importable (the package boundary holds)."""
    import jaxonomy

    assert hasattr(jaxonomy, "__version__")


def test_subpackages_are_importable():
    """Every declared subpackage imports (the skeleton is wired up)."""
    import importlib

    for name in (
        "robot",
        "actuators",
        "sensors",
        "sysid",
        "wbc",
        "policies",
        "diagnostics",
        "data",
        "mcp",
        "cli",
        "zoo",
        "testing",
    ):
        importlib.import_module(f"jaxterity.{name}")
