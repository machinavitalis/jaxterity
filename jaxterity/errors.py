# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""The Jaxterity exception hierarchy (PATTERNS.md §4.2).

Library code raises one of these — never a bare ``ValueError`` /
``RuntimeError`` / ``AssertionError`` — so agents and tests can catch the
specific failure mode. Each message should name the offending object, the
violated invariant, and the likely fix (PATTERNS.md §4.1).

This module is introduced for the first typed error and is the prerequisite
for the domain errors raised later (``FrameError``, ``SchemaError``, and so
on).
"""

from __future__ import annotations


class JaxterityError(Exception):
    """Base class for every Jaxterity error."""


class FrameError(JaxterityError):
    """A coordinate-frame mismatch, or an unknown / unrelatable frame."""


class UnitError(JaxterityError):
    """A unit mismatch, or a non-SI value where SI is required."""


class CalibrationError(JaxterityError):
    """A system-identification recipe failed, or telemetry is incompatible."""


class AttestationError(JaxterityError):
    """An attestation-handle mismatch or invalidation."""


class SchemaError(JaxterityError):
    """A telemetry or model schema violation."""


class ZooError(JaxterityError):
    """A reference-robot-zoo invariant violation."""


class JaxterityWarning(UserWarning):
    """A non-fatal Jaxterity caveat (e.g. a sysid path with weak gradient).

    Library code uses this — never a bare ``UserWarning`` — so callers can
    filter Jaxterity's own warnings specifically. Emitted, for example, when a
    recipe is asked to fit a parameter whose identifiability is known to be
    weak (``sysid.contact(fit_friction=True)``).
    """


__all__ = [
    "JaxterityError",
    "FrameError",
    "UnitError",
    "CalibrationError",
    "AttestationError",
    "SchemaError",
    "ZooError",
    "JaxterityWarning",
]
