# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""The attestation handle: a versioned content hash of a calibrated robot
(ADR-008, CONTEXT.md invariant 5).

The handle is the provenance key Jaxility consumes. It must be stable across
machines, sensitive to anything that changes deployment behaviour (model
structure, fitted parameters), insensitive to irrelevant edits (whitespace), and
versioned so an algorithm change cannot silently collide with old handles.

``ALGORITHM_VERSION`` is folded into the hash; bumping it is a major-version
release (PATTERNS.md §8.2). BLAKE3 is the digest (ADR-008).

Version 3 digests the canonical bytes (:mod:`jaxterity.diagnostics.canonical`)
of one JSON document::

    {"version": 3, "mjcf": <whitespace-collapsed model>,
     "params": {name: float64 array}, ...extra}

so a handle can be recomputed from the printed document
(:func:`handle_document`) by any implementation of the canonical form, and
nothing in the hashed bytes is a Python ``repr``. Version 2 concatenated the
parts with separator bytes and ``str()`` of each extra; its code path is kept
so a handle recorded under it still verifies (:func:`verify_handle`,
:func:`matching_version`). New handles are always the current version.

The handle string is the bare hex digest (64 characters); prefix it with
``blake3:`` to obtain the portable handle form the canonical-bytes contract
defines.
"""

from __future__ import annotations

from typing import Callable, Mapping

import blake3
import numpy as np

from .canonical import canonical_handle

#: Increment on ANY change to the handle computation. Folded into the digest.
#: v2: hash parameters in fixed little-endian byte order (cross-machine stability).
#: v3: hash the canonical bytes of one JSON document (no separators, no repr).
ALGORITHM_VERSION = 3

#: Versions a recorded handle may carry, newest first; what verification tries.
SUPPORTED_VERSIONS = (3, 2)

_RESERVED_KEYS = frozenset({"version", "mjcf", "params"})


def _canonical(text: str) -> str:
    """Whitespace-collapsed form of a model string (ADR-008 canonicalization)."""
    return " ".join(text.split())


def handle_document(
    *,
    model_description: str,
    parameters: Mapping[str, object],
    extra: Mapping[str, object] | None = None,
    version: int | None = None,
) -> dict[str, object]:
    """The JSON document a version-3 (or later) handle digests.

    Args
    ----
    model_description : str
        The canonical model source (the lowered MJCF); whitespace-collapsed.
    parameters : Mapping[str, object]
        The bound parameter values by canonical path. Each becomes a float64
        array, so the canonical form records its dtype, shape and values.
    extra : Mapping[str, object] | None
        Further provenance, spliced in at the top level as JSON values (a
        list, a string, a number, a bool) — never a ``repr`` string. The
        keys ``version``, ``mjcf`` and ``params`` are reserved.
    version : int | None
        The algorithm version to record; the current one by default.

    Returns
    -------
    dict[str, object]
        ``{"version", "mjcf", "params", **extra}``; the handle is the hex
        part of ``canonical_handle(document)``.
    """
    if version is None:
        version = ALGORITHM_VERSION
    if version < 3:
        raise ValueError(f"version {version} has no handle document (v3 onwards)")
    extras = dict(extra or {})
    clash = _RESERVED_KEYS & set(extras)
    if clash:
        raise ValueError(
            f"extra keys {sorted(clash)} are reserved in a handle document"
        )
    params = {
        str(key): np.ascontiguousarray(np.asarray(parameters[key], dtype="<f8"))
        for key in parameters
    }
    return {
        "version": int(version),
        "mjcf": _canonical(model_description),
        "params": params,
        **extras,
    }


def compute_handle(
    *,
    model_description: str,
    parameters: Mapping[str, object],
    extra: Mapping[str, object] | None = None,
    version: int | None = None,
) -> str:
    """Compute the attestation handle.

    Args
    ----
    model_description : str
        The canonical model source (the lowered MJCF). Canonicalised
        (whitespace-collapsed) before hashing, so cosmetic edits don't change
        the handle.
    parameters : Mapping[str, object]
        The bound parameter values, keyed by canonical path (e.g. ``"bob.mass"``).
        Hashed as float64 arrays, so byte order never matters.
    extra : Mapping[str, object] | None
        Additional provenance (block/recipe versions, telemetry hashes, …).
        JSON values under version 3; strings under version 2.
    version : int | None
        The algorithm version; :data:`ALGORITHM_VERSION` by default. ``2``
        runs the legacy computation so an old handle can be checked.

    Returns
    -------
    str
        The BLAKE3 hex digest. Deterministic across machines for identical input.
    """
    if version is None:
        version = ALGORITHM_VERSION
    if version == 2:
        return _compute_handle_v2(
            model_description=model_description, parameters=parameters, extra=extra
        )
    document = handle_document(
        model_description=model_description,
        parameters=parameters,
        extra=extra,
        version=version,
    )
    return canonical_handle(document, algorithm="blake3").split(":", 1)[1]


def _compute_handle_v2(
    *,
    model_description: str,
    parameters: Mapping[str, object],
    extra: Mapping[str, object] | None,
) -> str:
    """The version-2 computation, byte for byte, for verifying old handles."""
    h = blake3.blake3()
    h.update(b"jaxterity-attestation-v2")
    h.update(_canonical(model_description).encode())
    for key in sorted(parameters):
        h.update(b"\x00")
        h.update(key.encode())
        arr = np.ascontiguousarray(np.asarray(parameters[key], dtype="<f8"))
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    for key in sorted(extra or {}):
        h.update(b"\x01")
        h.update(key.encode())
        h.update(str((extra or {})[key]).encode())
    return h.hexdigest()


def matching_version(expected: str, compute: Callable[[int], str]) -> int | None:
    """The version in :data:`SUPPORTED_VERSIONS` whose handle is ``expected``.

    ``compute(version)`` returns the handle under that version. Tries the
    newest version first; returns ``None`` when no supported version
    reproduces ``expected``.
    """
    for version in SUPPORTED_VERSIONS:
        if compute(version) == expected:
            return version
    return None


def verify_handle(
    expected: str,
    *,
    model_description: str,
    parameters: Mapping[str, object],
    extra: Mapping[str, object] | None = None,
    legacy_extra: Mapping[str, str] | None = None,
) -> int | None:
    """The algorithm version under which ``expected`` is the handle of these
    inputs, or ``None``.

    A handle recorded under version 2 keeps verifying after the bump to 3.
    ``legacy_extra`` is the string-valued ``extra`` the version-2 computation
    used; when omitted, ``extra`` is stringified with ``str`` for that attempt.
    """

    def compute(version: int) -> str:
        if version >= 3:
            chosen: Mapping[str, object] | None = extra
        elif legacy_extra is not None:
            chosen = legacy_extra
        else:
            chosen = {key: str(value) for key, value in (extra or {}).items()}
        return compute_handle(
            model_description=model_description,
            parameters=parameters,
            extra=chosen,
            version=version,
        )

    return matching_version(expected, compute)


__all__ = [
    "ALGORITHM_VERSION",
    "SUPPORTED_VERSIONS",
    "compute_handle",
    "handle_document",
    "matching_version",
    "verify_handle",
]
