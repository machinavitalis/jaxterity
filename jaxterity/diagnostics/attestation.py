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
"""

from __future__ import annotations

import blake3
import numpy as np

#: Increment on ANY change to the handle computation. Folded into the digest.
#: v2: hash parameters in fixed little-endian byte order (cross-machine stability).
ALGORITHM_VERSION = 2


def _canonical(text: str) -> str:
    """Whitespace-collapsed form of a model string (ADR-008 canonicalization)."""
    return " ".join(text.split())


def compute_handle(
    *,
    model_description: str,
    parameters: dict[str, object],
    extra: dict[str, str] | None = None,
) -> str:
    """Compute the attestation handle.

    Args
    ----
    model_description : str
        The canonical model source (the lowered MJCF). Canonicalised
        (whitespace-collapsed) before hashing, so cosmetic edits don't change
        the handle.
    parameters : dict[str, object]
        The bound parameter values, keyed by canonical path (e.g. ``"bob.mass"``).
        Hashed in sorted key order as float64 bytes.
    extra : dict[str, str] | None
        Additional provenance strings (e.g. block/recipe versions, telemetry
        hashes) hashed in sorted key order.

    Returns
    -------
    str
        The BLAKE3 hex digest. Deterministic across machines for identical input.
    """
    h = blake3.blake3()
    h.update(f"jaxterity-attestation-v{ALGORITHM_VERSION}".encode())
    h.update(_canonical(model_description).encode())
    for key in sorted(parameters):
        h.update(b"\x00")
        h.update(key.encode())
        # Fixed little-endian float64 bytes so the digest is identical across
        # machines of differing endianness (the value, of course, must match —
        # cross-*device* reproducibility of fitted values is OQ-4).
        arr = np.ascontiguousarray(np.asarray(parameters[key], dtype="<f8"))
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    for key in sorted(extra or {}):
        h.update(b"\x01")
        h.update(key.encode())
        h.update(str((extra or {})[key]).encode())
    return h.hexdigest()


__all__ = ["ALGORITHM_VERSION", "compute_handle"]
