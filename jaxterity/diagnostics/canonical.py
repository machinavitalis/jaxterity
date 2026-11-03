# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Canonical bytes: the one byte form under every content hash.

The attestation handle (:mod:`jaxterity.diagnostics.attestation`) digests
the canonical bytes of a JSON document produced by :func:`canonical_dumps`.
The byte form is a contract shared across the stack (the deployment
compiler keeps the reference text of it); the shared test vectors in
``test/diagnostics/canonical_hash_vectors.json`` pin it byte for byte and
are copied verbatim between implementations.

The rules, in one screen:

* UTF-8, compact JSON (no whitespace), object keys sorted by Unicode
  code point at every depth.
* Strings are NFC-normalised. Only ``"``, ``\\`` and U+0000–U+001F are
  escaped (short forms for ``\\b \\f \\n \\r \\t``, ``\\u00xx`` for the
  rest); everything else is raw UTF-8. A lone surrogate is an error.
* Object keys must be strings; two keys equal after NFC are an error.
* ``true`` / ``false`` / ``null``. Integers are JSON integers of any
  size. Booleans are not integers.
* Floats are JSON **strings** holding the shortest decimal that
  round-trips to the same binary64 (CPython ``repr``): ``"1.5"``,
  ``"-0.0"``, ``"1e+300"``. NaN and ±Infinity are errors.
* Bytes are ``{"$b16": "<lowercase hex>"}``.
* Numpy arrays are ``{"$nd": {"dtype": "<f8", "shape": [...], ...}}``
  with ``"values"`` (flat, C order) when the array has at most
  :data:`ARRAY_INLINE_LIMIT` finite bool/int/uint/float elements, else
  ``"digest"``: the handle of the little-endian C-order bytes. Numpy
  scalars are plain scalars.
* A Pydantic model is its ``model_dump(mode="python")``.
* Anything else is a ``TypeError``: ``repr()`` is never hashed.

A handle is ``"<algorithm>:<lowercase hex>"`` over the canonical bytes;
the algorithm is BLAKE3 (32-byte digest). ``blake2b-256`` (``hashlib``)
is the sanctioned fallback where BLAKE3 cannot be a dependency; the
bytes under it are identical.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import unicodedata
from typing import Any

CANONICAL_FORM_VERSION = 1
"""Version of the byte form. Any change to it is a breaking change."""

DEFAULT_ALGORITHM = "blake3"
"""Digest behind :func:`canonical_handle` unless the caller says otherwise."""

ALGORITHMS = ("blake3", "blake2b-256")
"""Digests a handle may name, each producing 32 bytes."""

BYTES_KEY = "$b16"
"""Single key of the object that encodes a ``bytes`` value."""

ARRAY_KEY = "$nd"
"""Single key of the object that encodes a numpy array."""

ARRAY_INLINE_LIMIT = 64
"""Largest element count an array is inlined at; above it, its bytes are digested."""


def digest(data: bytes, *, algorithm: str = DEFAULT_ALGORITHM) -> bytes:
    """The 32-byte digest of ``data`` under ``algorithm``."""
    if algorithm == "blake3":
        import blake3  # lazy: a host of this module may not ship it

        return blake3.blake3(data).digest()
    if algorithm == "blake2b-256":
        return hashlib.blake2b(data, digest_size=32).digest()
    raise ValueError(
        f"unknown digest algorithm {algorithm!r}; expected one of {ALGORITHMS}"
    )


def handle(data: bytes, *, algorithm: str = DEFAULT_ALGORITHM) -> str:
    """``"<algorithm>:<hex>"`` for already-canonical ``data``."""
    return f"{algorithm}:{digest(data, algorithm=algorithm).hex()}"


def canonical_dumps(value: Any, *, algorithm: str = DEFAULT_ALGORITHM) -> bytes:
    """Return the canonical UTF-8 bytes of ``value``.

    Equal values yield byte-identical output on every host, Python
    version and numpy version; that is the property every content hash
    rests on. ``algorithm`` only matters when ``value`` holds an array
    too large to inline: its bytes are digested with that algorithm.

    Raises
    ------
    ValueError
        NaN or infinite float, lone surrogate, duplicate key after NFC
        normalisation, unknown algorithm.
    TypeError
        Any value outside the documented value model, including a
        non-string object key and an array of object / string / complex
        dtype.
    """
    coerced = _coerce(value, algorithm)
    text = json.dumps(
        coerced,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(
            "canonical bytes cannot encode a lone surrogate code point"
        ) from exc


def canonical_handle(value: Any, *, algorithm: str = DEFAULT_ALGORITHM) -> str:
    """``"<algorithm>:<hex>"`` over :func:`canonical_dumps` of ``value``."""
    return handle(canonical_dumps(value, algorithm=algorithm), algorithm=algorithm)


def coerce(value: Any, *, algorithm: str = DEFAULT_ALGORITHM) -> Any:
    """The JSON value model form of ``value``: what :func:`canonical_dumps`
    serialises, before serialisation.

    Plain JSON (``json.dumps`` accepts it) with floats already strings and
    bytes and arrays already in their tagged forms, so a document can be
    printed and inspected. Idempotent: a document assembled from coerced
    parts canonicalises to the same bytes as one assembled from the raw
    values.
    """
    return _coerce(value, algorithm)


def _string(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _coerce(value: Any, algorithm: str) -> Any:
    """Recursively map ``value`` onto the JSON value model.

    Order matters: ``bool`` before ``int`` (a subclass), numpy scalars
    and arrays before the generic containers (an array iterates).
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError(
                "canonical bytes have no encoding for NaN / Infinity; use a "
                "string form the schema documents at the field level"
            )
        # ``float(...)`` strips numpy's float64 subclass, whose repr is
        # ``np.float64(1.0)`` on numpy >= 2.
        return repr(float(value))
    if isinstance(value, str):
        return _string(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {BYTES_KEY: bytes(value).hex()}
    np = sys.modules.get("numpy")
    if np is not None:
        if isinstance(value, np.generic):
            return _coerce(value.item(), algorithm)
        if isinstance(value, np.ndarray):
            return _array(value, algorithm, np)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        # Pydantic v2; mode="python" keeps bytes as bytes and tuples as
        # tuples (mode="json" would already have stringified them).
        return _coerce(dump(mode="python"), algorithm)
    if isinstance(value, dict):
        return _object(value, algorithm)
    if isinstance(value, (list, tuple)):
        return [_coerce(item, algorithm) for item in value]
    raise TypeError(
        f"canonical_dumps cannot encode value of type {type(value).__name__!r}: "
        f"{value!r}. Add an explicit schema field or convert at the call site."
    )


def _object(mapping: dict, algorithm: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, item in mapping.items():
        if not isinstance(key, str):
            raise TypeError(
                f"canonical bytes require string object keys; got {key!r} "
                f"of type {type(key).__name__!r}"
            )
        normalised = _string(key)
        if normalised in out:
            raise ValueError(
                f"duplicate object key after NFC normalisation: {normalised!r}"
            )
        out[normalised] = _coerce(item, algorithm)
    return out


def _array(array: Any, algorithm: str, np: Any) -> dict[str, Any]:
    kind = array.dtype.kind
    if kind not in "biufc":
        raise TypeError(
            f"canonical bytes cannot encode an array of dtype {array.dtype!s}; "
            "only bool, integer, float and complex arrays are supported"
        )
    little = array.dtype.newbyteorder("<")
    contiguous = np.ascontiguousarray(array.astype(little, copy=False))
    document: dict[str, Any] = {
        "dtype": little.str,
        "shape": [int(dim) for dim in array.shape],
    }
    inline = (
        kind in "biuf"
        and array.size <= ARRAY_INLINE_LIMIT
        and (kind != "f" or bool(np.isfinite(array).all()))
    )
    if inline:
        document["values"] = [
            _coerce(item, algorithm) for item in contiguous.ravel().tolist()
        ]
    else:
        document["digest"] = handle(contiguous.tobytes(), algorithm=algorithm)
    return {ARRAY_KEY: document}


__all__ = [
    "ALGORITHMS",
    "ARRAY_INLINE_LIMIT",
    "ARRAY_KEY",
    "BYTES_KEY",
    "CANONICAL_FORM_VERSION",
    "DEFAULT_ALGORITHM",
    "canonical_dumps",
    "canonical_handle",
    "coerce",
    "digest",
    "handle",
]
