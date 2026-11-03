# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""The shared canonical-bytes test vectors.

``canonical_hash_vectors.json`` is copied verbatim between every implementation
of the canonical byte form; this test runs it
against :mod:`jaxterity.diagnostics.canonical`. A vector that fails here means
the byte form moved, and every handle computed elsewhere would stop matching.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from jaxterity.diagnostics.canonical import canonical_dumps, canonical_handle

VECTORS_PATH = Path(__file__).with_name("canonical_hash_vectors.json")
VECTORS = json.loads(VECTORS_PATH.read_text(encoding="utf-8"))


def _decode(value: Any) -> Any:
    """Turn a vector's JSON input into the Python value the encoder sees."""
    if isinstance(value, dict):
        if set(value) == {"$bytes"}:
            return bytes.fromhex(value["$bytes"])
        if set(value) == {"$float"}:
            return float(value["$float"])
        if set(value) == {"$array"}:
            spec = value["$array"]
            values = [_decode(item) for item in spec["values"]]
            return np.asarray(values, dtype=spec["dtype"]).reshape(spec["shape"])
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


def _names(entries: list[dict[str, Any]]) -> list[str]:
    return [entry["name"] for entry in entries]


def test_vectors_file_is_pure_ascii() -> None:
    """Non-ASCII text is escaped so no editor can renormalise the file."""
    VECTORS_PATH.read_bytes().decode("ascii")
    assert VECTORS["canonical_form_version"] == 1


@pytest.mark.parametrize("vector", VECTORS["vectors"], ids=_names(VECTORS["vectors"]))
def test_vector_blake3(vector: dict[str, Any]) -> None:
    value = _decode(vector["input"])
    assert canonical_dumps(value).decode("utf-8") == vector["canonical"]
    assert canonical_handle(value) == vector["handle"]


@pytest.mark.parametrize("vector", VECTORS["vectors"], ids=_names(VECTORS["vectors"]))
def test_vector_blake2b_256(vector: dict[str, Any]) -> None:
    value = _decode(vector["input"])
    expected = vector.get("canonical_blake2b_256", vector["canonical"])
    assert canonical_dumps(value, algorithm="blake2b-256").decode("utf-8") == expected
    assert (
        canonical_handle(value, algorithm="blake2b-256") == vector["handle_blake2b_256"]
    )


@pytest.mark.parametrize("vector", VECTORS["rejected"], ids=_names(VECTORS["rejected"]))
def test_rejected_input_raises(vector: dict[str, Any]) -> None:
    with pytest.raises((ValueError, TypeError)):
        canonical_dumps(_decode(vector["input"]))
