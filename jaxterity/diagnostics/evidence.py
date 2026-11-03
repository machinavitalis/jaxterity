# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Structured calibration evidence record — unsigned.

A calibration is only as trustworthy as the evidence behind it: how well it held
out, how tightly its parameters were identified, whether it recovered planted
truth. This module collects that evidence into one typed, stable-hashable
:class:`EvidenceRecord` **bound by hash to the unit's twin attestation handle**,
so a verifier can check "this evidence is for *this* unit version" and a
deployment manifest can reference the evidence by its content hash.

The record is **unsigned**. It is the open-source primitive the commercial layer
*seals*: cryptographic signing, key management, tamper-evidence, and any
compliance-framework mapping live above this repo (ATT-03 / ATT-05). What ships
here is the structure + a deterministic content hash.

Cross-repo: the hash :meth:`EvidenceRecord.content_hash` returns is the value a
**Jaxility** deployment manifest can carry to point at the calibration evidence
for an artifact — the same BLAKE3 / ``ALGORITHM_VERSION`` scheme as the
attestation handle (:mod:`jaxterity.diagnostics.attestation`), so the two are
verifiable with one implementation.
"""

from __future__ import annotations

import dataclasses
import json
import math

from . import attestation

#: Evidence-record schema version (bump on any field change that moves the hash).
SCHEMA_VERSION = "1.0"

_FloatMap = tuple[tuple[str, float], ...]
_VectorMap = tuple[tuple[str, tuple[float, ...]], ...]


def _float_map(d) -> _FloatMap:
    """Canonicalize a name→float mapping to a sorted, hashable tuple."""
    items = d.items() if isinstance(d, dict) else d
    return tuple(sorted((str(k), float(v)) for k, v in items))


def _vector_map(d) -> _VectorMap:
    """Canonicalize a name→vector mapping to a sorted, hashable tuple."""
    items = d.items() if isinstance(d, dict) else d
    return tuple(sorted((str(k), tuple(float(x) for x in v)) for k, v in items))


def _json_float(value: float) -> object:
    """A float the canonical form can carry: NaN / ±Inf as the documented
    strings ``"nan"`` / ``"inf"`` / ``"-inf"``."""
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


@dataclasses.dataclass(frozen=True)
class EvidenceRecord:
    """A hashable record of a calibration's validation evidence.

    Fields
    ------
    twin_handle : str
        The unit's twin attestation handle this evidence is for — the binding.
    recipe_name, recipe_version, telemetry_hash : str
        The fit's provenance (which recipe, which version, which telemetry).
    held_out_error : tuple[tuple[str, float], ...]
        Held-out validation error per quantity (e.g. ``("temperature_rms", …)``).
    uncertainty : tuple[tuple[str, tuple[float, ...]], ...]
        The per-leaf marginal variance, keyed by canonical path.
    recovery : tuple[tuple[str, float], ...]
        Planted-param recovery outcomes (e.g. per-param relative error), if a
        recovery check was run; empty otherwise.
    identifiability : tuple[tuple[str, float], ...]
        Identifiability outcomes (e.g. ``("rank", …)``, ``("condition_number",
        …)``), if a check was run; empty otherwise.
    schema_version : str
        The record schema version.
    """

    twin_handle: str
    recipe_name: str
    recipe_version: str
    telemetry_hash: str
    held_out_error: _FloatMap = ()
    uncertainty: _VectorMap = ()
    recovery: _FloatMap = ()
    identifiability: _FloatMap = ()
    schema_version: str = SCHEMA_VERSION

    def _extra(self, version: int) -> dict[str, object]:
        """The handle's ``extra`` under ``version``: JSON values from 3 on,
        the ``repr`` strings version 2 hashed."""
        recipe = f"{self.recipe_name}:{self.recipe_version}:{self.telemetry_hash}"
        if version == 2:
            return {
                "schema_version": self.schema_version,
                "twin_handle": self.twin_handle,
                "recipe": recipe,
                "held_out_error": repr(self.held_out_error),
                "uncertainty": repr(self.uncertainty),
                "recovery": repr(self.recovery),
                "identifiability": repr(self.identifiability),
            }
        return {
            "schema_version": self.schema_version,
            "twin_handle": self.twin_handle,
            "recipe": recipe,
            "held_out_error": [[k, _json_float(v)] for k, v in self.held_out_error],
            "uncertainty": [
                [k, [_json_float(x) for x in v]] for k, v in self.uncertainty
            ],
            "recovery": [[k, _json_float(v)] for k, v in self.recovery],
            "identifiability": [[k, _json_float(v)] for k, v in self.identifiability],
        }

    def content_hash(self, *, version: int | None = None) -> str:
        """The deterministic content hash binding this evidence to the unit.

        Reuses the attestation BLAKE3 scheme (:func:`compute_handle`) over a
        JSON document of every field — so identical evidence hashes identically
        across machines, and any change (including the ``twin_handle``) moves
        it. This is the value a Jaxility manifest references. ``version=2``
        recomputes the form recorded before the algorithm bump.

        Returns
        -------
        str
            The BLAKE3 hex digest (64 hex chars).
        """
        if version is None:
            version = attestation.ALGORITHM_VERSION
        return attestation.compute_handle(
            model_description="jaxterity-evidence-record",
            parameters={},
            extra=self._extra(version),
            version=version,
        )

    def verify_content_hash(self, expected: str) -> int | None:
        """The algorithm version under which ``expected`` is this record's
        content hash, or ``None``."""
        return attestation.matching_version(
            expected, lambda version: self.content_hash(version=version)
        )

    def to_dict(self) -> dict:
        """A JSON-serializable dict of the record (round-trips via ``from_dict``)."""
        return {
            "schema_version": self.schema_version,
            "twin_handle": self.twin_handle,
            "recipe_name": self.recipe_name,
            "recipe_version": self.recipe_version,
            "telemetry_hash": self.telemetry_hash,
            "held_out_error": [list(p) for p in self.held_out_error],
            "uncertainty": [[k, list(v)] for k, v in self.uncertainty],
            "recovery": [list(p) for p in self.recovery],
            "identifiability": [list(p) for p in self.identifiability],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EvidenceRecord":
        """Reconstruct a record from :meth:`to_dict` output."""
        return cls(
            twin_handle=d["twin_handle"],
            recipe_name=d["recipe_name"],
            recipe_version=d["recipe_version"],
            telemetry_hash=d["telemetry_hash"],
            held_out_error=_float_map(d.get("held_out_error", ())),
            uncertainty=_vector_map(d.get("uncertainty", ())),
            recovery=_float_map(d.get("recovery", ())),
            identifiability=_float_map(d.get("identifiability", ())),
            schema_version=d.get("schema_version", SCHEMA_VERSION),
        )

    def to_json(self) -> str:
        """Canonical JSON (sorted keys) of the record."""
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "EvidenceRecord":
        """Reconstruct a record from :meth:`to_json` output."""
        return cls.from_dict(json.loads(s))


def build_evidence(
    twin,
    *,
    held_out_error,
    recovery=(),
    identifiability=(),
) -> EvidenceRecord:
    """Build an :class:`EvidenceRecord` from a calibrated twin and its checks.

    Reads the twin's attestation handle (the binding), its fit uncertainty,
    and its latest off-body provenance record, and packages them with the
    caller-supplied held-out error and (optional) recovery / identifiability
    outcomes.

    Args
    ----
    twin : jaxterity.twin.Twin
        The calibrated unit the evidence is for.
    held_out_error : dict[str, float] or sequence of (str, float)
        Held-out validation error per quantity.
    recovery : dict[str, float] or sequence, optional
        Planted-param recovery outcomes.
    identifiability : dict[str, float] or sequence, optional
        Identifiability outcomes (rank / condition number / …).

    Returns
    -------
    EvidenceRecord
        Bound to ``twin.attestation_handle``.
    """
    prov = twin.provenance[-1] if twin.provenance else ("", "", "")
    return EvidenceRecord(
        twin_handle=twin.attestation_handle,
        recipe_name=prov[0],
        recipe_version=prov[1],
        telemetry_hash=prov[2],
        held_out_error=_float_map(held_out_error),
        uncertainty=tuple(twin.param_uncertainty),
        recovery=_float_map(recovery),
        identifiability=_float_map(identifiability),
    )


__all__ = ["EvidenceRecord", "build_evidence", "SCHEMA_VERSION"]
