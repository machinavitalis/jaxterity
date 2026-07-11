# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Telemetry loaders for CSV, Parquet, and HDF5.

Loaders convert at the boundary into the SI-unit, ``float64`` schema in
:mod:`jaxterity.data.schema` (PATTERNS §7.2). Where the source format carries
per-channel unit metadata (HDF5 dataset attributes, Parquet schema metadata),
the loader checks it against the schema's :class:`~jaxterity.data.schema.ChannelSpec`
and raises :class:`~jaxterity.errors.SchemaError` on a mismatch — catching the
"someone logged degrees" class of bug at ingestion.

Scope: these loaders cover :class:`~jaxterity.data.schema.JointTelemetry`
(the channel used by ``sysid.actuator``, the most common recipe). The other
schema models are defined; their loaders extend the same flat representation as
the recipes that consume them land.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Callable

import numpy as np

from ..errors import SchemaError
from .schema import JointTelemetry

_JOINT_CHANNELS = ("q", "qd", "tau")
_UNITS = {"q": "rad", "qd": "rad/s", "tau": "N*m"}


def _check_units(found: dict[str, str]) -> None:
    """Raise SchemaError if any provided channel unit is not the SI unit."""
    for channel, unit in found.items():
        expected = _UNITS.get(channel)
        if expected is not None and unit != expected:
            raise SchemaError(
                f"Telemetry channel {channel!r} is in {unit!r}, but JointTelemetry "
                f"requires SI {expected!r}. Convert at the source (e.g. deg→rad) "
                "before loading; Jaxterity's internal representation is SI only."
            )


def _stack(per_channel: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        k: np.ascontiguousarray(v, dtype=np.float64) for k, v in per_channel.items()
    }


# --- CSV -------------------------------------------------------------------
# Columns: ``time``, then ``<channel>.<joint>`` (e.g. ``q.shoulder``). Units are
# assumed SI (CSV carries no unit metadata); pass ``sample_rate_hz`` on load.


def save_csv(tel: JointTelemetry, path: str | pathlib.Path) -> None:
    """Write ``JointTelemetry`` to CSV (full float64 precision)."""
    import pandas as pd

    cols: dict[str, np.ndarray] = {"time": np.asarray(tel.time, dtype=np.float64)}
    for ch in _JOINT_CHANNELS:
        arr = np.asarray(getattr(tel, ch), dtype=np.float64)
        for j, name in enumerate(tel.joint_names):
            cols[f"{ch}.{name}"] = arr[:, j]
    # %.17g is round-trip-exact for IEEE-754 float64, so CSV->schema is bit-exact.
    pd.DataFrame(cols).to_csv(path, index=False, float_format="%.17g")


def load_csv(path: str | pathlib.Path, *, sample_rate_hz: float) -> JointTelemetry:
    """Load ``JointTelemetry`` from a CSV written by :func:`save_csv`."""
    import pandas as pd

    # ``round_trip`` is required for bit-exact float64 recovery — pandas'
    # default C float parser is fast but can be off by 1 ULP.
    df = pd.read_csv(path, float_precision="round_trip")
    if "time" not in df.columns:
        raise SchemaError(f"CSV {path} has no 'time' column.")
    joint_names: list[str] = []
    for col in df.columns:
        if col.startswith("q."):
            joint_names.append(col[2:])
    if not joint_names:
        raise SchemaError(
            f"CSV {path} has no 'q.<joint>' columns; cannot infer joints."
        )
    data = {}
    for ch in _JOINT_CHANNELS:
        data[ch] = np.stack(
            [df[f"{ch}.{j}"].to_numpy(dtype=np.float64) for j in joint_names], axis=1
        )
    return JointTelemetry(
        time=df["time"].to_numpy(dtype=np.float64),
        sample_rate_hz=float(sample_rate_hz),
        joint_names=tuple(joint_names),
        **_stack(data),
    )


# --- Parquet ---------------------------------------------------------------
# Self-contained: sample_rate, joint_names, and units travel in the Arrow
# schema metadata.


def save_parquet(tel: JointTelemetry, path: str | pathlib.Path) -> None:
    """Write ``JointTelemetry`` to Parquet with metadata in the schema."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    cols: dict[str, np.ndarray] = {"time": np.asarray(tel.time, dtype=np.float64)}
    for ch in _JOINT_CHANNELS:
        arr = np.asarray(getattr(tel, ch), dtype=np.float64)
        for j, name in enumerate(tel.joint_names):
            cols[f"{ch}.{name}"] = arr[:, j]
    table = pa.Table.from_pandas(pd.DataFrame(cols), preserve_index=False)
    meta = {
        b"sample_rate_hz": str(tel.sample_rate_hz).encode(),
        b"joint_names": json.dumps(list(tel.joint_names)).encode(),
        b"units": json.dumps(_UNITS).encode(),
    }
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), **meta})
    pq.write_table(table, path)


def load_parquet(path: str | pathlib.Path) -> JointTelemetry:
    """Load ``JointTelemetry`` from a Parquet file written by :func:`save_parquet`."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    meta = table.schema.metadata or {}
    if b"sample_rate_hz" not in meta:
        raise SchemaError(f"Parquet {path} is missing 'sample_rate_hz' metadata.")
    _check_units(json.loads(meta[b"units"].decode()))
    joint_names = json.loads(meta[b"joint_names"].decode())
    df = table.to_pandas()
    data = {
        ch: np.stack(
            [df[f"{ch}.{j}"].to_numpy(dtype=np.float64) for j in joint_names], axis=1
        )
        for ch in _JOINT_CHANNELS
    }
    return JointTelemetry(
        time=df["time"].to_numpy(dtype=np.float64),
        sample_rate_hz=float(meta[b"sample_rate_hz"].decode()),
        joint_names=tuple(joint_names),
        **_stack(data),
    )


# --- HDF5 ------------------------------------------------------------------
# Self-contained: one dataset per channel with ``unit`` attrs; sample_rate and
# joint_names as file attrs.


def save_hdf5(tel: JointTelemetry, path: str | pathlib.Path) -> None:
    """Write ``JointTelemetry`` to HDF5 with unit attributes per channel."""
    import h5py

    with h5py.File(path, "w") as f:
        f.attrs["sample_rate_hz"] = float(tel.sample_rate_hz)
        f.attrs["joint_names"] = list(tel.joint_names)
        f.create_dataset("time", data=np.asarray(tel.time, dtype=np.float64))
        for ch in _JOINT_CHANNELS:
            ds = f.create_dataset(
                ch, data=np.asarray(getattr(tel, ch), dtype=np.float64)
            )
            ds.attrs["unit"] = _UNITS[ch]


def load_hdf5(path: str | pathlib.Path) -> JointTelemetry:
    """Load ``JointTelemetry`` from HDF5; validates per-channel ``unit`` attrs."""
    import h5py

    with h5py.File(path, "r") as f:
        if "sample_rate_hz" not in f.attrs:
            raise SchemaError(f"HDF5 {path} is missing the 'sample_rate_hz' attr.")
        found_units = {
            ch: str(f[ch].attrs.get("unit", "")) for ch in _JOINT_CHANNELS if ch in f
        }
        _check_units(found_units)
        data: dict[str, Any] = {ch: f[ch][()] for ch in _JOINT_CHANNELS}
        return JointTelemetry(
            time=np.asarray(f["time"][()], dtype=np.float64),
            sample_rate_hz=float(f.attrs["sample_rate_hz"]),
            joint_names=tuple(str(n) for n in f.attrs["joint_names"]),
            **_stack(data),
        )


_LOADERS: dict[str, Callable[..., JointTelemetry]] = {
    ".csv": load_csv,
    ".parquet": load_parquet,
    ".pq": load_parquet,
    ".h5": load_hdf5,
    ".hdf5": load_hdf5,
}


def load_telemetry(path: str | pathlib.Path, **kwargs: Any) -> JointTelemetry:
    """Load joint telemetry, dispatching on the file extension.

    Args
    ----
    path : str | pathlib.Path
        ``.csv`` / ``.parquet`` / ``.pq`` / ``.h5`` / ``.hdf5``.
    **kwargs
        Forwarded to the format loader. CSV requires ``sample_rate_hz=`` (CSV
        carries no metadata); Parquet/HDF5 are self-contained.

    Returns
    -------
    JointTelemetry
        Validated telemetry in SI units.

    Raises
    ------
    SchemaError
        On an unknown extension, a missing column/attr, or a non-SI unit.
    """
    suffix = pathlib.Path(path).suffix.lower()
    loader = _LOADERS.get(suffix)
    if loader is None:
        raise SchemaError(
            f"Unsupported telemetry extension {suffix!r}. Expected one of "
            f"{sorted(_LOADERS)}."
        )
    return loader(path, **kwargs)


__all__ = [
    "load_telemetry",
    "save_csv",
    "load_csv",
    "save_parquet",
    "load_parquet",
    "save_hdf5",
    "load_hdf5",
]
