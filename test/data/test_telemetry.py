# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Telemetry schema + loader tests."""

from __future__ import annotations

import numpy as np
import pytest

from jaxterity.data import (
    ActuatorChirpTelemetry,
    ActuatorRampTelemetry,
    ActuatorStepTelemetry,
    BusVoltageTelemetry,
    JointTelemetry,
    SeriesElasticTelemetry,
    ThermalTelemetry,
    TransmissionTelemetry,
    load_telemetry,
    save_csv,
    save_hdf5,
    save_parquet,
)
from jaxterity.data.loaders import load_csv, load_hdf5, load_parquet
from jaxterity.errors import SchemaError


def _make(n: int = 50, seed: int = 0) -> JointTelemetry:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64) / 100.0  # 100 Hz
    return JointTelemetry(
        time=t,
        sample_rate_hz=100.0,
        joint_names=("shoulder", "elbow"),
        q=rng.standard_normal((n, 2)),
        qd=rng.standard_normal((n, 2)),
        tau=rng.standard_normal((n, 2)),
    )


# --- construction validation ----------------------------------------------


def test_time_must_be_float64():
    with pytest.raises(SchemaError, match="float64"):
        JointTelemetry(
            time=np.arange(3, dtype=np.float32),
            sample_rate_hz=100.0,
            joint_names=("a",),
            q=np.zeros((3, 1)),
            qd=np.zeros((3, 1)),
            tau=np.zeros((3, 1)),
        )


def test_time_must_be_monotonic():
    with pytest.raises(SchemaError, match="increasing"):
        JointTelemetry(
            time=np.array([0.0, 0.2, 0.1]),
            sample_rate_hz=10.0,
            joint_names=("a",),
            q=np.zeros((3, 1)),
            qd=np.zeros((3, 1)),
            tau=np.zeros((3, 1)),
        )


def test_channel_row_mismatch_raises():
    with pytest.raises(SchemaError, match="rows"):
        JointTelemetry(
            time=np.arange(3, dtype=np.float64),
            sample_rate_hz=10.0,
            joint_names=("a",),
            q=np.zeros((2, 1)),  # wrong length
            qd=np.zeros((3, 1)),
            tau=np.zeros((3, 1)),
        )


# --- hash ------------------------------------------------------------------


def test_hash_is_stable_and_sensitive():
    a, b = _make(seed=1), _make(seed=1)
    assert a.hash() == b.hash()
    assert len(a.hash()) == 32
    perturbed = _make(seed=2)
    assert a.hash() != perturbed.hash()


# --- round trips -----------------------------------------------------------


def _assert_same(a: JointTelemetry, b: JointTelemetry):
    assert a.joint_names == b.joint_names
    assert a.sample_rate_hz == b.sample_rate_hz
    for ch in ("time", "q", "qd", "tau"):
        assert (
            np.asarray(getattr(a, ch)).tobytes() == np.asarray(getattr(b, ch)).tobytes()
        )


def test_csv_roundtrip_bit_exact(tmp_path):
    jt = _make()
    p1, p2 = tmp_path / "a.csv", tmp_path / "b.csv"
    save_csv(jt, p1)
    jt2 = load_csv(p1, sample_rate_hz=jt.sample_rate_hz)
    _assert_same(jt, jt2)  # schema arrays bit-exact through CSV
    save_csv(jt2, p2)
    assert p1.read_bytes() == p2.read_bytes()  # CSV -> schema -> CSV byte-identical


def test_parquet_roundtrip(tmp_path):
    jt = _make()
    p = tmp_path / "t.parquet"
    save_parquet(jt, p)
    _assert_same(jt, load_telemetry(p))  # self-contained (dispatch on extension)


def test_hdf5_roundtrip(tmp_path):
    jt = _make()
    p = tmp_path / "t.h5"
    save_hdf5(jt, p)
    _assert_same(jt, load_telemetry(p))


# --- mis-unit rejection ----------------------------------------------------


def test_hdf5_rejects_misunit_channel(tmp_path):
    """A channel logged in the wrong unit is rejected at load with a clear error."""
    import h5py

    jt = _make()
    p = tmp_path / "bad.h5"
    save_hdf5(jt, p)
    with h5py.File(p, "a") as f:
        f["q"].attrs["unit"] = "deg"  # corrupt: degrees instead of radians
    with pytest.raises(SchemaError, match="rad"):
        load_hdf5(p)


def test_parquet_rejects_misunit_channel(tmp_path):
    import json

    import pyarrow.parquet as pq

    jt = _make()
    p = tmp_path / "bad.parquet"
    save_parquet(jt, p)
    table = pq.read_table(p)
    meta = dict(table.schema.metadata)
    meta[b"units"] = json.dumps({"q": "deg", "qd": "rad/s", "tau": "N*m"}).encode()
    pq.write_table(table.replace_schema_metadata(meta), p)
    with pytest.raises(SchemaError, match="rad"):
        load_parquet(p)


def test_unknown_extension_raises(tmp_path):
    (tmp_path / "x.txt").write_text("nope")
    with pytest.raises(SchemaError, match="Unsupported"):
        load_telemetry(tmp_path / "x.txt")


# --- validation report -----------------------------------------------------


def test_validate_flags_nan(tmp_path):
    rng = np.random.default_rng(0)
    n = 20
    q = rng.standard_normal((n, 1))
    q[5, 0] = np.nan
    jt = JointTelemetry(
        time=np.arange(n, dtype=np.float64) / 50.0,
        sample_rate_hz=50.0,
        joint_names=("a",),
        q=q,
        qd=np.zeros((n, 1)),
        tau=np.zeros((n, 1)),
    )
    report = jt.validate()
    assert not report.ok
    assert any("non-finite" in w for w in report.warnings)


def test_validate_flags_rate_drift():
    n = 20
    jt = JointTelemetry(
        time=np.arange(n, dtype=np.float64) / 10.0,  # actually 10 Hz
        sample_rate_hz=100.0,  # claimed 100 Hz -> drift
        joint_names=("a",),
        q=np.zeros((n, 1)),
        qd=np.zeros((n, 1)),
        tau=np.zeros((n, 1)),
    )
    report = jt.validate()
    assert any("sample interval" in w for w in report.warnings)


def test_clean_telemetry_validates_ok():
    assert _make().validate().ok


# --- ActuatorRampTelemetry ----------------------------------------


def _make_ramp(n: int = 50, seed: int = 0) -> ActuatorRampTelemetry:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64) / 200.0  # 200 Hz, WPILib SysId default
    return ActuatorRampTelemetry(
        time=t,
        sample_rate_hz=200.0,
        actuator_name="motor_0",
        command=rng.standard_normal(n),
        velocity=rng.standard_normal(n),
    )


def test_actuator_ramp_constructs_and_hashes():
    a, b = _make_ramp(seed=1), _make_ramp(seed=1)
    assert a.actuator_name == "motor_0"
    assert a.n_samples == 50
    assert a.hash() == b.hash()
    assert a.hash() != _make_ramp(seed=2).hash()


def test_actuator_ramp_channel_length_must_match_time():
    with pytest.raises(SchemaError, match="rows"):
        ActuatorRampTelemetry(
            time=np.arange(3, dtype=np.float64),
            sample_rate_hz=200.0,
            actuator_name="m",
            command=np.zeros(2),  # wrong length
            velocity=np.zeros(3),
        )


def test_actuator_ramp_channel_arrays_returns_command_and_velocity():
    arrays = _make_ramp().channel_arrays()
    assert set(arrays) == {"command", "velocity"}


# --- ActuatorStepTelemetry ----------------------------------------


def _make_step(n: int = 60, seed: int = 0) -> ActuatorStepTelemetry:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64) / 200.0
    return ActuatorStepTelemetry(
        time=t,
        sample_rate_hz=200.0,
        actuator_name="servo_0",
        setpoint=rng.standard_normal(n),
        position=rng.standard_normal(n),
    )


def test_actuator_step_constructs_and_hashes():
    a, b = _make_step(seed=1), _make_step(seed=1)
    assert a.actuator_name == "servo_0"
    assert a.n_samples == 60
    assert a.hash() == b.hash()
    assert a.hash() != _make_step(seed=2).hash()


def test_actuator_step_channel_arrays_returns_setpoint_and_position():
    arrays = _make_step().channel_arrays()
    assert set(arrays) == {"setpoint", "position"}


# --- ActuatorChirpTelemetry ---------------------------------------


def _make_chirp(n: int = 100, seed: int = 0) -> ActuatorChirpTelemetry:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64) / 1000.0  # 1 kHz, BLDC default
    return ActuatorChirpTelemetry(
        time=t,
        sample_rate_hz=1000.0,
        actuator_name="bldc_0",
        setpoint=rng.standard_normal(n),
        velocity=rng.standard_normal(n),
        current=rng.standard_normal(n),
    )


def test_actuator_chirp_constructs_and_hashes():
    a, b = _make_chirp(seed=1), _make_chirp(seed=1)
    assert a.actuator_name == "bldc_0"
    assert a.n_samples == 100
    assert a.hash() == b.hash()
    assert a.hash() != _make_chirp(seed=2).hash()


def test_actuator_chirp_channel_arrays_returns_all_three():
    arrays = _make_chirp().channel_arrays()
    assert set(arrays) == {"setpoint", "velocity", "current"}


# --- TransmissionTelemetry ----------------------------------------


def _make_transmission(n: int = 50, seed: int = 0) -> TransmissionTelemetry:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64) / 200.0
    return TransmissionTelemetry(
        time=t,
        sample_rate_hz=200.0,
        transmission_name="hd_0",
        tau_motor=rng.standard_normal(n),
        v_motor=rng.standard_normal(n),
        tau_joint=rng.standard_normal(n),
        v_joint=rng.standard_normal(n),
    )


def test_transmission_constructs_and_hashes():
    a, b = _make_transmission(seed=1), _make_transmission(seed=1)
    assert a.transmission_name == "hd_0"
    assert a.n_samples == 50
    assert a.hash() == b.hash()
    assert a.hash() != _make_transmission(seed=2).hash()


def test_transmission_channel_arrays_returns_all_four():
    arrays = _make_transmission().channel_arrays()
    assert set(arrays) == {"tau_motor", "v_motor", "tau_joint", "v_joint"}


# --- SeriesElasticTelemetry --------------------------------------


def _make_sea(n: int = 50, seed: int = 0) -> SeriesElasticTelemetry:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64) / 500.0  # 500 Hz, SEA default
    return SeriesElasticTelemetry(
        time=t,
        sample_rate_hz=500.0,
        actuator_name="sea_0",
        tau_input=rng.standard_normal(n),
        q_motor=rng.standard_normal(n),
        q_load=rng.standard_normal(n),
    )


def test_series_elastic_constructs_and_hashes():
    a, b = _make_sea(seed=1), _make_sea(seed=1)
    assert a.actuator_name == "sea_0"
    assert a.n_samples == 50
    assert a.hash() == b.hash()
    assert a.hash() != _make_sea(seed=2).hash()


def test_series_elastic_channel_arrays_returns_all_three():
    arrays = _make_sea().channel_arrays()
    assert set(arrays) == {"tau_input", "q_motor", "q_load"}


# --- ThermalTelemetry ---------------------------------------------


def _make_thermal(n: int = 50, n_servos: int = 2, seed: int = 0) -> ThermalTelemetry:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64) / 100.0
    return ThermalTelemetry(
        time=t,
        sample_rate_hz=100.0,
        servo_names=tuple(f"servo_{i}" for i in range(n_servos)),
        temperature=300.0 + rng.standard_normal((n, n_servos)),
    )


def test_thermal_constructs_and_hashes():
    a, b = _make_thermal(seed=1), _make_thermal(seed=1)
    assert a.n_samples == 50
    assert a.hash() == b.hash()
    assert a.hash() != _make_thermal(seed=2).hash()


def test_thermal_channel_arrays_returns_temperature():
    assert set(_make_thermal().channel_arrays()) == {"temperature"}


def test_thermal_column_count_must_match_servo_names():
    with pytest.raises(SchemaError, match="column per servo_name"):
        ThermalTelemetry(
            time=np.arange(10, dtype=np.float64) / 100.0,
            sample_rate_hz=100.0,
            servo_names=("a", "b", "c"),
            temperature=300.0 + np.zeros((10, 2)),  # 2 cols, 3 names
        )


# --- BusVoltageTelemetry ------------------------------------------


def _make_bus(
    n: int = 50, n_servos: int = 2, with_current: bool = False, seed: int = 0
):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64) / 100.0
    kwargs = dict(
        time=t,
        sample_rate_hz=100.0,
        voltage=12.0 - rng.uniform(0, 1, size=n),
        servo_names=tuple(f"servo_{i}" for i in range(n_servos)),
    )
    if with_current:
        kwargs["current"] = rng.uniform(0, 2, size=(n, n_servos))
    return BusVoltageTelemetry(**kwargs)


def test_bus_voltage_constructs_without_current():
    bus = _make_bus()
    assert bus.current is None
    assert set(bus.channel_arrays()) == {"voltage"}  # optional channel omitted


def test_bus_voltage_with_current_includes_both_channels():
    bus = _make_bus(with_current=True)
    assert set(bus.channel_arrays()) == {"voltage", "current"}


def test_bus_voltage_must_be_one_dimensional():
    with pytest.raises(SchemaError, match="1-D"):
        BusVoltageTelemetry(
            time=np.arange(10, dtype=np.float64) / 100.0,
            sample_rate_hz=100.0,
            voltage=np.zeros((10, 1)),  # 2-D — bus is a single shared rail
        )


def test_bus_voltage_current_column_count_must_match_servo_names():
    with pytest.raises(SchemaError, match="column per servo_name"):
        BusVoltageTelemetry(
            time=np.arange(10, dtype=np.float64) / 100.0,
            sample_rate_hz=100.0,
            voltage=np.zeros(10),
            servo_names=("a", "b", "c"),
            current=np.zeros((10, 2)),  # 2 cols, 3 names
        )
