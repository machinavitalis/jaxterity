# Jaxterity — Numerical Tolerance Policy

This is the human-readable policy for the numerical tolerances used across the
test suite (PATTERNS.md §6.3: no magic numbers in tests — every comparison goes
through `jaxterity.testing.tolerances`).

## Authority and sync

The **programmatic source** is `jaxterity/testing/tolerances.py` (it ships in
the wheel, so tests can read it regardless of working directory). The table
below is generated from that module, and
`test/determinism/test_determinism.py::test_doc_table_matches_code` parses this
file and asserts it matches the code — so the doc and the code cannot drift.
**To change a tolerance, edit `tolerances.py` and regenerate this table.**

## How the values are derived

Tolerances are computed from three factor tables (not hand-typed rows), so the
policy stays internally consistent:

```
atol = base_atol[quantity] * dtype_atol_factor[dtype] * solver_factor[solver]
rtol = dtype_rtol[dtype] * solver_factor[solver]
```

- **Quantity** sets the absolute floor by physical scale (a contact force in N
  is many orders larger than a joint angle in rad).
- **dtype** loosens the absolute floor for `float32` (~8 fewer decimal digits
  than `float64`) and sets the relative floor above each dtype's epsilon.
- **Solver** scales both: adaptive `dopri5` is the reference (×1); fixed-step
  `rk4` (×5) and stiff, dissipative `bdf` (×10) accumulate more error.

> **Status: initial defaults.** These are grounded in machine epsilon and
> solver order, *not yet* validated against real reference trajectories. Once
> `Robot.to_diagram()` lands and the conservation/gradient suites
> produce measured error floors, the factors in `tolerances.py` are tightened to
> those floors and this table is regenerated.

## Table

| quantity | solver | dtype | atol | rtol |
|----------|--------|-------|------|------|
| joint_position | rk4 | float32 | 5e-04 | 5e-04 |
| joint_position | rk4 | float64 | 5e-09 | 5e-06 |
| joint_position | dopri5 | float32 | 1e-04 | 1e-04 |
| joint_position | dopri5 | float64 | 1e-09 | 1e-06 |
| joint_position | bdf | float32 | 1e-03 | 1e-03 |
| joint_position | bdf | float64 | 1e-08 | 1e-05 |
| joint_velocity | rk4 | float32 | 5e-03 | 5e-04 |
| joint_velocity | rk4 | float64 | 5e-08 | 5e-06 |
| joint_velocity | dopri5 | float32 | 1e-03 | 1e-04 |
| joint_velocity | dopri5 | float64 | 1e-08 | 1e-06 |
| joint_velocity | bdf | float32 | 1e-02 | 1e-03 |
| joint_velocity | bdf | float64 | 1e-07 | 1e-05 |
| contact_force | rk4 | float32 | 5e-01 | 5e-04 |
| contact_force | rk4 | float64 | 5e-06 | 5e-06 |
| contact_force | dopri5 | float32 | 1e-01 | 1e-04 |
| contact_force | dopri5 | float64 | 1e-06 | 1e-06 |
| contact_force | bdf | float32 | 1e+00 | 1e-03 |
| contact_force | bdf | float64 | 1e-05 | 1e-05 |
| actuator_torque | rk4 | float32 | 5e-02 | 5e-04 |
| actuator_torque | rk4 | float64 | 5e-07 | 5e-06 |
| actuator_torque | dopri5 | float32 | 1e-02 | 1e-04 |
| actuator_torque | dopri5 | float64 | 1e-07 | 1e-06 |
| actuator_torque | bdf | float32 | 1e-01 | 1e-03 |
| actuator_torque | bdf | float64 | 1e-06 | 1e-05 |
| actuator_voltage | rk4 | float32 | 5e-02 | 5e-04 |
| actuator_voltage | rk4 | float64 | 5e-07 | 5e-06 |
| actuator_voltage | dopri5 | float32 | 1e-02 | 1e-04 |
| actuator_voltage | dopri5 | float64 | 1e-07 | 1e-06 |
| actuator_voltage | bdf | float32 | 1e-01 | 1e-03 |
| actuator_voltage | bdf | float64 | 1e-06 | 1e-05 |
| sensor_reading | rk4 | float32 | 5e-02 | 5e-04 |
| sensor_reading | rk4 | float64 | 5e-07 | 5e-06 |
| sensor_reading | dopri5 | float32 | 1e-02 | 1e-04 |
| sensor_reading | dopri5 | float64 | 1e-07 | 1e-06 |
| sensor_reading | bdf | float32 | 1e-01 | 1e-03 |
| sensor_reading | bdf | float64 | 1e-06 | 1e-05 |
| motor_temperature | rk4 | float32 | 5e+00 | 5e-04 |
| motor_temperature | rk4 | float64 | 5e-05 | 5e-06 |
| motor_temperature | dopri5 | float32 | 1e+00 | 1e-04 |
| motor_temperature | dopri5 | float64 | 1e-05 | 1e-06 |
| motor_temperature | bdf | float32 | 1e+01 | 1e-03 |
| motor_temperature | bdf | float64 | 1e-04 | 1e-05 |
| bus_voltage | rk4 | float32 | 5e-02 | 5e-04 |
| bus_voltage | rk4 | float64 | 5e-07 | 5e-06 |
| bus_voltage | dopri5 | float32 | 1e-02 | 1e-04 |
| bus_voltage | dopri5 | float64 | 1e-07 | 1e-06 |
| bus_voltage | bdf | float32 | 1e-01 | 1e-03 |
| bus_voltage | bdf | float64 | 1e-06 | 1e-05 |

## Gradient-agreement tolerances

A separate, looser tolerance governs how closely an **autodiff gradient**
(forward- or reverse-mode) must match a **central finite difference** (used by
the gradient-correctness tests and the forward-mode sysid primitive).
These are not derived from the table above: the comparison is bounded by
finite-difference *truncation* error (O(h²)), not by the autodiff — the gradient
itself is exact — so the bound is solver-independent. The same tolerance also
bounds forward-mode-vs-reverse-mode cross-checks (which agree far tighter). Read
them via `tolerances.gradient_rtol(dtype)` / `gradient_atol(dtype)`.

| dtype | gradient_atol | gradient_rtol |
|-------|---------------|---------------|
| float64 | 1e-04 | 5e-03 |
| float32 | 1e-02 | 5e-02 |
