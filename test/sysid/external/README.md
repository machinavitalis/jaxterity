# External-hardware sysid validation (EMPS)

The one sysid check whose ground truth is a **real physical rig outside the
jax\*y stack**. Every other recipe oracle — round-trip, Radau, jaxonomy — is
in-stack, so a physics error shared across the stack could survive it. This one
uses real measured data, so it cannot.

## What it validates

The **DC-motor recipe** (`sysid.actuator` on `DCMotorWithFriction`, model
`V = Ks*sgn(v) + Kv*v + Ka*a`) against **EMPS** — the Electro-Mechanical
Positioning System benchmark (a DC motor driving a low-friction ball-screw
stage, logged at 1 kHz). The rig's physics `F = M*a + Fv*v + Fc*sgn(v) + offset`
is exactly the recipe's form (`Ka↔M`, `Kv↔Fv`, `Ks↔Fc`; the recipe has no offset
term). The test checks that our fit (a) matches the paper's own identification
method (IDIM-LS, reproduced independently of our code) on the estimation set,
and (b) predicts the held-out *pulses* validation trace. See `RESULTS.md` for
the numbers.

The DC-motor recipe is exactly the recipe with **no** in-stack oracle (the
Radau/jaxonomy work covered the Servo/BLDC/SEA LM recipes), so this fills the
untested one. It does **not** cover the LM recipes — no clean open real dataset
exists for a current-loop BLDC, a two-inertia SEA, or a position-setpoint servo;
those still need our own bench capture.

## Mode B: fetched, not vendored

The EMPS data is **not committed**. Its license is unclear — the
[benchmark page](https://www.nonlinearbenchmark.org/benchmarks/emps) carries only
a liability disclaimer and a cite-the-paper request, with no redistribution
grant — so jaxterity uses it transiently (like a human downloading it to check)
and never republishes it. The data lives in the gitignored `_emps_data/`.

Consequently the test is **opt-in**: it is marked `external_data`, excluded from
`scripts/check.sh` and CI, and *skips* unless the data is present.

## Running it

```
python test/sysid/external/fetch_emps.py     # download + verify (sha256-pinned)
pytest -m external_data                       # run the validation
```

`fetch_emps.py` pulls the ~2 MB zip from the benchmark's Google-Drive mirror and
checks each `.mat` against a pinned hash. If Google Drive serves a confirm page
instead of the file, download manually from the benchmark page and unzip
`DATA_EMPS.mat` / `DATA_EMPS_PULSES.mat` into `_emps_data/`.

## Citation

A. Janot, M. Gautier, M. Brunot, *Data Set and Reference Models of EMPS* (2019),
Nonlinear System Identification Benchmarks. Cite it if you use this data.
