# External-hardware validation

Two datasets, two different claims — see `RESULTS.md` for both sets of numbers.

---

## EMPS — the DC-motor recipe

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


---

## NAD SO-101 — model completeness against a labelled unmodelled mass

The second external anchor, and it tests something the EMPS check cannot:
not whether a *parameter* is right, but whether the **model-completeness metric**
(`jaxterity.sysid.hybrid`) actually detects physics the model does not carry.

### Why this dataset

NAD (Neural Actuation Dataset, Dou et al., RSS 2026) includes an SO-101 subset —
**the same arm this repo calibrates** — logged by other people on another unit,
with the payload deliberately stepped empty → 200 g → 300 g → 400 g → 500 g. Our
model has no payload term, so the payload is *labelled unmodelled physics*:
exactly the thing the metric claims to spot, with the label supplied by someone
who has no stake in our result.

Every prior test of that claim was in-house — synthetic captures where we injected
the extra term ourselves, or LK-0001 where nothing tells us what is missing.

### What it does

Fits the parametric parameters **once** on the empty-payload training set, holds
them fixed, and scores every payload's held-out test trajectory. The error rises
monotonically with a mass the model was never told about (r = 0.99 against the
gram values at a 10-step rollout), while the empty-payload fit sits at half an
encoder quantum — so the model does fit their data, and the rise is attributable.

### Mode B: fetched, not vendored

Same posture as EMPS, for the same reason: the upstream repository declares **no
license** (the GitHub API reports `license: null`), so there is no redistribution
grant. Only the SO-101 CSVs are fetched — 100 files, ~24 MB — not the ~169 MB
repository and not the Hugging Face mirror's camera recordings, neither of which
a dynamics model needs. Integrity is one pinned aggregate digest over the whole
subset rather than a hundred per-file hashes, so a missing or extra file fails
too.

```
python test/sysid/external/fetch_nad.py     # download + verify (sha256-pinned)
pytest -m external_data                      # run both validations
```

### Citation

Z. Dou, J. U. Onyemelukwe, H. Zhang, H. Zhang, M. Guo, Y. Tian, M. P. Lipiec,
J. Jacob, C. Liu, P. Y. Chen, Y. Ivanov, W. Matusik, *NeuralActuator: Neural
Actuation Modeling for Robot Dynamics and External Force Perception*, RSS 2026
([arXiv 2607.11734](https://arxiv.org/abs/2607.11734)). Cite it if you use this
data.
