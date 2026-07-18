# EMPS external validation — recorded results

Outcome of `test/sysid/external/test_emps_dc_motor.py` on the EMPS dataset
(`DATA_EMPS.mat` estimation set, `DATA_EMPS_PULSES.mat` held-out set). Regenerate
by fetching the data and running `pytest -m external_data`.

## Parameter recovery (estimation set)

The rig's model is `F = M*a + Fv*v + Fc*sgn(v) + offset`. Our DC-motor recipe
fits `V = Ks*sgn(v) + Kv*v + Ka*a` (no offset term), with `Ka↔M`, `Kv↔Fv`,
`Ks↔Fc`. Both fitters run on the same real 1 kHz telemetry (position
Butterworth-filtered at 100 Hz, central-differenced to velocity).

| Quantity | Paper's IDIM-LS (reproduced) | Our `sysid.actuator` | Agreement |
|---|---|---|---|
| Mass `M` / `Ka` (kg) | 95.085 | 95.151 | 0.07 % |
| Viscous `Fv` / `Kv` (N·s/m) | 204.668 | 204.989 | 0.16 % |
| Coulomb `Fc` / `Ks` (N) | 20.281 | 20.243 | 0.19 % |
| offset (N) | −3.170 | — (recipe has none) | n/a |

The reproduced IDIM-LS values match the published EMPS reference
(M ≈ 95.1 kg, Fv ≈ 204.7, Fc ≈ 20.3; Janot/Gautier/Brunot 2019), confirming the
data + preprocessing are correct. Our recipe lands within **0.2 %** of that
independent fit on all three physical parameters — despite carrying no offset
term — from real hardware data. A model-form or sign error in the recipe would
not survive this.

## Held-out prediction (pulses validation set)

Using the estimation-set fit to predict the *pulses* trace (never seen; a
different excitation with force swings of ±300 N):

- **RMS error: 7.94 N**
- **Relative error: 13.6 %**

The residual is dominated by the recipe's missing constant-offset term
(the rig's ≈ −3.2 N `offset`) and mild friction asymmetry — a known,
documented limitation, not a fit failure. The recipe generalizes to an
independent real trajectory it was not fit on.

## Scope

This validates the **DC-motor recipe** only. The Servo / BLDC / SEA LM recipes
have no equivalent open real dataset and still rely on in-stack oracles plus
(eventually) our own bench capture — see `../../../KNOWN_GAPS.md`.
