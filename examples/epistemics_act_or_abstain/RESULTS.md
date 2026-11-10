> **SYNTHETIC PILOT - NOT THE SPRINT RESULT.** PREEXISTING: pipeline demonstration on synthetic data, generated before the sprint.

# Results (held-out)

| arm | loss | coverage | hit rate | n |
|---|---|---|---|---|
| A always_act | 0.522 | 1.000 | 0.478 | 10380 |
| B persistence + self-stated conf | 0.170 | 0.670 | 0.894 | 10380 |
| C evidence-derived conf (relative) (FAILS coverage floor) | 0.405 | 0.273 | 0.315 | 10380 |
| C' evidence-conformal (absolute, calibrated) | 0.313 | 0.300 | 0.656 | 10380 |

**Verdict (mechanical, from prereg.json): inconclusive (below coverage floor)**

- pre-registered margin: 0.0200; observed improvement of C over A: 0.1164, over B: -0.2355
- coverage floor: 0.3; cuts B/C: 0.57/0.25
- pre-registered at 2026-11-09T02:00:10.348443+00:00, git a89fe6b71e69, held-out seeds [201, 202, 203, 204, 205], warm-up seeds [8801, 8802, 8803]

**Verdict for C' (the arm that carries the sprint question; same rule, its own pre-registered margin 0.0200): premise falsified**

- observed improvement of C' over A: 0.2085, over B: -0.1434; cuts C/C': 0.25/0.68

C' constants (fixed before the pilot): W=50, k=10, alpha=0.2, outcome range 2.0.

## Dark rises error (pre-registered check)

Realized mean |error| of the ENGINE's forecast, by what the arm did with the case. Arm A never abstains (it is the `acted` row only). B has no dark concept (its `warming` = no self-stated confidence yet).

| arm | bucket | n | mean engine abs err |
|---|---|---|---|
| B persistence + self-stated conf | acted | 6955 | 0.308 |
| B persistence + self-stated conf | dark | 0 | - |
| B persistence + self-stated conf | warming | 0 | - |
| B persistence + self-stated conf | lowconf | 3425 | 0.574 |
| B persistence + self-stated conf | total | 10380 | |
| C evidence-derived conf (relative) | acted | 2836 | 0.452 |
| C evidence-derived conf (relative) | dark | 1730 | 0.007 |
| C evidence-derived conf (relative) | warming | 0 | - |
| C evidence-derived conf (relative) | lowconf | 5814 | 0.485 |
| C evidence-derived conf (relative) | total | 10380 | |
| C' evidence-conformal (absolute, calibrated) | acted | 3116 | 0.257 |
| C' evidence-conformal (absolute, calibrated) | dark | 1730 | 0.007 |
| C' evidence-conformal (absolute, calibrated) | warming | 325 | 0.491 |
| C' evidence-conformal (absolute, calibrated) | lowconf | 5209 | 0.603 |
| C' evidence-conformal (absolute, calibrated) | total | 10380 | |

Pre-registered check (arms C, C'): mean |error| on abstained-dark exceeds mean on acted by at least the pilot-set margin; needs >= 30 cases in both buckets.

- C evidence-derived conf (relative): effect (dark - acted) = -0.4456, margin = 0.0200, n_dark = 1730, n_acted = 2836 -> **dark rises error: NOT supported**
- C' evidence-conformal (absolute, calibrated): effect (dark - acted) = -0.2503, margin = 0.0301, n_dark = 1730, n_acted = 3116 -> **dark rises error: NOT supported**

### C' calibration

Fraction of cases whose realized engine |error| <= the predicted (1-alpha) quantile (target 0.80 if calibrated):

| cases | n | fraction within predicted quantile |
|---|---|---|
| acted | 3116 | 0.763 |
| all issued (acted or low-conf; excludes dark and warming) | 8325 | 0.797 |
