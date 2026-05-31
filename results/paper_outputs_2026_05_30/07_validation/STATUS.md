# 07 Validation — Status

## VROOM out-of-sample validation: PENDING

The full VROOM re-routing of the NEW optimized schedules (α=1.343 model,
fixed bugs) has NOT yet been run. The prior VROOM validation
(tab_validation_per_pp.csv, 3.05% MAPE) was computed on the OLD α=1.0
optimization output and is therefore not directly comparable.

## What IS validated (model-intrinsic, no VROOM needed):

* **GroupKFold-CV on training pool** (04_model/tab_cv_battery.csv):
  Hybrid α=1.343 achieves 2.96% MAPE out-of-sample (group=PLZ, 5 folds).
  This is genuine out-of-sample validation against VROOM ground-truth in
  the 2733-sample training pool.

## To complete this section, run:

```
python scripts/paper_vroom_full_sweep.py   # ~8-12h overnight
```

This routes the chosen schedules at each (P, share) operating point through
VROOM + Valhalla and compares ML-predicted vs VROOM-actual cost on the
optimised tours (out-of-sample, since these schedule sizes/mixes were not
in the training pool).
