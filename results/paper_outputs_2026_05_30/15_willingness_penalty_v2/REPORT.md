# Waiting-Penalty Sensitivity Report

**Mode**: v2 | **Output**: willingness_penalty_v2
**Method**: cost_objective = base_cost + PENALTY * n_parcels * avg_wait_days
**Schedules tested**: 39 (MAX_HOLDING_DAYS=3)
**Penalty grid**: [0.0, 0.25, 0.5, 1.0, 2.0, 4.0] €/parcel/day

## Grid

 penalty_eur_per_day  total_base_cost_eur  total_penalty_cost_eur  total_parcels_per_week  avg_wait_days  n_plz  n_size_2  n_size_3  n_size_4  n_size_5  n_size_6
                0.00         1.726528e+06                     0.0                 1263130       0.723519    312       114        83        69        27        19
                0.25         1.798266e+06                 47123.5                 1263130       0.149228    312        10        51        53        59       139
                0.50         1.843209e+06                 28035.5                 1263130       0.044391    312         0        25        23        39       225
                1.00         1.871175e+06                 16112.0                 1263130       0.012756    312         0         3        14        18       277
                2.00         1.891170e+06                  2870.0                 1263130       0.001136    312         0         0         1         5       306
                4.00         1.894455e+06                     0.0                 1263130       0.000000    312         0         0         0         0       312

## Break-even: 50% wechseln zu daily delivery bei **€0.50/parcel/day**

## Figures

- `figP1_schedule_mix_vs_penalty.png`
- `figP2_avg_wait_vs_penalty.png`
- `figP3_pareto_cost_wait.png`
- `figP4_breakeven_curves.png`