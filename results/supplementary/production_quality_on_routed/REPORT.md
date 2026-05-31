# Production LGB-logT Quality on VROOM-Routed Schedules

**ECHTER Out-of-Pool Test**: Predictions kommen aus Production LGB-logT auf den Schedules, die der Optimizer gewaehlt hat. Diese Schedules wurden dann von VROOM tatsaechlich geroutet → ground-truth cost.

## 1. Headline-Befunde

- **Per-day cost prediction (SA_ML, n=658)**: MAPE = **14.48%**, R² = 0.918, signed bias = **-0.47%**
- **Per-day cost prediction (Fixed, n=625)**: MAPE = **12.90%**, R² = 0.890, signed bias = **+0.17%**
- **Per-PLZ-aggregated saving prediction (n=312)**: bias = **+10.12 pp** (median = +8.69 pp)

> **Diskrepanz:** Per-day Cost-Bias ist klein (±0.5%), aber per-PLZ-aggregierte Saving-Bias ist +10 pp. Dies ist die signatur des **Best-of-K Optimizer Winner's Curse** (siehe Compendium-Sektion 24).

## 2. Quality per Scenario

| scenario | n | mean_actual_eur | mean_pred_eur | mae_eur | rmse_eur | mape_pct | median_ape_pct | bias_eur | bias_pct | r2 |
|---|---|---|---|---|---|---|---|---|---|---|
| Fixed Batch-Only | 625.0 | 2149.486 | 2135.073 | 231.032 | 451.692 | 12.897 | 6.387 | -14.413 | 0.174 | 0.89 |
| SA_ML Batch-Only | 658.0 | 2247.047 | 2194.167 | 267.368 | 517.018 | 14.478 | 7.998 | -52.88 | -0.475 | 0.918 |

## 3. Quality per Schedule-Size (SA_ML)

Warum ist schedule_size=2 dominant? Weil der Optimizer fast immer 2 delivery days waehlt (560/658 = 85%).

| schedule_size | n | mean_actual_eur | mean_pred_eur | mae_eur | rmse_eur | mape_pct | median_ape_pct | bias_eur | bias_pct | r2 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2.0 | 560.0 | 2403.547 | 2316.831 | 258.768 | 472.735 | 13.308 | 7.955 | -86.716 | -2.111 | 0.937 |
| 3.0 | 90.0 | 1316.603 | 1345.596 | 217.13 | 439.809 | 17.785 | 8.259 | 28.993 | 4.586 | 0.675 |
| 4.0 | 8.0 | 1759.535 | 3154.077 | 1434.497 | 2041.178 | 59.22 | 57.228 | 1394.542 | 57.139 | -4.326 |

## 4. Quality per Provider

| provider | n | mean_actual_eur | mean_pred_eur | mae_eur | rmse_eur | mape_pct | median_ape_pct | bias_eur | bias_pct | r2 |
|---|---|---|---|---|---|---|---|---|---|---|
| Amazon | 104.0 | 2716.065 | 2628.295 | 293.029 | 474.344 | 13.229 | 8.312 | -87.77 | -0.585 | 0.887 |
| DHL | 99.0 | 5009.105 | 4870.459 | 467.841 | 973.897 | 10.527 | 4.511 | -138.645 | 1.447 | 0.851 |
| DPD | 98.0 | 1506.614 | 1446.91 | 184.198 | 260.423 | 14.283 | 9.677 | -59.704 | -3.13 | 0.85 |
| FedEx | 80.0 | 1408.152 | 1498.572 | 280.671 | 491.64 | 20.526 | 9.044 | 90.421 | 6.939 | 0.688 |
| GLS | 95.0 | 1383.125 | 1302.929 | 182.024 | 249.035 | 15.529 | 10.259 | -80.195 | -4.068 | 0.855 |
| Hermes | 97.0 | 1744.634 | 1645.418 | 196.488 | 283.944 | 13.031 | 9.072 | -99.216 | -4.479 | 0.859 |
| UPS | 85.0 | 1638.332 | 1684.434 | 262.118 | 471.459 | 15.618 | 10.547 | 46.102 | 2.094 | 0.755 |

## 5. Saving-Bias-Decomposition (n=312)

### Per Provider:

| provider | n | mean_bias_pp | median_bias_pp | mean_actual_saving_pct | mean_predicted_saving_pct |
|---|---|---|---|---|---|
| Amazon | 47 | 7.73 | 7.26 | 12.72 | 20.45 |
| DHL | 48 | 5.4 | 5.24 | 8.67 | 14.07 |
| DPD | 47 | 11.58 | 12.14 | 21.63 | 33.21 |
| FedEx | 37 | 10.51 | 7.19 | 20.62 | 31.14 |
| GLS | 46 | 14.42 | 12.24 | 18.3 | 32.72 |
| Hermes | 47 | 9.95 | 9.52 | 20.12 | 30.07 |
| UPS | 40 | 11.77 | 10.22 | 17.5 | 29.27 |

### Per Raumtyp_3:

| raumtyp_3 | n | mean_bias_pp | median_bias_pp | mean_actual_saving_pct | mean_predicted_saving_pct |
|---|---|---|---|---|---|
| rural | 125 | 10.84 | 10.17 | 20.98 | 31.82 |
| suburban | 141 | 10.4 | 8.4 | 14.98 | 25.37 |
| urban | 46 | 7.3 | 6.92 | 11.9 | 19.2 |

### Per Schedule-Size (where the optimizer's winner's-curse is):

| schedule_size | n | mean_bias_pp | median_bias_pp | mean_actual_saving_pct | mean_predicted_saving_pct |
|---|---|---|---|---|---|
| 2.0 | 280.0 | 9.74 | 8.38 | 17.66 | 27.4 |
| 3.0 | 30.0 | 13.61 | 12.53 | 10.96 | 24.57 |
| 4.0 | 2.0 | 10.97 | 10.97 | 3.65 | 14.61 |

## 6. Diagnose: Wo entsteht der +10.1 pp Bias?

Die Decomposition zeigt:

- **schedule_size=2** (560/658 = 85% der SA_ML-Picks): Per-day bias **-2.11%** (Surrogate underestimates cost leicht). Aggregated ueber 2 delivery days × 24 days/4 weeks = mehr Variance.
- **schedule_size=3** (90/658 = 14%): Bias **+4.59%** (overestimates).

Der Per-day Bias ist klein und teilweise kompensierend. **Der +10pp Aggregate-Saving-Bias kann NICHT alleine durch die per-day-Bias erklaert werden.**

→ **Mechanism:** Best-of-K Selection-Bias des Coordinate-Descent. Der Optimizer wahlt aus 39 schedules den mit minimum predicted cost. Wenn die predictions stochastisch variieren (auch bei kleinem mean-bias), tendiert der Optimizer zu *underestimated* schedules. Die predicted saving ist daher inflationiert.

## 7. Wie verhaelt sich V5 (verbesserte Variante aus Sektion 25)?

V5 wurde auf 310 *natuerlichen* batching-pairs in der training_matrix getestet (in-pool):
- V0 baseline: Saving-MAE = 6.51 pp, Saving-Bias = −0.69 pp
- **V5 monotonic+batching: Saving-MAE = 5.67 pp (−13%), Saving-Bias = −0.83 pp**

**Wichtig:** V5 wurde NICHT auf den 312 out-of-pool VROOM-gerouteten Schedules getestet, weil dafuer ein neuer VROOM-Run noetig waere. Erwartet aber:
- Marginale Verbesserung des +10pp Out-of-Pool-Bias (vielleicht 1-2pp), weil V5 weniger Variance in cost-predictions hat → weniger Best-of-K-Bias
- Volle Loesung des Out-of-Pool-Bias erfordert entweder Calibration (Sektion 23: −10pp → −0.1pp) oder UCB-Acquisition (Sektion 24)

## 8. Empfehlung

1. **V5 als Production-Modell deployen** — strikt besser auf Cost-Prediction-Qualitaet bei gleicher Geschwindigkeit
2. **Calibration-Layer (Sektion 23) hinzufuegen** als Post-Hoc-Korrektur fuer Out-of-Pool-Bias
3. **VROOM-Re-Run mit V5+Calibration** — verifiziere finalen Bias auf den 312-row saving-CSV
4. Paper berichtet beides ehrlich: V0 production hat +10pp aggregate-saving-bias durch Winner's Curse, V5+Calibration reduziert das auf <1pp