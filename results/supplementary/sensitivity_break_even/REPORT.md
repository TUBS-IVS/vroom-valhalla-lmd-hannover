# Sensitivity- und Break-Even-Analyse

Auswertungs-Einheit: 312 Cluster × Provider Rows (36 unique Cluster × 7 Provider).

## 1. Break-Even-Punkte je Threshold × Feature × Raumtyp_3

Smallest feature value where LOWESS-smoothed saving crosses the threshold. Bootstrap-Median + 95%-CI über 500 Resamples.

### Threshold = 0% saving

| Feature | Raumtyp_3 | n | Break-Even | Bootstrap-Median | 95%-CI |
|---|---|---:|---:|---:|:---:|
| weekly_parcels | ALL | 312 | no crossing | — | — |
| weekly_parcels | urban | 46 | no crossing | 1171.1 | [1169.4, 1184.2] |
| weekly_parcels | suburban | 141 | no crossing | 17230.2 | [1672.7, 17469.4] |
| weekly_parcels | rural | 125 | no crossing | — | — |
| area_km2 | ALL | 312 | no crossing | — | — |
| area_km2 | urban | 46 | no crossing | 12.0 | [12.0, 12.0] |
| area_km2 | suburban | 141 | no crossing | — | — |
| area_km2 | rural | 125 | no crossing | — | — |
| demand_per_area | ALL | 312 | no crossing | 904.6 | [861.8, 990.6] |
| demand_per_area | urban | 46 | no crossing | — | — |
| demand_per_area | suburban | 141 | no crossing | — | — |
| demand_per_area | rural | 125 | no crossing | 65.9 | [62.4, 66.2] |
| hub_dist_km | ALL | 312 | no crossing | — | — |
| hub_dist_km | urban | 46 | no crossing | — | — |
| hub_dist_km | suburban | 141 | no crossing | — | — |
| hub_dist_km | rural | 125 | no crossing | — | — |
| baseline_routes | ALL | 312 | no crossing | — | — |
| baseline_routes | urban | 46 | no crossing | 6.3 | [6.0, 7.4] |
| baseline_routes | suburban | 141 | no crossing | 77.1 | [76.2, 79.1] |
| baseline_routes | rural | 125 | no crossing | — | — |
| parcels_per_route_baseline | ALL | 312 | no crossing | — | — |
| parcels_per_route_baseline | urban | 46 | no crossing | — | — |
| parcels_per_route_baseline | suburban | 141 | no crossing | — | — |
| parcels_per_route_baseline | rural | 125 | no crossing | — | — |
| b2c_share | ALL | 312 | no crossing | — | — |
| b2c_share | urban | 46 | no crossing | — | — |
| b2c_share | suburban | 141 | no crossing | — | — |
| b2c_share | rural | 125 | no crossing | — | — |

### Threshold = 10% saving

| Feature | Raumtyp_3 | n | Break-Even | Bootstrap-Median | 95%-CI |
|---|---|---:|---:|---:|:---:|
| weekly_parcels | ALL | 312 | 8247.3 | 8227.7 | [6776.2, 11105.4] |
| weekly_parcels | urban | 46 | 1265.7 | 1301.6 | [1194.3, 4289.8] |
| weekly_parcels | suburban | 141 | 7421.3 | 7227.4 | [831.5, 8684.2] |
| weekly_parcels | rural | 125 | no crossing | 16371.5 | [8537.6, 18611.1] |
| area_km2 | ALL | 312 | no crossing | 2.4 | [1.9, 4.2] |
| area_km2 | urban | 46 | 1.9 | 1.9 | [1.8, 6.3] |
| area_km2 | suburban | 141 | no crossing | 11.1 | [2.6, 71.4] |
| area_km2 | rural | 125 | no crossing | 13.7 | [10.7, 229.9] |
| demand_per_area | ALL | 312 | 448.3 | 426.4 | [83.8, 633.3] |
| demand_per_area | urban | 46 | 442.8 | 257.1 | [31.3, 628.5] |
| demand_per_area | suburban | 141 | 112.1 | 106.9 | [74.2, 451.1] |
| demand_per_area | rural | 125 | 45.2 | 45.3 | [32.4, 63.9] |
| hub_dist_km | ALL | 312 | 1.8 | 2.1 | [0.7, 3.6] |
| hub_dist_km | urban | 46 | 5.2 | 5.2 | [3.1, 9.6] |
| hub_dist_km | suburban | 141 | no crossing | 2.6 | [1.4, 5.6] |
| hub_dist_km | rural | 125 | no crossing | 1.2 | [0.4, 2.6] |
| baseline_routes | ALL | 312 | 40.2 | 39.6 | [6.2, 63.4] |
| baseline_routes | urban | 46 | 8.0 | 8.1 | [7.0, 20.2] |
| baseline_routes | suburban | 141 | 35.7 | 32.7 | [6.0, 40.6] |
| baseline_routes | rural | 125 | no crossing | 75.2 | [31.4, 85.1] |
| parcels_per_route_baseline | ALL | 312 | 209.2 | 209.4 | [204.7, 212.7] |
| parcels_per_route_baseline | urban | 46 | 187.6 | 181.0 | [162.5, 192.9] |
| parcels_per_route_baseline | suburban | 141 | 208.8 | 208.7 | [204.6, 211.6] |
| parcels_per_route_baseline | rural | 125 | no crossing | 217.8 | [210.8, 221.4] |
| b2c_share | ALL | 312 | no crossing | — | — |
| b2c_share | urban | 46 | no crossing | — | — |
| b2c_share | suburban | 141 | no crossing | — | — |
| b2c_share | rural | 125 | no crossing | — | — |

### Threshold = 20% saving

| Feature | Raumtyp_3 | n | Break-Even | Bootstrap-Median | 95%-CI |
|---|---|---:|---:|---:|:---:|
| weekly_parcels | ALL | 312 | 626.5 | 1232.8 | [657.6, 2005.6] |
| weekly_parcels | urban | 46 | 1587.8 | 1544.4 | [1397.1, 1705.8] |
| weekly_parcels | suburban | 141 | 1651.7 | 1517.9 | [648.2, 1841.2] |
| weekly_parcels | rural | 125 | 3096.2 | 3019.6 | [861.3, 4356.4] |
| area_km2 | ALL | 312 | 88.6 | 85.4 | [48.3, 113.5] |
| area_km2 | urban | 46 | no crossing | 2.2 | [1.8, 7.1] |
| area_km2 | suburban | 141 | 124.0 | 110.7 | [23.3, 128.5] |
| area_km2 | rural | 125 | 52.8 | 52.7 | [30.9, 105.5] |
| demand_per_area | ALL | 312 | 22.3 | 21.1 | [11.0, 25.1] |
| demand_per_area | urban | 46 | no crossing | 169.8 | [67.9, 202.8] |
| demand_per_area | suburban | 141 | 9.3 | 9.7 | [4.5, 15.2] |
| demand_per_area | rural | 125 | 7.6 | 7.5 | [4.6, 9.5] |
| hub_dist_km | ALL | 312 | 20.5 | 20.5 | [18.5, 22.9] |
| hub_dist_km | urban | 46 | no crossing | 16.1 | [6.4, 17.0] |
| hub_dist_km | suburban | 141 | no crossing | 21.2 | [9.3, 25.8] |
| hub_dist_km | rural | 125 | 17.1 | 15.8 | [8.0, 21.6] |
| baseline_routes | ALL | 312 | 9.6 | 9.5 | [8.7, 10.4] |
| baseline_routes | urban | 46 | 10.7 | 10.4 | [9.0, 11.0] |
| baseline_routes | suburban | 141 | no crossing | 10.3 | [9.5, 11.3] |
| baseline_routes | rural | 125 | 24.5 | 14.2 | [6.2, 27.3] |
| parcels_per_route_baseline | ALL | 312 | 159.3 | 159.6 | [156.1, 164.5] |
| parcels_per_route_baseline | urban | 46 | 151.2 | 151.1 | [146.1, 158.7] |
| parcels_per_route_baseline | suburban | 141 | 153.2 | 153.2 | [146.7, 158.0] |
| parcels_per_route_baseline | rural | 125 | 175.1 | 174.7 | [165.3, 184.4] |
| b2c_share | ALL | 312 | no crossing | — | — |
| b2c_share | urban | 46 | no crossing | — | — |
| b2c_share | suburban | 141 | no crossing | — | — |
| b2c_share | rural | 125 | no crossing | — | — |

## 2. Cost-Decomposition: Routes vs Distance pro Raumtyp_3

| Raumtyp_3 | n | Total Saving | aus Route-Reduktion (€) | aus Distance-Reduktion (€) | %Routes | %Distance | Residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| urban | 35 | 22,473 | 9,458 | 1,731 | 42.1% | 7.7% | +11,285 |
| suburban | 98 | 98,813 | 27,616 | 9,226 | 27.9% | 9.3% | +61,970 |
| rural | 119 | 151,899 | 35,938 | 17,533 | 23.7% | 11.5% | +98,428 |
| ALL | 252 | 273,185 | 73,012 | 28,490 | 26.7% | 10.4% | +171,683 |

## 3. Surrogate-Bias als Funktion der Features

Spearman ρ zwischen Feature und Bias (predicted − actual, in pp).

| Feature | n | Mean Bias (pp) | Median Bias (pp) | Spearman ρ | p-value |
|---|---:|---:|---:|---:|---:|
| weekly_parcels | 312 | +10.12 | +8.69 | -0.474 * | 7.42e-19 |
| area_km2 | 312 | +10.12 | +8.69 | +0.065 | 2.51e-01 |
| demand_per_area | 312 | +10.12 | +8.69 | -0.300 * | 6.83e-08 |
| hub_dist_km | 312 | +10.12 | +8.69 | +0.363 * | 3.69e-11 |
| baseline_routes | 312 | +10.12 | +8.69 | -0.497 * | 7.58e-21 |
| parcels_per_route_baseline | 312 | +10.12 | +8.69 | -0.270 * | 1.30e-06 |
| b2c_share | 312 | +10.12 | +8.69 | +nan | nan |

## 4. Random-Forest Permutation-Importance

Ranking der Features nach Permutation-Importance für saving_pct (n_estimators=500, n_repeats=30).

| Rank | Feature | Permutation Importance | Std |
|---:|---|---:|---:|
| 1 | parcels_per_route_baseline | 1.1937 | 0.0664 |
| 2 | demand_per_area | 0.2975 | 0.0212 |
| 3 | weekly_parcels | 0.1711 | 0.0143 |
| 4 | area_km2 | 0.0595 | 0.0044 |
| 5 | hub_dist_km | 0.0474 | 0.0044 |
| 6 | baseline_routes | 0.0451 | 0.0040 |
| 7 | b2c_share | 0.0000 | 0.0000 |

## 5. Decision-Tree Rules (max_depth=4)

```
|--- parcels_per_route_baseline <= 161.33
|   |--- demand_per_area <= 2.55
|   |   |--- value: [35.01]
|   |--- demand_per_area >  2.55
|   |   |--- demand_per_area <= 27.86
|   |   |   |--- weekly_parcels <= 1395.50
|   |   |   |   |--- value: [24.38]
|   |   |   |--- weekly_parcels >  1395.50
|   |   |   |   |--- value: [27.54]
|   |   |--- demand_per_area >  27.86
|   |   |   |--- weekly_parcels <= 1494.00
|   |   |   |   |--- value: [16.31]
|   |   |   |--- weekly_parcels >  1494.00
|   |   |   |   |--- value: [22.31]
|--- parcels_per_route_baseline >  161.33
|   |--- demand_per_area <= 9.12
|   |   |--- area_km2 <= 83.93
|   |   |   |--- value: [14.46]
|   |   |--- area_km2 >  83.93
|   |   |   |--- parcels_per_route_baseline <= 184.09
|   |   |   |   |--- value: [24.64]
|   |   |   |--- parcels_per_route_baseline >  184.09
|   |   |   |   |--- value: [19.42]
|   |--- demand_per_area >  9.12
|   |   |--- parcels_per_route_baseline <= 179.31
|   |   |   |--- weekly_parcels <= 1827.50
|   |   |   |   |--- value: [10.06]
|   |   |   |--- weekly_parcels >  1827.50
|   |   |   |   |--- value: [16.97]
|   |   |--- parcels_per_route_baseline >  179.31
|   |   |   |--- area_km2 <= 19.35
|   |   |   |   |--- value: [7.83]
|   |   |   |--- area_km2 >  19.35
|   |   |   |   |--- value: [12.17]

```

## 6. Output-Dateien

- `figS1_sensitivity_curves_3.{pdf,png}` — 1D Sensitivity per Feature × Raumtyp
- `figS2_cost_per_parcel_curves.{pdf,png}` — €/Parcel vs Volume
- `figS3_2D_break_even_map.{pdf,png}` — Volume × Area Heatmap mit Break-Even-Contour
- `figS4_cost_decomposition.{pdf,png}` — Saving aus Routes vs Distance
- `figS5_provider_sensitivity.{pdf,png}` — Per-Provider Saving-Curves
- `figS6_surrogate_bias.{pdf,png}` — Predicted − Actual als Feature-Function
- `figS7_break_even_summary.{pdf,png}` — Übersicht aller Break-Even-Punkte