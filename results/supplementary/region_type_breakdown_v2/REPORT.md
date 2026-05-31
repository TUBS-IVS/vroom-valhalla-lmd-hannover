# Raumtyp-Breakdown der Batching-Auswertungen

Auswertungseinheit: PLZ-Cluster × Provider (cluster definition: union of 7 merge_maps, 36 clusters).

## Saving je Raumtyp_3 (urban / suburban / rural)

| raumtyp_3 | # cluster | mean saving | 95% CI | median | std | Δ routes | Δ km | total EUR saved |
|---|---:|---:|:---:|---:|---:|---:|---:|---:|
| urban | 5 | 12.54% | [10.05, 15.03] | 10.26% | 7.62 | -7.38% | -18.93% | 22,931 |
| suburban | 14 | 15.32% | [13.79, 16.88] | 14.73% | 7.84 | -6.49% | -22.83% | 94,907 |
| rural | 17 | 21.37% | [19.51, 23.25] | 20.76% | 10.03 | -10.04% | -29.16% | 153,152 |

## Kruskal-Wallis-Test (urban / suburban / rural)

H = 31.947, p = 1.16e-07 (**signifikant** bei α=0.05)

## Saving je Raumtyp_8 (detailliert)

| raumtyp_8 | name | # cluster | mean | 95% CI | median | total EUR |
|---:|---|---:|---:|:---:|---:|---:|
| 2 | Zentrumsnah hochverdichtete Wohnnutzung | 2 | 9.32% | [6.79, 12.30] | 8.43% | 8,152 |
| 3 | Zentrumsnah verdichtete Mischnutzung | 3 | 14.69% | [11.18, 18.14] | 13.92% | 14,779 |
| 4 | Städtisch mit Verdichtungsansätzen | 8 | 15.21% | [12.93, 17.46] | 13.39% | 48,293 |
| 5 | Städtisch mit gewerblicher Prägung | 2 | 15.37% | [12.11, 18.73] | 15.87% | 16,441 |
| 6 | Umland Verstädtert | 4 | 15.51% | [13.10, 17.79] | 16.19% | 30,173 |
| 7 | Umland dörflich mit geringem gewerblichem Einfluss | 9 | 23.36% | [20.80, 25.96] | 23.90% | 100,842 |
| 8 | Umland dörflich ohne gewerblichen Einfluss | 8 | 19.12% | [16.80, 21.64] | 18.33% | 52,310 |

## Surrogate-MAPE je Raumtyp_3 (Frozen Extreme Holdout)

| raumtyp_3 | n samples | MAPE | median APE | p90 APE |
|---|---:|---:|---:|---:|
| urban | 538 | 6.17% | 3.94% | 13.37% |
| suburban | 1424 | 4.71% | 3.11% | 10.74% |
| rural | 113 | 5.11% | 4.07% | 11.53% |
