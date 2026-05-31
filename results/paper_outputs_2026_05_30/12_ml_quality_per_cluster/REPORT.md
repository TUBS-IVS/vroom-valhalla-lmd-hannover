# ML Accuracy per Cluster + Raumtyp

Vollständige ML-Quality-Auswertung des Production-LGB-logT auf den **echten**
VROOM-routed cells (out-of-pool). Granularität: Cluster, Provider×Raumtyp_3, Raumtyp_8.

## Per Raumtyp_3

| Raumtyp_3 | n_cells | Cost-MAPE | Cost-Bias % | n_clusters | Saving-bias pp | Saving-MAE pp |
|---|---:|---:|---:|---:|---:|---:|
| rural | 509 | 7.12% | -5.14% | 17 | +4.83 | 5.35 |
| suburban | 588 | 11.29% | -2.71% | 14 | +3.59 | 9.92 |
| urban | 190 | 33.76% | +25.90% | 5 | -20.93 | 29.23 |

## Per Raumtyp_8 (BBSR-style)

| RT | Name | n_cells | Cost-MAPE | Cost-Bias % | n_clusters | Saving-bias pp |
|---:|---|---:|---:|---:|---:|---:|
| 2 | Zentrumsnah hochverd. Wohnen | 78 | 38.05% | +28.77% | 2 | -23.26 |
| 3 | Zentrumsnah verd. Mischung | 112 | 30.77% | +23.90% | 3 | -19.29 |
| 4 | Städt. mit Verdichtungsansätzen | 327 | 11.05% | -5.01% | 8 | +6.13 |
| 5 | Städt. gewerblich geprägt | 121 | 15.35% | +4.09% | 2 | -2.76 |
| 6 | Umland verstädtert | 140 | 8.36% | -3.19% | 4 | +3.08 |
| 7 | Umland dörflich m. Gewerbe | 278 | 6.74% | -4.56% | 9 | +4.08 |
| 8 | Umland dörflich rein | 231 | 7.58% | -5.85% | 8 | +5.77 |

## Provider × Raumtyp_3 Cost-MAPE

| Provider | urban | suburban | rural |
|---|---:|---:|---:|
| Amazon | 37.2% | 6.6% | 6.0% |
| DHL | 23.6% | 4.0% | 3.8% |
| DPD | 36.6% | 10.6% | 8.3% |
| FedEx | 34.7% | 27.4% | 9.4% |
| GLS | 31.5% | 13.3% | 9.2% |
| Hermes | 39.1% | 9.7% | 7.4% |
| UPS | 42.8% | 15.7% | 7.2% |

## Provider × Raumtyp_3 Cost-Bias %

| Provider | urban | suburban | rural |
|---|---:|---:|---:|
| Amazon | +29.06 | -5.26 | -4.76 |
| DHL | +20.06 | -3.15 | -3.10 |
| DPD | +25.98 | -8.84 | -5.81 |
| FedEx | +32.64 | +14.82 | -5.51 |
| GLS | +19.33 | -7.31 | -7.96 |
| Hermes | +24.13 | -9.28 | -4.34 |
| UPS | +35.75 | +4.47 | -4.80 |

## Top-10 Worst-Predicted Cluster

| Cluster | Raumtyp_3 | n_cells | Cost-MAPE | Cost-Bias % | Saving-bias pp | n_members | Einwohner |
|---|---|---:|---:|---:|---:|---:|---:|
| 30159 | urban | 49 | 57.30% | +48.49% | -40.27 | 4 | 59203 |
| 30167 | urban | 29 | 53.34% | +53.34% | -44.55 | 2 | 33453 |
| 30449 | urban | 55 | 31.34% | +22.30% | -18.42 | 3 | 52274 |
| 30163 | suburban | 92 | 18.46% | +6.59% | -4.64 | 4 | 74338 |
| 30625 | suburban | 51 | 16.61% | +0.40% | +3.31 | 2 | 33250 |
| 30853 | suburban | 53 | 13.11% | -1.13% | +1.54 | 2 | 30958 |
| 30559 | suburban | 102 | 12.73% | -3.16% | +4.42 | 4 | 71927 |
| 30457 | suburban | 30 | 12.11% | -11.53% | +12.90 | 1 | 18093 |
| 30982 | rural | 29 | 10.27% | -9.80% | +7.42 | 1 | 13750 |
| 30453 | suburban | 28 | 9.44% | -8.68% | +7.98 | 1 | 17864 |
