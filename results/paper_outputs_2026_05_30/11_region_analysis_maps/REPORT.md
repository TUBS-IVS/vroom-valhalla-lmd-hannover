# Paper Maps Final — Merge-Forwarded Choropleths + Raumtyp-Aggregate

Alle Karten in diesem Verzeichnis verwenden **Merge-Forwarding**: PLZ die durch
`merge_small_plz()` in einen Cluster gefaltet wurden, zeigen den Wert ihres
Cluster-Repräsentanten — kein 'no data'-Grau auf den 17 Member-PLZ.

## Cluster-Level Saving (n_cluster mit Daten)

- Anzahl Cluster mit Saving-Daten: 36
- Mean actual saving across clusters: 17.69 %
- Mean predicted saving: 19.30 %
- Mean bias (pred − actual): +1.61 pp

## Per Raumtyp_3 (urban / suburban / rural)

| Raumtyp_3 | # Cluster | Mean actual saving | Mean predicted saving | Mean bias pp | Total weekly EUR saved |
|---|---:|---:|---:|---:|---:|
| rural | 17 | 21.37% | 26.28% | +4.90 | 153,152 |
| suburban | 14 | 15.12% | 19.94% | +4.82 | 94,907 |
| urban | 5 | 12.37% | -6.23% | -18.60 | 22,931 |

## Per Raumtyp_8 (BBSR)

| RT | Name | # Cluster | Mean actual saving | Mean bias pp |
|---:|---|---:|---:|---:|
| 2 | Zentrumsnah hochverdichtete Wohnnutzung | 2 | 9.57% | -17.19 |
| 3 | Zentrumsnah verdichtete Mischnutzung | 3 | 14.24% | -19.53 |
| 4 | Städtisch mit Verdichtungsansätzen | 8 | 15.03% | +6.94 |
| 5 | Städtisch mit gewerblicher Prägung | 2 | 15.38% | -0.76 |
| 6 | Umland Verstädtert | 4 | 15.19% | +3.36 |
| 7 | Umland dörflich mit geringem gewerblichem Einfluss | 9 | 23.38% | +4.13 |
| 8 | Umland dörflich ohne gewerblichen Einfluss | 8 | 19.12% | +5.77 |
