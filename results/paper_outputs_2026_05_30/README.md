# Paper Final — 2026-05-30 (Path-2 results, clean)

Konsistenter Datensatz für das EWGT/Transportation-Research-Procedia-Paper.
Alle Figuren in **Elsevier TRPro Single-/Double-Column Layout**, serif +
dejavuserif mathtext, durchgängiges Branding.

---

## Headline-Zahlen am System-Sweet-Spot

**Sweet-Spot $P^* = 0{,}5$ €/Paket/Tag** (geometric knee, chord-distance method)

| | Wert |
|---|---|
| Cost saving vs daily baseline | **14.14 %** |
| Average customer wait | **0.213 d** |
| Mean delivery frequency | 3.95 d/wk |
| Postal-code areas batched | 236 / 312 |
| Peak-fleet reduction vs baseline | 12.9 % |
| Mo-Sa CV reduction vs baseline | **54.1 %** |
| Total weekly fleet reduction | 5.9 % |

Per-provider individuelle Knies (P\* aus chord-distance pro LSP):

| Provider | Daily baseline | Max saving | Knie $P^*$ | Knie saving | Typ |
|---|---|---|---|---|---|
| GLS | 187 k€/Wo | 33.7 % | **0.75** | 23.2 % | Cost-friendly |
| DPD | 210 k€/Wo | 33.3 % | **0.75** | 21.4 % | Cost-friendly |
| Hermes | 232 k€/Wo | 30.1 % | **0.50** | 22.9 % | Cost-friendly |
| FedEx | 168 k€/Wo | 28.6 % | **0.50** | 20.0 % | Cost-friendly |
| UPS | 198 k€/Wo | 27.2 % | **0.50** | 18.8 % | Cost-friendly |
| Amazon | 351 k€/Wo | 21.2 % | **0.25** | 16.6 % | Cost-friendly |
| **DHL** | 564 k€/Wo | **11.1 %** | **0.25** | 4.5 % | **Service-bound** |

---

## Empfehlungs-Framework für Operator-Typen

| LSP-Typ | Charakteristik | Empfohlenes $P$ | Beispiel |
|---|---|---|---|
| **Service-Premium** | Wait ≤ 0.1 d garantiert, hohe Dichte | $P \geq 0{,}75$ | DHL-like |
| **Hybrid / Standard** | Moderate Cost-Service-Balance | $P \approx 0{,}50$ (System-Knie) | UPS, FedEx, Hermes |
| **Cost-Aggressiv** | Max Konsolidierung, Wait OK | $P \leq 0{,}25$ | DPD, GLS, Discount-Operator |

---

## Folder-Struktur (NEUE Path-2-Pipeline)

### 05_optimization/ — Kosten-Service-Optimierung

| Datei | Inhalt | Layout |
|---|---|---|
| `fig_PF1_pareto.{png,pdf}` | Pareto-Front bei $\theta = 1$ mit Sweet-Spot-Markierung | 1-col, 3.5" |
| `fig_PF2_saving_fleet_heatmaps.{png,pdf}` | 3-Panel-Heatmap: init-Saving / balanced-Saving / Peak-Reduktion | 2-col, 7" |
| `fig_PF3_sweetspot.{png,pdf}` | 2-Panel: Pareto mit Knie-Region + Diminishing returns | 2-col, 7" |
| `fig_PF4_per_provider_pareto.{png,pdf}` | 7 Provider-Pareto-Fronten mit Sweet-Spot-Stern | 2-col, 7" |
| `fig_PF5_per_provider_sweetspots.{png,pdf}` | (a) Per-Provider-Knies (b) Knee-Saving-Bar-Chart | 2-col, 7" |
| `fig_O1_value_of_optimization.{png,pdf}` | 4-Panel: Saving/Peak/Spread/Wait × 4 Baselines × 5 Test-Zellen | 2-col, 7" |
| `fig_SM1_schedule_mix.{png,pdf}` | Frequenz-Mix init vs balanced × 4 P × θ | 2-col, 7" |
| `tab_per_provider_knees.csv` | Provider-Knee-Tabelle (CSV) | |

### 06_balancing/ — Flotten-Glättung

| Datei | Inhalt | Layout |
|---|---|---|
| `fig_B1_fleet_impact_vs_baseline.{png,pdf}` | 3-Panel: Peak / CV / Total fleet reduction vs daily baseline | 2-col, 7" |
| `fig_B2_weekly_fleet_per_provider.{png,pdf}` | Mo-Sa Provider-Vergleich: Baseline vs Final, 5 Zellen × 7 Provider | 2-col, 7" |

### 11_spatial_maps/ — Geografische Choropleths

| Datei | Inhalt | Layout |
|---|---|---|
| `fig_M1_median_freq_at_sweetspot_by_share.{png,pdf}` | Mediane Lieferfrequenz pro PLZ am Sweet-Spot $P = 0{,}5$, 4 θ-Stufen | 2-col, 7" |

### `_archive_v1/` in jedem Folder

Alte Figuren aus dem 2026-05-28-Lauf, **nicht** Pfad-2-konsistent.
Nur als Vergleichs-Referenz aufbewahrt.

### 00_overview/ – 04_model/, 07_validation/ – 10_sensitivity/

Aus dem 2026-05-28-Lauf kopiert. **Daten unverändert** (Input/Training/Modell
sind identisch zur Pfad-2-Optimierung).
**Achtung:** 07_validation/ enthält noch die alten VROOM-Diagnostiken — bei
Pfad-2-Schedules muss VROOM neu validiert werden (TODO).

---

## Daten-Quellen

- `results/overnight_2026_05_29_path2/` — Roh-Daten aus dem Pfad-2-Lauf (88 cells)
- `results/overnight_2026_05_29_path2/_tab_chosen_with_system_smoothing.csv` — Schedules nach System-Smoothing
- `results/overnight_2026_05_29_path2/_system_spread_per_cell.csv` — Spread-Statistik per Cell

---

## Headline-Aussage fürs Paper-Wording

> "Across the full service-penalty by willingness-to-wait grid, the proposed
> ML-surrogate optimization pipeline (Path 2: CD on hub-bundled cost with
> penalty + per-hub balancing + system-level smoothing) achieves a cost saving
> of **14.1 %** vs daily delivery at the system sweet-spot
> $P^\ast = 0.5$ €/parcel/day, with a parcels-weighted mean customer wait of
> **0.21 days**. The Mo-Sa weekly fleet coefficient of variation is reduced
> by **54 %**, peak vehicles by **13 %**, and total weekly fleet capacity by
> **6 %**. Provider-individual sweet-spots span $P^\ast \in [0.25, 0.75]$:
> dispersed LSPs (GLS, DPD) consolidate optimally near $P^\ast = 0.75$,
> mid-density LSPs (Hermes, FedEx, UPS) near $P^\ast = 0.50$, while DHL —
> with three-times the per-PLZ demand density of the smaller operators — is
> structurally service-bound, capped at 11 % maximum cost saving even at
> $P = 0$."

---

## Was noch fehlt (TODO)

1. **VROOM-Re-Validierung am Sweet-Spot $P = 0{,}5$** (statt 0.4 vom Vortag)
   für $\theta \in \{0.5, 1.0\}$ → 07_validation/ neu rendern
2. Paper-Text-Patches auf $P = 0{,}5$ und neue Headline-Zahlen umschreiben
3. Optionales Fine-Sweep mit Pfad-2-Methodik um Sweet-Spot präziser zu
   identifizieren (aktuell auf 8-Punkt-Grid, geometrischer Knie bei P = 0.5)
