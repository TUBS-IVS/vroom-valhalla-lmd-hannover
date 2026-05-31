# Paper Results Final — Path-2 Optimization Pipeline

Final clean datapack for the EWGT/Transportation-Research-Procedia paper.
All figures in **Elsevier TRPro layout** (serif + dejavuserif mathtext,
single-column 3.5" or double-column 7"), all data Path-2-consistent.

---

## Headline numbers at the system sweet-spot

**Sweet-spot $P^* = 0{,}5$ €/parcel/day** (geometric knee via chord-distance method)

| Metric | Value |
|---|---|
| Cost saving vs daily baseline | **14.14 %** |
| Average customer wait | **0.213 d** |
| Mean delivery frequency | 3.95 d/wk |
| Postal-code areas batched | 236 / 312 |
| Peak-fleet reduction vs baseline | 12.9 % |
| Mo-Sa CV reduction vs baseline | **54.1 %** |
| Total weekly fleet reduction | 5.9 % |

## Operating-point summary (full table in `00_overview/tab_summary_by_operating_point.csv`)

| Operating point | $P$ | $\theta$ | Saving | Wait | Mean freq | Batched | Peak | Peak red |
|---|---|---|---|---|---|---|---|---|
| Daily baseline | 10 | 0 | 0.0 % | 0.00 d | 6.0 | 0 / 312 | 894 | 0.0 % |
| Cost-optimal end | 0 | 1 | **23.1 %** | 0.97 d | 2.0 | 312 / 312 | 1005 → 748 | **25.6 %** |
| **Sweet-spot (System)** | **0.5** | **1** | **14.1 %** | **0.21 d** | 3.95 | 236 / 312 | 862 → 738 | 14.4 % |
| Service-priority | 1 | 1 | 7.9 % | 0.08 d | 4.93 | 143 / 312 | 860 → 800 | 7.0 % |

## Per-provider individual knees (chord-distance on each LSP's Pareto)

| Provider | Daily baseline | Max saving | Knee $P^\ast$ | Knee saving | Type |
|---|---|---|---|---|---|
| GLS | 187 k€/wk | 33.7 % | **0.75** | 23.2 % | Cost-friendly |
| DPD | 210 k€/wk | 33.3 % | **0.75** | 21.4 % | Cost-friendly |
| Hermes | 232 k€/wk | 30.1 % | **0.50** | 22.9 % | Cost-friendly |
| FedEx | 168 k€/wk | 28.6 % | **0.50** | 20.0 % | Cost-friendly |
| UPS | 198 k€/wk | 27.2 % | **0.50** | 18.8 % | Cost-friendly |
| Amazon | 351 k€/wk | 21.2 % | **0.25** | 16.6 % | Cost-friendly |
| **DHL** | 564 k€/wk | **11.1 %** | **0.25** | 4.5 % | **Service-bound** |

---

## Recommendation framework

| LSP type | Profile | Recommended $P$ | Example |
|---|---|---|---|
| **Service-Premium** | Guaranteed Wait ≤ 0.1 d, high density | $P \geq 0{,}75$ | DHL-like |
| **Hybrid / Standard** | Moderate cost-service balance | $P \approx 0{,}50$ (system knee) | UPS, FedEx, Hermes |
| **Cost-Aggressive** | Max consolidation, Wait OK | $P \leq 0{,}25$ | DPD, GLS, discount-operator |

---

## Folder structure

### 00_overview/
- `tab_summary_by_operating_point.csv` — headline summary table (4 reference operating points)

### 01_input_data/ – 04_model/
Aus 2026-05-28 kopiert (Daten/Modell unverändert zur Pfad-2-Pipeline).

### 05_optimization/

| File | Content | Layout |
|---|---|---|
| `fig_PF1_pareto.{png,pdf}` | Pareto frontier $\theta = 1$ with sweet-spot star | 1-col, 3.5" |
| `fig_PF2_saving_fleet_heatmaps.{png,pdf}` | 3-panel heatmap: init-saving / balanced-saving / peak-reduction | 2-col, 7" |
| `fig_PF3_sweetspot.{png,pdf}` | 2-panel: Pareto with knee-region + diminishing returns | 2-col, 7" |
| `fig_PF4_per_provider_pareto.{png,pdf}` | 7 provider Pareto frontiers with sweet-spot star | 2-col, 7" |
| `fig_PF5_per_provider_sweetspots.{png,pdf}` | (a) per-provider knees (b) knee-saving bar chart | 2-col, 7" |
| `fig_O1_value_of_optimization.{png,pdf}` | 4-panel: Saving / Peak / Spread / Wait across 4 baselines × 5 cells | 2-col, 7" |
| `fig_O2_pipeline_contribution.{png,pdf}` | Per-stage saving contribution (CD-init / balancing / smoothing) | 2-col, 7" |
| `fig_O3_weekday_pattern_share.{png,pdf}` | Top-5 weekday-combinations per provider at sweet-spot | 2-col, 7" |
| `fig_O4_wait_distribution.{png,pdf}` | Per-PLZ wait-day distribution at sweet-spot, by provider | 2-col, 7" |
| `fig_O5_freq_histogram_per_provider.{png,pdf}` | Per-provider frequency histogram at sweet-spot | 2-col, 7" |
| `fig_SM1_schedule_mix.{png,pdf}` | Frequency-mix init vs balanced × 4 $P$ × $\theta$ | 2-col, 7" |
| `tab_per_provider_knees.csv` | Provider knee table (CSV) | |

### 06_balancing/

| File | Content | Layout |
|---|---|---|
| `fig_B1_fleet_impact_vs_baseline.{png,pdf}` | 3-panel: Peak / CV / Total fleet reduction vs daily baseline | 2-col, 7" |
| `fig_B2_weekly_fleet_per_provider.{png,pdf}` | Mo-Sa Provider comparison: Baseline vs Final | 2-col, 7" |

### 07_validation/

VROOM-Validierung läuft overnight für **4 Zellen** ($P \in \{0, 0.25, 0.5, 0.75\}$, $\theta = 1$):
- deckt alle Provider-individuellen Knies ab
- plus cost-aggressive Baseline ($P = 0$)
- alle im Top-Szenario (volle Willingness)

Outputs nach Completion:
- `tab_vroom_path2.csv` — pro (cell, provider, plz, day)
- `tab_diagnostics_path2.csv` — MAPE / Bias / $R^2$ pro Cell
- `fig_V1_vroom_vs_ml.{png,pdf}` — Scatter VROOM vs ML

### 08_region_analysis/
Aus 2026-05-28 (region-typ-Aufschlüsselung, Demand-Cluster).

### 09_maps/

| File | Content | Layout |
|---|---|---|
| `fig_M1_median_freq_at_sweetspot_by_share.{png,pdf}` | Median delivery frequency per PLZ at sweet-spot $P = 0{,}5$, 4 $\theta$ levels | 2-col, 7" |

---

## Key interpretive findings (für Diskussion)

### 1. Optimization carries most of the work; balancing trades fleet smoothness for tiny cost

`fig_O2_pipeline_contribution.png` zeigt: CD-init liefert **~23.1pp Saving am
cost-optimalen Ende, 14.1pp am Sweet-Spot**. Per-Hub-Balancing kostet
**~0.3-0.6pp** extra Cost für die Flottenglättung. System-Smoothing ist
**near-zero** auf Cost-Seite (bringt aber Flotten-CV-Verbesserung).

### 2. Provider economics dominate optimal-$P$ selection

GLS/DPD (kleine LSPs, ~200 k€/Wo Baseline) erreichen Knie bei $P = 0.75$
mit 22-23 % Saving. DHL (sehr große, dichte LSP, 564 k€/Wo) hat strukturelle
Decke bei nur **11 % maximum Saving**, selbst bei $P = 0$.

### 3. $P = 10$, $\theta = 0.1$ spike: hub-bundling unlocks unusual operating regimes

130 von 312 PLZs gehen unter daily bei $P = 10/\theta = 0.1$ — **26× mehr**
als der naive per-PLZ-Argmin findet. Hub-Bündelung macht "skip-one-day"
selbst unter prohibitiver Penalty wirtschaftlich, wenn nur 10 % der Kunden
warten müssen.

### 4. Daily baseline has its own natural Mo-Sa wave

Baseline Mo-Sa: [1058, 1120, 1239, 1181, 1005, 794] — Sa-Tief, Mi-Peak.
Pfad 2 **reduziert die CV um 54 %** weil die Optimierung die Liefertage
gezielt auf Demand-Schwachstellen legt.

### 5. Weekday patterns favour day-cluster splits

`fig_O3_weekday_pattern_share.png`: am Sweet-Spot wählt die Optimierung
systematisch zwei Cluster (early-week / late-week), nicht zufällig.

---

## Data sources

- `results/overnight_2026_05_29_path2/` — raw 88-cell Pfad-2-Lauf
- `_tab_chosen_with_system_smoothing.csv` — Schedules nach System-Smoothing
- `_system_spread_per_cell.csv` — Spread-Statistik pro Cell

---

## Headline paper-wording

> "Across the full service-penalty by willingness-to-wait grid, the proposed
> ML-surrogate optimization pipeline (Path 2: coordinate-descent on hub-bundled
> cost with service penalty + frequency-preserving fleet balancing +
> system-level smoothing) achieves a cost saving of **14.1 %** vs daily
> delivery at the system sweet-spot $P^\ast = 0.5$ €/parcel/day, with a
> parcels-weighted mean customer wait of **0.21 days**. The Mo-Sa weekly
> fleet coefficient of variation is reduced by **54 %**, peak vehicles by
> **13 %**, and total weekly fleet capacity by **6 %**. Provider-individual
> sweet-spots span $P^\ast \in [0.25, 0.75]$: dispersed LSPs (GLS, DPD)
> consolidate optimally near $P^\ast = 0.75$, mid-density LSPs (Hermes,
> FedEx, UPS) near $P^\ast = 0.50$, while DHL — with three-times the per-PLZ
> demand density of the smaller operators — is structurally service-bound,
> capped at 11 % maximum cost saving even at $P = 0$."

---

## What still needs (TODO)

1. **VROOM-Re-Validation** läuft overnight (Background-Task `bpm131iz8`)
2. Nach Completion: MAPE / Bias / $R^2$ für Path-2-Schedules berechnen, fig_V1 rendern
3. Paper-Text-Patches mit $P^\ast = 0{,}5$ und neuen Headline-Zahlen
