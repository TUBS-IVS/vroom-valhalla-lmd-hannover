# CHANGES rev1 — model revision of 2026-08 (paper text, part A)

Scope: the changes the 2026-08 model revision forces on
`paper/EWGT_2026_rev1/tbc_preprint_main.tex`. This file continues
`paper/EWGT_2026_rev1/CHANGES_rev1.md`, which logs the earlier
reviewer-response round (11 reviewer points, mirrored into the preprint on
2026-08-18). Nothing there is retracted; everything here is on top of it.

**Status: PART A.** All grid numbers below are from
`results/revision_2026_08_v5/tables/` and are marked in the manuscript with the
`\provisional{...}` macro (prints a light superscript `p`). Part B replaces
every one of them from the final head-priced grid and then deletes the macro.
Before submission, `grep -c 'provisional{' tbc_preprint_main.tex` must reach 0.

Evidence keys: `§n` = section of `docs/PAPER_COMPENDIUM_2026_05_24.md`;
table files are relative to `results/revision_2026_08_v5/tables/`.

---

## A. Method statements

### A1 — Cost model (documentation fix; no number changes)

- **Old:** "The rate $c_d$ prices distance and $c_f$ is the fixed daily cost per
  vehicle, including driver labor, **so no separate time term is needed**."
- **New:** the clause is deleted, and a new paragraph states the effective cost
  model of every label: 189.15 EUR per vehicle-day + 0.3864 EUR/km + 36 EUR per
  route-hour, with VROOM's default `per_hour` active in **all** labels
  (72.2 % fixed / 6.0 % distance / 21.7 % time). The Daganzo backbone has only
  the vehicle and distance terms, so `alpha` and the learned residual absorb the
  time component. Driver wage is inside `c_f`; the route-time term is an
  **additional** charge and is explicitly *not* claimed to be inside it.
- **Why:** the documented model (`COST_PER_HOUR_EUR = 0.0`, "labour in fixed")
  is not what VROOM computed. Measured on the training pool:
  `(cost − 189.15·n_routes − 0.3864·km) / duration_h` = 36.09 EUR/h, ratio to
  36.00 = 1.002.
- **Evidence:** §40.12 (finding), §40.13 (Lasse's ruling: accept for this
  revision, no relabel), `task-6e-brief.md`.
- **Effect on numbers:** none. Baseline, scenarios, prediction and validation all
  carry the same term; the paper text was wrong, the pipeline was not.

### A2 — Universal minimum/maximum tour rule (new Section 2.2)

- **Old:** no such statement. The surrogate priced whatever instance the
  optimizer formed, including hub-wide pooled tours far outside the training
  family (median 2,020 stops, max 9,644; 86.1 % above the training stop maximum).
- **New:** every priced tour, in the baseline exactly as in every scenario,
  carries at least 230 parcels, at most 556 stops, at most 159 km², and a convex
  hull at most 1.22x the summed member area. Cells at or above the parcel
  minimum are always standalone; cells below it are pooled with same-provider,
  same-depot neighbours until the rule holds, recomputed per weekday.
- **Why:** the surrogate is only defensible inside its measured validity domain,
  and the rule removes the baseline/scenario asymmetry (see B2).
- **Evidence:** §39.7 (calibration of 230 / 556 / 159 from OOF error, p99, p95),
  §40.1 (hull ratio 1.22 = training maximum; caps alone control size, not shape),
  `src/batch_delivery/config/constants.py:154-157`, `task-1-brief.md`.

### A3 — Express residuals are ordinary instances

- **Old:** "Cells served by the same depot share a pooled daily delivery tour for
  the non-consolidated share of their demand."
- **New:** on a day without a batched tour, a cell's non-willing parcels form an
  ordinary instance of that cell and enter the same partition as any other tour.
  There is no separate express mechanism.
- **Evidence:** §40.7 (per-cell express instead of bounded pooling; five-point
  justification), §40.8 (why a minimum tour size is nonetheless required).

### A4 — Equation (3): coupling term removed, residual coupling stated

- **Old:** "These shared tours couple the cells --- their load depends on the
  hub's joint schedule."
- **New:** the hub sum is separable wherever the tour rule leaves cells
  standalone; a residual coupling remains **inside pooling groups**, because a
  group's membership on a day depends on which of its cells deliver that day.
  The hub objective is retained precisely to keep that residual coupling priced.
  The text does **not** claim separability.
- **Evidence:** §40.7 item 5, `task-1-brief.md` (partition module).

### A5 — Bundle head (new, two sentences plus Gate U numbers)

- **Old:** no such component.
- **New:** pooled instances are priced by a dedicated head (alpha-Daganzo
  backbone + LightGBM residual) trained on 877 solver-routed bundles, validated
  out-of-fold with folds grouped by member set. It prices only certified bins
  (>= 6 labels and |OOF bias| <= 5 %): 21 of 235 bins, 44.0 % of the pooled
  instances that occur in the deployed solutions, 2.85 % OOF MAPE (2.71 %
  occurrence-weighted), 1.35 % absolute bias. Pool-wide: 5.37 % MAPE, +0.57 %
  bias. Outside its support the head refuses and the pooled tour is charged as
  the sum of its members' single-cell prices; every fallback is counted.
- **Evidence:** §40.16, `task-10b-brief.md` (Gate U ruling on the deployed
  population).

### A6 — Two cost lenses (new Section 2.3, Equations 4 and 5)

- **Old:** a single cost accounting (every vehicle-day charged in full).
- **New:** routing lens `C_route = sum_i C_i` (the submitted accounting) and
  operator lens `C_op = sum_i (C_i − c_f v_i) + 6 c_f sum_h max_d v_hd`, i.e.
  1,134.90 EUR per peak vehicle per depot per week. Both lenses are evaluated
  against the same baseline; neither contains the service penalty.
- **Evidence:** §40.11 (why routing euro is the wrong currency for balancing),
  §40.12 (baseline in both lenses: routing 1,909,432 EUR/wk; operator
  2,109,742 EUR/wk; sum of depot peaks 1,239), `task-6e-brief.md`.

### A7 — Stage 2 rebuilt, stage 3 dropped

- **Old:** "A two-stage post-processing therefore **preserves each cell's chosen
  delivery frequency**: a per-depot greedy local search reassigns weekdays to
  flatten each depot's peak, accepting only swaps that keep total weekly cost
  **within 5 % of the cost-optimal value**, and a subsequent smoothing stage
  ... redistributes weekday assignments across that provider's hubs."
- **New:** stage 2 is an operator-cost polish. It accepts a (cell, pattern) move
  iff `dC_op + P*theta_z*p_z*dwait < 0`. No budget, no spread objective. It is
  **frequency-free at theta > 0** (a cell may gain or lose delivery days; every
  consequence is priced) and **pinned to daily at theta = 0** (the baseline is
  recovered bit-for-bit). Three starts (routing-optimal, range-balanced,
  frequency-preserving), lowest operator cost kept. The text states plainly that
  it is a local search and reports the gap to the flat-profile bound
  `sum_h ceil(vehicle-days_h / 6)`. The provider-level smoothing stage is
  dropped from the production path.
- **Why:** the 5 % budget never bound (max 2.7 % per provider); the range
  objective bought trough-filling rather than peak reduction; the frequency lock
  blocked the moves that carry the operator-lens value (9 of DHL's 16 depots
  serve one cell and cannot rotate).
- **Evidence:** §40.10 (what the old balancer bought), §40.11, §40.14 (the
  frequency lock was the leak), `task-6e-brief.md`, `task-6f-brief.md`,
  `tab_stage2_ablation_v2.csv`.

### A8 — The operator plan is not service-neutral

- **Old:** the post-processing "preserves each cell's chosen delivery frequency"
  (implying identical service).
- **New:** the operator plan's mean delivery frequency and average waiting time
  differ from the routing plan's, in **either** direction depending on the
  operating point; the paper reports both plans side by side and never combines
  a cost from one with a service figure from the other. The term
  "service-neutral" is not used anywhere.
- **Evidence:** §40.15 (wait moves both ways: 0.97 -> 0.77 d at (0, 1);
  0.21 -> 0.23 d at (0.5, 1)), `tab_headline_theta1_v2.csv`.

---

## B. Result statements

### B1 — Headline saving

- **Old:** "Cost saving relative to the daily-delivery baseline peaks at
  **22.8 %** in the fully consolidated regime (0, 1)."
- **New:** routing lens **23.10 %** (routing-optimal plan) and **20.43 %**
  (operator plan); operator lens **−7.79 %** (routing-optimal plan) and
  **+24.69 %** (operator plan). The routing-lens headline is essentially
  unchanged; the submitted 22.8 % was the post-balancing value of the same
  stage-1 selection (stage 1 is bit-identical to run 2 across all 616 triples).
- **Evidence:** §40.15, `tab_headline_theta1_v2.csv`, `tab_stage2_ablation_v2.csv`.

### B2 — The theta = 10 % bulge (retracted explanation)

- **Old:** "at theta = 0.1 even P = 10 still leaves **42 %** of areas non-daily
  (f = 5.05) ... Two mechanisms compound. First, the penalty scales as
  P*theta_z*p_z*w ... Second, **hub-bundling lets rural and low-density
  suburban cells share a depot tour with neighbors**, so the routing saving on
  the 90 % standard-delivery parcels covers the small penalty on the willing
  10 %."
- **New:** the bulge was an artefact of the baseline/scenario pooling asymmetry,
  not a mechanism. Under the universal tour rule the corner is flat: at
  (10, 0.1) the routing-optimal plan saves **0.03 %** routing and **0.02 %**
  operator cost, and only **2.9 %** of areas remain non-daily. A symmetric
  re-computation of the submitted grid lowers the baseline by 0.74 % and the
  (0, 1) headline from 22.79 % to 22.22 %, which bounds the effect at full
  adoption. What remains at partial adoption: (0, 0.1) saves 1.4 % routing
  (routing-optimal plan) and 3.9 % operator cost (operator plan).
- **Evidence:** §39.2 (the 2.08 EUR/parcel saving against 1.51 EUR/parcel total
  cost that exposed it), §39.3 (asymmetry measured: ~24 % of demand on tours the
  baseline cannot have; symmetric recomputation; selective-pooling counter-test
  0.74 %), `tab_grid_full_v2.csv`, `_tab_chosen_v2.csv`.

### B3 — Partial adoption

- **Old:** partial adoption reported as broadly positive, with the low-theta
  fleet gain attributed partly to spatial pooling.
- **New:** partial adoption is worthwhile in the **operator lens only**. The
  operator plan is never negative anywhere on the grid (3.9 % at (0, 0.1)
  rising to 11.0 % at (0, 0.8)); the
  routing-optimal plan turns negative in the operator lens under partial
  adoption, down to **−10.4 %** at (0, 0.9), and its routing-lens saving is
  small (1.4–6.3 %).
- **Evidence:** §40.15, `tab_grid_full_v2.csv`.

### B4 — Cost of balancing

- **Old:** "fleet balancing costs only **0.3–0.6 pp** of saving in the
  high-saving region."
- **New:** at (0, 1) the operator polish gives up **2.7 pp** of routing saving
  (23.10 -> 20.43 %) and buys **32.5 pp** of operator saving (−7.79 -> +24.69 %)
  together with a 16.9 % lower summed depot peak.
- **Evidence:** §40.15, `tab_headline_theta1_v2.csv`.

### B5 — Fleet coefficient of variation and its baseline

- **Old:** "the Monday–Saturday coefficient of variation falls by **54.1 %** at
  (0.5, 1) and peaks at **78.2 %** at (0.25, 1)"; baseline CV 0.135.
- **New:** baseline **0.139** under this grid's partition-aware vehicle
  accounting; the operator plan reaches 0.024 at (0.25, 1) and 0.047 at
  (0.5, 1), i.e. **−83 %** and **−66 %**, with a grid maximum of **−88 %** at
  (0, 0.7). The manuscript states explicitly that CV figures must not be quoted
  across grid versions.
- **Evidence:** §40.18, `tab_fleet_diagnostics_v2.csv`.

### B6 — Total weekly fleet

- **Old (already struck in the previous round, stays struck):** "total weekly
  fleet can grow by up to +4.6 % at intermediate theta."
- **Old (submitted revision):** "The total weekly fleet declines across the whole
  adoption range, by 6.3–6.8 % already at theta = 0.1 ... Part of that reduction
  at low theta is structural rather than a consolidation effect."
- **New:** both are removed. The fleet statement is now made on the **summed
  depot peak** per lens and plan (−16.9 % at (0, 1), −17.1 % at (0.25, 1) for
  the operator plan; +34.5 % at (0, 1) for the routing-optimal plan), because
  the peak is what the operator pays for. The structural caveat about spatial
  pooling is obsolete: the mechanism that caused it no longer exists.
- **Evidence:** §39.1 (+4.6 % refuted as Bug-A artefact), §39.3, §40.15,
  `tab_headline_theta1_v2.csv`, `tab_fleet_diagnostics_v2.csv`.

### B7 — Carrier classes are lens-dependent

- **Old:** "This groups the seven operators into service-bound (Amazon, DHL) at
  P* = 0.25, hybrid (FedEx, Hermes, UPS) at 0.5, and cost-aggressive (DPD, GLS)
  at 0.75."
- **New:** unchanged **in the routing lens on the routing-optimal plan**. In the
  operator lens on the operator plan three carriers move up one class: Amazon
  0.25 -> 0.5, FedEx 0.5 -> 0.75, Hermes 0.5 -> 0.75; DHL, UPS, DPD, GLS
  unchanged. Every statement of the taxonomy now carries its lens.
- **Evidence:** §40.18, `tab_pstar_knees_v2.csv`.

### B8 — Recommended operating point

- **New:** P = 0.25 EUR/parcel-day under both lenses. At (0.25, 1) the operator
  plan saves 22.82 % operator and 17.07 % routing cost, cuts the summed depot
  peak by 17.1 %, and halves the additional wait against P = 0 (0.39 vs 0.77 d).
  The operator lens alone would prefer P = 0 by 1.9 pp; the flat-discount
  scenario (B9) makes P = 0.25 strictly better.
- **Evidence:** §40.15, §40.17, `tab_headline_theta1_v2.csv`.

### B9 — Discount scenario (new subsection)

- **New:** two payout rules on the same grid. (a) P EUR per parcel and waiting
  day: the operator plan retains 24.7 / 17.0 / 11.6 / 8.4 / 6.2 % operator
  saving at P = 0 / 0.25 / 0.5 / 0.75 / 1 (20.4 / 10.6 / 5.6 / 2.6 / 1.0 % in
  the routing lens). (b) flat 0.50 EUR per delayed parcel: 8.6 / 13.2 / 12.6 /
  10.9 / 9.2 %, so the optimum moves off P = 0 to P = 0.25–0.5. Break-even
  discount per delayed parcel: 0.77 / 1.19 / 1.57 / 1.89 / 2.24 EUR (operator
  lens), 0.57–1.24 EUR (routing lens). Delayed parcels per week at full
  adoption: 680 k / 404 k / 248 k / 165 k / 111 k of 1.263 M. At partial
  adoption the discount pushes the routing lens negative (−0.3 to −2.4 %) while
  the operator lens stays at +3 to +6 %.
- **Evidence:** §40.17.

### B10 — Out-of-sample validation

- **Old (abstract):** "Out-of-sample VROOM re-routing confirms the surrogate is
  conservative, underestimating realized savings by 0.9–2.7 pp across four
  validated operating points."
- **Old (Section 3.4):** the four theta = 1 points presented as validating the
  reported solutions.
- **New:** the four points are retained but re-framed. They predate the tour rule
  and the operator polish, so they are evidence about the **surrogate**, not a
  validation of the plans reported here. The section adds the direct per-cell
  evidence (3.03 % MAPE, +2.73 % bias over 1,247 solver-routed cells;
  +1.75 % bias on daily vs +3.51 % on 3-per-week schedules, i.e. consolidated
  schedules are overpriced more, so savings are understated; accuracy improves
  with instance size, 5.67 % -> 1.12 % MAPE). Re-validation of the revised
  pipeline in both lenses, including at least one partial-adoption point with
  pooled tours, is stated as **pending**.
- **Evidence:** §39.5 (conservative bias, extrapolation is safe), §39.6 (all
  four validated points sit at theta = 1, where no pooled tour exists),
  `task-12-brief.md`.

### B11 — Limitations (new subsection)

- **New:** (i) DHL depot structure (9 of 16 depots serve one cell) drives the
  operator-lens magnitude; (ii) pooling is within-depot only, inter-depot and
  inter-provider cooperation is future work; (iii) stage 2 is a local search,
  gap to the flat bound reported (50 of 1,030 peak vehicles at (0, 1));
  (iv) partial adoption is not solver-validated; (v) bundle-head coverage is
  44.0 %, the fallback is coarse, and certification is region-specific;
  (vi) baseline fleet CV is 0.139 here against 0.135 in the submission;
  (vii) the D3c feature skew (six of seven providers fall back to one parcel per
  demand point for two of the 25 features) — the deployed model was validated
  end-to-end in that state, so the numbers stand, but accuracy is lost.
- **Evidence:** §40.14 (i), §39.8 and §40.7 (ii), §40.15 and
  `tab_fleet_diagnostics_v2.csv` (iii), §39.6 (iv), §40.16 (v), §40.18 (vi),
  §40.6 (vii).

---

## C. Front matter, figures and structure

### C1 — Abstract

Rewritten to state the tour rule, both lenses, both plans, the recommended
operating point, and the lens-dependence of the carrier classes. The
"0.9–2.7 pp across four validated operating points" claim is replaced by the
per-cell bias evidence (+1.8 % daily vs +3.5 % consolidated over 1,247 cells).
Evidence: §39.5, §40.15, §40.18.

### C2 — Contributions

Three contributions become four: the tour rule and the two-lens accounting are
now named as contributions. No evidence claim attached.

### C3 — Figure captions 4, 5 and 6

Rewritten for the regenerated panel layouts, with plan and lens named in every
panel description and the baseline named per figure:

- **Fig. 4** — two-plan frequency mix (row 1 stage 1, row 2 stage 2); mean
  delivery days moved to supplementary.
- **Fig. 5** — 2x3: (a) routing saving, routing lens, stage-1 plan; (b) operator
  saving, operator lens, stage-2 plan; (c) wait, stage-2 plan; (d) summed depot
  peak vs baseline, stage-2 plan; (e) Mon–Sat fleet CV, stage-2 plan, against
  its own 0.139 baseline; (f) weekly vehicle-days vs baseline, stage-2 plan. The
  caption warns that (a) and (b) are a different lens **and** a different plan.
  The off-diagonal companion figure is referenced as supplementary.
- **Fig. 6** — (a)/(b) Pareto fronts in the two lenses; (c) per-LSP knees in both
  lenses; (d)–(f) structural breakdown at P = 0 on the operator plan, showing
  **delivery frequency** rather than per-area euro saving, bucketed by hub
  distance, parcels per drop-site, and depot size. The caption states that the
  sub-13-cell buckets in (f) are almost entirely DHL depots.

**Open:** the caption text describes the regenerated figures; the PDF files in
`paper/EWGT_2026_rev1/figures/` are still the previous renders. Part B syncs
them with `scripts/revision/71_sync_paper_figs.py` (md5 PASS required).
Evidence: `task-13-report.md` (panel lists), §40.18.

### C4 — Page budget

The preprint layout grew from 11 to 16 pages. The Elsevier camera-ready has a
hard 8-page limit in a much denser layout, so a trim pass is mandatory. Ranked
trim candidates are listed in a comment block at the top of
`tbc_preprint_main.tex`.

---

## D. Code hygiene (no paper claim)

Loud `DeprecationWarning`s at import plus a docstring pointer to
`scripts/revision/61_grid_run_v2.py` and `70_figs_tables_v2.py` were added to
the stale entry points: `scripts/pipeline/02_optimize_grid.py`,
`scripts/pipeline/03_apply_smoothing.py`,
`scripts/revision/10_recompute_stage3_outputs.py`, and the 19
`scripts/paper/paper_final_*.py` scripts. These recompute totals without the
pool term and must never be used for v2-semantics numbers. Nothing is deleted.
Evidence: `.superpowers/sdd/2026-08-25-realistic-tours-implementation/progress.md`
lines 68 and 100.

---

## E. Carried over, still open

1. HAGRID acronym full form (reviewer point 3) — still not supplied.
2. Final 8-page acceptance in the MiKTeX/Elsevier environment (Lasse), now with
   a mandatory trim pass (C4).
3. Stale `%TODO` headers in the Elsevier master.
4. Human read-through of the preprint.
5. Part B: replace every `\provisional{}` value, insert the validation
   paragraph, sync the figures, and do a final lens/plan consistency pass.
