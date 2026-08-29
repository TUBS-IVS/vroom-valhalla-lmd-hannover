# CHANGES rev1 — model revision of 2026-08 (paper text, parts A, B and C)

Scope: the changes the 2026-08 model revision forces on
`paper/EWGT_2026_rev1/tbc_preprint_main.tex`. This file continues
`paper/EWGT_2026_rev1/CHANGES_rev1.md`, which logs the earlier
reviewer-response round (11 reviewer points, mirrored into the preprint on
2026-08-18). Nothing there is retracted; everything here is on top of it.

**Status: PART C complete, fix round 3 applied. No open markers.** Sections A-D record part A (v5 grid), section F
part B (grid v6), section G part C (the v6 VROOM validation and the part-B
review fixes). **Later sections win**: where A-D, F and G disagree, G is
authoritative, then F. No `\provisional{}` marker remains; the macro is
deleted. One `% PART C3` marker is left in the manuscript, for validation
item 3 (the partial-adoption point), which was still solving.

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
  (72.2 % fixed / 6.0 % distance / 21.7 % time **over the surrogate's training
  pool**; 70.7 / 6.1 / 23.2 % over the re-routed validation labels — both
  populations are now named in the paper, see G9). The Daganzo backbone has only
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

### A4 — Schedule objective, Eq. (5) of the revision (Eq. (3) as submitted): coupling term removed, residual coupling stated

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

### A6 — Two cost lenses (new Section 2.3, Eqs. (3) and (4) of the revision)

- **Old:** a single cost accounting (every vehicle-day charged in full).
- **New:** routing lens `C_route = sum_i C_i` (the submitted accounting) and
  operator lens `C_op = sum_i (C_i − c_f v_i) + 6 c_f sum_h max_d v_hd`, i.e.
  1,134.90 EUR per peak vehicle per depot per week. Both lenses are evaluated
  against the same baseline; neither contains the service penalty. The paragraph
  after Eq. (4) discloses that the operator lens counts driver labour twice (the
  36 EUR/h route-time charge stays in the variable bucket while 6 c_f already
  contains the wage), so its absolute LEVELS are high by roughly a fifth, the
  time share of label cost, while savings are unaffected because the inflated
  term sits in baseline and scenario alike. Re-labelling with per_hour = 0 is
  named as future work (Task 15).
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
  a cost from one with a service figure from the other. The phrase
  "service-neutral" appears exactly once, in the negated form the spec requires:
  the manuscript states that the operator plan is *not* service-neutral.
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

- **New:** Read as a pure cost objective, NEITHER lens points to P = 0.25: at
  full adoption the saving decreases with P in both, so on cost alone the
  routing lens prefers P = 0 by 3.9 pp and the operator lens by 1.9 pp. P = 0.25
  is the recommendation because it is where the service side is priced in: at
  (0.25, 1) the operator plan saves 22.82 % operator and 17.07 % routing cost,
  cuts the summed depot peak by 17.1 % (as deeply as P = 0 does), and halves the
  additional wait (0.39 vs 0.77 d) for 1.9 pp of operator saving. Under a flat
  0.50 EUR discount (B9) it stops being a trade and is the best point on the
  grid outright. The earlier phrasing "both lenses point to P = 0.25" is
  withdrawn: it was contradicted by Table 2.
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
  `task-12-brief.md`. **SUPERSEDED by section G**: pooled tours *do* occur at
  theta = 1 (379 multi-cell groups), and the v6 validation shows the bias is
  not conservative in the savings sense — see G2.

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
per-cell bias evidence. After the review it also carries: the unit as "312
provider-postal-code cells formed by seven LSPs over 48 postal-code areas";
the truthful recommendation structure (both lenses prefer P = 0 on cost, by
3.9 and 1.9 pp; P = 0.25 recommended once waiting is priced and under a real
50 ct discount); the CV band as 52-83 % across P in [0.25, 0.75] under the
OPERATOR-POLISHED plan (66-83 % covers only P = 0.25-0.5; at P = 0.75 it is
52 %); the consolidated bias as "+2.5 to +3.5 %" rather than the 3-per-week
bucket alone; and the rural/structural claim replaced by the rotation
mechanism, since the per-area block it rested on is withdrawn (B12).
Evidence: §39.5, §40.15, §40.18, `tab_fleet_diagnostics_v2.csv`,
`_tab_chosen_v2.csv`.

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

## F. Fix round after review (2026-08-27)

Review: `.superpowers/sdd/2026-08-25-realistic-tours-implementation/task-14a-review.md`
(2 Critical, 6 Important, 11 Minor; spec compliance PASS, no wrong number found).
Entries A1-E above stand; the following are the additional claim changes.

### B12 — The carried-over per-area block is withdrawn, not merely marked provisional

- **Old** (submitted text, still present after the first part-A pass): "the
  per-area saving correlates strongly with parcels per drop-site (rho = -0.72),
  hub distance (+0.53) and area size (+0.31) ... rural areas save a median of
  25 % against 9 % for urban areas ... At low adoption (theta <= 0.3) the same
  mechanism pushes 35-44 % of service-bound cells into negative saving."
- **New:** the block is cut. In its place, one sentence stating that the
  per-area figures rest on the grid whose partial-adoption basis is retracted
  and are therefore not carried over, plus a pointer that the spatial breakdown
  is re-derived from the revised grid's per-cell costs (pending). What remains
  is the frequency-side evidence the revised grid does support, Fig. 6 (d)-(f).
- **Why:** "provisional" means same pipeline, newer numbers; this block's
  *source* is retracted. The theta <= 0.3 figure in particular comes from
  exactly the adoption range whose basis the paper withdraws 25 lines earlier.
- **Evidence:** §39.3 (roughly a quarter of demand at theta < 1 rode tours the
  baseline could not form); review C2/I6. Part B re-derives the breakdown from
  Task 11's per-cell plan costs.

### B13 — Fig. 6 (f) is a within-DHL statement

- **Old:** "the buckets below 13 cells consist almost entirely of DHL depots";
  the body read panel (f) as a general mechanism, and limitation (i) claimed
  "the finding that consolidation needs rotation is general".
- **New:** "exclusively" in the caption, followed by "panel (f) is therefore a
  within-DHL statement, not a general law"; the body names DHL as the only
  multi-depot network in the case study; limitation (i) now says the mechanism
  is evidenced within a single carrier's network and that neither its magnitude
  nor its generality can be established from these data.
- **Evidence:** `results/revision_2026_08_v5/figures/fig6_bucket_composition.csv`
  (buckets 1, 2-4 and 5-12 are all `single_carrier = True, DHL`),
  `task-13-report.md`.

### C5 — The unit is 312 cells over 48 postal-code areas

- **Old:** "the 312 postal-code areas", and "% of areas" throughout §3.2.
- **New:** "312 provider-postal-code cells formed by seven LSPs over 48
  postal-code areas" at first use in the abstract, in the data section and in
  the conclusion; "% of cells" in every distributional statement; Fig. 4 caption
  likewise. The methods already defined the cell correctly.
- **Evidence:** `_tab_chosen_v2.csv` at any grid point: 312 rows over 48 distinct
  PLZ and 7 providers (DHL 48, Amazon 47, DPD 47, Hermes 47, GLS 46, UPS 40,
  FedEx 37).

### C6 — Abstract corrections beyond the recommendation

- CV band: "66-83 % in that range" -> "52-83 % across that range under the
  operator-polished plan". At (0.75, 1) the reduction is 52 %, outside the old
  band, and the band had named no plan. Evidence: `tab_fleet_diagnostics_v2.csv`.
- Consolidated bias: "+3.5 %" -> "+2.5 to +3.5 %", so the modal 2-per-week
  bucket (+2.51 %, n = 458) is not dropped in favour of the 3-per-week bucket
  (+3.51 %, n = 363); the body now lists both. Evidence: §39.5.
- The rural/structural sentence is replaced by the rotation mechanism, since the
  per-area block it rested on is withdrawn (B12).

### C7 — Trim executed against the page budget

Cut in the reviewer's order: the per-area block (B12); the rule-(a) five-number
series reduced to two ranges; the instance-size accuracy series reduced to its
claim; the duplicated flat-bound sentence (stated once, kilo-euro figures
dropped); the P = 0.5 and P = 1 daily-share clauses of the mix paragraph;
Table 2's P = 1 and P = 2 rows moved into its caption. The feature-skew
limitation was KEPT, on the reviewer's advice. Remaining candidates are listed
in the header comment of `tbc_preprint_main.tex`.

### C8 — Minor corrections

Section 2.4 retitled "Schedule optimization and operator-cost polish" (it no
longer describes fleet balancing); the 159 km2 cap now reads "the bias drifts
negative, reaching -4.3 % at the training maximum of 358 km2" (the -4.33 % is
measured at 358, not immediately past 159); "Six limitations" -> "Seven
limitations and one methodological caveat"; the equation numbers in A4 and A6
and in the response skeleton corrected to the revision's numbering; the
"service-neutral is not used anywhere" sentence in A8 corrected.

---

## E. Carried over, still open

1. HAGRID acronym full form (reviewer point 3) — still not supplied.
2. Final 8-page acceptance in the MiKTeX/Elsevier environment (Lasse), now with
   a mandatory trim pass (C4).
3. Stale `%TODO` headers in the Elsevier master.
4. Human read-through of the preprint.
5. Part B: replace every `\provisional{}` value, insert the validation
   paragraph, sync the figures, and do a final lens/plan consistency pass.

---

# F. PART B — final numbers from grid v6 (2026-08-28)

**Status: PART B done.** Every `\provisional{}` value whose source was the
optimization grid has been replaced from
`results/revision_2026_08_v6/` (grid v6 = v5 + the certified BundleHead, 616/616
triples, no errors). **21 markers remain**, all of them validation-dependent
(predicted-vs-actual, actual saving in per cent, MAPE/bias); each carries a
`% PART C` comment and is replaced from the v6 VROOM validation report.

**Baseline change, read this first.** Grid v6 prices the pooled tours that the
*baseline itself* runs with the bundle head, so its θ = 0 denominator is
1,898,091 EUR/wk routing and 2,098,401 EUR/wk operator, **0.6 % below** v5's
1,909,432 / 2,109,742. Σ hub peaks are unchanged at 1,239. A v6 saving must
never be normalised against a v5 or 2026-07 denominator. The manuscript now says
this once, in the paragraph that introduces Table 1.

Evidence keys below: table files are relative to
`results/revision_2026_08_v6/tables/` unless stated; `_peek/` is
`results/revision_2026_08_v6/_peek/`; `§n` is
`docs/PAPER_COMPENDIUM_2026_05_24.md`.

## B-I. Numbers replaced (old = part A / v5, new = v6)

### Headline, θ = 1 — `tab_headline_theta1_v2.csv`, §40.21

| Quantity | Old (v5) | New (v6) | Where |
|---|---|---|---|
| Routing saving, routing plan, (0,1) | 23.1 % | **22.6 %** | abstract, §3.2, Table 2, Conclusion |
| Operator saving, routing plan, (0,1) | −7.8 % | **−8.4 %** | abstract, §3.2, Table 2, Conclusion |
| Operator saving, operator plan, (0,1) | 24.7 % | **24.3 %** | abstract, §3.2, Table 2, Conclusion |
| Routing saving, operator plan, (0,1) | 20.4 % | **20.0 %** | §3.2 |
| Σ hub peak vs base, routing plan, (0,1) | +34.5 % | +34.5 % (unchanged) | abstract, §3.2, Conclusion |
| Σ hub peak vs base, operator plan, (0,1) | −16.9 % | −16.9 % (unchanged) | abstract, §3.2, Conclusion |
| Routing pp given up by the polish | 2.7 pp | 2.7 pp (unchanged) | §3.2, Conclusion |
| Operator saving, operator plan, (0.25,1) | 22.8 % | **22.6 %** | §3.2, Conclusion |
| Routing saving, operator plan, (0.25,1) | 17.1 % | **16.7 %** | §3.2 |
| Σ hub peak vs base, operator plan, (0.25,1) | −17.1 % | **−17.2 %** | §3.2 |
| Cost-only gap P = 0 vs 0.25, operator lens | 1.9 pp | **1.7 pp** | abstract, §3.2, Conclusion |
| Cost-only gap P = 0 vs 0.25, routing lens | 3.9 pp | 3.9 pp (unchanged) | abstract, §3.2, Conclusion |
| Worst operator-lens value of the routing plan | −10.4 % at (0,0.9) | **−10.3 %** at (0,0.9) | §3.2 |

Table 1 (`tables/tab_two_lens.tex`, the only table in the document) was rebuilt cell by cell from
`tab_headline_theta1_v2.csv` and **regained its P = 1 row** (part A had moved it
into the caption), which restores table backing for the mean-frequency figure
quoted at P = 1 — this closes handover item M-g. The caption's tail now carries
P = 2, 5 and 10 and no longer claims the routing-optimal plan yields 0 % there.

### Frequency mix — recomputed from `_tab_chosen_v2.csv` + `enumerate_valid_schedules()`

| Quantity | Old (v5) | New (v6) |
|---|---|---|
| Two-day share, routing plan, (0,1) | 97.4 % | 97.4 % (unchanged) |
| f-bar, routing plan, (0,1) | 2.03 | 2.03 (unchanged) |
| f-bar, routing plan, P = 0.5 | 3.95 | **4.12** |
| f-bar, routing plan, P = 1 | 4.93 | **5.01** |
| Two-day share, operator plan, (0,1) | 72.8 % | **72.4 %** |
| f-bar, operator plan, (0,1) | 2.38 | **2.39** |
| Cells off daily, operator plan, (5,1) | 4.8 % | 4.8 % (unchanged) |
| Wait, (0.5,1), routing -> operator plan | 0.21 -> 0.23 d | **0.20 -> 0.22 d** |

**Claim withdrawn.** Old: "for P >= 5 and theta >= 0.3 the routing-optimal plan
reverts to fully daily delivery." On v6 that is false: 0.6 % of cells stay
non-daily at (5,1), 3.5 % at (5,0.3) and 15.4 % at (5,0.1). New text names the
0.6 % at full adoption and the 15.4 % at (5,0.1).

### The retracted (10, 0.1) bulge — `tab_grid_full_v2.csv`, §40.22

| Quantity | Submitted | Old (v5 text) | New (v6) |
|---|---|---|---|
| Routing saving, routing plan | 3.6 % | 0.03 % | **0.40 %** |
| Operator saving, routing plan | — | 0.02 % | **0.36 %** |
| Non-daily cells, routing plan | 42 % | 2.9 % | **9.6 %** |

The retraction of the *mechanism* stands unchanged, but the wording no longer
says the corner is "flat". It now says the corner "shrinks by almost an order of
magnitude but does not disappear", per §40.22's ruling ("reduced to", not
"gone"). The 0.74 % / 22.79 -> 22.22 % symmetric-recomputation bound is
unchanged (it is a property of the submitted grid, not of v6). The surviving
(0, 0.1) pair was cut in the trim; on v6 it would have read 1.8 % routing saving
for the routing-optimal plan and 4.1 % operator saving for the operator-polished
plan (two plans, as in part A's 1.4 % and 3.9 %).

### Fleet — `tab_fleet_diagnostics_v2.csv`

| Quantity | Old (v5) | New (v6) |
|---|---|---|
| Baseline Mo–Sa CV | 0.139 | 0.139 (unchanged) |
| CV at (0.25,1) | 0.024 (−83 %) | **0.023 (−84 %)** |
| CV at (0.5,1) | 0.047 (−66 %) | **0.044 (−68 %)** |
| Grid-max CV reduction | 88 % at (0,0.7) | **87 % at (0,0.7)** |
| Abstract CV band over P in [0.25,0.75] | 52–83 % | **51–84 %** |
| Σ peak vs flat bound, (0,1) | 1,030 vs 980 | 1,030 vs 980 (unchanged) |
| Residual gap, (0.25,1) | 32 | 32 (unchanged) |

**Old claim retracted (part A had already dropped it; recorded here for the
response letter).** The submitted "total weekly fleet declines by 6.3–6.8 % at
theta = 0.1" is replaced by the two quantities that actually exist under the
tour rule: routing-plan vehicle-days −1.5 to −1.9 % and operator-plan Σ hub
peaks −4.8 to −5.4 % over P in {0, 0.25, 0.5} at theta = 0.1
(`tab_grid_full_v2.csv`, §40.21). The old figure was mostly spatial pooling of
non-consolidated demand, which the universal tour rule removes.

**New distinction (13B observation O2).** Fig. 5 (d) plots the *system peak day*
(−17.5 % at (0,1)), not the summed depot peak (−16.9 %). The manuscript now
states both and labels each; "peak fleet" alone no longer appears — the abstract
and the Conclusion say "summed depot peak".

### P* by lens — `tab_pstar_knees_v2.csv`, §40.22

- Routing lens (routing-optimal plan): unchanged from the submission —
  Amazon 0.25, DHL 0.25, FedEx 0.5, Hermes 0.5, UPS 0.5, DPD 0.75, GLS 0.75.
- Operator lens (operator plan), **v6 corrects part A's v5 set**:

| Provider | Routing P* | Part A said (v5) | v6 |
|---|---|---|---|
| Amazon | 0.25 | 0.5 | **0.5** |
| Hermes | 0.5 | 0.75 | **0.75** |
| FedEx | 0.5 | 0.75 | **0.5 (no move)** |
| GLS | 0.75 | 0.75 (no move) | **1.0 — outside the 0.25–0.75 band** |
| DHL / UPS / DPD | 0.25 / 0.5 / 0.75 | no move | no move |

The manuscript now says the three-class taxonomy is a **routing-lens**
statement, names Amazon and Hermes as the two that move up a class, and states
that GLS leaves the band entirely and is therefore unclassified in the operator
lens. The abstract carries the same qualification. This is the strongest
qualification the revision puts on a submitted result and is called that in the
text.

### The discount scenario — `_peek/discount_scenarios_v6.csv`, §40.22

Moved out of the main text into `paper/EWGT_2026_rev1/supplementary.tex`,
Section S1 (page trim; one sentence with the break-even range stays in §3.2).
All values from the CSV, which counts the realised demand on skipped days; the
deep-dive prose in `DEEP_DIVE_V6_PAPER_IMPACT.md` used an even-weekday
approximation and is **not** the source.

| Quantity (theta = 1, operator plan) | Old (v5) | New (v6) |
|---|---|---|
| Delayed parcels, P = 0/0.25/0.5/0.75/1 (thousand) | 680/404/248/165/111 | **679/405/237/153/109** |
| Rule (a) net, operator lens (%) | 24.7/17.0/11.6/8.4/6.2 | **24.3/16.6/11.2/8.1/6.1** |
| Rule (a) net, routing lens (%) | 20.4/10.6/5.6/2.6/1.0 | **20.0/10.2/5.1/2.4/0.8** |
| Rule (b) net, operator lens (%) | 8.6/13.2/12.6/10.9/9.2 | **8.1/12.9/12.1/10.4/9.1** |
| Rule (b) net, routing lens (%) | 2.6/6.5/6.7/5.4/4.3 | **2.1/6.1/6.0/5.0/4.1** |
| Break-even, operator lens (EUR) | 0.77 … 2.24 | **0.75/1.17/1.57/1.93/2.25** |
| Break-even, routing lens (EUR) | 0.57 … 1.24 | **0.56/0.78/0.98/1.12/1.21** |
| Partial adoption, P = 0.25, routing lens | −0.3 … −2.4 % | **+0.1 … −2.1 %** |
| Partial adoption, P = 0.25, operator lens | +3 … +6 % | **+3.4 … +6.2 %** |

**Claim corrected.** Part A said the routing lens prefers P = 0.5 by 0.2 pp
under the flat rule. On v6 the routing lens returns 6.052 % at P = 0.25 against
6.043 % at P = 0.5 — **0.01 pp, a tie at this grid's resolution**. The
abstract, §3.2, the Conclusion and Supplementary S1 now say so, and the operator
lens's margin for P = 0.25 grows from 0.6 to **0.8 pp**.

### Bundle-head coverage — `tab_head_usage_summary*_v2.csv`, deep dive §1

Part A reported one number, "certified on 44.0 % of the pooled instances that
occur". That conflated three different rates. The manuscript now reports all
three and says they are not interchangeable:

- **44.0 %** — pre-run certified coverage of the expected pooled population.
- **53.9 %** — of the pooled tours that actually occurred in the optimized grid.
- **27.2 %** — of the pooled euro (68.9 % of tours and 53.0 % of euro at full
  adoption); the conservative fallback keeps the largest groups.

Limitation (v) now quotes the two realised rates rather than the pre-run one.

### Partial-adoption mechanism (new; Amendment 6, §40.20)

New paragraph in §3.2, all numbers recomputed on v6 from
`tab_grid_full_v2.csv` (`mean_days_plan1` / `mean_days_plan2`, the cell-level
means, **not** the `*_provmean` columns) and `_peek/results_overview_v6.csv`:
at P = 0.25 the mean delivery frequency rises from theta = 0.8 to 0.9,
4.93 -> 5.16 d (routing plan) and 4.68 -> 4.89 d (operator plan), while the
express residual falls from 9.6 % to 7.4 % of routing cost; at theta = 1 the residual vanishes and the frequency
drops to 3.10 / 3.27 d. Two causes are named (the penalty scales with theta; the
thin residual no longer fills a vehicle in small depots), and the theta =
0.9 -> 1 jump is attributed to the disappearance of the express obligation.
§40.20's v5 figures (4.66 -> 4.83 d, 8.3 -> 6.1 %) are superseded by these.
The euro-per-vehicle-day figures mentioned in the Part B brief (339 -> 519 EUR
vs ~298 EUR) are **not** in the compendium and could not be reproduced from any
v6 table, so they are not in the manuscript; the cost-share evidence above
carries the mechanism instead.

### Spatial breakdown — the part A "(pending)" is closed

Part A cut the submitted per-area block as resting on the retracted grid and
left a "(pending)" placeholder plus a `% PART B EXIT GATE` comment questioning
the abstract's "spatial signature" claim. Grid v6 plus
`72_per_cell_costs_v2.py` supply the replacement, so the placeholder, the exit
gate and the conditional "should the spatial pattern be confirmed" hedge are all
gone, and the abstract's claim and contribution 4 stand. Values from
`tab_per_cell_structural_v2.csv` (cell level, routing lens, operator plan, each
LSP at its own routing-lens P*, theta = 1):

| Breakdown | Submitted | v6 |
|---|---|---|
| Hub distance, Q1 -> Q4 | 8 % -> 28 % | **2.0 % -> 25.8 %** |
| Area size, Q1 -> Q4 | 10 % -> 25 % | **8.8 % -> 24.6 %** |
| Parcels per drop-site, Q1 -> Q4 | 27 % -> 4 % | **24.2 % -> 2.3 %** |
| Rural vs urban median | 25 % vs 9 % | **23.7 % vs 7.9 %** |
| Service-bound class cap | 10 % | **6.5 %** |
| Hybrid / cost-aggressive at their P* | 22–24 % | **20.3 / 17.3 %** |
| Service-bound cells with negative saving, theta <= 0.3 | 35–44 % | **21–41 %** |

The ordering is reproduced in every breakdown, so the equity concern is restated
unconditionally. The rho values of the submitted version are **not** restored:
the medians shown in the figure carry the claim, and re-deriving correlations
would add numbers no panel shows.

**New sentence (13B observation O1).** Fig. 6 (b)–(f) are flat at exactly 0.0 %
for theta <= 0.9 and jump only at theta = 1. The text now explains why — below
full adoption the non-willing parcels still force a daily tour, so at each
carrier's own P* the *median* cell saves nothing — so that a reader does not
take the panels for a rendering fault.

## B-II. Figures and captions

Per the author's steer of 2026-08-28 (Amendment 5), the accepted paper keeps its
**submitted figure layouts**; only the numbers change. Figs. 4, 5 and 6 are
re-rendered on v6 by the frozen builders `30_`/`31_`/`32_` through
`74_v2_to_legacy_tables.py --render`, and part A's 2x3 two-lens caption rewrites
are therefore **undone**:

- **Fig. 4** — back to the submitted single-plan mix caption, with three
  corrections: "cells" not "areas"; the plotted plan is named (stage 1); and the
  false frequency-invariance claim is replaced by the measured fact that the
  delivery-day count differs in **20.1 % of the 27,456 cell–grid-point pairs**
  (5,525), with a pointer to Supplementary Figs. S1–S2 (13B observation O4).
- **Fig. 5** — back to the submitted caption. (a), (b) = cost saving of the
  routing-optimal and of the operator-polished plan, both in the paper's routing
  cost; (c) wait; (d)–(f) fleet metrics of the operator plan; the CV baseline is
  named as this grid's own 0.139. Supplementary Figs. S3–S4 carry both lenses
  and the two off-diagonal lens/plan combinations.
- **Fig. 6** — back to the submitted caption: Pareto frontier (a) and median
  per-cell saving by carrier type, region type, hub distance, area size and
  parcels per drop-site (b)–(f), at each LSP's P*. The caption now defines that
  P* as the routing-lens knee (13B observation O3) and states the express
  allocation rule (per realised tour, O6). The hub-size panel that carried the
  within-DHL rotation caveat is not in the submitted layout, so that caveat now
  lives in the body text, in limitation (i) and in Supplementary Fig. S6.

The figure PDFs in `paper/EWGT_2026_rev1/figures/` are still the previous
renders; `71_sync_paper_figs.py` is out of this task's scope and must report an
md5 PASS before any build circulates.

## B-III. New in the methods

- **G1a audit** (§40.23, `gates_report.md`): §2.4 now records that for every
  cell that clears the tour minimum on all six weekdays, 1,651 of 1,656
  cell–penalty pairs reproduce the plain-enumeration optimum, and the five that
  differ are single-day neighbours worth at most 0.5 % of the cell's weekly
  objective — near-ties produced by the paired move in the local search, not a
  breach of the decoupling.
- **Baseline note**: the −0.6 % head effect on the θ = 0 denominator, with the
  explicit instruction never to normalise across grid versions.

## B-IV. Validation — the one block still provisional

Rewritten to keep three evidence levels apart, per the deep dive's §7:
(1) the single-cell surrogate validation; (2) the bundle head's out-of-fold
certification, which is a cross-validation result and not a re-routing; and
(3) re-routing of the optimized plans, which is under way. **The false claim
"at theta = 1 no pooled tour exists at all" is removed** — pooled delivery
groups occur at full adoption (379 multi-cell groups, `tab_head_usage_summary_
theta_kind_v2.csv`); the partial-adoption point is now motivated by pooled tours
being *far more frequent* there, not by their absence at theta = 1.

All 21 remaining `\provisional{}` markers are in this subsection, in the
abstract's validation clause and in the feature-skew caveat, each with a
`% PART C` comment.

## B-V. Page trim

17 pages -> **16 pages** (tectonic, preprint layout; the submitted version is
11). Executed:

1. The discount subsection moved to `supplementary.tex` S1; one sentence with
   the break-even range kept in §3.2 with a pointer.
2. Limitation (vii), the labour double count, moved to supplementary S2; one
   clause kept in limitation (vi) and one in §2.1.
3. The (0, 0.1) surviving-effect pair cut from the retraction paragraph
   (header-comment candidate a).
4. Fig. 5's panel list dropped with the return to the submitted caption
   (candidate c); Fig. 4 and Fig. 6 captions shortened likewise.
5. The CV cross-version warning stated once (limitation vi) instead of twice.
6. A density pass on nine passages that lost no claim (surrogate calibration,
   three-stage progression, feature list, willingness power law, the Nash
   clause, the shadow-price clause, the retraction's asymmetry sentence, two
   abstract sentences).

Candidate b (the discount subsection's partial-adoption sentence) is subsumed by
item 1. **Not** executed, and left as an author decision: any further reduction
now has to come out of substance — the abstract, the surrogate subsection, or
the validation subsection are the only blocks left with a page in them, and each
carries claims that a review round explicitly asked for. The four protected
items (two-lens table, the P* lens statement, the retraction, the limitations)
are all intact.

## B-VI. Supplementary document

New file `paper/EWGT_2026_rev1/supplementary.tex` (7 pages, compiles with
tectonic). Section S1 = the discount scenario with the v6 table; Section S2 =
the labour double count; Figs. S1–S11 = the `supp_*` set from
`results/revision_2026_08_v6/figures/` plus the maps. It reaches those PDFs
through a second `\graphicspath` entry,
`../../results/revision_2026_08_v6/figures/`, so it builds today; once
`71_sync_paper_figs.py` has copied them into `paper/EWGT_2026_rev1/figures/`
under the same stems the first entry wins and the relative path can be dropped.
This closes handover item M9 — the main text's "supplementary material"
pointers now resolve.

---

# G. PART C — the v6 VROOM validation, and the part-B review fixes (2026-08-28)

**Status: PART C done.** The manuscript carries no `\provisional{}` marker any
more (the macro is deleted, so no undecodable superscript reaches print). One
`% PART C3` marker remains, for validation item 3.

Source for every number in this section:
`results/revision_2026_08_v6/validation/validation_report.md` (items 0, 1, 2)
and `validation/tab_vroom_v2.csv`, independently recomputed over
`item in {0,1,2}` with the report's own formulas
(`variable = Σ(vroom_cost_eur − 189.15·vroom_n_routes)`,
`peak_h = max_d Σ vroom_n_routes of hub h`,
`OpCost = Σ variable + 1134.90·Σ_h peak_h`). Compendium §40.24 / §40.25 carry the
same figures.

## G1 — The validation itself (new, replaces the part-A/B placeholder text)

7,610 instances re-routed: the complete daily baseline (item 0, 1,683
instances), the operator-polished plan at θ = 1 and P ∈ {0, 0.25, 0.5, 0.75}
(item 1) and the routing-optimal plan at θ = 1 and P ∈ {0, 0.25} (item 2). One
PARTIAL (a DHL depot instance, PLZ 30855, one unassigned job) — excluded from
the error statistics, kept in the cost totals.

| Quantity | Value | Source |
|---|---|---|
| Overall error, 7,609 clean instances | 3.51 % MAPE, +3.05 % bias | recomputed |
| Daily baseline | 5.80 % MAPE, +5.39 % bias | recomputed |
| — single-cell tours / pooled tours | +4.88 % / +14.50 % bias | report, "Routing lens" |
| Consolidated points | 2.85 % MAPE, +2.39 % bias; 2.3–3.7 % MAPE per point | recomputed |
| Baseline totals, predicted vs actual | routing 1,898,091 / 1,818,360 €; OpCost 2,098,401 / 2,016,778 € | report, "Predicted vs actual" |
| Surrogate above solver | +4.4 % (routing) / +4.0 % (OpCost) at the baseline; +0.9 … +2.9 % at the points | derived |
| Σ depot peaks, predicted / actual | 1,239/1,249; 1,030/1,032; 1,026/1,026; 1,062/1,064; 1,091/1,096; 1,666/1,667; 1,314/1,314 | report, "Both lenses" |

## G2 — The claim that changes direction

- **Old (submitted, and repeated in parts A and B as "evidence on the
  surrogate"):** "the surrogate is conservative — 22.8 % predicted against
  23.7 % actual — so the reported savings are understated rather than
  overstated."
- **New:** the surrogate over-prices *every* instance class, and the thin daily
  baseline tours (+5.4 %) much more than the consolidated ones (+2.4 %). The
  over-priced baseline is the denominator of every saving, so **the predicted
  savings are an upper bound, by 1.3 to 2.5 pp**.
- **Why the submitted version saw the opposite:** its comparison predates the
  universal tour rule, and no earlier validation re-routed the daily baseline
  itself. Item 0 supplies that baseline for the first time.
- **Realised against predicted (θ = 1):**

| plan / lens | P = 0 | 0.25 | 0.5 | 0.75 |
|---|---|---|---|---|
| operator plan, routing lens | 17.48 (20.0) | 14.53 (16.7) | 10.59 (12.3) | 7.69 (9.0) |
| operator plan, operator lens | 22.08 (24.3) | 20.73 (22.6) | 16.29 (17.8) | 12.84 (14.1) |
| routing plan, routing lens | 20.58 (22.6) | 16.43 (18.7) | — | — |
| routing plan, operator lens | −12.09 (−8.4) | +5.22 (+7.8) | — | — |

  (20.58 % is the clean figure; 19.93 % if the PARTIAL row is kept.)
- **Nothing changes sign or order, and the central result strengthens:** the
  operator-lens penalty of the routing-optimal plan is *larger* in reality
  (−12.1 % against −8.4 % predicted at (0, 1)), so a plan chosen on per-day
  routing cost is a worse weekly proposition than the surrogate suggested.
- The fleet counting rule `⌈p/Q⌉` is near-exact: over all seven re-routed
  settings the summed depot peaks never differ by more than five vehicles in a
  thousand.

Text affected: the abstract's closing clause, the whole of §3.3, limitation
(iv), and the feature-skew caveat (now "3.51 % MAPE at +3.05 % bias over 7,609
solver-routed instances", replacing the v5 "3.03 % / +2.73 % / 1,247 cells").

## G3 — Gate G1a in the methods (compendium §40.23, `gates_report.md`)

§2.4 now states the tolerance rule rather than only the count: of 1,656
cell–penalty pairs, 1,651 agree exactly with plain enumeration and five are
accepted under a stated tolerance — a single-day change of the pattern whose
objective differs by at most 20 € or 0.5 % of the cell's weekly objective,
whichever is larger — all five arising from the paired move in the local search.
`62_` on v6 reports **G1a PASS** (0 hard mismatches, 5 tolerated).

## G4 — Fixes from the part-B review (`task-14b-review.md`)

| ID | Fix |
|---|---|
| **I1** | The "all five breakdowns are flat at exactly zero for θ ≤ 0.9" sentence was false — five of six panels carry a rising mid-θ line. Replaced by the reviewer's verified wording: the medians stay in low single digits and jump at θ = 1, and only the service-bound class, urban cells and the nearest hub-distance quartile are flat at exactly zero. |
| **I2** | The retracted "at θ = 1 no pooled tour exists" removed from `RESPONSE_TO_REVIEWERS_skeleton.md` Theme A and Theme G (it had been removed from the manuscript in part B but not from the reviewer letter). Theme G rewritten around the actual reason the partial-adoption regime is unvalidated. |
| **I3** | Closed by the controller's `71_` run: `paper/EWGT_2026_rev1/figures/fig4/5/6*.pdf` are now md5-identical to `results/revision_2026_08_v6/figures/manifest.json`. Verified here. |
| **I4** | Ruled by the controller: the in-figure label strings are corrected in the frozen builders by a separate task, so the captions describe the corrected labels and carry no clause about legacy strings. |
| **I5** | The abstract's superseded validation clause replaced by the real v6 result, and the `\provisional` macro deleted entirely, so no undefined superscript reaches print. |
| **I6** | `1.91 M€` → **1.90 M€** of baseline operating cost (the v5 value had survived because it was never wrapped). |
| **M1** | Table 1's caption: the "all but 0.6 % of cells" clause scoped to P ≥ 5 (it is 9.94 % at P = 2). |
| **M2** | Abstract: the 51–84 % CV band scoped to full adoption. |
| **M3** | Supplementary Fig. S11 caption: maximum area wait 0.7 → **0.6 d**. |
| **M4** | Supplementary Fig. S10 caption: "a two-day cluster" → "a single area reaches two" (n = 1 of 48). |
| **M5** | The 159 km² cap: "the 95th percentile … −4.3 % at the training maximum" → "above which only 6.4 % of training rows lie … averaging −4.3 % over the 159–358 km² range". |
| **M6** | "triples to 7.1 %" → "more than doubles, to 7.1 %" (7.14 against 2.91 % is a factor 2.4). |
| **M7** | "comfortably above the 50 cents" → "above the 50 cents at every operating point" (the P = 0 routing-lens break-even is 0.558 €). |
| **M8** | Fig. 5's caption now names the plan for panel (c). |
| **M9** | This changelog corrected: the mechanism frequencies to the cell-level `mean_days_*` (4.93 → 5.16, matching the manuscript), the (10, 0.1) operator saving to 0.36 %, and the cut (0, 0.1) pair labelled with its two plans. |
| **M10** | "Table 2" → "Table 1" throughout; the validation subsection is §3.3 and the limitations §3.4. |
| **M11** | The file header no longer says "UNDER REVIEW — not yet accepted". |
| **M13** | Closed: `62_` on v6 gives G1a PASS; §2.4's sentence matches the ruling (see G3). |

`M12` needed no change: the express €/vehicle-day figures are in §40.20 but are
ruled not citable until `77_mechanism_v2.py` lands, so not writing them was and
remains correct.

## G5 — Page budget

Not applied. The review's §6.3 cut list is prepared as
`paper/EWGT_2026_rev1/page_budget_cuts.patch` with
`page_budget_cuts_README.md`; the author decides. The manuscript as committed is
**16 pages** and the supplementary **7**.

## G6 — Still open

1. Validation item 3, the partial-adoption point (P, θ) = (0.25, 0.5) on the
   operator plan — the single `% PART C3` marker.
2. CO₂: the manuscript contains no CO₂ statement at all, so there is nothing to
   refresh; `70_` was not re-run because `validation/67.lock` exists.
3. `71_` does not yet carry `supp_fig7_*` or the four `supp_map_*` /
   `supp_penalty_*` figures into `paper/EWGT_2026_rev1/figures/`; the
   supplementary reaches them through its relative path.
4. HAGRID acronym, the Elsevier 8-page check, the stale `%TODO` headers, the
   human read-through.

---

## G7 — Fix round 2 (`task-14c-review.md`, verdict SPEC FAIL)

### The blocker: three paragraphs were commented out of the printed paper

Part C added five `% src:` provenance comments. **Three of them were placed at
the start of a line that still carried body text**, and a LaTeX `%` swallows the
rest of its line, so the printed manuscript silently lost:

| site | what print lost |
|---|---|
| `:290` | the only in-text references to Fig. 1 and Fig. 2, the code-availability footnote, the postal-code-decomposition sentence, and the full stop ending the preceding sentence |
| `:364` | the cross-provider independence statement, the shadow-price definition of $P$, and the definition of the entire $(P, \theta)$ grid |
| `:497` | the equity sentence (a protected item), the per-depot mechanism sentence and its Fig. S6 pointer |

Consequences measured: the bibliography fell from **23 to 21** entries
(`BoydVandenberghe2004` and `Pereira04032017` uncited), and the "16 pages,
unchanged" of the part-C report was an artefact — with the text live the
document is **17 pages**. LaTeX cannot warn about this: no `\ref` breaks and
unused `\label`s are silent.

**Fixed.** Every `% src:` comment now sits on its own line; all three passages
are restored and verified string-for-string against `32cd104`. A guard now runs
before commit and checks four things: no comment line hides a control sequence
(`\section|caption|label|ref|autoref|cite|citep|footnote|url|includegraphics|item|begin|end`),
the PDF page count, `\bibitem` count = 23, and that every cited key resolves.
Current state: **17 pages, 23 bibitems, 0 hidden control sequences, 0 unresolved
citations.**

### Number bases in the validation subsection

Part C mixed two bases in one paragraph and disclosed neither. The subsection now
states **one** basis — the *clean* basis, in which the single PARTIAL instance is
dropped from the error statistics **and** from the cost totals — and says so, with
the alternative given: keeping it lowers the one saving it touches from
$20.6$ to $19.9\%$.

| ID | Old | New | Recomputed value |
|---|---|---|---|
| **I2** | "excluded from the error statistics while remaining in the cost totals", beside a clean-basis $20.6\%$ | one named basis + the $19.9\%$ alternative | 20.580 clean / 19.930 incl. PARTIAL |
| **I2/M3** | point gaps "$0.9$ to $2.9\%$" (PARTIAL-inclusive, routing lens only, presented after a both-lens sentence) | "$1.2$ to $2.9\%$ and $0.5$ to $2.5\%$ respectively", clean basis, both lenses named | routing +1.246…+2.855, operator +0.49…+2.54 |
| **I3** | "The shortfall is $1.3$ to $2.5$~pp **at every point**" — contradicted four lines later by $-12.1$ vs $-8.4\%$ | "$1.3$ to $2.5$~pp for the operator-polished plan and $2.1$ to $3.7$~pp for the routing-optimal one" | operator plan 1.264–2.481; routing plan 2.057–3.726 |
| **I4** | "never differ by more than five vehicles in a thousand" — false at the baseline | "…at the six consolidated settings, or by ten in the baseline's $1{,}239$" | baseline 1,239/1,249 = 10; max elsewhere 5 of 1,091 |
| **I5** | "a realized $23.7\%$ against $22.8\%$ predicted" | "a realized $24.3\%$ against $22.8\%$ predicted at $P = 0$, and called the surrogate conservative" | `paper/EWGT_2026/tbc_preprint_main.tex:387`, verbatim; 23.7 % was revision 1's re-derivation |
| **M1** | "positive in every group we can form" | "positive in every group we report" | DHL at (item 2, P = 0) is −0.29 % |
| **M2** | the abstract's biases and §3.3's gaps are different quantities, undisclosed | "mean per-tour bias" in one, "total-cost gaps, not the mean per-tour biases above" in the other | — |
| **M5** | the Conclusion restated the predicted figures with no upper-bound caveat | "All figures above are the surrogate's; re-routing them puts the realized savings $1.3$ to $3.7$~pp lower" | — |

The same four defects were corrected in `RESPONSE_TO_REVIEWERS_skeleton.md`.

### The page-budget patch, regenerated

Regenerated on top of the I1 fix (the old patch would have silently locked the
`:290` and `:497` deletions in). Four casualties from the review's §5 are
repaired:

1. **The G1a tolerance sentence stays in §2.4** — it is an explicit part-C brief
   requirement and answers a reviewer point, so block 4 now moves only the
   restart-stability sentence.
2. **Block 8 is dropped.** Its premise was false: Table 1's caption carries
   "from $P = 5$ on **it** delivers daily in all but $0.6\%$ of cells" about the
   *routing-optimal* plan, not the operator polish's $4.8\%$ / $\bar f = 5.95$ at
   $(5, 1)$ — a distinctive two-plan finding that would have vanished from both
   documents.
3. **The cross-provider independence statement is kept verbatim** in
   Supplementary S4 (the previous version repunctuated it, which is why a search
   found no hits).
4. **Limitation (vi) keeps its labour clause** and the $0.139$ vs $0.135$
   cross-version numbers; block 14 now only removes the duplicated warning.

### Not fixed here

`M4` — five supplementary figures (`supp_fig7_*` and the four `supp_map_*` /
`supp_penalty_*`) still resolve through `../../results/`, so a standalone build
of the paper folder hard-fails. They are registered in the 70_ manifest but have
**not** appeared in `paper/EWGT_2026_rev1/figures/`; the six that 71_ does carry
are already referenced by their destination names.

---

## G8 — Fix round 3

### Validation item 3: partial adoption is now validated

The last `% PART C3` marker is gone; `grep -c "PART C"` returns **0**.
Item 3 of the v6 validation re-routed $(P, \theta) = (0.25, 0.5)$ on the
operator-polished plan over a **stratified subset of 1,000 of that point's 1,594
instances** (every instance of the three smallest providers plus at least half of
each remaining provider's demand, drawn round-robin over instance-kind and
n_jobs-tercile strata; `validation/G6_sampling_note.md`). 0 PARTIAL. Every value
below was recomputed from `validation/tab_vroom_v2.csv` (`item == 3`) and matches
`validation_report.md` exactly.

| instance kind | n | MAPE % | bias % | Σ pred € | Σ actual € | Σ gap % |
|---|---:|---:|---:|---:|---:|---:|
| delivery_single | 626 | 4.33 | +3.96 | 826,520 | 802,847 | +2.95 |
| delivery_group | 65 | 7.43 | +6.52 | 69,704 | 66,016 | +5.59 |
| express_single | 230 | 12.90 | +12.03 | 189,966 | 170,528 | +11.40 |
| express_group | 79 | 22.64 | +21.46 | 77,018 | 65,051 | +18.40 |

Totals: routing 1,163,209 vs 1,104,442 € (**+5.32 %**); OpCost 1,354,818 vs
1,299,077 € (**+4.29 %**); Σ hub peaks 811 predicted vs 819 actual (**−0.98 %**).

**What it means, and what is deliberately not said.** The error is strongly
kind-dependent: express tours — the part of a partial-adoption plan that only
consolidation creates — are the least accurately priced, at $+12$ to $+21\%$.
That bias runs **opposite** to the one at full adoption: it inflates the
consolidated schedule's cost and makes consolidation look *less* attractive than
it is, where the over-priced baseline at $\theta = 1$ makes it look *more*
attractive. The two roughly offset here ($+5.3\%$ for the point against
$+4.4\%$ for the baseline).

**No realized saving is quoted for this point.** The report's item-3 saving row
(+38.7 %) divides a 1,000-instance subset by the full baseline and is
meaningless; `67_` is being corrected to print n/a. The manuscript says so
explicitly.

Limitation (iv) rewritten to match: the partial-adoption regime is validated at
one point and only on a subset, its savings remain predictions, and the express
tours it creates are named as the least accurately priced part of the model.

### Supplementary figures now build standalone

`71_sync_paper_figs.py` (PASS, 32 copies) has landed the remaining seven stems in
`paper/EWGT_2026_rev1/figures/`. The five `\includegraphics` that still reached
into `../../results/` now use the tracked destination names
(`supp_map_saving_P`, `supp_penalty_raumtyp`, `supp_map_freq_theta`,
`supp_map_wait_theta`; `supp_fig7_fleet_week_classes` already matched), the
results path is out of `\graphicspath`, and the header note is rewritten around
the one remaining source. **Verified by copying the paper folder alone into a
scratch tree and building there: 7 pages, no "Unable to load" error.** This
closes M4 and the standing concern from fix round 2.

### Patch: the cross-document `\ref`

`page_budget_cuts.patch` moved a sentence into Supplementary S4 that referred to
`Section~\ref{sec:schedule_opt}`. That label lives in the main tex and the two
documents are built separately, so the patched supplementary printed
"Section ??". It now reads **"Section~2.4"** as literal text, the convention the
file already uses. The patch was regenerated on the fix-round-3 base.

Patched page counts move with the manuscript: **17 → 16** (not 15, as in round 2)
because item 3 added roughly 1,200 characters to §3.3. Both patched documents
build clean at 23 bibitems.

## G9 — Final branch-review fixes (`final-branch-review.md`, I1 / I3 / M1)

Three paper-side items from the whole-branch review. None changes a result; all
three were recomputed independently from the tracked artefacts before editing.

### I1 — the two over-pricing ranges now cover the same population

`tbc_preprint_main.tex` §3.4 quoted the consolidated points as over-priced by
"$1.2$ to $2.9\%$ and $0.5$ to $2.5\%$ respectively". The operator range spanned
all six consolidated points; the routing range spanned only the four item-1
points, so the clause joined two different populations with "respectively". The
routing range is now **$0.8$ to $2.9\%$** and the clause says the two ranges span
the same six points.

Recomputed from `results/revision_2026_08_v6/validation/tab_vroom_v2.csv` on the
clean basis (`n_unassigned == 0`, `jobs_removed == 0`, status in `{OK, CACHED}`),
gap = `(Σ pred − Σ actual) / Σ actual`:

| item | P | routing gap % | operator gap % |
|---|---:|---:|---:|
| 2 | 0 | 0.821 | 0.491 |
| 1 | 0 | 1.246 | 1.081 |
| 2 | 0.25 | 1.539 | 1.233 |
| 1 | 0.25 | 1.695 | 1.648 |
| 1 | 0.5 | 2.404 | 2.203 |
| 1 | 0.75 | 2.862 | 2.539 |

Baseline (item 0, n = 1,683): routing +4.385 %, operator +4.047 % — the printed
$4.4$ / $4.0\%$. The argument is unchanged: every consolidated gap is still
smaller than the baseline's, so every predicted saving remains an upper bound.
The G-round table above (row **I2/M3**) records the earlier, narrower scoping and
is left as the historical record of that round.

### I3 — the label-cost shares now name their population and carry a `% src:`

"the three terms contribute $72.2$, $6.0$, and $21.7\%$ of label cost" was the
only headline-adjacent number in the manuscript without a `% src:` line. It is
**exactly reproducible**, but on the **surrogate's training pool**, not on the
validation labels:

| population | rows | fixed | distance | time |
|---|---:|---:|---:|---:|
| `results/supplementary/sweep_v3_mergefix/training_matrix.csv` | 2,733 (all `OK`) | 72.207 % | 5.989 % | 21.748 % |
| `results/revision_2026_08_v6/validation/tab_vroom_v2.csv`, clean | 8,609 | 70.703 % | 6.057 % | 23.183 % |

Shares are `189.15·Σ n_routes` / `0.3864·Σ km` / `36·Σ duration_h` over the summed
solver cost. Both the main text (§2.2) and Supplementary S2 now name the training
pool for $72.2 / 6.0 / 21.7$ and give $70.7 / 6.1 / 23.2$ for the validation
labels, and a `% src:` block above the paragraph records both computations. The
compendium's §40.12 line `(72 / 6 / 22 %)` is refreshed to the same precision;
the paper's population is the right one for a statement about *label* cost, so
the paper's figure stands and the compendium follows it.

### M1 — the $P \cdot \theta = 0$ corner is stated in full

`balancing.py` and `61_grid_run_v2.py` both record that the service penalty is
identically zero over the whole corner $P \cdot \theta = 0$, "not merely at
$\theta = 0$". §2.4 stated only the $\theta$ half. It now states both, and
distinguishes them: at $\theta = 0$ stage 1 is daily and stage 2 is pinned, so
the baseline is recovered exactly; at $P = 0$ with $\theta > 0$ nothing is
pinned and the polish's wait and frequency changes are simply unpriced — which
is why the service differences reported at $(0, 1)$ are a by-product, not an
optimized trade.

---

## G9 - Task 14D: per-LSP weekly fleet figure (author decision, 2026-08-29)

The author ruled that the 78_ per-LSP weekly-fleet figures are to be used as
rendered, in absolute vehicle counts.

- **New Supplementary Fig. S12**, two panels from the tracked stems
  `supp_fig_fleet_week_P0.pdf` and `supp_fig_fleet_week_P025.pdf` (synced by
  71_ in commit `b56c86c`): Monday-Saturday vehicles per LSP at full adoption
  for the daily baseline, the routing-optimal plan (stage 1, recomputed with the
  partition-aware fleet counter) and the operator plan (stage 2). Appended at the
  END of the supplementary so it numbers as S12 and **every existing S1-S11
  pointer stays valid** (verified in the built `.aux`).
- Summed depot peaks in the caption: **1,239 -> 1,666 -> 1,030** at P = 0 and
  **1,239 -> 1,314 -> 1,026** at P = 0.25. Verified by summing
  `peak_baseline` / `peak_plan1` / `peak_plan2` over the seven providers in
  `results/revision_2026_08_v6/tables/tab_fleet_week_by_provider_v2.csv`,
  cross-checked against that file's own "All seven LSPs" row and against
  compendium 40.22b (1 239 -> 1 666, +34 % at P = 0).
- **Disclosure sentence in the caption**: the vehicle counts are model outputs
  computed from the HAGRID-based demand and the tour rule, not carrier data.
- One pointer clause in the main text's fleet passage, with its own `% src:`
  block naming the table and the columns summed.
- Supplementary grows 7 -> **8 pages**; the main text stays at 17 with 23
  bibitems.

### The guard gained a second check, because the first one missed a swallow

Adding that `% src:` block re-created the I1 defect: the second comment line
absorbed "The Monday--Saturday coefficient of variation ...". The guard did not
catch it, because 1a only looks for LaTeX **control sequences** and that sentence
contains none. A second rule now flags any `% src:` line whose body contains a
body-text marker -- math (`$`), an em-dash (`---`) or a tie (`~`) -- which no
provenance line ever needs. It caught the defect immediately, and then caught a
**second** instance of it in the regenerated page-budget patch, where block 10's
replacement text had been absorbed the same way. Both are fixed; a bare
sentence-boundary rule was tried first and rejected because it false-positives on
legitimate multi-line provenance prose.

### The patch

`page_budget_cuts.patch` no longer applied once the pointer clause landed
(`git apply --check` rc = 1). Regenerated with `git am -3` onto the 14D base: one
content conflict, in block 10's rewrite of exactly that fleet sentence, resolved
by keeping the block-10 text **and** carrying the Fig. S12 pointer into it, so
the pointer survives the cut. `git apply --check` rc = **0**; patched builds are
**16 + 10 pages**, 23 bibitems, guard clean after both patches and after the
first alone.
