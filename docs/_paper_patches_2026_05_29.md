# Paper text patches after the bundled-accounting fix (2026-05-29)

## What changed in the data

- `init_cost_eur` in `tab_balancing_summary.csv` is now the **hub-bundled**
  routing cost (dd + bundled express) of the per-postal-code-area cost-optimal
  selection. It was previously the per-postal-code-area UNBUNDLED total, which
  systematically overstated cost at low θ (each cell's express assumed to
  travel its own dedicated vehicle).
- `balanced_cost_eur` is unchanged (already bundled).
- All per-cell `cost_delta_eur` / `cost_delta_pct` are recomputed.
- **Schedules do not change.** Mix figure, maps figure and all peak-fleet
  reduction numbers stay exactly as before.

## Methodology — cost-accounting note (insert in §2 or footnote)

> All reported routing costs use the **hub-bundled** express model: parcels of
> different postal-code areas served by the same depot and not delivered on a
> given weekday are pooled into a shared express tour for that depot-day. The
> per-postal-code-area selection step uses a separable unbundled proxy to keep
> the assignment problem tractable; the bundled cost is then evaluated on the
> chosen schedules for reporting, ensuring consistency between init,
> fleet-balanced and VROOM-validated cost numbers.

## Methodology — fleet balancing (replace earlier text)

```latex
Fleet balancing is a post-processing step that flattens per-depot daily vehicle
peaks. Starting from the cost-optimal selection, the balancer performs greedy
single-cell schedule swaps under the hub-bundled cost model. A swap is accepted
only if it lowers the worst hub's peak vehicle range and keeps the penalised
objective (routing cost plus service penalty) within five percent of its
cost-optimal value. The procedure terminates once no depot's vehicle spread
exceeds one vehicle or the swap budget is exhausted.
```

(replaces the earlier "one percent" wording; corresponds to the actual
`FLEET_COST_BUDGET_PCT = 5.0` in the orchestrator)

## Results — saving heatmap paragraph (replaces the prior § around §3.2)

```latex
\autoref{fig:saving_fleet_heatmaps} maps the cost saving over the
service-penalty by willingness-to-wait grid for (a) the cost-optimal init plan
and (b) its fleet-balanced refinement, alongside (c) the peak-fleet reduction
achieved by balancing. With both plans evaluated on a single consistent
hub-bundled cost basis, panels (a) and (b) lie close to one another across the
grid: the cost-optimal selection already captures the bulk of the consolidation
benefit, and the balancer's remaining wiggle room corresponds either to small
hub-level bundling synergies that the per-postal-code-area selection cannot
see by construction (yielding a slightly lower cost at marginally higher wait)
or to extra cost spent to flatten the fleet (yielding a slightly higher cost at
unchanged service). The peak-fleet reduction in panel (c) reaches up to fifty
percent at $P = 0$ and twenty percent at the sweet-spot $P = 0.4$~€/p/d, with
the largest reductions naturally occurring where the selection has the most
schedule diversity to redistribute across days.
```

## Validation paragraph (closing, around §3.X)

```latex
The optimised schedules are then re-solved with VROOM as an independent
validation of the surrogate's predictions at the deployed operating point.
Across the 624 routing rows the ML cost agrees with VROOM within \mape{4.21}{\%}
(R$^2 = 0.997$); at full willingness ($\theta = 1$) the agreement tightens to
\mape{2.99}{\%} ($R^2 = 0.998$), while at half willingness ($\theta = 0.5$) the
ML predictions remain conservative ($+5.4 \%$ bias, $R^2 = 0.995$). This
confirms that the optimised plans transfer cleanly from the surrogate-driven
optimisation back to operationally realistic routing, with the largest residual
concentrated in the express-heavy intermediate-willingness regime where
hub-bundled express makes up a larger share of total cost.
```

## Numbers that remain unchanged

| Quantity | Value | Source |
|---|---|---|
| Headline saving (sweet-spot $P=0.4$, $\theta=1$) | **15.27 %** | `tab_pareto_optimal.csv` (1D sweep, $\theta=1$, dd-only equals bundled) |
| Average wait at sweet-spot | **0.249 d** | same |
| Areas batched / daily at sweet-spot | 249 / 63 | same |
| Surrogate MAPE (GroupKFold) | **2.95 %** | model_battery |
| Surrogate $R^2$ | **0.997** | same |
| ML-vs-VROOM ALL MAPE | **4.21 %** | `tab_diagnostics_balanced.csv` |
| Mix-paragraph numbers (96.5 → 65.4 %; mean 2.04 → 2.51; 6.00 → 5.40) | as in submitted draft | `tab_chosen_schedules.csv`, both columns intact |
