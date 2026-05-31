"""Assemble the paper-final folder: generate fresh plots from the balanced
run + import the best existing analyses.

Output: results/paper_final_2026_05_28/

Sections:
  01_input_data/         Demand summary, hubs, raumtyp
  02_baseline/           Daily-delivery baseline KPIs
  03_training/           Permuted training data + sample generation
  04_model/              CV battery, model comparison, Daganzo physics + LGB hybrid
  05_optimization/       Pareto frontier, P×share grid, schedule mix, sweet-spot
  06_balancing/          Fleet-balancing impact, before/after
  07_validation/         VROOM validation, per-provider/per-schedule
  08_interpretation/     Hybrid waterfall, decision trees
  README.md              All in one
"""
from __future__ import annotations
import shutil
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.patches import Patch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
BAL = ROOT / "results" / "overnight_2026_05_27_balanced"
UNBAL = ROOT / "results" / "overnight_2026_05_27"
OUT = ROOT / "results" / "paper_final_2026_05_28"

rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 13,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})

PROV_COLOR = {"Amazon": "#003049", "DHL": "#d62828", "DPD": "#f77f00",
              "FedEx": "#5a189a", "GLS": "#2a9d8f", "Hermes": "#9d4edd",
              "UPS": "#7d5a50"}


def _mkdir(*parts):
    p = OUT.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _copy(src, dst):
    if Path(src).exists():
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def load_data():
    """Load everything we need from the balanced run."""
    s = pd.read_csv(BAL / "tab_balancing_summary.csv")
    sched = pd.read_csv(BAL / "tab_chosen_schedules.csv")
    fleet = pd.read_csv(BAL / "tab_fleet_per_hub.csv") if (BAL / "tab_fleet_per_hub.csv").exists() else None

    # Per-(P, share) aggregate
    cost_agg = s.groupby(["penalty", "share_willing"], as_index=False).agg(
        init_cost_eur=("init_cost_eur", "sum"),
        bal_cost_eur=("balanced_cost_eur", "sum"),
        max_fleet_before=("max_fleet_before", "sum"),
        max_fleet_after=("max_fleet_after", "sum"),
        total_swaps=("swaps_made", "sum"),
        total_routes_before=("total_routes_before", "sum"),
        total_routes_after=("total_routes_after", "sum"),
    )
    # Parcel-weighted wait per cell (init + balanced)
    wait_rows = []
    for (P, sh), g in sched.groupby(["penalty", "share_willing"]):
        pkts = g.weekly_parcels.sum()
        w_init = (g.avg_wait_d_init * g.weekly_parcels).sum() / pkts
        w_bal = (g.avg_wait_d_balanced * g.weekly_parcels).sum() / pkts
        wait_rows.append({"penalty": P, "share_willing": sh,
                          "wait_init": w_init, "wait_balanced": w_bal})
    waits = pd.DataFrame(wait_rows)
    cost_agg = cost_agg.merge(waits, on=["penalty", "share_willing"])

    baseline_cost = float(cost_agg[cost_agg.share_willing == 0.0].bal_cost_eur.max())
    cost_agg["saving_pct_init"] = 100 * (baseline_cost - cost_agg.init_cost_eur) / baseline_cost
    cost_agg["saving_pct_bal"] = 100 * (baseline_cost - cost_agg.bal_cost_eur) / baseline_cost
    cost_agg["fleet_red_pct"] = 100 * (cost_agg.max_fleet_before - cost_agg.max_fleet_after) / cost_agg.max_fleet_before.clip(lower=1)
    cost_agg["delta_pct"] = 100 * (cost_agg.bal_cost_eur - cost_agg.init_cost_eur) / cost_agg.init_cost_eur.clip(lower=1)

    return {"summary": s, "schedules": sched, "fleet": fleet,
            "agg": cost_agg, "baseline": baseline_cost}


def make_pareto_frontier(data, outdir):
    """Cool visualisation of the Pareto frontier with confidence band."""
    agg = data["agg"]
    baseline = data["baseline"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left panel: balanced Pareto with confidence band (init = lighter)
    ax = axes[0]
    PENALTY_COLORS = plt.cm.viridis(np.linspace(0.15, 0.9, agg.penalty.nunique()))
    P_VALUES = sorted(agg.penalty.unique())

    # Background: area covered by init vs balanced
    all_x_init = []
    all_y_init = []
    all_x_bal = []
    all_y_bal = []
    for P in P_VALUES:
        sub = agg[agg.penalty == P].sort_values("share_willing")
        all_x_init.extend(sub.wait_init.values)
        all_y_init.extend(sub.saving_pct_init.values)
        all_x_bal.extend(sub.wait_balanced.values)
        all_y_bal.extend(sub.saving_pct_bal.values)
    # Convex hull of init points = upper bound
    # Convex hull of balanced = lower bound
    # Use simple sort+max-saving-per-wait
    init_sorted = sorted(zip(all_x_init, all_y_init))
    bal_sorted = sorted(zip(all_x_bal, all_y_bal))

    # Pareto envelope: for each wait, max saving achievable
    init_pts = pd.DataFrame(init_sorted, columns=["w", "s"])
    bal_pts = pd.DataFrame(bal_sorted, columns=["w", "s"])

    # Pareto upper envelope (best saving for each wait threshold)
    def pareto_envelope(df, increasing_wait=True):
        # Find points where saving is non-dominated
        df = df.sort_values("w" if increasing_wait else "s").reset_index(drop=True)
        env = []
        max_s = -np.inf
        for _, row in df.iterrows():
            if row.s > max_s:
                env.append((row.w, row.s))
                max_s = row.s
        return env
    env_init = pareto_envelope(init_pts)
    env_bal = pareto_envelope(bal_pts)

    # Fill area between init (top) and balanced (top)
    ex = [p[0] for p in env_init]
    ey = [p[1] for p in env_init]
    bx = [p[0] for p in env_bal]
    by = [p[1] for p in env_bal]

    # Plot per-P trajectories
    for pi, P in enumerate(P_VALUES):
        sub = agg[agg.penalty == P].sort_values("share_willing")
        ax.plot(sub.wait_balanced, sub.saving_pct_bal,
                 "o-", color=PENALTY_COLORS[pi], linewidth=2, markersize=6,
                 label=f"P = {P} €/p/d", alpha=0.92)
        ax.plot(sub.wait_init, sub.saving_pct_init,
                 "x--", color=PENALTY_COLORS[pi], linewidth=1, markersize=5,
                 alpha=0.35)

    # Envelope band
    ax.plot(ex, ey, "-", color="black", linewidth=1.2, alpha=0.6,
             label="Cost-optimal envelope (init)")
    ax.plot(bx, by, "-", color="darkred", linewidth=1.8,
             label="Fleet-balanced envelope")

    # Highlight sweet-spot
    op = agg[(np.isclose(agg.penalty, 0.5)) & (np.isclose(agg.share_willing, 1.0))]
    if len(op):
        ax.scatter(op.wait_balanced.iloc[0], op.saving_pct_bal.iloc[0],
                    marker="*", s=400, c="gold", edgecolor="black",
                    zorder=10, label=f"Sweet-spot (P=0.5, share=1.0)")

    ax.set_xlabel("Average customer wait [days]")
    ax.set_ylabel("Cost saving vs daily baseline [%]")
    ax.set_title("Pareto frontier — Cost-Service trade-off\n"
                  "Solid: fleet-balanced  /  Dashed: cost-optimal (init)")
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)

    # Right panel: marginal saving
    ax = axes[1]
    for pi, P in enumerate(P_VALUES):
        sub = agg[agg.penalty == P].sort_values("wait_balanced")
        if len(sub) < 3:
            continue
        x = sub.wait_balanced.values
        y = sub.saving_pct_bal.values
        # Discrete derivative
        dy_dx = np.gradient(y, x + 1e-9)
        ax.plot(x[1:-1], dy_dx[1:-1], "o-", color=PENALTY_COLORS[pi],
                 label=f"P={P}", alpha=0.8, markersize=4, linewidth=1.5)
    ax.set_xlabel("Customer wait [days]")
    ax.set_ylabel("d(saving)/d(wait)  [% per day]")
    ax.set_title("Marginal saving per wait-day\n"
                  "Sweet-spot: where slope intersects service-cost P")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / "fig_PF1_pareto_frontier.png")
    fig.savefig(outdir / "fig_PF1_pareto_frontier.pdf")
    plt.close(fig)
    print(f"  → fig_PF1_pareto_frontier")

    # ── Additional: P×share heatmap of saving %
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    saving_init = agg.pivot(index="penalty", columns="share_willing", values="saving_pct_init")
    saving_bal = agg.pivot(index="penalty", columns="share_willing", values="saving_pct_bal")
    vmax = max(saving_init.values.max(), saving_bal.values.max())
    vmin = min(saving_init.values.min(), saving_bal.values.min())

    for ax, mat, title in [(axes[0], saving_init, "Cost-optimal (init)"),
                            (axes[1], saving_bal, "Fleet-balanced")]:
        im = ax.imshow(mat.values, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels([f"{x*100:.0f}%" for x in mat.columns], rotation=0)
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels([f"P={p}" for p in mat.index])
        ax.set_xlabel("Share willing to wait")
        ax.set_ylabel("Service penalty")
        ax.set_title(f"Saving [%] — {title}")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                color = "white" if v < (vmax - vmin) * 0.5 + vmin else "black"
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        color=color, fontsize=8)
        plt.colorbar(im, ax=ax, label="Saving %", shrink=0.8, pad=0.02)

    fig.suptitle("Saving heatmap across the (P × share) grid", fontsize=14)
    fig.tight_layout()
    fig.savefig(outdir / "fig_PF2_saving_heatmap.png")
    fig.savefig(outdir / "fig_PF2_saving_heatmap.pdf")
    plt.close(fig)
    print(f"  → fig_PF2_saving_heatmap")

    # ── Cost-fleet tradeoff plot at share=1.0 across P
    fig, ax = plt.subplots(figsize=(10, 6))
    s1 = agg[np.isclose(agg.share_willing, 1.0)].sort_values("penalty")
    P_str = [f"P={p}" for p in s1.penalty]

    # Init: cost vs fleet (cost-optimal)
    ax.scatter(s1.max_fleet_before, s1.bal_cost_eur / 1e3,
                marker="x", s=120, color="gray", label="Cost-optimal init",
                zorder=5)
    # Balanced: cost vs fleet (after balancing)
    ax.scatter(s1.max_fleet_after, s1.bal_cost_eur / 1e3,
                marker="o", s=120, c=PENALTY_COLORS, label="After fleet-balance",
                zorder=10, edgecolor="black")

    # Annotate each point
    for _, r in s1.iterrows():
        ax.annotate(f"P={r.penalty}",
                    (r.max_fleet_after, r.bal_cost_eur / 1e3),
                    xytext=(8, 5), textcoords="offset points",
                    fontsize=8)

    # Arrows from before to after
    for _, r in s1.iterrows():
        ax.annotate("", xy=(r.max_fleet_after, r.bal_cost_eur / 1e3),
                    xytext=(r.max_fleet_before, r.bal_cost_eur / 1e3),
                    arrowprops=dict(arrowstyle="->", color="gray", alpha=0.4))

    ax.set_xlabel("Peak weekly fleet (max trucks per hub × day)")
    ax.set_ylabel("Weekly routing cost [k€]")
    ax.set_title("Cost-Fleet trade-off at share = 100%\n"
                  "Arrows = effect of fleet-balancing (gray X → coloured circle)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "fig_PF3_cost_fleet_tradeoff.png")
    fig.savefig(outdir / "fig_PF3_cost_fleet_tradeoff.pdf")
    plt.close(fig)
    print(f"  → fig_PF3_cost_fleet_tradeoff")


def make_schedule_mix(data, outdir):
    """Schedule-size mix as stacked area over share, per P."""
    sched = data["schedules"]
    P_VALUES = sorted(sched.penalty.unique())
    SCHED_SIZE_COLORS = {2: "#9b2226", 3: "#bb3e03", 4: "#ee9b00",
                          5: "#94d2bd", 6: "#005f73"}

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for pi, P in enumerate(P_VALUES):
        ax = axes[pi // 4, pi % 4]
        sub = sched[sched.penalty == P]
        mix_df = sub.groupby(["share_willing", "schedule_size_balanced"]).size().unstack(fill_value=0)
        shares = mix_df.index.values * 100
        bottom = np.zeros(len(shares))
        for sz in sorted(mix_df.columns):
            vals = mix_df[sz].values
            ax.fill_between(shares, bottom, bottom + vals,
                             color=SCHED_SIZE_COLORS.get(sz, "gray"),
                             alpha=0.88, label=f"{sz}d/wk" if pi == 0 else None)
            bottom += vals
        ax.set_xlabel("Share willing [%]")
        ax.set_ylabel("Cells" if pi % 4 == 0 else "")
        ax.set_title(f"P = {P} €/p/d", fontsize=11)
        ax.set_xlim(0, 100)
        ax.grid(axis="y", alpha=0.3)

    handles = [Patch(color=SCHED_SIZE_COLORS[s], label=f"{s}d/wk", alpha=0.88)
               for s in sorted(SCHED_SIZE_COLORS.keys())]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=11,
                bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle("Schedule-size distribution across (P × share) — fleet-balanced output",
                  fontsize=14, y=0.99)
    fig.tight_layout()
    fig.savefig(outdir / "fig_SM1_schedule_mix_grid.png", bbox_inches="tight")
    fig.savefig(outdir / "fig_SM1_schedule_mix_grid.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  → fig_SM1_schedule_mix_grid")


def make_balancing_impact(data, outdir):
    """Fleet-balancing impact: cost vs peak-fleet reduction trade-off."""
    agg = data["agg"]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for sh in sorted(agg.share_willing.unique()):
        sub = agg[np.isclose(agg.share_willing, sh)].sort_values("penalty")
        ax.plot(sub.delta_pct, sub.fleet_red_pct, "o-",
                 color=plt.cm.plasma(sh), label=f"share={sh:.0f}".replace("0.0", "0").replace(".", "0%"),
                 alpha=0.85, markersize=6)
    ax.set_xlabel("Cost increase [%]   (5% budget)")
    ax.set_ylabel("Peak-fleet reduction [%]")
    ax.set_title("Fleet-balancing trade-off — Cost vs Fleet reduction\n"
                  "Each line: one share-willing level across all penalties")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(5, color="red", linewidth=1, linestyle="--", label="5% budget")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2, title="share willing")
    fig.tight_layout()
    fig.savefig(outdir / "fig_FB1_cost_vs_fleet_tradeoff.png")
    fig.savefig(outdir / "fig_FB1_cost_vs_fleet_tradeoff.pdf")
    plt.close(fig)
    print(f"  → fig_FB1_cost_vs_fleet_tradeoff")

    # Bar chart: total swaps + fleet reduction per (P, share)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    pivot_swaps = agg.pivot(index="penalty", columns="share_willing", values="total_swaps")
    pivot_fleet_red = agg.pivot(index="penalty", columns="share_willing", values="fleet_red_pct")

    for ax, mat, title, cmap in [
        (axes[0], pivot_swaps, "Total swaps performed", "BuGn"),
        (axes[1], pivot_fleet_red, "Peak-fleet reduction [%]", "RdYlGn"),
    ]:
        im = ax.imshow(mat.values, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels([f"{x*100:.0f}%" for x in mat.columns])
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels([f"P={p}" for p in mat.index])
        ax.set_xlabel("Share willing")
        ax.set_ylabel("Service penalty")
        ax.set_title(title)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                if v == 0 or np.isnan(v):
                    ax.text(j, i, "—", ha="center", va="center", fontsize=8)
                else:
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                            color="white" if v > mat.values.max()*0.55 else "black",
                            fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

    fig.tight_layout()
    fig.savefig(outdir / "fig_FB2_swap_and_reduction.png")
    fig.savefig(outdir / "fig_FB2_swap_and_reduction.pdf")
    plt.close(fig)
    print(f"  → fig_FB2_swap_and_reduction")


def import_existing(outdir):
    """Copy best existing analyses into the new folder."""
    print("\n[importing existing best analyses]")
    items = [
        # CV battery (best paper-relevant model comparison)
        ("results/overnight_2026_05_27/diagnosis_v2/cv_battery/tab_model_comparison_cv.csv",
         "04_model/tab_model_comparison_cv.csv"),
        ("results/overnight_2026_05_27/diagnosis_v2/cv_battery/fig_cv_mape_bars.png",
         "04_model/fig_M1_cv_mape_bars.png"),
        ("results/overnight_2026_05_27/diagnosis_v2/cv_battery/fig_cv_mape_bars.pdf",
         "04_model/fig_M1_cv_mape_bars.pdf"),
        # Daganzo physics decomposition
        ("results/overnight_2026_05_27/diagnosis_v2/fig_D1_component_decomposition.png",
         "04_model/fig_M2_daganzo_components.png"),
        ("results/overnight_2026_05_27/diagnosis_v2/fig_D1_component_decomposition.pdf",
         "04_model/fig_M2_daganzo_components.pdf"),
        # Validation
        ("results/overnight_2026_05_27/fig15_validation_scatter_hybrid_vs_pure.png",
         "07_validation/fig_V1_hybrid_vs_pure.png"),
        ("results/overnight_2026_05_27/fig15_validation_scatter_hybrid_vs_pure.pdf",
         "07_validation/fig_V1_hybrid_vs_pure.pdf"),
        ("results/overnight_2026_05_27/fig16_validation_per_provider.png",
         "07_validation/fig_V2_per_provider.png"),
        ("results/overnight_2026_05_27/fig16_validation_per_provider.pdf",
         "07_validation/fig_V2_per_provider.pdf"),
        ("results/overnight_2026_05_27/fig17_validation_per_schedule_size.png",
         "07_validation/fig_V3_per_schedule_size.png"),
        ("results/overnight_2026_05_27/fig17_validation_per_schedule_size.pdf",
         "07_validation/fig_V3_per_schedule_size.pdf"),
        # Hybrid interpretation
        ("results/overnight_2026_05_27/diagnosis_v2/interpretation/fig_H1_waterfall_per_regime.png",
         "08_interpretation/fig_H1_waterfall_per_regime.png"),
        ("results/overnight_2026_05_27/diagnosis_v2/interpretation/fig_H1_waterfall_per_regime.pdf",
         "08_interpretation/fig_H1_waterfall_per_regime.pdf"),
        ("results/overnight_2026_05_27/diagnosis_v2/interpretation/fig_H3_lgb_feature_importance.png",
         "08_interpretation/fig_H3_lgb_feature_importance.png"),
        ("results/overnight_2026_05_27/diagnosis_v2/interpretation/fig_H3_lgb_feature_importance.pdf",
         "08_interpretation/fig_H3_lgb_feature_importance.pdf"),
        # Decision trees
        ("results/overnight_2026_05_27/diagnosis_v2/interpretation/fig_DT1_schedule_classification.png",
         "08_interpretation/fig_DT1_schedule_classification.png"),
        ("results/overnight_2026_05_27/diagnosis_v2/interpretation/fig_DT1_schedule_classification.pdf",
         "08_interpretation/fig_DT1_schedule_classification.pdf"),
        ("results/overnight_2026_05_27/diagnosis_v2/interpretation/fig_DT2_saving_regression.png",
         "08_interpretation/fig_DT2_saving_regression.png"),
        ("results/overnight_2026_05_27/diagnosis_v2/interpretation/fig_DT2_saving_regression.pdf",
         "08_interpretation/fig_DT2_saving_regression.pdf"),
        ("results/overnight_2026_05_27/diagnosis_v2/interpretation/fig_DT3_lgb_residual.png",
         "08_interpretation/fig_DT3_lgb_residual.png"),
        ("results/overnight_2026_05_27/diagnosis_v2/interpretation/fig_DT3_lgb_residual.pdf",
         "08_interpretation/fig_DT3_lgb_residual.pdf"),
        # Willingness curve
        ("results/overnight_2026_05_27/fig_willingness_curve.png",
         "05_optimization/fig_O0_willingness_curve.png"),
        ("results/overnight_2026_05_27/fig_willingness_curve.pdf",
         "05_optimization/fig_O0_willingness_curve.pdf"),
    ]
    for src, dst in items:
        if _copy(ROOT / src, OUT / dst):
            print(f"  ✓ {dst}")


def write_readme(data):
    """Generate the master README."""
    agg = data["agg"]
    baseline = data["baseline"]

    sweet_init = agg[(np.isclose(agg.penalty, 0.5)) & (np.isclose(agg.share_willing, 1.0))]
    sweet_bal = sweet_init.iloc[0] if len(sweet_init) else None

    lines = [
        f"# Paper Final — Region Hannover Last-Mile Batched Delivery",
        f"## MobilTUM 2026  ·  Generated 2026-05-28",
        "",
        f"Daily baseline weekly cost: **{baseline/1e3:.1f} k€**",
        f"Best operating point (P=0.5 €/parcel/day, share=100%):  "
        f"**{sweet_bal.saving_pct_bal:.1f}% saving** "
        f"at {sweet_bal.wait_balanced:.3f} d avg wait" if sweet_bal is not None else "",
        "",
        "## Folder structure",
        "",
        "| Section | Purpose |",
        "|---|---|",
        "| **01_input_data/** | HAGRID demand, hubs, PLZ raumtyp |",
        "| **02_baseline/** | Daily delivery baseline KPIs |",
        "| **03_training/** | Permuted training samples (sweep/runner) |",
        "| **04_model/** | Daganzo-LGB-Hybrid + CV battery comparison |",
        "| **05_optimization/** | Pareto frontier, P×share grid, sweet-spot |",
        "| **06_balancing/** | Fleet-balancing impact + cost-fleet tradeoff |",
        "| **07_validation/** | VROOM out-of-sample validation |",
        "| **08_interpretation/** | LGB residual waterfall + decision trees |",
        "",
        "## Headline figures",
        "",
        "* **fig_PF1_pareto_frontier**: Cost-vs-Wait Pareto with init/balanced + sweet-spot",
        "* **fig_PF2_saving_heatmap**: P×share matrix of saving %, init vs balanced",
        "* **fig_PF3_cost_fleet_tradeoff**: at share=1.0, fleet reduction per P",
        "* **fig_SM1_schedule_mix_grid**: stacked-area of schedule sizes per P",
        "* **fig_FB1_cost_vs_fleet_tradeoff**: cost-fleet point cloud per share",
        "* **fig_FB2_swap_and_reduction**: heatmaps of swaps + fleet reduction",
        "* **fig_M1_cv_mape_bars**: GroupKFold-CV model comparison (Hybrid wins 2.96%)",
        "* **fig_M2_daganzo_components**: route/km/cost decomposition vs VROOM",
        "* **fig_V1-V3**: per-provider/per-schedule VROOM validation",
        "* **fig_H1, H3**: LGB residual waterfall + feature importance",
        "* **fig_DT1-DT3**: decision trees on schedule, saving, LGB residual",
        "",
        "## Method",
        "",
        "1. **Training**: 2,733 perturbed (provider, PLZ, day, agg_k) samples → VROOM solve → 25 base features.",
        "2. **Model**: Daganzo-LGB-Hybrid, α=1.343 median-calibrated, MAPE 2.96% GroupKFold OOS.",
        "3. **Optimization**: per-(P, share) cell, ML cost matrix on 312 (provider, PLZ) cells × 39 schedules, "
        "argmin with service penalty `P × share × parcels × wait`.",
        "4. **Bundling**: PLZ with ≥230 parcels stand-alone; <230 LPT-packed into multi-PLZ tours.",
        "5. **Fleet balancing**: greedy swap with 5% cost budget on TOTAL cost.",
        "6. **Willingness model**: smooth power-law (B2B-priority k=2), aggregate matches share globally.",
        "",
        "## Key numbers",
        "",
        f"* Daily baseline: {baseline/1e3:.1f} k€/week",
        f"* Max saving (P=0, share=1.0): {agg[(agg.penalty==0)&(agg.share_willing==1.0)].saving_pct_bal.iloc[0]:.1f}%",
        f"* Sweet-spot (P=0.5, share=1.0): {sweet_bal.saving_pct_bal:.1f}% saving at {sweet_bal.wait_balanced:.3f}d wait" if sweet_bal is not None else "",
        f"* Fleet reduction at share=1.0, P=0: "
        f"{agg[(agg.penalty==0)&(agg.share_willing==1.0)].fleet_red_pct.iloc[0]:.1f}%  "
        f"(peak {agg[(agg.penalty==0)&(agg.share_willing==1.0)].max_fleet_before.iloc[0]:.0f} → "
        f"{agg[(agg.penalty==0)&(agg.share_willing==1.0)].max_fleet_after.iloc[0]:.0f})",
    ]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✓ README.md")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(f"Paper Final Assembly  →  {OUT}")
    print("=" * 70)

    data = load_data()
    print(f"  data: {len(data['agg'])} cells (P × share)")

    # ── 05 Optimization (Pareto + heatmaps)
    optim_dir = _mkdir("05_optimization")
    make_pareto_frontier(data, optim_dir)
    make_schedule_mix(data, optim_dir)

    # ── 06 Balancing
    bal_dir = _mkdir("06_balancing")
    make_balancing_impact(data, bal_dir)

    # ── Import existing best analyses
    for s in ["01_input_data", "02_baseline", "03_training", "04_model",
              "05_optimization", "06_balancing", "07_validation",
              "08_interpretation"]:
        _mkdir(s)
    import_existing(OUT)

    # ── Save the aggregated data CSV
    data["agg"].to_csv(OUT / "05_optimization" / "tab_optimization_full_grid.csv", index=False)
    data["schedules"].to_csv(OUT / "05_optimization" / "tab_chosen_schedules_all.csv", index=False)
    data["summary"].to_csv(OUT / "06_balancing" / "tab_balancing_summary.csv", index=False)

    write_readme(data)
    print(f"\nDone. Full paper-ready output in {OUT}")


if __name__ == "__main__":
    main()
