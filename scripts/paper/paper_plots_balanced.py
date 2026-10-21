"""Plotting suite for the balanced overnight outputs.

Generates the same set of figures as paper_plots_overnight.py + the additional
balancing-impact plots requested by the user:

  fig01b_heatmap_balanced.{png,pdf}        — same as fig01 but on balanced cost
  fig02b_schedule_mix_balanced.{png,pdf}   — using balanced schedules
  fig03b_pareto_cost_wait_balanced.{png,pdf}
  fig04b_schedule_mix_vs_share_balanced.{png,pdf}
  fig05b_provider_cost_balanced.{png,pdf}
  fig06b_schedule_mix_vs_share_per_P_balanced.{png,pdf}

  fig_FB1_max_fleet_reduction_per_cell.{png,pdf}    — gap heatmap
  fig_FB2_hub_day_profile_compare.{png,pdf}         — fleet load per hub × day
  fig_FB3_cost_vs_imbalance_pareto.{png,pdf}        — cost cost / imbalance trade
  fig_FB4_max_fleet_per_provider.{png,pdf}          — peak fleet per LSP before/after

Inputs:
  results/overnight_2026_05_27_balanced/tab_balancing_summary.csv
  results/overnight_2026_05_27_balanced/tab_chosen_schedules.csv
  results/overnight_2026_05_27_balanced/tab_fleet_per_hub.csv

Status B (Task 19): fig01b/fig06b (schedule-mix/cost-heatmap) and FB3
(cost-vs-imbalance) map onto 74_-legacy's tab_balancing_summary.csv /
tab_chosen_schedules.csv directly. FB1 (system fleet-reduction heatmap) and
FB4 (peak-fleet-per-provider bar) need max_fleet_before, which the legacy
adapter leaves NaN (74_'s NO_SOURCE) -- reconstructed from v6-native
tab_costs_v2.csv's sum_hub_peak_before/after via --rev-dir (see
_paper_v6_common). FB2 (per-HUB-day fleet profile, before vs after) has NO
v6 source at any grain -- v6 only ever computes a per-hub-day fleet at the
FINAL plan -- so FB2 is E, documented, not produced.
"""
from __future__ import annotations
import argparse
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _paper_v6_common as V6  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BAL = ROOT / "results" / "overnight_2026_05_27_balanced"
OUT = BAL
REV_DIR = None
SCRIPT_NAME = "paper_plots_balanced.py"
PLAN_NOTE = "operator-polished (balanced) vs stage-1 (before)"
SRC_NOTE = "B: 74_-legacy tab_balancing_summary.csv/tab_chosen_schedules.csv"

rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})

FREQ_COLOR = {1: "#9d2226", 2: "#1d3557", 3: "#2a9d8f",
              4: "#e9c46a", 5: "#f4a261", 6: "#e76f51"}
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
OPERATING_P = 0.5
OPERATING_SHARE = 1.0


def load_data():
    summary = pd.read_csv(BAL / "tab_balancing_summary.csv")
    chosen = pd.read_csv(BAL / "tab_chosen_schedules.csv")
    fleet_path = BAL.parent / "rev" / "tab_fleet_per_hub_fixed.csv"
    if fleet_path.exists():
        fleet = pd.read_csv(fleet_path).rename(
            columns={"fleet_old": "fleet_before", "fleet_fixed": "fleet_after"})
    else:
        fleet = pd.read_csv(BAL / "tab_fleet_per_hub.csv")
    return summary, chosen, fleet


def plot_balanced_heatmap(summary):
    # aggregate per (P, share)
    agg = summary.groupby(["penalty", "share_willing"], as_index=False).agg(
        balanced_cost=("balanced_cost_eur", "sum"),
        init_cost=("init_cost_eur", "sum"),
    )
    pen = sorted(agg.penalty.unique())
    shares = sorted(agg.share_willing.unique())
    M = np.zeros((len(shares), len(pen)))
    for i, s in enumerate(shares):
        for j, p in enumerate(pen):
            row = agg[(np.isclose(agg.penalty, p)) & (np.isclose(agg.share_willing, s))]
            if len(row):
                M[i, j] = row.balanced_cost.iloc[0] / 1e3
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    im = ax.imshow(M, aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(pen)))
    ax.set_xticklabels([f"{p:g}" for p in pen])
    ax.set_yticks(range(len(shares)))
    ax.set_yticklabels([f"{int(s*100)}%" for s in shares])
    ax.set_xlabel("Service penalty $P$ [€/parcel/day]")
    ax.set_ylabel("Share willing to wait")
    ax.set_title("Total weekly cost [k€] — AFTER hub fleet-balancing")
    for i in range(len(shares)):
        for j in range(len(pen)):
            v = M[i, j]
            color = "white" if (v - M.min()) / max(1, M.max() - M.min()) < 0.55 else "black"
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", color=color, fontsize=10)
    plt.colorbar(im, ax=ax, label="Weekly cost [k€]")
    fig.tight_layout()
    V6.add_provenance_footer(fig, plan="operator-polished (balanced)",
                             script=SCRIPT_NAME, source=SRC_NOTE)
    V6.savefig_pair(fig, OUT / "fig01b_heatmap_balanced.png",
                    OUT / "fig01b_heatmap_balanced.pdf")
    plt.close(fig)
    print("  fig01b_heatmap_balanced")


def plot_mix_vs_share_per_P(chosen):
    pen_values = sorted(chosen.penalty.unique())
    fig, axes = plt.subplots(1, len(pen_values), figsize=(4.0 * len(pen_values), 4.5),
                              sharey=True, sharex=True)
    if len(pen_values) == 1:
        axes = [axes]
    for ax, P in zip(axes, pen_values):
        sub = chosen[np.isclose(chosen.penalty, P)]
        agg = sub.groupby(["share_willing", "schedule_size_balanced"]).size().reset_index(name="cnt")
        pivot = (agg.pivot(index="share_willing", columns="schedule_size_balanced",
                          values="cnt").fillna(0).sort_index())
        pivot = pivot.div(pivot.sum(axis=1), axis=0) * 100
        x = pivot.index.values * 100
        bottom = np.zeros(len(pivot))
        for sz in (2, 3, 4, 5, 6):
            if sz not in pivot.columns:
                continue
            h = pivot[sz].values
            ax.fill_between(x, bottom, bottom + h,
                             color=FREQ_COLOR[sz], alpha=0.92,
                             label=f"{sz} day/wk" if P == pen_values[-1] else None)
            bottom += h
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Share willing [%]")
        ax.set_title(f"$P = {P:g}$")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Delivery-frequency mix [%]")
    axes[-1].legend(title="Delivery days/week", loc="upper left",
                    bbox_to_anchor=(1.02, 1.0), frameon=True)
    fig.suptitle("Balanced delivery-frequency mix per service-penalty",
                  fontsize=13, y=1.02)
    fig.tight_layout()
    V6.add_provenance_footer(fig, plan="operator-polished (balanced)",
                             script=SCRIPT_NAME, source=SRC_NOTE)
    V6.savefig_pair(fig, OUT / "fig06b_schedule_mix_vs_share_per_P_balanced.png",
                    OUT / "fig06b_schedule_mix_vs_share_per_P_balanced.pdf")
    plt.close(fig)
    print("  fig06b_schedule_mix_vs_share_per_P_balanced")


def plot_max_fleet_reduction(summary):
    """Gap heatmap: (P × share) → max_fleet_before - max_fleet_after.

    max_fleet_before is NaN in the v6 legacy adapter (74_'s NO_SOURCE); the
    (penalty, share_willing) SYSTEM aggregate (summed over providers, this
    panel's own grain) is reconstructed from v6-native tab_costs_v2.csv's
    sum_hub_peak_before/after via REV_DIR when set. Returns None (E, caller
    skips the figure) when neither source is available.
    """
    try:
        V6.assert_has_data(summary, "max_fleet_before", context="FB1 heatmap")
        agg = summary.groupby(["penalty", "share_willing"], as_index=False).agg(
            max_fleet_before=("max_fleet_before", "sum"),
            max_fleet_after=("max_fleet_after", "sum"))
        source = SRC_NOTE
    except V6.NoV6Source as exc:
        if REV_DIR is None:
            print(f"  [E] FB1 max-fleet-reduction heatmap skipped: {exc}")
            return
        agg = V6.load_fleet_before_after(REV_DIR).groupby(
            ["penalty", "share_willing"], as_index=False).agg(
            max_fleet_before=("sum_hub_peak_before", "sum"),
            max_fleet_after=("sum_hub_peak_after", "sum"))
        source = "tab_costs_v2.csv (sum_hub_peak_before/after)"
    agg["reduction"] = agg.max_fleet_before - agg.max_fleet_after
    pen = sorted(agg.penalty.unique())
    shares = sorted(agg.share_willing.unique())
    M = np.zeros((len(shares), len(pen)))
    for i, s in enumerate(shares):
        for j, p in enumerate(pen):
            row = agg[(np.isclose(agg.penalty, p)) & (np.isclose(agg.share_willing, s))]
            if len(row):
                M[i, j] = row.reduction.iloc[0]
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    im = ax.imshow(M, aspect="auto", cmap="Greens")
    ax.set_xticks(range(len(pen)))
    ax.set_xticklabels([f"{p:g}" for p in pen])
    ax.set_yticks(range(len(shares)))
    ax.set_yticklabels([f"{int(s*100)}%" for s in shares])
    ax.set_xlabel("Service penalty $P$ [€/parcel/day]")
    ax.set_ylabel("Share willing")
    ax.set_title("Max-fleet reduction through hub balancing\n"
                  "(sum of (peak fleet before − after) across hubs)")
    for i in range(len(shares)):
        for j in range(len(pen)):
            v = M[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    color="white" if v > M.max() * 0.55 else "black", fontsize=9)
    plt.colorbar(im, ax=ax, label="Vehicle-day reduction")
    fig.tight_layout()
    V6.add_provenance_footer(fig, plan=PLAN_NOTE, script=SCRIPT_NAME, source=source)
    V6.savefig_pair(fig, OUT / "fig_FB1_max_fleet_reduction_per_cell.png",
                    OUT / "fig_FB1_max_fleet_reduction_per_cell.pdf")
    plt.close(fig)
    print("  fig_FB1_max_fleet_reduction_per_cell")


def plot_hub_day_profile(fleet):
    """For the operating point, compare BEFORE/AFTER hub × day fleet.

    E on v6: a per-hub-day (or even per-provider-day) "before" fleet has NO
    v6 source at all -- v6 only ever computes a per-hub-day fleet at the
    FINAL plan (74_'s NO_SOURCE on fleet_old/fleet_before). Unlike FB1/FB4
    there is no coarser v6-native aggregate that recovers this DAY-level
    split either (tab_costs_v2.csv's sum_hub_peak_before is a single number
    per (P, theta, provider), not six per-day values), so this stays E.
    """
    try:
        V6.assert_has_data(fleet, "fleet_before", context="FB2 hub-day profile")
    except V6.NoV6Source as exc:
        print(f"  [E] fig_FB2_hub_day_profile_compare skipped: {exc}")
        return
    sub = fleet[(np.isclose(fleet.penalty, OPERATING_P)) &
                (np.isclose(fleet.share_willing, OPERATING_SHARE))].copy()
    if sub.empty:
        print("  fig_FB2 skipped: no data at operating point")
        return
    # Pivot per provider × hub × day, then sum hubs to provider × day
    prov_day_before = sub.groupby(["provider", "day"], as_index=False).fleet_before.sum()
    prov_day_after = sub.groupby(["provider", "day"], as_index=False).fleet_after.sum()
    providers = sorted(sub.provider.unique())
    fig, axes = plt.subplots(1, len(providers), figsize=(2.4 * len(providers), 4.5),
                              sharey=True)
    if len(providers) == 1:
        axes = [axes]
    for ax, prov in zip(axes, providers):
        b = prov_day_before[prov_day_before.provider == prov].set_index("day")["fleet_before"]
        a = prov_day_after[prov_day_after.provider == prov].set_index("day")["fleet_after"]
        days = np.arange(6)
        bvals = [b.get(d, 0) for d in days]
        avals = [a.get(d, 0) for d in days]
        width = 0.4
        ax.bar(days - width/2, bvals, width, color="#e76f51", label="Before")
        ax.bar(days + width/2, avals, width, color="#1f4f8f", label="After")
        ax.set_xticks(days)
        ax.set_xticklabels(WEEKDAYS, rotation=45, ha="right")
        ax.set_title(prov, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Daily fleet (vehicles)")
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Hub fleet profile per LSP × weekday — before vs after balancing\n"
                  f"(operating point $P={OPERATING_P}$, share=100%)", fontsize=13, y=1.02)
    fig.tight_layout()
    V6.add_provenance_footer(fig, plan=PLAN_NOTE, script=SCRIPT_NAME, source=SRC_NOTE)
    V6.savefig_pair(fig, OUT / "fig_FB2_hub_day_profile_compare.png",
                    OUT / "fig_FB2_hub_day_profile_compare.pdf")
    plt.close(fig)
    print("  fig_FB2_hub_day_profile_compare")


def plot_cost_vs_imbalance(summary):
    """Per cell: cost increase % vs imbalance reduction %."""
    summary = summary.copy()
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for prov, g in summary.groupby("provider"):
        ax.scatter(g.imbalance_reduction_pct, g.cost_delta_pct,
                    s=24, alpha=0.6, label=prov)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Imbalance reduction [%]")
    ax.set_ylabel("Cost increase from balancing [%]")
    ax.set_title("Cost / fleet-smoothness trade-off per (P, share, provider)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    V6.add_provenance_footer(fig, plan=PLAN_NOTE, script=SCRIPT_NAME, source=SRC_NOTE)
    V6.savefig_pair(fig, OUT / "fig_FB3_cost_vs_imbalance_pareto.png",
                    OUT / "fig_FB3_cost_vs_imbalance_pareto.pdf")
    plt.close(fig)
    print("  fig_FB3_cost_vs_imbalance_pareto")


def plot_max_fleet_per_provider(summary):
    """Bar chart: peak fleet per LSP, before vs after, at operating point.

    Same max_fleet_before NaN issue as FB1; reconstructed from v6-native
    tab_costs_v2.csv via REV_DIR at this same (P, share) operating point
    when the legacy column has no data.
    """
    try:
        V6.assert_has_data(summary, "max_fleet_before", context="FB4 per-provider bar")
        sub = summary[(np.isclose(summary.penalty, OPERATING_P)) &
                      (np.isclose(summary.share_willing, OPERATING_SHARE))].copy()
        fb4_source = SRC_NOTE
    except V6.NoV6Source as exc:
        if REV_DIR is None:
            print(f"  [E] fig_FB4_max_fleet_per_provider skipped: {exc}")
            return
        fba = V6.load_fleet_before_after(REV_DIR)
        sub = fba[(np.isclose(fba.penalty, OPERATING_P))
                 & (np.isclose(fba.share_willing, OPERATING_SHARE))].rename(
            columns={"sum_hub_peak_before": "max_fleet_before",
                    "sum_hub_peak_after": "max_fleet_after"}).copy()
        fb4_source = "tab_costs_v2.csv (sum_hub_peak_before/after)"
    if sub.empty:
        print("  fig_FB4 skipped: no data at operating point")
        return
    sub = sub.sort_values("max_fleet_before", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(sub))
    width = 0.38
    ax.bar(x - width/2, sub.max_fleet_before, width, color="#e76f51",
            label="Before balancing", edgecolor="black")
    ax.bar(x + width/2, sub.max_fleet_after, width, color="#1f4f8f",
            label="After balancing", edgecolor="black")
    for i, (b, a) in enumerate(zip(sub.max_fleet_before, sub.max_fleet_after)):
        pct = -100 * (b - a) / max(1, b)
        ax.text(i, max(b, a) + 5, f"{pct:+.1f}%", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(sub.provider, rotation=0)
    ax.set_ylabel("Peak fleet (vehicles)")
    ax.set_title(f"Peak weekly fleet per LSP — before vs after balancing "
                  f"(operating point $P={OPERATING_P}$, share=100%)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    V6.add_provenance_footer(fig, plan=PLAN_NOTE, script=SCRIPT_NAME, source=fb4_source)
    V6.savefig_pair(fig, OUT / "fig_FB4_max_fleet_per_provider.png",
                    OUT / "fig_FB4_max_fleet_per_provider.pdf")
    plt.close(fig)
    print("  fig_FB4_max_fleet_per_provider")


def main():
    global BAL, OUT, REV_DIR
    ap = argparse.ArgumentParser(description=__doc__)
    V6.add_v6_cli_args(ap, needs_legacy=True)
    args = ap.parse_args()
    if args.legacy_dir is not None:
        BAL = Path(args.legacy_dir)
    elif args.rev_dir is not None:
        BAL, _ = V6.run_legacy_adapter(args.rev_dir,
                                       Path(args.out_dir or OUT) / "_legacy")
    if args.rev_dir is not None:
        REV_DIR = Path(args.rev_dir)
    if args.out_dir is not None:
        OUT = Path(args.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading balanced data ...")
    summary, chosen, fleet = load_data()
    print(f"  summary rows: {len(summary)}")
    print(f"  chosen rows: {len(chosen)}")
    print(f"  fleet rows: {len(fleet)}")

    print("\nBalanced versions of paper plots:")
    plot_balanced_heatmap(summary)
    plot_mix_vs_share_per_P(chosen)

    print("\nFleet-balancing impact plots:")
    plot_max_fleet_reduction(summary)
    plot_hub_day_profile(fleet)
    plot_cost_vs_imbalance(summary)
    plot_max_fleet_per_provider(summary)

    print(f"\nDone. Outputs in {OUT}")


if __name__ == "__main__":
    main()
