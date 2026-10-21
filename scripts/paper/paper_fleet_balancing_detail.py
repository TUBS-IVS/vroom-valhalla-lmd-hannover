"""Detailed fleet-balancing impact analysis.

After running overnight_orchestrator_balanced.py we have, per cell,
both the unbalanced (cost-optimal) and the hub-balanced schedules
plus the per-(provider, hub, day) fleet load before/after.

This script answers the user's question: *what does hub balancing buy
us, and where?*

  fig_FB5_imbalance_distribution.{png,pdf}    — boxplot of imbalance Δ per cell
  fig_FB6_peak_day_heatmap.{png,pdf}          — peak-day fleet vs penalty/share
  fig_FB7_avg_peak_per_hub.{png,pdf}          — bar chart per (provider, hub)
                                                  showing peak fleet shrink
  fig_FB8_swap_distribution.{png,pdf}          — histogram of # swaps per cell
  fig_FB9_imbalance_reduction_vs_n_plz.{png,pdf}
  tab_fleet_balancing_summary.csv
  tab_per_hub_summary.csv

Inputs (from results/overnight_2026_05_27_balanced/):
  tab_balancing_summary.csv
  tab_chosen_schedules.csv
  tab_fleet_per_hub.csv

Status B (Task 19): 74_-legacy's tab_balancing_summary.csv carries
imbalance_before/after and swaps_made directly (max_fleet_before is
NaN -- NO_SOURCE), so FB5/FB8/FB9 and the per-provider summary port
cleanly. FB6's (P, share) peak-fleet heatmap and the per-provider summary's
peak columns are reconstructed from v6-native tab_costs_v2.csv's
sum_hub_peak_before/after via --rev-dir (see _paper_v6_common). FB7 (top-20
PER-HUB peak-fleet reduction) and tab_per_hub_summary.csv have NO v6 source
at all -- the v6 fleet table is only ever written at the FINAL plan, so a
per-hub-day "before" does not exist anywhere, not even via tab_costs_v2.csv
(which is only per-provider) -- both are E, documented, not produced.
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

rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
OPERATING_P = 0.5
OPERATING_SHARE = 1.0


def _fig_fb6_heatmap(agg: pd.DataFrame, out: Path, plan_note: str,
                     script_name: str, source_note: str) -> None:
    """The (P, share) before/after peak-fleet heatmap, factored out of
    main() so both the legacy-column and the v6-native-aggregate source can
    feed the same plotting code without duplicating it."""
    pen = sorted(agg.penalty.unique())
    shares = sorted(agg.share_willing.unique())
    M_before = np.zeros((len(shares), len(pen)))
    M_after = np.zeros_like(M_before)
    for i, s in enumerate(shares):
        for j, p in enumerate(pen):
            r = agg[(np.isclose(agg.penalty, p)) & (np.isclose(agg.share_willing, s))]
            if len(r):
                M_before[i, j] = r.max_fleet_before.iloc[0]
                M_after[i, j] = r.max_fleet_after.iloc[0]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    vmin = min(M_before.min(), M_after.min())
    vmax = max(M_before.max(), M_after.max())
    for ax, M, title in [(axA, M_before, "BEFORE balancing"),
                          (axB, M_after, "AFTER balancing")]:
        im = ax.imshow(M, aspect="auto", cmap="viridis_r", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(pen)))
        ax.set_xticklabels([f"{p:g}" for p in pen])
        ax.set_yticks(range(len(shares)))
        ax.set_yticklabels([f"{int(s*100)}%" for s in shares])
        ax.set_xlabel("Service penalty $P$")
        ax.set_title(f"Peak weekly fleet — {title}")
        for i in range(len(shares)):
            for j in range(len(pen)):
                v = M[i, j]
                color = "white" if (v - vmin) / max(1, vmax - vmin) > 0.55 else "black"
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        color=color, fontsize=8)
    axA.set_ylabel("Share willing")
    plt.colorbar(im, ax=[axA, axB], label="Peak fleet (vehicles)",
                  shrink=0.8, pad=0.02)
    fig.suptitle("Peak fleet across the (P, share) grid — balancing flattens hub demand\n"
                 "(system total, summed over providers)",
                  fontsize=13, y=1.02)
    V6.add_provenance_footer(fig, plan=plan_note, script=script_name, source=source_note)
    V6.savefig_pair(fig, out / "fig_FB6_peak_day_heatmap.png",
                    out / "fig_FB6_peak_day_heatmap.pdf")
    plt.close(fig)


def main():
    global BAL, OUT, REV_DIR
    ap = argparse.ArgumentParser(description=__doc__)
    V6.add_v6_cli_args(ap, needs_legacy=True)
    args = ap.parse_args()
    if args.legacy_dir is not None:
        legacy_run = Path(args.legacy_dir)
        legacy_rev = legacy_run.parent / "rev"
    elif args.rev_dir is not None:
        legacy_run, legacy_rev = V6.run_legacy_adapter(
            args.rev_dir, Path(args.out_dir or OUT) / "_legacy")
    else:
        legacy_run = legacy_rev = None
    if legacy_run is not None:
        BAL = legacy_run
    if args.rev_dir is not None:
        REV_DIR = Path(args.rev_dir)
    if args.out_dir is not None:
        OUT = Path(args.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    script_name = "paper_fleet_balancing_detail.py"
    plan_note = "operator-polished (balanced) vs stage-1 (before)"
    src_note = "B: 74_-legacy tab_balancing_summary.csv"

    summary = pd.read_csv(BAL / "tab_balancing_summary.csv")
    fleet_path = (legacy_rev / "tab_fleet_per_hub_fixed.csv"
                 if legacy_rev is not None else BAL / "tab_fleet_per_hub.csv")
    fleet = pd.read_csv(fleet_path)
    if legacy_rev is not None:
        # legacy schema: fleet_old (NaN, NO_SOURCE) / fleet_fixed (final plan)
        fleet = fleet.rename(columns={"fleet_old": "fleet_before",
                                      "fleet_fixed": "fleet_after"})
    print(f"  summary rows: {len(summary)}")
    print(f"  fleet rows:   {len(fleet)}")

    # ── Plot FB5: imbalance reduction per cell (boxplot per provider)
    fig, ax = plt.subplots(figsize=(9, 5))
    provs = sorted(summary.provider.unique())
    data = [summary[summary.provider == p].imbalance_reduction_pct.values for p in provs]
    bp = ax.boxplot(data, labels=provs, patch_artist=True)
    for patch, color in zip(bp["boxes"], plt.cm.viridis(np.linspace(0.15, 0.9, len(provs)))):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Imbalance reduction [%]")
    n_per_provider = len(data[0]) if data else 0
    ax.set_title("Fleet-imbalance reduction per LSP "
                 f"(across {n_per_provider} (penalty, share) points per LSP)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    V6.add_provenance_footer(fig, plan=plan_note, script=script_name, source=src_note)
    V6.savefig_pair(fig, OUT / "fig_FB5_imbalance_distribution.png",
                    OUT / "fig_FB5_imbalance_distribution.pdf")
    plt.close(fig)
    print("  fig_FB5_imbalance_distribution")

    # ── Plot FB6: peak fleet after balancing per cell heatmap
    # max_fleet_before is NaN in the v6 legacy adapter (74_'s NO_SOURCE): the
    # v6 fleet table is only ever written at the FINAL plan. The (penalty,
    # share_willing) SYSTEM aggregate (summed over providers, this panel's
    # own grain) does have a v6-native source: tab_costs_v2.csv's
    # sum_hub_peak_before/after, via --rev-dir. Fail loud rather than
    # silently sum an all-NaN column to a fabricated 0.
    try:
        V6.assert_has_data(summary, "max_fleet_before", context="FB6 heatmap")
        agg = summary.groupby(["penalty", "share_willing"], as_index=False).agg(
            max_fleet_before=("max_fleet_before", "sum"),
            max_fleet_after=("max_fleet_after", "sum"))
        fb6_note = src_note
    except V6.NoV6Source as exc:
        if REV_DIR is None:
            print(f"  [E] FB6 peak-fleet heatmap skipped: {exc}")
            agg = None
        else:
            agg = V6.load_fleet_before_after(REV_DIR).groupby(
                ["penalty", "share_willing"], as_index=False).agg(
                max_fleet_before=("sum_hub_peak_before", "sum"),
                max_fleet_after=("sum_hub_peak_after", "sum"))
            fb6_note = "tab_costs_v2.csv (sum_hub_peak_before/after)"

    if agg is not None:
        _fig_fb6_heatmap(agg, OUT, plan_note, script_name, fb6_note)
    print("  fig_FB6_peak_day_heatmap" + ("" if agg is not None else " -- E, skipped"))

    # ── Plot FB7: average peak-fleet per (provider, hub) bar
    # PER-HUB-DAY "before" (fleet_before/fleet_old) has NO v6 source at all
    # -- unlike FB6's system aggregate, there is no tab_costs_v2.csv
    # equivalent at hub grain (v6 only ever computes a per-hub-day fleet at
    # the FINAL plan). E, not approximated -- do not substitute the final
    # plan's own value as a fake "before".
    try:
        V6.assert_has_data(fleet, "fleet_before", context="FB7 per-hub bar")
        sub_fleet = fleet[(np.isclose(fleet.penalty, OPERATING_P)) &
                           (np.isclose(fleet.share_willing, OPERATING_SHARE))]
        hub_peak = sub_fleet.groupby(["provider", "hub"], as_index=False).agg(
            peak_before=("fleet_before", "max"),
            peak_after=("fleet_after", "max"))
        hub_peak["delta"] = hub_peak.peak_before - hub_peak.peak_after
        hub_peak = hub_peak.sort_values("delta", ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(11, 6))
        x = np.arange(len(hub_peak))
        width = 0.38
        ax.bar(x - width/2, hub_peak.peak_before, width, color="#e76f51",
                label="Peak before", edgecolor="black")
        ax.bar(x + width/2, hub_peak.peak_after, width, color="#1f4f8f",
                label="Peak after", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r.provider}\n{r.hub[:18]}"
                              for _, r in hub_peak.iterrows()],
                              rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Peak weekly fleet (vehicles)")
        ax.set_title(f"Top-20 hubs by fleet reduction (operating point $P={OPERATING_P}$)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        V6.add_provenance_footer(fig, plan=plan_note, script=script_name,
                                 source="tab_fleet_per_hub_fixed.csv")
        V6.savefig_pair(fig, OUT / "fig_FB7_avg_peak_per_hub.png",
                        OUT / "fig_FB7_avg_peak_per_hub.pdf")
        plt.close(fig)
        print("  fig_FB7_avg_peak_per_hub")
    except V6.NoV6Source as exc:
        print(f"  [E] FB7 per-hub fleet-reduction bar skipped: {exc}")

    # ── Plot FB8: swap distribution
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(summary.swaps_made, bins=30, color="#1f4f8f",
             edgecolor="white", alpha=0.85)
    ax.set_xlabel("Number of fleet-balancing swaps per cell")
    ax.set_ylabel("Count of (P, share, provider) cells")
    ax.set_title(f"Distribution of swaps made — median {summary.swaps_made.median():.0f}, "
                  f"max {summary.swaps_made.max()}")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    V6.add_provenance_footer(fig, plan=plan_note, script=script_name, source=src_note)
    V6.savefig_pair(fig, OUT / "fig_FB8_swap_distribution.png",
                    OUT / "fig_FB8_swap_distribution.pdf")
    plt.close(fig)
    print("  fig_FB8_swap_distribution")

    # ── Plot FB9: imbalance reduction vs n_plz
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for prov, g in summary.groupby("provider"):
        ax.scatter(g.n_plz, g.imbalance_reduction_pct, s=22, alpha=0.6, label=prov)
    ax.set_xlabel("Number of PLZ at the LSP")
    ax.set_ylabel("Imbalance reduction [%]")
    ax.set_title("Fleet-balancing impact scales with LSP coverage")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    V6.add_provenance_footer(fig, plan=plan_note, script=script_name, source=src_note)
    V6.savefig_pair(fig, OUT / "fig_FB9_imbalance_reduction_vs_n_plz.png",
                    OUT / "fig_FB9_imbalance_reduction_vs_n_plz.pdf")
    plt.close(fig)
    print("  fig_FB9_imbalance_reduction_vs_n_plz")

    # ── Per-provider summary table: imbalance/cost columns always come from
    # the legacy adapter; the peak-fleet columns reuse whichever source FB6
    # resolved to above (v6-native aggregate when the legacy one is NaN).
    per_prov = summary.groupby("provider").agg(
        n_cells=("provider", "count"),
        mean_imbalance_reduction_pct=("imbalance_reduction_pct", "mean"),
        mean_cost_delta_pct=("cost_delta_pct", "mean"),
        sum_swaps=("swaps_made", "sum"),
    ).reset_index()
    if agg is not None:
        peak_src = (summary if fb6_note == src_note else
                   V6.load_fleet_before_after(REV_DIR).rename(
                       columns={"sum_hub_peak_before": "max_fleet_before",
                               "sum_hub_peak_after": "max_fleet_after"}))
        peak_by_prov = peak_src.groupby("provider").agg(
            sum_peak_before=("max_fleet_before", "sum"),
            sum_peak_after=("max_fleet_after", "sum")).reset_index()
        per_prov = per_prov.merge(peak_by_prov, on="provider", how="left")
        per_prov["peak_reduction_pct"] = (
            100 * (per_prov.sum_peak_before - per_prov.sum_peak_after)
            / per_prov.sum_peak_before.clip(lower=1))
    per_prov.to_csv(OUT / "tab_per_provider_fleet_summary.csv", index=False)
    print("\nPer-provider summary:")
    print(per_prov.round(2).to_string(index=False))

    # Per-hub summary (operating point): E, same reason as FB7 -- no v6
    # source for a per-hub-day "before" fleet exists at all.
    try:
        V6.assert_has_data(fleet, "fleet_before", context="tab_per_hub_summary.csv")
        if not sub_fleet.empty:
            hub_summary = sub_fleet.groupby(["provider", "hub"]).agg(
                peak_before=("fleet_before", "max"),
                peak_after=("fleet_after", "max"),
            ).reset_index()
            hub_summary["peak_reduction"] = hub_summary.peak_before - hub_summary.peak_after
            hub_summary["peak_reduction_pct"] = (
                100 * hub_summary.peak_reduction / hub_summary.peak_before.clip(lower=1)
            )
            hub_summary.to_csv(OUT / "tab_per_hub_summary.csv", index=False)
            print(f"  per-hub summary: {len(hub_summary)} (provider, hub) rows")
    except V6.NoV6Source as exc:
        print(f"  [E] tab_per_hub_summary.csv skipped: {exc}")

    print(f"\nDone. Outputs in {OUT}")


if __name__ == "__main__":
    main()
