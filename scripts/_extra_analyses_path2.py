"""Additional interpretive analyses for the Path-2 paper results.

  fig_O2_pipeline_contribution.png   Stepwise saving contribution of each
                                      pipeline stage (CD-init / balancing /
                                      system smoothing) vs daily baseline.
  fig_O3_weekday_pattern_share.png   Which Mo-Sa day combinations the chosen
                                      schedules favour at the sweet-spot.
  fig_O4_wait_distribution.png       Per-PLZ wait-day distribution at the
                                      sweet-spot, by provider.
  fig_O5_freq_histogram_per_provider.png  Per-provider frequency histogram at
                                            the sweet-spot.
  tab_summary_by_operating_point.csv  Headline table for the paper covering
                                       baseline / cost-optimal / sweet-spot /
                                       service-priority operating points.
"""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
from matplotlib import rcParams

ROOT = Path(__file__).resolve().parents[1]
PATH2 = ROOT / "results" / "overnight_2026_05_29_path2"
OUT = ROOT / "results" / "paper_results_final" / "05_optimization"
TBL_OUT = ROOT / "results" / "paper_results_final" / "00_overview"
TBL_OUT.mkdir(parents=True, exist_ok=True)
BASE = 1909747.75
SWEET_P = 0.5
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

PROV_COLORS = {
    "Amazon":  "#1d3557", "DHL": "#e63946", "DPD": "#2a9d8f",
    "FedEx":   "#e76f51", "GLS": "#f4a261", "Hermes": "#264653",
    "UPS":     "#9d4edd",
}
FREQ_COLORS = {2: "#1d3557", 3: "#2a9d8f", 4: "#e9c46a",
               5: "#f4a261", 6: "#e76f51"}

rcParams.update({
    "font.family": "serif", "font.size": 9,
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 9, "axes.titlesize": 9.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})

W_2COL = 7.0


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.png + pdf")


def load_data():
    s = pd.read_csv(PATH2 / "tab_balancing_summary.csv")
    c = pd.read_csv(PATH2 / "tab_chosen_schedules.csv"); c["plz"] = c.plz.astype(str)
    sm = (pd.read_csv(PATH2 / "_tab_chosen_with_system_smoothing.csv")
          if (PATH2 / "_tab_chosen_with_system_smoothing.csv").exists() else None)
    if sm is not None: sm["plz"] = sm.plz.astype(str)
    sp = (pd.read_csv(PATH2 / "_system_spread_per_cell.csv")
          if (PATH2 / "_system_spread_per_cell.csv").exists() else None)
    return s, c, sm, sp


# ────────────────────────────────────────────────────────────────────────
def fig_O2_pipeline_contribution(s, c, sm, sp):
    """For each penalty (theta = 1), show how much saving comes from each
    pipeline step."""
    # init_cost is bundled init (= 'init_cost_eur')
    # balanced_cost is per-hub-balanced
    # smoothed_cost requires bundled-cost recomputation -> approximate with
    # cost_pre_smoothing - smoothed_delta (already in our _tab_*_smoothing file)
    sm_summ = (pd.read_csv(PATH2 / "_tab_balancing_summary_with_smoothing.csv")
               if (PATH2 / "_tab_balancing_summary_with_smoothing.csv").exists()
               else None)
    th = 1.0
    ag = s[np.isclose(s.share_willing, th)].groupby("penalty").agg(
        init=("init_cost_eur", "sum"), bal=("balanced_cost_eur", "sum")).reset_index()
    ag["init_sav"] = 100 * (BASE - ag.init) / BASE
    ag["bal_sav"]  = 100 * (BASE - ag.bal) / BASE
    if sm_summ is not None:
        smag = sm_summ[np.isclose(sm_summ.share_willing, th)].groupby("penalty").agg(
            sm_cost=("system_smoothed_cost_eur", "sum")).reset_index()
        ag = ag.merge(smag, on="penalty", how="left")
        ag["sm_cost"] = ag.sm_cost.fillna(ag.bal)
        ag["sm_sav"] = 100 * (BASE - ag.sm_cost) / BASE
    else:
        ag["sm_sav"] = ag.bal_sav

    ag = ag.sort_values("penalty")
    # Contributions: init = bundled selection saving (vs baseline)
    #                balancing_delta = bal_sav - init_sav  (small / negative)
    #                smoothing_delta = sm_sav - bal_sav  (~0)
    ag["init_contrib"] = ag.init_sav
    ag["bal_contrib"]  = ag.bal_sav - ag.init_sav
    ag["sm_contrib"]   = ag.sm_sav - ag.bal_sav

    fig, ax = plt.subplots(figsize=(W_2COL, 3.6))
    x = np.arange(len(ag))
    width = 0.7
    ax.bar(x, ag.init_contrib, width, color="#1d3557",
           edgecolor="black", linewidth=0.4, label="Optimization (CD init)")
    bottoms = ag.init_contrib.values
    ax.bar(x, ag.bal_contrib.clip(lower=-100), width,
           bottom=bottoms, color="#f4a261",
           edgecolor="black", linewidth=0.4, label="Per-hub balancing")
    bottoms = bottoms + ag.bal_contrib.values
    ax.bar(x, ag.sm_contrib.clip(lower=-100), width,
           bottom=bottoms, color="#2a9d8f",
           edgecolor="black", linewidth=0.4, label="System smoothing")
    for i, r in ag.iterrows():
        total = r.sm_sav
        ax.text(i, total + 0.5, f"{total:.1f}%", ha="center",
                fontsize=7, fontweight="bold", color="#222")
    ax.set_xticks(x)
    ax.set_xticklabels([rf"$P={p:g}$" for p in ag.penalty])
    ax.set_ylabel("Cumulative cost saving vs daily baseline [%]")
    ax.set_xlabel(r"Service penalty $P$ [€/p/d] ($\theta = 1$)")
    ax.set_title("Per-stage saving contribution along the Path-2 pipeline",
                 fontsize=10)
    ax.axhline(0, color="black", linewidth=0.4)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig_O2_pipeline_contribution")


# ────────────────────────────────────────────────────────────────────────
def fig_O3_weekday_pattern_share(c):
    """At sweet-spot, which Mo-Sa day-combinations dominate the chosen
    schedules? Look at most common patterns per provider."""
    cell = c[(np.isclose(c.penalty, SWEET_P)) &
             (np.isclose(c.share_willing, 1.0))]

    # Decode weekdays string -> tuple of day indices
    def pat_to_tuple(s):
        days = s.split(",")
        return tuple(sorted(DAYS.index(d) for d in days if d in DAYS))

    cell = cell.copy()
    cell["pattern"] = cell.weekdays_init.apply(pat_to_tuple)
    # Group: per (provider, pattern size, pattern) → count
    fig, axes = plt.subplots(2, 4, figsize=(W_2COL, 4.5), sharex=False)
    axes = axes.flatten()
    providers = sorted(cell.provider.unique())
    # Use top-5 weekday-patterns globally
    top = Counter(cell.pattern.tolist()).most_common(8)
    top_patterns = [p for p, _ in top]
    for ax_i, prov in enumerate(providers):
        ax = axes[ax_i]
        sub = cell[cell.provider == prov]
        n = len(sub)
        counts = Counter(sub.pattern.tolist())
        # Plot top patterns of THIS provider
        prov_top = counts.most_common(5)
        labels = []
        vals = []
        for pat, cnt in prov_top:
            days_str = "-".join(DAYS[d] for d in pat)
            labels.append(f"{days_str}\n({len(pat)}d/wk)")
            vals.append(100 * cnt / n)
        y = np.arange(len(labels))
        ax.barh(y, vals, color=PROV_COLORS[prov], edgecolor="black",
                linewidth=0.4, alpha=0.88)
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=6)
        ax.invert_yaxis()
        ax.set_title(f"{prov} (n={n})", fontsize=8)
        ax.set_xlabel("Share of PLZs [%]", fontsize=7)
        ax.tick_params(axis="x", labelsize=6)
        ax.grid(axis="x", alpha=0.25)
    axes[-1].axis("off")
    fig.suptitle(rf"Top-5 schedule patterns per provider at sweet-spot "
                  rf"$P = {SWEET_P:g}$, $\theta = 1$",
                  fontsize=10, y=1.02)
    save(fig, "fig_O3_weekday_pattern_share")


# ────────────────────────────────────────────────────────────────────────
def fig_O4_wait_distribution(c):
    """Per-PLZ wait-day distribution at sweet-spot, coloured by provider."""
    cell = c[(np.isclose(c.penalty, SWEET_P)) &
             (np.isclose(c.share_willing, 1.0))].copy()
    fig, ax = plt.subplots(figsize=(W_2COL, 3.3))
    bins = np.linspace(0, max(cell.avg_wait_d_init.max() * 1.05, 1.0), 36)
    bottoms = np.zeros(len(bins) - 1)
    for prov in sorted(cell.provider.unique()):
        vals = cell[cell.provider == prov].avg_wait_d_init.values
        hist, _ = np.histogram(vals, bins=bins)
        ax.bar(bins[:-1], hist, width=np.diff(bins),
               bottom=bottoms, color=PROV_COLORS[prov], edgecolor="black",
               linewidth=0.2, align="edge", label=prov, alpha=0.9)
        bottoms += hist
    weighted = (cell.avg_wait_d_init * cell.weekly_parcels).sum() / cell.weekly_parcels.sum()
    ax.axvline(weighted, color="black", linestyle="--", linewidth=1.0,
                label=f"parcels-weighted mean = {weighted:.3f}d")
    ax.set_xlabel(r"Average customer wait per PLZ [days] ($\theta = 1$)")
    ax.set_ylabel("Number of postal-code areas")
    ax.set_title(rf"Wait-time distribution at sweet-spot $P = {SWEET_P:g}$",
                  fontsize=10)
    ax.legend(loc="upper right", ncol=2, fontsize=7.5)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig_O4_wait_distribution")


# ────────────────────────────────────────────────────────────────────────
def fig_O5_freq_histogram_per_provider(c):
    """Distribution of chosen delivery frequencies per provider at sweet-spot."""
    cell = c[(np.isclose(c.penalty, SWEET_P)) &
             (np.isclose(c.share_willing, 1.0))]
    providers = sorted(cell.provider.unique())
    fig, ax = plt.subplots(figsize=(W_2COL, 3.3))
    x = np.arange(2, 7)
    w = 0.11
    for i, prov in enumerate(providers):
        vals = cell[cell.provider == prov].schedule_size_init.values
        counts = np.array([(vals == k).sum() for k in x])
        ax.bar(x + (i - 3) * w, 100 * counts / counts.sum(), w,
               color=PROV_COLORS[prov], edgecolor="black",
               linewidth=0.4, label=prov)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k} d/wk" for k in x])
    ax.set_ylabel("Share of provider's PLZs [%]")
    ax.set_title(rf"Per-provider frequency histogram at $P = {SWEET_P:g}$, "
                  rf"$\theta = 1$", fontsize=10)
    ax.legend(loc="upper left", ncol=2, fontsize=7.5)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig_O5_freq_histogram_per_provider")


# ────────────────────────────────────────────────────────────────────────
def tab_summary_by_operating_point(s, c):
    """Headline summary table covering 4 reference operating points."""
    rows = []
    for label, P, th in [
        ("Daily baseline",         10.0, 0.0),   # = all daily
        ("Cost-optimal end",        0.0, 1.0),
        ("Sweet-spot (System)",     0.5, 1.0),
        ("Service-priority",        1.0, 1.0),
    ]:
        gp = s[(np.isclose(s.penalty, P)) & (np.isclose(s.share_willing, th))]
        cp = c[(np.isclose(c.penalty, P)) & (np.isclose(c.share_willing, th))]
        init_cost = gp.init_cost_eur.sum()
        bal_cost  = gp.balanced_cost_eur.sum()
        wait = (cp.avg_wait_d_init * cp.weekly_parcels).sum() / cp.weekly_parcels.sum()
        mean_freq = cp.schedule_size_init.mean()
        n_batched = int((cp.schedule_size_init < 6).sum())
        peak_b = gp.max_fleet_before.sum(); peak_a = gp.max_fleet_after.sum()
        rows.append({
            "operating_point": label,
            "P": P, "theta": th,
            "init_cost_eur": init_cost, "init_saving_pct": 100*(BASE-init_cost)/BASE,
            "balanced_cost_eur": bal_cost, "balanced_saving_pct": 100*(BASE-bal_cost)/BASE,
            "avg_wait_d": wait,
            "mean_freq": mean_freq,
            "n_batched": n_batched, "n_daily": 312 - n_batched,
            "peak_fleet_init": peak_b, "peak_fleet_balanced": peak_a,
            "peak_reduction_pct": 100*(peak_b-peak_a)/max(1, peak_b),
        })
    df = pd.DataFrame(rows)
    df.to_csv(TBL_OUT / "tab_summary_by_operating_point.csv", index=False)
    print(f"  saved tab_summary_by_operating_point.csv ({len(df)} rows)")
    print(df.round(2).to_string(index=False))


# ────────────────────────────────────────────────────────────────────────
def main():
    s, c, sm, sp = load_data()
    print("=== Generating extra analyses ===")
    fig_O2_pipeline_contribution(s, c, sm, sp)
    fig_O3_weekday_pattern_share(c)
    fig_O4_wait_distribution(c)
    fig_O5_freq_histogram_per_provider(c)
    print("\n=== Summary table ===")
    tab_summary_by_operating_point(s, c)


if __name__ == "__main__":
    main()
