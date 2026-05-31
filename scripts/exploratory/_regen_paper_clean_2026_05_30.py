"""Final clean regeneration of all Path-2 paper figures for Transportation
Research Procedia (Elsevier) single-column layout, with consistent naming
across the 9 paper-folder categories.

Categorisation logic:
  05_optimization/    Pareto, saving, sweet-spot, schedule-mix figures
  06_balancing/       Fleet-impact, smoothing, per-provider Mo-Sa figures
  11_spatial_maps/    Choropleth maps

All figures use the same matplotlib rcParams (serif / dejavuserif mathtext),
sized to fit Elsevier TRPro layouts (single-column ~3.5 in, full-width ~7 in).
Cost-vs-service references default to the system sweet-spot P=0.5; per-provider
analyses use each LSP's individual knee in [0.25, 0.75].
"""
from __future__ import annotations
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch

# ────────────────────────────────────────────────────────────────────────
#   Paths
# ────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
PATH2 = ROOT / "results" / "overnight_2026_05_29_path2"
PAPER = ROOT / "results" / "paper_final_2026_05_30"

OPT_DIR = PAPER / "05_optimization"
BAL_DIR = PAPER / "06_balancing"
MAPS_DIR = PAPER / "11_spatial_maps"
for d in (OPT_DIR, BAL_DIR, MAPS_DIR):
    d.mkdir(parents=True, exist_ok=True)

BASE = 1909747.75
SWEET_P = 0.5
KNEE_LO, KNEE_HI = 0.25, 0.75
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
PROVIDERS = ["Amazon", "DHL", "DPD", "FedEx", "GLS", "Hermes", "UPS"]

PROV_COLORS = {
    "Amazon":  "#1d3557",
    "DHL":     "#e63946",
    "DPD":     "#2a9d8f",
    "FedEx":   "#e76f51",
    "GLS":     "#f4a261",
    "Hermes":  "#264653",
    "UPS":     "#9d4edd",
}
FREQ_COLORS = {2: "#1d3557", 3: "#2a9d8f", 4: "#e9c46a",
               5: "#f4a261", 6: "#e76f51"}

# ────────────────────────────────────────────────────────────────────────
#   TRPro rcParams
# ────────────────────────────────────────────────────────────────────────
rcParams.update({
    "font.family": "serif", "font.size": 9,
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})

# Figure widths (inches) — TRPro guidelines
W_1COL = 3.5      # single column
W_15COL = 5.1     # 1.5 column
W_2COL = 7.0      # full width (2 columns)


def savefig(fig: plt.Figure, name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_dir.name}/{name}.png + pdf")


# ────────────────────────────────────────────────────────────────────────
#   Data loaders
# ────────────────────────────────────────────────────────────────────────
def load_data():
    s = pd.read_csv(PATH2 / "tab_balancing_summary.csv")
    c = pd.read_csv(PATH2 / "tab_chosen_schedules.csv"); c["plz"] = c.plz.astype(str)
    f = pd.read_csv(PATH2 / "tab_fleet_per_hub.csv")
    base = (s[np.isclose(s.share_willing, 0.0)]
            .groupby("provider").balanced_cost_eur.mean())
    return s, c, f, base


def pareto_at_theta1(s, c, base):
    th = 1.0
    s1 = s[np.isclose(s.share_willing, th)]
    c1 = c[np.isclose(c.share_willing, th)]
    rows = []
    for P in sorted(s1.penalty.unique()):
        gp = s1[np.isclose(s1.penalty, P)]
        cp = c1[np.isclose(c1.penalty, P)]
        cost = gp.init_cost_eur.sum()
        wait = (cp.avg_wait_d_init * cp.weekly_parcels).sum() / cp.weekly_parcels.sum()
        rows.append({"penalty": P, "wait_d": wait,
                      "sav_pct": 100 * (BASE - cost) / BASE,
                      "mean_freq": cp.schedule_size_init.mean()})
    return pd.DataFrame(rows).sort_values("penalty").reset_index(drop=True)


def per_provider_pareto(s, c, base):
    th = 1.0
    rows = []
    for prov in PROVIDERS:
        b = float(base[prov])
        for P in sorted(s.penalty.unique()):
            gp = s[(s.provider == prov) & np.isclose(s.penalty, P)
                   & np.isclose(s.share_willing, th)]
            cp = c[(c.provider == prov) & np.isclose(c.penalty, P)
                   & np.isclose(c.share_willing, th)]
            if not len(gp) or not len(cp): continue
            cost = gp.init_cost_eur.iloc[0]
            wait = float((cp.avg_wait_d_init * cp.weekly_parcels).sum()
                          / cp.weekly_parcels.sum())
            rows.append({"provider": prov, "penalty": P, "wait_d": wait,
                          "sav_pct": 100 * (b - cost) / b,
                          "baseline_k": b / 1000})
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────────────
#   05_optimization/  figures
# ────────────────────────────────────────────────────────────────────────
def fig_PF1_pareto_path2(s, c, base):
    par = pareto_at_theta1(s, c, base)
    fig, ax = plt.subplots(figsize=(W_1COL, 3.0))
    ax.plot(par.wait_d, par.sav_pct, "o-", color="#1d3557",
            linewidth=1.6, markersize=6)
    sweet = par[np.isclose(par.penalty, SWEET_P)].iloc[0]
    ax.scatter([sweet.wait_d], [sweet.sav_pct], marker="*", s=300,
                color="gold", edgecolor="black", linewidth=0.9, zorder=10,
                label=rf"Sweet-spot $P = {SWEET_P:g}$")
    for _, r in par.iterrows():
        ax.annotate(rf"$P={r.penalty:g}$",
                     (r.wait_d, r.sav_pct), xytext=(4, 4),
                     textcoords="offset points", fontsize=6.5, color="#444")
    ax.set_xlabel(r"Average customer wait [d] ($\theta = 1$)")
    ax.set_ylabel("Cost saving vs daily baseline [%]")
    ax.legend(loc="lower right", frameon=True, framealpha=0.92)
    ax.grid(alpha=0.3)
    savefig(fig, "fig_PF1_pareto", OPT_DIR)


def fig_PF2_saving_fleet_heatmaps(s):
    agg = s.groupby(["penalty", "share_willing"], as_index=False).agg(
        init=("init_cost_eur", "sum"),
        bal=("balanced_cost_eur", "sum"),
        peak_b=("max_fleet_before", "sum"),
        peak_a=("max_fleet_after", "sum"))
    agg = agg[~np.isclose(agg.penalty, 0.4)]
    agg["init_sav"] = 100 * (BASE - agg.init) / BASE
    agg["bal_sav"]  = 100 * (BASE - agg.bal)  / BASE
    agg["peak_red"] = 100 * (agg.peak_b - agg.peak_a) / agg.peak_b.clip(lower=1)

    pivI = agg.pivot(index="penalty", columns="share_willing", values="init_sav")
    pivB = agg.pivot(index="penalty", columns="share_willing", values="bal_sav")
    pivP = agg.pivot(index="penalty", columns="share_willing", values="peak_red")

    sV = min(pivI.values.min(), pivB.values.min()), max(pivI.values.max(), pivB.values.max())

    fig, axes = plt.subplots(1, 3, figsize=(W_2COL, 2.6))

    def heat(ax, mat, cmap, vmin, vmax, title):
        im = ax.imshow(mat.values, aspect="auto", cmap=cmap,
                        vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels([f"{x*100:.0f}" for x in mat.columns], fontsize=6)
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels([f"{p:g}" for p in mat.index], fontsize=6)
        ax.set_xlabel(r"$\theta$ [%]", fontsize=7)
        ax.set_title(title, fontsize=8)
        thr = vmin + (vmax - vmin) * 0.55
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                if np.isnan(v): continue
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        color="white" if v < thr else "black", fontsize=4.5)
        return im

    im_a = heat(axes[0], pivI, "viridis", *sV, "(a) Cost saving, cost-optimal [%]")
    axes[0].set_ylabel(r"$P$ [€/p/d]", fontsize=7)
    heat(axes[1], pivB, "viridis", *sV, "(b) Cost saving, fleet-balanced [%]")
    im_c = heat(axes[2], pivP, "magma", 0,
                 float(np.ceil(pivP.values.max() / 5) * 5),
                 "(c) Peak-fleet reduction [%]")
    cb1 = fig.colorbar(im_a, ax=axes[1], fraction=0.046, pad=0.03)
    cb1.set_label("[%]", fontsize=6); cb1.ax.tick_params(labelsize=6)
    cb2 = fig.colorbar(im_c, ax=axes[2], fraction=0.046, pad=0.03)
    cb2.set_label("[%]", fontsize=6); cb2.ax.tick_params(labelsize=6)
    savefig(fig, "fig_PF2_saving_fleet_heatmaps", OPT_DIR)


def fig_PF3_sweetspot(s, c, base):
    par = pareto_at_theta1(s, c, base)
    sweet = par[np.isclose(par.penalty, SWEET_P)].iloc[0]
    max_sav = par.sav_pct.max(); max_wait = par.wait_d.max()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W_2COL, 3.2))
    par["sav_n"] = par.sav_pct / max_sav * 100
    par["wait_n"] = par.wait_d / max_wait * 100

    # Panel A: Pareto curve
    ax1.axvspan(par[np.isclose(par.penalty, KNEE_LO)].wait_d.iloc[0],
                 par[np.isclose(par.penalty, KNEE_HI)].wait_d.iloc[0],
                 color="orange", alpha=0.18,
                 label=rf"Knee region $P \in [{KNEE_LO:g}, {KNEE_HI:g}]$")
    ax1.plot(par.wait_d, par.sav_pct, "o-", color="#1d3557",
              linewidth=1.6, markersize=6)
    ax1.scatter([sweet.wait_d], [sweet.sav_pct], marker="*",
                 s=320, color="gold", edgecolor="black", linewidth=1.0, zorder=10,
                 label=rf"Geometric knee $P = {SWEET_P:g}$")
    for _, r in par.iterrows():
        ax1.annotate(rf"${r.penalty:g}$", (r.wait_d, r.sav_pct),
                      xytext=(4, 3), textcoords="offset points",
                      fontsize=6.5, color="#444")
    ax1.set_xlabel(r"Average wait [d] ($\theta = 1$)", fontsize=8.5)
    ax1.set_ylabel("Cost saving vs daily baseline [%]", fontsize=8.5)
    ax1.set_title("(a) Cost-service Pareto frontier", fontsize=9)
    ax1.grid(alpha=0.3); ax1.legend(loc="lower right")

    # Panel B: Diminishing returns
    ax2.plot(par.penalty, par.sav_n, "o-", color="#1d3557",
              linewidth=1.6, markersize=6, label="% of max saving")
    ax2.plot(par.penalty, par.wait_n, "s--", color="#e76f51",
              linewidth=1.6, markersize=6, label="% of max wait")
    ax2.axvspan(KNEE_LO, KNEE_HI, color="orange", alpha=0.18)
    ax2.axvline(SWEET_P, color="#e63946", linestyle=":", linewidth=1)
    ax2.fill_between(par.penalty, par.sav_n, par.wait_n,
                      color="#2a9d8f", alpha=0.13)
    ax2.set_xlabel(r"Service penalty $P$ [€/p/d]", fontsize=8.5)
    ax2.set_ylabel("Fraction of maximum [%]", fontsize=8.5)
    ax2.set_title("(b) Diminishing returns", fontsize=9)
    ax2.legend(loc="upper right"); ax2.grid(alpha=0.3)
    ax2.set_ylim(-5, 105)
    savefig(fig, "fig_PF3_sweetspot", OPT_DIR)


def fig_PF4_per_provider_pareto(s, c, base):
    df = per_provider_pareto(s, c, base)
    fig, ax = plt.subplots(figsize=(W_2COL, 4.0))
    for prov in PROVIDERS:
        d = df[df.provider == prov].sort_values("penalty")
        ax.plot(d.wait_d, d.sav_pct, "o-", color=PROV_COLORS[prov],
                 linewidth=1.6, markersize=5,
                 label=rf"{prov} (baseline {d.baseline_k.iloc[0]:.0f}~k€/wk)")
        sweet = d[np.isclose(d.penalty, SWEET_P)]
        if len(sweet):
            ax.scatter([sweet.wait_d.iloc[0]], [sweet.sav_pct.iloc[0]],
                        marker="*", s=200, color=PROV_COLORS[prov],
                        edgecolor="black", linewidth=0.9, zorder=10)
    ax.set_xlabel(r"Average customer wait [d] ($\theta = 1$)")
    ax.set_ylabel(r"Cost saving vs provider's daily baseline [%]")
    ax.set_title(rf"Per-provider Pareto frontiers — sweet-spot $P = {SWEET_P:g}$ marked",
                  fontsize=9)
    ax.grid(alpha=0.3); ax.legend(loc="lower right", ncol=2, fontsize=7)
    ax.set_xlim(-0.02); ax.set_ylim(-1)
    savefig(fig, "fig_PF4_per_provider_pareto", OPT_DIR)


def chord_knee(wait, sav):
    if len(wait) < 3: return 0
    x = (wait - wait.min()) / (wait.max() - wait.min() + 1e-9)
    y = (sav - sav.min()) / (sav.max() - sav.min() + 1e-9)
    return int(np.argmax(y - x))


def fig_PF5_per_provider_sweetspots(s, c, base):
    df = per_provider_pareto(s, c, base)
    knees = []
    curves = {}
    for prov in PROVIDERS:
        d = df[df.provider == prov].sort_values("penalty").reset_index(drop=True)
        curves[prov] = d
        ki = chord_knee(d.wait_d.values, d.sav_pct.values)
        kr = d.iloc[ki]
        knees.append({"provider": prov,
                       "baseline_k": d.baseline_k.iloc[0],
                       "max_sav": d.sav_pct.max(),
                       "knee_P": kr.penalty,
                       "knee_sav": kr.sav_pct,
                       "knee_wait": kr.wait_d})
    kdf = pd.DataFrame(knees).sort_values("max_sav", ascending=False)
    kdf["category"] = np.where(kdf.max_sav >= 15.0,
                                 "Cost-friendly", "Service-bound")
    kdf.to_csv(OPT_DIR / "tab_per_provider_knees.csv", index=False)

    fig = plt.figure(figsize=(W_2COL, 3.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1], wspace=0.30)
    ax_par = fig.add_subplot(gs[0, 0]); ax_kn = fig.add_subplot(gs[0, 1])

    for prov in kdf.provider:
        d = curves[prov]
        ax_par.plot(d.wait_d, d.sav_pct, "o-", color=PROV_COLORS[prov],
                     linewidth=1.6, markersize=4.5, label=prov)
        kr = kdf[kdf.provider == prov].iloc[0]
        ax_par.scatter([kr.knee_wait], [kr.knee_sav], marker="*",
                        s=260, color=PROV_COLORS[prov],
                        edgecolor="black", linewidth=0.9, zorder=10)
        ax_par.annotate(rf"${kr.knee_P:g}$", (kr.knee_wait, kr.knee_sav),
                         xytext=(5, 4), textcoords="offset points",
                         fontsize=6.5, color=PROV_COLORS[prov])
    ax_par.set_xlabel(r"Average wait [d] ($\theta = 1$)")
    ax_par.set_ylabel(r"Cost saving vs provider's baseline [%]")
    ax_par.set_title("(a) Per-provider Pareto fronts with knees", fontsize=9)
    ax_par.grid(alpha=0.3); ax_par.legend(loc="lower right", ncol=2, fontsize=7)
    ax_par.set_xlim(-0.02); ax_par.set_ylim(-1)

    y = np.arange(len(kdf))
    for i, (_, r) in enumerate(kdf.iterrows()):
        ax_kn.barh(i, r.knee_sav, color=PROV_COLORS[r.provider],
                    edgecolor="black", linewidth=0.4, alpha=0.85)
        ax_kn.text(r.knee_sav + 0.7, i,
                    rf"{r.knee_sav:.1f}% @ $P^* = {r.knee_P:g}$",
                    va="center", fontsize=7)
    ax_kn.set_yticks(y); ax_kn.set_yticklabels(kdf.provider, fontsize=8)
    ax_kn.invert_yaxis()
    ax_kn.set_xlabel("Saving at individual knee [%]")
    ax_kn.set_xlim(0, kdf.knee_sav.max() * 1.7)
    ax_kn.set_title("(b) Individual sweet-spots", fontsize=9)
    ax_kn.grid(axis="x", alpha=0.3)
    savefig(fig, "fig_PF5_per_provider_sweetspots", OPT_DIR)


def fig_O1_value_of_optimization(s, c, base):
    """Random vs Argmin vs Final on 5 representative cells (no recomputation)."""
    src = PATH2 / "_final_vs_baselines.csv"
    if not src.exists():
        print(f"  skip O1 (need {src.name})"); return
    df = pd.read_csv(src)
    metrics = [
        ("sav_pct", "Cost saving [%]", "{:.1f}"),
        ("peak",    "Peak Mo-Sa vehicles", "{:.0f}"),
        ("spread",  "Mo-Sa fleet spread", "{:.0f}"),
        ("wait",    "Mean wait [d]", "{:.2f}"),
    ]
    cols = {"baseline": "#1d3557", "random": "#888",
            "argmin": "#f4a261", "final": "#2a9d8f"}
    labels = {"baseline": "Daily baseline", "random": "Random schedule",
              "argmin": "Per-PLZ argmin (proxy)",
              "final": "Final (Path 2 + smoothing)"}
    fig, axes = plt.subplots(1, 4, figsize=(W_2COL, 2.5))
    x = np.arange(len(df)); w = 0.20
    names = ("baseline", "random", "argmin", "final")
    for ax, (key, label, fmt) in zip(axes, metrics):
        for i, name in enumerate(names):
            vals = df[f"{name}_{key}"].values
            ax.bar(x + (i - 1.5) * w, vals, w, color=cols[name],
                   edgecolor="black", linewidth=0.3, label=labels[name])
        ax.set_xticks(x)
        ax.set_xticklabels(
            [rf"$P={r.penalty:g}$" "\n" rf"$\theta={r.share_willing:g}$"
             for _, r in df.iterrows()], fontsize=6)
        ax.set_ylabel(label, fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        if ax is axes[0]:
            ax.legend(fontsize=6.5, loc="upper left", framealpha=0.9)
    savefig(fig, "fig_O1_value_of_optimization", OPT_DIR)


def fig_SM1_schedule_mix_pct(s, c):
    KEEP_P = [0.0, 0.5, 1.0, 10.0]
    avail = sorted(c.penalty.unique())
    P_VALUES = [p for p in KEEP_P
                 if any(np.isclose(p, a) for a in avail)]
    n = len(P_VALUES)

    fig, axes = plt.subplots(2, n, figsize=(W_2COL, 4.6),
                              sharey=True, sharex=True)

    def stack(ax, sub, col):
        agg = sub.groupby(["share_willing", col]).size().reset_index(name="cnt")
        piv = (agg.pivot(index="share_willing", columns=col, values="cnt")
                .fillna(0).sort_index())
        piv = piv.div(piv.sum(axis=1), axis=0) * 100
        x = piv.index.values * 100
        bottom = np.zeros(len(piv))
        for sz in (2, 3, 4, 5, 6):
            if sz not in piv.columns: continue
            h = piv[sz].values
            ax.fill_between(x, bottom, bottom + h, color=FREQ_COLORS[sz],
                             alpha=0.92)
            bottom += h
        ax.set_xlim(0, 100); ax.set_ylim(0, 100)
        ax.grid(alpha=0.2)

    rows = [("schedule_size_init", "Cost-optimal (init)"),
            ("schedule_size_balanced", "Fleet-balanced")]
    for ri, (col, label) in enumerate(rows):
        for ci, P in enumerate(P_VALUES):
            ax = axes[ri, ci]
            stack(ax, c[np.isclose(c.penalty, P)], col)
            if ri == 0:
                ax.set_title(rf"$P = {P:g}$ €/p/d", fontsize=8.5)
        axes[ri, 0].set_ylabel(label, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel(r"$\theta$ [%]", fontsize=8)
    # Shared legend at bottom
    handles = [Patch(facecolor=FREQ_COLORS[s], label=f"{s} d/wk")
               for s in (2, 3, 4, 5, 6)]
    fig.legend(handles=handles, title="Delivery days per week",
                loc="lower center", ncol=5, frameon=True,
                bbox_to_anchor=(0.5, -0.04),
                fontsize=7.5, title_fontsize=8)
    fig.suptitle(r"Schedule-size mix across $(\theta)$ per service penalty $P$",
                 fontsize=9, y=1.0)
    savefig(fig, "fig_SM1_schedule_mix", OPT_DIR)


# ────────────────────────────────────────────────────────────────────────
#   06_balancing/ figures
# ────────────────────────────────────────────────────────────────────────
def fig_B1_fleet_impact_vs_baseline(f):
    sysd = (f.groupby(["penalty", "share_willing", "day"])
             .agg(fb=("fleet_before", "sum"), fa=("fleet_after", "sum")).reset_index())
    base_by_day = (sysd[np.isclose(sysd.share_willing, 0.0)]
                    .groupby("day").fb.mean().values)
    baseline_peak = float(base_by_day.max())
    baseline_cv = float(base_by_day.std() / base_by_day.mean())
    baseline_total = float(base_by_day.sum())

    rows = []
    for (P, sh), g in sysd.groupby(["penalty", "share_willing"]):
        g = g.sort_values("day"); fa = g.fa.values
        rows.append({"penalty": P, "share_willing": sh,
                      "peak_red": 100 * (baseline_peak - fa.max()) / baseline_peak,
                      "cv_red":   100 * (baseline_cv - fa.std() / fa.mean())
                                    / baseline_cv if fa.mean() > 0 else 0,
                      "total_red": 100 * (baseline_total - fa.sum()) / baseline_total})
    cells = pd.DataFrame(rows)
    cells = cells[~np.isclose(cells.penalty, 0.4)]

    pivP = cells.pivot(index="penalty", columns="share_willing", values="peak_red")
    pivCV = cells.pivot(index="penalty", columns="share_willing", values="cv_red")
    pivT = cells.pivot(index="penalty", columns="share_willing", values="total_red")

    fig, axes = plt.subplots(1, 3, figsize=(W_2COL, 2.6))

    def heat(ax, mat, cmap, vmin, vmax, title, cb_label):
        im = ax.imshow(mat.values, aspect="auto", cmap=cmap,
                        vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels([f"{x*100:.0f}" for x in mat.columns], fontsize=6)
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels([f"{p:g}" for p in mat.index], fontsize=6)
        ax.set_xlabel(r"$\theta$ [%]", fontsize=7)
        ax.set_title(title, fontsize=8)
        thr = vmin + (vmax - vmin) * 0.55
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                if np.isnan(v): continue
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        color="white" if v < thr else "black", fontsize=4.5)
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(cb_label, fontsize=6); cb.ax.tick_params(labelsize=6)

    heat(axes[0], pivP, "magma",
          float(np.floor(pivP.values.min() / 5) * 5),
          float(np.ceil(pivP.values.max() / 5) * 5),
          "(a) Peak-fleet reduction [%]", "%")
    axes[0].set_ylabel(r"$P$ [€/p/d]", fontsize=7)
    heat(axes[1], pivCV, "viridis", 0,
          float(np.ceil(pivCV.values.max() / 10) * 10),
          "(b) Coefficient of variation reduction [%]", "%")
    heat(axes[2], pivT, "magma", 0,
          float(np.ceil(pivT.values.max() / 5) * 5),
          "(c) Total weekly fleet reduction [%]", "%")
    fig.suptitle("Fleet impact of Path-2 pipeline vs daily baseline",
                  fontsize=9, y=1.02)
    savefig(fig, "fig_B1_fleet_impact_vs_baseline", BAL_DIR)


def fig_B2_weekly_fleet_per_provider(f):
    """5-cell x 7-provider grid of Mo-Sa fleet under the FINAL plan vs daily baseline."""
    cells = [(0.0, 0.1), (0.0, 0.5), (0.0, 1.0), (0.5, 0.5), (0.75, 1.0)]
    # baseline pattern from share=0 fleet_before
    bp = (f[np.isclose(f.share_willing, 0.0)]
          .groupby(["provider", "day"]).fleet_before.mean()
          .unstack().T.reindex(range(6)).reindex(columns=PROVIDERS))

    fig, axes = plt.subplots(len(cells), len(PROVIDERS),
                              figsize=(W_2COL, 6.5), sharex=True)
    x = np.arange(6); w = 0.4
    for ri, (P, sh) in enumerate(cells):
        for ci, prov in enumerate(PROVIDERS):
            ax = axes[ri, ci]
            sub = f[(np.isclose(f.penalty, P)) &
                    (np.isclose(f.share_willing, sh)) &
                    (f.provider == prov)]
            after = sub.groupby("day").fleet_after.sum().reindex(range(6)).values
            baseline = bp[prov].values
            ax.bar(x - w/2, baseline, w, color="#1d3557",
                   edgecolor="black", linewidth=0.2)
            ax.bar(x + w/2, after, w, color="#2a9d8f",
                   edgecolor="black", linewidth=0.2)
            ax.set_xticks(x); ax.set_xticklabels(DAYS, fontsize=5.5)
            ax.tick_params(axis="y", labelsize=5.5)
            if ri == 0: ax.set_title(prov, fontsize=8)
            if ci == 0:
                ax.set_ylabel(rf"$P={P:g}$" "\n" rf"$\theta={sh:g}$",
                               fontsize=7, rotation=0, labelpad=22,
                               ha="right", va="center")
            ax.grid(axis="y", alpha=0.25)
    handles = [Patch(color="#1d3557", label="Daily baseline"),
               Patch(color="#2a9d8f", label="Final (Path 2 + balancing)")]
    fig.legend(handles=handles, loc="lower center", ncol=2,
                bbox_to_anchor=(0.5, -0.01), fontsize=8, frameon=False)
    fig.suptitle("Mo-Sa fleet per provider — daily baseline vs final plan",
                  fontsize=9.5, y=1.02)
    savefig(fig, "fig_B2_weekly_fleet_per_provider", BAL_DIR)


# ────────────────────────────────────────────────────────────────────────
#   Main
# ────────────────────────────────────────────────────────────────────────
def main():
    s, c, f, base = load_data()
    print("=== Regenerating 05_optimization/ ===")
    fig_PF1_pareto_path2(s, c, base)
    fig_PF2_saving_fleet_heatmaps(s)
    fig_PF3_sweetspot(s, c, base)
    fig_PF4_per_provider_pareto(s, c, base)
    fig_PF5_per_provider_sweetspots(s, c, base)
    fig_O1_value_of_optimization(s, c, base)
    fig_SM1_schedule_mix_pct(s, c)
    print("=== Regenerating 06_balancing/ ===")
    fig_B1_fleet_impact_vs_baseline(f)
    fig_B2_weekly_fleet_per_provider(f)
    # Clean up old _-prefixed scratch files from 05_optimization
    print("=== Cleaning up scratch files ===")
    for stub in OPT_DIR.glob("_*.png"):
        stub.unlink()
        print(f"  removed {stub.name}")
    for stub in OPT_DIR.glob("_*.csv"):
        stub.unlink()
        print(f"  removed {stub.name}")
    print("\nDone.")


if __name__ == "__main__":
    main()
