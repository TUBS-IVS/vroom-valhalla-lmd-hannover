"""3D Sensitivity — (postponement_window, share_willing, service_penalty P).

Extension of `willingness_to_wait_daganzo.py`:
  * adds a third dimension `P ∈ {0, 0.10, 0.25, 0.50, 0.75, 1.0, 2.0, 5.0}`
  * 3 × 11 × 8 = 264 cells, all evaluated on the cached cost matrix (~1 s).

Outputs (results/willingness_3d/):
    fig3d_heatmap_cost.{png,pdf}        # 3 panels: x=penalty, y=share, color=cost (per window)
    fig3d_heatmap_wait.{png,pdf}        # same layout, color=wait
    fig3d_pareto_lines_per_window.{png,pdf}    # one panel per window, lines per penalty
    fig3d_scatter.{png,pdf}             # 3D scatter (wait, cost, share) coloured by penalty
    fig3d_savings_envelope.{png,pdf}    # saving curve per (window, P) overlay
    tab_3d_grid.csv                     # 264 rows × KPIs
"""
from __future__ import annotations
import pickle, sys, time, warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import pandas as pd
from matplotlib import rcParams

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 11, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})

OUT = ROOT / "results" / "willingness_3d"
OUT.mkdir(parents=True, exist_ok=True)

N_DAYS = 6
MAX_HOLD_CACHED = 3
SHARE_GRID    = np.linspace(0.0, 1.0, 11)
WINDOW_GRID   = [1, 2, 3]
PENALTY_GRID  = np.array([0.0, 0.10, 0.25, 0.50, 0.75, 1.0, 2.0, 5.0])

WIN_COLOR  = {1: "#003049", 2: "#2a9d8f", 3: "#e76f51"}


def log(msg):
    print(msg, flush=True)


def enumerate_schedules(max_hold):
    out = []
    for k in range(1, N_DAYS + 1):
        for combo in combinations(range(N_DAYS), k):
            days = sorted(combo)
            ok = True
            for i in range(len(days)):
                gap = (days[(i + 1) % len(days)] - days[i]) % N_DAYS
                if gap == 0:
                    gap = N_DAYS
                if gap > max_hold:
                    ok = False
                    break
            if ok:
                out.append(frozenset(days))
    return out


def avg_wait_days(schedule_days):
    if not schedule_days:
        return 0.0
    ds = sorted(schedule_days)
    total = 0.0
    for di in range(N_DAYS):
        next_dd = min(((d - di) % N_DAYS, d) for d in ds)[1]
        wait = (next_dd - di) % N_DAYS
        total += wait
    return total / N_DAYS


def main():
    t0 = time.time()
    log("3D Sweep: window × share × penalty  (Daganzo-LGB-Hybrid, blended cost)")

    cache = np.load(ROOT / "results/penalty_sweep/sched_cost_cache.npz")
    sched_cost = cache["sched_cost"]
    prov_cache = list(cache["prov_order"])
    plz_cache  = list(cache["plz_order"])
    log(f"  cache shape: {sched_cost.shape}")

    chk = pickle.load(open(ROOT / "results/checkpoints/01_demand.pkl", "rb"))
    provider_data = chk["provider_data"]
    n_pp_arr = []
    pp_meta = []
    for prov, plz in zip(prov_cache, plz_cache):
        pld = provider_data[prov]["plz_demand"]
        row = pld[pld.plz == plz]
        if row.empty:
            continue
        row = row.iloc[0]
        n_pp_arr.append(int(row.weekly_parcels))
        pp_meta.append({"provider": prov, "plz": plz})
    n_pp_arr = np.array(n_pp_arr, dtype=np.float64)
    n_pp = len(pp_meta)
    tot_pkts = float(n_pp_arr.sum())

    schedules = enumerate_schedules(MAX_HOLD_CACHED)
    sched_sizes = np.array([len(s) for s in schedules])
    sched_waits = np.array([avg_wait_days(sorted(s)) for s in schedules])
    daily_idx = int(np.where(sched_sizes == N_DAYS)[0][0])
    cost_daily = sched_cost[:, daily_idx]
    tot_cost_daily = float(cost_daily.sum())

    window_mask = {w: np.array([s in set(enumerate_schedules(w)) for s in schedules])
                   for w in WINDOW_GRID}

    log("  Sweeping 3 × 11 × 8 = 264 cells ...")
    rows = []
    # cache best batched cost & wait per (window, penalty)
    batched_cache = {}  # (w, P) -> (cost_batched[n_pp], wait_batched[n_pp])
    for w in WINDOW_GRID:
        wm = window_mask[w]
        for P in PENALTY_GRID:
            combined = sched_cost + P * n_pp_arr[:, None] * sched_waits[None, :]
            combined_w = np.where(wm[None, :], combined, np.inf)
            best_si = np.argmin(combined_w, axis=1)
            cost_b = sched_cost[np.arange(n_pp), best_si]
            wait_b = sched_waits[best_si]
            batched_cache[(w, float(P))] = (cost_b, wait_b)

    for w in WINDOW_GRID:
        for P in PENALTY_GRID:
            cost_b, wait_b = batched_cache[(w, float(P))]
            for s in SHARE_GRID:
                blended_cost = (1 - s) * cost_daily + s * cost_b
                blended_wait = s * wait_b
                total_cost = float(blended_cost.sum())
                avg_wait = float((blended_wait * n_pp_arr).sum() / tot_pkts)
                saving_pct = 100.0 * (tot_cost_daily - total_cost) / tot_cost_daily
                rows.append({
                    "window": w, "penalty": float(P), "share_willing": float(s),
                    "total_cost_eur": total_cost,
                    "avg_wait_days": avg_wait,
                    "cost_savings_pct": saving_pct,
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tab_3d_grid.csv", index=False)
    log(f"  {len(df)} cells written")

    # ---------- Plot 1: cost heatmaps (3 panels per window) ----------
    log("[plot] cost heatmaps per window ...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    cmap = "viridis_r"
    vmin = df.total_cost_eur.min() / 1e3
    vmax = df.total_cost_eur.max() / 1e3
    for ai, w in enumerate(WINDOW_GRID):
        sub = df[df.window == w]
        piv = sub.pivot(index="share_willing", columns="penalty",
                          values="total_cost_eur").sort_index() / 1e3
        im = axes[ai].imshow(piv.values, aspect="auto", cmap=cmap,
                              vmin=vmin, vmax=vmax, origin="lower")
        axes[ai].set_xticks(range(len(PENALTY_GRID)))
        axes[ai].set_xticklabels([f"{p:g}" for p in PENALTY_GRID])
        axes[ai].set_yticks(range(len(SHARE_GRID)))
        axes[ai].set_yticklabels([f"{int(s*100)}%" for s in SHARE_GRID])
        axes[ai].set_xlabel("Service penalty $P$ [€/parcel/day]")
        axes[ai].set_title(f"Window = {w} day{'s' if w != 1 else ''}")
        for i, srow in enumerate(piv.index):
            for j, pcol in enumerate(piv.columns):
                v = piv.values[i, j]
                axes[ai].text(j, i, f"{v:.0f}", ha="center", va="center",
                                color="white" if (v - vmin) / max(1, vmax - vmin) < 0.55 else "black",
                                fontsize=7)
    axes[0].set_ylabel("Share of customers willing to wait")
    cax = fig.add_axes([0.92, 0.18, 0.014, 0.65])
    plt.colorbar(im, cax=cax, label="Weekly routing cost [k€]")
    fig.suptitle("Weekly routing cost across the (share, penalty, window) cube",
                  fontsize=12, y=1.03)
    fig.tight_layout(rect=[0, 0, 0.91, 1.0])
    fig.savefig(OUT / "fig3d_heatmap_cost.png")
    fig.savefig(OUT / "fig3d_heatmap_cost.pdf")
    plt.close(fig)

    # ---------- Plot 2: wait heatmaps ----------
    log("[plot] wait heatmaps ...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    vmin_w = df.avg_wait_days.min()
    vmax_w = df.avg_wait_days.max()
    for ai, w in enumerate(WINDOW_GRID):
        sub = df[df.window == w]
        piv = sub.pivot(index="share_willing", columns="penalty",
                          values="avg_wait_days").sort_index()
        im = axes[ai].imshow(piv.values, aspect="auto", cmap="RdPu",
                              vmin=vmin_w, vmax=vmax_w, origin="lower")
        axes[ai].set_xticks(range(len(PENALTY_GRID)))
        axes[ai].set_xticklabels([f"{p:g}" for p in PENALTY_GRID])
        axes[ai].set_yticks(range(len(SHARE_GRID)))
        axes[ai].set_yticklabels([f"{int(s*100)}%" for s in SHARE_GRID])
        axes[ai].set_xlabel("Service penalty $P$ [€/parcel/day]")
        axes[ai].set_title(f"Window = {w} day{'s' if w != 1 else ''}")
        for i, srow in enumerate(piv.index):
            for j, pcol in enumerate(piv.columns):
                v = piv.values[i, j]
                axes[ai].text(j, i, f"{v:.2f}", ha="center", va="center",
                                color="white" if (v - vmin_w) / max(1e-6, vmax_w - vmin_w) > 0.55 else "black",
                                fontsize=7)
    axes[0].set_ylabel("Share of customers willing to wait")
    cax = fig.add_axes([0.92, 0.18, 0.014, 0.65])
    plt.colorbar(im, cax=cax, label="Average customer wait [days]")
    fig.suptitle("Customer-wait burden across the (share, penalty, window) cube",
                  fontsize=12, y=1.03)
    fig.tight_layout(rect=[0, 0, 0.91, 1.0])
    fig.savefig(OUT / "fig3d_heatmap_wait.png")
    fig.savefig(OUT / "fig3d_heatmap_wait.pdf")
    plt.close(fig)

    # ---------- Plot 3: per-window Pareto lines per penalty ----------
    log("[plot] per-window Pareto lines ...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    pen_colors = plt.cm.viridis(np.linspace(0.1, 0.95, len(PENALTY_GRID)))
    for ai, w in enumerate(WINDOW_GRID):
        sub_w = df[df.window == w]
        for pi, P in enumerate(PENALTY_GRID):
            sub = sub_w[sub_w.penalty == P].sort_values("share_willing")
            axes[ai].plot(sub.avg_wait_days, sub.total_cost_eur / 1e3,
                            "o-", color=pen_colors[pi], linewidth=1.5,
                            markersize=4, alpha=0.85,
                            label=f"$P={P:g}$" if ai == 2 else None)
        axes[ai].set_title(f"Window = {w} day{'s' if w != 1 else ''}")
        axes[ai].set_xlabel("Average customer wait [days]")
        axes[ai].grid(alpha=0.3)
    axes[0].set_ylabel("Weekly routing cost [k€]")
    axes[-1].legend(title="Service penalty\n[€/parcel/day]",
                    bbox_to_anchor=(1.0, 1.0), loc="upper left", fontsize=8)
    fig.suptitle(
        "Pareto trade-off across willingness, penalty and postponement window\n"
        "(curves run from 0% → 100% willing; each curve is one penalty)",
        fontsize=12, y=1.04,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig3d_pareto_lines_per_window.png")
    fig.savefig(OUT / "fig3d_pareto_lines_per_window.pdf")
    plt.close(fig)

    # ---------- Plot 4: 3D scatter (wait, cost, share) coloured by penalty ----------
    log("[plot] 3D scatter ...")
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")
    markers = {1: "o", 2: "s", 3: "^"}
    for w in WINDOW_GRID:
        sub = df[df.window == w]
        sc = ax.scatter(sub.avg_wait_days, sub.share_willing * 100,
                          sub.total_cost_eur / 1e3,
                          c=sub.penalty, cmap="viridis", marker=markers[w],
                          s=28, alpha=0.85, edgecolor="none")
    ax.set_xlabel("Avg customer wait [d]", labelpad=8)
    ax.set_ylabel("Share willing [%]", labelpad=8)
    ax.set_zlabel("Weekly cost [k€]", labelpad=8)
    cbar = fig.colorbar(sc, ax=ax, pad=0.07, shrink=0.7)
    cbar.set_label("Service penalty $P$ [€/parcel/day]")
    # Window legend
    handles = [plt.Line2D([0], [0], marker=markers[w], linestyle="",
                           color="black", markersize=7, label=f"Window {w}d")
                for w in WINDOW_GRID]
    ax.legend(handles=handles, loc="upper left")
    ax.view_init(elev=22, azim=-58)
    ax.set_title("(Wait, share, cost) cube coloured by service penalty",
                  y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "fig3d_scatter.png")
    fig.savefig(OUT / "fig3d_scatter.pdf")
    plt.close(fig)

    # ---------- Plot 5: savings-envelope (cost-saving curves) ----------
    log("[plot] savings envelopes ...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ai, w in enumerate(WINDOW_GRID):
        sub_w = df[df.window == w]
        for pi, P in enumerate(PENALTY_GRID):
            sub = sub_w[sub_w.penalty == P].sort_values("share_willing")
            axes[ai].plot(sub.share_willing * 100, sub.cost_savings_pct,
                            "-", color=pen_colors[pi], linewidth=1.6,
                            label=f"$P={P:g}$" if ai == 2 else None)
        axes[ai].set_title(f"Window = {w} day{'s' if w != 1 else ''}")
        axes[ai].set_xlabel("Share willing [%]")
        axes[ai].grid(alpha=0.3)
    axes[0].set_ylabel("Cost saving vs all-daily baseline [%]")
    axes[-1].legend(title="$P$ [€/pkt/d]",
                    bbox_to_anchor=(1.0, 1.0), loc="upper left", fontsize=8)
    fig.suptitle("Cost-saving envelopes across willingness, penalty and window",
                  fontsize=12, y=1.04)
    fig.tight_layout()
    fig.savefig(OUT / "fig3d_savings_envelope.png")
    fig.savefig(OUT / "fig3d_savings_envelope.pdf")
    plt.close(fig)

    # REPORT
    log("[3] Writing REPORT.md ...")
    lines = [
        "# 3D Sensitivity (window × share × service-penalty) — Daganzo-LGB-Hybrid",
        f"\nBaseline (all-daily): **{tot_cost_daily/1e3:,.0f} k€**",
        f"Grid: 3 windows × 11 shares × 8 penalties = {len(df)} cells",
        "\n## Cost saving at maximum willingness (share = 100%)",
        "| Window | $P=0$ | $P=0.25$ | $P=0.50$ | $P=1.0$ | $P=5.0$ |",
        "|---|---|---|---|---|---|",
    ]
    for w in WINDOW_GRID:
        sub = df[(df.window == w) & (df.share_willing == 1.0)].set_index("penalty")
        cells = [f"{sub.loc[float(P), 'cost_savings_pct']:.1f}%"
                 if float(P) in sub.index else "—"
                 for P in [0.0, 0.25, 0.5, 1.0, 5.0]]
        lines.append(f"| {w} day | " + " | ".join(cells) + " |")
    lines.append(
        "\n*Reading the cube:* Increasing **share** moves the customer-side "
        "(more accept batching). Increasing **window** loosens the structural "
        "constraint (more valid schedules). Increasing **penalty $P$** raises "
        "the operator-side cost of service, pushing the optimizer toward "
        "denser delivery. Saving is greatest at high share + wide window + "
        "low penalty; service-cost trade-off elbow remains near $P=0.5$."
    )
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    log(f"\nDone in {time.time()-t0:.0f}s. Outputs in: {OUT}")
    for p in sorted(OUT.glob("*")):
        log(f"  {p.name}")


if __name__ == "__main__":
    main()
