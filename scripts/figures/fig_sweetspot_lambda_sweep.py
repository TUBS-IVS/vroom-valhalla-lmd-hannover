"""Lambda (value-of-time) sensitivity sweep for the system sweet-spot.

The Pareto front of (saving, wait) at theta=1 admits more than one
defensible sweet-spot depending on how cost saving is traded against
customer waiting time. We define a utility function

    U(P, lambda) = saving(P) - lambda * wait(P)

where lambda is the "value of time" expressed in percentage points of
cost saving per day of customer wait. Sweeping lambda over a plausible
range identifies the regimes in which different operating points P
dominate.

Left panel: U(P) as a function of lambda for each P in the grid.
Right panel: argmax_P U as a step function of lambda, with the
dominance regions of each operating point shaded.

Output: results/EWGT_Results/fig_sweetspot_lambda.{png,pdf}
"""
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "EWGT_Results"
OUT.mkdir(parents=True, exist_ok=True)

rcParams.update({
    "font.family": "serif", "font.size": 12,
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 12, "axes.titlesize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})

# Pareto front at theta = 1, fleet-balanced pipeline
P_VALUES   = np.array([0.0,  0.25, 0.5,  0.75, 1.0,  2.0,  5.0,  10.0])
SAVING_PCT = np.array([22.8, 18.6, 13.5, 10.2, 7.5,  1.2,  0.0,  0.0])
WAIT_D     = np.array([0.98, 0.45, 0.23, 0.14, 0.09, 0.04, 0.01, 0.01])

# Lambda sweep (pp of saving worth one day of wait)
LAMBDAS = np.linspace(0.0, 50.0, 501)

PALETTE = plt.cm.viridis(np.linspace(0.1, 0.9, len(P_VALUES)))


def main():
    # U(P, lambda) for each P
    U = SAVING_PCT[:, None] - LAMBDAS[None, :] * WAIT_D[:, None]
    best_idx = np.argmax(U, axis=0)
    best_P = P_VALUES[best_idx]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # ---- Left: utility curves ----
    for i, P in enumerate(P_VALUES):
        if SAVING_PCT[i] == 0:
            continue  # Skip dominated points (P >= 5 with zero saving)
        axL.plot(LAMBDAS, U[i], color=PALETTE[i], linewidth=2.2,
                  label=rf"$P = {P:g}$")
    axL.set_xlabel(r"$\lambda$: value of time "
                    r"[pp of cost saving per day of wait]")
    axL.set_ylabel(r"Utility $U(P, \lambda) = "
                    r"\mathrm{saving}(P) - \lambda \cdot \mathrm{wait}(P)$")
    axL.set_title("(a) Utility as a function of the value-of-time weight")
    axL.axhline(0, color="black", linewidth=0.6, alpha=0.5)
    axL.grid(alpha=0.3)
    axL.legend(loc="upper right", ncol=2, fontsize=9.5)

    # ---- Right: dominance regions ----
    # Find transitions
    transitions = []
    cur = best_idx[0]
    cur_start = LAMBDAS[0]
    for i, idx in enumerate(best_idx[1:], start=1):
        if idx != cur:
            transitions.append((cur_start, LAMBDAS[i], cur))
            cur, cur_start = idx, LAMBDAS[i]
    transitions.append((cur_start, LAMBDAS[-1], cur))

    print("Dominance regions:")
    for lo, hi, idx in transitions:
        print(f"  P = {P_VALUES[idx]:5g}  for  lambda in [{lo:5.2f}, {hi:5.2f}]")

    # Shade regions and draw step function
    for lo, hi, idx in transitions:
        axR.axvspan(lo, hi, color=PALETTE[idx], alpha=0.30)
    # Step function of optimal P
    axR.step(LAMBDAS, best_P, where="post", color="black",
              linewidth=2.0)
    # Annotate each region with its P value
    for lo, hi, idx in transitions:
        x_mid = (lo + hi) / 2.0
        y_pos = P_VALUES[idx]
        axR.annotate(rf"$P^\star = {P_VALUES[idx]:g}$",
                      xy=(x_mid, y_pos), xytext=(0, 8),
                      textcoords="offset points", ha="center",
                      fontsize=10.5,
                      bbox=dict(boxstyle="round,pad=0.25",
                                 fc="white", ec="0.6"))
    # Vertical guide markers at common literature values
    for lam_lit, lab in [(15, "literature low"),
                          (25, "literature mid"),
                          (35, "literature high")]:
        axR.axvline(lam_lit, color="grey", linewidth=0.7,
                     linestyle=":")
    axR.set_xlabel(r"$\lambda$: value of time "
                    r"[pp of cost saving per day of wait]")
    axR.set_ylabel(r"Optimal service penalty $P^\star$ [€/p/d]")
    axR.set_title("(b) Dominance regions of the operating point")
    axR.set_ylim(-0.2, max(P_VALUES) + 0.3)
    axR.grid(alpha=0.3)

    fig.tight_layout(w_pad=1.6)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_sweetspot_lambda.{ext}",
                     bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {OUT/'fig_sweetspot_lambda.png'}")


if __name__ == "__main__":
    main()
