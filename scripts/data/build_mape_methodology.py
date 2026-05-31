"""Saubere MAPE-Methodik-Tabelle fuer das Paper.

Reportet pro Iteration drei klar benannte Metriken:
  * mape_cv_oof    : Mean APE auf der Sweep-Validation (Out-of-Fold per iter).
  * mape_holdout   : Mean APE auf dem Frozen Extreme Holdout (Iter-uebergreifend).
  * mape_cv_oof_per_provider : per-LSP Breakdown der OOF MAPE.

Plus Bootstrap-95%-CIs auf dem Holdout, und eine Lernkurve ueber iter01..iter20.

Liest aus:
  results/oracle_loop_extended_2026_05_22/iterNN/iteration_summary.json
  results/oracle_loop_extended_2026_05_22/iterNN/holdout_eval/validation_residuals.csv
  results/oracle_loop_extended_2026_05_22/iterNN/sweep/validation_matrix.csv

Outputs:
  results/paper_figures/final/tab_mape_methodology.csv
  results/paper_figures/final/tab_mape_methodology_by_provider.csv
  results/paper_figures/final/fig_M1_mape_learning_curve.{pdf,png}
  results/paper_figures/final/methods_mape_snippet.md
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "oracle_loop_extended_2026_05_22"
OUT = ROOT / "results" / "paper_figures" / "final"
OUT.mkdir(parents=True, exist_ok=True)


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)])
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def compute_mape_ci(residuals_csv: Path) -> tuple[float, float, float, int]:
    if not residuals_csv.exists():
        return (float("nan"), float("nan"), float("nan"), 0)
    df = pd.read_csv(residuals_csv)
    ape = (df["actual_eur"] - df["predicted_eur"]).abs() / df["actual_eur"].clip(lower=1)
    lo, hi = bootstrap_ci(ape.to_numpy())
    return float(ape.mean() * 100), float(lo * 100), float(hi * 100), len(df)


def collect_per_iter():
    rows = []
    iter_dirs = sorted([d for d in RUN.iterdir() if d.is_dir() and re.match(r"iter\d+$", d.name)])
    for d in iter_dirs:
        it_num = int(d.name.replace("iter", ""))
        summary_path = d / "iteration_summary.json"
        if not summary_path.exists():
            continue
        s = json.load(open(summary_path))

        cv = s.get("validation", {}).get("overall", {})
        ho = s.get("holdout_eval", {}) or {}

        # Bootstrap CIs from raw residuals (more honest than summary point)
        cv_resid_csv = d / "validation" / "validation_residuals.csv"
        ho_resid_csv = d / "holdout_eval" / "validation_residuals.csv"
        cv_mape, cv_lo, cv_hi, cv_n = compute_mape_ci(cv_resid_csv)
        ho_mape, ho_lo, ho_hi, ho_n = compute_mape_ci(ho_resid_csv)

        rows.append({
            "iteration": it_num,
            "n_train": s.get("training_composition", {}).get("n_rows"),
            "cv_oof_mape_pct": cv.get("mape_pct"),  # from summary
            "cv_oof_n": cv.get("n"),
            "cv_oof_mape_ci_lo": cv_lo,
            "cv_oof_mape_ci_hi": cv_hi,
            "holdout_mape_pct": ho.get("mape_pct"),
            "holdout_n": ho.get("n"),
            "holdout_mape_ci_lo": ho_lo,
            "holdout_mape_ci_hi": ho_hi,
            "stability_pct": s.get("stability_pct"),
            "stability_streak": s.get("stability_streak"),
        })
    return pd.DataFrame(rows)


def collect_per_provider_last(iter_dir: Path) -> pd.DataFrame:
    summary_path = iter_dir / "iteration_summary.json"
    s = json.load(open(summary_path))
    by_prov = s.get("validation", {}).get("by_provider", {})
    rows = []
    for prov, m in by_prov.items():
        rows.append({
            "provider": prov,
            "n_oof": m.get("n"),
            "cv_oof_mape_pct": m.get("mape_pct"),
            "cv_oof_r2": m.get("r2"),
            "cv_oof_bias_pct": m.get("bias_pct"),
        })

    # Plus per-provider holdout MAPE
    ho_csv = iter_dir / "holdout_eval" / "validation_residuals.csv"
    df_ho = pd.DataFrame()
    if ho_csv.exists():
        df_ho = pd.read_csv(ho_csv)
        df_ho["ape"] = (df_ho["actual_eur"] - df_ho["predicted_eur"]).abs() / df_ho["actual_eur"].clip(lower=1)
        ho_by_prov = df_ho.groupby("provider").agg(
            n_holdout=("ape", "size"), holdout_mape_pct=("ape", "mean")
        )
        ho_by_prov["holdout_mape_pct"] = ho_by_prov["holdout_mape_pct"] * 100
        ho_by_prov = ho_by_prov.reset_index()

    df = pd.DataFrame(rows)
    if not df_ho.empty:
        df = df.merge(ho_by_prov, on="provider", how="left")
    return df.sort_values("provider")


def plot_learning_curve(df: pd.DataFrame, out_png: Path):
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # CV-OOF with CI band
    ax.plot(df["iteration"], df["cv_oof_mape_pct"], "o-", color="#1f77b4",
             label="CV out-of-fold MAPE", lw=2)
    ax.fill_between(df["iteration"], df["cv_oof_mape_ci_lo"], df["cv_oof_mape_ci_hi"],
                      color="#1f77b4", alpha=0.18)

    # Holdout with CI band
    ax.plot(df["iteration"], df["holdout_mape_pct"], "s-", color="#d62728",
             label="Frozen Extreme Holdout MAPE", lw=2)
    ax.fill_between(df["iteration"], df["holdout_mape_ci_lo"], df["holdout_mape_ci_hi"],
                      color="#d62728", alpha=0.18)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("MAPE  [%]")
    ax.set_title("Fig M1 — Surrogate learning curve  (oracle_loop_extended_2026_05_22)",
                  loc="left", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    # Right-axis: training pool size
    ax2 = ax.twinx()
    ax2.plot(df["iteration"], df["n_train"], "k--", lw=1, alpha=0.5)
    ax2.set_ylabel("Training-pool size  [rows]", color="k", fontsize=9, alpha=0.7)
    fig.tight_layout()
    fig.savefig(out_png.with_suffix(".pdf"))
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def write_methods_snippet(df_iter: pd.DataFrame, df_prov: pd.DataFrame, out_md: Path):
    last = df_iter.iloc[-1]
    lines = []
    lines.append("# MAPE-Methodik (paper methods section snippet)\n")
    lines.append(
        "Wir berichten drei nicht-vermischte MAPE-Schaetzer fuer das Surrogate-MLP-Ensemble:\n"
    )
    lines.append("1. **CV out-of-fold (cv_oof_mape_pct)** — pro Iteration k=5-faltige GroupKFold-CV "
                  "ueber PLZ-Gruppen auf dem akkumulierten Trainings-Pool. Ist der ehrliche In-Pool-Generalisierungsschaetzer.\n")
    lines.append("2. **Frozen Extreme Holdout (holdout_mape_pct)** — eingefrorenes Out-of-Pool-Set "
                  "mit Demand-Perturbationen, das nie ins Training kommt. Ist der ehrliche Out-of-Pool-Schaetzer.\n")
    lines.append("3. Wir verwenden **nicht** den naiv-berechneten In-Sample-MAPE (fit-then-predict auf dem Trainingsdatensatz),"
                  " weil dieser den Generalisierungsfehler unterschaetzt.\n")
    lines.append(
        f"\nFinal-Iteration ({int(last['iteration'])}) — Surrogate-Quality:\n\n"
    )
    lines.append("| Metrik | n | MAPE | Bootstrap-95%-CI |")
    lines.append("|---|---:|---:|:---:|")
    lines.append(
        f"| CV out-of-fold | {int(last['cv_oof_n']) if pd.notna(last['cv_oof_n']) else 'n/a'} | "
        f"{last['cv_oof_mape_pct']:.2f}% | [{last['cv_oof_mape_ci_lo']:.2f}, {last['cv_oof_mape_ci_hi']:.2f}] |"
    )
    lines.append(
        f"| Frozen Extreme Holdout | {int(last['holdout_n'])} | "
        f"{last['holdout_mape_pct']:.2f}% | [{last['holdout_mape_ci_lo']:.2f}, {last['holdout_mape_ci_hi']:.2f}] |"
    )
    lines.append("")
    lines.append("## Per-LSP Breakdown (Iter " + str(int(last["iteration"])) + ")\n")
    lines.append("| Provider | CV OOF n | CV OOF MAPE | Holdout n | Holdout MAPE |")
    lines.append("|---|---:|---:|---:|---:|")
    for _, r in df_prov.iterrows():
        lines.append(
            f"| {r['provider']} | {int(r['n_oof']) if pd.notna(r['n_oof']) else '-'} | "
            f"{r['cv_oof_mape_pct']:.2f}% | "
            f"{int(r['n_holdout']) if pd.notna(r.get('n_holdout')) else '-'} | "
            f"{r['holdout_mape_pct']:.2f}%" if pd.notna(r.get('holdout_mape_pct')) else "n/a"
        )
        lines[-1] = lines[-1] + " |"
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main():
    print("Collecting per-iter MAPE...")
    df_iter = collect_per_iter()
    df_iter.to_csv(OUT / "tab_mape_methodology.csv", index=False)
    print(f"  {len(df_iter)} iterations")
    print(df_iter[["iteration", "n_train", "cv_oof_mape_pct", "holdout_mape_pct", "stability_pct"]].to_string(index=False))

    last_iter_dir = RUN / f"iter{int(df_iter['iteration'].max()):02d}"
    df_prov = collect_per_provider_last(last_iter_dir)
    df_prov.to_csv(OUT / "tab_mape_methodology_by_provider.csv", index=False)
    print("\nPer-provider (last iter):")
    print(df_prov.to_string(index=False))

    plot_learning_curve(df_iter, OUT / "fig_M1_mape_learning_curve.png")
    write_methods_snippet(df_iter, df_prov, OUT / "methods_mape_snippet.md")

    print(f"\nOutputs in {OUT}")


if __name__ == "__main__":
    main()
