"""Surrogate-model commands: train/tune/validate/learn-loop."""
from __future__ import annotations

from pathlib import Path

import typer

from batch_delivery.cli._app import app, config_app  # noqa: F401


@app.command(name="train-surrogate")
def train_surrogate(
    data: Path = typer.Option(
        Path("results/sweep_multi/training_matrix.csv"),
        "--data", "-d",
        help="Sweep training-matrix CSV.",
    ),
    out_dir: Path = typer.Option(
        Path("results/surrogate"),
        "--out-dir",
        help="Where to write model + history.",
    ),
    cv_folds: int = typer.Option(5, "--cv-folds", help="Number of CV folds."),
    cv_group: str = typer.Option(
        "plz", "--cv-group",
        help="Group column for GroupKFold CV (default: plz).",
    ),
    arch: str = typer.Option(
        "256,128,64,32", "--arch",
        help="Hidden-layer sizes (comma-separated).",
    ),
    alpha: float = typer.Option(1e-4, "--alpha", help="L2 regularisation strength."),
    iteration: int = typer.Option(
        1, "--iteration",
        help="Iteration number to record in the history CSV (1 = single fit).",
    ),
    no_history: bool = typer.Option(False, "--no-history", help="Skip history append."),
) -> None:
    """Train MLCostPredictor on a sweep CSV with k-fold CV reporting."""
    import json

    from batch_delivery.surrogate import (
        append_iteration_row,
        cross_validate,
        load_training_data,
        train_full_model,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    arch_tuple = tuple(int(x) for x in arch.split(",") if x.strip())

    typer.echo(f"train: loading {data}")
    td = load_training_data(data, group_col=cv_group)
    typer.echo(
        f"  rows={len(td.y):,}  groups={len(set(td.groups)):,}  "
        f"baselines={int(td.is_baseline.sum()):,}"
    )

    metrics = cross_validate(
        td, k=cv_folds, arch=arch_tuple, alpha=alpha, group_aware=True,
    )

    o = metrics["overall"]
    b = metrics["baseline_overall"]
    typer.echo("")
    typer.echo(f"CV ({metrics['cv_kind']}):")
    typer.echo(
        f"  overall   MAE={o['mae']:>7.1f} EUR  "
        f"MAPE={o['mape']:>5.2f}%  R²={o['r2']:>5.3f}"
    )
    typer.echo(
        f"  baseline  MAE={b['mae']:>7.1f} EUR  "
        f"MAPE={b['mape']:>5.2f}%  R²={b['r2']:>5.3f}  "
        f"(n={metrics['n_baseline']})"
    )

    typer.echo("")
    typer.echo("train: fitting final model on all data ...")
    model = train_full_model(td, arch=arch_tuple, alpha=alpha)
    model_path = out_dir / "ml_cost_predictor.pkl"
    model.save(model_path)

    metrics_path = out_dir / "cv_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {k: v for k, v in metrics.items() if k != "oof_pred"},
            indent=2, default=float,
        ),
        encoding="utf-8",
    )

    if not no_history:
        history = out_dir / "iteration_history.csv"
        append_iteration_row(
            history, iteration=iteration, metrics=metrics,
            extra={"data_csv": str(data)},
        )
        typer.echo(f"  history:  {history}")

    typer.echo(f"  model:    {model_path}")
    typer.echo(f"  metrics:  {metrics_path}")


@app.command(name="learn-loop")
def learn_loop(
    sweep_config: Path = typer.Option(
        Path("conf/sweep_multi_provider.yaml"),
        "--sweep-config", "-c",
        help="Base sweep YAML — its seeds are extended each iteration.",
    ),
    iterations: int = typer.Option(3, "--iterations", "-n", help="Number of train iterations."),
    seeds_per_iter: int = typer.Option(
        3, "--seeds-per-iter", "-s",
        help="Fresh seed batch added each iteration.",
    ),
    base_seed: int = typer.Option(
        2026, "--base-seed",
        help="Starting seed for the per-iteration batches.",
    ),
    out_dir: Path = typer.Option(
        Path("results/surrogate"),
        "--out-dir",
        help="Where to write model, history, and per-iteration logs.",
    ),
    cv_folds: int = typer.Option(5, "--cv-folds"),
    cv_group: str = typer.Option("plz", "--cv-group"),
    arch: str = typer.Option("256,128,64,32", "--arch"),
    alpha: float = typer.Option(1e-4, "--alpha"),
    jobs: int = typer.Option(8, "--jobs", "-j", help="Parallel sweep workers."),
    max_combinations: int = typer.Option(
        None, "--max",
        help="Optional cap on combos *per iteration* (after caching).",
    ),
) -> None:
    """Iteratively grow the training set and retrain the surrogate.

    Each iteration:
      1. Adds a new seed-batch of size ``seeds-per-iter`` to the sweep config.
      2. Runs the sweep (cache reuses previous iterations' routings).
      3. Loads the cumulative training_matrix.csv.
      4. Cross-validates + fits MLCostPredictor.
      5. Appends one row to ``iteration_history.csv``.

    Watch the history CSV's MAE / R² columns to see the learning curve.
    """
    from batch_delivery.runtime import RunContext
    from batch_delivery.surrogate import (
        append_iteration_row,
        cross_validate,
        load_training_data,
        train_full_model,
    )
    from batch_delivery.sweep import load_sweep_yaml, run_sweep

    out_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = load_sweep_yaml(sweep_config)
    arch_tuple = tuple(int(x) for x in arch.split(",") if x.strip())

    cumulative_seeds: list[int] = list(base_cfg.seeds)
    history_csv = out_dir / "iteration_history.csv"

    for it in range(1, iterations + 1):
        new_seeds = [base_seed + (it - 1) * seeds_per_iter + i
                     for i in range(seeds_per_iter)]
        cumulative_seeds = sorted(set(cumulative_seeds) | set(new_seeds))
        typer.echo("")
        typer.echo(f"=== iteration {it}/{iterations} ===")
        typer.echo(f"new seeds:    {new_seeds}")
        typer.echo(f"total seeds:  {len(cumulative_seeds)}")

        # --- sweep with extended seed list (cache reuses prior runs) ------
        cfg = base_cfg.model_copy(update={
            "seeds": cumulative_seeds,
            "parallel_jobs": jobs,
            "max_combinations": max_combinations,
            "progress": False,
        })
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        ctx = RunContext.create(
            runs_root=cfg.out_dir / "runs",
            cache_root=cfg.out_dir / "cache",
            config=cfg,
            use_cache=cfg.use_cache,
            parallel_jobs=cfg.parallel_jobs,
            run_name=f"learn_iter_{it:02d}",
        )
        try:
            df_sweep = run_sweep(cfg, ctx=ctx)
            ctx.finalize(kpis={"n_rows": len(df_sweep), "iteration": it})
        except Exception:
            ctx.finalize(kpis={"_failed": True, "iteration": it})
            raise

        # --- load full cumulative CSV + cross-validate --------------------
        data_csv = cfg.out_dir / cfg.out_csv
        td = load_training_data(data_csv, group_col=cv_group)
        typer.echo(
            f"training data: rows={len(td.y):,} "
            f"groups={len(set(td.groups)):,} "
            f"baselines={int(td.is_baseline.sum()):,}"
        )
        metrics = cross_validate(
            td, k=cv_folds, arch=arch_tuple, alpha=alpha, group_aware=True,
        )
        o = metrics["overall"]
        b = metrics["baseline_overall"]
        typer.echo(
            f"CV: MAE={o['mae']:.1f}€  MAPE={o['mape']:.2f}%  R²={o['r2']:.3f}"
            f"  | baseline R²={b['r2']:.3f}"
        )

        # --- append history ----------------------------------------------
        append_iteration_row(
            history_csv,
            iteration=it,
            metrics=metrics,
            extra={
                "data_csv": str(data_csv),
                "n_seeds": len(cumulative_seeds),
                "new_seeds": ",".join(str(s) for s in new_seeds),
                "run_id": ctx.run_id,
            },
        )

        # --- save iteration model ----------------------------------------
        model = train_full_model(td, arch=arch_tuple, alpha=alpha)
        model_path = out_dir / f"ml_cost_predictor_iter{it:02d}.pkl"
        model.save(model_path)
        latest_path = out_dir / "ml_cost_predictor.pkl"
        model.save(latest_path)

    typer.echo("")
    typer.echo(f"learn-loop done. history: {history_csv}")
    typer.echo(f"  latest model: {out_dir / 'ml_cost_predictor.pkl'}")


@app.command(name="plot-learning-curve")
def plot_learning_curve_cmd(
    history: Path = typer.Option(
        Path("results/surrogate/iteration_history.csv"),
        "--history", "-h",
        help="Iteration-history CSV from train-surrogate / learn-loop.",
    ),
    metrics_json: Path = typer.Option(
        None, "--metrics-json",
        help="Optional cv_metrics.json for per-fold detail plot.",
    ),
    data_csv: Path = typer.Option(
        None, "--data",
        help="Optional sweep CSV for an honest pred-vs-actual scatter.",
    ),
    out_dir: Path = typer.Option(
        Path("results/surrogate/figures"),
        "--out-dir",
        help="Where to write PNGs.",
    ),
    scatter_arch: str = typer.Option(
        "64,32", "--scatter-arch",
        help="Small arch for the pred-vs-actual scatter (kept fast).",
    ),
    scatter_k: int = typer.Option(5, "--scatter-k"),
) -> None:
    """Render learning-curve / fold-detail / pred-vs-actual PNGs."""
    from batch_delivery.surrogate.plots import (
        plot_fold_detail,
        plot_learning_curve,
        plot_pred_vs_actual,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    if not history.exists():
        raise typer.BadParameter(f"history not found: {history}")
    plot_learning_curve(history, out_dir)

    if metrics_json is not None and metrics_json.exists():
        plot_fold_detail(metrics_json, out_dir)
    elif metrics_json is not None:
        typer.echo(f"  (skip fold detail — {metrics_json} missing)")

    if data_csv is not None and data_csv.exists():
        arch_t = tuple(int(x) for x in scatter_arch.split(",") if x.strip())
        plot_pred_vs_actual(
            data_csv, out_dir, k=scatter_k, arch=arch_t, seeds=[42, 123],
        )
    elif data_csv is not None:
        typer.echo(f"  (skip scatter — {data_csv} missing)")

    typer.echo(f"plots written to {out_dir}")


@app.command(name="tune-surrogate")
def tune_surrogate_cmd(
    data: Path = typer.Option(
        Path("results/sweep_multi/training_matrix.csv"),
        "--data", "-d",
        help="Sweep training-matrix CSV.",
    ),
    out_dir: Path = typer.Option(
        Path("results/surrogate/tuning"),
        "--out-dir",
        help="Where to write tuning_results.csv + tuning_best.json.",
    ),
    cv_folds: int = typer.Option(3, "--cv-folds"),
    cv_group: str = typer.Option("plz", "--cv-group"),
    n_random: int = typer.Option(
        12, "--n-random",
        help="Random-search budget; 0 = full grid.",
    ),
    seeds: str = typer.Option(
        "42,123", "--seeds",
        help="Ensemble seeds *per trial* (kept short for speed).",
    ),
    max_iter: int = typer.Option(
        400, "--max-iter",
        help="MLP max_iter per trial (early stopping still applies).",
    ),
    objective: str = typer.Option(
        "mape_pct", "--objective",
        help="Metric to minimise: mape_pct | mae_eur | r2 (maximised).",
    ),
    refit_full: bool = typer.Option(
        True, "--refit-full/--no-refit-full",
        help="After tuning, refit the full 5-seed ensemble with the best config.",
    ),
) -> None:
    """Random/grid hyperparameter search over arch x alpha x learning_rate.

    Each trial uses a small ensemble + reduced max_iter for affordability;
    the winner is then refit with the full :data:`ENSEMBLE_SEEDS` ensemble.
    """
    from batch_delivery.surrogate import (
        ENSEMBLE_SEEDS,
        load_training_data,
        train_full_model,
        tune_hyperparameters,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    seed_list = [int(s) for s in seeds.split(",") if s.strip()]

    typer.echo(f"tune: loading {data}")
    td = load_training_data(data, group_col=cv_group)
    typer.echo(
        f"  rows={len(td.y):,}  groups={len(set(td.groups)):,}  "
        f"baselines={int(td.is_baseline.sum()):,}"
    )

    best_cfg, df_results = tune_hyperparameters(
        td,
        n_random=n_random if n_random > 0 else None,
        k=cv_folds,
        seeds=seed_list,
        max_iter=max_iter,
        group_aware=True,
        out_dir=out_dir,
        objective=objective,
    )

    typer.echo("")
    typer.echo(f"best config: {best_cfg.label()}")
    sort_col = "r2" if objective == "r2" else objective
    head = df_results.sort_values(
        sort_col, ascending=(objective != "r2")
    ).head(5)
    typer.echo("top 5 trials:")
    for _, r in head.iterrows():
        typer.echo(
            f"  arch={r['arch']:<18s} alpha={r['alpha']:.0e} "
            f"lr={r['lr_init']:.0e}  "
            f"MAPE={r['mape_pct']:>5.2f}%  R²={r['r2']:>5.3f}  "
            f"MAE={r['mae_eur']:>6.1f}€"
        )

    if refit_full:
        typer.echo("")
        typer.echo("tune: refitting full 5-seed ensemble with best config ...")
        model = train_full_model(
            td,
            arch=best_cfg.arch,
            alpha=best_cfg.alpha,
            seeds=ENSEMBLE_SEEDS,
        )
        model_path = out_dir / "ml_cost_predictor_best.pkl"
        model.save(model_path)
        typer.echo(f"  best model: {model_path}")


@app.command(name="validate-surrogate")
def validate_surrogate_cmd(
    sweep_csv: Path = typer.Option(
        ..., "--sweep-csv", "-s",
        help="Fresh VROOM sweep CSV (acts as the oracle).",
    ),
    model_path: Path = typer.Option(
        Path("results/surrogate/ml_cost_predictor.pkl"),
        "--model", "-m",
        help="Trained MLCostPredictor pickle.",
    ),
    out_dir: Path = typer.Option(
        Path("results/surrogate/validation"),
        "--out-dir",
        help="Where to write residuals + summary.",
    ),
    append_to: Path = typer.Option(
        None, "--append-to",
        help="If given, append the sweep rows to this training CSV "
        "(dedup by cache-key) — closes the trust-region loop.",
    ),
) -> None:
    """Compare surrogate predictions to VROOM truth on a fresh sweep batch.

    Computes residuals, MAPE / R² overall and per-provider, and writes
    ``validation_residuals.csv`` + ``validation_summary.json``.  Use
    ``--append-to`` to fold the new rows into the training set so the
    next ``train-surrogate`` iteration learns from them — this is the
    surrogate-with-oracle-correction loop (trust-region style).
    """
    from batch_delivery.surrogate import (
        MLCostPredictor,
        append_to_training,
        validate_against_vroom,
    )

    if not model_path.exists():
        raise typer.BadParameter(f"model not found: {model_path}")
    if not sweep_csv.exists():
        raise typer.BadParameter(f"sweep CSV not found: {sweep_csv}")

    typer.echo(f"validate: loading model {model_path}")
    model = MLCostPredictor.load(model_path)

    summary = validate_against_vroom(sweep_csv, model, out_dir=out_dir)
    o = summary["overall"]
    typer.echo("")
    typer.echo(
        f"VROOM-validation:  n={o['n']:,}  "
        f"MAE={o['mae_eur']:.1f}€  MAPE={o['mape_pct']:.2f}%  "
        f"R²={o['r2']:.3f}  bias={o['bias_pct']:+.2f}%"
    )
    typer.echo("per-provider:")
    for prov, s in summary["by_provider"].items():
        typer.echo(
            f"  {prov:<8s} n={s['n']:>4d}  "
            f"MAPE={s['mape_pct']:>5.2f}%  R²={s['r2']:>5.3f}  "
            f"bias={s['bias_pct']:+5.2f}%"
        )

    if append_to is not None:
        added = append_to_training(sweep_csv, append_to)
        typer.echo("")
        typer.echo(f"append: +{added} rows -> {append_to}")


def _plot_oracle_loop_history(history_csv: Path, out_path: Path) -> None:
    """Render a 3-panel learning curve.

    Panel 1 — Validation MAPE: overall MAPE (red) on the *current iter's*
    fresh sweep plus, when available, the MAPE on the cumulative leak-free
    extreme-tour holdout (purple, dashed). The gap between the two lines
    shows how much harder extreme tours are than the average input.

    Panel 2 — Validation R² on the current iter's sweep.

    Panel 3 — Cumulative training rows; the holdout count is annotated so
    the share of routed VROOM calls that became training vs. permanent
    holdout is visible at a glance.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(history_csv)
    if df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    axes[0].plot(df["iteration"], df["mape_pct"], "o-", color="C3",
                 label="overall (this iter)")
    if "holdout_mape_pct" in df.columns:
        ho = df[df["holdout_n"] > 0]
        if not ho.empty:
            axes[0].plot(
                ho["iteration"], ho["holdout_mape_pct"],
                "s--", color="C4",
                label="extreme holdout (cumulative)",
            )
    if "extreme_mape_pct" in df.columns:
        ex = df[df["extreme_mape_pct"].notna()]
        if not ex.empty:
            axes[0].plot(
                ex["iteration"], ex["extreme_mape_pct"],
                "^:", color="C1", alpha=0.7,
                label="extreme (this iter)",
            )
    if "frozen_mape_pct" in df.columns:
        fz = df[df["frozen_n"] > 0]
        if not fz.empty:
            axes[0].plot(
                fz["iteration"], fz["frozen_mape_pct"],
                "D-", color="black", linewidth=2,
                label="frozen test set (paper)",
            )
    axes[0].set_ylabel("MAPE (%)")
    axes[0].set_xlabel("Iteration")
    axes[0].set_title("Surrogate convergence")
    axes[0].legend(fontsize=8, loc="best")
    axes[0].grid(alpha=0.3)

    axes[1].plot(df["iteration"], df["r2"], "o-", color="C2")
    axes[1].set_ylabel("Validation R²")
    axes[1].set_xlabel("Iteration")
    axes[1].grid(alpha=0.3)
    r2_min = float(df["r2"].min())
    axes[1].set_ylim(min(0.99, r2_min - 0.005), 1.0)

    axes[2].plot(df["iteration"], df["n_training_after"], "o-", color="C0",
                 label="training pool")
    if "holdout_n" in df.columns:
        axes[2].plot(df["iteration"], df["holdout_n"], "s--", color="C4",
                     label="extreme holdout")
        axes[2].legend(fontsize=8, loc="best")
    axes[2].set_ylabel("Cumulative rows")
    axes[2].set_xlabel("Iteration")
    axes[2].grid(alpha=0.3)

    fig.suptitle("Oracle-loop learning curve", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
