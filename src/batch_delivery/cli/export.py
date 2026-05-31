"""Result-export commands: optimization-results, build-holdout."""
from __future__ import annotations

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from batch_delivery import __version__
from batch_delivery.config import load_config

from batch_delivery.cli._app import app, config_app  # noqa: F401


@app.command(name="export-optimization-results")
def export_optimization_results_cmd(
    config_path: Path = typer.Option(
        None, "--config", "-c",
        help="Pipeline YAML (defaults to conf/default.yaml). Used only for "
             "out_dir / providers context.",
    ),
    checkpoint: str = typer.Option(
        "08_sa_ml_optimization", "--checkpoint",
        help="Stage checkpoint name to load.",
    ),
    out_dir: Path = typer.Option(
        None, "--out-dir",
        help="Override output root (default: <results>/optimization).",
    ),
) -> None:
    """Re-emit per-provider schedule CSVs + plots from an existing pipeline run.

    Loads the SA_ML stage checkpoint (no re-routing), reconstructs matrices
    from ``ml_optimization_data``, and writes:

    * ``schedule_assignment.csv`` — per-PLZ delivery days
    * ``daily_volume.csv`` — weekday-level shares
    * ``hub_fleet_by_day.csv``
    * ``schedule_heatmap.png``, ``daily_volume.png``
    * ``optimization_summary.json``

    Outputs go under ``out_dir/<provider>/<scenario>/``.
    """
    from batch_delivery.optimization.results import export_optimization_results
    from batch_delivery.utils.core import load_checkpoint

    cfg = load_config(config_path)
    base_out = out_dir or (Path(cfg.results_dir) / "optimization")
    base_out.mkdir(parents=True, exist_ok=True)

    typer.echo(f"export: loading checkpoint '{checkpoint}'")
    ck = load_checkpoint(checkpoint)
    if ck is None:
        raise typer.BadParameter(
            f"checkpoint '{checkpoint}' not found — run the pipeline first."
        )

    # The provider data is needed to reconstruct hub_names / hub_plz_list etc.
    # We reload it from the optimization-prep checkpoint.
    od_ck = load_checkpoint("07_optimization_prep")
    if od_ck is None:
        raise typer.BadParameter(
            "checkpoint '07_optimization_prep' not found — needed for hub arrays."
        )
    optimization_data = od_ck["optimization_data"]
    ml_optimization_data = ck["ml_optimization_data"]

    n_written = 0
    for provider, ml in ml_optimization_data.items():
        odata = optimization_data[provider]
        for sc_tag, sc_key, mx_key in (
            ("sa_ml_express", "bal_ml_expr", "matrices_ml_expr"),
            ("sa_ml_batch", "bal_ml_batch", "matrices_ml_batch"),
        ):
            cd_result = ml[sc_key]
            matrices = ml[mx_key]
            target = base_out / provider.lower() / sc_tag
            export_optimization_results(
                plz_keys=odata["plz_keys"],
                plz_hub_arr=odata["plz_hub_arr"],
                hub_names=odata["hub_names"],
                hub_plz_list=odata["hub_plz_list"],
                schedules=odata["schedules"],
                matrices=matrices,
                cd_result=cd_result,
                out_dir=target,
                summary_extras={
                    "provider": provider,
                    "scenario": sc_tag,
                    "total_cost": float(cd_result.get("cost",
                                                       cd_result.get("best_cost", 0.0))),
                },
            )
            typer.echo(f"  {provider:<8s} {sc_tag:<16s} -> {target}")
            n_written += 1

    typer.echo("")
    typer.echo(f"export-optimization-results done. {n_written} scenarios written.")


@app.command(name="build-holdout")
def build_holdout_cmd(
    sweep_config: Path = typer.Option(
        Path("conf/sweep_multi_provider.yaml"),
        "--sweep-config", "-c",
        help="Base sweep YAML — providers / agg_ks / PLZs are taken from here.",
    ),
    out_csv: Path = typer.Option(
        Path("results/holdout/frozen_holdout.csv"),
        "--out-csv",
        help="Where to write the frozen-holdout matrix.",
    ),
    n_target: int = typer.Option(
        500, "--n-target",
        help="Target number of VROOM-routed rows (stratified random sample). "
             "Increase for a more statistically powerful paper test set.",
    ),
    holdout_seed_base: int = typer.Option(
        9000, "--holdout-seed-base",
        help="Seed base for the frozen holdout. Use a value disjoint from the "
             "training-sweep seed range so a row can never appear in both. "
             "The default 9000 leaves the conventional 0..8999 range free for "
             "training sweeps.",
    ),
    n_seeds: int = typer.Option(
        3, "--n-seeds",
        help="Number of distinct holdout seeds (rows are stratified across them).",
    ),
    jobs: int = typer.Option(8, "--jobs", "-j"),
) -> None:
    """Build a stratified, frozen VROOM test set for paper-grade evaluation.

    Why a separate frozen holdout?
      * The per-iter validation sweep inside ``oracle-loop`` always feeds
        most of its rows back into the training pool — so MAPE on it is
        out-of-sample for *that iteration's seeds* but does not give a
        clean before/after comparison across iterations.
      * The cumulative ``holdout_extreme.csv`` only contains *extreme*
        perturbations and grows monotonically.
      * For the paper we want a single, immutable test set that:
        - covers all 7 providers x 4 agg_k values x 3 extremity buckets,
        - includes the new B2C/B2B share-shift axis,
        - uses seeds disjoint from any training run (``>= 9000``),
        - is flagged with ``frozen_holdout=True`` so :func:`append_to_training`
          will refuse to absorb it even if mis-pointed at it.

    The output CSV is stratified-random-sampled to ``--n-target`` rows over
    a comprehensive perturbation grid that spans:

    * scales        ``[0.5, 1.0, 1.5, 2.0]``        off-peak / normal / Christmas / extreme peak
    * p_keeps       ``[1.0, 0.8, 0.5, 0.3]``        full sparsity bandwidth
    * noise_sigmas  ``[0.0, 0.3]``                  smooth / noisy demand
    * b2c_scales    ``[1.0, 1.5, 2.0]``             normal / Christmas-surge B2C
    * b2b_scales    ``[1.0, 0.7]``                  normal / summer-quiet B2B
    """
    from batch_delivery.runtime import RunContext
    from batch_delivery.sweep import load_sweep_yaml, run_sweep
    import pandas as _pd

    base_cfg = load_sweep_yaml(sweep_config)
    holdout_seeds = [holdout_seed_base + i for i in range(n_seeds)]

    # Comprehensive paper-grade grid; explicit so the holdout is reproducible
    # regardless of what the active training-sweep config currently has.
    holdout_grid = {
        "scales": [0.5, 1.0, 1.5, 2.0],
        "p_keeps": [1.0, 0.8, 0.5, 0.3],
        "noise_sigmas": [0.0, 0.3],
        "b2c_scales": [1.0, 1.5, 2.0],
        "b2b_scales": [1.0, 0.7],
    }

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    work_dir = out_csv.parent / "_build"
    work_dir.mkdir(parents=True, exist_ok=True)

    cfg = base_cfg.model_copy(update={
        "seeds": holdout_seeds,
        **holdout_grid,
        "max_combinations": n_target,
        "shuffle_seed": holdout_seed_base,  # deterministic stratification
        "out_dir": work_dir,
        "out_csv": "frozen_holdout_raw.csv",
        "parallel_jobs": jobs,
        "progress": True,
    })
    typer.echo(
        f"build-holdout: target={n_target} rows  seeds={holdout_seeds}  "
        f"providers={cfg.providers}"
    )
    ctx = RunContext.create(
        runs_root=work_dir / "runs",
        cache_root=work_dir / "cache",
        config=cfg,
        use_cache=cfg.use_cache,
        parallel_jobs=cfg.parallel_jobs,
        run_name="build_holdout",
    )
    try:
        df = run_sweep(cfg, ctx=ctx)
        ctx.finalize(kpis={"n_rows": int(len(df))})
    except Exception:
        ctx.finalize(kpis={"_failed": True})
        raise

    raw_csv = cfg.out_dir / cfg.out_csv
    df = _pd.read_csv(raw_csv)
    df["frozen_holdout"] = True
    df.to_csv(out_csv, index=False)

    # Quick coverage report so paper review sees the design.
    typer.echo("")
    typer.echo(f"frozen-holdout written: {out_csv}  ({len(df)} rows)")
    if "provider" in df.columns and "agg_k" in df.columns:
        cov = df.groupby(["provider", "agg_k"]).size().unstack(fill_value=0)
        typer.echo("coverage (rows per provider x agg_k):")
        for line in cov.to_string().splitlines():
            typer.echo(f"  {line}")
    typer.echo(
        "use it from oracle-loop with: "
        "--frozen-holdout " + str(out_csv)
    )
