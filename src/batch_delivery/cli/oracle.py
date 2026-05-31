"""``batch-delivery oracle-loop`` — variance-driven sample-gen loop."""
from __future__ import annotations

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from batch_delivery import __version__
from batch_delivery.config import load_config

from batch_delivery.cli._app import app, config_app  # noqa: F401


@app.command(name="oracle-loop")
def oracle_loop_cmd(
    sweep_config: Path = typer.Option(
        Path("conf/sweep_multi_provider.yaml"),
        "--sweep-config", "-c",
        help="Base sweep YAML — extended each iteration with new seeds.",
    ),
    iterations: int = typer.Option(
        3, "--iterations", "-n",
        help="Number of oracle-correction rounds.",
    ),
    seeds_per_iter: int = typer.Option(
        3, "--seeds-per-iter", "-s",
        help="Fresh VROOM seeds added per iteration.",
    ),
    base_seed: int = typer.Option(
        3000, "--base-seed",
        help="Starting seed for the per-iteration validation batches.",
    ),
    out_dir: Path = typer.Option(
        Path("results/oracle_loop"),
        "--out-dir",
        help="Where iteration logs, models, and validation summaries go.",
    ),
    training_csv: Path = typer.Option(
        Path("results/sweep_multi/training_matrix.csv"),
        "--training-csv",
        help="Cumulative training CSV — appended each iteration.",
    ),
    arch: str = typer.Option("256,128,64,32", "--arch"),
    alpha: float = typer.Option(1e-4, "--alpha"),
    cv_folds: int = typer.Option(5, "--cv-folds"),
    cv_group: str = typer.Option("plz", "--cv-group"),
    jobs: int = typer.Option(8, "--jobs", "-j"),
    mape_target: float = typer.Option(
        2.0, "--mape-target",
        help="Stop early when validation MAPE drops below this (%%).",
    ),
    variance_growth: float = typer.Option(
        0.35, "--variance-growth",
        help="Per-iteration expansion factor for demand variance "
             "(scales/p_keeps/noise_sigmas). 0 disables curriculum.",
    ),
    variance_max_extra: int = typer.Option(
        2, "--variance-max-extra",
        help="Max extra (low/high) values appended per perturbation axis per iter.",
    ),
    samples_per_iter: int = typer.Option(
        2000, "--samples-per-iter",
        help="Cap on (provider x day x PLZ x perturbation) combos per iteration. "
             "Combined with shuffle_seed yields stratified random sampling.",
    ),
    all_providers: bool = typer.Option(
        True, "--all-providers/--no-all-providers",
        help="Override sweep config to use ALL 7 known providers each iter.",
    ),
    all_days: bool = typer.Option(
        True, "--all-days/--no-all-days",
        help="Override sweep config to use base_days [0..5] (Mon-Sat) each iter.",
    ),
    plzs_per_iter: int = typer.Option(
        0, "--plzs-per-iter",
        help="Number of PLZs sampled per iteration (rotated across iters via "
             "RNG seed). 0 means use sweep config's plzs as-is (full set if null).",
    ),
    max_runtime_min: float = typer.Option(
        0.0, "--max-runtime-min",
        help="Wall-clock budget in minutes. 0 disables. The loop stops "
             "BEFORE starting an iteration once the elapsed time exceeds "
             "this budget (last completed iteration is preserved).",
    ),
    active_sampling: bool = typer.Option(
        True, "--active-sampling/--no-active-sampling",
        help="Bias each iteration's sweep towards (provider, day, PLZ) "
             "regions where the surrogate ensemble disagrees the most "
             "(top --hot-regions rows by per-row prediction std).",
    ),
    hot_regions: int = typer.Option(
        150, "--hot-regions",
        help="Number of high-variance (provider, day, PLZ, agg_k) tuples "
             "to seed the next iteration's sweep with (active sampling).",
    ),
    stability_threshold: float = typer.Option(
        0.5, "--stability-threshold",
        help="Stop early when the mean absolute relative change in "
             "predictions between two consecutive retrain models drops "
             "below this %% on a fixed validation set.",
    ),
    stability_window: int = typer.Option(
        2, "--stability-window",
        help="Number of consecutive iterations the stability criterion "
             "must be satisfied before stopping.",
    ),
    extreme_holdout_frac: float = typer.Option(
        0.2, "--extreme-holdout-frac",
        help="Fraction of each iteration's *extreme* perturbation rows "
             "(scale outside [0.8, 1.2], p_keep<0.85, noise>0.25, or 2+ "
             "knobs perturbed) to reserve as a permanent VROOM-truth "
             "holdout. These rows are NEVER appended to the training pool, "
             "so the per-iter MAPE on the cumulative holdout is a leak-free "
             "measure of how well the surrogate generalises to extreme "
             "tours. 0 disables.",
    ),
    initial_model: Path | None = typer.Option(
        None, "--initial-model",
        help="Reuse a previously-trained MLCostPredictor pickle as the "
             "starting model and SKIP the initial fit. The CLI will still "
             "retrain after every iteration's append. Useful for resuming "
             "an oracle-loop or warm-starting a fresh budget.",
    ),
    frozen_holdout: Path | None = typer.Option(
        None, "--frozen-holdout",
        help="Optional path to a paper-grade frozen test set (built by "
             "`batch-delivery build-holdout`). When given, every iteration's "
             "retrained model is also evaluated on it and the result is "
             "logged to history (cols: frozen_n, frozen_mape_pct, frozen_r2, "
             "frozen_bias_pct). The CSV must carry frozen_holdout=True so "
             "append_to_training refuses to absorb it.",
    ),
) -> None:
    """Closed-loop surrogate-with-oracle-correction (trust-region style).

    For each iteration:

    1. **Sweep** with a fresh VROOM seed batch (validation set).
    2. **Validate** the current surrogate against the new sweep CSV.
    3. **Append** the new rows to the cumulative training CSV (dedup).
    4. **Retrain** the surrogate on the augmented data.
    5. **Stop** when validation MAPE crosses ``--mape-target`` or
       ``--iterations`` is exhausted.

    Per-iteration metrics land in ``oracle_loop_history.csv`` and each model
    snapshot is saved as ``ml_cost_predictor_iter{NN}.pkl``.
    """
    import csv
    import json
    import sys
    import datetime as _dt
    from batch_delivery.runtime import RunContext
    from batch_delivery.surrogate import (
        MLCostPredictor, append_to_training, compute_feature_importance,
        load_training_data, split_extreme_holdout, summarize_training_data,
        train_full_model, validate_against_vroom,
    )
    from batch_delivery.sweep import load_sweep_yaml, run_sweep

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Persistent text log (tee stdout/stderr → oracle_loop.log) ────────
    # Captures every typer.echo + every nested print so we can review the
    # whole run offline (incl. sweep tqdm summaries written to stderr).
    log_path = out_dir / "oracle_loop.log"

    class _Tee:
        def __init__(self, *streams):
            self._streams = streams
        def write(self, data):
            for s in self._streams:
                try:
                    s.write(data)
                    s.flush()
                except Exception:
                    pass
            return len(data) if isinstance(data, str) else 0
        def flush(self):
            for s in self._streams:
                try:
                    s.flush()
                except Exception:
                    pass
        def isatty(self):
            return False

    log_fh = open(log_path, "a", encoding="utf-8", buffering=1)
    log_fh.write(
        f"\n\n========== oracle-loop start "
        f"{_dt.datetime.now().isoformat(timespec='seconds')} ==========\n"
    )
    log_fh.flush()
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(_orig_stdout, log_fh)
    sys.stderr = _Tee(_orig_stderr, log_fh)
    typer.echo(f"oracle-loop: tee log -> {log_path}")

    base_cfg = load_sweep_yaml(sweep_config)
    arch_tuple = tuple(int(x) for x in arch.split(",") if x.strip())
    history_csv = out_dir / "oracle_loop_history.csv"
    holdout_csv = out_dir / "holdout_extreme.csv"

    # snapshot base perturbation grid for the variance curriculum
    base_scales = list(base_cfg.scales)
    base_p_keeps = list(base_cfg.p_keeps)
    base_noise = list(base_cfg.noise_sigmas)
    base_b2c = list(base_cfg.b2c_scales)
    base_b2b = list(base_cfg.b2b_scales)

    # Resolve full coverage overrides — by default the oracle-loop covers
    # ALL providers and ALL six weekdays so the surrogate sees the entire
    # weekly demand distribution. PLZ rotation is handled implicitly via
    # shuffle_seed + samples_per_iter (each iter draws a fresh random subset
    # over the full PLZ pool).
    from batch_delivery.config.constants import PROVIDERS, N_DAYS
    full_providers = list(PROVIDERS) if all_providers else list(base_cfg.providers)
    full_base_days = list(range(N_DAYS)) if all_days else list(base_cfg.base_days)
    if plzs_per_iter and plzs_per_iter > 0 and base_cfg.plzs is not None:
        typer.echo(
            "  (note) --plzs-per-iter ignored: sweep config has an explicit "
            "plz list; shuffle_seed will rotate within that list."
        )
    typer.echo(
        f"oracle-loop coverage: providers={full_providers} days={full_base_days} "
        f"samples/iter={samples_per_iter}"
    )

    # Initial model: either load an existing pickle (warm start) or train
    # from scratch on the cumulative training_csv.
    if initial_model is not None:
        if not initial_model.exists():
            raise typer.BadParameter(
                f"--initial-model not found: {initial_model}"
            )
        typer.echo(f"oracle-loop: warm-starting from {initial_model}")
        model = MLCostPredictor.load(initial_model)
        # Save as iter00 snapshot for traceability.
        model.save(out_dir / "ml_cost_predictor_iter00.pkl")
        model.save(out_dir / "ml_cost_predictor.pkl")
    else:
        typer.echo(f"oracle-loop: training initial model on {training_csv}")
        if not training_csv.exists():
            raise typer.BadParameter(
                f"training CSV not found: {training_csv} — run a sweep first."
            )
        td = load_training_data(training_csv, group_col=cv_group)
        typer.echo(f"  initial rows={len(td.y):,}")
        model = train_full_model(td, arch=arch_tuple, alpha=alpha)
        model_path = out_dir / "ml_cost_predictor_iter00.pkl"
        model.save(model_path)
        model.save(out_dir / "ml_cost_predictor.pkl")

    last_mape: float | None = None
    import time as _time
    t_loop_start = _time.perf_counter()
    budget_sec = max_runtime_min * 60.0 if max_runtime_min and max_runtime_min > 0 else 0.0
    if budget_sec > 0:
        typer.echo(f"oracle-loop: wall-clock budget = {max_runtime_min:.1f} min")

    # ── Active sampling + model-stability state ──────────────────────────
    # `priority_keys`     — top-K (provider, day, plz, agg_k) tuples by
    #                       ensemble std on the current training set, fed
    #                       into the next iteration's sweep so VROOM is
    #                       routed where the surrogate is most uncertain.
    # `prev_pred_snapshot`— predictions of the previous model on a fixed
    #                       validation feature matrix; used to measure
    #                       model-prediction stability between retrains.
    priority_keys: list[tuple[str, int, str, int]] | None = None
    prev_pred_snapshot: object = None  # np.ndarray | None
    stability_streak = 0
    last_stability_pct: float | None = None
    for it in range(1, iterations + 1):
        # ── wall-clock budget check ───────────────────────────────────────
        elapsed_min = (_time.perf_counter() - t_loop_start) / 60.0
        if budget_sec > 0 and (_time.perf_counter() - t_loop_start) >= budget_sec:
            typer.echo(
                f"\nstop: elapsed {elapsed_min:.1f} min >= budget "
                f"{max_runtime_min:.1f} min -- skipping further iterations."
            )
            break
        # ── graceful stop sentinel (created by GUI / external stop) ──────
        stop_sentinel = out_dir / "STOP_REQUESTED"
        if stop_sentinel.exists():
            typer.echo(
                f"\nstop: STOP_REQUESTED sentinel found at {stop_sentinel} "
                f"-- finishing after last completed iteration."
            )
            break
        typer.echo("")
        typer.echo(
            f"=== oracle-loop iteration {it}/{iterations} "
            f"(elapsed {elapsed_min:.1f} min) ==="
        )

        # Phase-level wall-clock timings; written to iteration_summary.json
        # for every iteration so paper plots can break down where the
        # oracle-loop spends its time.
        phase_times: dict[str, float] = {}
        _t_iter_start = _time.perf_counter()
        _t_phase = _time.perf_counter()

        # ── 1) Validation sweep with fresh seeds ──────────────────────────
        new_seeds = [base_seed + (it - 1) * seeds_per_iter + i
                     for i in range(seeds_per_iter)]
        typer.echo(f"validation seeds: {new_seeds}")
        # Variance curriculum: progressively widen the perturbation grid so
        # the surrogate sees both normal and extreme per-PLZ demand levels.
        cur_scales, cur_p_keeps, cur_noise, cur_b2c, cur_b2b = _expand_variance_grid(
            base_scales, base_p_keeps, base_noise,
            iter_idx=it, growth=variance_growth, max_extra=variance_max_extra,
            base_b2c_scales=base_b2c, base_b2b_scales=base_b2b,
        )
        typer.echo(
            f"variance grid: scales={cur_scales}  "
            f"p_keeps={cur_p_keeps}  noise_sigmas={cur_noise}"
        )
        typer.echo(
            f"               b2c_scales={cur_b2c}  b2b_scales={cur_b2b}"
        )
        val_dir = out_dir / f"iter{it:02d}" / "sweep"
        val_dir.mkdir(parents=True, exist_ok=True)
        cfg = base_cfg.model_copy(update={
            "providers": full_providers,
            "base_days": full_base_days,
            "seeds": new_seeds,
            "scales": cur_scales,
            "p_keeps": cur_p_keeps,
            "noise_sigmas": cur_noise,
            "b2c_scales": cur_b2c,
            "b2b_scales": cur_b2b,
            "max_combinations": samples_per_iter,
            "shuffle_seed": base_seed + 1000 * it,
            "priority_keys": priority_keys if active_sampling else None,
            "out_dir": val_dir,
            "out_csv": "validation_matrix.csv",
            "parallel_jobs": jobs,
            "progress": False,
        })
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        ctx = RunContext.create(
            runs_root=cfg.out_dir / "runs",
            cache_root=cfg.out_dir / "cache",
            config=cfg,
            use_cache=cfg.use_cache,
            parallel_jobs=cfg.parallel_jobs,
            run_name=f"oracle_iter_{it:02d}",
        )
        try:
            df_val = run_sweep(cfg, ctx=ctx)
            ctx.finalize(kpis={"n_rows": int(len(df_val)), "iteration": it})
        except Exception:
            ctx.finalize(kpis={"_failed": True, "iteration": it})
            raise
        val_csv = cfg.out_dir / cfg.out_csv
        phase_times["sweep_sec"] = _time.perf_counter() - _t_phase
        _t_phase = _time.perf_counter()

        # ── 2) Validate current surrogate against fresh oracle ─────────────
        val_summary = validate_against_vroom(
            val_csv, model, out_dir=out_dir / f"iter{it:02d}" / "validation",
        )
        phase_times["validate_sec"] = _time.perf_counter() - _t_phase
        _t_phase = _time.perf_counter()
        o = val_summary["overall"]
        typer.echo(
            f"validation: n={o['n']:,}  MAE={o['mae_eur']:.1f}€  "
            f"MAPE={o['mape_pct']:.2f}%  R²={o['r2']:.3f}  "
            f"bias={o['bias_pct']:+.2f}%"
        )
        # Stratified MAPE by perturbation extremity — shows whether the
        # current model improves uniformly or only on the easy middle.
        for bucket in ("baseline", "mild", "extreme"):
            stat = val_summary.get("by_extremity", {}).get(bucket)
            if stat and stat.get("n", 0) > 0:
                typer.echo(
                    f"  [{bucket:>8}]  n={stat['n']:,}  "
                    f"MAPE={stat['mape_pct']:.2f}%  "
                    f"R²={stat['r2']:.3f}  bias={stat['bias_pct']:+.2f}%"
                )
        last_mape = float(o["mape_pct"])
        last_extreme_mape = float(
            val_summary.get("by_extremity", {})
            .get("extreme", {}).get("mape_pct", float("nan"))
        )

        # ── 3a) Reserve extreme rows as permanent leak-free holdout ───────
        # Done BEFORE append_to_training so the held-out rows can never enter
        # the training pool. Each iter contributes a deterministic random
        # slice (per-iter seed) of its extreme perturbations to the growing
        # holdout CSV.
        moved_to_holdout = 0
        if extreme_holdout_frac > 0:
            try:
                _, moved_to_holdout, _ = split_extreme_holdout(
                    val_csv, holdout_csv,
                    fraction=extreme_holdout_frac,
                    seed=base_seed + it,
                )
                if moved_to_holdout:
                    typer.echo(
                        f"holdout: reserved {moved_to_holdout} extreme rows "
                        f"-> {holdout_csv.name} (cumulative)"
                    )
            except Exception as exc:
                typer.echo(f"  (holdout split skipped: {exc})")

        # ── 3) Append remaining rows to cumulative training CSV ───────────
        added = append_to_training(val_csv, training_csv)
        typer.echo(f"append: +{added} net-new rows -> {training_csv}")
        phase_times["append_sec"] = _time.perf_counter() - _t_phase
        _t_phase = _time.perf_counter()

        # ── 4) Retrain on augmented data ──────────────────────────────────
        td = load_training_data(training_csv, group_col=cv_group)
        model = train_full_model(td, arch=arch_tuple, alpha=alpha)
        model_path = out_dir / f"ml_cost_predictor_iter{it:02d}.pkl"
        model.save(model_path)
        model.save(out_dir / "ml_cost_predictor.pkl")
        typer.echo(f"retrain: rows={len(td.y):,}  saved -> {model_path.name}")
        phase_times["retrain_sec"] = _time.perf_counter() - _t_phase
        _t_phase = _time.perf_counter()

        # ── 4b) Evaluate retrained model on cumulative extreme-holdout ────
        # The holdout grows monotonically and was NEVER appended to training,
        # so this is a leak-free measure of generalisation to extreme tours.
        holdout_mape = float("nan")
        holdout_n = 0
        if holdout_csv.exists():
            try:
                ho_summary = validate_against_vroom(
                    holdout_csv, model,
                    out_dir=out_dir / f"iter{it:02d}" / "holdout_eval",
                )
                hb = ho_summary["overall"]
                holdout_mape = float(hb["mape_pct"])
                holdout_n = int(hb["n"])
                typer.echo(
                    f"holdout-eval: n={holdout_n:,}  "
                    f"MAPE={holdout_mape:.2f}%  R²={hb['r2']:.3f}  "
                    f"bias={hb['bias_pct']:+.2f}%  "
                    f"(cumulative leak-free extreme-tour score)"
                )
            except Exception as exc:
                typer.echo(f"  (holdout eval skipped: {exc})")
        phase_times["holdout_eval_sec"] = _time.perf_counter() - _t_phase
        _t_phase = _time.perf_counter()

        # ── 4c) Evaluate on FROZEN paper-grade test set, if provided ──────
        # This file is immutable across iterations and stratified across
        # providers / agg_k / extremity buckets including the new B2C/B2B
        # share-shift axis. It is the cleanest comparison signal across
        # the full oracle-loop (no distribution drift between iterations).
        frozen_mape = float("nan")
        frozen_r2 = float("nan")
        frozen_bias = float("nan")
        frozen_n = 0
        if frozen_holdout is not None and Path(frozen_holdout).exists():
            try:
                fz_summary = validate_against_vroom(
                    frozen_holdout, model,
                    out_dir=out_dir / f"iter{it:02d}" / "frozen_eval",
                )
                fb = fz_summary["overall"]
                frozen_mape = float(fb["mape_pct"])
                frozen_r2 = float(fb["r2"])
                frozen_bias = float(fb["bias_pct"])
                frozen_n = int(fb["n"])
                typer.echo(
                    f"frozen-eval:  n={frozen_n:,}  "
                    f"MAPE={frozen_mape:.2f}%  R²={frozen_r2:.3f}  "
                    f"bias={frozen_bias:+.2f}%  "
                    f"(immutable paper test set)"
                )
                # Per-provider breakdown — useful in the paper appendix.
                for prov, s in fz_summary.get("by_provider", {}).items():
                    if s.get("n", 0) >= 5:
                        typer.echo(
                            f"  frozen[{prov:<8s}] n={s['n']:>4d}  "
                            f"MAPE={s['mape_pct']:>5.2f}%  R²={s['r2']:>5.3f}"
                        )
            except Exception as exc:
                typer.echo(f"  (frozen-holdout eval skipped: {exc})")
        phase_times["frozen_eval_sec"] = _time.perf_counter() - _t_phase
        _t_phase = _time.perf_counter()

        # ── 4d) Permutation feature importance on the validation sweep ────
        # Cheap (few thousand rows × 25 features × 3 repeats) and gives the
        # paper a per-iteration signal of which spatial / demand inputs the
        # surrogate currently relies on. Saved as feature_importance.csv.
        feat_imp_top: list[dict] = []
        try:
            fi = compute_feature_importance(
                val_csv, model,
                out_dir=out_dir / f"iter{it:02d}" / "feature_importance",
                n_repeats=3,
                sample_n=min(2000, int(o["n"])) or 1,
                seed=base_seed + it,
            )
            if not fi.empty:
                top = fi.head(5)
                feat_imp_top = top.to_dict(orient="records")
                typer.echo(
                    "feature importance top-5: "
                    + ", ".join(
                        f"{r['feature']}(+{r['delta_mape_mean']:.2f}%)"
                        for r in feat_imp_top
                    )
                )
        except Exception as exc:
            typer.echo(f"  (feature importance skipped: {exc})")
        phase_times["feature_importance_sec"] = _time.perf_counter() - _t_phase
        _t_phase = _time.perf_counter()

        # ── 4e) Snapshot training-set composition ─────────────────────────
        train_composition: dict = {}
        try:
            train_composition = summarize_training_data(
                training_csv,
                out_path=out_dir / f"iter{it:02d}" / "train_composition.json",
            )
        except Exception as exc:
            typer.echo(f"  (train composition skipped: {exc})")
        phase_times["composition_sec"] = _time.perf_counter() - _t_phase

        # ── append to history CSV ──────────────────────────────────────────
        write_header = not history_csv.exists()
        iter_total_sec = _time.perf_counter() - _t_iter_start
        with history_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow([
                    "iteration", "n_validation", "mae_eur", "mape_pct", "r2",
                    "bias_pct", "extreme_mape_pct", "holdout_n",
                    "holdout_mape_pct", "frozen_n", "frozen_mape_pct",
                    "frozen_r2", "frozen_bias_pct",
                    "n_training_after", "rows_added",
                    "moved_to_holdout",
                    "sweep_sec", "validate_sec", "append_sec", "retrain_sec",
                    "holdout_eval_sec", "frozen_eval_sec",
                    "feature_importance_sec", "composition_sec",
                    "iter_total_sec",
                    "model_path",
                ])
            w.writerow([
                it, o["n"], f"{o['mae_eur']:.4f}", f"{o['mape_pct']:.4f}",
                f"{o['r2']:.6f}", f"{o['bias_pct']:.4f}",
                f"{last_extreme_mape:.4f}",
                holdout_n, f"{holdout_mape:.4f}",
                frozen_n, f"{frozen_mape:.4f}",
                f"{frozen_r2:.6f}", f"{frozen_bias:.4f}",
                len(td.y), added, moved_to_holdout,
                f"{phase_times.get('sweep_sec', 0.0):.2f}",
                f"{phase_times.get('validate_sec', 0.0):.2f}",
                f"{phase_times.get('append_sec', 0.0):.2f}",
                f"{phase_times.get('retrain_sec', 0.0):.2f}",
                f"{phase_times.get('holdout_eval_sec', 0.0):.2f}",
                f"{phase_times.get('frozen_eval_sec', 0.0):.2f}",
                f"{phase_times.get('feature_importance_sec', 0.0):.2f}",
                f"{phase_times.get('composition_sec', 0.0):.2f}",
                f"{iter_total_sec:.2f}",
                str(model_path),
            ])

        # ── consolidated per-iteration summary JSON ───────────────────────
        # One file per iteration capturing EVERYTHING needed offline:
        # validation slices, holdout scores, frozen-test scores, model meta,
        # phase timings, training-data composition, feature importance.
        # Lets the paper re-render any plot without re-running the loop.
        iter_summary = {
            "iteration": it,
            "wall_clock_min_so_far": elapsed_min,
            "iter_total_sec": iter_total_sec,
            "phase_times_sec": phase_times,
            "config": {
                "new_seeds": new_seeds,
                "samples_per_iter": samples_per_iter,
                "providers": full_providers,
                "base_days": full_base_days,
                "scales": cur_scales,
                "p_keeps": cur_p_keeps,
                "noise_sigmas": cur_noise,
                "b2c_scales": cur_b2c,
                "b2b_scales": cur_b2b,
                "active_sampling": bool(active_sampling),
                "hot_regions": hot_regions,
                "n_priority_keys_used": (
                    len(priority_keys) if priority_keys else 0
                ),
            },
            "model": {
                "arch": list(arch_tuple),
                "alpha": alpha,
                "n_pipelines": int(len(getattr(model, "pipelines", []))),
                "saved_to": str(model_path),
            },
            "validation": val_summary,
            "holdout_eval": {
                "n": holdout_n, "mape_pct": holdout_mape,
            },
            "frozen_eval": {
                "n": frozen_n, "mape_pct": frozen_mape,
                "r2": frozen_r2, "bias_pct": frozen_bias,
            },
            "rows_added_to_training": added,
            "moved_to_holdout": moved_to_holdout,
            "training_composition": train_composition,
            "feature_importance_top5": feat_imp_top,
            "stability_pct": last_stability_pct,
            "stability_streak": stability_streak,
        }
        iter_summary_path = out_dir / f"iter{it:02d}" / "iteration_summary.json"
        iter_summary_path.parent.mkdir(parents=True, exist_ok=True)
        iter_summary_path.write_text(
            json.dumps(iter_summary, indent=2, default=float),
            encoding="utf-8",
        )

        # ── plot learning curve so far ────────────────────────────────────
        try:
            _plot_oracle_loop_history(history_csv, out_dir / "learning_curve.png")
        except Exception as exc:
            typer.echo(f"  (plot skipped: {exc})")

        # ── 5a) Active sampling: compute new priority_keys for next iter ──
        if active_sampling and hot_regions > 0:
            try:
                priority_keys = _compute_hot_regions(
                    training_csv, model, top_k=hot_regions,
                )
                typer.echo(
                    f"active sampling: {len(priority_keys)} hot "
                    f"(provider, day, plz, agg_k) regions selected by "
                    f"ensemble std for next iter"
                )
            except Exception as exc:
                typer.echo(f"  (hot-region scan skipped: {exc})")
                priority_keys = None

        # ── 5b) Model-stability check ─────────────────────────────────────
        try:
            import numpy as _np
            import pandas as _pd
            from batch_delivery.features import ALL_COLS as _ALL
            df_val_full = _pd.read_csv(val_csv)
            cur_pred = model.predict(df_val_full[_ALL])
            if prev_pred_snapshot is not None and len(prev_pred_snapshot) == len(cur_pred):  # type: ignore[arg-type]
                denom = _np.maximum(1.0, _np.abs(prev_pred_snapshot))  # type: ignore[arg-type]
                stab_pct = float(
                    _np.mean(_np.abs(cur_pred - prev_pred_snapshot) / denom) * 100.0  # type: ignore[operator]
                )
                last_stability_pct = stab_pct
                typer.echo(
                    f"stability: mean |Δpred|/pred = {stab_pct:.3f}% "
                    f"(threshold {stability_threshold:.3f}%, "
                    f"streak {stability_streak}/{stability_window})"
                )
                if stab_pct <= stability_threshold:
                    stability_streak += 1
                else:
                    stability_streak = 0
            prev_pred_snapshot = cur_pred
        except Exception as exc:
            typer.echo(f"  (stability check skipped: {exc})")

        # ── 6) Early stopping (MAPE OR stability) ──────────────────────────
        if last_mape is not None and last_mape <= mape_target:
            typer.echo(
                f"early stop: MAPE {last_mape:.2f}% ≤ target {mape_target:.2f}%"
            )
            break
        if stability_streak >= stability_window:
            typer.echo(
                f"early stop: model predictions stable "
                f"({last_stability_pct:.3f}% ≤ {stability_threshold:.3f}%) "
                f"for {stability_streak} consecutive iterations"
            )
            break

    typer.echo("")
    typer.echo("oracle-loop done.")
    # ── Final report: aggregate all per-iter summaries into one JSON ──────
    try:
        final_report = _build_final_report(
            out_dir=out_dir,
            history_csv=history_csv,
            holdout_csv=holdout_csv if holdout_csv.exists() else None,
            frozen_holdout=frozen_holdout if (
                frozen_holdout is not None and Path(frozen_holdout).exists()
            ) else None,
            training_csv=training_csv,
            wall_clock_min=(_time.perf_counter() - t_loop_start) / 60.0,
        )
        report_path = out_dir / "oracle_loop_final_report.json"
        report_path.write_text(
            json.dumps(final_report, indent=2, default=float),
            encoding="utf-8",
        )
        typer.echo(f"  final report:  {report_path}")
    except Exception as exc:
        typer.echo(f"  (final report skipped: {exc})")
    typer.echo(f"  history:       {history_csv}")
    typer.echo(f"  latest model:  {out_dir / 'ml_cost_predictor.pkl'}")
    typer.echo(f"  learning curve: {out_dir / 'learning_curve.png'}")
    if holdout_csv.exists():
        typer.echo(f"  holdout (leak-free): {holdout_csv}")
    if frozen_holdout is not None and Path(frozen_holdout).exists():
        typer.echo(f"  frozen test set:     {frozen_holdout}")
    if last_mape is not None:
        typer.echo(f"  final MAPE:    {last_mape:.2f}%")


def _build_final_report(
    *,
    out_dir: Path,
    history_csv: Path,
    holdout_csv: Path | None,
    frozen_holdout: Path | None,
    training_csv: Path,
    wall_clock_min: float,
) -> dict:
    """Aggregate every iter{NN}/iteration_summary.json into one report.

    The report consolidates the entire run into a single JSON suitable for
    the paper's reproducibility appendix:

    * cross-iteration MAPE / R² / bias trajectories (overall + frozen-set);
    * per-iteration training-set growth and timings;
    * final-iteration model metadata + feature-importance top-10;
    * pointers to the cumulative artefacts (training CSV, holdout CSV,
      frozen test set, history CSV, learning curve PNG).
    """
    import json
    import pandas as pd

    iter_files = sorted(out_dir.glob("iter*/iteration_summary.json"))
    iters: list[dict] = []
    for fp in iter_files:
        try:
            iters.append(json.loads(fp.read_text(encoding="utf-8")))
        except Exception:
            continue

    history = (
        pd.read_csv(history_csv).to_dict(orient="records")
        if history_csv.exists() else []
    )

    # Final-iter feature importance (top-10) for the paper headline.
    final_fi: list[dict] = []
    if iter_files:
        last = iter_files[-1].parent / "feature_importance" / "feature_importance.csv"
        if last.exists():
            final_fi = (
                pd.read_csv(last).head(10).to_dict(orient="records")
            )

    # Cross-iter MAPE trajectory (compact form for paper plots).
    trajectories: dict[str, list] = {
        "iteration": [r.get("iteration") for r in history],
        "mape_pct_validation": [r.get("mape_pct") for r in history],
        "mape_pct_extreme_holdout": [r.get("holdout_mape_pct") for r in history],
        "mape_pct_frozen": [r.get("frozen_mape_pct") for r in history],
        "r2_validation": [r.get("r2") for r in history],
        "r2_frozen": [r.get("frozen_r2") for r in history],
        "n_training_after": [r.get("n_training_after") for r in history],
    }

    return {
        "wall_clock_min": float(wall_clock_min),
        "n_iterations": len(iters),
        "training_csv": str(training_csv),
        "history_csv": str(history_csv),
        "holdout_csv": str(holdout_csv) if holdout_csv else None,
        "frozen_holdout": str(frozen_holdout) if frozen_holdout else None,
        "learning_curve_png": str(out_dir / "learning_curve.png"),
        "trajectories": trajectories,
        "final_feature_importance_top10": final_fi,
        "iterations": iters,
    }


def _compute_hot_regions(
    training_csv: Path,
    model: object,
    *,
    top_k: int,
) -> list[tuple[str, int, str, int]]:
    """Return the top-K (provider, base_day, plz, agg_k) tuples ranked by the
    surrogate ensemble's per-row prediction std (= active-sampling targets).

    Uses :py:meth:`MLCostPredictor.predict_with_variance` which evaluates each
    of the 5 seed pipelines independently and returns their disagreement. Rows
    with high std mark regions where the surrogate is most uncertain — routing
    perturbations of those rows through VROOM yields the highest information
    gain per oracle call.
    """
    import pandas as pd
    from batch_delivery.features import ALL_COLS as _ALL

    df = pd.read_csv(training_csv)
    if df.empty:
        return []
    _, std = model.predict_with_variance(df[_ALL])  # type: ignore[attr-defined]
    df = df.copy()
    df["_ens_std"] = std

    keycols = ["provider", "base_day", "plz", "agg_k"]
    keycols = [c for c in keycols if c in df.columns]
    if len(keycols) < 4:
        return []
    agg = (
        df.groupby(keycols, as_index=False)["_ens_std"]
        .max()
        .nlargest(top_k, "_ens_std")
    )
    out: list[tuple[str, int, str, int]] = []
    for _, row in agg.iterrows():
        out.append((
            str(row["provider"]),
            int(row["base_day"]),
            str(row["plz"]),
            int(row["agg_k"]),
        ))
    return out


def _expand_variance_grid(
    base_scales: list[float],
    base_p_keeps: list[float],
    base_noise: list[float],
    *,
    iter_idx: int,
    growth: float,
    max_extra: int,
    base_b2c_scales: list[float] | None = None,
    base_b2b_scales: list[float] | None = None,
) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
    """Per-iteration curriculum: append progressively more extreme perturbations.

    The base grid is always preserved. Each iteration ``k`` (1-based) appends
    extras shaped to match what is *physically realistic* in last-mile
    parcel delivery:

    * ``scales``       — **asymmetric** widening. The upper end grows
                         aggressively (Christmas peak / consolidation
                         simulates 2-3× normal volume) while the lower end
                         only goes down to ~0.5 (slowest off-peak weeks).
                         Geometric multiplier: ``smax * (1 + 0.6*g*iter*j)``
                         vs. ``smin / (1 + 0.3*g*iter*j)``.
    * ``p_keeps``      — bounded by 0.2. We deliberately span the full
                         realistic bandwidth from "almost every stop kept"
                         down to "only ~20 % of stops served" so the model
                         learns sparse-tour cost behaviour as well as the
                         dense-day regime. Below 0.2 the input becomes
                         degenerate (too few stops for a meaningful VRP),
                         so we hard-clip there.
    * ``noise_sigmas`` — additive: ``max(base) + g*iter*j*0.25``.
    * ``b2c_scales``   — **asymmetric upward** only (Christmas surge).
                         Iter ``k`` adds ``1 + 0.4*g*iter*j`` so by iter 3
                         B2C reaches ~2× while B2B stays modest.
    * ``b2b_scales``   — **mild symmetric** (business cycles): adds
                         ``1 ± 0.15*g*iter*j``, capped to ``[0.5, 1.5]``.

    When ``base_b2c_scales`` / ``base_b2b_scales`` are ``None`` or ``[1.0]``
    the curriculum still widens them (so they become a useful learning
    axis); pass an explicit non-trivial base list to override.

    Returns deduplicated, sorted lists in the order
    ``(scales, p_keeps, noise, b2c, b2b)``.
    """
    base_b2c = list(base_b2c_scales) if base_b2c_scales else [1.0]
    base_b2b = list(base_b2b_scales) if base_b2b_scales else [1.0]
    if growth <= 0 or iter_idx <= 0:
        return (
            list(base_scales),
            list(base_p_keeps),
            list(base_noise),
            base_b2c,
            base_b2b,
        )

    smin, smax = min(base_scales), max(base_scales)
    pmin = min(base_p_keeps)
    nmax = max(base_noise) if base_noise else 0.0
    b2c_max = max(base_b2c)
    b2b_min, b2b_max = min(base_b2b), max(base_b2b)

    P_KEEP_FLOOR = 0.2  # span the full realistic bandwidth (sparse..dense)
    SCALE_UP = 0.6      # aggressive upward growth (consolidation peaks)
    SCALE_DN = 0.3      # mild downward growth (slow off-peak)
    B2C_UP = 0.4        # Christmas-style B2C surge growth
    B2B_SYM = 0.15      # B2B mild symmetric variation
    B2B_FLOOR, B2B_CEIL = 0.5, 1.5

    extra_scales: list[float] = []
    extra_pk: list[float] = []
    extra_noise: list[float] = []
    extra_b2c: list[float] = []
    extra_b2b: list[float] = []
    for j in range(1, max_extra + 1):
        up_factor = 1.0 + growth * iter_idx * j * SCALE_UP
        dn_factor = 1.0 + growth * iter_idx * j * SCALE_DN
        extra_scales.append(round(smax * up_factor, 3))
        extra_scales.append(round(smin / dn_factor, 3))
        pk_low = round(max(P_KEEP_FLOOR, pmin - growth * iter_idx * j * 0.6), 3)
        if pk_low < pmin:
            extra_pk.append(pk_low)
        extra_noise.append(round(nmax + growth * iter_idx * j * 0.25, 3))

        b2c_up = round(b2c_max * (1.0 + growth * iter_idx * j * B2C_UP), 3)
        extra_b2c.append(b2c_up)
        b2b_up = round(min(B2B_CEIL, b2b_max * (1.0 + growth * iter_idx * j * B2B_SYM)), 3)
        b2b_dn = round(max(B2B_FLOOR, b2b_min / (1.0 + growth * iter_idx * j * B2B_SYM)), 3)
        extra_b2b.append(b2b_up)
        extra_b2b.append(b2b_dn)

    out_scales = sorted({*base_scales, *(s for s in extra_scales if s > 0.05)})
    out_pk = sorted({
        *(p for p in base_p_keeps if p >= P_KEEP_FLOOR),
        *(p for p in extra_pk if P_KEEP_FLOOR <= p <= 1.0),
    })
    if not out_pk:
        out_pk = [1.0]
    out_noise = sorted({*base_noise, *(n for n in extra_noise if n >= 0)})
    out_b2c = sorted({*base_b2c, *(b for b in extra_b2c if b > 0)})
    out_b2b = sorted({*base_b2b, *(b for b in extra_b2b if B2B_FLOOR <= b <= B2B_CEIL)})
    return out_scales, out_pk, out_noise, out_b2c, out_b2b
