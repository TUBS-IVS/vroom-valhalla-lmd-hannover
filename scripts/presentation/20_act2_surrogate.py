"""Act 2 -- the surrogate: how a routing solver is replaced by a model.

Figures
  fig21_progression    pure Daganzo -> scaled Daganzo -> hybrid, out-of-fold
  fig22_pool           what the training pool is made of
  fig23_alpha          why one global alpha, not one per provider
  fig24_determinism    coordinate descent restart spread
  fig25_importance     which features carry the learned residual

fig21 and fig25 refit the residual model here rather than reading a stored
prediction, using the same GroupKFold-by-PLZ protocol and the same
hyper-parameters as scripts/revision/11_alpha_sensitivity.py. Grouping by
postal-code area matters: the pool contains several perturbed samples of the
same area, so a random split would leak and flatter the model.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "revision"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

import _data as D
import _plots as P
import _style as S

ACT = "2 - Surrogate"
BASIS = ("sweep_v3_mergefix training pool (2733 VROOM samples); model and pool "
         "are unchanged by the Stage-3 revision")

# Same hyper-parameters as the production/alpha-sensitivity fits.
LGB_HPS = dict(n_estimators=1000, learning_rate=0.05, num_leaves=31,
               max_depth=-1, subsample=0.85, colsample_bytree=0.85,
               reg_lambda=0.5, min_child_samples=10, n_jobs=4,
               random_state=42, verbosity=-1)

_CACHE: dict = {}


def _oof():
    """Out-of-fold hybrid prediction on the training pool, cached per run."""
    if "oof" in _CACHE:
        return _CACHE["oof"]

    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    import _stage3_common as C
    from batch_delivery.features import ALL_COLS
    from batch_delivery.surrogate import build_combo_features

    pool = D.load_training_pool()
    y = pool.actual_cost_eur.values.astype(float)
    dag = C.DaganzoLGBHybrid._daganzo_vec(
        pool.n_parcels.values, pool.n_stops.values,
        pool.area_km2.values, pool.hub_dist_km.values)
    groups = pool.plz.astype(str).values
    combo = build_combo_features(pool[ALL_COLS])
    X = combo.values.astype(float)

    alpha = float(np.median(y / np.maximum(dag, 1.0)))
    model = C.load_model()
    assert abs(alpha - model.alpha) < 0.01, (
        f"recomputed alpha {alpha:.3f} != production alpha {model.alpha:.3f}")
    print(f"  alpha = {alpha:.3f} (matches production model)")

    base = alpha * dag
    oof = np.zeros_like(y)
    gains = np.zeros(X.shape[1])
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = lgb.LGBMRegressor(**LGB_HPS)
        m.fit(X[tr], y[tr] - base[tr])
        oof[te] = base[te] + m.predict(X[te])
        gains += m.booster_.feature_importance(importance_type="gain")

    _CACHE["oof"] = (pool, y, dag, alpha, oof,
                     pd.Series(gains / 5, index=combo.columns))
    return _CACHE["oof"]


def _met(actual, pred):
    e = (pred - actual) / np.maximum(1e-6, actual)
    r2 = 1 - ((actual - pred) ** 2).sum() / ((actual - actual.mean()) ** 2).sum()
    return float(np.abs(e).mean() * 100), float(e.mean() * 100), float(r2)


# ---------------------------------------------------------------- fig21
def fig21_progression():
    pool, y, dag, alpha, oof, _ = _oof()
    panels = [
        (r"(a) Daganzo, $\alpha = 1$", dag),
        (rf"(b) Daganzo, $\alpha = {alpha:.3f}$", alpha * dag),
        (r"(c) Daganzo-LGB hybrid (out-of-fold)", oof),
    ]
    for title, pred in panels:
        mape, bias, r2 = _met(y, pred)
        print(f"  {title}: MAPE {mape:.1f}% bias {bias:+.1f}% R2 {r2:.3f}")

    fmt = FuncFormatter(lambda v, _: f"{int(v):,}".replace(",", " "))
    lo, hi = y.min() * 0.8, y.max() * 1.25

    for style in S.styles():
        S.apply(style)
        fig, axes = plt.subplots(1, 3, sharex=True, sharey=True,
                                 figsize=S.figsize(style, (11.5, 4.1),
                                                   (16.0, 5.6)))
        colors = np.array([D.PROVIDER_COLOR.get(p, "#888888")
                           for p in pool.provider])
        for ax, (title, pred) in zip(axes, panels):
            ax.scatter(y, pred, s=S.scale(style, 6, 2.2), c=colors, alpha=0.45,
                       edgecolor="none", rasterized=True)
            ax.plot([lo, hi], [lo, hi], "k--",
                    linewidth=S.scale(style, 0.9, 1.5), alpha=0.8)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_locator(LogLocator(base=10))
                axis.set_minor_locator(NullLocator())
                axis.set_major_formatter(fmt)
            mape, bias, r2 = _met(y, pred)
            ax.set_title(f"{title}\nMAPE {mape:.1f}%  ·  bias {bias:+.1f}%  ·  "
                         rf"$R^2$ {r2:.3f}",
                         fontsize=9 if style == "paper" else 14)
            ax.grid(alpha=0.25)
        axes[0].set_ylabel("Predicted cost [€]")
        fig.supxlabel("VROOM cost per postal-code area [€]")
        handles = [plt.Line2D([], [], marker="o", linestyle="", color=c,
                              label=p, markersize=S.scale(style, 5, 1.4))
                   for p, c in D.PROVIDER_COLOR.items()]
        axes[2].legend(handles=handles, fontsize=None, framealpha=0.9,
                       loc="upper left", ncol=2)
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        S.save(fig, "fig21_progression", style, S.TIER_A)

    m1 = _met(y, dag)
    m2 = _met(y, alpha * dag)
    m3 = _met(y, oof)
    D.prov.write("fig21_progression",
                 title="From analytical proxy to hybrid surrogate",
                 tier=S.TIER_A, act=ACT, basis=BASIS,
                 claim=f"The analytical Daganzo proxy is badly biased "
                       f"({m1[0]:.0f}% MAPE, {m1[1]:+.0f}% bias); a single "
                       f"global scale factor alpha = {alpha:.3f} removes the "
                       f"level error ({m2[0]:.0f}% MAPE) and the learned "
                       f"residual brings out-of-fold error to {m3[0]:.1f}% MAPE "
                       f"at R2 = {m3[2]:.3f}.",
                 caveats="Panel (c) is out-of-fold under GroupKFold by postal "
                         "code, so perturbed samples of one area never span the "
                         "train/test split. Panels (a) and (b) need no split.")


# ---------------------------------------------------------------- fig22
def fig22_pool():
    pool = D.load_training_pool()
    print(f"  {len(pool)} samples, {pool.plz.nunique()} areas, "
          f"{pool.provider.nunique()} providers, "
          f"{int(pool.is_baseline.sum())} baseline")

    for style in S.styles():
        S.apply(style)
        fig, axes = plt.subplots(1, 3,
                                 figsize=S.figsize(style, (13.0, 4.0),
                                                   (16.0, 5.2)))
        # (a) samples per provider, split baseline vs perturbed
        g = (pool.groupby(["provider", "is_baseline"]).size()
             .unstack(fill_value=0).reindex(D.PROVIDERS))
        x = np.arange(len(g))
        axes[0].bar(x, g[True], 0.6, label="Baseline day",
                    color=D.PALETTE["baseline"])
        axes[0].bar(x, g[False], 0.6, bottom=g[True], label="Perturbed",
                    color=D.PALETTE["stage3"])
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(g.index, rotation=30 if style == "slides" else 0)
        axes[0].set_ylabel("VROOM samples")
        axes[0].set_title("(a) Pool composition")
        axes[0].grid(alpha=0.25, axis="y")
        axes[0].legend(framealpha=0.9)

        # (b) cost distribution
        axes[1].hist(pool.actual_cost_eur, bins=40,
                     color=D.PALETTE["stage2"], alpha=0.9)
        axes[1].set_xlabel("VROOM cost per sample [€]")
        axes[1].set_ylabel("Samples")
        axes[1].set_title("(b) Target distribution")
        axes[1].grid(alpha=0.25, axis="y")

        # (c) aggregation level -- the batching depth the pool covers
        ak = pool.groupby("agg_k").size()
        bars = axes[2].bar(ak.index.astype(str), ak.values, 0.6,
                           color=D.PALETTE["accent2"])
        axes[2].set_xlabel("Days aggregated into one tour ($k$)")
        axes[2].set_ylabel("Samples")
        axes[2].set_title("(c) Batching depth covered")
        axes[2].grid(alpha=0.25, axis="y")
        P.bar_labels(axes[2], bars, ak.values, "{:.0f}", style=style)

        P.footnote(fig, f"{len(pool)} VROOM solves over "
                        f"{pool.plz.nunique()} areas and "
                        f"{pool.provider.nunique()} providers; all returned OK. "
                        f"Cost {pool.actual_cost_eur.min():.0f}–"
                        f"{pool.actual_cost_eur.max():.0f} €.", style)
        fig.tight_layout(rect=[0, 0.07, 1, 1])
        S.save(fig, "fig22_pool", style, S.TIER_B)

    D.prov.write("fig22_pool", title="Training-pool composition",
                 tier=S.TIER_B, act=ACT, basis=BASIS,
                 claim=f"The surrogate is fit on {len(pool)} real VROOM solves "
                       f"spanning {pool.plz.nunique()} areas, all seven "
                       f"providers and batching depths k = "
                       f"{sorted(pool.agg_k.unique())}; "
                       f"{int(pool.is_baseline.sum())} are unperturbed baseline "
                       f"days and the rest are demand perturbations that widen "
                       f"the feature envelope.")


# ---------------------------------------------------------------- fig23
def fig23_alpha():
    a = D.load_alpha_sensitivity()
    label = {"no_alpha": r"No scaling ($\alpha = 1$)",
             "global_median": "One global $\\alpha$\n(production)",
             "per_lsp_median": r"One $\alpha$ per provider"}
    a["label"] = a.variant.map(label).fillna(a.variant)
    print(a[["variant", "oof_mape_pct", "r2"]].round(3).to_string(index=False))

    for style in S.styles():
        S.apply(style)
        fig, ax = plt.subplots(figsize=S.figsize(style, (7.0, 4.4), (10.0, 5.4)))
        x = np.arange(len(a))
        colors = [D.PALETTE["stage3"] if v == "global_median"
                  else D.PALETTE["baseline"] for v in a.variant]
        bars = ax.bar(x, a.oof_mape_pct, 0.55, yerr=a.oof_mape_std,
                      capsize=5, color=colors, edgecolor="#333333",
                      linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(a.label)
        ax.set_ylabel("Out-of-fold MAPE [%]")
        ax.set_title(r"Does the Daganzo scale factor need to be per provider?")
        ax.grid(alpha=0.25, axis="y")
        for xi, row in zip(x, a.itertuples()):
            ax.text(xi, row.oof_mape_pct + row.oof_mape_std + 0.12,
                    f"{row.oof_mape_pct:.2f}%", ha="center",
                    fontsize=9 if style == "paper" else 13)
        P.footnote(fig, "Error bars are the standard deviation of the per-sample "
                        "absolute percentage error. Per-provider scaling is "
                        "worse, not better: the learned residual already "
                        "absorbs provider-level scale.", style)
        fig.tight_layout(rect=[0, 0.07, 1, 1])
        S.save(fig, "fig23_alpha", style, S.TIER_B)

    gm = a[a.variant == "global_median"].iloc[0]
    pl = a[a.variant == "per_lsp_median"].iloc[0]
    D.prov.write("fig23_alpha", title="Global vs per-provider alpha",
                 tier=S.TIER_B, act=ACT, basis=BASIS,
                 claim=f"A single global alpha gives {gm.oof_mape_pct:.2f}% "
                       f"out-of-fold MAPE against {pl.oof_mape_pct:.2f}% for "
                       f"per-provider alphas, so the extra parameters do not "
                       f"pay: the LightGBM residual already absorbs "
                       f"provider-level scale differences.",
                 caveats="Answers a reviewer question; the three variants differ "
                         "only in how the Daganzo base is scaled.")


# ---------------------------------------------------------------- fig24
def fig24_determinism():
    cd = D.load_cd_restart_spread()
    print(f"  {len(cd)} provider-cells, max spread "
          f"{cd.spread_pct.abs().max():.2e}% "
          f"({cd.spread_eur.abs().max():.2e} €), restarts="
          f"{sorted(cd.n_restarts.unique())}")

    for style in S.styles():
        S.apply(style)
        fig, ax = plt.subplots(figsize=S.figsize(style, (7.4, 4.4), (10.5, 5.4)))
        vals = cd.spread_pct.abs().replace(0, 1e-18).values
        ax.hist(np.log10(vals), bins=18, color=D.PALETTE["stage3"], alpha=0.9)
        ax.axvline(np.log10(1e-12), color=D.PALETTE["accent"],
                   linestyle="--", linewidth=S.scale(style, 1.6, 1.4),
                   label=r"$10^{-12}\%$ reference")
        ax.set_xlabel(r"$\log_{10}$ of relative cost spread across restarts [%]")
        ax.set_ylabel("Provider-cells")
        ax.set_title("Coordinate descent converges to the same point")
        ax.grid(alpha=0.25, axis="y")
        ax.legend(framealpha=0.9)
        P.footnote(fig, f"{int(cd.n_restarts.max())} random restarts per "
                        f"provider-cell over {len(cd)} cells. The largest "
                        f"spread is {cd.spread_eur.abs().max():.1e} € — "
                        f"floating-point noise, not a different solution.",
                   style)
        fig.tight_layout(rect=[0, 0.07, 1, 1])
        S.save(fig, "fig24_determinism", style, S.TIER_B)

    D.prov.write("fig24_determinism", title="Coordinate-descent restart spread",
                 tier=S.TIER_B, act=ACT, basis="Stage-3 restart analysis",
                 claim=f"Across {len(cd)} provider-cells and "
                       f"{int(cd.n_restarts.max())} restarts each, the spread "
                       f"between best and worst converged cost never exceeds "
                       f"{cd.spread_eur.abs().max():.1e} € "
                       f"({cd.spread_pct.abs().max():.1e}%). The coordinate "
                       f"descent is effectively deterministic.",
                 caveats="Determinism is not global optimality: this shows the "
                         "restarts agree, i.e. a Nash point of the shared hub "
                         "objective, not that it is the global optimum.")


# ---------------------------------------------------------------- fig25
def fig25_importance():
    _pool, _y, _dag, _alpha, _oof_pred, gains = _oof()
    top = gains.sort_values(ascending=False).head(15).iloc[::-1]
    share = gains.sort_values(ascending=False).head(15).sum() / gains.sum() * 100
    print(f"  top-15 features carry {share:.1f}% of total split gain")
    print("  top 5: " + ", ".join(
        f"{k} ({v / gains.sum() * 100:.1f}%)"
        for k, v in gains.sort_values(ascending=False).head(5).items()))

    for style in S.styles():
        S.apply(style)
        fig, ax = plt.subplots(figsize=S.figsize(style, (7.6, 5.4), (10.5, 6.4)))
        y = np.arange(len(top))
        ax.barh(y, top.values / gains.sum() * 100, 0.65,
                color=D.PALETTE["stage2"])
        ax.set_yticks(y)
        ax.set_yticklabels(top.index)
        ax.set_xlabel("Share of total LightGBM split gain [%]")
        ax.set_title("What the learned residual actually uses")
        ax.grid(alpha=0.25, axis="x")
        P.footnote(fig, f"Gain averaged over the five out-of-fold models. "
                        f"These 15 of {len(gains)} features carry "
                        f"{share:.0f}% of total gain.", style)
        fig.tight_layout(rect=[0, 0.06, 1, 1])
        S.save(fig, "fig25_importance", style, S.TIER_B)

    D.prov.write("fig25_importance", title="Residual-model feature importance",
                 tier=S.TIER_B, act=ACT, basis=BASIS,
                 claim=f"The residual is concentrated: 15 of {len(gains)} "
                       f"engineered features carry {share:.0f}% of the split "
                       f"gain, led by {gains.idxmax()}.",
                 caveats="Split gain is a within-model diagnostic, not a causal "
                         "or permutation-based importance; correlated features "
                         "share credit arbitrarily.")


def main():
    for fn in (fig21_progression, fig22_pool, fig23_alpha, fig24_determinism,
               fig25_importance):
        print(f"\n=== {fn.__name__} ===")
        fn()


if __name__ == "__main__":
    main()
