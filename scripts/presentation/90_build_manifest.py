"""Assemble MANIFEST.md from the provenance sidecars the figure scripts wrote.

The manifest is generated, never hand-maintained: every entry comes from a
_prov/<figure>.json written by the script that actually produced the figure, so
the recorded sources are the files that were really read, with their sizes and
modification times.

Also verifies that every sidecar has its rendered files on disk and that no
rendered file lacks a sidecar, appends the exclusion list -- the analyses
deliberately left out, with the reason each one cannot be put on the Stage-3
basis -- and records the DECK PROVENANCE: which grid the rebuilt talk decks
were generated from, at which repository commit, with the md5 of every grid
table they read. That last block is what makes a deck falsifiable after the
fact: an md5 that no longer matches the file on disk says the deck predates
the grid it claims to be built on.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _data as D
import _style as S

OUT = S.OUT_ROOT
MANIFEST = OUT / "MANIFEST.md"
DECK_DIR = Path(
    r"C:/Users/bienzeisler/Documents/Präsentationen/EWGT/2026")

ACT_ORDER = [
    "1 - Setting",
    "2 - Surrogate",
    "3 - Core results",
    "4 - Spatial",
    "5 - Operational",
    "6 - Validation",
    "7 - Where it pays off",
    "8 - Adopted paper figures",
]

EXCLUDED = [
    dict(
        what="fig_value_of_optimization, fig_value_of_path2_pipeline, "
             "fig_sweetspot_lambda",
        where="results/paper_ewgt_2026/, scripts/figures/",
        why="Built on results/supplementary/penalty_sweep/sched_cost_cache.npz "
            "(2026-05-28). That cache holds per-PLZ *unbundled* schedule costs, "
            "which is the selection path only: the express component is bundled "
            "per hub and cannot be split per area. Using it for savings claims "
            "would mix a bundled quantity into a per-area one. It was not "
            "recomputed at Stage 3.",
    ),
    dict(
        what="Per-area saving maps and threshold analyses on VROOM-actual costs "
             "(figT1/T2/T8, figU1-U5, fig_BE1-BE9, figS3/S7)",
        where="results/_archive/final_optimization/, "
              "results/paper_outputs_2026_05_30/09_region_analysis/",
        why="Their source table results/supplementary/sensitivity_break_even/"
            "tab_sensitivity_master_plz.csv carries per-area VROOM baseline "
            "costs on a *different demand allocation*: Amazon 30159 has 3 562 "
            "parcels there against 9 019 in Stage 3, because cluster merge "
            "forwarding reassigns demand. Pairing that baseline with Stage-3 "
            "batched routes would produce invented savings. Act 7 answers the "
            "same questions on the ML basis instead, where baseline and "
            "optimised cost come from one reconciled computation.",
    ),
    dict(
        what="fig_M02_cluster_bias, fig_MLA3_cluster_bias_choropleth",
        where="results/supplementary/paper_maps_final_v2/, "
              "results/paper_outputs_2026_05_30/12_ml_quality_per_cluster/",
        why="Carry the known PLZ merge artefact: cluster 30159 shows -30.7% "
            "predicted against +9.6% actual saving because the pre-merge "
            "geometry split demand that Stage 3 keeps together. The bias these "
            "maps display is a data-preparation artefact, not model error.",
    ),
    dict(
        what="Everything derived from tab_vroom_path2.csv",
        where="results/paper_results_2026_05_30/07_validation/",
        why="Every row of that file is an error: "
            "ERR:TypeError: solve_single_plz() got an unexpected keyword "
            "argument 'seed_key'. No usable routing result in it.",
    ),
    dict(
        what="tab_vroom_balanced_BUGGY_backup.csv, "
             "tab_vroom_balanced_prebudgetfix.csv and their figures",
        where="results/paper_outputs_2026_05_30/07_validation/",
        why="Superseded pre-bugfix validation runs, retained only as history. "
            "The Stage-3 validation in results/revision_2026_07/validation/ "
            "replaces them.",
    ),
    dict(
        what="_fig_maps_per_share_P04.py",
        where="scripts/figures/",
        why="Selects PENALTY = 0.5 while its title and output filename both "
            "claim P = 0.4. Rewritten from scratch as fig41-fig43 rather than "
            "patched, with the penalty flowing from one constant into both the "
            "data selection and the label.",
    ),
    dict(
        what="fig_PF2_saving_fleet_heatmaps, fig_PF3_sweetspot, "
             "fig_SM_mix_pct_init_vs_balanced (+_P5)",
        where="results/paper_ewgt_2026/",
        why="Superseded rather than wrong: fig31/32/33 carry the same content on "
            "the Stage-3 grid, and fig35 covers the frequency mix across all "
            "eight penalty levels instead of two.",
    ),
    dict(
        what="Per-area saving choropleth at theta < 1",
        where="not produced",
        why="Only theta = 1 has an exactly zero express component, which is what "
            "makes the per-area cost decomposition exact. Below theta = 1 the "
            "hub-bundled express cost would have to be attributed to individual "
            "areas, which is not defined. Act 7 is therefore theta = 1 "
            "throughout.",
    ),
]


# The decks this manifest vouches for. Each is generated by the script named,
# from the grid `_data.REV` points at; the sizes and times come off disk.
DECKS = [
    ("EWGT_26_Bienzeisler_TBC_deck_rev2026-08.pptx",
     "scripts/presentation/91_build_pptx.py",
     "v1 grammar: kicker, real bulleted lists, native-shape schematics"),
    ("EWGT_26_Bienzeisler_TBC_house_deck_rev2026-08.pptx",
     "scripts/presentation/94_build_house_deck.py",
     "house grammar: two-line titles, progressive builds, icon badges"),
    ("EWGT_26_Bienzeisler_new_plus_explainers_rev2026-08.pptx",
     "scripts/presentation/96_explainers.py",
     "the author's working deck plus the question-by-question backup"),
]

# The grid tables the revision slides read. Anything a deck asserts against is
# in here, so the md5 list is the deck's data fingerprint.
GRID_TABLES = [
    "tab_costs_v2.csv",
    "tab_wait_v2.csv",
    "tab_fleet_per_hub_v2.csv",
    "_tab_chosen_v2.csv",
    "tables/tab_headline_theta1_v2.csv",
    "tables/tab_pstar_knees_v2.csv",
    "tables/tab_grid_full_v2.csv",
    "tables/tab_fleet_diagnostics_v2.csv",
    "_peek/results_overview*.csv",
    "_peek/discount_scenarios*.csv",
    "tables/tab_per_cell_costs_v2.csv",
]


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _deck_provenance() -> list[str]:
    """The deck section of the manifest: grid, commit, table fingerprints."""
    L = ["## Talk decks", ""]
    L.append(f"Revision grid: `{D.REV.name}` (schema {D.SCHEMA}), read via "
             f"`$PRES_REV_DIR` / `--rev-dir`.")
    L.append("")
    L.append(f"VROOM validation: `{D.VAL.parent.name}/validation` "
             f"(schema {D.val_schema()}), chosen SEPARATELY via "
             f"`$PRES_VAL_DIR` because a validation run is much longer than "
             f"the grid it validates and the two are not in step. Realised "
             f"saving available: "
             f"`{D.vroom_actual_baseline_available()}` -- when False, the "
             f"validation figures show predicted against actual COST on the "
             f"same tours, never a saving formed from an actual numerator and "
             f"a predicted denominator.")
    L.append("")
    L.append(f"Repository commit at manifest time: `{D._git_head()}`.")
    L.append("")
    L.append("The decks are written OUTSIDE the repository, into the author's "
             "presentation folder, and are copy-only: every builder routes "
             "its output through `_outguard.resolve()`, which refuses to "
             "overwrite an existing file without `--overwrite`. The originals "
             "were copied to `_originals_2026-08-27/` before the first "
             "rebuild.")
    L.append("")
    L.append("| Deck | Built by | Grammar | Size | Written |")
    L.append("|---|---|---|---|---|")
    for name, script, grammar in DECKS:
        f = DECK_DIR / name
        if f.exists():
            st = f.stat()
            size = f"{st.st_size / 1048576:.1f} MB"
            when = dt.datetime.fromtimestamp(st.st_mtime).strftime(
                "%Y-%m-%d %H:%M")
        else:
            size, when = "not built", "-"
        L.append(f"| `{name}` | `{script}` | {grammar} | {size} | {when} |")
    L.append("")
    L.append("Grid tables the revision slides read, with their md5 at "
             "manifest time:")
    L.append("")
    L.append("| Table | Bytes | md5 |")
    L.append("|---|---|---|")
    for rel in GRID_TABLES:
        # a pattern, because the peek files carry their grid version in the
        # name (results_overview_v5.csv, results_overview_v6.csv, ...)
        hits = sorted(D.REV.glob(rel)) if "*" in rel else (
            [D.REV / rel] if (D.REV / rel).exists() else [])
        if not hits:
            L.append(f"| `{rel}` | missing | - |")
            continue
        for f in hits:
            rel_name = f.relative_to(D.REV).as_posix()
            size = f"{f.stat().st_size:,}".replace(",", " ")
            L.append(f"| `{rel_name}` | {size} | `{_md5(f)}` |")
    L.append("")
    L.append("Pinned to the submission grid on purpose, whatever `PRES_REV_DIR` "
             "says: `load_alpha_sensitivity()`, `load_cd_restart_spread()`, "
             "`load_per_plz()` and the VROOM-validation loaders. None of those "
             "analyses was re-run for the revision, and the revision's own "
             "validation is a different schema still being produced.")
    L.append("")
    return L


def _load_sidecars() -> list[dict]:
    if not D.PROV_DIR.exists():
        raise SystemExit(f"no provenance directory at {D.PROV_DIR}; "
                         f"run the act scripts first")
    out = []
    for p in sorted(D.PROV_DIR.glob("*.json")):
        out.append(json.loads(p.read_text()))
    return out


def _verify(cards: list[dict]) -> list[str]:
    """Cross-check sidecars against the rendered files. Returns warnings."""
    warn = []
    named = {c["name"] for c in cards}

    for c in cards:
        paper = OUT / "paper" / f"{c['name']}.pdf"
        if not paper.exists():
            warn.append(f"{c['name']}: no paper PDF at {paper.name}")
        slide = (OUT / "slides" / c["tier"] / f"{c['name']}.png")
        if not slide.exists():
            other = OUT / "slides" / (
                S.TIER_B if c["tier"] == S.TIER_A else S.TIER_A
            ) / f"{c['name']}.png"
            if other.exists():
                warn.append(f"{c['name']}: slide PNG sits in "
                            f"{other.parent.name}, manifest says {c['tier']}")
            elif c["act"] != "8 - Adopted paper figures":
                warn.append(f"{c['name']}: no slide PNG")

    for f in sorted((OUT / "paper").glob("*.pdf")):
        if f.stem not in named:
            warn.append(f"{f.name}: rendered but has no provenance sidecar")
    for tier in (S.TIER_A, S.TIER_B):
        d = OUT / "slides" / tier
        if not d.exists():
            continue
        for f in sorted(d.glob("*.png")):
            if f.stem not in named:
                warn.append(f"slides/{tier}/{f.name}: rendered but has no "
                            f"provenance sidecar")
    return warn


def main():
    cards = _load_sidecars()
    warn = _verify(cards)

    by_act: dict[str, list[dict]] = {}
    for c in cards:
        by_act.setdefault(c["act"], []).append(c)
    acts = [a for a in ACT_ORDER if a in by_act]
    acts += [a for a in sorted(by_act) if a not in acts]

    n_a = sum(1 for c in cards if c["tier"] == S.TIER_A)
    n_b = len(cards) - n_a
    commits = {c.get("git_commit", "unknown") for c in cards}

    L: list[str] = []
    L.append("# Presentation figure set — Region Hannover time-based "
             "consolidation")
    L.append("")
    L.append(f"Generated {dt.datetime.now():%Y-%m-%d %H:%M} from "
             f"`scripts/presentation/`. "
             f"{len(cards)} figures — {n_a} tier A (core narrative), "
             f"{n_b} tier B (backup / deep dive).")
    L.append("")
    L.append("## Layout")
    L.append("")
    L.append("```")
    L.append("paper/          serif, PDF + PNG at 300 dpi, column width")
    L.append("slides/tierA/   sans, 16:9, PNG at 200 dpi — the deck's spine")
    L.append("slides/tierB/   sans, 16:9, PNG at 200 dpi — backup material")
    L.append("_prov/          per-figure provenance sidecars (JSON)")
    L.append("```")
    L.append("")
    L.append("Every figure is rendered in both styles from a single code path "
             "(`_style.apply()`), so the slide and paper variants cannot drift "
             "apart. Two adopted paper figures are the documented exception.")
    L.append("")
    L.append(f"Repository state: commit {', '.join(sorted(commits))}.")
    L.append("")

    L.append("## Verification built into the figure scripts")
    L.append("")
    L.append("These run on every render and abort the figure if they fail:")
    L.append("")
    L.append("| Check | Where | Guards against |")
    L.append("|---|---|---|")
    L.append("| Per-area cost sums to the provider Stage-3 total | "
             "`00_recompute_per_plz_costs.py` gate A | a per-area "
             "decomposition that silently drops or double-counts cost |")
    L.append("| Per-area baseline sums to the pinned 1 909 747.75 € | "
             "gate B | a baseline reference drifting from the paper's |")
    L.append("| Express component is exactly 0 at θ = 1 | "
             "`00_recompute` premise check | mixing hub-bundled express into "
             "per-area figures |")
    L.append("| `schedule_size_init == schedule_size_system_smoothed` | "
             "`_data.load_chosen_stage3()` | mislabelling every frequency map "
             "if smoothing ever changed frequencies |")
    L.append("| Observed delivery frequencies ⊆ {2,…,6} | "
             "`_data.load_chosen_stage3()` | inventing a 1 day/wk class that "
             "`MAX_HOLDING_DAYS = 3` forbids |")
    L.append("| Baseline fleet CV == 0.135 | `fig33_fleet_grid` | the Stage-2 "
             "fleet reference drifting from the submitted value |")
    L.append("| Recomputed α == production α (1.343) | `fig21_progression` | "
             "plotting a progression for a different model than ships |")
    L.append("| Every polygon resolves to a model unit | `40_act4_maps._view` | "
             "areas silently dropped from a choropleth |")
    L.append("| Weekday strings parse to a non-zero profile | "
             "`fig55_pattern_clock` | an all-zero figure from a format change |")
    L.append("| 39 admissible weekly patterns | `00_recompute` | the "
             "holding-days invariant changing unnoticed |")
    L.append("")

    L.append("## Figures")
    L.append("")
    for act in acts:
        L.append(f"### Act {act}")
        L.append("")
        for c in sorted(by_act[act], key=lambda z: z["name"]):
            L.append(f"#### `{c['name']}` — {c['title']}")
            L.append("")
            L.append(f"- **Tier:** {c['tier']}")
            L.append(f"- **Says:** {c['claim']}")
            if c.get("caveats"):
                L.append(f"- **Read with care:** {c['caveats']}")
            L.append(f"- **Data basis:** {c['basis']}")
            L.append(f"- **Script:** `{c['script']}`")
            L.append(f"- **Rendered:** {c['generated']}")
            L.append("- **Sources:**")
            for src, meta in sorted(c["sources"].items()):
                if meta.get("mtime") == "n/a":
                    L.append(f"  - `{src}`")
                else:
                    # Space as thousands separator, applied to the number only:
                    # a blanket comma replacement would eat sentence commas.
                    size = f"{meta['bytes']:,}".replace(",", " ")
                    L.append(f"  - `{src}` — {size} bytes, "
                             f"modified {meta['mtime']}")
            L.append("")

    L += _deck_provenance()

    L.append("## Deliberately excluded")
    L.append("")
    L.append("These analyses exist in the repository but are **not** in this "
             "folder, because they cannot be put on the Stage-3 basis without "
             "changing what they mean. Each entry says why.")
    L.append("")
    for e in EXCLUDED:
        L.append(f"### {e['what']}")
        L.append("")
        L.append(f"*Location:* `{e['where']}`")
        L.append("")
        L.append(e["why"])
        L.append("")

    if warn:
        L.append("## Manifest self-check warnings")
        L.append("")
        for w in warn:
            L.append(f"- {w}")
        L.append("")

    MANIFEST.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {MANIFEST.relative_to(D.ROOT)} "
          f"({len(cards)} figures, {n_a} tier A / {n_b} tier B)")
    if warn:
        print(f"\n{len(warn)} self-check warning(s):")
        for w in warn:
            print(f"  - {w}")
    else:
        print("self-check clean: every figure has a sidecar and every sidecar "
              "has its files")


if __name__ == "__main__":
    main()
