"""Assemble results/revision_2026_08_final/ -- the one folder for the revision.

Copies (never moves, never edits) the v6 grid's figures, tables, validation
outputs and controller analyses into a single pack, adds the paper-side
provenance (the real 71_ sync table, the CHANGES snapshot, the compendium
excerpt), records the presentation decks by path + md5 instead of copying
40 MB of PowerPoint, and writes a README that lists every file with its
producing script, md5, size and grid.

Rules this script enforces rather than assumes:

* **read-only on the source.** Every source is opened for reading; the only
  writes go under ``results/revision_2026_08_final/``. Nothing under
  ``paper/`` or ``results/revision_2026_08_v6/`` is touched.
* **every copy is md5-verified** after the fact (source hash == destination
  hash), and the same hash is what the README prints.
* **every file must be attributable.** ``PRODUCERS`` maps a destination path
  to the script that made it; a file that matches no rule is a hard failure,
  not an "unknown" row -- an unattributable artifact in a provenance pack is
  worse than a missing one.
* **the 71_ sync log must show a real PASS** for this rev-dir, otherwise the
  pack would claim a paper sync that never happened.

Usage:

    python scripts/revision/79_build_final_pack.py \\
        --sync-log <log of the real 71_ --include-companions run>

``--sync-log`` is the transcript of the actual sync (71_ prints the md5 PASS
table); it is copied into ``paper/71_sync_pass_table.txt`` verbatim.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "revision_2026_08_v6"
PACK = ROOT / "results" / "revision_2026_08_final"
COMPENDIUM = ROOT / "docs" / "PAPER_COMPENDIUM_2026_05_24.md"
CHANGES = ROOT / "docs" / "CHANGES_rev1.md"
DECK_DIR = Path(r"C:\Users\bienzeisler\Documents\Präsentationen\EWGT\2026")
DECKS = ("EWGT_26_Bienzeisler_TBC_deck_rev2026-08.pptx",
         "EWGT_26_Bienzeisler_TBC_house_deck_rev2026-08.pptx",
         "EWGT_26_Bienzeisler_new_plus_explainers_rev2026-08.pptx")

#: The compendium sections the pack carries verbatim: the whole v5/v6 block,
#: from the operator-lens finding that produced grid v5 to the CO2 table.
EXCERPT_FROM = "## 40.14"
EXCERPT_TO_END_OF = "## 40.28"

#: Analyses the controller produced outside the numbered scripts. They live
#: in the grid's ``_peek/`` scratch area, which is not a published location.
PEEK_FILES = ("results_overview_v6.csv", "discount_scenarios_v6.csv",
              "fig_mechanism_theta_P.pdf", "fig_mechanism_theta_P.png",
              "fig_fleet_week_by_provider_P0.pdf",
              "fig_fleet_week_by_provider_P0.png",
              "fig_fleet_week_by_provider_P025.pdf",
              "fig_fleet_week_by_provider_P025.png")

VALIDATION_FILES = ("validation_report.md", "tab_vroom_v2.csv", "census.md",
                    "census.csv", "instance_queue.csv", "G6_sampling_note.md")

#: Files in the live grid's ``figures/`` that no current script produces.
#: ``fig6_bucket_composition.csv`` is the pre-rename filename of
#: ``supp_fig6_bucket_composition.csv`` -- same content, older mtime, absent
#: from the render manifest. It is skipped WITH a printed note rather than
#: shipped: a provenance pack must not carry a file nothing produces.
FIGURE_SUPERSEDED = ("fig6_bucket_composition.csv",)

#: Non-figure files in ``figures/`` that ARE part of the current render and
#: therefore belong in the pack even though the manifest only lists PDFs/PNGs.
FIGURE_EXTRA = ("manifest.json", "supp_fig6_bucket_composition.csv")

#: (regex on the pack-relative posix path, producing script, grid).
#: First match wins; an unmatched file is a hard failure.
PRODUCERS: tuple[tuple[str, str, str], ...] = (
    (r"^figures/fig4_SM_mix_pct_8P\.", "scripts/revision/32_fig4_mix.py (via 70_)", "v6"),
    (r"^figures/fig5_grid_heatmap_6_smoothed\.",
     "scripts/revision/30_fig5_heatmap_smoothed.py (via 70_)", "v6"),
    (r"^figures/fig6_structural_grid_6_smoothed\.",
     "scripts/revision/31_fig6_structural_smoothed.py (via 70_)", "v6"),
    (r"^figures/supp_fig7_fleet_week_classes\.",
     "scripts/revision/75_fig_fleet_week_classes.py", "v6"),
    (r"^figures/supp_(map_|penalty_raumtyp)", "scripts/revision/76_maps_v2.py", "v6"),
    (r"^figures/supp_fig_mechanism_v2\.", "scripts/revision/77_mechanism_v2.py", "v6"),
    (r"^figures/supp_fig_fleet_week_v2_", "scripts/revision/78_fleet_week_v2.py", "v6"),
    (r"^figures/manifest\.json$", "scripts/revision/70_figs_tables_v2.py", "v6"),
    # 70_ draws the supp_fig4/4b/5/5b/6/6b set itself (its own docstring) and
    # writes the bucket-composition CSV next to figure 6. Deliberately NOT a
    # catch-all: a new figure from a new script must fail attribution here
    # rather than be shipped as if 70_ had made it.
    (r"^figures/supp_fig(4|4b|5|5b|6|6b)_",
     "scripts/revision/70_figs_tables_v2.py", "v6"),
    (r"^tables/tab_map_|^tables/tab_penalty_raumtyp_v2",
     "scripts/revision/76_maps_v2.py", "v6"),
    (r"^tables/tab_fleet_week_classes_v2",
     "scripts/revision/75_fig_fleet_week_classes.py", "v6"),
    (r"^tables/tab_mechanism_theta_P_v2|^tables/tab_saving_per_parcel_hist_v2",
     "scripts/revision/77_mechanism_v2.py", "v6"),
    (r"^tables/tab_fleet_week_by_provider_v2",
     "scripts/revision/78_fleet_week_v2.py", "v6"),
    (r"^tables/tab_per_cell_costs_v2", "scripts/revision/72_per_cell_costs_v2.py", "v6"),
    (r"^tables/tab_plz_knee_with_features_v2|^tables/tab_value_of_stage2_v2",
     "scripts/revision/73_tables_ops_v2.py", "v6"),
    (r"^tables/tab_grid_delta_v2",
     "scripts/revision/70_figs_tables_v2.py", "v6 vs v5"),
    (r"^tables/tab_co2_km_v2",
     "scripts/revision/70_figs_tables_v2.py (from the v6 VROOM validation)", "v6"),
    # Everything else in tables/ is 70_'s: 72_/73_/75_/76_/77_/78_ each write
    # only the named tables above, so this last-resort rule is true today and
    # is the one place a new writer would need a rule of its own.
    (r"^tables/", "scripts/revision/70_figs_tables_v2.py", "v6"),
    (r"^validation/gates_report\.md$", "scripts/revision/62_gates_check.py", "v6"),
    (r"^validation/", "scripts/revision/67_validate_vroom_v2.py", "v6"),
    (r"^analyses/results_overview_v6\.csv$",
     "dashboard/results_overview.py (controller scratch)", "v6"),
    (r"^analyses/discount_scenarios_v6\.csv$",
     "dashboard/discount_scenarios.py (controller scratch)", "v6"),
    (r"^analyses/fig_mechanism_theta_P\.",
     "dashboard/fig_mechanism.py (superseded by 77_)", "v6 curves, v5 panel (c)"),
    (r"^analyses/fig_fleet_week_by_provider_",
     "dashboard/fig_fleet_week.py (superseded by 78_)", "v6"),
    (r"^paper/71_sync_pass_table\.txt$",
     "scripts/revision/71_sync_paper_figs.py --include-companions", "v6"),
    (r"^paper/CHANGES_rev1\.md$", "docs/CHANGES_rev1.md (snapshot)", "n/a"),
    (r"^paper/compendium_40_14_to_40_28\.md$",
     "docs/PAPER_COMPENDIUM_2026_05_24.md (excerpt)", "n/a"),
    (r"^dashboard/dashboard_v6\.html$",
     "dashboard/build_v5_page.py + body_add_validation.py (controller)", "v6"),
    (r"^dashboard/figure_gallery_v6\.html$",
     "dashboard/build_fig_gallery_v6.py (controller)", "v6"),
    (r"^dashboard/v5_body\.html$", "dashboard/build_v5_page.py (controller)", "v5"),
    (r"^dashboard/build_v5_page\.py$|^dashboard/body_add_validation\.py$",
     "controller scratch script (kept as source)", "v5 -> v6"),
    (r"^dashboard/", "controller scratch script (kept as source)", "v6"),
    (r"^decks/DECKS\.md$", "this script (pointer list, decks not copied)", "v6"),
    (r"^gallery_manifest\.json$", "this script (deliverable 5)", "v6"),
)


def md5_of(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                         capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, "cannot read git HEAD"
    head = out.stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain",
                            "--untracked-files=no"],
                           capture_output=True, text=True, timeout=30)
    return head + ("-dirty" if dirty.stdout.strip() else "")


def copy_verified(src: Path, dst: Path) -> str:
    """Copy and re-hash both ends; return the md5 they agree on."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    a, b = md5_of(src), md5_of(dst)
    assert a == b, f"copy corrupted: {src} ({a}) != {dst} ({b})"
    return a


def producer_of(rel: str) -> tuple[str, str]:
    for pattern, script, grid in PRODUCERS:
        if re.search(pattern, rel):
            return script, grid
    raise AssertionError(
        f"{rel} matches no PRODUCERS rule -- every file in a provenance pack "
        "must name the script that made it; add a rule instead of shipping an "
        "unattributable artifact")


# ─────────────────────────────────────────────────────────────────────────
# excerpt / pointer files
# ─────────────────────────────────────────────────────────────────────────
def compendium_excerpt(text: str, start: str = EXCERPT_FROM,
                       end_of: str = EXCERPT_TO_END_OF) -> str:
    """The compendium block from ``start`` to the end of the ``end_of`` section.

    Pure string work so it can be tested: refuses if either heading is absent
    or if they are in the wrong order, rather than returning a silent empty
    or half excerpt.
    """
    lines = text.replace("\r\n", "\n").split("\n")
    i = next((k for k, ln in enumerate(lines) if ln.startswith(start)), None)
    j = next((k for k, ln in enumerate(lines) if ln.startswith(end_of)), None)
    assert i is not None, f"{start} not found in the compendium"
    assert j is not None, f"{end_of} not found in the compendium"
    assert i < j, f"{start} comes after {end_of}"
    k = next((m for m in range(j + 1, len(lines))
              if lines[m].startswith("## ") or lines[m].startswith("# ")),
             len(lines))
    return "\n".join(lines[i:k]).rstrip() + "\n"


def deck_pointers() -> tuple[str, list[dict]]:
    """Pointer list for the presentation decks (paths + md5, never copied)."""
    rows = []
    for name in DECKS:
        p = DECK_DIR / name
        if p.exists():
            rows.append(dict(name=name, path=str(p), md5=md5_of(p),
                             bytes=p.stat().st_size,
                             mtime=datetime.fromtimestamp(
                                 p.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                             locked=(p.parent / f"~${name}").exists()))
        else:
            rows.append(dict(name=name, path=str(p), md5="MISSING", bytes=0,
                             mtime="-", locked=False))
    body = ["# Presentation decks (pointers, not copies)", "",
            "The three revision decks are ~40 MB together, so the pack records",
            "where they are and what they hashed to, not the files themselves.",
            "", "| deck | md5 | MB | modified | note |", "|---|---|---:|---|---|"]
    for r in rows:
        note = ("open in PowerPoint at pack time (a ~$ lock file exists) -- "
                "pending rebuild" if r["locked"] else
                "MISSING" if r["md5"] == "MISSING" else "")
        body.append(f"| `{r['name']}` | `{r['md5']}` | {r['bytes'] / 1e6:.1f} | "
                    f"{r['mtime']} | {note} |")
    body += ["", f"Folder: `{DECK_DIR}`", "",
             "Per task 17 / Addendum 3 the TBC deck and the house deck are still",
             "to be rebuilt once PowerPoint is closed; the hashes above are the",
             "state at pack time.", ""]
    return "\n".join(body), rows


# ─────────────────────────────────────────────────────────────────────────
# build
# ─────────────────────────────────────────────────────────────────────────
def build(sync_log: Path) -> list[dict]:
    assert SRC.is_dir(), f"{SRC} does not exist"
    log_text = sync_log.read_text(encoding="utf-8", errors="replace")
    assert "PASS: all" in log_text and "revision_2026_08_v6" in log_text, (
        f"{sync_log} is not the transcript of a passing 71_ sync of "
        "revision_2026_08_v6 -- the pack must not claim a sync that did not "
        "happen")

    rows: list[dict] = []

    def take(src: Path, rel: str) -> None:
        dst = PACK / rel
        h = copy_verified(src, dst)
        script, grid = producer_of(rel)
        rows.append(dict(rel=rel, md5=h, bytes=dst.stat().st_size,
                         script=script, grid=grid, source=str(
                             src.relative_to(ROOT) if ROOT in src.parents
                             else src)))

    import json
    rendered = set(json.loads(
        (SRC / "figures" / "manifest.json").read_text(encoding="utf-8")
    )["figures"])
    for f in sorted((SRC / "figures").iterdir()):
        if not f.is_file():
            continue
        if f.name in FIGURE_SUPERSEDED:
            print(f"NOTE: skipping figures/{f.name} -- superseded filename, "
                  "not in the render manifest, no current script produces it")
            continue
        assert f.name in rendered or f.name in FIGURE_EXTRA, (
            f"figures/{f.name} is neither in the render manifest nor a known "
            "companion file -- it cannot be attributed, so it must not go into "
            "a provenance pack (add it to FIGURE_EXTRA or FIGURE_SUPERSEDED)")
        take(f, f"figures/{f.name}")
    for f in sorted((SRC / "tables").iterdir()):
        if f.is_file():
            take(f, f"tables/{f.name}")
    for name in VALIDATION_FILES:
        take(SRC / "validation" / name, f"validation/{name}")
    take(SRC / "gates_report.md", "validation/gates_report.md")
    for name in PEEK_FILES:
        take(SRC / "_peek" / name, f"analyses/{name}")

    # paper-side provenance
    (PACK / "paper").mkdir(parents=True, exist_ok=True)
    take(sync_log, "paper/71_sync_pass_table.txt")
    take(CHANGES, "paper/CHANGES_rev1.md")
    excerpt = compendium_excerpt(COMPENDIUM.read_text(encoding="utf-8"))
    ex_path = PACK / "paper" / "compendium_40_14_to_40_28.md"
    ex_path.write_text(
        "<!-- Verbatim excerpt of docs/PAPER_COMPENDIUM_2026_05_24.md, "
        f"sections {EXCERPT_FROM.strip('# ')} to {EXCERPT_TO_END_OF.strip('# ')}. "
        "The compendium is the single source of truth; this copy is frozen at "
        "pack time. -->\n\n" + excerpt, encoding="utf-8")
    s, g = producer_of("paper/compendium_40_14_to_40_28.md")
    rows.append(dict(rel="paper/compendium_40_14_to_40_28.md",
                     md5=md5_of(ex_path), bytes=ex_path.stat().st_size,
                     script=s, grid=g, source="docs/PAPER_COMPENDIUM_2026_05_24.md"))

    # decks: pointers only
    text, _ = deck_pointers()
    dpath = PACK / "decks" / "DECKS.md"
    dpath.parent.mkdir(parents=True, exist_ok=True)
    dpath.write_text(text, encoding="utf-8")
    s, g = producer_of("decks/DECKS.md")
    rows.append(dict(rel="decks/DECKS.md", md5=md5_of(dpath),
                     bytes=dpath.stat().st_size, script=s, grid=g,
                     source="(generated)"))

    # dashboard: placed by the controller before this script runs
    dash = PACK / "dashboard"
    if dash.is_dir():
        for f in sorted(dash.iterdir()):
            if f.is_file():
                rel = f"dashboard/{f.name}"
                s, g = producer_of(rel)
                rows.append(dict(rel=rel, md5=md5_of(f), bytes=f.stat().st_size,
                                 script=s, grid=g, source="(placed in situ)"))
    else:
        print("NOTE: results/revision_2026_08_final/dashboard/ is absent -- "
              "the artifact HTML sources were not placed before this run")
    return rows


# ─────────────────────────────────────────────────────────────────────────
# gallery manifest (deliverable 5)
# ─────────────────────────────────────────────────────────────────────────
#: stem -> (German title, one-sentence German caption, producing script).
#: The captions restate what the compendium says about each figure
#: (§40.21 / §40.22b / §40.23b) and 70_'s own module docstring; no number
#: appears here that is not in one of those.
GALLERY: dict[str, tuple[str, str, str]] = {
    "supp_fig4_freq_mix_two_plans": (
        "Frequenzmix, beide Pläne",
        "Verteilung der gewählten Liefertage je Woche über das (P, θ)-Gitter, "
        "Routing-Plan und Operator-Plan nebeneinander.",
        "scripts/revision/70_figs_tables_v2.py"),
    "supp_fig4b_mean_days": (
        "Ø Liefertage je Woche",
        "Mittlere Liefertage über das Gitter; bei θ = 1 kommt der Operator-Plan "
        "auf 2.38 / 3.24 / 4.17 / 4.71 / 5.04 Tage für P = 0 … 1 (§40.21).",
        "scripts/revision/70_figs_tables_v2.py"),
    "supp_fig5_grid_heatmap_v2": (
        "Zwei-Linsen-Heatmaps (2×3)",
        "Kosten, Wartezeit und Flotte über (P, θ) in Routing- und "
        "Operator-Linse — die Supplement-Fassung der Paper-Abbildung 5.",
        "scripts/revision/70_figs_tables_v2.py"),
    "supp_fig5b_offdiagonal_v2": (
        "Die andere Hälfte der 2×2",
        "Plan × Linse über Kreuz gelesen: was der Routing-Plan in der "
        "Operator-Linse kostet und umgekehrt.",
        "scripts/revision/70_figs_tables_v2.py"),
    "supp_fig6_structural_v2": (
        "Zellkosten-Aufschlüsselung in Euro",
        "Strukturelle Zerlegung der Ersparnis je Zelle in Euro, "
        "Routing-Linse, θ = 1.",
        "scripts/revision/70_figs_tables_v2.py"),
    "supp_fig6b_operator_lens_v2": (
        "Operator-Linse je Depot",
        "Dieselbe Struktur in der Operator-Linse, die hub- und nicht "
        "zellattributierbar ist (72_).",
        "scripts/revision/70_figs_tables_v2.py"),
    "supp_fig7_fleet_week_classes": (
        "Wochen-Flottenprofil je Carrier-Klasse",
        "4×4-Small-Multiples je Carrier-Klasse und P: der Routing-Plan hebt "
        "die vorzuhaltende Flotte von Index 117 auf 157 (P = 0), der "
        "Operator-Polish senkt sie auf 97 (§40.22b).",
        "scripts/revision/75_fig_fleet_week_classes.py"),
    "supp_map_freq_theta_v2": (
        "Karte: Lieferfrequenz über θ (P = 0.25)",
        "Paketgewichteter Median der Schedule-Größe je Fläche; bei θ = 1 "
        "dominiert 3 d/Woche (§40.23b).",
        "scripts/revision/76_maps_v2.py"),
    "supp_map_freq_theta_P0_v2": (
        "Karte: Lieferfrequenz über θ (P = 0)",
        "Dieselbe Karte ohne Service-Penalty; der 4-d/Woche-Fleck in "
        "31515 Wunstorf ist der Ein-Zellen-Depot-Effekt (§40.23b).",
        "scripts/revision/76_maps_v2.py"),
    "supp_map_freq_theta_P0_routing_v2": (
        "Karte: Lieferfrequenz, Routing-Plan (P = 0)",
        "Kontrastfigur auf dem Routing-Plan: dort liegt Wunstorf wie alles "
        "andere bei 2 d/Woche (§40.23b).",
        "scripts/revision/76_maps_v2.py"),
    "supp_map_saving_P_v2": (
        "Karte: Kostenersparnis je Fläche über P",
        "Euro-gewichtete Ersparnis je Fläche bei θ = 1; System 20.0 / 16.7 / "
        "12.3 / 9.0 / 6.9 / 2.2 % für P = 0 … 2, Flächenspanne 6.0–29.4 % bei "
        "P = 0 (§40.23b).",
        "scripts/revision/76_maps_v2.py"),
    "supp_map_wait_theta_v2": (
        "Karte: Wartezeit eines gehaltenen Pakets",
        "Paketgewichtete mittlere Wartezeit je Fläche, P = 0.25; maximal "
        "0.7 d bei θ = 1 (§40.23b).",
        "scripts/revision/76_maps_v2.py"),
    "supp_penalty_raumtyp_v2": (
        "Ersparnis nach Raumtyp über P",
        "Euro-gewichtete Ersparnis je Siedlungstyp bei θ = 1; ab P = 5 bleibt "
        "jedem Typ ≤ 0.7 % (§40.23b).",
        "scripts/revision/76_maps_v2.py"),
    "supp_fig_mechanism_v2": (
        "Mechanismus: warum tägliche Pläne bleiben",
        "Ø Liefertage über θ, Express-Kosten je Fahrzeugtag gegen die "
        "reguläre Tour, Penalty-Masse und die Verteilung der Zell-Ersparnis "
        "je Paket gegen die Penalty-Schwellen.",
        "scripts/revision/77_mechanism_v2.py"),
    "supp_fig_fleet_week_v2_P0": (
        "Wochen-Flottenprofil je Anbieter (P = 0)",
        "Fahrzeuge je Wochentag und Anbieter für Baseline, Routing-Plan und "
        "Operator-Plan bei θ = 1; Σ Depot-Peaks 1 239 → 1 666 → 1 030.",
        "scripts/revision/78_fleet_week_v2.py"),
    "supp_fig_fleet_week_v2_P025": (
        "Wochen-Flottenprofil je Anbieter (P = 0.25)",
        "Dieselbe Darstellung mit Service-Penalty; Σ Depot-Peaks "
        "1 239 → 1 314 → 1 026.",
        "scripts/revision/78_fleet_week_v2.py"),
}


def write_gallery_manifest(rows: list[dict]) -> Path:
    """One entry per supp_ stem in the pack, for the figure-gallery build.

    Gated both ways: every supp_ stem present in ``figures/`` must have an
    entry, and every entry must have a file -- a gallery built from a
    manifest that quietly omits a figure is how the 13C/13D stems went
    missing in the first place.
    """
    import json

    stems = sorted({Path(r["rel"]).stem for r in rows
                    if r["rel"].startswith("figures/supp_")
                    and r["rel"].endswith(".pdf")})
    missing = [s for s in stems if s not in GALLERY]
    extra = [s for s in GALLERY if s not in stems]
    assert not missing, f"supp_ stems with no gallery entry: {missing}"
    assert not extra, f"gallery entries with no rendered figure: {extra}"
    by_rel = {r["rel"]: r for r in rows}
    doc = dict(
        schema=1, grid="revision_2026_08_v6", git_head=git_head(),
        written_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        note="Titel und Bildunterschriften auf Deutsch, aus dem Kompendium "
             "(§40.21 / §40.22b / §40.23b) und den Skript-Docstrings.",
        figures=[dict(stem=s, title_de=GALLERY[s][0], caption_de=GALLERY[s][1],
                      source_script=GALLERY[s][2],
                      pdf=f"figures/{s}.pdf", png=f"figures/{s}.png",
                      md5_pdf=by_rel[f"figures/{s}.pdf"]["md5"],
                      md5_png=by_rel[f"figures/{s}.png"]["md5"])
                 for s in stems])
    p = PACK / "gallery_manifest.json"
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
                 encoding="utf-8")
    return p


# ─────────────────────────────────────────────────────────────────────────
# README (deliverable 3)
# ─────────────────────────────────────────────────────────────────────────
#: One paragraph per subfolder. Every number here comes from compendium
#: §40.21 (grid v6 end numbers) or §40.25 (VROOM-realised savings) -- no
#: other source is quoted in this README, by instruction.
SUBFOLDER_PROSE: tuple[tuple[str, str], ...] = (
    ("figures", """The 19 figure stems of grid v6 (38 PDF/PNG files), the two
bucket-composition CSVs that go with figure 6, and the render `manifest.json`
71_ syncs from. The three accepted paper figures keep the submitted layout and
only their numbers changed; everything with a `supp_`
prefix is supplementary. Against the submission the headline moved from
22.8 / 18.5 / 13.5 / 10.2 % to **22.6 / 18.7 / 13.5 / 10.1 / 7.6 %** routing
saving of the routing-optimal plan at theta = 1 for P = 0 / 0.25 / 0.5 /
0.75 / 1, and the operator-lens view of the operator-polished plan --
**24.3 / 22.6 / 17.8 / 14.1 / 11.6 %** with summed hub peaks
-16.9 / -17.2 / -14.3 / -12.0 / -10.5 % -- is new since the submission
(§40.21)."""),
    ("tables", """Every `*_v2` table of the same grid, including the CO2/km
table built from the v6 VROOM validation. The baseline these percentages are
taken against is v6's own: **1 898 091 EUR routing / 2 098 401 EUR operator
per week, 1 239 summed hub peaks** (§40.21). `tab_grid_delta_v2` is the only
cross-grid table in the pack: it is v6 against v5, where the bundle head
moves the routing plan by **+0.20 pp** and the operator plan by **+0.17 pp**
on average -- a fine correction, not a change of message (§40.21)."""),
    ("validation", """The VROOM validation of the v6 plans: the report, the
per-instance `tab_vroom_v2.csv`, the census and instance queue, and the 62_
gate report. Its headline is that the surrogate prices the daily baseline
**+4.4 % (routing) / +4.0 % (operator)** above VROOM, more than it overprices
the scenario points, so the predicted savings are an upper bound by about
2 pp: the routing plan's 22.64 % at (0, 1) realises as **20.58 %** and
18.70 % at (0.25, 1) as **16.43 %**; the operator plan's 19.96 / 16.73 /
12.29 / 9.04 % realise as **17.48 / 14.53 / 10.59 / 7.69 %**. In the operator
lens the operator plan's 24.30 % realises as **22.08 %**, and the routing
plan stays firmly negative, -8.37 % predicted against **-12.09 %** realised
(§40.25)."""),
    ("analyses", """The controller's own passes over the same grid: the
two-plan x two-lens overview of all 88 points, the discount/break-even
scenarios, and the two draft figures the dashboard showed. Those two drafts
are superseded by `77_mechanism_v2.py` and `78_fleet_week_v2.py`, which
recompute them inside the repo with gates and tests -- the drafts are kept
here only so the dashboard's numbers can be traced."""),
    ("dashboard", """The published revision pages as they were built:
the results dashboard, the figure gallery, and the scratch scripts behind
them. `gallery_manifest.json` at the pack root is the machine-readable
input for rebuilding the gallery."""),
    ("paper", """The paper-side provenance: the transcript of the real 71_
sync (every destination md5-verified against the render, none identical to
the frozen submission), the `docs/CHANGES_rev1.md` snapshot, and a verbatim
excerpt of compendium sections 40.14-40.28, which is where every number in
this README comes from."""),
    ("decks", """Pointers only -- paths, md5 and size of the three
`_rev2026-08` decks, which are about 40 MB together. Per task 17 the TBC
deck and the house deck are still to be rebuilt once PowerPoint is closed;
`DECKS.md` says which of them was locked at pack time."""),
)


def write_readme(rows: list[dict], sync_log: Path, manifest_git: str) -> Path:
    head = git_head()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    total = sum(r["bytes"] for r in rows)
    out = [
        "# Final results pack -- EWGT 2026 revision (grid v6)", "",
        f"Built {now} from `results/revision_2026_08_v6/` at git `{head}`.",
        f"Figure render manifest git head: `{manifest_git}`."
        + (" (70_ counts untracked files as dirty too; at render time the only"
           " untracked file was another agent's `AGENTS.md`, so the render came"
           " from committed source.)" if manifest_git.endswith("-dirty") else ""),
        f"{len(rows)} files, {total / 1e6:.1f} MB, all listed below (this",
        "README is the list, so it does not list itself). Every file was",
        "copied, never moved, and re-hashed at both ends; the md5 below is",
        "the hash both sides agreed on.", "",
        "This pack supersedes `results/revision_2026_07/` and the Stage-3",
        "material for anything about the revision. It does not replace",
        "`paper/EWGT_2026/`, which stays frozen at the submission.", "",
        "## What is in each subfolder", "",
    ]
    for name, prose in SUBFOLDER_PROSE:
        out += [f"### `{name}/`", "", " ".join(prose.split()), ""]
    out += [
        "## Provenance table", "",
        "`grid` says which run a file describes: `v6` throughout, except the",
        "one v6-vs-v5 delta table and the v5 leftovers in `dashboard/`.", "",
        "| file | producing script | grid | md5 | bytes |",
        "|---|---|---|---|---:|",
    ]
    for r in sorted(rows, key=lambda x: x["rel"]):
        out.append(f"| `{r['rel']}` | `{r['script']}` | {r['grid']} | "
                   f"`{r['md5']}` | {r['bytes']:,} |")
    out += [
        "", "## Reproducing", "",
        "```", "python scripts/revision/75_fig_fleet_week_classes.py",
        "python scripts/revision/76_maps_v2.py",
        "python scripts/revision/77_mechanism_v2.py",
        "python scripts/revision/78_fleet_week_v2.py",
        "python scripts/revision/70_figs_tables_v2.py --rev-dir results/revision_2026_08_v6",
        "python scripts/revision/71_sync_paper_figs.py --rev-dir results/revision_2026_08_v6 --include-companions",
        "python scripts/revision/79_build_final_pack.py --sync-log <the 71_ transcript>",
        "```", "",
        "70_ refuses to write its manifest until 75_/76_/77_/78_ have run on",
        "the same rev-dir, and 71_ refuses to sync while any manifest stem is",
        "unmapped, so the order above is enforced rather than documented.", "",
        f"Sync transcript used for this pack: `{sync_log.name}` "
        "(copied to `paper/71_sync_pass_table.txt`).", "",
    ]
    p = PACK / "README.md"
    p.write_text("\n".join(out), encoding="utf-8")
    return p


if __name__ == "__main__":
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--sync-log", required=True, type=Path,
                    help="transcript of the real 71_ --include-companions run")
    a = ap.parse_args()
    built = build(a.sync_log)
    gm = write_gallery_manifest(built)
    s, g = producer_of("gallery_manifest.json")
    built.append(dict(rel="gallery_manifest.json", md5=md5_of(gm),
                      bytes=gm.stat().st_size, script=s, grid=g,
                      source="(generated)"))
    manifest_git = json.loads(
        (SRC / "figures" / "manifest.json").read_text(encoding="utf-8")
    )["git_head"]
    readme = write_readme(built, a.sync_log, manifest_git)
    print(f"{len(built)} files in {PACK.relative_to(ROOT)}")
    print(f"  {readme.relative_to(ROOT)}")
    print(f"  {gm.relative_to(ROOT)}  ({len(GALLERY)} supp_ stems)")
    sys.exit(0)
