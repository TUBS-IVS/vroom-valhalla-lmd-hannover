"""Backup slides that explain the findings behind each results slide.

These are appended to the author's own working deck
(`EWGT_26_Bienzeisler_new.pptx`, 41 slides) as a backup block: one to three
slides per results slide, each answering the question that slide provokes when
somebody in the audience looks closely.

Two style rules, both from the author:

* **Body in the v1 language** — the kicker, real bulleted lists, stat blocks,
  tables and native-shape schematics of `91_build_pptx.py`. Not the icon-badge
  house style of `94_build_house_deck.py`.
* **Headlines in the house form** — two lines in the title placeholder: the
  running section, then this slide's subject.

Every number is measured from the Stage-3 grid or the repository's constants;
the analysis behind the frequency-mix block is reproduced by `--audit`, which
recomputes each figure and fails loudly if it has moved.

The source deck is never written to. Output goes to a new file.

Usage:
    python scripts/presentation/96_explainers.py [--out PATH] [--audit]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):      # theta and rho in the audit log
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from pptx import Presentation
from pptx.enum.text import PP_ALIGN

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "revision"))

import _house as H                                                # noqa: E402
from _house import (AMBER, BODY_T, CRIM, DIM, GREEN, INK, INK2, L, LINE,
                    PANEL, RED, S1, S3, S4, S5, S6, SW, TEAL, W, WHITE,
                    hrule, hslide, label_box, rect, txt)           # noqa: E402

B = H.B
SRC = Path(r"C:/Users/bienzeisler/Documents/Präsentationen/EWGT/2026/"
           r"EWGT_26_Bienzeisler_new.pptx")
DEFAULT_OUT = SRC.parent / "EWGT_26_Bienzeisler_new_plus_explainers.pptx"

# The results slides in the source deck these blocks belong to.
EXPLAINS = {
    "mix": (32, "The penalty shifts the delivery-frequency mix"),
    "trade": (33, "Service improves faster than savings disappear"),
    "range": (34, "The efficient range sits between P = 0.25 and P = 0.5"),
    "maps": (35, "Where the delivery days land"),
    "where": (36, "TBC pays where delivery is sparse and far"),
    "valid": (38, "Real routing confirms a conservative surrogate"),
}

VAN = 189.15        # EUR per van-day, config/constants.py
CAP = 230           # parcels per van


# ── slide scaffolding: house headline, v1 body ──────────────────────────────
def xslide(prs, key, section, subject, source=None):
    """A backup slide, tagged with the results slide it unpacks."""
    s = hslide(prs, section, subject, source)
    n, title = EXPLAINS[key]
    txt(s, L, 1.00, W, 0.30, f"explains slide {n}  ·  {title}", 11.5,
        bold=True, color=RED, spc=1.4, caps=True)
    return s


def vbullets(slide, items, t, *, size=20.0, floor=17.0, l=L, w=W, label=""):
    """A v1 bulleted list that stops above this slide's source line.

    `B.bullets` takes a fixed height and happily writes past the bottom of the
    slide. Here the block is measured first and the type steps down until it
    fits the room the citation leaves; if even the floor is not enough the
    build says so rather than printing into the footer.
    """
    limit = getattr(slide, "body_bottom", H.BODY_B) - t
    gap = 12.0

    def need(sz):
        return sum(H.text_height(it, w, sz, 1.22) + gap / 72.0 for it in items)

    while size > floor and need(size) > limit:
        size -= 0.5
    over = need(size) - limit
    if over > 0.03:
        print(f"  ! {label or 'bullets'}: overruns by {over:.2f} in at "
              f"{size:g} pt", file=sys.stderr)
    return B.bullets(slide, items, t, l=l, w=w, size=size,
                     h=max(0.4, limit), gap=int(gap))


def divider(prs, num, kicker, headline, body):
    return B.divider(prs, num, kicker, headline, body)


# ═══════════════════════════════════════════════════════════════════════════
# 32 · the frequency mix — the bulge at theta = 10 %
# ═══════════════════════════════════════════════════════════════════════════
def block_mix(prs):
    # ── 1 · the observation ──────────────────────────────────────────────
    s = xslide(prs, "mix", "Backup: The frequency mix",
               "At θ = 10 % the mix consolidates even under a punitive penalty",
               "Share of the 312 cells choosing fewer than six delivery days · "
               "Stage-3 grid.")
    # The table's header row is set in caps, and a capitalised theta reads as
    # a different letter entirely — so the symbol goes above the table instead.
    txt(s, L + 2.30, BODY_T + 0.06, W - 2.30, 0.28,
        "share of parcels willing to wait  ·  θ", 12, color=DIM, spc=0.8)
    y = B.table(s, ["Penalty", "10 %", "20 %", "30 %", "40 %", "100 %"],
                [[("P = 0", "key"), "87.5 %", "92.0 %", "94.2 %", "96.2 %",
                  "100 %"],
                 [("P = 5", "key"), ("49.7 %", "num"), ("42.9 %", "num"),
                  "14.7 %", "0 %", "0 %"],
                 [("P = 10", "key"), ("41.7 %", "num"), ("10.9 %", "num"),
                  "0 %", "0 %", "0 %"]],
                BODY_T + 0.40, widths=[2.3, 2.0, 2.0, 2.0, 2.0, 2.0],
                reserve=2.6)
    label_box(s, L, y + 0.30, W, 0.85, H.BLUSH,
              [("More participation, less consolidation — the opposite of "
                "what a penalty should do.", 22, True, RED)], line_col=RED)
    vbullets(s, ["At P = 10 four cells in ten still batch at θ = 10 %.",
                 "From θ = 30 % upwards not a single one does.",
                 "It is not the penalty rate — it is what the rate is "
                 "charged on."],
             y + 1.30)

    # ── 2 · the two price tags ───────────────────────────────────────────
    s = xslide(prs, "mix", "Backup: The frequency mix",
               "Two price tags decide it: van-days against waiting",
               "Medians per chosen frequency at P = 5, nominal θ = 10 %. "
               "Van-day = 189.15 € (config/constants.py); penalty = "
               "P · local willing share · weekly parcels · added wait.")
    y = B.table(s, ["Chosen takt", "Cells", "Van-days dropped", "Gain / week",
                    "Penalty / week", "Margin"],
                [[("2 days", "key"), "33", "4", ("757 €", "num"), "129 €",
                  ("5.9 ×", "good")],
                 [("3 days", "key"), "48", "3", ("567 €", "num"), "164 €",
                  ("3.5 ×", "good")],
                 [("4 days", "key"), "37", "2", ("378 €", "num"), "117 €",
                  ("3.2 ×", "good")],
                 [("5 days", "key"), "37", "1", ("189 €", "num"), "83 €",
                  ("2.3 ×", "good")],
                 [("daily", "key"), "157", "0", "—", "—", "—"]],
                BODY_T + 0.30, widths=[2.2, 1.4, 2.4, 2.0, 2.2, 1.6],
                reserve=2.2)
    vbullets(s, ["A van-day costs the same whether it carries 5 parcels or 230.",
                  [("So the gain is a fixed ", False), ("189.15 €", True),
                   (" per dropped delivery day — it barely depends on volume.",
                    False)],
                  [("The penalty is charged per waiting parcel, so it scales "
                    "with ", False), ("θ", True), (".", False)],
                  "At θ = 10 % the gain outruns the penalty three- to sixfold."],
              y + 0.28)

    # ── 3 · who actually batches ─────────────────────────────────────────
    s = xslide(prs, "mix", "Backup: The frequency mix",
               "The cells that batch hardest are the ones nobody opted into",
               "At nominal θ = 10 %, 45.8 % of business customers accept a "
               "wait but only 0.1 % of private ones (fs_b2c / fs_b2b).")
    y = B.table(s, ["Chosen takt", "B2C share", "Locally willing",
                    "Batched parcels per week"],
                [[("2 days", "key"), ("97 %", "num"), ("1 %", "num"),
                  ("26", "num")],
                 [("3 days", "key"), "95 %", "2 %", "52"],
                 [("4 days", "key"), "94 %", "3 %", "62"],
                 [("5 days", "key"), "89 %", "5 %", "99"],
                 [("daily", "key"), ("73 %", "num"), ("12 %", "num"),
                  ("512", "num")]],
                BODY_T + 0.30, widths=[2.6, 2.4, 2.8, 4.0], reserve=2.3)
    label_box(s, L, y + 0.26, W, 0.85, PANEL,
              [("26 parcels a week, driven out six times — that is the whole "
                "story.", 22, True, INK)], line_col=LINE)
    vbullets(s, ["Almost nobody waits, so the penalty is tiny — but four "
                 "near-empty van-days still fall away.",
                 "Business-heavy cells already carry 512 batched parcels; "
                 "there the vans are needed anyway."],
             y + 1.28)

    # ── 4 · why it dies at 20 % ──────────────────────────────────────────
    s = xslide(prs, "mix", "Backup: The frequency mix",
               "Why the bulge dies between θ = 10 % and θ = 20 %",
               "Willingness curves evaluated at both levels; gain and penalty "
               "as on the previous slide.")
    B.stats(s, [("45.8 → 72.4 %", "business customers willing", False),
                ("0.1 → 5.5 %", "private customers willing", False),
                ("4 ×", "more parcels waiting", True)], BODY_T + 0.30,
            h=1.15, sz=34)
    y = B.table(s, ["Chosen takt", "Gain / week", "Penalty at θ = 10 %",
                    "Penalty at θ = 20 %", "Margin left"],
                [[("2 days", "key"), "757 €", "129 €", ("544 €", "num"),
                  ("1.4 ×", "num")],
                 [("3 days", "key"), "567 €", "164 €", "477 €", "1.2 ×"],
                 [("4 days", "key"), "378 €", "117 €", "355 €", "1.1 ×"],
                 [("5 days", "key"), "189 €", "83 €", ("194 €", "num"),
                  ("gone", "num")]],
                BODY_T + 1.75, widths=[2.2, 2.2, 2.6, 2.6, 2.2], reserve=1.5)
    txt(s, L, y + 0.28, W, 0.95,
        "The gain is fixed; the penalty quadruples. Deep batching goes first, "
        "the gentle five-day takt survives longest — and at θ = 30 % nothing "
        "is left.", 22, bold=True, color=RED, line=1.28)


# ═══════════════════════════════════════════════════════════════════════════
# 33 / 34 · the trade and the efficient range
# ═══════════════════════════════════════════════════════════════════════════
def block_trade(prs):
    s = xslide(prs, "trade", "Backup: The cost–service trade",
               "What each penalty step actually costs and buys",
               "Stage-3 grid at θ = 1. Saving against the 1 909 748 € weekly "
               "baseline; wait is the volume-weighted average over all parcels.")
    y = B.table(s, ["Penalty", "Saving", "Added wait", "Saving given up",
                    "Wait removed"],
                [[("P = 0", "key"), ("22.8 %", "num"), "0.98 d", "—", "—"],
                 [("P = 0.25", "key"), ("18.5 %", "num"), "0.46 d",
                  "4.3 pp", ("53 %", "good")],
                 [("P = 0.5", "key"), ("13.5 %", "num"), "0.23 d",
                  "9.3 pp", ("77 %", "good")],
                 [("P = 0.75", "key"), "10.2 %", "0.14 d", "12.6 pp", "86 %"],
                 [("P = 1", "key"), "7.5 %", "0.09 d", "15.3 pp", "91 %"],
                 [("P = 2", "key"), "1.2 %", "0.01 d", "21.6 pp", "99 %"]],
                BODY_T + 0.30, widths=[2.2, 2.0, 2.2, 2.6, 2.4], reserve=1.6)
    txt(s, L, y + 0.24, W, 1.32,
        "The first step is the bargain: it hands back a fifth of the saving "
        "and removes half of the waiting.\nEvery step after that buys less "
        "and costs more.", 22, bold=True, color=RED, line=1.28)

    s = xslide(prs, "range", "Backup: The cost–service trade",
               "Why the knee and not the extreme is the operating point",
               "15 of 80 grid points lie on the efficient front. Fleet figures "
               "are Stage-3, per-hub balanced and system-smoothed.")
    for i, (nm, sav, wait, note, hot) in enumerate([
            ("P = 0", "22.8 %", "0.98 d",
             "the cost-optimal extreme — a full day of waiting", False),
            ("P = 0.25", "18.5 %", "0.46 d",
             "wait halves for 4.3 pp of saving", True),
            ("P = 0.5", "13.5 %", "0.23 d",
             "12.9 % peak-fleet cut, 54 % less weekday variation", True)]):
        x = L + i * (W / 3 + 0.02)
        cw = W / 3 - 0.30
        rect(s, x, BODY_T + 0.30, cw, 0.10, RED if hot else LINE)
        txt(s, x, BODY_T + 0.52, cw, 0.44, nm, 24, bold=True,
            color=RED if hot else INK)
        txt(s, x, BODY_T + 1.05, cw, 0.60, sav, 40, bold=True,
            color=RED if hot else INK)
        txt(s, x, BODY_T + 1.72, cw, 0.40, f"+{wait} waiting", 20, color=DIM)
        txt(s, x, BODY_T + 2.20, cw, 0.90, note, 20, color=INK2, line=1.22)
    vbullets(s, ["Beyond the knee, waiting grows faster than saving.",
                  "The efficient front is where no point gives more saving at "
                  "the same wait.",
                  "Which knee is right is a service decision, not a modelling "
                  "one."],
              BODY_T + 3.35)


# ═══════════════════════════════════════════════════════════════════════════
# 35 · where the days land
# ═══════════════════════════════════════════════════════════════════════════
def block_maps(prs):
    s = xslide(prs, "maps", "Backup: Where the days land",
               "The periphery consolidates first, and by a wide margin",
               "Chosen delivery frequency per provider–area cell at P = 0.25, "
               "θ = 1, by BBSR settlement class (plz_raumtyp.csv).")
    y = B.table(s, ["Settlement type", "Cells", "Median takt",
                    "Mean takt", "On two days a week"],
                [[("Rural", "key"), "118", ("2 days", "num"), "2.58",
                  ("60.2 %", "num")],
                 [("Suburban", "key"), "124", "3 days", "3.12", "29.0 %"],
                 [("Urban", "key"), "70", "3 days", "3.77", ("4.3 %", "num")]],
                BODY_T + 0.30, widths=[2.8, 1.6, 2.4, 2.0, 3.0], reserve=2.5)
    # a bar showing the 2-day share per class, so the gap is visible at a glance
    for i, (nm, pct, col) in enumerate([("rural", 60.2, S6),
                                        ("suburban", 29.0, S4),
                                        ("urban", 4.3, S1)]):
        yy = y + 0.34 + i * 0.52
        txt(s, L, yy, 1.9, 0.36, nm, 20, color=INK2)
        rect(s, L + 2.0, yy + 0.03, 8.6 * pct / 100.0, 0.30, col)
        txt(s, L + 2.15 + 8.6 * pct / 100.0, yy, 1.6, 0.36, f"{pct:.1f} %", 20,
            bold=True, color=INK)
    txt(s, L, y + 1.92, W, 0.48,
        "Fourteen times the rural share of the urban one.", 22, bold=True,
        color=RED)


# ═══════════════════════════════════════════════════════════════════════════
# 36 · where it pays
# ═══════════════════════════════════════════════════════════════════════════
def block_where(prs):
    s = xslide(prs, "where", "Backup: Where it pays",
               "Three structural drivers, and what each one means",
               "Spearman correlations across the 312 provider–area cells, each "
               "at its provider's own P*.")
    y = B.table(s, ["Driver", "ρ", "Why it works that way"],
                [[("Distance to the depot", "key"), ("+0.53", "num"),
                  "Every tour pays the stem twice; batching amortises it over "
                  "more parcels"],
                 [("Area size", "key"), ("+0.31", "num"),
                  "A large area means long local tours, which get denser when "
                  "days are merged"],
                 [("Parcels per drop site", "key"), ("−0.72", "num"),
                  "Where many parcels already arrive at one address, the tour "
                  "is dense — nothing left to win"]],
                BODY_T + 0.30, widths=[3.2, 1.6, 7.0], reserve=2.4)
    label_box(s, L, y + 0.28, W, 0.85, H.BLUSH,
              [("The strongest driver is negative: consolidation pays where "
                "density is missing.", 22, True, RED)], line_col=RED)
    vbullets(s, ["The median rural area saves 25 %, the median urban area 9 %.",
                 "Structural, not an artefact — and why a uniform rollout "
                 "wastes the mechanism."],
             y + 1.26)


# ═══════════════════════════════════════════════════════════════════════════
# 38 · validation
# ═══════════════════════════════════════════════════════════════════════════
def block_valid(prs):
    s = xslide(prs, "valid", "Backup: Validation",
               "The surrogate is wrong in the direction that is safe",
               "Four out-of-sample Stage-3 operating points at θ = 1, "
               "re-routed with VROOM/Valhalla. n = 1 248 · R² = 0.997 · "
               "bias +2.73 %.")
    y = B.table(s, ["Operating point", "Predicted", "Realised by the solver",
                    "Gap"],
                [[("P = 0", "key"), "22.8 %", ("23.7 %", "good"), "+0.9 pp"],
                 [("P = 0.25", "key"), "18.5 %", ("19.8 %", "good"), "+1.3 pp"],
                 [("P = 0.5", "key"), "13.5 %", ("15.6 %", "good"), "+2.1 pp"],
                 [("P = 0.75", "key"), "10.2 %", ("13.0 %", "good"),
                  "+2.8 pp"]],
                BODY_T + 0.30, widths=[3.0, 2.4, 3.4, 2.2], reserve=2.4)
    vbullets(s, ["Every validated point beats its own prediction.",
                  "The gap widens with the penalty, so the margin is largest "
                  "where service is protected most.",
                  [("A schedule chosen on the surrogate therefore does ",
                    False), ("at least as well", True), (" in reality.", False)],
                  "What this does not prove: that VROOM matches the street."],
              y + 0.28)


# ═══════════════════════════════════════════════════════════════════════════
def build(out: Path) -> Path:
    prs = Presentation(str(SRC))
    n_before = len(prs.slides)
    divider(prs, "B", "Backup", "Why the results\nlook like this",
            "One to three slides behind each results slide — the questions a "
            "close reader asks")
    block_mix(prs)
    block_trade(prs)
    block_maps(prs)
    block_where(prs)
    block_valid(prs)
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out))
    print(f"  {n_before} original slides kept, "
          f"{len(prs.slides) - n_before} appended")
    return out


def audit() -> int:
    """Recompute every measured figure on these slides; fail if one moved."""
    import _data as D
    import _stage3_common as C
    bad = []

    def check(name, got, want, tol):
        ok = abs(got - want) <= tol
        print(f"  {'ok ' if ok else 'FAIL'} {name}: {got:.4g} "
              f"(slide says {want:g})")
        if not ok:
            bad.append(name)

    s = D.load_chosen_stage3()
    col = "schedule_size_system_smoothed"
    for pen, th, want in ((5.0, 0.1, 49.7), (5.0, 0.2, 42.9), (5.0, 0.3, 14.7),
                          (10.0, 0.1, 41.7), (10.0, 0.2, 10.9),
                          (0.0, 0.1, 87.5)):
        sub = s[np.isclose(s.penalty, pen) & np.isclose(s.share_willing, th)]
        check(f"consolidating share P={pen:g} θ={th:g}",
              100 * (sub[col] < 6).mean(), want, 0.15)

    check("B2B willing at θ=0.1", 100 * C._willing_b2b(0.1), 45.8, 0.1)
    check("B2C willing at θ=0.1", 100 * C._willing_b2c(0.1), 0.1, 0.05)
    check("B2B willing at θ=0.2", 100 * C._willing_b2b(0.2), 72.4, 0.1)
    check("B2C willing at θ=0.2", 100 * C._willing_b2c(0.2), 5.5, 0.1)
    check("van-day cost", VAN, 189.15, 0.001)

    rt = D.load_raumtyp()
    sub = (s[np.isclose(s.penalty, 0.25) & np.isclose(s.share_willing, 1.0)]
           .merge(rt, on="plz", how="left"))
    for cls, want in (("rural", 60.2), ("suburban", 29.0), ("urban", 4.3)):
        m = sub[sub.raumtyp_3 == cls]
        check(f"two-day share, {cls}", 100 * (m[col] == 2).mean(), want, 0.15)

    g = D.saving_grid().merge(D.load_wait(), on=["penalty", "share_willing"])
    at1 = g[np.isclose(g.share_willing, 1.0)]
    for pen, want in ((0.0, 22.8), (0.25, 18.5), (0.5, 13.5), (0.75, 10.2)):
        check(f"saving at P={pen:g}, θ=1",
              float(at1[np.isclose(at1.penalty, pen)].saving_pct.iloc[0]),
              want, 0.06)
    print("\n" + ("AUDIT FAILED: " + ", ".join(bad) if bad else "audit clean"))
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--audit", action="store_true",
                    help="recompute every number on these slides and stop")
    a = ap.parse_args()
    if a.audit:
        return audit()
    if not SRC.exists():
        print(f"source deck not found: {SRC}", file=sys.stderr)
        return 1
    keep = SRC.with_name(SRC.stem + "_untouched_copy.pptx")
    if not keep.exists():
        shutil.copy2(SRC, keep)
        print(f"kept an untouched copy at {keep.name}")
    p = build(a.out)
    print(f"wrote {p}")
    print(f"  {len(Presentation(str(p)).slides)} slides total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
