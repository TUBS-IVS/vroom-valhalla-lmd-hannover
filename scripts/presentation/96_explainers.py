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
                    hrule, hslide, label_box, pic, rect, txt)      # noqa: E402

import _conclusion as CONC                                        # noqa: E402
import _dummies as DUM                                            # noqa: E402

B = H.B
FIG = H.FIG / "tierB"          # the per-carrier and bulge panels
SRC = Path(r"C:/Users/bienzeisler/Documents/Präsentationen/EWGT/2026/"
           r"EWGT_26_Bienzeisler_new.pptx")
DEFAULT_OUT = SRC.parent / "EWGT_26_Bienzeisler_new_plus_explainers.pptx"

# The results slide each block unpacks, matched by a fragment of its title.
# Not by slide number: the working deck is edited between builds, and a
# hard-coded index silently starts pointing at the wrong slide. The number is
# resolved from the deck at build time and simply omitted if the title is gone.
EXPLAINS = {
    "mix": "delivery-frequency mix",
    "trade": "service improves faster",
    "range": "efficiency range",
    "maps": "where the delivery days land",
    "where": "pays where delivery is sparse",
    "valid": "validation",
}
_RESOLVED: dict = {}


def resolve_targets(prs) -> dict:
    """Find each block's results slide in the deck, by title fragment."""
    titles = []
    for i, sl in enumerate(prs.slides, 1):
        ph = [sh for sh in sl.shapes if sh.is_placeholder
              and sh.placeholder_format.idx == 0 and sh.has_text_frame]
        titles.append((i, ph[0].text_frame.text.replace("", " ") if ph else ""))
    out = {}
    for key, frag in EXPLAINS.items():
        hit = next(((i, t) for i, t in titles if frag in t.lower()), None)
        if hit is None:
            print(f"  ! no slide matches {frag!r} — tag will omit the number")
        else:
            print(f"  {key:6s} -> slide {hit[0]:2d}  {hit[1][:60]}")
        out[key] = hit
    return out


VAN = 189.15        # EUR per van-day, config/constants.py
CAP = 230           # parcels per van


# ── slide scaffolding: house headline, v1 body ──────────────────────────────
def xslide(prs, key, section, subject, source=None):
    """A backup slide, tagged with the results slide it unpacks."""
    s = hslide(prs, section, subject, source)
    hit = _RESOLVED.get(key)
    tag = (f"explains slide {hit[0]}  ·  {hit[1]}" if hit
           else f"explains  ·  {EXPLAINS[key]}")
    txt(s, L, 1.00, W, 0.30, tag, 11.5, bold=True, color=RED, spc=1.4,
        caps=True)
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
# the backup as a chain of questions
# ═══════════════════════════════════════════════════════════════════════════
# Each slide answers exactly one question, and the question is the headline.
# The order is the order somebody actually asks them in: first what looks odd,
# then whether it is real, then why, then whether the numbers can be trusted.
# `chapter()` records where each part starts so the contents slide can be
# filled in with real slide numbers once everything has been laid down.

_CHAPTERS: list = []
_CONTENTS = None


def chapter(prs, title, sub):
    _CHAPTERS.append((title, sub, len(prs.slides) + 1))


def block_contents(prs):
    """A map of the backup, filled in with real numbers once it is built."""
    global _CONTENTS
    s = hslide(prs, "Backup", "What is in here, and what each part answers",
               "Every slide in this section answers one question. Jump "
               "straight to the one you were asked.")
    _CONTENTS = s
    return s


def fill_contents():
    """Write the recorded chapters onto the contents slide."""
    if _CONTENTS is None:
        return
    y = BODY_T + 0.22
    for title, sub, num in _CHAPTERS:
        label_box(_CONTENTS, L, y, 0.95, 0.54, RED,
                  [(str(num), 21, True, WHITE)])
        txt(_CONTENTS, L + 1.18, y - 0.03, W - 1.18, 0.34, title, 20,
            bold=True, color=INK)
        txt(_CONTENTS, L + 1.18, y + 0.31, W - 1.18, 0.30, sub, 15, color=DIM)
        y += 0.70


# ═══════════════════════════════════════════════════════════════════════════
# part 1 · the odd thing in the frequency picture
# ═══════════════════════════════════════════════════════════════════════════
def block_mix(prs):
    chapter(prs, "The odd thing in the frequency picture",
            "Why bundling survives a huge fee when almost nobody takes part")

    # ── 1 · what looks odd ───────────────────────────────────────────────
    s = xslide(prs, "mix", "Part 1 · The odd thing", "What looks odd here?",
               "Share of the 312 delivery areas that give up daily delivery · "
               "Stage-3 grid.")
    txt(s, L + 2.30, BODY_T + 0.06, W - 2.30, 0.28,
        "share of customers willing to wait", 12, color=DIM, spc=0.8)
    y = B.table(s, ["Fee", "10 %", "20 %", "30 %", "40 %", "100 %"],
                [[("none", "key"), "87.5 %", "92.0 %", "94.2 %", "96.2 %",
                  "100 %"],
                 [("P = 5", "key"), ("49.7 %", "num"), ("42.9 %", "num"),
                  "14.7 %", "0 %", "0 %"],
                 [("P = 10", "key"), ("41.7 %", "num"), ("10.9 %", "num"),
                  "0 %", "0 %", "0 %"]],
                BODY_T + 0.40, widths=[2.3, 2.0, 2.0, 2.0, 2.0, 2.0],
                reserve=2.6)
    label_box(s, L, y + 0.30, W, 0.85, H.BLUSH,
              [("Follow the bottom row from left to right. The more people "
                "join in, the less gets bundled.", 22, True, RED)],
              line_col=RED)
    vbullets(s, ["That feels backwards. More volunteers should make bundling "
                 "easier, not harder.",
                 "And the top row, with no fee at all, does exactly what you "
                 "expect: it climbs.",
                 "The next slides work out why the two rows point in opposite "
                 "directions."],
             y + 1.30)

    # ── 2 + 3 · the argument, in pictures ────────────────────────────────
    DUM.slide_two_price_tags(prs, xslide)
    DUM.slide_the_proof(prs, xslide)

    # ── 3b · the fee is not a cost ───────────────────────────────────────
    # A reader stops here and asks: surely the shadow price is not added to
    # the bill? It is not. It steers the choice and nothing else, and the
    # slide has to say so before the next one shows what that steering costs.
    s = xslide(prs, "mix", "Part 1 · The odd thing",
               "Wait — is the fee added to the cost? No.",
               "Reported cost at the harshest fee, 10 % taking part: "
               "1 841 323 € with and without the penalty term. Identical.")
    for i, (head, body, col, fill, strike) in enumerate([
            ("What we count", "driving, vehicles, labour — real routing money",
             INK, PANEL, False),
            ("What we never count", "the service fee. Nobody is ever charged "
             "it", RED, H.BLUSH, True)]):
        x = L + i * (W / 2 + 0.20)
        cwd = W / 2 - 0.20
        rect(s, x, BODY_T + 0.35, cwd, 1.45, fill, line_col=LINE)
        txt(s, x + 0.30, BODY_T + 0.58, cwd - 0.60, 0.42, head, 24, bold=True,
            color=col)
        txt(s, x + 0.30, BODY_T + 1.06, cwd - 0.60, 0.60, body, 19, color=INK2,
            line=1.22)
        if strike:
            hrule(s, x + 0.30, BODY_T + 1.22, cwd - 0.60, RED, 2.5)
    label_box(s, L, BODY_T + 2.10, W, 0.95, WHITE,
              [("The fee lives only inside the optimiser. It decides which "
                "weekly plan wins — it is never money.", 22, True, INK)],
              line_col=LINE)
    vbullets(s, ["So why does a higher fee reduce the saving at all?",
                 "Because it makes the optimiser pick gentler plans. Gentler "
                 "plans deliver more often, and delivering more often costs "
                 "more.",
                 "The money that disappears is routing money never saved — not "
                 "a fee anybody hands over."],
             BODY_T + 3.25)

    # ── 3b2 · the same thing, with the numbers written out ───────────────
    DUM.slide_worked_example(prs, xslide)

    # ── 3c · why 10 % beats 20 % ─────────────────────────────────────────
    s = xslide(prs, "mix", "Part 1 · The odd thing",
               "Why is 10 % better than 20 %?",
               "Bar height is the most that could be saved; green is what is "
               "actually saved at the harshest fee.")
    pic(s, FIG / "figB5_prize_and_bill.png", L, BODY_T + 0.24, W, 3.35)
    vbullets(s, [[("Twice as many wait, so the fee steers twice as hard: "
                   "waiting retreats from ", False),
                  ("0.125 to 0.017 days", True), (".", False)],
                 "A gentler plan saves less — the region gives up 76 616 € "
                 "at 10 %, but 144 037 € at 20 %.",
                 "But the prize grows only a tenth — so 68 425 € are left, "
                 "then 15 109 €, then nothing."],
             BODY_T + 3.66)

    # ── 4 · are they outliers? ───────────────────────────────────────────
    # An earlier version of this block claimed the saving came from dropping
    # whole van-days. It does not: 24 of 6 397 are saved and 107 of the 130
    # areas drop none. The slides below carry what is in the data.
    s = xslide(prs, "mix", "Part 1 · The odd thing",
               "Are these just a few tiny areas we should throw out?",
               "The areas that give up daily delivery at the harshest fee and "
               "10 % participation, by their weekly parcel volume.")
    pic(s, FIG / "figB1_who_consolidates.png", L, BODY_T + 0.24, W, 3.35)
    vbullets(s, ["No. There are 130 of them, and together they carry a "
                 "quarter of all parcels in the region.",
                 [("The smallest still handles ", False),
                  ("774 parcels a week", True),
                  (", the typical one 2 172. Not one is a curiosity.", False)],
                 "Throwing them out would delete a quarter of the study area."],
             BODY_T + 3.75)

    # ── 5 · do we save vans? ─────────────────────────────────────────────
    s = xslide(prs, "mix", "Part 1 · The odd thing",
               "So do we save delivery vans? No.",
               "Vehicle-days from the chosen schedules; the network saving is "
               "measured against the 1 909 748 € weekly baseline.")
    pic(s, FIG / "figB2_where_the_money_is.png", L, BODY_T + 0.24, W, 3.35)
    vbullets(s, [[("Almost none: ", False), ("24 of 6 397", True),
                  (" van-days, and 107 of the 130 areas save not a single one.",
                   False)],
                 "The van still drives there daily for everyone who did not "
                 "join in. What gets shorter is the driving, not the fleet.",
                 "Counted area by area it looks like 251 823 €. For the whole "
                 "network it is 68 425 €."],
             BODY_T + 3.75)

    # ── 6 · what decides it ──────────────────────────────────────────────
    s = xslide(prs, "mix", "Part 1 · The odd thing",
               "What decides it, then?",
               "Each dot is one setting of fee and participation from the "
               "Stage-3 grid.")
    pic(s, FIG / "figB3_ptheta_collapse.png", L, BODY_T + 0.24, W, 3.35)
    vbullets(s, ["Not the fee on its own, and not the participation on its "
                 "own — the two multiplied.",
                 "That product is simply the bill you end up paying.",
                 "A fee of 10 with one in ten joining behaves just like a fee "
                 "of 1 with everybody joining."],
             BODY_T + 3.75)

    # ── 7 · which areas ──────────────────────────────────────────────────
    s = xslide(prs, "mix", "Part 1 · The odd thing",
               "Why do small areas bundle and big ones not?",
               "Weekly parcel volume of the areas choosing each number of "
               "delivery days. The boxes cover the middle half of them.")
    pic(s, FIG / "figB4_size_vs_takt.png", L, BODY_T + 0.24, W, 3.35)
    vbullets(s, ["The fewer delivery days an area picks, the smaller it is. "
                 "Big areas stay daily.",
                 "Double the participation and the size limit halves, so the "
                 "two-day option empties out first.",
                 "Five days a week lasts longest: it saves a little and costs "
                 "hardly any waiting."],
             BODY_T + 3.75)


# ═══════════════════════════════════════════════════════════════════════════
# part 2 · what a fee buys
# ═══════════════════════════════════════════════════════════════════════════
def block_trade(prs):
    chapter(prs, "What a fee actually buys",
            "Why the middle of the range beats both ends")

    s = xslide(prs, "trade", "Part 2 · What a fee buys",
               "What does each step of the fee cost us, and what does it buy?",
               "Everyone taking part. Saving measured against the 1 909 748 € "
               "weekly baseline; waiting averaged over all parcels, including "
               "those that never wait.")
    y = B.table(s, ["Fee", "We save", "People wait", "Saving given up",
                    "Waiting removed"],
                [[("none", "key"), ("22.8 %", "num"), "0.98 d", "—", "—"],
                 [("P = 0.25", "key"), ("18.5 %", "num"), "0.46 d",
                  "4.3 points", ("half of it", "good")],
                 [("P = 0.5", "key"), ("13.5 %", "num"), "0.23 d",
                  "9.3 points", ("three quarters", "good")],
                 [("P = 0.75", "key"), "10.2 %", "0.14 d", "12.6 points",
                  "86 %"],
                 [("P = 1", "key"), "7.5 %", "0.09 d", "15.3 points", "91 %"],
                 [("P = 2", "key"), "1.2 %", "0.01 d", "21.6 points", "99 %"]],
                BODY_T + 0.20, widths=[2.2, 2.0, 2.2, 2.6, 2.6], reserve=1.6)
    txt(s, L, y + 0.24, W, 1.32,
        "The first step is the bargain: give up a fifth of the saving and "
        "half the waiting disappears.\nEvery step after that buys less and "
        "costs more.", 22, bold=True, color=RED, line=1.28)

    s = xslide(prs, "range", "Part 2 · What a fee buys",
               "Why not simply take the biggest saving?",
               "Fleet figures are for the balanced and smoothed schedules; "
               "15 of the 80 settings sit on the efficient front.")
    for i, (nm, sav, wait, note, hot) in enumerate([
            ("No fee", "22.8 %", "0.98 d",
             "the cheapest week — but a full day of waiting", False),
            ("P = 0.25", "18.5 %", "0.46 d",
             "half the waiting for a fifth of the saving", True),
            ("P = 0.5", "13.5 %", "0.23 d",
             "peak fleet down 12.9 %, weekday swings down 54 %", True)]):
        x = L + i * (W / 3 + 0.02)
        cw = W / 3 - 0.30
        rect(s, x, BODY_T + 0.30, cw, 0.10, RED if hot else LINE)
        txt(s, x, BODY_T + 0.52, cw, 0.44, nm, 24, bold=True,
            color=RED if hot else INK)
        txt(s, x, BODY_T + 1.05, cw, 0.60, sav, 40, bold=True,
            color=RED if hot else INK)
        txt(s, x, BODY_T + 1.72, cw, 0.40, f"+{wait} waiting", 20, color=DIM)
        txt(s, x, BODY_T + 2.20, cw, 0.90, note, 20, color=INK2, line=1.22)
    vbullets(s, ["Push the fee to zero and you get the most money — and the "
                 "longest wait.",
                 "Push it high and the waiting vanishes, but so does the point "
                 "of doing it.",
                 "Which of the middle settings is right is a service decision, "
                 "not a modelling one."],
             BODY_T + 3.35)


# ═══════════════════════════════════════════════════════════════════════════
# part 3 · where it happens
# ═══════════════════════════════════════════════════════════════════════════
def block_maps(prs):
    chapter(prs, "Where it happens first",
            "Why the countryside changes and the city centre does not")

    s = xslide(prs, "maps", "Part 3 · Where it happens",
               "Why does the countryside change first?",
               "Delivery days chosen per area at P = 0.25, everyone taking "
               "part, grouped by settlement type.")
    y = B.table(s, ["Type of area", "How many", "Typical",
                    "Average", "On two days a week"],
                [[("Countryside", "key"), "118", ("2 days", "num"), "2.58",
                  ("60.2 %", "num")],
                 [("Suburb", "key"), "124", "3 days", "3.12", "29.0 %"],
                 [("City", "key"), "70", "3 days", "3.77", ("4.3 %", "num")]],
                BODY_T + 0.20, widths=[2.8, 1.6, 2.4, 2.0, 3.0], reserve=2.5)
    for i, (nm, pct, col) in enumerate([("countryside", 60.2, S6),
                                        ("suburb", 29.0, S4),
                                        ("city", 4.3, S1)]):
        yy = y + 0.30 + i * 0.52
        txt(s, L, yy, 1.9, 0.36, nm, 20, color=INK2)
        rect(s, L + 2.0, yy + 0.03, 8.6 * pct / 100.0, 0.30, col)
        txt(s, L + 2.15 + 8.6 * pct / 100.0, yy, 1.6, 0.36, f"{pct:.1f} %", 20,
            bold=True, color=INK)
    txt(s, L, y + 1.90, W, 0.48,
        "Fourteen rural areas drop to two days for every one city area that "
        "does.", 22, bold=True, color=RED)


# ═══════════════════════════════════════════════════════════════════════════
# part 4 · why some areas gain nothing
# ═══════════════════════════════════════════════════════════════════════════
def block_where(prs):
    chapter(prs, "Why dense areas gain nothing",
            "The three things that decide whether an area profits")

    s = xslide(prs, "where", "Part 4 · Why dense areas gain nothing",
               "What makes one area profit and its neighbour not?",
               "How strongly each property moves together with the saving, "
               "across all 312 areas. +1 would be a perfect match, −1 a "
               "perfect opposite.")
    y = B.table(s, ["Property of the area", "Link", "What it means"],
                [[("Far from the depot", "key"), ("+0.53", "num"),
                  "Every tour drives out and back. Bundling spreads that trip "
                  "over more parcels"],
                 [("Large area", "key"), ("+0.31", "num"),
                  "Long tours inside the area. Merging days makes them denser"],
                 [("Many parcels per address", "key"), ("−0.72", "num"),
                  "The tour is already full and short. There is nothing left "
                  "to win"]],
                BODY_T + 0.20, widths=[3.4, 1.6, 7.0], reserve=2.4)
    label_box(s, L, y + 0.28, W, 0.85, H.BLUSH,
              [("The strongest link is the negative one: bundling pays where "
                "density is missing.", 22, True, RED)], line_col=RED)
    vbullets(s, ["The typical rural area saves 25 %, the typical city area 9 %.",
                 "That gap is geography, not modelling — which is why a "
                 "uniform rollout wastes it."],
             y + 1.26)


# ═══════════════════════════════════════════════════════════════════════════
# part 5 · can we trust the numbers
# ═══════════════════════════════════════════════════════════════════════════
def block_valid(prs):
    chapter(prs, "Can the numbers be trusted",
            "What happens when a real routing solver checks the answer")

    s = xslide(prs, "valid", "Part 5 · Can we trust it",
               "What happens when a real routing solver checks the answer?",
               "Four settings recomputed from scratch with VROOM/Valhalla on "
               "1 248 observations the model had never seen.")
    y = B.table(s, ["Setting", "The model promised", "The solver delivered",
                    "Difference"],
                [[("No fee", "key"), "22.8 %", ("23.7 %", "good"),
                  "+0.9 points"],
                 [("P = 0.25", "key"), "18.5 %", ("19.8 %", "good"),
                  "+1.3 points"],
                 [("P = 0.5", "key"), "13.5 %", ("15.6 %", "good"),
                  "+2.1 points"],
                 [("P = 0.75", "key"), "10.2 %", ("13.0 %", "good"),
                  "+2.8 points"]],
                BODY_T + 0.20, widths=[3.0, 3.0, 3.4, 2.4], reserve=2.4)
    vbullets(s, ["Every one of the four came out better than promised.",
                 "So the model is wrong in the direction that is safe: it "
                 "under-promises.",
                 "What this does not prove is that the routing solver matches "
                 "the real street."],
             y + 0.28)


# ═══════════════════════════════════════════════════════════════════════════
# part 6 · the seven carriers
# ═══════════════════════════════════════════════════════════════════════════
def block_providers(prs):
    chapter(prs, "The seven carriers, side by side",
            "Why one average curve describes none of them")
    for key, subject, fig_name, src, bullets in [
        ("mix", "Do all seven carriers behave the same? No.",
         "figP1_mix_by_provider",
         "Delivery days chosen by each carrier's areas as participation rises, "
         "at P = 0.25.",
         ["DHL is the odd one out: most of its network keeps five or six "
          "delivery days.",
          "DPD, GLS and FedEx put roughly half their areas on two days.",
          "The single curve in the talk is the average of these two — and "
          "describes neither."]),
        ("range", "How much can each of them actually save?",
         "figP2_saving_by_provider",
         "Each carrier measured against its own daily-delivery cost, not "
         "against the regional total.",
         ["The best DHL can do is 10.6 %. GLS reaches 33.4 %.",
          "The shape is the same everywhere — only the ceiling differs.",
          "So one shared operating point is a compromise for everybody."]),
        ("where", "Does each carrier gain in the same places?",
         "figP3_map_saving_provider",
         "Saving per area at P = 0.25 with everyone taking part. Same colour "
         "scale for all seven.",
         ["Typical saving runs from 3.7 % for DHL to 32.6 % for FedEx.",
          "But the pattern is identical: the outside gains, the centre does "
          "not.",
          "What differs is how much of each network sits on the outside."]),
        ("maps", "And do they end up with the same delivery days?",
         "figP4_map_freq_provider",
         "Typical delivery days per area at P = 0.25, everyone taking part.",
         ["DHL holds five days; DPD, GLS and FedEx drop to two.",
          "Every carrier keeps a high frequency in the dense centre.",
          "How often you can deliver is a property of the network, not of the "
          "region."]),
    ]:
        sl = xslide(prs, key, "Part 6 · The seven carriers", subject, src)
        pic(sl, FIG / f"{fig_name}.png", L, BODY_T + 0.24, W, 3.35)
        vbullets(sl, bullets, BODY_T + 3.75)


# ═══════════════════════════════════════════════════════════════════════════
# each results figure, in full, once per carrier
# ═══════════════════════════════════════════════════════════════════════════
# The small multiples in block_providers compare the seven carriers at one
# operating point. This section gives each carrier the *whole* figure, so a
# question about one carrier can be answered with the same picture the talk
# already showed, just for that network. 4 families x 7 carriers = 28 slides.

CARRIERS = ["DHL", "Amazon", "Hermes", "UPS", "DPD", "GLS", "FedEx"]

FAMILIES = [
    ("mix", "figQ1_mix", "How often each area is served",
     "Eight panels, one per service fee. Left to right in each panel: more "
     "customers willing to wait. Dark blue = two delivery days a week, red = "
     "still daily.",
     "Read it as: how much of this network leaves daily delivery as the "
     "fee and the participation change."),
    ("range", "figQ2_saving", "What it saves, fee by fee",
     "Every square is one setting: a service fee (down the side) and a "
     "participation level (across the bottom). The number is the weekly cost "
     "saving against this carrier delivering daily.",
     "Read it as: bright is good. The bright corner is where this carrier "
     "would want to operate."),
    ("maps", "figQ3_freqmap", "Where the delivery days go",
     "The same region four times, as more customers join in. Colour is how "
     "often that area gets served.",
     "Read it as: the outside of the region turns dark first — that is where "
     "delivery days are dropped."),
    ("where", "figQ4_savingmap", "Where the money is saved",
     "The same region six times, one per service fee. Darker green = more "
     "saving in that area. The number above each map is what the whole "
     "network saves.",
     "Read it as: the greener the ring around the city, the more this carrier "
     "gains outside the core."),
]


def block_carrier_full(prs):
    chapter(prs, "Every figure, one carrier at a time",
            "The same four pictures, redrawn for each of the seven")
    for key, stem, subject, source, how in FAMILIES:
        for carrier in CARRIERS:
            path = FIG / f"{stem}_{carrier}.png"
            if not path.exists():
                print(f"  ! missing {path.name}")
                continue
            s = xslide(prs, key, f"Backup: {carrier}", subject, source)
            # The figure is the slide; give it the whole body and keep the
            # reading aid to one line underneath.
            pic(s, path, L, BODY_T + 0.10, W, 4.45)
            hh = H.text_height(how, W, 20, 1.22) + 0.06
            txt(s, L, BODY_T + 4.66, W, hh, how, 20, color=INK2,
                line=1.22)



# ═══════════════════════════════════════════════════════════════════════════
def build(out: Path) -> Path:
    prs = Presentation(str(SRC))
    n_before = len(prs.slides)
    _RESOLVED.update(resolve_targets(prs))

    # The closing summary belongs in the talk, not in the backup: build it,
    # then move it in front of the contact slide, which is the last one.
    CONC.slide_conclusion(prs)
    CONC.slide_takeaway(prs)
    contact_at = n_before - 1                    # 0-based index of the last
    H.move_slide(prs, len(prs.slides) - 2, contact_at)
    H.move_slide(prs, len(prs.slides) - 1, contact_at + 1)
    print(f"  conclusion + takeaway inserted as slides "
          f"{contact_at + 1}–{contact_at + 2}, before the contact slide")

    divider(prs, "B", "Backup", "Why the results\nlook like this",
            "Seven parts, each answering the questions a close reader asks — "
            "in the order they come up")
    block_contents(prs)
    block_mix(prs)         # 1 · the odd thing in the frequency picture
    block_trade(prs)       # 2 · what a fee actually buys
    block_maps(prs)        # 3 · where it happens first
    block_where(prs)       # 4 · why dense areas gain nothing
    block_valid(prs)       # 5 · can the numbers be trusted
    block_providers(prs)   # 6 · the seven carriers, side by side
    block_carrier_full(prs)  # 7 · every figure, one carrier at a time
    fill_contents()
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

    rt = D.load_raumtyp()
    sub = (s[np.isclose(s.penalty, 0.25) & np.isclose(s.share_willing, 1.0)]
           .merge(rt, on="plz", how="left"))
    for cls, want in (("rural", 60.2), ("suburban", 29.0), ("urban", 4.3)):
        m = sub[sub.raumtyp_3 == cls]
        check(f"two-day share, {cls}", 100 * (m[col] == 2).mean(), want, 0.15)

    c = D.load_costs()
    t = (c.groupby(["penalty", "share_willing"], as_index=False)
          .total_stage3_eur.sum())
    t["saved"] = D.BASE_TOTAL - t.total_stage3_eur

    def at(pen, th):
        return float(t[np.isclose(t.penalty, pen)
                       & np.isclose(t.share_willing, th)].saved.iloc[0])

    for th, want_prize, want_kept in ((0.1, 145041, 68425), (0.2, 159146, 15109)):
        check(f"prize at θ={th:g} (no fee)", at(0.0, th), want_prize, 60)
        check(f"kept at θ={th:g}, P=10", at(10.0, th), want_kept, 60)
        check(f"the fee costs at θ={th:g}", at(0.0, th) - at(10.0, th),
              want_prize - want_kept, 90)

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
