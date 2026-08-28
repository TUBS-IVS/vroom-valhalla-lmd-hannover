"""The two slides that explain the θ = 10 % effect without any notation.

The measured backup slides show *that* the effect exists. These two show *why*,
in the plainest terms the material allows: the saving is earned per trip, the
fee is paid per parcel, and only one of the two has a ceiling.

Everything is drawn from native shapes so it stays editable, and the only
numbers on them are the two rows of the measured table.

Imported by 96_explainers.py; not meant to be run on its own.

WITHDRAWN NOTICE (2026-08-28)
-----------------------------
Every slide in this module illustrates the same finding: that bundling survives
a punitive fee when few customers take part, because the fee scales with
adoption and the routing gain does not. That finding has been WITHDRAWN. It was
an artefact of the pre-revision express price -- the standard parcels of every
non-delivering area of a hub rode ONE pooled tour, which no operator would
dispatch and which only the scenario could ever have. Under the universal tour
rule the cell it was built on consolidates 2.9 % of areas and saves 0.03 %
(compendium 40.7-40.9, 40.15).

The drawings are kept as the record of what was argued. Calling any of them
raises unless `allow_withdrawn()` has been called first, so a deck builder
cannot put them back on a slide by accident; `96_explainers.py --no-revision`
is the one caller that legitimately does.
"""
from __future__ import annotations

from pptx.enum.text import PP_ALIGN

import _house as H
from _house import (BODY_T, DIM, GREEN, INK, INK2, L, LINE, PANEL, RED, S6,
                    W, WHITE, hrule, label_box, rect, txt)

B = H.B


_ALLOW_WITHDRAWN = False


def allow_withdrawn(on: bool = True) -> None:
    """Permit the withdrawn illustrations, for the submission-era rebuild."""
    global _ALLOW_WITHDRAWN
    _ALLOW_WITHDRAWN = on


def _withdrawn(name: str) -> None:
    if not _ALLOW_WITHDRAWN:
        raise SystemExit(
            f"_dummies.{name} illustrates the WITHDRAWN theta = 10 % finding "
            f"(compendium 40.7-40.9 / 40.15) and is not drawn. Call "
            f"_dummies.allow_withdrawn() first if you are deliberately "
            f"rebuilding the submission-era deck.")


def _van(slide, x, y, w, h, *, skipped: bool, day: str):
    """One delivery day: a van that either rolls or does not."""
    body = PANEL if skipped else S6
    rect(slide, x, y, w, h, body, line_col=LINE if skipped else None)
    if not skipped:
        txt(slide, x, y + h * 0.28, w, 0.34, "VAN", 13, bold=True,
            color=WHITE, align=PP_ALIGN.CENTER, spc=1.0)
    if skipped:
        # a cross, drawn as two connectors, so "not driven" reads at a glance
        for x1, y1, x2, y2 in ((x + 0.10, y + 0.10, x + w - 0.10, y + h - 0.10),
                               (x + w - 0.10, y + 0.10, x + 0.10, y + h - 0.10)):
            ln = slide.shapes.add_connector(
                1, H.Inches(x1), H.Inches(y1), H.Inches(x2), H.Inches(y2))
            ln.line.color.rgb = RED
            ln.line.width = H.Pt(2.2)
    txt(slide, x, y + h + 0.06, w, 0.30, day, 15, color=DIM,
        align=PP_ALIGN.CENTER)


def _coins(slide, x, y, n, *, colour=RED, d=0.30, gap=0.10, per_row=5):
    """n little discs — one per parcel that has to wait, so the bill is visible."""
    for i in range(n):
        cx = x + (i % per_row) * (d + gap) + d / 2
        cy = y + (i // per_row) * (d + gap) + d / 2
        B.dot(slide, cx, cy, d, colour)
    rows = (n + per_row - 1) // per_row
    return y + rows * (d + gap)


def slide_two_price_tags(prs, xslide):
    """Why more participants can make bundling stop paying.

    An earlier version of this slide drew six vans with four crossed out. That
    is what full participation looks like, not 10 % — at 10 % the van still
    goes out daily for the 90 % who did not opt in, which is exactly why only
    24 of 6 397 vehicle-days are ever saved. The saving side now shows the
    measured figure instead of a picture that is true only at the far end.
    """
    _withdrawn("slide_two_price_tags")
    s = xslide(prs, "mix", "Backup: The frequency mix",
               "Ten times the people, but not ten times the saving",
               "Weekly cost saving of the whole region against daily delivery, "
               "with no service fee applied at all (P = 0) · Stage-3 grid.")
    txt(s, L, BODY_T + 0.02, W, 0.34,
        "Two separate scenarios — not two things happening at the same time.",
        17, color=DIM, align=PP_ALIGN.CENTER)
    txt(s, L, BODY_T + 0.40, W, 0.50,
        "The bill grows with every person. The saving does not.", 28,
        bold=True, color=INK, align=PP_ALIGN.CENTER)

    cw = (W - 0.55) / 2
    xr = L + cw + 0.55
    lab_w, bar_x = 2.30, 3.05
    rows = [("10 % join in", "the other 90 % keep daily delivery", 7.6, 1),
            ("everyone joins", "every parcel can be held back", 22.8, 10)]

    # ── left: what it is actually worth ──────────────────────────────────
    rect(s, L, BODY_T + 1.02, cw, 2.85, WHITE, line_col=LINE)
    rect(s, L, BODY_T + 1.02, cw, 0.09, GREEN)
    txt(s, L + 0.28, BODY_T + 1.22, cw - 0.56, 0.36, "What you save", 20,
        bold=True, color=GREEN, spc=1.2, caps=True)
    for i, (name, sub, val, _) in enumerate(rows):
        y = BODY_T + 1.72 + i * 0.86
        txt(s, L + 0.28, y, lab_w, 0.32, name, 17, bold=True, color=INK)
        txt(s, L + 0.28, y + 0.30, lab_w, 0.46, sub, 13, color=DIM,
            line=1.18)
        rect(s, L + bar_x, y + 0.06, (cw - bar_x - 1.35) * val / 22.8, 0.40,
             GREEN)
        txt(s, L + cw - 1.30, y + 0.02, 1.05, 0.40, f"{val:.1f} %", 22,
            bold=True, color=GREEN)
    txt(s, L + 0.28, BODY_T + 3.36, cw - 0.56, 0.42,
        "10 × the people, 3 × the saving.", 21, bold=True, color=INK2)

    # ── right: what the waiting costs ────────────────────────────────────
    rect(s, xr, BODY_T + 1.02, cw, 2.85, WHITE, line_col=LINE)
    rect(s, xr, BODY_T + 1.02, cw, 0.09, RED)
    txt(s, xr + 0.28, BODY_T + 1.22, cw - 0.56, 0.36, "What you pay", 20,
        bold=True, color=RED, spc=1.2, caps=True)
    for i, (name, sub, _, coins) in enumerate(rows):
        y = BODY_T + 1.72 + i * 0.86
        txt(s, xr + 0.28, y, lab_w, 0.32, name, 17, bold=True, color=INK)
        txt(s, xr + 0.28, y + 0.30, lab_w, 0.46, sub, 13, color=DIM,
            line=1.18)
        _coins(s, xr + bar_x, y + 0.04, coins)
    txt(s, xr + 0.28, BODY_T + 3.36, cw - 0.56, 0.42,
        "10 × the people, 10 × the bill.", 21, bold=True, color=RED)

    # ── the asymmetry, spelled out ───────────────────────────────────────
    label_box(s, L, BODY_T + 4.06, W, 0.95, H.BLUSH,
              [("The bill grows tenfold, the saving threefold. That is why "
                "more participants can make bundling stop paying.",
                21, True, RED)], line_col=RED)
    return s


def slide_the_proof(prs, xslide):
    """The same table, two rows, opposite directions — the fee is the cause."""
    _withdrawn("slide_the_proof")
    s = xslide(prs, "mix", "Backup: The frequency mix",
               "The proof: two rows of the same table",
               "Share of the 312 delivery areas that give up daily delivery, "
               "as participation rises from 10 % to 100 % · Stage-3 grid.")
    rows = [("No fee at all", ["87.5", "92.0", "94.2", "96.2", "100"],
             GREEN, "more people joining only helps", "▲"),
            ("Harshest fee", ["41.7", "10.9", "0", "0", "0"],
             RED, "more people joining kills it", "▼")]
    heads = ["10 %", "20 %", "30 %", "40 %", "100 %"]
    bw = 1.62
    x0 = L + 3.35
    txt(s, x0, BODY_T + 0.10, 5 * bw, 0.30,
        "share of customers willing to wait", 14, color=DIM,
        align=PP_ALIGN.CENTER)
    for i, h in enumerate(heads):
        txt(s, x0 + i * bw, BODY_T + 0.42, bw, 0.32, h, 17, color=DIM,
            align=PP_ALIGN.CENTER)
    for r, (name, vals, col, note, arrow) in enumerate(rows):
        y = BODY_T + 0.90 + r * 1.42
        rect(s, L, y, 3.10, 1.02, PANEL, line_col=LINE)
        txt(s, L + 0.18, y + 0.12, 2.74, 0.36, name, 20, bold=True, color=col)
        txt(s, L + 0.18, y + 0.52, 2.74, 0.52, note, 15, color=DIM,
            line=1.20)
        for i, v in enumerate(vals):
            txt(s, x0 + i * bw, y + 0.22, bw, 0.52, f"{v} %", 26, bold=True,
                color=col, align=PP_ALIGN.CENTER)
        txt(s, x0 + 5 * bw + 0.15, y + 0.18, 0.70, 0.60, arrow, 34, bold=True,
            color=col)
    hrule(s, L, BODY_T + 3.66, W, LINE, 1.25)
    txt(s, L, BODY_T + 3.86, W, 0.95,
        "The only difference between these two rows is whether a fee is "
        "charged at all.\nSo the fee is what turns the direction around — not "
        "the participation.", 22, bold=True, color=INK, line=1.28)
    return s


def _option(slide, x, y, w, name, drive, fee, total, *, wins):
    """One row of the comparison: what it costs, and what the total comes to."""
    rect(slide, x, y, w, 0.86, H.BLUSH if wins else PANEL,
         line_col=GREEN if wins else LINE, line_w=2.0 if wins else 1.0)
    rect(slide, x, y, 0.09, 0.86, GREEN if wins else LINE)
    txt(slide, x + 0.26, y + 0.09, w - 2.30, 0.34, name, 19, bold=True,
        color=INK)
    txt(slide, x + 0.26, y + 0.45, w - 2.30, 0.32,
        f"{drive} driving  +  {fee} fee", 15, color=DIM)
    txt(slide, x + w - 2.00, y + 0.14, 1.80, 0.44, total, 24, bold=True,
        color=GREEN if wins else INK, align=PP_ALIGN.RIGHT)
    if wins:
        txt(slide, x + w - 2.00, y + 0.58, 1.80, 0.24, "← this one wins", 13,
            bold=True, color=GREEN, align=PP_ALIGN.RIGHT)


def slide_worked_example(prs, xslide):
    """The same area under both scenarios, with the arithmetic written out."""
    _withdrawn("slide_worked_example")
    s = xslide(prs, "mix", "Part 1 · The odd thing",
               "The same area, two scenarios — with the arithmetic",
               "Illustrative figures for one area of 2 000 parcels a week at a "
               "fee of 10 € per parcel per day of waiting. The measured "
               "regional figures follow on the next slide.")

    # ── the rule the optimiser applies ───────────────────────────────────
    rect(s, L, BODY_T + 0.02, W, 0.86, PANEL, line_col=LINE)
    rect(s, L, BODY_T + 0.02, 0.09, 0.86, RED)
    txt(s, L + 0.30, BODY_T + 0.10, W - 0.60, 0.42,
        "Total  =  driving cost   +   fee × waiting parcels × waiting days",
        24, bold=True, color=INK, align=PP_ALIGN.CENTER)
    txt(s, L + 0.30, BODY_T + 0.54, (W - 0.60) * 0.42, 0.28,
        "real money — this is what gets reported", 14, color=DIM,
        align=PP_ALIGN.CENTER)
    txt(s, L + 0.30 + (W - 0.60) * 0.48, BODY_T + 0.54, (W - 0.60) * 0.52,
        0.28, "never paid — it only decides which line wins", 14, color=RED,
        align=PP_ALIGN.CENTER)

    # ── the two scenarios, side by side ──────────────────────────────────
    cw = (W - 0.55) / 2
    for i, (tag, joiners, waiting, rows) in enumerate([
            ("A", "10 % join in", "200 parcels would wait",
             [("deliver daily", "20 000 €", "0 €", "20 000 €", False),
              ("deliver on 2 days", "17 500 €", "2 000 €", "19 500 €", True)]),
            ("B", "20 % join in", "400 parcels would wait",
             [("deliver daily", "20 000 €", "0 €", "20 000 €", True),
              ("deliver on 2 days", "17 250 €", "4 000 €", "21 250 €", False)]),
    ]):
        x = L + i * (cw + 0.55)
        label_box(s, x, BODY_T + 1.10, 0.62, 0.52, RED,
                  [(tag, 22, True, WHITE)])
        txt(s, x + 0.80, BODY_T + 1.10, cw - 0.80, 0.34, joiners, 21,
            bold=True, color=INK)
        txt(s, x + 0.80, BODY_T + 1.44, cw - 0.80, 0.30, waiting, 16,
            color=DIM)
        for j, (name, drive, fee, total, wins) in enumerate(rows):
            _option(s, x, BODY_T + 1.86 + j * 0.98, cw, name, drive, fee,
                    total, wins=wins)

    # ── what actually changed between them ───────────────────────────────
    label_box(s, L, BODY_T + 4.00, W, 1.00, WHITE,
              [("Driving saving: 2 500 € → 2 750 €.   Fee number: "
                "2 000 € → 4 000 €.", 21, True, INK),
               ("One grew by a tenth, the other doubled. That is why the "
                "comparison flips.", 21, True, RED)], line_col=RED)
    return s
