"""The closing summary: what we did, what we found, what it means.

The deck ends on a list of managerial implications, which is an action list
rather than a conclusion — it tells the room what to do without first telling
them what was established. These two slides go in front of it: one that puts
the method, the result and the meaning side by side, and one that carries the
single sentence worth remembering.

Everything on them is a number the deck has already shown.

Imported by 96_explainers.py; not meant to be run on its own.
"""
from __future__ import annotations

from pptx.enum.text import PP_ALIGN

import _house as H
from _house import (BODY_T, DIM, GREEN, INK, INK2, L, LINE, PANEL, RED, S6,
                    SW, W, WHITE, hrule, hslide, label_box, rect, txt)

B = H.B

COLUMNS = [
    ("1", "What we did",
     "Solve the routing once. Learn from it. Then search millions of "
     "schedules for the price of one.",
     [("2.95 %", "surrogate error on postal-code areas it never saw"),
      ("39 × 312", "weekly patterns, over cells coupled through 22 depots"),
      ("4 points", "re-routed with the real solver as a check")]),
    ("2", "What we found",
     "13.5 to 18.5 % cheaper per week in the efficient range, at under half "
     "a day of added waiting.",
     [("25 % vs 9 %", "median saving, rural against urban areas"),
      ("−12.9 %", "peak fleet at P = 0.5, and 54 % less weekday variation"),
      ("+0.9…2.8 pp", "the real solver beat every prediction")]),
    ("3", "What it means",
     "Time is the lever where space is not. Target it, offer it, and scale it "
     "only when enough people take part.",
     [("the periphery", "not the network — the gap is structural"),
      ("opt-in, funded", "from the saving, never a downgrade of the default"),
      ("per carrier", "DHL tops out at 10.6 %, GLS at 33.4 %")]),
]


def slide_conclusion(prs):
    """Method, result and meaning, side by side."""
    s = hslide(prs, "Conclusion",
               "What we did, what we found, what it means",
               "All figures as reported earlier in this talk · Region "
               "Hannover, seven carriers, 1.26 M parcels a week against a "
               "1 909 748 € daily-delivery baseline.")
    cw = (W - 2 * 0.42) / 3
    # one shared height for the three claims, so the detail rows below them
    # sit on the same line across all three columns
    ch = max(H.text_height(c[2], cw, 20, 1.24) for c in COLUMNS) + 0.06
    for i, (num, head, claim, rows) in enumerate(COLUMNS):
        x = L + i * (cw + 0.42)
        rect(s, x, BODY_T + 0.10, cw, 0.09, RED)
        label_box(s, x, BODY_T + 0.32, 0.62, 0.62, RED,
                  [(num, 24, True, WHITE)])
        txt(s, x + 0.80, BODY_T + 0.38, cw - 0.80, 0.44, head, 24, bold=True,
            color=INK)
        txt(s, x, BODY_T + 1.10, cw, ch, claim, 20, bold=True, color=RED,
            line=1.24)
        for j, (key, note) in enumerate(rows):
            y = BODY_T + 1.10 + ch + 0.16 + j * 0.88
            txt(s, x, y, cw, 0.36, key, 22, bold=True, color=INK)
            txt(s, x, y + 0.36, cw, 0.52, note, 15, color=DIM, line=1.20)
    return s


def slide_takeaway(prs):
    """The one sentence, on red, with the number under it."""
    s = prs.slides.add_slide(prs.slide_layouts[H.LAYOUT_BLANK])
    rect(s, 0, 0, SW, H.SH, RED)
    hrule(s, 1.05, 2.30, 2.0, H.RGBColor(0xE4, 0x9A, 0xA8), 3.0)
    txt(s, 1.05, 2.55, 11.2, 1.75, "Batch where it is\nsparse and far.", 60,
        bold=True, color=WHITE, line=1.06)
    txt(s, 1.05, 4.58, 10.4, 0.95,
        "Temporal flexibility creates the density that non-urban delivery is "
        "missing — without a single new building.", 24,
        color=H.RGBColor(0xF7, 0xE0, 0xE5), line=1.32)
    hrule(s, 1.05, 5.62, 11.2, H.RGBColor(0xE4, 0x9A, 0xA8), 1.5)
    txt(s, 1.05, 5.82, 11.2, 0.50,
        "13.5 – 18.5 % of a weekly delivery bill, verified against real "
        "routing.", 24, bold=True, color=WHITE)
    return s
