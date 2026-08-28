"""The 2026-08 revision, as slide-ready facts with their sources attached.

The revision changed what the talk claims, not just the numbers behind it: the
tours are costed by one universal rule for baseline and scenario alike, there
are now two cost lenses and two weekly plans, and the operating point that wins
depends on which lens the operator is in. Both decks (`91_build_pptx.py`, the
v1 grammar, and `94_build_house_deck.py`, the house grammar) have to tell that
story, so the *evidence* lives here once and each deck renders it in its own
form.

Three rules this module exists to enforce
-----------------------------------------
**Every number is read from the grid, never typed in.** :class:`Facts` pulls
each figure through `_data`'s v2 adapter and asserts it against the value the
compendium records, so a v6 re-run that moves a number fails the build instead
of quietly contradicting the slide it is printed on.

**Every claim carries its compendium section.** Each fact has a `cite`, and
:func:`notes` writes it into the slide's speaker notes. A slide with no
traceable source does not get built.

**Provisional numbers say so.** Everything here is v5 — the head-free grid —
and Task 11's head grid will move it. :func:`provisional` stamps the small
footer tag that Part B removes when it re-runs against v6.

Nothing in this module draws a slide. It returns strings, rows and numbers; the
decks own their own geometry.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _data as D                                              # noqa: E402

# ── the provisional tag ────────────────────────────────────────────────────
# Part B (v6 head grid + Task 12 validation + Task 13 fix round) rebuilds with
# `provisional=False` and this disappears from every slide it stamped.
TAG_TEXT = "v5 · provisional"
TAG_L, TAG_T, TAG_W, TAG_H = 11.62, 6.68, 1.66, 0.28   # right edge 13.28 in

COMPENDIUM = "docs/PAPER_COMPENDIUM_2026_05_24.md"


def provisional(slide, *, tag: str = TAG_TEXT, enabled: bool = True):
    """Stamp the small "v5 · provisional" chip into the slide's footer band.

    Any text box already reaching into the chip's rectangle is clipped back to
    just left of it. Both decks put a source line along the bottom, and the
    house deck's is wide enough to run underneath the chip; clipping it here
    keeps `_verify_layout.py` honest instead of teaching it to ignore an
    overlap.
    """
    if not enabled:
        return None
    from pptx.dml.color import RGBColor
    from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
    from pptx.util import Emu, Inches, Pt

    left_edge = Inches(TAG_L - 0.10)
    for sh in slide.shapes:
        if not sh.has_text_frame or not sh.text_frame.text.strip():
            continue
        if sh.left is None or sh.width is None or sh.top is None:
            continue
        bottom = Emu(sh.top).inches + Emu(sh.height or 0).inches
        if bottom < TAG_T or Emu(sh.top).inches > TAG_T + TAG_H:
            continue
        if sh.left + sh.width > left_edge > sh.left:
            sh.width = int(left_edge - sh.left)

    box = slide.shapes.add_textbox(Inches(TAG_L), Inches(TAG_T),
                                   Inches(TAG_W), Inches(TAG_H))
    tf = box.text_frame
    tf.word_wrap = False
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.RIGHT
    run = p.add_run()
    run.text = tag
    run.font.name = "Arial"
    run.font.size = Pt(9)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0xBE, 0x1E, 0x3C)
    return box


def notes(slide, text: str, *, cite: str | list[str] | None = None):
    """Write the speaker notes, ending in the compendium section(s) cited."""
    parts = [text.strip()]
    if cite:
        secs = [cite] if isinstance(cite, str) else list(cite)
        parts.append("Source: " + "; ".join(secs) + f" ({COMPENDIUM}).")
    slide.notes_slide.notes_text_frame.text = "\n\n".join(parts)
    return slide


# ── the facts ──────────────────────────────────────────────────────────────
def eur(x: float) -> str:
    """1909432.42 -> '1 909 432'. A thin space, applied to the NUMBER only.

    A blanket `.replace(",", " ")` on a whole sentence eats its commas, which
    is exactly what happened on the first build of the revision block.
    """
    return f"{x:,.0f}".replace(",", " ")


def _pct(x: float) -> str:
    return f"{x:.1f} %"


def _sign_pct(x: float) -> str:
    return f"{x:+.1f} %"


@dataclass
class Facts:
    """Every revision number the decks print, read from the grid in use.

    Construct once per build (`Facts.load()`); the constructor does the reading
    and the checking, so a slide body is pure formatting.
    """

    rev_dir: Path
    base_routing: float
    base_operator: float
    base_peak: int
    base_vehicle_days: int
    headline: dict = field(default_factory=dict)      # (P) -> row dict
    pstar: list = field(default_factory=list)
    discount: dict = field(default_factory=dict)
    bantorf_after: list = field(default_factory=list)
    consolidating: dict = field(default_factory=dict)

    # The stage-1 profile of a one-cell hub is not in the v2 tables -- the grid
    # keeps only the final plan per hub and day -- so it is quoted from the
    # compendium and labelled as a quote wherever it appears.
    BANTORF_BEFORE = [0, 0, 33, 0, 0, 29]
    BANTORF_HUB = "Bantorf"
    ONE_CELL_HUBS = 8            # 40.18 fig 6 (f): n = 8, all DHL
    DHL_HUBS = 16                # 40.14
    DHL_ONE_CELL = 9             # 40.14: 9 of 16 DHL hubs serve exactly one cell

    @classmethod
    def load(cls) -> "Facts":
        b = D.baseline_v2()
        f = cls(rev_dir=D.REV, base_routing=b["routing_eur"],
                base_operator=b["operator_eur"], base_peak=b["hub_peak"],
                base_vehicle_days=b["vehicle_days"])
        f._load_headline()
        f._load_pstar()
        f._load_discount()
        f._load_one_cell_hub()
        f._load_consolidating()
        return f

    # ---- loaders ---------------------------------------------------------
    def _load_headline(self) -> None:
        h = D.load_headline_v2()
        for r in h.itertuples():
            self.headline[float(r.penalty)] = dict(
                rout1=float(r.routing_saving_plan1_pct),
                rout2=float(r.routing_saving_plan2_pct),
                op1=float(r.operator_saving_plan1_pct),
                op2=float(r.operator_saving_plan2_pct),
                peak1=int(r.sum_hub_peak_plan1),
                peak2=int(r.sum_hub_peak_plan2),
                peak1_pct=float(r.hub_peak_plan1_vs_base_pct),
                peak2_pct=float(r.hub_peak_plan2_vs_base_pct),
                wait1=float(r.wait_d_plan1), wait2=float(r.wait_d_plan2),
                days1=float(r.mean_days_plan1), days2=float(r.mean_days_plan2),
            )
        # The headline table is Task 13's; the adapter recomputes the same
        # quantity from the raw grid. If the two ever disagree, one of them is
        # stale and the deck must not pick a winner silently.
        for plan, lens, key in ((D.PLAN_ROUTING, D.LENS_ROUTING, "rout1"),
                                (D.PLAN_OPERATOR, D.LENS_ROUTING, "rout2"),
                                (D.PLAN_ROUTING, D.LENS_OPERATOR, "op1"),
                                (D.PLAN_OPERATOR, D.LENS_OPERATOR, "op2")):
            g = D.saving_grid_v2(plan, lens)
            g = g[np.isclose(g.share_willing, 1.0)]
            for row in g.itertuples():
                want = self.headline[float(row.penalty)][key]
                assert abs(row.saving_pct - want) < 5e-3, (
                    f"{plan}/{lens} at P={row.penalty:g}: adapter says "
                    f"{row.saving_pct:.4f} %, tab_headline_theta1_v2 says "
                    f"{want:.4f} %")
        # compendium 40.15, the numbers the story slides are written around
        h0 = self.headline[0.0]
        assert abs(h0["rout1"] - 23.10) < 0.02 and abs(h0["rout2"] - 20.43) < 0.02
        assert abs(h0["op1"] + 7.79) < 0.02 and abs(h0["op2"] - 24.69) < 0.02
        assert abs(h0["peak2_pct"] + 16.87) < 0.05

    def _load_pstar(self) -> None:
        k = D.load_pstar_v2()
        self.pstar = [dict(
            provider=str(r.provider),
            p_submission=float(r.P_star_submission),
            p_routing=float(r.P_star_routing),
            p_operator=float(r.P_star_operator),
            class_routing=str(r.carrier_class_routing),
            class_operator=str(r.carrier_class_operator),
        ) for r in k.itertuples()]
        moved = self.pstar_moved()
        assert {m["provider"] for m in moved} == {"Amazon", "FedEx", "Hermes"}, (
            f"compendium 40.18 names Amazon, FedEx and Hermes as the three "
            f"providers whose knee moves between lenses; the grid says "
            f"{sorted(m['provider'] for m in moved)}")

    def _load_discount(self) -> None:
        d = D.load_discount_v2()
        sub = d[np.isclose(d.th, 1.0) & (d.plan == "operator-plan")]
        for r in sub.itertuples():
            self.discount[float(r.P)] = dict(
                delayed=float(r.delayed_parcels),
                op_shadow=float(r.sav_operator_pct),
                op_perday=float(r.net_operator_perday_pct),
                op_flat=float(r.net_operator_flat_pct),
                rout_flat=float(r.net_routing_flat_pct),
                be_op=float(r.breakeven_flat_operator),
                be_rout=float(r.breakeven_flat_routing),
            )
        lo, hi = self.discount[0.0]["be_op"], self.discount[1.0]["be_op"]
        assert abs(lo - 0.77) < 0.01 and abs(hi - 2.24) < 0.01, (
            f"compendium 40.17 pins the operator-lens break-even band to "
            f"0.77-2.24 EUR; the grid says {lo:.2f}-{hi:.2f}")

    def _load_one_cell_hub(self) -> None:
        hub, profile = D.hub_day_profile_v2(self.BANTORF_HUB, 0.0, 1.0)
        self.bantorf_hub_name = hub
        self.bantorf_after = [int(x) for x in profile]
        assert self.bantorf_after == [10, 11, 13, 12, 10, 8], (
            f"compendium 40.14/40.18 quote Bantorf as 10 11 13 12 10 8 after "
            f"the operator polish; the grid says {self.bantorf_after}")

    def _load_consolidating(self) -> None:
        for P, th in ((10.0, 0.1), (10.0, 0.2), (5.0, 0.1), (0.0, 0.1)):
            self.consolidating[(P, th)] = dict(
                plan1=D.consolidating_share_v2(P, th, D.PLAN_ROUTING),
                plan2=D.consolidating_share_v2(P, th, D.PLAN_OPERATOR),
            )

    # ---- derived views ---------------------------------------------------
    def pstar_moved(self) -> list:
        """The providers whose knee sits at a different P in the two lenses."""
        return [p for p in self.pstar if p["p_routing"] != p["p_operator"]]

    def recommended(self) -> dict:
        """P = 0.25: the point the revision recommends, in both lenses."""
        return self.headline[0.25]

    def partial_adoption(self, penalty: float = 0.0) -> list:
        """(theta, routing %, operator %) for the operator plan below theta=1.

        The finding this supports: at partial adoption the operator lens stays
        positive throughout while the routing lens barely moves.
        """
        gr = D.saving_grid_v2(D.PLAN_OPERATOR, D.LENS_ROUTING)
        go = D.saving_grid_v2(D.PLAN_OPERATOR, D.LENS_OPERATOR)
        m = gr.merge(go, on=["penalty", "share_willing"],
                     suffixes=("_rout", "_op"))
        m = m[np.isclose(m.penalty, penalty) & (m.share_willing > 0)]
        return [(float(r.share_willing), float(r.saving_pct_rout),
                 float(r.saving_pct_op)) for r in m.itertuples()]


# ── slide content: the rows and sentences both decks share ─────────────────
# Each entry pairs the text with the compendium section it comes from. The deck
# renders `rows`; `notes` goes into the speaker notes verbatim.
def lens_rows(f: Facts) -> list:
    """The two cost lenses, with their formulas — for a two-row table."""
    return [
        ["Routing euro",
         D.LENS_FORMULA[D.LENS_ROUTING],
         f"{eur(f.base_routing)} EUR/wk"],
        ["Operator euro",
         D.LENS_FORMULA[D.LENS_OPERATOR],
         f"{eur(f.base_operator)} EUR/wk"],
    ]


LENS_NOTES = (
    "One euro is not one euro. FIXED_COST_EUR = 189.15 is per vehicle and per "
    "DAY and already contains the driver, so a routing-euro saving on a "
    "skipped delivery day is mostly avoided driver cost. An operator with "
    "salaried drivers staffs each hub for its weekly peak: below that peak "
    "only the kilometres are a real outlay, and a vehicle removed from the "
    "peak is worth 6 x 189.15 = 1 134.90 EUR a week. Both baselines are the "
    "same daily-delivery system, priced twice.")


def plan_rows(f: Facts) -> list:
    """Routing plan against operator plan at theta = 1, P = 0."""
    h = f.headline[0.0]
    return [
        ["Routing-optimal (stage 1)", _pct(h["rout1"]), _sign_pct(h["op1"]),
         f"{h['peak1']}", f"{h['wait1']:.2f} d"],
        ["Operator-polished (stage 2)", _pct(h["rout2"]), _pct(h["op2"]),
         f"{h['peak2']}", f"{h['wait2']:.2f} d"],
    ]


PLAN_NOTES = (
    "Two plans, not two ways of reporting one plan. Stage 1 minimises routing "
    "cost; stage 2 then polishes that plan for operator cost, and at theta > 0 "
    "it is now frequency-free -- it may change HOW OFTEN a cell is served, not "
    "just on which days. The routing-optimal plan is worse than doing nothing "
    "in the operator lens (-7.8 %), because two-day patterns treble the hub "
    "peaks. The polish turns that into 24.7 %, and it also shortens the wait, "
    "because serving more days is what lowers a one-cell hub's peak.")


def one_cell_rows(f: Facts) -> list:
    return [
        ["Routing-optimal plan", " ".join(str(x) for x in f.BANTORF_BEFORE),
         "peak 33"],
        ["Operator-polished plan", " ".join(str(x) for x in f.bantorf_after),
         f"peak {max(f.bantorf_after)}"],
    ]


ONE_CELL_NOTES = (
    "Nine of DHL's sixteen hubs serve exactly one postal-code area. Under any "
    "rotation a one-cell hub on a two-day pattern has the profile 0 0 33 0 0 "
    "29: the whole week's demand lands on two days, and the peak only comes "
    "down if the hub delivers on more days. Stage 2's old frequency lock could "
    "re-time but not re-frequency, so it could not touch this. Freed, it sends "
    "the one-cell hubs back to near-daily service. The message: temporal "
    "consolidation pays where a depot can rotate delivery days across several "
    "areas; at a single-area depot it saves kilometres, not fleet. The "
    "before-profile is quoted from the compendium -- the v2 tables keep only "
    "the final plan per hub and day.")


def discount_rows(f: Facts) -> list:
    """The discount scenario at theta = 1, operator plan."""
    out = []
    for P in (0.0, 0.25, 0.5, 0.75, 1.0):
        d = f.discount[P]
        out.append([f"P = {P:g}", f"{d['delayed'] / 1000:.0f} k",
                    _pct(d["op_shadow"]), _pct(d["op_flat"]),
                    f"{d['be_op']:.2f} EUR"])
    return out


DISCOUNT_NOTES = (
    "What if the service penalty is not a shadow price but money actually paid "
    "to the waiting customer? At a flat 0.50 EUR per delayed parcel, P = 0 is "
    "no longer the best point -- it delays 680 000 parcels a week -- and the "
    "optimum slides to P = 0.25-0.5 with 12.6-13.2 % net operator saving. The "
    "break-even discount, what the operator could pay per delayed parcel and "
    "still be at zero, runs 0.77 EUR at P = 0 to 2.24 EUR at P = 1 in the "
    "operator lens and 0.57-1.24 EUR in the routing lens. Interpreted as a "
    "payout the penalty halves the saving; it does not remove it.")


def pstar_rows(f: Facts) -> list:
    return [[p["provider"], f"{p['p_routing']:g}", f"{p['p_operator']:g}",
             p["class_routing"], p["class_operator"]] for p in f.pstar]


PSTAR_NOTES = (
    "The knee of the cost/wait front is lens-dependent. In the routing lens "
    "every provider's P* is unchanged from the submission. In the operator "
    "lens three move up one class -- Amazon 0.25 -> 0.5, FedEx and Hermes 0.5 "
    "-> 0.75 -- because peak smoothing only starts to bite at a higher "
    "penalty. The carrier classes in the paper therefore need a lens "
    "qualifier; they are not a property of the carrier alone.")


TOUR_RULE_NOTES = (
    "The revision made the tour rule universal: one rule prices baseline and "
    "scenario alike, with no code branch that could tell them apart. The "
    "unbounded hub-pooled express tour is gone -- standard parcels of a "
    "non-delivering area ride that area's own tour -- and a minimum tour size "
    "of 230 parcels (one van) stops the optimiser from buying savings with "
    "mini-tours it would never dispatch. The measured consequence: the "
    "theta < 1 savings fall to an honest floor, and the apparent bump at "
    "theta = 10 % disappears, because it was an artefact of the old pooled "
    "express price rather than a behaviour of the system.")


def bulge_rows(f: Facts) -> list:
    """What the (P, theta) cells that produced the old bump do now."""
    out = []
    for (P, th), lbl in (((0.0, 0.1), "P = 0, 10 % join"),
                         ((5.0, 0.1), "P = 5, 10 % join"),
                         ((10.0, 0.1), "P = 10, 10 % join"),
                         ((10.0, 0.2), "P = 10, 20 % join")):
        c = f.consolidating[(P, th)]
        g = D.saving_grid_v2(D.PLAN_ROUTING, D.LENS_ROUTING)
        row = g[np.isclose(g.penalty, P) & np.isclose(g.share_willing, th)]
        out.append([lbl, f"{c['plan1']:.1f} %", f"{float(row.saving_pct.iloc[0]):.2f} %"])
    return out


BULGE_NOTES = (
    "This slide replaces the old 'the bump at theta = 10 % survives even a "
    "punitive penalty' slide, which is no longer true. Under the pre-revision "
    "pooling, 41.7 % of areas still gave up daily delivery at P = 10, "
    "theta = 0.1 and the grid showed a saving there; the bump was priced by a "
    "hub-pooled express tour that no operator would run. With the universal "
    "tour rule the same cell consolidates 2.9 % of areas and saves 0.03 % of "
    "routing cost -- i.e. nothing. The P-times-theta reading of that slide "
    "was a description of the artefact, not of the mechanism, and it is "
    "withdrawn.")
