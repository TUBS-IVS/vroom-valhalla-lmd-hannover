"""Lift the stock icon set out of the author's own decks.

His slides mark each bullet with a white circle carrying a PowerPoint stock
icon. Those icons ship inside the .pptx as an SVG plus a PNG fallback; the SVG
carries the icon's name in its root ``id`` (``Icons_Truck``), which is what
makes them addressable here. This script pairs the two through the slide
relationships, keeps the PNG, and names it after the SVG.

Run it once; ``_house.icon_path()`` reads the result and tints on demand.

    python scripts/presentation/_extract_icons.py
"""
from __future__ import annotations

import io
import os
import re
import zipfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "presentation_2026_08" / "icons"

DECKS = [
    Path(r"C:/Users/bienzeisler/Documents/Präsentationen/Berlin Freight 2025"
         r"/FreightSim_Bienzeisler.pptx"),
    Path(r"C:/Users/bienzeisler/Documents/Präsentationen/mobilTUM 2025"
         r"/mobilTUM_25_Bienzeisler.pptx"),
    Path(r"C:/Users/bienzeisler/Documents/Präsentationen/Universitätstagung"
         r"/2023/Universitätstagung_2023_Bienzeisler.pptx"),
    Path(r"C:/Users/bienzeisler/Documents/Promotion/Disputation_Bienzeisler.pptx"),
]

# One <a:blip> holds the raster and, in an extLst, the SVG it falls back from.
# The match must not run past the end of that element: a picture without an SVG
# would otherwise pair its PNG with the *next* picture's SVG, which silently
# mislabels the icons. `(?:(?!<a:blip).)*?` keeps the match inside one blip.
# The svgBlip element also carries an xmlns declaration between its tag name
# and r:embed, so those two attributes are not adjacent.
_PAIR = re.compile(
    r'<a:blip r:embed="(rId\d+)"(?:(?!<a:blip).)*?svgBlip[^>]*r:embed="(rId\d+)"',
    re.S)
_NAME = re.compile(r'id="Icons_([A-Za-z0-9_]+)"')
_REL = re.compile(r'Id="(rId\d+)"[^>]*Target="([^"]+)"')


def extract(decks=DECKS, out: Path = OUT) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    found: dict[str, bytes] = {}
    for deck in decks:
        if not deck.exists():
            print(f"  skip (missing) {deck.name}")
            continue
        z = zipfile.ZipFile(deck)
        for slide in [n for n in z.namelist()
                      if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)]:
            rels = z.read(
                f"ppt/slides/_rels/{os.path.basename(slide)}.rels").decode("utf8")
            rmap = dict(_REL.findall(rels))
            for png_r, svg_r in _PAIR.findall(z.read(slide).decode("utf8")):
                if png_r not in rmap or svg_r not in rmap:
                    continue
                svg_t = "ppt/" + rmap[svg_r].replace("../", "")
                png_t = "ppt/" + rmap[png_r].replace("../", "")
                if not png_t.lower().endswith(".png"):
                    continue
                try:
                    name = _NAME.search(z.read(svg_t).decode("utf8", "replace"))
                except KeyError:
                    continue
                if name and name.group(1) not in found:
                    found[name.group(1)] = z.read(png_t)
    written = {}
    for name, data in sorted(found.items()):
        try:
            im = Image.open(io.BytesIO(data)).convert("RGBA")
        except Exception as exc:                      # noqa: BLE001
            print(f"  skip (unreadable) {name}: {exc}")
            continue
        p = out / f"{name}.png"
        im.save(p)
        written[name] = p
    return written


if __name__ == "__main__":
    got = extract()
    print(f"{len(got)} icons -> {OUT.relative_to(ROOT)}")
    print("  " + ", ".join(sorted(got)))
