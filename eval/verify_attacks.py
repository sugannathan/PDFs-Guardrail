"""Sanity-check an attacked test set produced by make_attacks.py.

For every attacked PDF it confirms two properties:

1. The payload reaches a real pipeline parser (pypdf for page text, or the
   metadata / annotation channel for those attacks).
2. The payload is not readable by a human, measured by rendering the visible
   page and comparing it with the clean original. A change of essentially zero
   pixels means invisible; the covered-text attack legitimately adds a box, and
   the tiny-font attack leaves an unreadable sub-pixel speck, both reported
   rather than failed.

Usage:
    python eval/verify_attacks.py data/clean data/attacked
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pymupdf
import pypdf

ZERO_WIDTH = {0x200B, 0x200C, 0x200D, 0xFEFF}
SENTINEL = "PDFGUARD-TEST"
DPI = 150
# Gray-level difference below this is imperceptible (anti-aliasing of
# background-matched glyphs), so it does not count as a visible change.
PERCEPTIBLE_LEVEL = 24
# Fraction of pixels allowed to change perceptibly and still count as hidden:
# absorbs a sub-point speck such as the 0.6 pt tiny-font attack.
HIDDEN_MAX_PX_FRAC = 0.001


def strip_zw(s: str) -> str:
    return "".join(c for c in s if ord(c) not in ZERO_WIDTH)


def payload_reaches_parser(path: Path, attack: str) -> bool:
    doc = pymupdf.open(path)
    try:
        if attack == "metadata":
            return SENTINEL in json.dumps(doc.metadata)
        if attack == "annotation":
            return any(SENTINEL in (a.info.get("content") or "")
                       for a in doc[0].annots())
    finally:
        doc.close()
    # All page-text attacks: use pypdf, a common pipeline parser that does not
    # clip to the cropbox, so it sees off-page text too.
    text = pypdf.PdfReader(str(path)).pages[0].extract_text()
    return SENTINEL in strip_zw(text)


def visible_change_fraction(clean: Path, attacked: Path) -> float:
    """Fraction of pixels that change perceptibly between the two renders.

    Rendered in grayscale; differences below PERCEPTIBLE_LEVEL (anti-aliasing of
    background-matched text) are ignored, so this measures what a human would
    actually notice rather than every byte.
    """
    cpix = pymupdf.open(clean)[0].get_pixmap(dpi=DPI, colorspace=pymupdf.csGRAY)
    apix = pymupdf.open(attacked)[0].get_pixmap(dpi=DPI, colorspace=pymupdf.csGRAY)
    if (cpix.width, cpix.height) != (apix.width, apix.height):
        return 1.0
    cs, as_ = cpix.samples, apix.samples
    changed = sum(1 for a, b in zip(cs, as_) if abs(a - b) > PERCEPTIBLE_LEVEL)
    total = len(cs) or 1
    return changed / total


def main() -> int:
    here = Path(__file__).resolve().parent.parent / "data"
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clean", nargs="?", default=str(here / "clean"),
                    help="directory of clean source PDFs")
    ap.add_argument("attacked", nargs="?", default=str(here / "attacked"),
                    help="directory of attacked PDFs + manifest.json")
    args = ap.parse_args()

    clean_dir = Path(args.clean)
    attacked_dir = Path(args.attacked)
    manifest = json.loads((attacked_dir / "manifest.json").read_text())

    agg = defaultdict(lambda: {"reach": 0, "hidden": 0, "total": 0})
    failures = []

    for row in manifest:
        attack = row["attack"]
        out = attacked_dir / row["output"]
        clean = clean_dir / row["source"]

        reached = payload_reaches_parser(out, attack)
        frac = visible_change_fraction(clean, out)

        if attack == "covered_text":
            # The opaque box is a deliberate, visible decoy; the payload text
            # sits underneath it and is not itself readable.
            hidden = frac > 0
        else:
            hidden = frac <= HIDDEN_MAX_PX_FRAC

        a = agg[attack]
        a["total"] += 1
        a["reach"] += reached
        a["hidden"] += hidden
        if not (reached and hidden):
            failures.append((row["output"], reached, hidden, round(frac, 5)))

    order = ["white_text", "tiny_font", "off_page", "covered_text",
             "invisible_render", "metadata", "annotation", "unicode"]
    print(f"{'attack':17}{'reaches parser':16}{'visually hidden':16}")
    print("-" * 49)
    ok = True
    for attack in order:
        if attack not in agg:
            continue
        a = agg[attack]
        print(f"{attack:17}{f'{a['reach']}/{a['total']}':16}"
              f"{f'{a['hidden']}/{a['total']}':16}")
        ok = ok and a["reach"] == a["total"] and a["hidden"] == a["total"]

    total = sum(a["total"] for a in agg.values())
    print(f"\n{total} files checked. ALL PASS: {ok}")
    if failures:
        print("\nfailures (file, reached, hidden, visible_change_frac):")
        for f in failures[:20]:
            print(" ", f)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
