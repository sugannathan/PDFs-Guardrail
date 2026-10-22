"""Which PDF parsers leak hidden content into an LLM pipeline?

For every attacked PDF in the manifest, this extracts text (or the relevant
channel) with each available parser and checks whether the hidden
[PDFGUARD-TEST] payload survives. A high rate means that parser would hand the
hidden text to a model. This fills the "which parsers pass hidden content"
table in the top-level README.

Parsers: PyMuPDF, pdfplumber, pypdf, and Docling if installed. Zero-width
characters are stripped before matching so the unicode attack is scored on its
text content, not on exact bytes.

Usage:
    python eval/parser_leak.py                       # uses data/attacked + manifest
    python eval/parser_leak.py --attacked data/attacked --csv reports/parsers.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import pymupdf

ZERO_WIDTH = {0x200B, 0x200C, 0x200D, 0xFEFF}
SENTINEL = "PDFGUARD-TEST"
DATA = Path(__file__).resolve().parent.parent / "data"

ATTACK_ORDER = ["white_text", "tiny_font", "off_page", "covered_text",
                "invisible_render", "metadata", "annotation", "unicode"]


def strip_zw(s: str) -> str:
    return "".join(c for c in s if ord(c) not in ZERO_WIDTH)


# ---- per-parser text extraction. Each returns the full text it would feed a
# model, or "" if the parser is unavailable for that file. --------------------

def extract_pymupdf(path: Path) -> str:
    doc = pymupdf.open(path)
    parts = []
    for page in doc:
        # Clip larger than the page so off-page text is included, matching what
        # a careful extractor should do.
        big = pymupdf.Rect(-2000, -2000, page.rect.width + 2000,
                           page.rect.height + 2000)
        parts.append(page.get_text(clip=big))
    # Metadata and annotations are part of what a pipeline may ingest.
    parts.append(json.dumps(doc.metadata or {}))
    for page in doc:
        for annot in page.annots() or []:
            parts.append(annot.info.get("content") or "")
    doc.close()
    return "\n".join(parts)


def extract_pypdf(path: Path) -> str:
    import pypdf
    reader = pypdf.PdfReader(str(path))
    parts = [page.extract_text() or "" for page in reader.pages]
    meta = reader.metadata or {}
    parts.append(" ".join(str(v) for v in meta.values()))
    return "\n".join(parts)


def extract_pdfplumber(path: Path) -> str:
    import pdfplumber
    parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
            for annot in page.annots or []:
                parts.append(str(annot.get("contents") or ""))
        parts.append(json.dumps(pdf.metadata or {}))
    return "\n".join(parts)


def extract_docling(path: Path) -> str:
    from docling.document_converter import DocumentConverter
    conv = DocumentConverter()
    return conv.convert(str(path)).document.export_to_markdown()


PARSERS = {
    "pymupdf": extract_pymupdf,
    "pdfplumber": extract_pdfplumber,
    "pypdf": extract_pypdf,
    "docling": extract_docling,
}


def available_parsers() -> dict:
    out = {}
    for name, fn in PARSERS.items():
        mod = {"pymupdf": "pymupdf", "pdfplumber": "pdfplumber",
               "pypdf": "pypdf", "docling": "docling"}[name]
        try:
            __import__(mod)
            out[name] = fn
        except Exception:
            pass
    return out


def leaks(fn, path: Path) -> bool:
    try:
        return SENTINEL in strip_zw(fn(path))
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attacked", default=str(DATA / "attacked"),
                    help="directory of attacked PDFs + manifest.json")
    ap.add_argument("--csv", default=None, help="optional path to write a CSV")
    args = ap.parse_args()

    attacked = Path(args.attacked)
    manifest = json.loads((attacked / "manifest.json").read_text())
    parsers = available_parsers()
    if not parsers:
        print("no parsers available", file=sys.stderr)
        return 1

    # counts[attack][parser] = [leaked, total]
    counts = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for row in manifest:
        path = attacked / row["output"]
        for pname, fn in parsers.items():
            c = counts[row["attack"]][pname]
            c[0] += leaks(fn, path)
            c[1] += 1

    pnames = list(parsers)
    head = f"{'attack':18}" + "".join(f"{p:13}" for p in pnames)
    print(head)
    print("-" * len(head))
    rows_out = []
    for attack in ATTACK_ORDER:
        if attack not in counts:
            continue
        cells = []
        for p in pnames:
            leaked, total = counts[attack][p]
            pct = 100.0 * leaked / total if total else 0.0
            cells.append(f"{leaked}/{total} ({pct:.0f}%)")
            rows_out.append({"attack": attack, "parser": p,
                             "leaked": leaked, "total": total,
                             "leak_rate": round(pct, 1)})
        print(f"{attack:18}" + "".join(f"{c:13}" for c in cells))

    print(f"\nparsers tested: {', '.join(pnames)}")
    missing = [p for p in PARSERS if p not in parsers]
    if missing:
        print(f"not installed (skipped): {', '.join(missing)}")
    print("A higher leak rate means that parser hands more hidden text to an LLM.")

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["attack", "parser", "leaked",
                                              "total", "leak_rate"])
            w.writeheader()
            w.writerows(rows_out)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
