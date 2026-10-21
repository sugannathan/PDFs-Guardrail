"""Which PDF parsers pass hidden content on to the LLM?

PDFGuard finds the hidden snippets in each file; this script then checks
whether each snippet appears in the output of common parsers.

    python eval/parsers.py data/attacked --out results/parsers.csv

Parsers tested when installed: PyMuPDF, pymupdf4llm, pdfplumber, pypdf, Docling.
Note: most parsers ignore metadata, annotations and embedded files, but many
pipelines read those separately (e.g. title/abstract from metadata).
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pdfguard import scan  # noqa: E402


def _pymupdf(path):
    import pymupdf
    with pymupdf.open(path) as d:
        return "\n".join(p.get_text() for p in d)


def _pymupdf4llm(path):
    import pymupdf4llm
    return pymupdf4llm.to_markdown(str(path), show_progress=False)


def _pdfplumber(path):
    import pdfplumber
    with pdfplumber.open(path) as pdf:
        return "\n".join((p.extract_text() or "") for p in pdf.pages)


def _pypdf(path):
    import pypdf
    return "\n".join((p.extract_text() or "") for p in pypdf.PdfReader(str(path)).pages)


def _docling(path):
    from docling.document_converter import DocumentConverter
    return DocumentConverter().convert(str(path)).document.export_to_markdown()


PARSERS = {"PyMuPDF": _pymupdf, "pymupdf4llm": _pymupdf4llm, "pdfplumber": _pdfplumber,
           "pypdf": _pypdf, "Docling": _docling}


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", t.lower())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", default="results/parsers.csv")
    ap.add_argument("--parsers", default=",".join(PARSERS), help="comma-separated subset")
    a = ap.parse_args()

    modules = {"PyMuPDF": "pymupdf", "pymupdf4llm": "pymupdf4llm", "pdfplumber": "pdfplumber",
               "pypdf": "pypdf", "Docling": "docling"}
    parsers = {}
    for name in a.parsers.split(","):
        if importlib.util.find_spec(modules[name]) is None:
            print(f"skipping {name} (not installed)", file=sys.stderr)
            continue
        parsers[name] = PARSERS[name]

    rows = []
    for pdf in sorted(Path(a.folder).rglob("*.pdf")):
        report = scan(pdf, use_ocr=True)
        hidden = [f for f in report.findings if f.label != "harmless" and f.source == "text"]
        if not hidden:
            continue
        outputs = {}
        for name, fn in parsers.items():
            try:
                outputs[name] = _norm(fn(pdf))
            except Exception as e:
                outputs[name] = None
                print(f"{name} failed on {pdf.name}: {e}", file=sys.stderr)
        for f in hidden:
            probe = _norm(f.text)[:40]
            row = {"file": pdf.name, "reasons": ",".join(f.reasons), "snippet": f.text[:80]}
            for name, text in outputs.items():
                row[name] = "error" if text is None else ("PASSED" if probe and probe in text else "blocked")
            rows.append(row)

    if not rows:
        print("no hidden text found")
        return
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    names = list(parsers)
    print("| File | Hidden by | " + " | ".join(names) + " |")
    print("|---|---|" + "---|" * len(names))
    for r in rows:
        print(f"| {r['file']} | {r['reasons']} | " + " | ".join(r[n] for n in names) + " |")
    print(f"\nPASSED = hidden text reaches the LLM. Written to {a.out}")


if __name__ == "__main__":
    main()
