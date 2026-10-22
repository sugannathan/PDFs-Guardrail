"""Simple entry points for using PDFGuard inside other pipelines.

    import pdfguard

    text = pdfguard.safe_text("paper.pdf")          # visible text only; raises if dangerous
    if pdfguard.is_safe("paper.pdf"): ...
    report = pdfguard.scan("paper.pdf")              # full report
"""
from __future__ import annotations

from pathlib import Path

from .report import Report, scan

_RANK = {"clean": 0, "suspicious": 1, "dangerous": 2}


class HiddenContentError(Exception):
    """Raised by safe_text() when a document is riskier than allowed."""

    def __init__(self, report: Report):
        self.report = report
        found = [f for f in report.findings if f.label != "harmless"]
        preview = "; ".join(f"{f.label}: {(f.decoded or f.text)[:60]!r}" for f in found[:3])
        super().__init__(f"{report.file} is {report.risk} ({len(found)} finding(s)). {preview}")


def safe_text(path: str | Path, max_risk: str = "suspicious", use_llm: bool | None = False,
              use_ocr: bool = False) -> str:
    """Return only the text a human can see.

    Raises HiddenContentError if the document's risk is above `max_risk`
    ("clean", "suspicious" or "dangerous"). With the default, dangerous files are
    rejected and suspicious ones are passed through with hidden content removed.
    """
    if max_risk not in _RANK:
        raise ValueError(f"max_risk must be one of {list(_RANK)}")
    report = scan(path, use_llm=use_llm, use_ocr=use_ocr)
    if _RANK[report.risk] > _RANK[max_risk]:
        raise HiddenContentError(report)
    return report.clean_text


def is_safe(path: str | Path, use_llm: bool | None = False, use_ocr: bool = False) -> bool:
    """True if nothing dangerous or suspicious is hidden in the document."""
    return scan(path, use_llm=use_llm, use_ocr=use_ocr).risk == "clean"
