"""Run the full pipeline on one PDF and build the report."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pymupdf

from .checks import INVISIBLE_RE, Finding, check_spans, offpage_findings, text_findings, _norm
from .classify import classify
from .ink import PageRenderer
from .split import DocumentParts, split_pdf


@dataclass
class Report:
    file: str
    risk: str
    counts: dict
    parts: dict
    findings: list[Finding] = field(default_factory=list)
    clean_text: str = ""
    raw_text: str = ""

    def to_dict(self, include_text: bool = False) -> dict:
        d = {"file": self.file, "risk": self.risk, "counts": self.counts, "parts": self.parts,
             "findings": [asdict(f) for f in self.findings]}
        if include_text:
            d["clean_text"] = self.clean_text
        return d

    def to_json(self, include_text: bool = False) -> str:
        return json.dumps(self.to_dict(include_text), indent=2, ensure_ascii=False)


def _risk(findings: list[Finding]) -> str:
    if any(f.label == "injection" or (f.label == "data" and f.contradicts_visible) or f.decoded
           or f.method == "heuristic+llm_disagrees" for f in findings):
        return "dangerous"
    if any(f.label in ("data", "unknown") for f in findings):
        return "suspicious"
    return "clean"


def _reading_text(spans) -> str:
    """Join spans in reading order: by page, then line (rounded y), then x."""
    lines: dict[tuple[int, int], list] = {}
    for s in spans:
        key = (s.page, round((s.bbox[1] + s.bbox[3]) / 2 / 3))
        lines.setdefault(key, []).append(s)
    out, last_page = [], None
    for (page, _), items in sorted(lines.items()):
        if last_page is not None and page != last_page:
            out.append("\f")
        last_page = page
        items.sort(key=lambda s: s.bbox[0])
        out.append(" ".join(INVISIBLE_RE.sub("", s.text).strip() for s in items if s.text.strip()))
    return "\n".join(line for line in out if line)


def _visible_page_texts(doc, hidden_spans) -> dict[int, str]:
    """Text a human can see, per page, in reading order.

    Uses PyMuPDF's word extraction (it restores spaces and column order) on the
    document with its real layer state, and drops every word that belongs to a
    hidden span.
    """
    hidden_by_page: dict[int, list] = {}
    for s in hidden_spans:
        hidden_by_page.setdefault(s.page, []).append((pymupdf.Rect(s.bbox), _norm(s.text)))
    out = {}
    for pno, page in enumerate(doc):
        lines: dict[tuple[int, int], list[str]] = {}
        order: list[tuple[int, int]] = []
        for x0, y0, x1, y1, word, block, line, _ in page.get_text("words", sort=False):
            r = pymupdf.Rect(x0, y0, x1, y1)
            if not r.intersects(page.rect):
                continue
            c = (r.tl + r.br) / 2
            w = _norm(word)
            if any(c in hb and (not w or w in ht) for hb, ht in hidden_by_page.get(pno, [])):
                continue
            key = (block, line)
            if key not in lines:
                lines[key] = []
                order.append(key)
            lines[key].append(INVISIBLE_RE.sub("", word))
        out[pno] = "\n".join(" ".join(lines[k]) for k in order if lines[k])
    return out


def scan(path: str | Path, use_llm: bool | None = None, use_ocr: bool = False, dpi: int = 110,
         parts: DocumentParts | None = None) -> Report:
    parts = parts or split_pdf(path)
    renderer = PageRenderer(str(path), dpi=dpi)
    try:
        verdicts = check_spans(parts, renderer)
        page_texts = _visible_page_texts(renderer.doc, [v.span for v in verdicts if v.hidden])
        visible_text = "\n\f\n".join(page_texts[p] for p in sorted(page_texts))

        findings = text_findings(parts, verdicts, renderer, use_ocr=use_ocr)
        findings += offpage_findings(parts, _norm(visible_text))
        classify(findings, visible_text, page_texts, use_llm=use_llm)
    finally:
        renderer.close()

    counts = {label: sum(f.label == label for f in findings) for label in ("injection", "data", "unknown", "harmless")}
    counts["hidden_spans"] = sum(v.hidden for v in verdicts)
    counts["total_spans"] = len(verdicts)
    return Report(file=str(path), risk=_risk(findings), counts=counts, parts=parts.summary(), findings=findings,
                  clean_text=visible_text, raw_text=_reading_text(parts.text_spans))


def format_report(r: Report) -> str:
    icon = {"clean": "OK  ", "suspicious": "WARN", "dangerous": "FAIL"}[r.risk]
    lines = [f"[{icon}] {r.file}  risk={r.risk}  "
             f"(injection={r.counts['injection']}, data={r.counts['data']}, unknown={r.counts['unknown']}, "
             f"harmless={r.counts['harmless']}, hidden spans={r.counts['hidden_spans']}/{r.counts['total_spans']})"]
    for f in r.findings:
        where = f"p{f.page + 1}" if f.page is not None else f.source
        text = (f.decoded and f"[decoded] {f.decoded}") or f.text
        text = text.replace("\n", " ")
        tag = " [llm]" if f.method == "llm" else (" [llm disagrees]" if "disagrees" in f.method else "")
        lines.append(f"   - {f.label:<9} {where:<6} {','.join(f.reasons):<45} {text[:90]!r}{tag}")
        if f.explanation and f.label != "harmless":
            lines.append(f"     {f.explanation}")
        if "llm_error" in f.details:
            lines.append(f"     (LLM not used: {f.details['llm_error'][:120]})")
    return "\n".join(lines)
