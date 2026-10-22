"""Split a PDF into its content types.

Everything a PDF parser could hand to an LLM ends up in one of these parts:
text spans (with their drawing properties), images, vector drawings,
annotations, form fields, metadata (Info + XMP), embedded files, links,
JavaScript and optional content layers.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pymupdf


@dataclass
class TextSpan:
    page: int
    text: str
    bbox: list[float]
    font: str
    size: float
    color: list[float]          # RGB in 0..1
    opacity: float
    render_mode: int            # 0 = fill, 3 = invisible, 7 = clip only
    layer: str | None
    seqno: int


@dataclass
class ImagePart:
    page: int
    xref: int
    bbox: list[float]
    width: int
    height: int


@dataclass
class DrawingPart:
    page: int
    rect: list[float]
    fill: list[float] | None
    fill_opacity: float | None
    seqno: int


@dataclass
class AnnotationPart:
    page: int
    type: str
    rect: list[float]
    content: str
    title: str
    subject: str
    hidden: bool


@dataclass
class FormFieldPart:
    page: int
    name: str
    type: str
    value: str
    rect: list[float]
    hidden: bool


@dataclass
class LinkPart:
    page: int
    uri: str
    rect: list[float]


@dataclass
class EmbeddedFilePart:
    name: str
    filename: str
    size: int
    description: str
    text_preview: str


@dataclass
class PageInfo:
    page: int
    width: float
    height: float
    rotation: int


class PDFGuardError(ValueError):
    """The file cannot be checked: not a PDF, password-protected, or without pages."""


@dataclass
class DocumentParts:
    path: str
    pages: list[PageInfo] = field(default_factory=list)
    text_spans: list[TextSpan] = field(default_factory=list)
    images: list[ImagePart] = field(default_factory=list)
    drawings: list[DrawingPart] = field(default_factory=list)
    annotations: list[AnnotationPart] = field(default_factory=list)
    form_fields: list[FormFieldPart] = field(default_factory=list)
    links: list[LinkPart] = field(default_factory=list)
    embedded_files: list[EmbeddedFilePart] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    xmp: str = ""
    javascript: list[str] = field(default_factory=list)
    layers: dict = field(default_factory=dict)       # name -> on (bool)
    repaired: bool = False                            # file was damaged and had to be reconstructed

    def summary(self) -> dict:
        return {
            "pages": len(self.pages),
            "text_spans": len(self.text_spans),
            "images": len(self.images),
            "drawings": len(self.drawings),
            "annotations": len(self.annotations),
            "form_fields": len(self.form_fields),
            "links": len(self.links),
            "embedded_files": len(self.embedded_files),
            "metadata_fields": sum(1 for v in self.metadata.values() if v),
            "xmp_bytes": len(self.xmp),
            "javascript": len(self.javascript),
            "repaired": self.repaired,
            "layers": len(self.layers),
        }


def _rect(r) -> list[float]:
    return [round(float(v), 2) for v in r]


def _rgb(color) -> list[float]:
    if color is None:
        return [0.0, 0.0, 0.0]
    c = list(color)
    if len(c) == 1:              # gray
        return [c[0]] * 3
    if len(c) == 4:              # CMYK -> RGB
        cy, m, y, k = c
        return [(1 - cy) * (1 - k), (1 - m) * (1 - k), (1 - y) * (1 - k)]
    return c[:3]


def _layers(doc) -> dict:
    out = {}
    try:
        for info in doc.get_ocgs().values():
            out[info.get("name", "")] = bool(info.get("on", True))
    except Exception:
        pass
    return out


def _text_spans(page, pno: int) -> list[TextSpan]:
    spans = []
    for tr in page.get_texttrace():
        for text, bbox in _line_segments(tr):
            if not text.strip() and not re.search(r"[​-‏⁠-⁤﻿\U000e0000-\U000e007f]", text):
                continue
            spans.append(_make_span(tr, pno, text, bbox))
    return spans


def _line_segments(tr):
    """A trace span can run over several lines; cut it where the baseline jumps."""
    chars = tr["chars"]
    size = max(float(tr.get("size", 1)), 0.1)
    segments, cur = [], []
    for ch in chars:
        if cur:
            px, py = cur[-1][2]
            x, y = ch[2]
            if abs(y - py) > size * 0.5 or x < px - size:
                segments.append(cur)
                cur = []
        cur.append(ch)
    if cur:
        segments.append(cur)
    if len(segments) == 1:
        yield "".join(chr(c[0]) for c in chars), tr["bbox"]
        return
    for seg in segments:
        r = pymupdf.Rect(seg[0][3])
        for c in seg[1:]:
            r |= pymupdf.Rect(c[3])
        yield "".join(chr(c[0]) for c in seg), tuple(r)


def _make_span(tr, pno, text, bbox) -> TextSpan:
    return TextSpan(
            page=pno,
            text=text,
            bbox=_rect(bbox),
            font=tr.get("font", ""),
            size=round(float(tr.get("size", 0)), 3),
            color=[round(v, 3) for v in _rgb(tr.get("color"))],
            opacity=float(tr.get("opacity", 1.0)),
            render_mode=int(tr.get("type", 0)),
            layer=tr.get("layer") or None,
            seqno=int(tr.get("seqno", 0)),
    )


# PDF string literals "(...)" (one level of nesting) and hex strings "<...>"; removed before looking
# for names so that a URL such as https://github.com/JSR-... inside a /URI string is not mistaken for /JS.
_PDF_STRING = re.compile(r"\((?:[^()\\]|\\.|\((?:[^()\\]|\\.)*\))*\)|<[0-9A-Fa-f\s]*>")
_JS_NAME = re.compile(r"/(?:JS|JavaScript)(?=[\s/\[\]<>(){}%]|$)")
_JS_CODE = re.compile(r"/JS\s*\(((?:[^()\\]|\\.)*)\)")


def has_javascript_name(obj: str) -> bool:
    """True if the object text contains the name /JS or /JavaScript outside of string literals."""
    return bool(_JS_NAME.search(_PDF_STRING.sub("()", obj)))


def _javascript(doc) -> list[str]:
    found = []
    for xref in range(1, doc.xref_length()):
        try:
            obj = doc.xref_object(xref, compressed=False)
        except Exception:
            continue
        if has_javascript_name(obj):
            m = _JS_CODE.search(obj)
            found.append(f"xref {xref}: {m.group(1)[:300] if m else obj[:300]}")
    return found


def _decode_preview(data: bytes, limit: int = 2000) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            text = data[:limit].decode(enc)
            printable = sum(ch.isprintable() or ch.isspace() for ch in text)
            if text and printable / len(text) > 0.9:
                return text
        except Exception:
            pass
    return ""


def open_pdf(path: str | Path) -> "pymupdf.Document":
    """Open a file that must be a readable PDF, or raise PDFGuardError with a clear reason."""
    path = str(path)
    if not Path(path).is_file():
        raise PDFGuardError(f"no such file: {path}")
    try:
        doc = pymupdf.open(path, filetype="pdf")
    except Exception as e:
        raise PDFGuardError(f"not a readable PDF ({e})") from e
    if not doc.is_pdf:
        doc.close()
        raise PDFGuardError("not a PDF file")
    if doc.needs_pass:
        doc.close()
        raise PDFGuardError("password-protected PDF: it cannot be checked without the password")
    if doc.page_count == 0:
        doc.close()
        raise PDFGuardError("PDF has no pages (the file may be truncated or damaged)")
    return doc


def split_pdf(path: str | Path) -> DocumentParts:
    """Open a PDF and return all of its content, split by type."""
    path = str(path)
    doc = open_pdf(path)
    parts = DocumentParts(path=path, repaired=bool(doc.is_repaired))
    parts.metadata = {k: v for k, v in (doc.metadata or {}).items() if k not in ("format", "encryption")}
    try:
        parts.xmp = doc.get_xml_metadata() or ""
    except Exception:
        parts.xmp = ""
    parts.layers = _layers(doc)
    parts.javascript = _javascript(doc)
    # Other parsers (pypdf, pdfplumber, ...) extract text from switched-off layers,
    # so we switch every layer on for extraction. The renderer keeps the real state.
    try:
        for cfg in doc.layer_ui_configs():
            if not cfg.get("on"):
                doc.set_layer_ui_config(cfg["number"], action=0)   # 0 = switch on
    except Exception:
        pass

    for pno, page in enumerate(doc):
        parts.pages.append(PageInfo(pno, round(page.rect.width, 2), round(page.rect.height, 2), page.rotation))
        parts.text_spans.extend(_text_spans(page, pno))

        for info in page.get_image_info(xrefs=True):
            parts.images.append(ImagePart(pno, info.get("xref", 0), _rect(info["bbox"]),
                                          info.get("width", 0), info.get("height", 0)))

        for d in page.get_drawings():
            if d.get("fill") is not None:
                parts.drawings.append(DrawingPart(pno, _rect(d["rect"]), _rgb(d.get("fill")),
                                                  d.get("fill_opacity"), int(d.get("seqno", 0))))

        for a in page.annots() or []:
            info = a.info or {}
            hidden = bool(a.flags & (pymupdf.PDF_ANNOT_IS_HIDDEN | pymupdf.PDF_ANNOT_IS_NO_VIEW))
            parts.annotations.append(AnnotationPart(pno, a.type[1], _rect(a.rect), info.get("content", "") or "",
                                                    info.get("title", "") or "", info.get("subject", "") or "", hidden))

        for w in page.widgets() or []:
            hidden = getattr(w, "field_display", 0) in (1, 3)
            parts.form_fields.append(FormFieldPart(pno, w.field_name or "", w.field_type_string or "",
                                                   str(w.field_value or ""), _rect(w.rect), hidden))

        for link in page.get_links():
            uri = link.get("uri") or ""
            if uri:
                parts.links.append(LinkPart(pno, uri, _rect(link["from"])))

    for name in doc.embfile_names():
        info = doc.embfile_info(name)
        try:
            data = doc.embfile_get(name)
        except Exception:
            data = b""
        parts.embedded_files.append(EmbeddedFilePart(name, info.get("filename", name), int(info.get("size", len(data))),
                                                     info.get("description", "") or "", _decode_preview(data)))
    doc.close()
    return parts


def write_parts(parts: DocumentParts, out_dir: str | Path, extract_images: bool = False) -> Path:
    """Write each content type to its own file so it can be inspected separately."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "text_spans.jsonl", "w", encoding="utf-8") as f:
        for s in parts.text_spans:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
    for name in ("images", "drawings", "annotations", "form_fields", "links", "embedded_files", "pages"):
        (out / f"{name}.json").write_text(json.dumps([asdict(x) for x in getattr(parts, name)], indent=2, ensure_ascii=False))
    (out / "metadata.json").write_text(json.dumps(parts.metadata, indent=2, ensure_ascii=False))
    (out / "xmp.xml").write_text(parts.xmp, encoding="utf-8")
    (out / "javascript.json").write_text(json.dumps(parts.javascript, indent=2))
    (out / "layers.json").write_text(json.dumps(parts.layers, indent=2, ensure_ascii=False))
    (out / "summary.json").write_text(json.dumps(parts.summary(), indent=2))

    if extract_images and parts.images:
        img_dir = out / "images"
        img_dir.mkdir(exist_ok=True)
        doc = pymupdf.open(parts.path)
        seen = set()
        for im in parts.images:
            if im.xref and im.xref not in seen:
                seen.add(im.xref)
                try:
                    pix = pymupdf.Pixmap(doc, im.xref)
                    if pix.n - pix.alpha > 3:
                        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                    pix.save(img_dir / f"page{im.page}_xref{im.xref}.png")
                except Exception:
                    pass
        doc.close()
    return out
