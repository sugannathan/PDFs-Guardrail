"""Hidden-content checks for every content type produced by split.py."""
from __future__ import annotations

import difflib
import io
import re
from dataclasses import dataclass, field

import pymupdf

from .ink import PageRenderer
from .split import DocumentParts, TextSpan

TINY_FONT_PT = 2.0
MIN_OPACITY = 0.05
BG_COLOR_DIST = 0.12          # RGB distance (0..1 per channel) treated as "same colour"

# Characters a reader cannot see. Tag characters (U+E0000..E007F) can encode a
# full ASCII message ("ASCII smuggling"), so we decode them.
INVISIBLE_RE = re.compile(r"[­᠎​-‏‪-‮⁠-⁤⁦-⁩﻿\U000e0000-\U000e007f]")
TAG_RE = re.compile(r"[\U000e0020-\U000e007e]")
BIDI_RE = re.compile(r"[‪-‮⁦-⁩]")

# Reasons that on their own mean "a reader cannot see this span".
HIDING_REASONS = {"no_ink", "off_page", "tiny_font", "invisible_render_mode", "transparent",
                  "hidden_layer", "color_matches_background", "covered"}
COVERED_MIN_FRACTION = 0.75


@dataclass
class Finding:
    source: str                          # text | annotation | form_field | metadata | xmp | embedded_file | link | javascript
    text: str
    reasons: list[str]
    page: int | None = None
    bbox: list[float] | None = None
    decoded: str | None = None           # e.g. message hidden in Unicode tag characters
    label: str = "unknown"               # injection | data | harmless | unknown
    confidence: float = 0.0
    contradicts_visible: bool = False
    explanation: str = ""
    method: str = "heuristic"
    details: dict = field(default_factory=dict)


@dataclass
class SpanVerdict:
    span: TextSpan
    reasons: list[str]

    @property
    def hidden(self) -> bool:
        return bool(HIDING_REASONS & set(self.reasons))


def decode_invisible(text: str) -> str:
    """Recover messages encoded in invisible characters.

    - Unicode tag characters map 1:1 to ASCII (U+E0041 -> 'A').
    - Runs of ZWSP/ZWNJ are a common binary encoding (U+200B = 0, U+200C = 1).
    """
    tags = "".join(chr(ord(c) - 0xE0000) for c in TAG_RE.findall(text))
    if tags:
        return tags
    # Binary encodings are an unbroken run of ZWSP/ZWNJ; scattered zero-width characters are not a message.
    for run in sorted(re.findall("[\u200b\u200c]{16,}", text), key=len, reverse=True):
        bits = "".join("0" if c == "\u200b" else "1" for c in run)
        msg = "".join(chr(int(bits[i:i + 8], 2)) for i in range(0, len(bits) - len(bits) % 8, 8))
        if msg and sum(c.isprintable() for c in msg) / len(msg) > 0.9 and re.search("[A-Za-z]{2}", msg):
            return msg
    return ""


def _color_dist(a, b) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


def check_spans(parts: DocumentParts, renderer: PageRenderer) -> list[SpanVerdict]:
    off_layers = {name for name, on in parts.layers.items() if not on}
    by_page: dict[int, list[TextSpan]] = {}
    for sp in parts.text_spans:
        by_page.setdefault(sp.page, []).append(sp)
    verdicts = []
    for s in parts.text_spans:
        reasons = []
        if s.size < TINY_FONT_PT:
            reasons.append("tiny_font")
        if s.render_mode in (3, 7):
            reasons.append("invisible_render_mode")
        if s.opacity < MIN_OPACITY:
            reasons.append("transparent")
        if s.layer and s.layer in off_layers:
            reasons.append("hidden_layer")

        me = pymupdf.Rect(s.bbox)
        others = [o.bbox for o in by_page.get(s.page, []) if o is not s and pymupdf.Rect(o.bbox).intersects(me)]
        ink = renderer.check(s.page, s.bbox, exclude=others)
        if renderer.covered_fraction(s.page, s.bbox, s.seqno) >= COVERED_MIN_FRACTION:
            reasons.append("covered")
        if ink.clipped_area < 0.5:
            reasons.append("off_page")
        elif not ink.visible:
            reasons.append("no_ink")
        elif ink.low_contrast:
            reasons.append("low_contrast")
        if ink.clipped_area >= 0.5 and _color_dist(s.color, ink.background) < BG_COLOR_DIST and s.render_mode == 0:
            reasons.append("color_matches_background")

        if INVISIBLE_RE.search(s.text):
            reasons.append("invisible_unicode")
        verdicts.append(SpanVerdict(s, reasons))
    return verdicts


def _merge_hidden(verdicts: list[SpanVerdict]) -> list[list[SpanVerdict]]:
    """Group neighbouring hidden spans into readable snippets."""
    hidden = sorted((v for v in verdicts if v.hidden or "low_contrast" in v.reasons),
                    key=lambda v: (v.span.page, v.span.seqno))
    groups: list[list[SpanVerdict]] = []
    for v in hidden:
        if groups:
            last = groups[-1][-1]
            same_kind = set(last.reasons) == set(v.reasons)
            close = abs(v.span.bbox[1] - last.span.bbox[1]) <= max(3 * max(v.span.size, 1), 8) or "off_page" in v.reasons
            if v.span.page == last.span.page and same_kind and close and v.span.seqno - last.span.seqno <= 3:
                groups[-1].append(v)
                continue
        groups.append([v])
    return groups


def _union(bboxes) -> list[float]:
    r = pymupdf.Rect(bboxes[0])
    for b in bboxes[1:]:
        r |= pymupdf.Rect(b)
    return [round(v, 2) for v in r]


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", INVISIBLE_RE.sub("", text).lower()).strip()


def _overlaps_image(parts: DocumentParts, page: int, bbox) -> bool:
    r = pymupdf.Rect(bbox)
    return any(im.page == page and r.intersects(pymupdf.Rect(im.bbox)) for im in parts.images)


def ocr_status() -> str:
    """'' if OCR works, otherwise what is missing."""
    try:
        import pytesseract
    except ImportError:
        return "pytesseract not installed (pip install pytesseract)"
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        return "tesseract program not found (brew install tesseract / sudo apt install tesseract-ocr / conda install -c conda-forge tesseract)"
    return ""


def _ocr(png: bytes) -> str | None:
    try:
        import pytesseract
        from PIL import Image
        return pytesseract.image_to_string(Image.open(io.BytesIO(png)), config="--psm 6")
    except Exception:
        return None


def text_findings(parts: DocumentParts, verdicts: list[SpanVerdict], renderer: PageRenderer,
                  use_ocr: bool = False) -> list[Finding]:
    findings = []
    for group in _merge_hidden(verdicts):
        raw = " ".join(v.span.text.strip() for v in group if v.span.text.strip())
        text = INVISIBLE_RE.sub("", raw)
        reasons = sorted(set(group[0].reasons))
        page = group[0].span.page
        bbox = _union([v.span.bbox for v in group])
        f = Finding("text", text, reasons, page, bbox, decoded=decode_invisible(raw) or None,
                    details={"font_size": group[0].span.size, "color": group[0].span.color,
                             "render_mode": group[0].span.render_mode, "spans": len(group)})
        if raw != text:
            f.details["invisible_chars_removed"] = len(raw) - len(text)
        # Invisible text lying on an image is either an OCR layer (harmless) or
        # text hidden under a picture. OCR of what the reader sees tells them apart.
        if use_ocr and ("no_ink" in reasons or "invisible_render_mode" in reasons) and _overlaps_image(parts, page, bbox):
            seen = _ocr(renderer.crop_png(page, bbox))
            if seen is None:
                f.details["ocr_error"] = ocr_status() or "OCR failed"
            else:
                ratio = difflib.SequenceMatcher(None, _norm(text), _norm(seen)).ratio()
                f.details["ocr_text"] = seen.strip()[:200]
                f.details["ocr_similarity"] = round(ratio, 2)
                if ratio >= 0.6:
                    f.reasons.append("matches_visible_image")
        findings.append(f)

    # Invisible Unicode inside otherwise visible text
    for v in verdicts:
        if "invisible_unicode" in v.reasons and not v.hidden:
            chars = INVISIBLE_RE.findall(v.span.text)
            decoded = decode_invisible(v.span.text)
            reasons = ["invisible_unicode"]
            if decoded:
                reasons.append("hidden_unicode_message")
            if BIDI_RE.search(v.span.text):
                reasons.append("bidi_override")
            findings.append(Finding("text", INVISIBLE_RE.sub("", v.span.text), reasons, v.span.page, v.span.bbox,
                                    decoded=decoded or None,
                                    details={"invisible_chars": [f"U+{ord(c):04X}" for c in chars[:20]],
                                             "count": len(chars)}))
    return findings


_XML_TEXT = re.compile(r">([^<>]{3,})<")
_STANDARD_META = {"title", "author", "creator", "producer", "creationDate", "modDate", "trapped"}


def offpage_findings(parts: DocumentParts, visible_norm: str) -> list[Finding]:
    """Content that is never drawn on the page but that many parsers pass on."""
    from .classify import looks_like_data, looks_like_instruction

    findings = []

    def interesting(text: str, long_limit: int = 200) -> bool:
        t = text.strip()
        if not t:
            return False
        if decode_invisible(t):
            return True
        if _norm(t) and _norm(t) in visible_norm:
            return False
        return looks_like_instruction(t) or (looks_like_data(t) and len(t) > 25) or len(t) > long_limit

    for key, value in parts.metadata.items():
        value = str(value or "")
        if interesting(value, 80 if key not in _STANDARD_META else 200):
            findings.append(Finding("metadata", value, [f"metadata:{key}"]))

    if parts.xmp:
        known = {_norm(str(v)) for v in parts.metadata.values() if v}
        for chunk in {c.strip() for c in _XML_TEXT.findall(parts.xmp)}:
            if _norm(chunk) not in known and interesting(chunk, 120):
                findings.append(Finding("xmp", chunk, ["xmp_metadata"]))

    for a in parts.annotations:
        text = " ".join(t for t in (a.content, a.subject) if t).strip()
        if not text:
            continue
        if a.type == "FreeText" and not a.hidden:
            continue                     # drawn on the page, a reader sees it
        reasons = ["annotation_content", f"annotation:{a.type}"] + (["hidden_annotation"] if a.hidden else [])
        findings.append(Finding("annotation", text, reasons, a.page, a.rect))

    for w in parts.form_fields:
        if w.value and (w.hidden or _norm(w.value) not in visible_norm):
            findings.append(Finding("form_field", w.value, ["form_field_value"] + (["hidden_field"] if w.hidden else []),
                                    w.page, w.rect, details={"name": w.name}))

    for e in parts.embedded_files:
        findings.append(Finding("embedded_file", e.text_preview or f"<binary file {e.filename}, {e.size} bytes>",
                                ["embedded_file"], details={"name": e.name, "description": e.description}))

    for link in parts.links:
        if link.uri.lower().startswith("javascript:"):
            findings.append(Finding("link", link.uri, ["javascript_link"], link.page, link.rect))

    for js in parts.javascript:
        findings.append(Finding("javascript", js, ["javascript"]))

    for f in findings:
        msg = decode_invisible(f.text)
        if msg:
            f.decoded = msg
            f.text = INVISIBLE_RE.sub("", f.text)
            f.reasons.append("hidden_unicode_message")
    return findings
