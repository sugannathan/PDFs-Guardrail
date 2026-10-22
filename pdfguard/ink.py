"""Ink check: does a text span actually leave visible marks on the page?

For every page we render two images:
  1. the page as a reader sees it
  2. the same page with all text removed (images and drawings kept)

The difference between them is exactly the ink produced by text. A span whose
area shows no difference is invisible, whatever trick was used to hide it
(white text, covered by an image, invisible render mode, hidden layer, zero
opacity, ...). The no-text image also tells us the real background colour
behind each span.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pymupdf

DIFF_THRESHOLD = 30          # per-channel difference (0..255) that counts as ink
MIN_INK_PIXELS = 2
MIN_INK_RATIO = 0.003
LOW_CONTRAST_MAX_DIFF = 60


@dataclass
class InkResult:
    ink_pixels: int
    area_pixels: int
    max_diff: int
    background: list[float]     # median RGB (0..1) behind the span
    visible: bool
    low_contrast: bool
    clipped_area: float         # fraction of bbox inside the page
    undetermined: bool = False  # the span's area is almost entirely shared with other text


def _to_array(pix) -> np.ndarray:
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]


class PageRenderer:
    """Renders each page with and without text, cached per page."""

    def __init__(self, path: str, dpi: int = 110):
        self.path = path
        self.dpi = dpi
        self.scale = dpi / 72.0
        self.doc = pymupdf.open(path)
        self.notext_doc = pymupdf.open(path)
        self._cache: dict[int, tuple[np.ndarray, np.ndarray, pymupdf.Matrix]] = {}
        self._covers: dict[int, list[tuple[int, pymupdf.Rect]]] = {}

    def close(self):
        self.doc.close()
        self.notext_doc.close()

    def _render(self, pno: int):
        if pno in self._cache:
            return self._cache[pno]
        mat = pymupdf.Matrix(self.scale, self.scale)
        page = self.doc[pno]
        full = _to_array(page.get_pixmap(matrix=mat, annots=False, alpha=False, colorspace=pymupdf.csRGB))

        nt_page = self.notext_doc[pno]
        nt_page.add_redact_annot(nt_page.rect, fill=False)
        nt_page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE,
                                 graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
                                 text=pymupdf.PDF_REDACT_TEXT_REMOVE)
        notext = _to_array(nt_page.get_pixmap(matrix=mat, annots=False, alpha=False, colorspace=pymupdf.csRGB))

        if full.shape != notext.shape:   # should not happen, but be safe
            h, w = min(full.shape[0], notext.shape[0]), min(full.shape[1], notext.shape[1])
            full, notext = full[:h, :w], notext[:h, :w]
        to_pix = page.rotation_matrix * mat
        self._cache[pno] = (full, notext, to_pix)
        return self._cache[pno]

    def check(self, pno: int, bbox, exclude=()) -> InkResult:
        """Ink left by the text in `bbox`.

        `exclude` are the boxes of other text spans on the page: their ink is
        ignored, so a visible caption printed over hidden text cannot make the
        hidden text look visible.
        """
        full, notext, to_pix = self._render(pno)
        page_rect = self.doc[pno].rect
        rect = pymupdf.Rect(bbox)
        if rect.is_empty or rect.width <= 0 or rect.height <= 0:
            rect = pymupdf.Rect(rect.x0, rect.y0, rect.x0 + 1, rect.y0 + 1)
        inside = rect & page_rect
        clipped = 0.0 if inside.is_empty else (inside.width * inside.height) / max(rect.width * rect.height, 1e-6)
        if inside.is_empty:
            return InkResult(0, 0, 0, [1.0, 1.0, 1.0], False, False, 0.0)

        r = (inside * to_pix).irect
        h, w = full.shape[:2]
        x0, y0 = max(r.x0, 0), max(r.y0, 0)
        x1, y1 = min(max(r.x1, x0 + 1), w), min(max(r.y1, y0 + 1), h)
        a, b = full[y0:y1, x0:x1].astype(np.int16), notext[y0:y1, x0:x1].astype(np.int16)
        if a.size == 0:
            return InkResult(0, 0, 0, [1.0, 1.0, 1.0], False, False, clipped)

        diff = np.abs(a - b).max(axis=2)
        keep = np.ones(diff.shape, dtype=bool)
        for other in exclude:
            o = (pymupdf.Rect(other) & inside)
            if o.is_empty:
                continue
            orr = (o * to_pix).irect
            keep[max(orr.y0 - y0, 0):max(orr.y1 - y0, 0), max(orr.x0 - x0, 0):max(orr.x1 - x0, 0)] = False
        area = int(keep.sum())
        background = (np.median(b.reshape(-1, 3), axis=0) / 255.0).round(3).tolist()
        if area < 0.25 * diff.size:
            # Almost all of this span's area is shared with other text: the pixels cannot
            # be attributed. Treat as visible and leave the decision to the other rules.
            return InkResult(0, area, 0, background, True, False, round(clipped, 3), undetermined=True)
        diff = np.where(keep, diff, 0)
        ink = int((diff > DIFF_THRESHOLD).sum())
        max_diff = int(diff.max())
        visible = ink >= MIN_INK_PIXELS and ink >= MIN_INK_RATIO * area
        return InkResult(ink, area, max_diff, background, visible,
                         visible and max_diff < LOW_CONTRAST_MAX_DIFF, round(clipped, 3))

    def _opaque_covers(self, pno: int) -> list[tuple[int, pymupdf.Rect]]:
        """(paint order, rect) of opaque filled rectangles and opaque images on a page."""
        if pno in self._covers:
            return self._covers[pno]
        page = self.doc[pno]
        covers: list[tuple[int, pymupdf.Rect]] = []
        try:
            drawings = {d.get("seqno"): d for d in page.get_drawings()}
            log = page.get_bboxlog()
        except Exception:
            self._covers[pno] = covers
            return covers
        images = []
        try:
            for info in page.get_image_info(xrefs=True):
                xref = info.get("xref", 0)
                smask = False
                if xref:
                    try:
                        smask = self.doc.xref_get_key(xref, "SMask")[0] != "null"
                    except Exception:
                        smask = False
                images.append((pymupdf.Rect(info["bbox"]), smask))
        except Exception:
            pass
        for seq, (kind, box) in enumerate(log):
            rect = pymupdf.Rect(box)
            if kind == "fill-path":
                d = drawings.get(seq)
                if not d or d.get("fill") is None:
                    continue
                if (d.get("fill_opacity") if d.get("fill_opacity") is not None else 1.0) < 0.95:
                    continue
                if not all(item[0] in ("re", "qu") for item in d.get("items", [])):
                    continue          # only rectangles: a curve's bounding box over-states coverage
                covers.append((seq, rect))
            elif kind == "fill-image":
                transparent = any(abs(r.x0 - rect.x0) < 1 and abs(r.y0 - rect.y0) < 1 and sm for r, sm in images)
                if not transparent:
                    covers.append((seq, rect))
        self._covers[pno] = covers
        return covers

    def covers_after(self, pno: int, bbox, seqno: int) -> list[pymupdf.Rect]:
        """Opaque rectangles/images painted after the span that overlap it."""
        inside = pymupdf.Rect(bbox) & self.doc[pno].rect
        if inside.is_empty:
            return []
        return [r for seq, r in self._opaque_covers(pno) if seq > seqno and r.intersects(inside)]

    def covered_fraction(self, pno: int, bbox, seqno: int) -> float:
        """Share of the span (inside the page) painted over by opaque shapes/images drawn after it."""
        inside = pymupdf.Rect(bbox) & self.doc[pno].rect
        if inside.is_empty or inside.width <= 0 or inside.height <= 0:
            return 0.0
        later = self.covers_after(pno, bbox, seqno)
        if not later:
            return 0.0
        # rasterise the span at ~2 cells per point and mark covered cells
        nx, ny = max(int(inside.width * 2), 1), max(int(inside.height * 2), 1)
        xs = np.linspace(inside.x0, inside.x1, nx, endpoint=False) + inside.width / nx / 2
        ys = np.linspace(inside.y0, inside.y1, ny, endpoint=False) + inside.height / ny / 2
        mask = np.zeros((ny, nx), dtype=bool)
        for r in later:
            mask |= ((ys[:, None] >= r.y0) & (ys[:, None] <= r.y1)) & ((xs[None, :] >= r.x0) & (xs[None, :] <= r.x1))
        return float(mask.mean())

    def ink_under(self, pno: int, bbox, rects, exclude=()) -> InkResult:
        """Ink check restricted to the part of the span that lies under `rects`.

        If the text still leaves ink where a "covering" shape supposedly is, the shape
        does not really hide it (transparency group, clipping, blend mode, ...).
        """
        span = pymupdf.Rect(bbox) & self.doc[pno].rect
        if span.is_empty:
            return InkResult(0, 0, 0, [1.0, 1.0, 1.0], False, False, 0.0)
        union = None
        for r in rects:
            part = pymupdf.Rect(r) & span
            if not part.is_empty:
                union = part if union is None else union | part
        if union is None:
            return InkResult(0, 0, 0, [1.0, 1.0, 1.0], False, False, 0.0)
        return self.check(pno, union, exclude=exclude)

    def crop_png(self, pno: int, bbox, pad: float = 2.0) -> bytes:
        """The rendered page area of a span, as a reader sees it (for OCR)."""
        page = self.doc[pno]
        rect = (pymupdf.Rect(bbox) + (-pad, -pad, pad, pad)) & page.rect
        if rect.is_empty:
            return b""
        pix = page.get_pixmap(matrix=pymupdf.Matrix(4, 4), clip=rect, annots=False, alpha=False)
        return pix.tobytes("png")
