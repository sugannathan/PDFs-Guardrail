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

    def check(self, pno: int, bbox) -> InkResult:
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
        ink = int((diff > DIFF_THRESHOLD).sum())
        area = int(diff.size)
        max_diff = int(diff.max())
        background = (np.median(b.reshape(-1, 3), axis=0) / 255.0).round(3).tolist()
        visible = ink >= MIN_INK_PIXELS and ink >= MIN_INK_RATIO * area
        return InkResult(ink, area, max_diff, background, visible,
                         visible and max_diff < LOW_CONTRAST_MAX_DIFF, round(clipped, 3))

    def crop_png(self, pno: int, bbox, pad: float = 2.0) -> bytes:
        """The rendered page area of a span, as a reader sees it (for OCR)."""
        page = self.doc[pno]
        rect = (pymupdf.Rect(bbox) + (-pad, -pad, pad, pad)) & page.rect
        if rect.is_empty:
            return b""
        pix = page.get_pixmap(matrix=pymupdf.Matrix(4, 4), clip=rect, annots=False, alpha=False)
        return pix.tobytes("png")
