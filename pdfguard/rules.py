import unicodedata
import pymupdf as fitz
from typing import List, Dict, Any, Optional

ZERO_WIDTH_CHARS = {
    '\u200B', '\u200C', '\u200D', '\uFEFF', '\u200E', '\u200F',
    '\u202A', '\u202B', '\u202C', '\u202D', '\u202E'
}

def is_near_white(rgb: tuple, threshold: int = 245) -> bool:
    r, g, b = rgb
    return r >= threshold and g >= threshold and b >= threshold

def check_unicode_tricks(text: str) -> List[str]:
    reasons = []
    for ch in text:
        if ch in ZERO_WIDTH_CHARS or unicodedata.category(ch) in ['Cf', 'Zl', 'Zp']:
            reasons.append("zero_width_unicode")
            break
    return reasons

def is_span_covered(span_bbox: fitz.Rect, covering_rects: List[fitz.Rect]) -> bool:
    """
    Checks if a text span's area is >= 80% covered by any overlaying
    solid vector shape or image bounding box.
    """
    span_area = span_bbox.get_area()
    if span_area <= 0:
        return False

    for rect in covering_rects:
        intersect = span_bbox & rect
        if not intersect.is_empty:
            overlap_ratio = intersect.get_area() / span_area
            if overlap_ratio >= 0.8:
                return True
    return False

def evaluate_span(
    span: Dict[str, Any],
    min_font_size: float = 2.0,
    covering_rects: Optional[List[fitz.Rect]] = None
) -> List[str]:
    reasons = []

    # 1. Micro-font check (< 2pt)
    if span["font_size"] < min_font_size:
        reasons.append(f"font_size < {min_font_size}pt")

    # 2. White / Low-contrast Invisible text
    if is_near_white(span["rgb"]):
        reasons.append("white_or_near_white_text")

    # 3. Text placed outside the visible page
    if span["is_outside_page"]:
        reasons.append("off_page_text")

    # 4. Invisible render mode (mode 3)
    if span["flags"] & 2:
        reasons.append("invisible_render_mode")

    # 5. Unicode tricks (zero-width characters)
    reasons.extend(check_unicode_tricks(span["text"]))

    # 6. Covered text (drawn underneath images or opaque filled shapes)
    if covering_rects:
        span_bbox = fitz.Rect(span["bbox"])
        if is_span_covered(span_bbox, covering_rects):
            reasons.append("covered_text")

    return reasons
