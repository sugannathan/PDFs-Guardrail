import unicodedata
from typing import List, Dict, Any

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

def evaluate_span(span: Dict[str, Any], min_font_size: float = 2.0) -> List[str]:
    reasons = []

    if span["font_size"] < min_font_size:
        reasons.append(f"font_size < {min_font_size}pt")

    if is_near_white(span["rgb"]):
        reasons.append("white_or_near_white_text")

    if span["is_outside_page"]:
        reasons.append("off_page_text")

    if span["flags"] & 2:
        reasons.append("invisible_render_mode")

    reasons.extend(check_unicode_tricks(span["text"]))

    return reasons
