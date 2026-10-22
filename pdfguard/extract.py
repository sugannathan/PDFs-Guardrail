import fitz
from typing import List, Dict, Any

def extract_spans(page: fitz.Page) -> List[Dict[str, Any]]:
    """
    Extracts all text spans from a PDF page along with coordinates,
    font sizes, colors, and flags.
    """
    spans = []
    text_dict = page.get_text("dict")
    cropbox = page.cropbox

    for block in text_dict.get("blocks", []):
        if "lines" not in block:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                text = span.get("text", "")
                if not text:
                    continue

                bbox = fitz.Rect(span["bbox"])
                color_int = span.get("color", 0)

                # Decode PyMuPDF sRGB integer into RGB (0-255)
                r = (color_int >> 16) & 255
                g = (color_int >> 8) & 255
                b = color_int & 255

                spans.append({
                    "text": text,
                    "bbox": [bbox.x0, bbox.y0, bbox.x1, bbox.y1],
                    "font_size": span.get("size", 0.0),
                    "font_name": span.get("font", ""),
                    "flags": span.get("flags", 0),
                    "rgb": (r, g, b),
                    "is_outside_page": not cropbox.contains(bbox)
                })
    return spans
