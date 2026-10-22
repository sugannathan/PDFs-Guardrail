import pymupdf as fitz
from typing import List, Dict, Any

def get_page_covering_rects(page: fitz.Page) -> List[fitz.Rect]:
    """
    Finds bounding boxes of solid filled vector drawings and images
    that could obscure underlying text.
    """
    covering_rects = []

    # 1. Vector drawings with fill color
    for draw in page.get_drawings():
        if draw.get("fill") is not None:  # Has solid fill color
            covering_rects.append(draw["rect"])

    # 2. Image bounding boxes
    for img_info in page.get_image_info(xrefs=True):
        bbox = img_info.get("bbox")
        if bbox:
            covering_rects.append(fitz.Rect(bbox))

    return covering_rects

def extract_spans(page: fitz.Page) -> List[Dict[str, Any]]:
    spans = []
    cropbox = page.cropbox
    mediabox = page.mediabox

    extract_rect = mediabox | cropbox
    extract_rect = extract_rect + (-200, -200, 200, 200)

    text_dict = page.get_text("dict", clip=extract_rect)

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

                r = (color_int >> 16) & 255
                g = (color_int >> 8) & 255
                b = color_int & 255

                is_outside = not cropbox.contains(bbox)

                spans.append({
                    "text": text,
                    "bbox": [bbox.x0, bbox.y0, bbox.x1, bbox.y1],
                    "font_size": span.get("size", 0.0),
                    "font_name": span.get("font", ""),
                    "flags": span.get("flags", 0),
                    "rgb": (r, g, b),
                    "is_outside_page": is_outside
                })
    return spans
