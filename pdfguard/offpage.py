import pymupdf as fitz
from typing import List, Dict, Any

SUSPICIOUS_KEYWORDS = [
    "system prompt", "ignore previous", "instructions",
    "reviewer", "groundbreaking", "excellent", "rating", "override"
]

def scan_offpage(doc: fitz.Document) -> List[Dict[str, Any]]:
    findings = []

    # 1. Metadata check
    metadata = doc.metadata or {}
    for key, val in metadata.items():
        if val and isinstance(val, str):
            val_lower = val.lower()
            if any(k in val_lower for k in SUSPICIOUS_KEYWORDS):
                findings.append({
                    "type": "metadata",
                    "source": f"metadata:{key}",
                    "text": val,
                    "reason": f"Suspicious keyword found in {key}"
                })

    # 2. Annotations check
    for page_idx, page in enumerate(doc):
        for annot in page.annots():
            info = annot.info or {}
            content = info.get("content", "")
            if content:
                findings.append({
                    "type": "annotation",
                    "page": page_idx + 1,
                    "source": "annotation:comment",
                    "text": content,
                    "reason": "Embedded annotation text found"
                })

    return findings
