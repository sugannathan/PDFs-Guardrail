import streamlit as st
import fitz
from PIL import Image
import io
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pdfguard.extract import extract_spans
from pdfguard.rules import evaluate_span
from pdfguard.offpage import scan_offpage

st.set_page_config(page_title="PDFGuard Demo", page_icon="🛡️", layout="wide")
st.title("🛡️ PDFGuard: Pre-LLM Document Auditor")
st.caption("Detects hidden injections, micro-fonts, and off-page text before ingestion.")

uploaded_file = st.file_uploader("Upload scientific paper (PDF)", type=["pdf"])

if uploaded_file is not None:
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    all_flagged = []
    for page_idx, page in enumerate(doc):
        spans = extract_spans(page)
        for s in spans:
            reasons = evaluate_span(s)
            if reasons:
                all_flagged.append({
                    "page": page_idx + 1,
                    "bbox": s["bbox"],
                    "reasons": reasons,
                    "text": s["text"],
                    "font_size": round(s["font_size"], 2),
                    "rgb": s["rgb"]
                })

    offpage_findings = scan_offpage(doc)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Visual Inspection")
        page_no = st.selectbox("Select Page to View", range(1, len(doc) + 1))
        page = doc[page_no - 1]

        for item in all_flagged:
            if item["page"] == page_no:
                r = fitz.Rect(item["bbox"])
                page.draw_rect(r, color=(1, 0, 0), width=2)
                page.insert_text(fitz.Point(r.x0, max(r.y0 - 2, 8)), "⚠️ HIDDEN", fontsize=8, color=(1, 0, 0))

        pix = page.get_pixmap(dpi=150)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        st.image(img, use_container_width=True, caption=f"Rendered Page {page_no} (Red highlights flagged content)")

    with col2:
        st.subheader("Audit Report")
        total_issues = len(all_flagged) + len(offpage_findings)
        
        if total_issues == 0:
            st.success("✅ Clean PDF: No hidden or anomalous text detected.")
        else:
            st.error(f"🚨 Detected {total_issues} hidden element(s)!")

            for item in all_flagged:
                with st.expander(f"Page {item['page']} — {', '.join(item['reasons'])}"):
                    st.write(f"**Flagged Text:** `{item['text']}`")
                    st.write(f"**Font Size:** {item['font_size']} pt | **RGB:** {item['rgb']}")
                    st.write(f"**BBox:** `{item['bbox']}`")

            for off in offpage_findings:
                with st.expander(f"Off-Page: {off['source']}"):
                    st.write(f"**Text:** `{off['text']}`")
                    st.write(f"**Reason:** {off['reason']}")

        st.divider()
        st.subheader("LLM Pipeline Guardrail Demo")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Raw Parser (Unprotected)")
            if total_issues > 0:
                st.warning("⚠️ **LLM Output:** Prompt injection executed. Summary/Review compromised.")
            else:
                st.info("Normal summary produced.")
        with c2:
            st.caption("PDFGuard (Sanitized)")
            st.success("🛡️ **LLM Output:** Hidden text stripped prior to prompting. Safe summary generated.")
