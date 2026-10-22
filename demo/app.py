# -*- coding: utf-8 -*-
"""PDFGuard Streamlit Demo – powered by the unified scan() pipeline."""
from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path

import pymupdf
import requests
import streamlit as st
from PIL import Image, ImageDraw

# ── ensure project root is importable ──────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pdfguard import scan  # noqa: E402

# ── constants ──────────────────────────────────────────────────────────
DPI = 110
SCALE = DPI / 72.0

EXAMPLES_DIR = ROOT / "examples"
ATTACKED_DIR = ROOT / "data" / "attacked"
REAL_ARXIV_DIR = ROOT / "data" / "real_arxiv"

REAL_ARXIV_IDS = [
    "2408.13940v3",
    "2501.13461v1",
    "2407.16803v3",
]

RISK_COLORS = {
    "clean":      ("#16a34a", "#dcfce7", "✅"),   # green
    "suspicious": ("#d97706", "#fef3c7", "⚠️"),   # amber
    "dangerous":  ("#dc2626", "#fee2e2", "🚨"),   # red
}

DEFAULT_EXAMPLE = "26_latex_tiny_white_prompt.pdf"

# ── page config ────────────────────────────────────────────────────────
st.set_page_config(page_title="PDFGuard Demo", page_icon="🛡️", layout="wide")

# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════
st.sidebar.image(
    "https://img.icons8.com/fluency/96/shield.png",
    width=64,
)
st.sidebar.title("🛡️ PDFGuard")
st.sidebar.caption("Pre-LLM Document Auditor")
st.sidebar.divider()

# ── scan options ───────────────────────────────────────────────────────
use_llm = st.sidebar.checkbox("🤖 Enable LLM reasoning (Stage 2)", value=True)
show_harmless = st.sidebar.checkbox("👁️ Show harmless findings", value=False)
st.sidebar.divider()

# ── document source selector ──────────────────────────────────────────
st.sidebar.subheader("📂 Document Source")
source = st.sidebar.radio(
    "Choose a source",
    [
        "Technique Gallery (examples/)",
        "Synthetic Attack Benchmark",
        "Real-World arXiv Preprints (Lin et al. 2025)",
        "Live arXiv Fetch",
        "Upload Custom PDF",
    ],
    index=0,
    label_visibility="collapsed",
)

pdf_bytes: bytes | None = None
source_label: str = ""

if source == "Technique Gallery (examples/)":
    files = sorted(f.name for f in EXAMPLES_DIR.glob("*.pdf")) if EXAMPLES_DIR.is_dir() else []
    if not files:
        st.sidebar.warning("No example PDFs found in `examples/`.")
    else:
        default_idx = files.index(DEFAULT_EXAMPLE) if DEFAULT_EXAMPLE in files else 0
        chosen = st.sidebar.selectbox("Select example PDF", files, index=default_idx)
        pdf_bytes = (EXAMPLES_DIR / chosen).read_bytes()
        source_label = f"Technique Gallery — {chosen}"

elif source == "Synthetic Attack Benchmark":
    st.sidebar.caption("_Synthetic attack inserted by our generator._")
    files = sorted(f.name for f in ATTACKED_DIR.glob("*.pdf")) if ATTACKED_DIR.is_dir() else []
    if not files:
        st.sidebar.warning("No attacked PDFs found in `data/attacked/`.")
    else:
        chosen = st.sidebar.selectbox("Select attacked PDF", files)
        pdf_bytes = (ATTACKED_DIR / chosen).read_bytes()
        source_label = f"Synthetic Attack — {chosen}"

elif source == "Real-World arXiv Preprints (Lin et al. 2025)":
    st.sidebar.caption("_Real arXiv preprint, documented in Lin 2025, unmodified._")
    if REAL_ARXIV_DIR.is_dir():
        files = sorted(f.name for f in REAL_ARXIV_DIR.glob("*.pdf"))
    else:
        files = []
    # Also offer known IDs that can be fetched live
    all_ids = list({f.replace(".pdf", "") for f in files} | set(REAL_ARXIV_IDS))
    all_ids.sort()
    chosen_id = st.sidebar.selectbox("Select preprint", all_ids)
    local = REAL_ARXIV_DIR / f"{chosen_id}.pdf" if REAL_ARXIV_DIR.is_dir() else None
    if local and local.exists():
        pdf_bytes = local.read_bytes()
    else:
        st.sidebar.info(f"Fetching `{chosen_id}` from arXiv …")
        try:
            resp = requests.get(f"https://arxiv.org/pdf/{chosen_id}.pdf", timeout=30)
            resp.raise_for_status()
            pdf_bytes = resp.content
        except Exception as exc:
            st.sidebar.error(f"Download failed: {exc}")
    source_label = f"arXiv Preprint — {chosen_id}"

elif source == "Live arXiv Fetch":
    arxiv_id = st.sidebar.text_input("arXiv ID (e.g. `2408.13940v3`)")
    if st.sidebar.button("📥 Fetch PDF") and arxiv_id.strip():
        with st.spinner(f"Downloading arXiv:{arxiv_id.strip()} …"):
            try:
                resp = requests.get(
                    f"https://arxiv.org/pdf/{arxiv_id.strip()}.pdf", timeout=30,
                )
                resp.raise_for_status()
                pdf_bytes = resp.content
                source_label = f"arXiv Live — {arxiv_id.strip()}"
            except Exception as exc:
                st.sidebar.error(f"Download failed: {exc}")

elif source == "Upload Custom PDF":
    uploaded = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded is not None:
        pdf_bytes = uploaded.read()
        source_label = f"Upload — {uploaded.name}"

# ══════════════════════════════════════════════════════════════════════
#  MAIN AREA – header
# ══════════════════════════════════════════════════════════════════════
st.title("🛡️ PDFGuard: Pre-LLM Document Auditor")
st.caption(
    "Detects hidden prompt injections, exfiltration payloads, and anomalous "
    "content *before* your LLM ingests the document."
)

if pdf_bytes is None:
    st.info("👈 Select or upload a PDF from the sidebar to begin scanning.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════
#  RUN SCAN
# ══════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="🔍 Scanning document …")
def run_scan(data: bytes, _use_llm: bool) -> dict:
    """Write to temp file, run scan(), return serialisable results."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        report = scan(tmp_path, use_llm=_use_llm)
        # Serialise findings so Streamlit can cache them
        findings = []
        for f in report.findings:
            findings.append({
                "label":       f.label,
                "source":      f.source,
                "text":        f.text,
                "decoded":     f.decoded,
                "reasons":     f.reasons,
                "page":        f.page,
                "bbox":        f.bbox,
                "method":      f.method,
                "explanation": f.explanation,
                "confidence":  f.confidence,
                "details":     f.details,
            })
        return {
            "risk":       report.risk,
            "counts":     report.counts,
            "raw_text":   report.raw_text,
            "clean_text": report.clean_text,
            "findings":   findings,
        }
    finally:
        os.unlink(tmp_path)


result = run_scan(pdf_bytes, use_llm)
risk = result["risk"]
counts = result["counts"]
findings = result["findings"]
raw_text = result["raw_text"]
clean_text = result["clean_text"]

# ══════════════════════════════════════════════════════════════════════
#  RISK BANNER
# ══════════════════════════════════════════════════════════════════════
fg, bg, icon = RISK_COLORS[risk]
st.markdown(
    f"""
    <div style="
        background: {bg};
        border-left: 6px solid {fg};
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        font-size: 1.1rem;
    ">
        <span style="font-size:1.4rem">{icon}</span>
        <strong style="color:{fg}; margin-left:.4rem">
            Document Risk: {risk.upper()}
        </strong>
        <span style="color:#374151; margin-left:1.5rem;">
            Injection: <b>{counts.get('injection',0)}</b> &nbsp;|&nbsp;
            Data: <b>{counts.get('data',0)}</b> &nbsp;|&nbsp;
            Unknown: <b>{counts.get('unknown',0)}</b> &nbsp;|&nbsp;
            Harmless: <b>{counts.get('harmless',0)}</b>
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(f"📄 Source: **{source_label}** &nbsp;|&nbsp; LLM reasoning: **{'on' if use_llm else 'off'}**")

# ══════════════════════════════════════════════════════════════════════
#  PAGE VIEWER WITH BOUNDING-BOX OVERLAY + FINDINGS TABLE
# ══════════════════════════════════════════════════════════════════════
col_view, col_findings = st.columns([3, 2])

doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")

with col_view:
    st.subheader("📑 Visual Inspection")
    page_no = st.selectbox("Page", range(1, len(doc) + 1), format_func=lambda x: f"Page {x}")
    page_idx = page_no - 1
    page = doc[page_idx]

    # Render at 110 DPI
    pix = page.get_pixmap(dpi=DPI)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    draw = ImageDraw.Draw(img)

    # Overlay red bounding boxes for findings on this page
    page_findings = [f for f in findings if f["page"] == page_idx and f["bbox"]]
    for f in page_findings:
        x0, y0, x1, y1 = [c * SCALE for c in f["bbox"]]
        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)

    st.image(img, width="stretch", caption=f"Page {page_no} — red boxes mark detected findings")

with col_findings:
    st.subheader("🔎 Findings")

    visible_findings = findings if show_harmless else [f for f in findings if f["label"] != "harmless"]

    if not visible_findings:
        if findings:
            st.success("All findings on this document are harmless. Toggle *Show harmless findings* to view them.")
        else:
            st.success("✅ **Clean** — no findings detected.")
    else:
        for i, f in enumerate(visible_findings):
            label = f["label"]
            label_icon = {"injection": "🔴", "data": "🟠", "unknown": "🟡", "harmless": "⚪"}.get(label, "❓")
            page_display = f"p{f['page'] + 1}" if f["page"] is not None else f["source"]
            method_tag = ""
            if "llm" in f["method"]:
                method_tag = " 🤖"
            elif f["method"] == "heuristic":
                method_tag = " ⚙️"

            text_preview = (f["decoded"] or f["text"] or "")[:120]
            with st.expander(f"{label_icon} **{label.upper()}** — {page_display} — {', '.join(f['reasons'][:3])}{method_tag}"):
                st.markdown(f"**Label:** `{label}` &nbsp; **Method:** `{f['method']}`")
                st.markdown(f"**Source:** `{f['source']}` &nbsp; **Confidence:** {f['confidence']:.0%}")
                if f["reasons"]:
                    st.markdown(f"**Reasons:** {', '.join(f['reasons'])}")
                if text_preview:
                    st.code(text_preview, language=None)
                if f["decoded"]:
                    st.warning(f"**Decoded hidden message:** `{f['decoded'][:200]}`")
                if f["explanation"]:
                    st.info(f"**Explanation:** {f['explanation']}")
                if f.get("details", {}).get("needs_llm_verification"):
                    st.caption("⏳ This finding needs LLM verification for higher confidence.")

doc.close()

# ══════════════════════════════════════════════════════════════════════
#  RAW vs CLEAN TEXT DIFF
# ══════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("📝 Raw Parser Text vs. PDFGuard Sanitized Text")
st.caption(
    "**Left:** what a naïve PDF parser would feed your LLM (may contain hidden injections). "
    "**Right:** cleaned output after PDFGuard removes hidden content."
)

diff_col1, diff_col2 = st.columns(2)
with diff_col1:
    st.markdown("##### ⚠️ Raw Text (unprotected)")
    st.text_area("raw_text", value=raw_text, height=320, disabled=True, label_visibility="collapsed")
with diff_col2:
    st.markdown("##### 🛡️ Clean Text (sanitized)")
    st.text_area("clean_text", value=clean_text, height=320, disabled=True, label_visibility="collapsed")

st.download_button(
    label="📥 Download Sanitized Text",
    data=clean_text.encode("utf-8"),
    file_name="pdfguard_clean.txt",
    mime="text/plain",
)
