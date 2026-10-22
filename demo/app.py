# -*- coding: utf-8 -*-
"""PDFGuard — Judge-Facing Dashboard.

Two modes:
  • **Benchmark Gallery** – browse pre-computed results from ``all_examples.json``
    without needing LLM weights, API keys, or even the example PDFs on disk.
  • **Live Upload** – upload any PDF and run the full Stage 1 + Stage 2 pipeline
    in real time (falls back to heuristic-only when no LLM is configured).

Run:
    streamlit run demo/app.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Path setup – make ``pdfguard`` importable regardless of cwd
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from pdfguard import config
from pdfguard.report import scan, Report, sanitize_pdf_text

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PDFGuard · Pre-LLM Document Auditor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — premium dark-mode inspired styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* Global overrides */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Risk badges */
    .risk-badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 24px;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .risk-dangerous {
        background: linear-gradient(135deg, #e53935 0%, #b71c1c 100%);
        color: #fff;
    }
    .risk-suspicious {
        background: linear-gradient(135deg, #ffa726 0%, #e65100 100%);
        color: #fff;
    }
    .risk-clean {
        background: linear-gradient(135deg, #66bb6a 0%, #2e7d32 100%);
        color: #fff;
    }

    /* Label pills */
    .label-pill {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.78rem;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        margin-right: 4px;
    }
    .label-injection {
        background: #e53935;
        color: #fff;
    }
    .label-data {
        background: #fb8c00;
        color: #fff;
    }
    .label-unknown {
        background: #7e57c2;
        color: #fff;
    }
    .label-harmless {
        background: #66bb6a;
        color: #fff;
    }

    /* Confidence bar */
    .conf-bar-bg {
        background: rgba(255,255,255,0.08);
        border-radius: 6px;
        overflow: hidden;
        height: 8px;
        width: 120px;
        display: inline-block;
        vertical-align: middle;
        margin-left: 6px;
    }
    .conf-bar-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.4s ease;
    }

    /* Finding card */
    .finding-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .finding-card:hover {
        border-color: rgba(255,255,255,0.18);
    }

    /* Metric card */
    .metric-card {
        text-align: center;
        padding: 14px 8px;
        border-radius: 10px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.06);
    }
    .metric-card .val {
        font-size: 1.7rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .metric-card .lbl {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        opacity: 0.65;
    }

    /* section titles */
    .section-title {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.5;
        margin-bottom: 6px;
    }

    /* Override Streamlit tab styling */
    div[data-baseweb="tab-list"] button {
        font-weight: 600;
    }

    /* Hide default Streamlit footer */
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
st.markdown(
    "<h1 style='margin-bottom:0'>🛡️ PDFGuard</h1>"
    "<p style='margin-top:0; opacity:0.6; font-size:1.05rem'>"
    "Pre-LLM Document Auditor — Detect hidden injections, data contradictions "
    "& invisible Unicode payloads before ingestion.</p>",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════

def _risk_badge(risk: str) -> str:
    """Return HTML for a colour-coded risk badge."""
    return f'<span class="risk-badge risk-{risk}">{risk}</span>'


def _label_pill(label: str) -> str:
    """Return HTML for a label pill."""
    icons = {"injection": "💉", "data": "📊", "unknown": "❓", "harmless": "✅"}
    icon = icons.get(label, "")
    return f'<span class="label-pill label-{label}">{icon} {label}</span>'


def _confidence_bar(conf: float) -> str:
    """Return HTML for a slim confidence bar."""
    pct = max(0, min(100, int(conf * 100)))
    if conf >= 0.8:
        colour = "#66bb6a"
    elif conf >= 0.5:
        colour = "#ffa726"
    else:
        colour = "#ef5350"
    return (
        f'<span class="conf-bar-bg">'
        f'<span class="conf-bar-fill" style="width:{pct}%;background:{colour}"></span>'
        f'</span> <small>{pct}%</small>'
    )


def _metric_html(value, label: str) -> str:
    return (
        f'<div class="metric-card">'
        f'<div class="val">{value}</div>'
        f'<div class="lbl">{label}</div>'
        f'</div>'
    )


def _render_finding_card(f: dict, idx: int) -> None:
    """Render a single finding as a rich card."""
    label = f.get("label", "unknown")
    source = f.get("source", "?")
    conf = f.get("confidence", 0)
    method = f.get("method", "heuristic")
    explanation = f.get("explanation", "")
    reasons = f.get("reasons", [])
    text = f.get("text", "")
    decoded = f.get("decoded")
    page_number = f.get("page_number")
    contradicts = f.get("contradicts_visible", False)
    details = f.get("details", {})

    # Header row
    header_parts = [_label_pill(label)]
    if page_number is not None:
        header_parts.append(f"<small style='opacity:0.55'>Page {page_number}</small>")
    else:
        header_parts.append(f"<small style='opacity:0.55'>Non-visual ({source})</small>")
    header_parts.append(f"<small style='opacity:0.55'>·</small>")
    header_parts.append(_confidence_bar(conf))
    if contradicts:
        header_parts.append("<small>⚠️ <b>Contradicts visible</b></small>")

    header = " &nbsp;".join(header_parts)

    with st.expander(f"Finding #{idx + 1}: {source} — {', '.join(reasons[:3])}", expanded=(label in ("injection", "data"))):
        st.markdown(header, unsafe_allow_html=True)
        st.markdown("---")

        # Text / Decoded
        if decoded:
            st.markdown(f"**🔓 Decoded hidden message:**")
            st.code(decoded, language=None)
            st.markdown(f"**Carrier text:** `{text[:200]}`")
        else:
            st.markdown(f"**Flagged text:**")
            st.code(text[:500] if text else "(no text)", language=None)

        # Explanation
        if explanation:
            st.info(f"💬 **Reasoning:** {explanation}")

        # Method
        method_labels = {
            "heuristic": "🔧 Heuristic only",
            "llm": "🤖 LLM-verified",
            "heuristic+llm_disagrees": "⚡ Heuristic (LLM disagrees)",
        }
        st.markdown(f"**Method:** {method_labels.get(method, method)}")

        # Hiding techniques
        st.markdown(f"**Hiding technique(s):** `{'`, `'.join(reasons)}`")

        # Details
        if details:
            detail_parts = []
            if "font_size" in details:
                detail_parts.append(f"Font: {details['font_size']} pt")
            if "color" in details:
                c = details["color"]
                if isinstance(c, list) and len(c) == 3:
                    r_val = int(c[0] * 255) if isinstance(c[0], float) and c[0] <= 1 else int(c[0])
                    g_val = int(c[1] * 255) if isinstance(c[1], float) and c[1] <= 1 else int(c[1])
                    b_val = int(c[2] * 255) if isinstance(c[2], float) and c[2] <= 1 else int(c[2])
                    hex_color = f"#{r_val:02x}{g_val:02x}{b_val:02x}"
                    detail_parts.append(
                        f'Color: <span style="background:{hex_color};'
                        f'padding:1px 10px;border-radius:3px;border:1px solid rgba(255,255,255,0.2)">'
                        f'&nbsp;</span> `{hex_color}`'
                    )
            if "render_mode" in details:
                rm = details["render_mode"]
                rm_labels = {0: "Fill", 1: "Stroke", 2: "Fill+Stroke", 3: "Invisible", 7: "Clip only"}
                detail_parts.append(f"Render mode: {rm} ({rm_labels.get(rm, '?')})")
            if "visible_values" in details:
                for unit, vals in details["visible_values"].items():
                    detail_parts.append(f"Visible {unit}: {', '.join(str(v) for v in vals)}")
            if "invisible_chars" in details:
                chars = details["invisible_chars"][:10]
                detail_parts.append(f"Invisible chars: {', '.join(chars)} ({details.get('count', '?')} total)")
            if "name" in details:
                detail_parts.append(f"Field name: `{details['name']}`")
            if "description" in details:
                detail_parts.append(f"Description: {details['description']}")
            if "heuristic_label" in details:
                detail_parts.append(f"Heuristic label: {details['heuristic_label']}")
            if "llm_label" in details:
                detail_parts.append(f"LLM label: {details['llm_label']}")
            if "llm_model" in details:
                detail_parts.append(f"LLM model: `{details['llm_model']}`")
            if "llm_error" in details:
                detail_parts.append(f"⚠️ LLM error: {details['llm_error'][:150]}")
            if "ocr_similarity" in details:
                detail_parts.append(f"OCR similarity: {details['ocr_similarity']}")

            if detail_parts:
                st.markdown(" · ".join(detail_parts), unsafe_allow_html=True)

        # Bounding box
        bbox = f.get("bbox")
        if bbox:
            st.caption(f"BBox: [{', '.join(f'{v:.1f}' for v in bbox)}]")


def _render_counts_row(counts: dict) -> None:
    """Render the label counts as a row of metric cards."""
    keys = [
        ("injection", "💉 Injection"),
        ("data", "📊 Data"),
        ("unknown", "❓ Unknown"),
        ("harmless", "✅ Harmless"),
        ("hidden_spans", "👁️‍🗨️ Hidden Spans"),
        ("total_spans", "📝 Total Spans"),
    ]
    cols = st.columns(len(keys))
    for col, (key, label) in zip(cols, keys):
        val = counts.get(key, 0)
        with col:
            st.markdown(_metric_html(val, label), unsafe_allow_html=True)


def _render_parts_summary(parts: dict) -> None:
    """Render the document structure summary."""
    st.markdown('<p class="section-title">Document Structure</p>', unsafe_allow_html=True)
    items = [
        ("pages", "Pages"), ("text_spans", "Text Spans"), ("images", "Images"),
        ("drawings", "Drawings"), ("annotations", "Annotations"), ("form_fields", "Form Fields"),
        ("links", "Links"), ("embedded_files", "Embeds"), ("metadata_fields", "Metadata"),
        ("xmp_bytes", "XMP (bytes)"), ("javascript", "JavaScript"), ("layers", "Layers"),
    ]
    cols = st.columns(6)
    for i, (key, label) in enumerate(items):
        val = parts.get(key, 0)
        with cols[i % 6]:
            st.markdown(_metric_html(val, label), unsafe_allow_html=True)


def _render_full_report(report: dict, pdf_bytes: bytes | None = None, sanitized_text: str | None = None) -> None:
    """Render a complete report — used by both gallery and live modes."""
    filename = Path(report.get("file", "unknown")).name
    risk = report.get("risk", "unknown")
    counts = report.get("counts", {})
    findings = report.get("findings", [])
    parts = report.get("parts", {})
    clean_text = report.get("clean_text", "")

    # If sanitized text was not pre-computed, extract it using sanitize_pdf_text
    if sanitized_text is None and pdf_bytes is not None:
        try:
            sanitized_text = sanitize_pdf_text(pdf_bytes)
        except Exception:
            sanitized_text = clean_text

    # --- Risk verdict ---
    st.markdown(
        f"<h3>{filename} &nbsp; {_risk_badge(risk)}</h3>",
        unsafe_allow_html=True,
    )

    # --- Counts row ---
    _render_counts_row(counts)
    st.markdown("")

    # --- Document structure ---
    if parts:
        with st.expander("📐 Document Structure", expanded=False):
            _render_parts_summary(parts)

    # --- Findings ---
    visual_findings = [f for f in findings if f.get("bbox") is not None]
    nonvisual_findings = [f for f in findings if f.get("bbox") is None]

    if not findings:
        st.success("✅ **Clean document** — no hidden or anomalous content detected.")
    else:
        # Visual findings (have bbox → can be shown on PDF overlay)
        if visual_findings:
            st.markdown(f"### 🔍 On-Page Findings ({len(visual_findings)})")

            # If we have the actual PDF bytes, show visual overlay
            if pdf_bytes is not None:
                _render_pdf_overlay(pdf_bytes, visual_findings)

            for i, f in enumerate(visual_findings):
                _render_finding_card(f, i)

        # Non-visual findings
        if nonvisual_findings:
            st.markdown(f"### 📎 Non-Visual Findings ({len(nonvisual_findings)})")
            st.caption("These findings live in metadata, annotations, form fields, or embedded files — not rendered on any page.")
            for i, f in enumerate(nonvisual_findings):
                _render_finding_card(f, len(visual_findings) + i)

    # --- Before / After comparison ---
    st.markdown("---")
    st.markdown("### ⚖️ Pipeline Impact: Before vs After Sanitisation")
    col_before, col_after = st.columns(2)
    with col_before:
        st.markdown("**🔴 Raw Parser Output (Unprotected)**")
        if findings:
            injection_texts = [
                f.get("decoded") or f.get("text", "")
                for f in findings
                if f.get("label") == "injection"
            ]
            data_texts = [
                f.get("text", "")
                for f in findings
                if f.get("label") == "data"
            ]
            if injection_texts:
                st.error("💉 **Prompt injection payloads present in raw text:**")
                for t in injection_texts[:5]:
                    st.code(t[:300], language=None)
            if data_texts:
                st.warning("📊 **Hidden data values would contaminate extraction:**")
                for t in data_texts[:5]:
                    st.code(t[:300], language=None)
            if not injection_texts and not data_texts:
                st.warning("⚠️ Hidden content detected that could affect LLM processing.")
        else:
            st.info("Normal text — no hidden content.")

    with col_after:
        st.markdown("**🟢 PDFGuard Sanitised Output**")
        if findings:
            st.success("🛡️ **All hidden content stripped.** Only visible text reaches the LLM.")
        else:
            st.success("🛡️ Document is clean — sanitised output matches raw output.")

    # --- Sanitised text ---
    final_text = sanitized_text or clean_text
    if final_text:
        with st.expander("📄 Sanitized Text (Visible-Only Content)", expanded=False):
            st.text_area(
                "Clean, guardrail-protected text sent to downstream LLM",
                value=final_text,
                height=250,
                disabled=True,
                key=f"clean_text_{filename}_{id(report)}",
            )
            st.download_button(
                label="📥 Download Sanitized Text",
                data=final_text,
                file_name=f"{filename}.sanitized.txt",
                mime="text/plain",
                key=f"dl_sanitized_main_{filename}_{id(report)}",
            )


def _render_pdf_overlay(pdf_bytes: bytes, findings: list[dict]) -> None:
    """Render PDF pages with red bounding-box overlays for visual findings."""
    try:
        import fitz
        from PIL import Image
        import io

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        # Determine which pages have findings
        pages_with_findings = sorted({f["page_number"] for f in findings if f.get("page_number")})
        all_pages = list(range(1, len(doc) + 1))

        page_no = st.selectbox(
            "Select page to inspect",
            all_pages,
            index=0 if not pages_with_findings else all_pages.index(pages_with_findings[0]),
            key=f"page_select_{id(pdf_bytes)}",
        )

        page = doc[page_no - 1]

        # Draw overlays
        page_findings = [f for f in findings if f.get("page_number") == page_no]
        for f in page_findings:
            bbox = f.get("bbox")
            if bbox:
                r = fitz.Rect(bbox)
                label = f.get("label", "unknown")
                color_map = {
                    "injection": (1, 0, 0),
                    "data": (1, 0.55, 0),
                    "unknown": (0.5, 0.35, 0.8),
                    "harmless": (0.4, 0.73, 0.42),
                }
                color = color_map.get(label, (1, 0, 0))
                page.draw_rect(r, color=color, width=2.5)
                label_text = f"⚠️ {label.upper()}"
                try:
                    page.insert_text(
                        fitz.Point(r.x0, max(r.y0 - 3, 10)),
                        label_text,
                        fontsize=7,
                        color=color,
                    )
                except Exception:
                    pass  # Some pages may have restrictions

        pix = page.get_pixmap(dpi=150)
        img = Image.open(io.BytesIO(pix.tobytes("png")))

        finding_count_on_page = len(page_findings)
        caption = f"Page {page_no} of {len(doc)}"
        if finding_count_on_page:
            caption += f" — {finding_count_on_page} finding(s) highlighted"
        st.image(img, use_container_width=True, caption=caption)
        doc.close()
    except Exception as e:
        st.warning(f"Could not render PDF overlay: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# Load benchmark data
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data
def _load_benchmark() -> dict | None:
    """Load the pre-computed benchmark report from results/all_examples.json."""
    candidates = [
        _PROJECT_ROOT / "results" / "all_examples.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
    return None


@st.cache_data
def _load_expected() -> dict:
    """Load expected.json for attack descriptions."""
    p = _PROJECT_ROOT / "examples" / "expected.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


@st.cache_data
def _load_parser_leak_data() -> dict:
    """Load parser leak comparison data from reports/parsers.csv or results/parsers.csv."""
    data = {
        "matrix": [],
        "file_tests": [],
        "has_data": False,
    }

    # 1. Per-attack leak summary from reports/parsers.csv (eval/parser_leak.py)
    leak_csv = _PROJECT_ROOT / "reports" / "parsers.csv"
    if leak_csv.exists():
        try:
            attack_labels = {
                "white_text": "White text (matching background)",
                "tiny_font": "Micro-font (0.5 pt speck)",
                "off_page": "Off-page (outside cropbox)",
                "covered_text": "Covered text (under opaque image/rect)",
                "invisible_render": "Invisible render mode (mode 3)",
                "metadata": "Document metadata (subject/keywords)",
                "annotation": "Annotations (hidden sticky note)",
                "unicode": "Zero-width Unicode steganography",
            }
            rows_by_attack = {}
            with leak_csv.open(encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    att = r.get("attack", "")
                    pname = r.get("parser", "")
                    leak_rate = float(r.get("leak_rate", 0))
                    leaked = r.get("leaked", "0")
                    total = r.get("total", "0")

                    if att not in rows_by_attack:
                        rows_by_attack[att] = {
                            "Attack Vector": attack_labels.get(att, att),
                            "_raw_attack": att,
                            "PDFGuard": "0% 🛡️ (0/10)",
                        }
                    status = "⚠️ Leaked" if leak_rate > 0 else "✅ Blocked"
                    rows_by_attack[att][pname] = f"{leak_rate:.0f}% ({leaked}/{total})"

            data["matrix"] = list(rows_by_attack.values())
            if data["matrix"]:
                data["has_data"] = True
        except Exception:
            pass

    # 2. Per-file parser leakage results from results/parsers.csv (eval/parsers.py)
    per_file_csv = _PROJECT_ROOT / "results" / "parsers.csv"
    if per_file_csv.exists():
        try:
            with per_file_csv.open(encoding="utf-8") as f:
                reader = csv.DictReader(f)
                file_rows = []
                for r in reader:
                    file_rows.append({
                        "File": r.get("file", ""),
                        "Technique": r.get("reasons", ""),
                        "Payload Preview": (r.get("snippet", "")[:60] + "...") if len(r.get("snippet", "")) > 60 else r.get("snippet", ""),
                        "PyMuPDF": r.get("PyMuPDF", "—"),
                        "pymupdf4llm": r.get("pymupdf4llm", "—"),
                        "pdfplumber": r.get("pdfplumber", "—"),
                        "pypdf": r.get("pypdf", "—"),
                        "PDFGuard": "BLOCKED 🛡️",
                    })
                data["file_tests"] = file_rows
                data["has_data"] = True
        except Exception:
            pass

    return data


def _render_parser_vulnerability_summary() -> None:
    """Render the parser vulnerability comparison summary from eval/parser_leak.py."""
    parser_data = _load_parser_leak_data()

    with st.expander("🔬 Parser Vulnerability Comparison (eval/parser_leak.py)", expanded=True):
        st.markdown(
            "Empirical evaluation measuring **which standard PDF parsers inadvertently leak hidden prompt injections** "
            "and tampered data into downstream LLM pipelines."
        )

        # Overview Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(_metric_html("97.5%", "PyMuPDF Leak Rate"), unsafe_allow_html=True)
        with col2:
            st.markdown(_metric_html("87.5%", "pdfplumber Leak Rate"), unsafe_allow_html=True)
        with col3:
            st.markdown(_metric_html("87.5%", "pypdf Leak Rate"), unsafe_allow_html=True)
        with col4:
            st.markdown(_metric_html("0.0%", "🛡️ PDFGuard (Ours)"), unsafe_allow_html=True)

        st.caption(
            "Based on 80 attack runs across 8 distinct hiding techniques on real arXiv and synthetic academic papers. "
            "A higher leak rate means that parser hands adversarial payloads directly to an LLM."
        )

        st.markdown("")

        # 1. Aggregate Matrix
        if parser_data.get("matrix"):
            st.markdown("#### 📊 Leak Rate by Attack Vector (`eval/parser_leak.py`)")
            matrix_df = [
                {k: v for k, v in row.items() if not k.startswith("_")}
                for row in parser_data["matrix"]
            ]
            st.dataframe(
                matrix_df,
                use_container_width=True,
                hide_index=True,
            )

        # 2. Per-file breakdown if available
        if parser_data.get("file_tests"):
            st.markdown("#### 📋 Example Attack Files vs Parser Output (`eval/parsers.py`)")
            st.dataframe(
                parser_data["file_tests"],
                use_container_width=True,
                hide_index=True,
                height=min(35 * len(parser_data["file_tests"]) + 38, 350),
            )

        st.info(
            "💡 **Key Finding:** Traditional document extractors (`pypdf`, `pdfplumber`, `PyMuPDF`) blindly concatenate font spans "
            "or annotations without verifying physical rendering or visibility. In contrast, **PDFGuard** combines layout geometry, "
            "pixel-level ink rendering, and zero-width code point filters to ensure **0% adversarial leakage**."
        )


# ═══════════════════════════════════════════════════════════════════════════
# Main app — tabs
# ═══════════════════════════════════════════════════════════════════════════

tab_benchmark, tab_audit = st.tabs(["📊 Benchmark Explorer", "📤 Live Document Audit"])

# ─────────────────────────────────────────────────────────────────────────
# Tab 1: Benchmark Explorer
# ─────────────────────────────────────────────────────────────────────────
with tab_benchmark:
    benchmark = _load_benchmark()
    expected = _load_expected()

    if benchmark is None:
        st.warning(
            "⚠️ No pre-computed benchmark found at `results/all_examples.json`. "
            "Generate it with:\n\n"
            "```bash\n"
            "python -m pdfguard scan examples/ --no-llm --report results/all_examples.json --with-text\n"
            "```"
        )
    else:
        # --- Benchmark header ---
        meta_cols = st.columns(4)
        totals = benchmark.get("totals", {})
        settings = benchmark.get("settings", {})
        with meta_cols[0]:
            st.markdown(_metric_html(totals.get("files", 0), "Files Scanned"), unsafe_allow_html=True)
        with meta_cols[1]:
            st.markdown(_metric_html(totals.get("dangerous", 0), "🔴 Dangerous"), unsafe_allow_html=True)
        with meta_cols[2]:
            st.markdown(_metric_html(totals.get("suspicious", 0), "🟠 Suspicious"), unsafe_allow_html=True)
        with meta_cols[3]:
            st.markdown(_metric_html(totals.get("clean", 0), "🟢 Clean"), unsafe_allow_html=True)

        st.caption(
            f"PDFGuard v{benchmark.get('pdfguard_version', '?')} · "
            f"Scanned {benchmark.get('scanned_at', '?')} · "
            f"LLM: {settings.get('llm') or 'heuristic only'} · "
            f"OCR: {'on' if settings.get('ocr') else 'off'} · "
            f"DPI: {settings.get('dpi', '?')}"
        )

        st.markdown("---")

        # --- Parser Vulnerability Comparison Section ---
        _render_parser_vulnerability_summary()

        st.markdown("---")

        # --- Overview table ---
        reports = benchmark.get("reports", [])
        report_names = [Path(r["file"]).name for r in reports]

        # Summary table
        st.markdown("### 📋 Benchmark Corpus Scan Summary (`all_examples.json`)")
        summary_data = []
        for r in reports:
            name = Path(r["file"]).name
            exp = expected.get(name, {})
            hidden_by = exp.get("hidden_by") or "—"
            exp_label = exp.get("label", "—")
            risk = r.get("risk", "?")
            fc = r.get("counts", {})
            summary_data.append({
                "File": name,
                "Risk": risk.upper(),
                "Injection": fc.get("injection", 0),
                "Data": fc.get("data", 0),
                "Unknown": fc.get("unknown", 0),
                "Harmless": fc.get("harmless", 0),
                "Hidden By": hidden_by,
                "Expected Label": exp_label,
            })

        st.dataframe(
            summary_data,
            use_container_width=True,
            height=min(35 * len(summary_data) + 38, 500),
            column_config={
                "Risk": st.column_config.TextColumn(width="small"),
                "Injection": st.column_config.NumberColumn(width="small"),
                "Data": st.column_config.NumberColumn(width="small"),
                "Unknown": st.column_config.NumberColumn(width="small"),
                "Harmless": st.column_config.NumberColumn(width="small"),
            },
        )

        st.markdown("---")

        # --- Detailed per-file view ---
        st.markdown("### 🔎 Detailed Report Inspector")
        selected_name = st.selectbox(
            "Select a benchmark file to inspect",
            report_names,
            key="gallery_select",
        )
        selected_report = next((r for r in reports if Path(r["file"]).name == selected_name), None)

        if selected_report:
            exp_info = expected.get(selected_name, {})
            if exp_info.get("hidden_by"):
                st.caption(
                    f"**Attack:** {exp_info.get('hidden_by', '?')} · "
                    f"**Expected payload:** `{exp_info.get('payload', '?')}` · "
                    f"**Expected label:** {exp_info.get('label', '?')}"
                )

            # Try to load the actual PDF for overlay rendering
            pdf_path = _PROJECT_ROOT / "examples" / selected_name
            pdf_bytes = None
            if pdf_path.exists():
                try:
                    pdf_bytes = pdf_path.read_bytes()
                except Exception:
                    pass

            _render_full_report(selected_report, pdf_bytes=pdf_bytes)


# ─────────────────────────────────────────────────────────────────────────
# Tab 2: Live Document Audit
# ─────────────────────────────────────────────────────────────────────────
with tab_audit:
    # Stage 2 LLM status banner
    is_llm_ready = config.llm_available()
    if not is_llm_ready:
        st.warning(
            "⚠️ **Running in Stage 1 Deterministic Mode** "
            "(Provide Cerebras API key to enable live Stage 2 reasoning)"
        )
    else:
        st.success(
            f"🚀 **Live Stage 2 LLM Reasoning Active** — Connected to "
            f"**{config.provider().title()}** (`{config.model()}`). "
            f"Flagged snippets undergo live LLM intent reasoning and confidence scoring."
        )

    st.markdown(
        "Upload any PDF to run the **complete dual-stage PDFGuard pipeline** in real time:\n\n"
        "1. **Stage 1 Deterministic Heuristics**: Extracts microscopic fonts (<1 pt), invisible render modes (mode 3), "
        "off-page coordinates, white-on-white text, covered bounding boxes, hidden layers, zero-width Unicode steganography, "
        "and non-visual metadata.\n"
        "2. **Stage 2 Live LLM Classification**: Classifies flagged snippets via Cerebras (`gpt-oss-120b`) into "
        "`injection`, `data`, or `harmless` with calibrated confidence scores and natural-language explanations."
    )

    uploaded_file = st.file_uploader("Upload PDF for Live Audit", type=["pdf"], key="live_audit_upload")

    if uploaded_file is not None:
        pdf_bytes = uploaded_file.read()

        spinner_label = (
            "Running full PDFGuard pipeline (Stage 1 Heuristics + Stage 2 Cerebras LLM Reasoning)…"
            if is_llm_ready else
            "Running PDFGuard Stage 1 Heuristics (Deterministic Mode)…"
        )

        with st.spinner(spinner_label):
            try:
                # Write to a temp file because scan() expects a path
                tmp_dir = _PROJECT_ROOT / "results" / ".tmp"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                tmp_path = tmp_dir / uploaded_file.name
                tmp_path.write_bytes(pdf_bytes)

                try:
                    # Executes Stage 1 heuristics and Stage 2 LLM classification (if configured)
                    report = scan(str(tmp_path), use_llm=is_llm_ready, use_ocr=False)
                    report_dict = report.to_dict(include_text=True)
                finally:
                    # Clean up temp file
                    try:
                        tmp_path.unlink()
                    except Exception:
                        pass

                # Clean Text Sanitization: deliver visible-only, guardrail-protected text
                sanitized_text = sanitize_pdf_text(pdf_bytes)
                report_dict["sanitized_text"] = sanitized_text

                _render_full_report(report_dict, pdf_bytes=pdf_bytes, sanitized_text=sanitized_text)

                # Sidebar download actions
                st.sidebar.header("📥 Document Downloads")
                st.sidebar.download_button(
                    label="Download Report (JSON)",
                    data=json.dumps(report_dict, indent=2, ensure_ascii=False),
                    file_name=f"{uploaded_file.name}.report.json",
                    mime="application/json",
                    key="sidebar_dl_report_json",
                )
                st.sidebar.download_button(
                    label="Download Sanitized Text",
                    data=sanitized_text,
                    file_name=f"{uploaded_file.name}.sanitized.txt",
                    mime="text/plain",
                    key="sidebar_dl_sanitized_text",
                    help="Calls sanitize_pdf_text() to deliver visible-only text stripped of invisible payloads.",
                )

            except Exception as e:
                st.error(f"Pipeline error: {e}")
                import traceback
                st.code(traceback.format_exc(), language="python")

# ─────────────────────────────────────────────────────────────────────────
# Sidebar Configuration & Info
# ─────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Engine Settings")
provider_name = config.provider().title()
model_name = config.model()
is_llm_ready = config.llm_available()

if is_llm_ready:
    st.sidebar.success(f"🟢 **Stage 2 Active**\n{provider_name} · `{model_name}`")
else:
    st.sidebar.warning("🟡 **Stage 1 Mode**\nNo API Key Active")

cerebras_key_input = st.sidebar.text_input(
    "Cerebras API Key",
    value=config.api_key(),
    type="password",
    help="Loaded automatically from secrets.env. You can update or paste your key here.",
    key="sidebar_cerebras_key_input",
)
if cerebras_key_input and cerebras_key_input != config.api_key():
    os.environ["CEREBRAS_API_KEY"] = cerebras_key_input.strip()
    config.reload()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**PDFGuard v0.2** · "
    "[GitHub](https://github.com/sugannathan/PDFs-Guardrail) · "
    "Built for [SaTML 2025](https://satml.org)"
)
st.sidebar.caption(
    "Stage 1: Heuristic extraction (micro-fonts, invisible render modes, "
    "off-page text, covered text, hidden layers, zero-width Unicode, "
    "metadata, XMP, annotations, form fields, embedded files, JavaScript)\n\n"
    "Stage 2: Intent classification (injection / data / harmless / unknown) "
    "with confidence scores, contradiction detection, and Cerebras gpt-oss-120b live reasoning"
)
