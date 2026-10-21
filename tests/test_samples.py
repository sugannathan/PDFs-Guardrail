"""Run with: pytest -q   (builds the sample PDFs on first run)"""
from pathlib import Path

import pytest

from pdfguard import scan, split_pdf

HERE = Path(__file__).parent
SAMPLES = HERE / "samples"

EXPECTED = {
    "00_clean": "clean",
    "01_white_text": "dangerous",
    "02_tiny_font": "dangerous",
    "03_off_page": "dangerous",
    "04_covered_by_image": "dangerous",
    "05_invisible_render_mode": "dangerous",
    "06_zero_opacity": "dangerous",
    "07_same_colour_as_background": "dangerous",
    "08_hidden_layer": "dangerous",
    "09_metadata": "dangerous",
    "10_xmp_metadata": "dangerous",
    "11_hidden_annotation": "dangerous",
    "12_hidden_form_field": "dangerous",
    "13_zero_width_message": "dangerous",
    "13b_unicode_tags_in_metadata": "dangerous",
    "14_embedded_file": "dangerous",
    "15_scanned_ocr_layer_benign": "clean",
    "16_scanned_ocr_layer_tampered": "dangerous",
}


@pytest.fixture(scope="session", autouse=True)
def samples():
    if not (SAMPLES / "00_clean.pdf").exists():
        import make_samples
        make_samples.main(str(SAMPLES))


@pytest.mark.parametrize("name,risk", EXPECTED.items())
def test_risk(name, risk):
    assert scan(SAMPLES / f"{name}.pdf", use_ocr=True).risk == risk


def test_clean_text_drops_hidden_value():
    r = scan(SAMPLES / "02_tiny_font.pdf")
    assert "4.8 eV" in r.raw_text and "4.8 eV" not in r.clean_text
    assert "3.2 eV" in r.clean_text


def test_split_has_all_types():
    parts = split_pdf(SAMPLES / "12_hidden_form_field.pdf")
    assert parts.form_fields and parts.text_spans and parts.metadata
