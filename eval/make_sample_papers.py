"""Generate diverse, fully synthetic scientific-paper PDFs for local testing.

These are NOT real papers. They exist only so the attack generator has varied,
license-free inputs to inject clearly-marked test payloads into. Content is
invented. Any resemblance to a real paper is coincidental.

Usage:
    python eval/make_sample_papers.py --out data/clean/ --n 6
"""
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import pymupdf  # PyMuPDF


# A small library of invented papers across different scientific domains, so the
# downstream test set covers varied layouts, fonts and visible "ground-truth"
# values that hidden payloads can contradict.
PAPERS = [
    {
        "slug": "mtrl_bandgap",
        "title": "Optical Band Gap of Sputtered Zinc Oxide Thin Films",
        "authors": "A. Rivera, L. Okonkwo, M. Haas",
        "abstract": (
            "We report the optical band gap of zinc oxide thin films deposited "
            "by RF magnetron sputtering. Tauc analysis of transmission spectra "
            "yields a direct band gap of 3.2 eV, consistent with the known "
            "value for wurtzite ZnO. Film thickness was 240 nm."
        ),
        "sections": [
            ("Introduction",
             "Zinc oxide is a wide band gap semiconductor used in transparent "
             "electronics and ultraviolet optoelectronics. Accurate band gap "
             "determination is essential for device modelling."),
            ("Methods",
             "Films were sputtered onto fused silica at 200 C. Transmission "
             "spectra were recorded from 300 to 900 nm. The absorption "
             "coefficient was extracted and fit with a Tauc plot assuming a "
             "direct allowed transition."),
            ("Results",
             "The Tauc fit gives a band gap of 3.2 eV. The films are highly "
             "transparent above 400 nm with an average transmission of 88 "
             "percent. X-ray diffraction confirms the wurtzite structure."),
            ("Conclusion",
             "Sputtered ZnO films show a band gap of 3.2 eV and high optical "
             "transparency, suitable for transparent conducting applications."),
        ],
        "key_value": "band gap = 3.2 eV",
    },
    {
        "slug": "chem_yield",
        "title": "Palladium-Catalysed Coupling Yields Under Mild Conditions",
        "authors": "S. Nakamura, P. Delacroix",
        "abstract": (
            "A palladium-catalysed cross-coupling is reported with an isolated "
            "yield of 74 percent at room temperature. The reaction tolerates "
            "a range of functional groups and completes within four hours."
        ),
        "sections": [
            ("Introduction",
             "Cross-coupling reactions form carbon-carbon bonds central to "
             "pharmaceutical synthesis. Mild conditions reduce side products."),
            ("Experimental",
             "The substrate was combined with the aryl halide, base and a "
             "palladium catalyst in tetrahydrofuran at 25 C for four hours. "
             "The product was isolated by column chromatography."),
            ("Results and Discussion",
             "The isolated yield was 74 percent. Nuclear magnetic resonance "
             "confirmed the product structure. Turnover number reached 148."),
            ("Conclusion",
             "The protocol delivers a 74 percent yield under mild conditions "
             "with broad functional group tolerance."),
        ],
        "key_value": "isolated yield = 74 percent",
    },
    {
        "slug": "bio_expression",
        "title": "Temperature Dependence of Reporter Gene Expression in E. coli",
        "authors": "R. Mbeki, T. Larsson, H. Qadir",
        "abstract": (
            "Reporter gene expression in Escherichia coli peaks at 37 C, with "
            "fluorescence intensity falling by 40 percent at 30 C. Growth was "
            "monitored over twelve hours."
        ),
        "sections": [
            ("Introduction",
             "Temperature strongly affects protein expression in bacterial "
             "hosts. We quantify this for a green fluorescent reporter."),
            ("Methods",
             "Cultures were grown at 30, 37 and 42 C. Optical density and "
             "fluorescence were recorded hourly for twelve hours."),
            ("Results",
             "Peak expression occurred at 37 C. At 30 C fluorescence dropped "
             "by 40 percent; at 42 C cells showed reduced viability."),
            ("Conclusion",
             "Optimal reporter expression occurs at 37 C in this system."),
        ],
        "key_value": "optimum temperature = 37 C",
    },
    {
        "slug": "phys_resistivity",
        "title": "Low-Temperature Resistivity of a Thin Copper Interconnect",
        "authors": "K. Fischer, D. Oyelaran",
        "abstract": (
            "The electrical resistivity of a 90 nm copper interconnect is "
            "2.1 microohm centimetre at room temperature, rising with grain "
            "boundary scattering as the line narrows."
        ),
        "sections": [
            ("Introduction",
             "Resistivity of narrow metal lines increases due to surface and "
             "grain boundary scattering, a key limit for chip interconnects."),
            ("Methods",
             "Four-point probe measurements were taken from 4 K to 300 K on "
             "lines patterned by electron beam lithography."),
            ("Results",
             "Room temperature resistivity was 2.1 microohm centimetre. The "
             "residual resistance ratio was 3.4, indicating strong scattering."),
            ("Conclusion",
             "Narrow copper lines show elevated resistivity of 2.1 microohm "
             "centimetre, consistent with scattering models."),
        ],
        "key_value": "resistivity = 2.1 microohm cm",
    },
    {
        "slug": "env_adsorption",
        "title": "Adsorption Capacity of Activated Carbon for a Model Dye",
        "authors": "N. Abara, J. Westerlund",
        "abstract": (
            "Activated carbon adsorbs a model dye with a maximum capacity of "
            "152 mg per gram, following a Langmuir isotherm. Equilibrium is "
            "reached within 90 minutes."
        ),
        "sections": [
            ("Introduction",
             "Adsorption is a common method for removing dyes from waste water. "
             "Capacity and kinetics determine process cost."),
            ("Methods",
             "Batch experiments varied initial dye concentration. Residual "
             "concentration was measured by ultraviolet-visible spectroscopy."),
            ("Results",
             "The maximum adsorption capacity was 152 mg per gram. The data fit "
             "a Langmuir isotherm with a correlation above 0.99."),
            ("Conclusion",
             "Activated carbon reaches 152 mg per gram capacity for the dye."),
        ],
        "key_value": "max capacity = 152 mg/g",
    },
    {
        "slug": "geo_porosity",
        "title": "Porosity and Permeability of a Sandstone Core",
        "authors": "F. Castellano, B. Nwosu",
        "abstract": (
            "A sandstone core shows a porosity of 18 percent and a "
            "permeability of 45 millidarcy, measured by helium porosimetry "
            "and steady-state gas flow."
        ),
        "sections": [
            ("Introduction",
             "Porosity and permeability govern fluid flow in reservoir rocks "
             "and are central to resource and storage assessments."),
            ("Methods",
             "Core plugs were dried and measured by helium porosimetry. "
             "Permeability was found from steady-state nitrogen flow."),
            ("Results",
             "Porosity was 18 percent and permeability 45 millidarcy. The two "
             "correlate across the sampled depth interval."),
            ("Conclusion",
             "The sandstone shows 18 percent porosity and 45 millidarcy "
             "permeability, typical of a moderate-quality reservoir."),
        ],
        "key_value": "porosity = 18 percent",
    },
]


def build_pdf(paper: dict, path: Path) -> None:
    """Render one invented paper to a clean multi-section PDF."""
    doc = pymupdf.open()
    page = doc.new_page()  # default A4-ish
    margin = 56
    width = page.rect.width - 2 * margin
    y = margin

    def write_block(text: str, fontsize: float, fontname: str, gap: float,
                    wrap: int) -> None:
        nonlocal y, page
        for para in text.split("\n"):
            for line in textwrap.wrap(para, wrap) or [""]:
                if y > page.rect.height - margin:
                    page = doc.new_page()
                    y = margin
                page.insert_text((margin, y), line, fontsize=fontsize,
                                 fontname=fontname, color=(0, 0, 0))
                y += fontsize * 1.35
            y += gap

    write_block(paper["title"], 17, "helv", 6, 60)
    write_block(paper["authors"], 10.5, "helv", 10, 90)
    write_block("Abstract", 12, "helv", 2, 90)
    write_block(paper["abstract"], 10.5, "helv", 12, 95)

    for heading, body in paper["sections"]:
        write_block(heading, 12.5, "helv", 2, 90)
        write_block(body, 10.5, "helv", 10, 95)

    # A small results table so there is a visible value to contradict.
    write_block("Table 1. Key measured value", 10, "helv", 2, 90)
    write_block(paper["key_value"], 10.5, "cour", 12, 90)

    doc.set_metadata({
        "title": paper["title"],
        "author": paper["authors"],
        "subject": "Synthetic test paper for PDFGuard. Not a real publication.",
        "keywords": "test, synthetic, pdfguard",
        "creator": "pdfguard make_sample_papers",
    })
    doc.save(path, garbage=4, deflate=True)
    doc.close()


def main() -> None:
    default_out = str(Path(__file__).resolve().parent.parent / "data" / "clean")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=default_out, help="output directory")
    ap.add_argument("--n", type=int, default=len(PAPERS),
                    help="number of papers to generate (max %d)" % len(PAPERS))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    chosen = PAPERS[: max(1, min(args.n, len(PAPERS)))]
    for paper in chosen:
        path = out / f"{paper['slug']}.pdf"
        build_pdf(paper, path)
        print(f"wrote {path}  ({paper['key_value']})")

    print(f"\n{len(chosen)} clean sample papers written to {out}/")


if __name__ == "__main__":
    main()
