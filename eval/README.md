# PDFGuard evaluation set

Scripts that build the test set PDFGuard is measured against: clean scientific
papers, and copies with hidden prompt-injection payloads inserted. Everything is
local and the payloads are clearly marked test strings.

## Responsible use

Every hidden payload begins with the sentinel `[PDFGUARD-TEST]` so it cannot be
mistaken for a real manipulation. Run these only on local, license-free or
self-generated PDFs, to test a detector. Do not apply them to real submissions,
reviews or applications. See the Responsible use section of the top-level
README.

## Scripts

| Script | What it does |
|---|---|
| `download_papers.py` | Downloads real open-access papers from arXiv (the maintainer's own paper plus a few materials-science papers), trimmed to their first pages. |
| `make_sample_papers.py` | Offline fallback: generates fully synthetic papers across six domains if no network is available. |
| `make_attacks.py` | For each input PDF, writes one copy per (attack type × payload kind) with a hidden payload, plus a `manifest.json`. |
| `verify_attacks.py` | Confirms every attacked file: the payload reaches a pipeline parser, and it is not perceptible on the rendered page. |
| `parser_leak.py` | Measures which PDF parsers leak each attack into an LLM pipeline (fills the parser-comparison table). |
| `score_attacks.py` | Scores the detector's detection and false-alarm rates; until the detector exists, reports the undefended attack-success baseline. |

## Layout

Code lives in this `eval/` folder; the generated data lives at the repo root in
`data/` (shared with the detector stage), and results in `reports/`:

```
eval/                        # this folder: code only
├── download_papers.py       # fetch real papers from arXiv
├── make_sample_papers.py    # offline synthetic fallback
├── make_attacks.py          # inject hidden payloads
├── verify_attacks.py        # confirm the test set is valid
├── parser_leak.py           # parser-leak comparison
├── score_attacks.py         # detection / baseline scoring
├── requirements.txt
└── README.md
data/                        # repo root: generated, gitignored
├── clean/                   # real (or synthetic) source papers
└── attacked/                # one subfolder per source paper + manifest.json
    ├── own_from_knowledge.../    # 16 files: 8 attacks x 2 payload kinds
    ├── arxiv_large_anisotropic.../
    └── manifest.json
reports/                     # CSV outputs, gitignored
```

Attacked copies are grouped into one subfolder per source paper, named
`<attack>__<kind>.pdf`. The manifest's `output` field is the path relative to
`data/attacked/`, e.g. `own_from_knowledge.../white_text__data.pdf`.

## Quick start

The scripts default to the repo-root `data/`, so they run from anywhere with no
arguments:

```bash
pip install -r eval/requirements.txt
python eval/download_papers.py      # or make_sample_papers.py if offline
python eval/make_attacks.py
python eval/verify_attacks.py
python eval/parser_leak.py --csv reports/parsers.csv
python eval/score_attacks.py
```

With the default five papers this produces 80 attacked copies (5 papers × 8
attacks × 2 payload kinds). Paths can be overridden as arguments.

### Papers used

`download_papers.py` fetches the maintainer's own arXiv paper (the safest
real-world case) plus the newest few `cond-mat.mtrl-sci` papers for layout
diversity, and trims each to its first 8 pages (`--max-pages`). Trimming keeps
the real page-1 layout and fonts, where the attacks land, while keeping
image-heavy PDFs small enough for a fast eval loop.

## Attack types

Each attack hides a payload a human cannot read while a parser still extracts
it. Two payload kinds are emitted for every attack: an `instruction` (aimed at a
downstream model) and a `data` value (a fabricated number meant to override the
visible one).

| Attack | Hiding mechanism | Channel a detector reads |
|---|---|---|
| `white_text` | Text coloured to match the page background | page text span colour |
| `tiny_font` | 0.6 pt glyphs (an unreadable speck of ink) | page text span size |
| `off_page` | Drawn 50 pt past the right page edge | span bbox outside the cropbox |
| `covered_text` | Opaque white box drawn over the text | ink check / OCR under image |
| `invisible_render` | PDF text render mode 3, never painted | span render mode |
| `metadata` | Stored in subject and keywords fields | document metadata |
| `annotation` | Free-text annotation near the page edge | annotation contents |
| `unicode` | Zero-width characters laced through the text | zero-width code points in text |

Hidden text is placed on a genuinely blank, colour-uniform band of page 1 (found
by rendering the page), so white or tiny text lands on empty space and stays
invisible even on real papers with figures and tinted backgrounds.

### Notes on fidelity

- **off_page** is clipped out by PyMuPDF's default extraction (it honours the
  cropbox) but is extracted by `pypdf` and `pdfplumber`, the parsers scientific
  pipelines commonly use. Even with a clip larger than the page, PyMuPDF still
  fails to recover it on some documents (see results), so a detector should not
  rely on PyMuPDF alone for off-page text.
- **tiny_font** leaves about 0.007% of page pixels changed at 150 dpi, far below
  anything a reader perceives. The verifier treats a sub-0.1% perceptible change
  as hidden.
- **unicode** needs a font whose cmap carries the zero-width code points
  (DejaVuSans, FreeSans or Liberation, searched automatically); otherwise the
  characters would be dropped by a base-14 font and the payload falls back to
  the metadata keywords channel so the sample stays a true positive.
- The verifier measures a *perceptible* pixel change (grayscale difference above
  24 of 255), not any byte difference, so anti-aliasing of background-matched
  glyphs does not count as visible.

## Results so far

On the default set of five real arXiv papers (80 attacked files).

**Test-set validity** (`verify_attacks.py`): all 80 files pass. Every payload
reaches a parser, and none is perceptible on the rendered page.

**Which parsers leak hidden content** (`parser_leak.py`): percentage of files where
the payload survives into that parser's output.

| Attack | PyMuPDF | pdfplumber | pypdf |
|---|---|---|---|
| white_text | 100% | 100% | 100% |
| tiny_font | 100% | 100% | 100% |
| off_page | 80% | 100% | 100% |
| covered_text | 100% | 0% | 100% |
| invisible_render | 100% | 100% | 100% |
| metadata | 100% | 100% | 100% |
| annotation | 100% | 100% | 0% |
| unicode | 100% | 100% | 100% |

Notes: PyMuPDF misses off-page text on one of the five papers even with an
oversized clip. pdfplumber does not surface the covered-text payload, and pypdf
`extract_text` does not include annotation contents (both are real extractor
limitations, not missing payloads: the text is present in the file).

**Undefended baseline** (`score_attacks.py`, no detector yet): 80/80 (100%) of
payloads reach a standard pipeline parser. This is the attack-success rate
PDFGuard must drive to zero. Once `pdfguard.scan` exists, the same script
reports detection and false-alarm rates instead.

## Manifest

`make_attacks.py` writes `manifest.json` in the output directory, one row per
file with its source, attack, payload kind, exact payload text and a short
description of how it was hidden. Use it as ground truth when scoring the
detector.
