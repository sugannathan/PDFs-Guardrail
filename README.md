# PDFGuard

**Prompt injection and hidden content detection for scientific PDFs, before they reach your LLM pipeline.**

Built at the [Open Scientific Intelligence (OSI) Hackathon 2026](https://llmhackathon.github.io/) · Category: Knowledge Extraction and Reasoning

---

## The problem

LLMs increasingly read scientific papers for us. They summarise literature, extract values into materials and chemistry databases, answer questions in RAG assistants and help with peer review. Every one of these pipelines starts the same way: a PDF is parsed into text, and the text goes into the prompt.

PDFs can contain text that a human reader never sees: white text, fonts smaller than a dot, text outside the page, text under images, invisible text layers, switched-off layers, metadata, annotations, form fields, embedded files and invisible Unicode characters. PDF parsers pass much of this to the model, and the model reads it like any other sentence. This makes documents a vehicle for **prompt injection**.

This is not hypothetical. In 2025, hidden prompts telling AI reviewers to give only positive feedback were found in published papers, and job applicants use the same trick in CVs. The same method can slip **fake values** into extracted data: a hidden line saying "the corrected band gap is 4.8 eV" can override the 3.2 eV a scientist sees in the table. Hidden content is also not always malicious: template leftovers and OCR layers in scanned papers end up in LLM input too.

## What PDFGuard does

PDFGuard runs **before** a document enters an LLM pipeline and answers one question: *does this file contain text that the model will read but a human cannot see?*

```
            ┌─ text spans (font, size, colour, opacity, render mode, layer, position)
            ├─ images and drawings
 PDF ──►  split ─ annotations, form fields, links, JavaScript
            ├─ metadata (Info dictionary + XMP)
            └─ embedded files
                       │
                       ▼
        check each type for hidden content (code, no LLM)
                       │
                       ▼
        label each finding: injection / data / harmless / unknown
        (heuristics, optionally an LLM)
                       │
                       ▼
        report + risk level + cleaned text (only what a human can see)
```

For each PDF you get:

1. **The document split into content types** (`pdfguard split`)
2. **A list of findings** with page, position, the reasons and the text
3. **A label per finding:** `injection`, `data` (and whether it contradicts visible values), `harmless` or `unknown`
4. **A risk level:** `clean`, `suspicious` or `dangerous`
5. **A cleaned text** with only visible content, to use instead of raw parser output

## How the checks work

### Text: the ink check

For every page we render two images: the page as a reader sees it, and the same page with all text removed (images and drawings kept). The difference between them is exactly the ink produced by text. A text span whose area shows no difference is invisible, **whatever trick was used**. The text-free image also gives the real background colour behind each span.

On top of that, each span is checked for:

| Reason | Meaning |
|---|---|
| `no_ink` | Text leaves no visible marks (white, covered, transparent, ...) |
| `color_matches_background` | Text colour equals the actual background behind it |
| `tiny_font` | Font size below 2 pt |
| `off_page` | More than half of the text lies outside the page |
| `invisible_render_mode` | Stored but not painted (render mode 3, used for OCR layers) |
| `transparent` | Fill opacity close to zero |
| `hidden_layer` | Inside an optional content layer that is switched off |
| `low_contrast` | Visible, but extremely faint |
| `invisible_unicode` | Zero-width, bidi-override or Unicode tag characters |

Messages encoded in invisible characters are decoded: Unicode tag characters (U+E0000 block, "ASCII smuggling") and binary zero-width encodings (ZWSP = 0, ZWNJ = 1).

**OCR layers in scanned papers** are invisible by design. With `--ocr`, invisible text lying on an image is compared with Tesseract's reading of that image: if they match, it is a harmless OCR layer; if not, it is flagged.

### Content that is never drawn on the page

Metadata and XMP fields, annotation contents (hidden or not), form field values, embedded files, `javascript:` links and JavaScript actions. Metadata is only reported when it is long, looks like an instruction or data, or carries invisible characters, so ordinary titles and authors don't cause alarms.

### Labelling

The **heuristic classifier** always runs:

- `injection`: instruction-like wording addressed to an AI ("ignore previous instructions", "as an AI reviewer", "rate this paper", ...), or JavaScript
- `data`: statements with quantities and units. If the same unit appears in the visible text with a different number, the finding is marked `contradicts_visible`
- `harmless`: duplicates of visible text, OCR layers that match the image, no words (glyph tables, stray symbols), a few zero-width characters
- `unknown`: hidden text that matches none of these; review manually

When a Cerebras key is configured, **gpt-oss-120b** labels every finding (most severe first, up to 25 per document). It sees only the isolated snippet, how it was hidden, and visible text from the same page, wrapped as untrusted data with the instruction never to follow it. It returns JSON with label, confidence and a one-sentence explanation.

The LLM can raise a label (e.g. `unknown` to `injection`) but **cannot lower** one: if the heuristics say `injection` and the LLM says `harmless`, the finding stays dangerous and the report shows the disagreement. A hidden "this text is harmless" therefore cannot talk its way out of a finding.

**Why the LLM does not do the detecting:** an LLM only sees extracted text, so it cannot know what is visible, and it can be injected by the very text it is checking. In PDFGuard, visibility is decided by code. A fooled LLM can mislabel a finding but cannot make it disappear.

### Risk level

- `dangerous`: any injection, any data value contradicting visible values, or a decoded hidden Unicode message
- `suspicious`: hidden data or unknown hidden content
- `clean`: nothing hidden, or only harmless findings

## Installation

```bash
git clone https://github.com/<your-org>/pdfguard.git
cd pdfguard
pip install -r requirements.txt
cp .env.example .env    # then paste your Cerebras key into .env
# optional, for --ocr: brew install tesseract   (or apt install tesseract-ocr)
```

## Usage

### Command line

```bash
# split a PDF into its content types (one file per type)
python -m pdfguard split paper.pdf --out parts/ --images

# scan one file or a whole folder
python -m pdfguard scan paper.pdf
python -m pdfguard scan data/attacked/ --out reports/ --ocr

# also write the split parts next to each report
python -m pdfguard scan data/attacked/ --out reports/ --split

# heuristic labels only (no API calls)
python -m pdfguard scan paper.pdf --no-llm

# visible-only text for your pipeline
python -m pdfguard clean paper.pdf --out paper_clean.txt
```

The exit code of `scan` is `0` (clean), `1` (suspicious) or `2` (dangerous), so it can gate a pipeline or CI job.

Example output:

```
[FAIL] paper.pdf  risk=dangerous  (injection=0, data=1, unknown=0, harmless=0, hidden spans=1/14)
   - data      p1     no_ink,tiny_font    'The corrected band gap of the TiO2-Pt sample is 4.8 eV.'
     Hidden value(s) 4.8 ev differ from visible value(s) 3.2 ev, 3.3 ev
```

### Cerebras setup (LLM labelling)

Findings are labelled by **gpt-oss-120b on [Cerebras](https://cloud.cerebras.ai)**. The key is never in the code: put it in a `.env` file, which git ignores.

```bash
cp .env.example .env
# open .env and paste your key after CEREBRAS_API_KEY=
python -m pdfguard llm-check        # sends one test snippet, prints OK or the error
```

Once the key is set, `scan` uses the LLM automatically. Use `--no-llm` for heuristics only. To use a different Cerebras model, set `PDFGUARD_MODEL` in `.env`.

If Cerebras is unreachable or rate-limited (PDFGuard retries 3 times), the heuristic labels are kept and the report says why the LLM was not used. Detection itself never depends on the LLM.

### Python

```python
from pdfguard import scan, split_pdf

parts = split_pdf("paper.pdf")          # all content, split by type
print(parts.summary())

report = scan("paper.pdf", use_ocr=True)   # uses Cerebras if the key is in .env
print(report.risk)                      # "clean", "suspicious" or "dangerous"
for f in report.findings:
    print(f.page, f.reasons, f.label, f.text)

text_for_llm = report.clean_text        # only what a human can see
```

## Split output

`pdfguard split` writes one file per content type:

| File | Contents |
|---|---|
| `text_spans.jsonl` | Every text span: text, bbox, font, size, colour, opacity, render mode, layer |
| `images.json`, `images/` | Image placements (and the images with `--images`) |
| `drawings.json` | Filled vector shapes (can cover text) |
| `annotations.json` | Type, contents, author, hidden flag |
| `form_fields.json` | Field names, values, hidden flag |
| `links.json` | Link targets |
| `metadata.json`, `xmp.xml` | Info dictionary and XMP metadata |
| `embedded_files.json` | Attached files with a text preview |
| `javascript.json` | Objects containing JavaScript |
| `layers.json` | Optional content layers and whether they are on |
| `summary.json` | Count per type |

## Evaluation

```bash
# detection rate per attack type and false alarms on clean papers
python eval/run_eval.py --attacked data/attacked --clean data/clean --out results/ --ocr

# which parsers pass hidden text on to the LLM?
python eval/parsers.py data/attacked --out results/parsers.csv
```

Name attacked files `<paper>__<attack_type>.pdf` (e.g. `paper07__white_text.pdf`) to get results per attack type.

### First results (synthetic test set)

`tests/make_samples.py` builds a small "paper" and one variant per hiding method. These are sanity checks, not the main benchmark:

| Attack | Detected |
|---|---|
| White text | ✓ |
| Tiny font (0.5 pt) | ✓ |
| Off-page text | ✓ |
| Text covered by an image | ✓ |
| Invisible render mode | ✓ |
| Zero opacity | ✓ |
| Same colour as a coloured background | ✓ |
| Switched-off layer | ✓ |
| Metadata (Info) | ✓ |
| XMP metadata | ✓ |
| Hidden annotation | ✓ |
| Hidden form field | ✓ |
| Zero-width encoded message (decoded) | ✓ |
| Unicode tag characters in metadata (decoded) | ✓ |
| Embedded file | ✓ |
| Tampered OCR layer on a scanned page | ✓ |
| **False alarms**: clean sample, benign OCR layer, a LaTeX two-column paper and 25 font documentation PDFs | **0 / 28** |

Which parsers passed the hidden text on (same test set):

| Hidden by | PyMuPDF | pymupdf4llm | pdfplumber | pypdf |
|---|---|---|---|---|
| White text | passed | blocked | passed | passed |
| Same colour as background | passed | passed | passed | passed |
| Tiny font | passed | blocked | passed | passed |
| Zero opacity | passed | blocked | passed | passed |
| Invisible render mode | passed | blocked | passed | passed |
| Covered by image | passed | passed | passed | passed |
| Off-page | blocked | blocked | passed | passed |
| Switched-off layer | blocked | blocked | passed | passed |
| Tampered OCR layer | passed | passed | passed | passed |

Results on the real attacked paper set will be added here.

## Repository structure

```
pdfguard/
├── pdfguard/
│   ├── split.py       # split a PDF into content types
│   ├── ink.py         # render with and without text, ink and background per span
│   ├── checks.py      # hidden-content checks for every content type
│   ├── classify.py    # heuristic labels + gpt-oss on Cerebras
│   ├── config.py      # reads CEREBRAS_API_KEY from .env
│   ├── report.py      # scan(), risk level, clean text, text report
│   └── cli.py         # split / scan / clean commands
├── eval/
│   ├── run_eval.py    # detection and false-alarm rates
│   └── parsers.py     # parser comparison
├── tests/
│   ├── make_samples.py
│   └── test_samples.py
└── README.md
```

Run the tests with `pytest -q`.

## Limitations

- Text that is tiny or faint but technically visible is a grey zone; thresholds are in `checks.py` and `ink.py`.
- Hidden text placed exactly on top of visible text of the same words is not separated by the ink check; the colour, opacity and render-mode rules cover the common cases.
- Text inside images (e.g. in figures) is not analysed.
- Unicode tag characters only survive in PDFs whose fonts map them; they are always caught in metadata, annotations and form fields.
- The heuristic classifier is English-only; use `--llm` for other languages.
- PDF only for now. Supplementary files such as CIF and XLSX are planned.

## Roadmap

- [ ] Supplementary files: CIF comment fields, hidden sheets and cells in XLSX
- [ ] Mark every value extracted by a pipeline as visible or hidden
- [ ] Prevalence scan over recent arXiv papers (aggregate results only)
- [ ] Hugging Face Space for drag-and-drop checks
- [ ] MCP tool so agents can check documents before reading them

## Responsible use

The sample generator exists only to build test data for evaluating defences. Please do not use it, or any attack set, to manipulate real submissions, reviews or applications. If you run PDFGuard on published papers and find hidden content, report results in aggregate and contact authors or publishers rather than naming individual papers publicly.

## Team

- **Sugan Kanagasenthinathan** – sugan.kanagasenthinathan@dlr.de
- **Prakul Pandit** – prakul.pandit@dlr.de

## License

MIT (to be confirmed)
