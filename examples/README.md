# Test PDFs

`00_clean.pdf` and `real_latex_paper_clean.pdf` contain nothing hidden and should come back **clean**.
`15_scanned_ocr_layer_benign.pdf` is a scanned page with a normal OCR layer and should also be **clean** (with `--ocr`).
`17_long_author_list_clean.pdf` (hundreds of authors in the metadata) and `18_link_url_contains_js_clean.pdf` (a link to a URL containing `/JS`) are clean documents that used to cause false alarms and should be **clean**.
`24_figure_leftovers_clean.pdf` and `25_white_label_on_dark_figure_clean.pdf` should also be **clean**.
`26_latex_tiny_white_prompt.pdf` is a real LaTeX file with a 0.1 pt white reviewer prompt and no stored spaces, built like the 2025 arXiv cases; it should be **dangerous**.
`20`–`23` are honeypot traps modelled on real cases collected by [pdf-injection-scanner](https://github.com/Andy8647/pdf-injection-scanner) and should be **dangerous**.
Every other file hides an injection or a fake value in a different way (see the file name) and should be **dangerous**.

    python -m pdfguard scan examples/ --ocr --quiet

Regenerate them with `python tests/make_samples.py examples`.
