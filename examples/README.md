# Test PDFs

`00_clean.pdf` and `real_latex_paper_clean.pdf` contain nothing hidden and should come back **clean**.
`15_scanned_ocr_layer_benign.pdf` is a scanned page with a normal OCR layer and should also be **clean** (with `--ocr`).
Every other file hides an injection or a fake value in a different way (see the file name) and should be **dangerous**.

    python -m pdfguard scan examples/ --ocr --quiet

Regenerate them with `python tests/make_samples.py examples`.
