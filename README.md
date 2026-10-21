# PDFGuard

**Prompt injection and hidden content detection for scientific PDFs, before they reach your LLM pipeline.**

Built at the [Open Scientific Intelligence (OSI) Hackathon 2026](https://llmhackathon.github.io/) · Category: Knowledge Extraction and Reasoning

---

## The problem

LLMs increasingly read scientific papers for us. They summarise literature, extract values into materials and chemistry databases, answer questions in RAG assistants and help with peer review. Every one of these pipelines starts the same way: a PDF is parsed into text, and the text goes into the prompt.

PDFs can contain text that a human reader never sees:

- white text on a white background
- fonts so small they look like a dot
- text placed outside the visible page
- text hidden behind images
- invisible text layers
- metadata, annotations and form fields
- zero-width Unicode characters

PDF parsers pass all of this to the model, and the model reads it like any other sentence. This makes documents a vehicle for **prompt injection**.

This is not hypothetical. In 2025, hidden prompts telling AI reviewers to give only positive feedback were found in published papers. Job applicants use the same trick in CVs to get past automated screening. And the same method can slip **fake values** into extracted data: a hidden line saying "the corrected band gap is 4.8 eV" can override the 3.2 eV a scientist sees in the table.

Hidden content is also not always malicious. Template leftovers, OCR layers in scanned papers and old annotations end up in LLM input too, and they can corrupt extraction just as easily.

Nothing in a typical pipeline checks for any of this.

## What PDFGuard does

PDFGuard runs **before** a document enters an LLM pipeline and answers one question: *does this file contain text that the model will read but a human cannot see?*
find hidden content, report results in aggregate and contact authors or publishers rather than naming individual papers publicly.

## Team

- **Sugan Kanagasenthinathan** – sugan.kanagasenthinathan@dlr.de
- **Prakul Pandit** – prakul.pandit@dlr.de

## License

MIT (to be confirmed)

## Acknowledgements

Thanks to the organisers, mentors and sponsors of the OSI Hackathon 2026.
