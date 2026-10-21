# PDFs-Guardrail

PDFGuard

LLMs increasingly read scientific papers for us: they summarise literature, extract values into materials databases and assist with peer review. This makes them a target for prompt injection through documents. Text can be hidden in a PDF using white font, tiny sizes, content outside the page, metadata or annotations. A human reader never sees it, but the PDF parser passes it to the model, which reads it like any other instruction. In 2025, hidden prompts telling AI reviewers to give positive reviews were found in published papers, and the same trick can be used to slip fake values into extracted data.

PDFGuard checks a PDF before it enters an LLM pipeline. It compares what is visible on the rendered page with what the parser extracts and flags all hidden content. An LLM then classifies each hidden snippet as a prompt injection, a fake data value or a harmless leftover. PDFGuard also returns a cleaned text containing only what a human can see, so pipelines can safely use that instead.
