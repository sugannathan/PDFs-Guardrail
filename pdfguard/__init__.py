"""PDFGuard: find hidden content and prompt injections in PDFs before they reach an LLM."""
from .report import Report, format_report, scan
from .split import DocumentParts, PDFGuardError, split_pdf, write_parts

__all__ = ["scan", "Report", "format_report", "split_pdf", "write_parts", "DocumentParts", "PDFGuardError"]
__version__ = "0.2.0"
