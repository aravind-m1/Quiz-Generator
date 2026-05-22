"""
PDF & Document Parser
=====================
Extracts structured text from PDF files, plain text files, and raw strings.
Uses PyMuPDF (fitz) as primary engine with pdfplumber as fallback for
table-heavy documents.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


# ─── Data class for parsed content ──────────────────────────────────────────────
class ParsedDocument:
    """Container for a parsed document with section-level granularity."""

    def __init__(
        self,
        title: str,
        full_text: str,
        sections: List[dict],
        metadata: Optional[dict] = None,
    ):
        self.title = title
        self.full_text = full_text
        self.sections = sections  # [{"heading": str, "content": str, "page": int}]
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return (
            f"ParsedDocument(title={self.title!r}, "
            f"sections={len(self.sections)}, "
            f"chars={len(self.full_text)})"
        )


# ─── Cleaning helpers ───────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Normalize whitespace, fix encoding artifacts, strip page numbers."""
    # Collapse multiple newlines / spaces
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Remove standalone page numbers (e.g., "  42  ")
    text = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", text)
    # Fix common PDF ligature artefacts
    replacements = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text.strip()


def _detect_heading(line: str) -> bool:
    """Heuristic: a line is a heading if it's short, title-cased / upper-cased,
    and does NOT end with a period."""
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return False
    if stripped.endswith((".", ",", ";", ":")):
        return False
    # Match patterns like "Chapter 3", "3.1 Introduction", "SECTION TITLE"
    if re.match(r"^(\d+\.?\d*\.?\d*\s+)?[A-Z]", stripped):
        words = stripped.split()
        if len(words) <= 12:
            return True
    if stripped.isupper() and len(stripped.split()) <= 10:
        return True
    return False


# ─── PDF Extraction (primary: PyMuPDF) ──────────────────────────────────────────

def _extract_with_fitz(pdf_path: str) -> ParsedDocument:
    """Extract text from a PDF using PyMuPDF (fitz)."""
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    title = doc.metadata.get("title", "") or Path(pdf_path).stem

    full_parts: list[str] = []
    sections: list[dict] = []
    current_heading = "Introduction"
    current_content: list[str] = []
    current_page = 1

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if not text:
            continue
        text = _clean_text(text)
        for line in text.split("\n"):
            if _detect_heading(line):
                # Flush previous section
                if current_content:
                    sections.append(
                        {
                            "heading": current_heading,
                            "content": "\n".join(current_content).strip(),
                            "page": current_page,
                        }
                    )
                current_heading = line.strip()
                current_content = []
                current_page = page_num
            else:
                current_content.append(line)
        full_parts.append(text)

    # Flush last section
    if current_content:
        sections.append(
            {
                "heading": current_heading,
                "content": "\n".join(current_content).strip(),
                "page": current_page,
            }
        )

    doc.close()
    full_text = _clean_text("\n".join(full_parts))
    return ParsedDocument(title=title, full_text=full_text, sections=sections)


def _extract_with_pdfplumber(pdf_path: str) -> ParsedDocument:
    """Fallback extractor using pdfplumber (better for tables)."""
    import pdfplumber

    full_parts: list[str] = []
    sections: list[dict] = []
    current_heading = "Introduction"
    current_content: list[str] = []
    current_page = 1

    with pdfplumber.open(pdf_path) as pdf:
        title = Path(pdf_path).stem
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = _clean_text(text)
            for line in text.split("\n"):
                if _detect_heading(line):
                    if current_content:
                        sections.append(
                            {
                                "heading": current_heading,
                                "content": "\n".join(current_content).strip(),
                                "page": current_page,
                            }
                        )
                    current_heading = line.strip()
                    current_content = []
                    current_page = page_num
                else:
                    current_content.append(line)
            full_parts.append(text)

    if current_content:
        sections.append(
            {
                "heading": current_heading,
                "content": "\n".join(current_content).strip(),
                "page": current_page,
            }
        )

    full_text = _clean_text("\n".join(full_parts))
    return ParsedDocument(title=title, full_text=full_text, sections=sections)


# ─── Plain text / raw string parsing ────────────────────────────────────────────

def parse_raw_text(
    text: str, title: str = "Untitled"
) -> ParsedDocument:
    """Parse a raw string of text into a ParsedDocument with heading detection."""
    text = _clean_text(text)
    sections: list[dict] = []
    current_heading = "Content"
    current_content: list[str] = []

    for line in text.split("\n"):
        if _detect_heading(line):
            if current_content:
                sections.append(
                    {
                        "heading": current_heading,
                        "content": "\n".join(current_content).strip(),
                        "page": 0,
                    }
                )
            current_heading = line.strip()
            current_content = []
        else:
            current_content.append(line)

    if current_content:
        sections.append(
            {
                "heading": current_heading,
                "content": "\n".join(current_content).strip(),
                "page": 0,
            }
        )

    # If no headings were detected, treat the whole text as one section
    if not sections:
        sections = [{"heading": "Content", "content": text, "page": 0}]

    return ParsedDocument(title=title, full_text=text, sections=sections)


def parse_text_file(file_path: str) -> ParsedDocument:
    """Read a .txt / .md file and parse it."""
    path = Path(file_path)
    text = path.read_text(encoding="utf-8", errors="replace")
    return parse_raw_text(text, title=path.stem)


# ─── Unified entry point ────────────────────────────────────────────────────────

def parse_document(source: str) -> ParsedDocument:
    """
    Detect the source type and parse accordingly.

    Parameters
    ----------
    source : str
        Either a file path (PDF, TXT) or raw text content (>200 chars heuristic).

    Returns
    -------
    ParsedDocument
    """
    path = Path(source)

    # If it looks like a file path and the file exists
    if path.exists() and path.is_file():
        ext = path.suffix.lower()
        if ext == ".pdf":
            try:
                return _extract_with_fitz(str(path))
            except Exception as exc:
                logger.warning(
                    "PyMuPDF extraction failed (%s), falling back to pdfplumber.",
                    exc,
                )
                return _extract_with_pdfplumber(str(path))
        elif ext in (".txt", ".md", ".rst"):
            return parse_text_file(str(path))
        else:
            # Attempt plain-text read for any other text-like file
            try:
                return parse_text_file(str(path))
            except Exception:
                raise ValueError(f"Unsupported file type: {ext}")

    # Otherwise treat as raw text
    return parse_raw_text(source)
