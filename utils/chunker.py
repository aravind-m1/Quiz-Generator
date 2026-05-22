"""
Semantic Text Chunker
=====================
Splits parsed documents into overlapping, semantically coherent chunks
suitable for embedding and retrieval in the RAG pipeline.

Strategy:
  1. Sentence-level tokenisation (via NLTK punkt).
  2. Accumulate sentences until CHUNK_SIZE words are reached.
  3. Apply CHUNK_OVERLAP by carrying trailing sentences forward.
  4. Respect section boundaries – never merge text from different sections.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

import nltk

# Ensure punkt tokeniser is available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

from nltk.tokenize import sent_tokenize

from utils.pdf_parser import ParsedDocument

logger = logging.getLogger(__name__)


# ─── Chunk data class ───────────────────────────────────────────────────────────

@dataclass
class TextChunk:
    """A single chunk of text with provenance metadata."""

    chunk_id: int
    text: str
    word_count: int
    source_heading: str
    source_page: int
    start_sentence_idx: int
    end_sentence_idx: int
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.text[:80].replace("\n", " ")
        return f"TextChunk(id={self.chunk_id}, words={self.word_count}, " \
               f"heading={self.source_heading!r}, preview={preview!r}…)"


# ─── Helpers ─────────────────────────────────────────────────────────────────────

def _word_count(text: str) -> int:
    return len(text.split())


def _clean_sentence(sentence: str) -> str:
    """Light normalisation for individual sentences."""
    sentence = re.sub(r"\s+", " ", sentence)
    return sentence.strip()


# ─── Core chunking logic ────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = 400,
    chunk_overlap: int = 80,
    heading: str = "Content",
    page: int = 0,
    start_id: int = 0,
) -> List[TextChunk]:
    """
    Split a block of text into overlapping chunks.

    Parameters
    ----------
    text : str
        The text to chunk.
    chunk_size : int
        Target number of words per chunk.
    chunk_overlap : int
        Number of overlapping words between consecutive chunks.
    heading : str
        Source heading for provenance tracking.
    page : int
        Source page number.
    start_id : int
        Starting chunk ID for numbering continuity.

    Returns
    -------
    List[TextChunk]
    """
    sentences = [_clean_sentence(s) for s in sent_tokenize(text) if s.strip()]

    if not sentences:
        return []

    chunks: list[TextChunk] = []
    current_sentences: list[str] = []
    current_wc = 0
    sent_start_idx = 0
    chunk_id = start_id

    for i, sentence in enumerate(sentences):
        s_wc = _word_count(sentence)
        current_sentences.append(sentence)
        current_wc += s_wc

        if current_wc >= chunk_size:
            chunk_text_str = " ".join(current_sentences)
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    text=chunk_text_str,
                    word_count=_word_count(chunk_text_str),
                    source_heading=heading,
                    source_page=page,
                    start_sentence_idx=sent_start_idx,
                    end_sentence_idx=i,
                )
            )
            chunk_id += 1

            # Calculate overlap: walk backwards to capture ~chunk_overlap words
            overlap_sentences: list[str] = []
            overlap_wc = 0
            for s in reversed(current_sentences):
                overlap_wc += _word_count(s)
                overlap_sentences.insert(0, s)
                if overlap_wc >= chunk_overlap:
                    break

            sent_start_idx = i - len(overlap_sentences) + 1
            current_sentences = overlap_sentences
            current_wc = overlap_wc

    # Flush remaining sentences
    if current_sentences:
        chunk_text_str = " ".join(current_sentences)
        # Only add if this isn't a near-duplicate of the last chunk
        if not chunks or chunk_text_str != chunks[-1].text:
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    text=chunk_text_str,
                    word_count=_word_count(chunk_text_str),
                    source_heading=heading,
                    source_page=page,
                    start_sentence_idx=sent_start_idx,
                    end_sentence_idx=len(sentences) - 1,
                )
            )

    return chunks


# ─── Document-level chunking ────────────────────────────────────────────────────

def chunk_document(
    document: ParsedDocument,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[TextChunk]:
    """
    Chunk an entire ParsedDocument, respecting section boundaries.

    Parameters
    ----------
    document : ParsedDocument
        Output from the pdf_parser module.
    chunk_size : int, optional
        Override config.CHUNK_SIZE.
    chunk_overlap : int, optional
        Override config.CHUNK_OVERLAP.

    Returns
    -------
    List[TextChunk]
    """
    from config import CHUNK_SIZE, CHUNK_OVERLAP

    size = chunk_size or CHUNK_SIZE
    overlap = chunk_overlap or CHUNK_OVERLAP

    all_chunks: list[TextChunk] = []
    running_id = 0

    for section in document.sections:
        content = section.get("content", "")
        if not content or _word_count(content) < 20:
            continue  # skip trivially short sections

        section_chunks = chunk_text(
            text=content,
            chunk_size=size,
            chunk_overlap=overlap,
            heading=section.get("heading", "Unknown"),
            page=section.get("page", 0),
            start_id=running_id,
        )
        all_chunks.extend(section_chunks)
        if section_chunks:
            running_id = section_chunks[-1].chunk_id + 1

    logger.info(
        "Chunked document '%s' into %d chunks (size=%d, overlap=%d).",
        document.title,
        len(all_chunks),
        size,
        overlap,
    )
    return all_chunks


def chunk_raw_text(
    text: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[TextChunk]:
    """
    Convenience wrapper: chunk a raw string without parsing sections.
    """
    from config import CHUNK_SIZE, CHUNK_OVERLAP

    size = chunk_size or CHUNK_SIZE
    overlap = chunk_overlap or CHUNK_OVERLAP
    return chunk_text(text, chunk_size=size, chunk_overlap=overlap)
