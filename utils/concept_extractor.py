"""
Key Concept & Entity Extractor
===============================
Extracts the most important concepts, entities, and keyphrases from text
chunks to guide targeted question generation.

Uses a multi-strategy approach:
  1. KeyBERT  – transformer-based keyphrase extraction (primary)
  2. spaCy NER – named entity recognition (supplementary)
  3. TF-IDF fallback – lightweight keyword scoring when models unavailable
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class ExtractedConcepts:
    """Container for extracted concepts from a text chunk."""

    keyphrases: List[str] = field(default_factory=list)
    named_entities: List[dict] = field(default_factory=list)  # {"text", "label"}
    top_terms: List[str] = field(default_factory=list)
    source_text_preview: str = ""

    @property
    def all_concepts(self) -> List[str]:
        """Deduplicated union of all extracted concepts."""
        seen: Set[str] = set()
        result: list[str] = []
        for phrase in self.keyphrases + self.top_terms:
            normalised = phrase.lower().strip()
            if normalised and normalised not in seen:
                seen.add(normalised)
                result.append(phrase.strip())
        for ent in self.named_entities:
            normalised = ent["text"].lower().strip()
            if normalised not in seen:
                seen.add(normalised)
                result.append(ent["text"].strip())
        return result


# ─── KeyBERT-based Extraction ───────────────────────────────────────────────────

_keybert_model = None


def _get_keybert():
    """Lazy-load KeyBERT model (singleton)."""
    global _keybert_model
    if _keybert_model is None:
        try:
            from keybert import KeyBERT
            _keybert_model = KeyBERT(model="all-MiniLM-L6-v2")
            logger.info("KeyBERT model loaded successfully.")
        except ImportError:
            logger.warning("keybert not installed; keyphrase extraction will fallback.")
            _keybert_model = False  # sentinel to avoid re-trying
    return _keybert_model


def extract_keyphrases(
    text: str, top_n: int = 10, diversity: float = 0.5
) -> List[str]:
    """
    Extract keyphrases using KeyBERT with Maximal Marginal Relevance
    for diversity.

    Parameters
    ----------
    text : str
        Input text to extract keyphrases from.
    top_n : int
        Number of keyphrases to return.
    diversity : float
        MMR diversity parameter (0 = no diversity, 1 = max diversity).

    Returns
    -------
    List[str]
        Ranked list of keyphrases.
    """
    model = _get_keybert()
    if not model:
        return _fallback_tfidf_keywords(text, top_n)

    try:
        keywords = model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            use_mmr=True,
            diversity=diversity,
            top_n=top_n,
        )
        return [kw for kw, _score in keywords]
    except Exception as exc:
        logger.warning("KeyBERT extraction failed: %s", exc)
        return _fallback_tfidf_keywords(text, top_n)


# ─── spaCy NER Extraction ──────────────────────────────────────────────────────

_spacy_nlp = None


def _get_spacy():
    """Lazy-load spaCy model (singleton)."""
    global _spacy_nlp
    if _spacy_nlp is None:
        try:
            import spacy
            _spacy_nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded (en_core_web_sm).")
        except (ImportError, OSError):
            logger.warning("spaCy model not available; NER will be skipped.")
            _spacy_nlp = False
    return _spacy_nlp


def extract_named_entities(text: str) -> List[dict]:
    """
    Extract named entities using spaCy.

    Returns
    -------
    List[dict]
        Each dict has keys 'text' and 'label' (e.g., PERSON, ORG, DATE).
    """
    nlp = _get_spacy()
    if not nlp:
        return []

    try:
        doc = nlp(text[:100000])  # spaCy has a max length; truncate for safety
        entities = []
        seen: set[str] = set()
        for ent in doc.ents:
            key = f"{ent.text.lower()}|{ent.label_}"
            if key not in seen:
                seen.add(key)
                entities.append({"text": ent.text, "label": ent.label_})
        return entities
    except Exception as exc:
        logger.warning("spaCy NER failed: %s", exc)
        return []


# ─── TF-IDF Fallback ────────────────────────────────────────────────────────────

# Simple English stop words for the fallback
_STOP_WORDS = set(
    "a an the is are was were be been being have has had do does did "
    "will would shall should may might can could of in to for on with "
    "at by from as into through during before after above below between "
    "out off over under again further then once here there when where "
    "why how all each every both few more most other some such no not "
    "only own same so than too very and but or nor if it its this that "
    "these those i me my myself we our ours ourselves you your yours "
    "he him his she her hers they them their what which who whom".split()
)


def _fallback_tfidf_keywords(text: str, top_n: int = 10) -> List[str]:
    """
    Lightweight keyword extraction based on term frequency when
    KeyBERT is unavailable.
    """
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    filtered = [w for w in words if w not in _STOP_WORDS]
    counter = Counter(filtered)
    return [word for word, _count in counter.most_common(top_n)]


# ─── Unified Extraction ─────────────────────────────────────────────────────────

def extract_concepts(
    text: str,
    top_n: int = 10,
    use_ner: bool = True,
    diversity: float = 0.5,
) -> ExtractedConcepts:
    """
    Extract key concepts from text using all available strategies.

    Parameters
    ----------
    text : str
        The input text.
    top_n : int
        Number of keyphrases to extract.
    use_ner : bool
        Whether to also run named entity recognition.
    diversity : float
        MMR diversity for KeyBERT.

    Returns
    -------
    ExtractedConcepts
    """
    keyphrases = extract_keyphrases(text, top_n=top_n, diversity=diversity)
    entities = extract_named_entities(text) if use_ner else []
    fallback_terms = _fallback_tfidf_keywords(text, top_n=top_n)

    return ExtractedConcepts(
        keyphrases=keyphrases,
        named_entities=entities,
        top_terms=fallback_terms,
        source_text_preview=text[:200],
    )
