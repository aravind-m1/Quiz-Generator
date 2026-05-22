"""
Difficulty Classifier
======================
Classifies or adjusts the difficulty level of generated questions based on
Bloom's Taxonomy and linguistic complexity metrics.

This module serves two purposes:
  1. POST-GENERATION VERIFICATION: verify that a generated question actually
     matches the requested difficulty level.
  2. AUTOMATIC CLASSIFICATION: classify an untagged question into easy/medium/hard.

Scoring Dimensions:
  - Bloom's cognitive level (keyword matching)
  - Sentence complexity (average sentence length, subordinate clauses)
  - Vocabulary sophistication (syllable count, rare words)
  - Concept density (named entities / keyphrases per sentence)
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─── Bloom's Taxonomy Keywords ──────────────────────────────────────────────────

BLOOMS_LEVELS = {
    1: {
        "name": "Remember",
        "keywords": [
            "define", "list", "name", "identify", "recall", "recognize",
            "state", "label", "match", "memorize", "select",
        ],
    },
    2: {
        "name": "Understand",
        "keywords": [
            "describe", "explain", "summarize", "interpret", "paraphrase",
            "classify", "discuss", "illustrate", "report", "translate",
        ],
    },
    3: {
        "name": "Apply",
        "keywords": [
            "apply", "demonstrate", "implement", "solve", "use",
            "calculate", "execute", "construct", "modify", "operate",
        ],
    },
    4: {
        "name": "Analyze",
        "keywords": [
            "analyze", "compare", "contrast", "differentiate", "distinguish",
            "examine", "categorize", "deconstruct", "investigate", "relate",
        ],
    },
    5: {
        "name": "Evaluate",
        "keywords": [
            "evaluate", "justify", "critique", "assess", "argue",
            "defend", "judge", "support", "validate", "appraise",
        ],
    },
    6: {
        "name": "Create",
        "keywords": [
            "create", "design", "develop", "formulate", "hypothesize",
            "invent", "compose", "construct", "plan", "synthesize",
        ],
    },
}

# Mapping: Bloom's level → difficulty
_BLOOM_TO_DIFFICULTY = {
    1: "easy",
    2: "easy",
    3: "medium",
    4: "medium",
    5: "hard",
    6: "hard",
}


# ─── Syllable Counter ──────────────────────────────────────────────────────────

def _count_syllables(word: str) -> int:
    """Estimate syllable count of an English word."""
    word = word.lower().strip()
    if len(word) <= 3:
        return 1
    # Count vowel groups
    count = len(re.findall(r"[aeiouy]+", word))
    # Adjust for silent e
    if word.endswith("e") and not word.endswith("le"):
        count = max(1, count - 1)
    return max(1, count)


# ─── Linguistic Complexity Metrics ──────────────────────────────────────────────

def compute_linguistic_complexity(text: str) -> Dict[str, float]:
    """
    Compute linguistic complexity metrics for a piece of text.

    Returns
    -------
    Dict with keys:
        - avg_word_length: average word length in characters
        - avg_sentence_length: average number of words per sentence
        - avg_syllables_per_word: average syllable count per word
        - flesch_kincaid_grade: FK grade level estimate
        - complex_word_ratio: fraction of words with 3+ syllables
        - subordinate_clause_count: number of subordinate conjunctions
    """
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    words = re.findall(r"\b[a-zA-Z]+\b", text)

    if not words or not sentences:
        return {
            "avg_word_length": 0,
            "avg_sentence_length": 0,
            "avg_syllables_per_word": 0,
            "flesch_kincaid_grade": 0,
            "complex_word_ratio": 0,
            "subordinate_clause_count": 0,
        }

    total_syllables = sum(_count_syllables(w) for w in words)
    num_words = len(words)
    num_sentences = len(sentences)
    complex_words = sum(1 for w in words if _count_syllables(w) >= 3)

    avg_word_len = sum(len(w) for w in words) / num_words
    avg_sent_len = num_words / num_sentences
    avg_syl = total_syllables / num_words
    fk_grade = 0.39 * avg_sent_len + 11.8 * avg_syl - 15.59

    # Count subordinate conjunctions (indicates complex sentence structure)
    subordinate_conjs = [
        "although", "because", "since", "while", "whereas", "unless",
        "if", "though", "even though", "provided that", "in order to",
        "so that", "whenever", "wherever", "however",
    ]
    sub_count = sum(
        1
        for conj in subordinate_conjs
        if re.search(rf"\b{conj}\b", text.lower())
    )

    return {
        "avg_word_length": round(avg_word_len, 2),
        "avg_sentence_length": round(avg_sent_len, 2),
        "avg_syllables_per_word": round(avg_syl, 2),
        "flesch_kincaid_grade": round(fk_grade, 2),
        "complex_word_ratio": round(complex_words / num_words, 3),
        "subordinate_clause_count": sub_count,
    }


# ─── Bloom's Level Detection ───────────────────────────────────────────────────

def detect_blooms_level(question_text: str) -> Dict[str, Any]:
    """
    Detect the Bloom's Taxonomy cognitive level of a question
    based on keyword matching.

    Returns
    -------
    Dict with keys: 'level' (1-6), 'level_name', 'matching_keywords'
    """
    text_lower = question_text.lower()
    best_level = 1
    best_matches: list[str] = []

    for level, info in BLOOMS_LEVELS.items():
        matches = [kw for kw in info["keywords"] if kw in text_lower]
        if matches and level > best_level:
            best_level = level
            best_matches = matches

    # If no explicit keywords found, use sentence complexity as heuristic
    if not best_matches:
        metrics = compute_linguistic_complexity(question_text)
        if metrics["flesch_kincaid_grade"] > 12:
            best_level = 5
        elif metrics["flesch_kincaid_grade"] > 8:
            best_level = 3
        else:
            best_level = 1

    return {
        "level": best_level,
        "level_name": BLOOMS_LEVELS[best_level]["name"],
        "matching_keywords": best_matches,
    }


# ─── Difficulty Classifier ─────────────────────────────────────────────────────

class DifficultyClassifier:
    """
    Classifies the difficulty of a generated question using a composite score
    of Bloom's level, linguistic complexity, and concept density.
    """

    # Weights for the composite difficulty score
    WEIGHTS = {
        "blooms": 0.40,
        "fk_grade": 0.25,
        "complex_word_ratio": 0.15,
        "sentence_length": 0.10,
        "subordinate_clauses": 0.10,
    }

    # Thresholds for difficulty bins
    THRESHOLDS = {
        "easy": 0.33,
        "medium": 0.66,
        # anything above 0.66 → hard
    }

    def classify(self, question_text: str) -> Dict[str, Any]:
        """
        Classify the difficulty of a question.

        Parameters
        ----------
        question_text : str
            The question text to classify.

        Returns
        -------
        Dict with keys:
            - 'predicted_difficulty': easy / medium / hard
            - 'composite_score': float in [0, 1]
            - 'blooms': Bloom's analysis dict
            - 'linguistics': linguistic metrics dict
        """
        blooms = detect_blooms_level(question_text)
        linguistics = compute_linguistic_complexity(question_text)

        # Normalize each dimension to [0, 1]
        blooms_norm = (blooms["level"] - 1) / 5.0  # level 1-6 → 0-1
        fk_norm = min(linguistics["flesch_kincaid_grade"] / 16.0, 1.0)
        cwr_norm = min(linguistics["complex_word_ratio"] / 0.3, 1.0)
        slen_norm = min(linguistics["avg_sentence_length"] / 30.0, 1.0)
        sub_norm = min(linguistics["subordinate_clause_count"] / 4.0, 1.0)

        composite = (
            self.WEIGHTS["blooms"] * blooms_norm
            + self.WEIGHTS["fk_grade"] * fk_norm
            + self.WEIGHTS["complex_word_ratio"] * cwr_norm
            + self.WEIGHTS["sentence_length"] * slen_norm
            + self.WEIGHTS["subordinate_clauses"] * sub_norm
        )

        if composite <= self.THRESHOLDS["easy"]:
            predicted = "easy"
        elif composite <= self.THRESHOLDS["medium"]:
            predicted = "medium"
        else:
            predicted = "hard"

        return {
            "predicted_difficulty": predicted,
            "composite_score": round(composite, 4),
            "blooms": blooms,
            "linguistics": linguistics,
        }

    def verify_difficulty(
        self, question_text: str, expected_difficulty: str
    ) -> Dict[str, Any]:
        """
        Check if a question's actual difficulty matches the expected level.

        Returns
        -------
        Dict with keys: 'matches', 'expected', 'predicted', 'details'
        """
        result = self.classify(question_text)
        matches = result["predicted_difficulty"] == expected_difficulty

        return {
            "matches": matches,
            "expected": expected_difficulty,
            "predicted": result["predicted_difficulty"],
            "composite_score": result["composite_score"],
            "details": result,
        }
