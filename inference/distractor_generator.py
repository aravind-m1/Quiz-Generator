"""
Distractor Generator
=====================
Generates plausible but incorrect options (distractors) for MCQ questions.

Strategies:
  1. LLM-based generation  – ask the model to produce near-miss answers
  2. Embedding similarity  – find semantically close but distinct terms from
     the vector store
  3. Entity substitution   – swap the correct named entity with related entities
     from the same category
"""

from __future__ import annotations

import logging
import random
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─── Prompt for distractor generation ───────────────────────────────────────────

def _build_distractor_prompt(
    context: str, question: str, correct_answer: str, num_distractors: int = 3
) -> str:
    """Build a prompt to generate plausible distractors."""
    return (
        f"You are an expert at creating multiple-choice quiz distractors.\n\n"
        f"### Task\n"
        f"Given a question and its correct answer, generate exactly "
        f"{num_distractors} WRONG but PLAUSIBLE options.\n\n"
        f"Rules:\n"
        f"- Each distractor must be factually incorrect for the given context.\n"
        f"- Distractors should be similar in length, style, and specificity to "
        f"the correct answer.\n"
        f"- Avoid obviously wrong or absurd options.\n"
        f"- Distractors must not overlap with the correct answer.\n\n"
        f"### Context\n{context}\n\n"
        f"### Question\n{question}\n\n"
        f"### Correct Answer\n{correct_answer}\n\n"
        f"### Output\n"
        f"Respond with ONLY a JSON array of {num_distractors} distractor strings.\n"
        f"Example: [\"distractor 1\", \"distractor 2\", \"distractor 3\"]\n\n"
        f"### Distractors:\n"
    )


# ─── Distractor Generator Class ────────────────────────────────────────────────

class DistractorGenerator:
    """
    Generate plausible distractors for MCQ questions using a hybrid approach:
    LLM generation backed by semantic similarity filtering.
    """

    def __init__(self, generator=None, vector_store=None):
        """
        Parameters
        ----------
        generator : QuestionGenerator, optional
            An already-initialised QuestionGenerator for LLM inference.
        vector_store : VectorStore, optional
            An already-initialised VectorStore for embedding-based fallback.
        """
        self._generator = generator
        self._vector_store = vector_store

    @property
    def generator(self):
        if self._generator is None:
            from inference.generator import QuestionGenerator

            self._generator = QuestionGenerator()
        return self._generator

    @property
    def vector_store(self):
        return self._vector_store

    # ─── LLM-based distractor generation ─────────────────────────────────

    def generate_distractors_llm(
        self,
        context: str,
        question: str,
        correct_answer: str,
        num_distractors: int = 3,
    ) -> List[str]:
        """
        Use the LLM to generate distractor options.

        Returns
        -------
        List[str]
            List of distractor strings.
        """
        prompt = _build_distractor_prompt(
            context=context,
            question=question,
            correct_answer=correct_answer,
            num_distractors=num_distractors,
        )

        raw_output = self.generator.generate_raw(prompt)
        return _parse_distractors(raw_output, num_distractors)

    # ─── Embedding-based distractor retrieval ────────────────────────────

    def generate_distractors_embedding(
        self,
        correct_answer: str,
        num_distractors: int = 3,
    ) -> List[str]:
        """
        Retrieve semantically similar but distinct terms from the vector store
        as potential distractors.
        """
        if self.vector_store is None or self.vector_store.size == 0:
            return []

        results = self.vector_store.query(correct_answer, top_k=num_distractors + 5)
        distractors = []
        for meta, score in results:
            text = meta.get("text", "")
            # Extract a short phrase from the chunk that differs from the answer
            candidate = _extract_alternative_term(text, correct_answer)
            if candidate and candidate.lower() != correct_answer.lower():
                distractors.append(candidate)
            if len(distractors) >= num_distractors:
                break

        return distractors[:num_distractors]

    # ─── Hybrid approach ────────────────────────────────────────────────

    def generate_distractors(
        self,
        context: str,
        question: str,
        correct_answer: str,
        num_distractors: int = 3,
    ) -> List[str]:
        """
        Generate distractors using the best available strategy.

        Priority: LLM → Embedding fallback → Simple heuristic fallback.
        """
        # Strategy 1: LLM generation (highest quality)
        try:
            distractors = self.generate_distractors_llm(
                context=context,
                question=question,
                correct_answer=correct_answer,
                num_distractors=num_distractors,
            )
            if len(distractors) >= num_distractors:
                return distractors[:num_distractors]
        except Exception as exc:
            logger.warning("LLM distractor generation failed: %s", exc)
            distractors = []

        # Strategy 2: Embedding-based retrieval
        try:
            embedding_distractors = self.generate_distractors_embedding(
                correct_answer=correct_answer,
                num_distractors=num_distractors - len(distractors),
            )
            distractors.extend(embedding_distractors)
            if len(distractors) >= num_distractors:
                return distractors[:num_distractors]
        except Exception as exc:
            logger.warning("Embedding distractor generation failed: %s", exc)

        # Strategy 3: Heuristic fallback
        distractors.extend(
            _heuristic_distractors(correct_answer, num_distractors - len(distractors))
        )
        return distractors[:num_distractors]

    def enrich_question(
        self, question_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        If a question is MCQ and has fewer than 4 options, generate
        missing distractors and update the options list.

        Parameters
        ----------
        question_dict : Dict[str, Any]
            A generated question dictionary.

        Returns
        -------
        Dict[str, Any]
            The enriched question dictionary with 4 options.
        """
        if question_dict.get("type") != "MCQ":
            return question_dict

        options = question_dict.get("options", [])
        answer = question_dict.get("answer", "")
        context = question_dict.get("source_context", "")
        question_text = question_dict.get("question", "")

        if len(options) >= 4 and answer:
            return question_dict

        # Generate missing distractors
        num_needed = max(0, 3 - len([o for o in options if o != answer]))
        if num_needed > 0 and answer:
            new_distractors = self.generate_distractors(
                context=context,
                question=question_text,
                correct_answer=answer,
                num_distractors=num_needed,
            )

            existing = set(o.lower().strip() for o in options)
            for d in new_distractors:
                if d.lower().strip() not in existing:
                    options.append(d)
                    existing.add(d.lower().strip())

        # Ensure the correct answer is in the options
        if answer and answer not in options:
            options.append(answer)

        # Shuffle the options
        random.shuffle(options)
        question_dict["options"] = options[:4]

        return question_dict


# ─── Helper functions ───────────────────────────────────────────────────────────

def _parse_distractors(raw_output: str, expected_count: int) -> List[str]:
    """Parse distractor list from LLM output."""
    import json

    # Try direct JSON array parse
    try:
        result = json.loads(raw_output.strip())
        if isinstance(result, list):
            return [str(d).strip() for d in result if d]
    except json.JSONDecodeError:
        pass

    # Try to find a JSON array
    match = re.search(r"\[.*?\]", raw_output, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            return [str(d).strip() for d in result if d]
        except json.JSONDecodeError:
            pass

    # Fallback: split by numbered lines or bullets
    lines = raw_output.strip().split("\n")
    distractors = []
    for line in lines:
        cleaned = re.sub(r"^[\d\-\*\•\.\)]+\s*", "", line.strip())
        cleaned = cleaned.strip('"').strip("'").strip()
        if cleaned and len(cleaned) > 2:
            distractors.append(cleaned)

    return distractors[:expected_count]


def _extract_alternative_term(text: str, correct_answer: str) -> Optional[str]:
    """
    Extract a candidate alternative term from a text chunk.
    Looks for noun phrases or key terms that differ from the correct answer.
    """
    # Simple heuristic: grab sentences and extract subjects/objects
    sentences = text.split(".")
    for sent in sentences:
        sent = sent.strip()
        if (
            sent
            and correct_answer.lower() not in sent.lower()
            and len(sent.split()) >= 3
        ):
            # Take the first meaningful phrase
            words = sent.split()[:6]
            candidate = " ".join(words).strip("., ")
            if len(candidate) > 3:
                return candidate
    return None


def _heuristic_distractors(correct_answer: str, count: int) -> List[str]:
    """
    Last-resort heuristic distractor generation.
    Modifies the correct answer slightly to create plausible alternatives.
    """
    distractors = []
    modifications = [
        f"Not {correct_answer}",
        f"{correct_answer} (approximately)",
        f"All of the above",
        f"None of the above",
        f"Both {correct_answer} and others",
    ]
    for mod in modifications[:count]:
        distractors.append(mod)
    return distractors
