"""
LLM Question Generator
=======================
Optimised for FLAN-T5 models using a multi-pass approach:
  Pass 1: Generate question text
  Pass 2: Generate answer for that question
  Pass 3: Generate MCQ options (if applicable)

This avoids the problem of T5-base producing incomplete single-pass outputs.
"""

from __future__ import annotations

import json
import logging
import random
import re
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


# ─── Generator Class ────────────────────────────────────────────────────────────

class QuestionGenerator:
    """
    LLM-based question generator using FLAN-T5 Seq2Seq models.
    Uses a multi-pass strategy for reliable output quality.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        from config import GENERATOR_MODEL_NAME, DEVICE

        self.model_name = model_name or GENERATOR_MODEL_NAME
        self.device = device or DEVICE
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Load model and tokenizer on first use."""
        if self._model is not None:
            return

        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        logger.info("Loading generator model: %s on %s", self.model_name, self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
        )
        self._model.to(self.device)
        self._model.eval()
        logger.info("Generator model loaded on %s.", self.device)

    @property
    def tokenizer(self):
        self._load_model()
        return self._tokenizer

    @property
    def model(self):
        self._load_model()
        return self._model

    # ─── Core inference ──────────────────────────────────────────────────

    def _generate(self, prompt: str, max_tokens: int = 128) -> str:
        """Run a single generation pass."""
        inputs = self.tokenizer(
            prompt, return_tensors="pt", max_length=512, truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.8,
                top_p=0.92,
                do_sample=True,
                no_repeat_ngram_size=3,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # ─── Multi-pass question generation ──────────────────────────────────

    def generate_question(
        self,
        context: str,
        question_type: str = "MCQ",
        difficulty: str = "medium",
        concepts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a structured quiz question using multi-pass T5 inference.

        Returns dict with: question, options, answer, explanation, type, difficulty.
        """
        ctx = context[:1000]  # Trim to keep within token limits
        concept_hint = ""
        if concepts and len(concepts) > 0:
            concept_hint = f" about {concepts[0]}"

        # ── Pass 1: Generate the question ────────────────────────────────
        if question_type == "TrueFalse":
            q_prompt = (
                f"Based on the following text, write a {difficulty} "
                f"true or false statement{concept_hint}:\n\n{ctx}"
            )
        elif question_type == "FillInTheBlank":
            q_prompt = (
                f"Based on the following text, write a {difficulty} "
                f"fill-in-the-blank sentence{concept_hint}. "
                f"Replace one key word with a blank ________:\n\n{ctx}"
            )
        else:
            q_prompt = (
                f"Based on the following text, ask a {difficulty}{concept_hint} "
                f"question:\n\n{ctx}"
            )

        question_text = self._generate(q_prompt, max_tokens=100)
        logger.info("Pass 1 (question): %s", question_text[:120])

        # ── Pass 2: Generate the answer ──────────────────────────────────
        if question_type == "TrueFalse":
            a_prompt = f"Is the following statement true or false?\n\n{question_text}\n\nContext: {ctx[:500]}\n\nAnswer:"
            answer = self._generate(a_prompt, max_tokens=20)
            # Normalise True/False
            answer = answer.strip().rstrip(".")
            if "true" in answer.lower():
                answer = "True"
            elif "false" in answer.lower():
                answer = "False"
            else:
                answer = random.choice(["True", "False"])
        elif question_type == "FillInTheBlank":
            # Extract the blanked word from context
            a_prompt = f"Fill in the blank:\n{question_text}\n\nContext: {ctx[:500]}\n\nAnswer:"
            answer = self._generate(a_prompt, max_tokens=30)
        else:
            a_prompt = f"Answer the following question in one sentence.\n\nQuestion: {question_text}\n\nContext: {ctx[:500]}\n\nAnswer:"
            answer = self._generate(a_prompt, max_tokens=60)

        logger.info("Pass 2 (answer): %s", answer[:100])

        # ── Pass 3: Generate MCQ options (if applicable) ─────────────────
        options = []
        if question_type == "MCQ":
            options = self._generate_mcq_options(question_text, answer, ctx)

        # ── Pass 4: Generate explanation ──────────────────────────────────
        e_prompt = f"Explain why the answer to \"{question_text}\" is \"{answer}\" based on this text:\n\n{ctx[:400]}"
        explanation = self._generate(e_prompt, max_tokens=80)
        if not explanation or len(explanation) < 5:
            explanation = "Based on the provided context."

        logger.info("Pass 4 (explanation): %s", explanation[:100])

        return {
            "question": question_text,
            "options": options,
            "answer": answer,
            "explanation": explanation,
            "type": question_type,
            "difficulty": difficulty,
            "source_context": context[:500],
        }

    def _generate_mcq_options(
        self, question: str, correct_answer: str, context: str
    ) -> List[str]:
        """Generate 4 MCQ options with the correct answer included."""
        # Generate 3 wrong options
        d_prompt = (
            f"Question: {question}\n"
            f"Correct answer: {correct_answer}\n\n"
            f"Generate 3 wrong but plausible answers, separated by newlines:"
        )
        raw_distractors = self._generate(d_prompt, max_tokens=100)
        logger.info("Pass 3 (distractors): %s", raw_distractors[:120])

        # Parse distractors from output
        distractors = []
        for line in raw_distractors.split("\n"):
            line = line.strip()
            # Remove numbering/lettering prefixes
            line = re.sub(r"^[\d\.\)\-\*]+\s*", "", line)
            line = re.sub(r"^[A-Da-d][\.\)\:]\s*", "", line)
            line = line.strip()
            if line and line.lower() != correct_answer.lower() and len(line) > 2:
                distractors.append(line)

        # If we didn't get enough, create simple fallbacks
        fallbacks = ["None of the above", "All of the above", "Cannot be determined"]
        while len(distractors) < 3:
            fb = fallbacks.pop(0) if fallbacks else f"Option {len(distractors) + 2}"
            if fb not in distractors:
                distractors.append(fb)

        # Combine correct + distractors and shuffle
        options = [correct_answer] + distractors[:3]
        random.shuffle(options)
        return options

    # ─── Batch generation ────────────────────────────────────────────────

    def generate_batch(
        self,
        contexts: List[str],
        question_types: Optional[List[str]] = None,
        difficulties: Optional[List[str]] = None,
        concepts_list: Optional[List[List[str]]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate questions for multiple contexts."""
        n = len(contexts)
        types = question_types or ["MCQ"] * n
        diffs = difficulties or ["medium"] * n
        concepts = concepts_list or [None] * n

        results = []
        for i in range(n):
            try:
                q = self.generate_question(
                    context=contexts[i],
                    question_type=types[i],
                    difficulty=diffs[i],
                    concepts=concepts[i],
                )
                results.append(q)
            except Exception as exc:
                logger.error("Failed to generate question %d: %s", i, exc)
                results.append(
                    {
                        "question": "",
                        "options": [],
                        "answer": "",
                        "explanation": "",
                        "type": types[i],
                        "difficulty": diffs[i],
                        "error": str(exc),
                    }
                )
        return results
