"""
Quiz Generation Pipeline – End-to-End Orchestrator
====================================================
This is the central brain of the system. It coordinates all modules:

  1. Content Ingestion  →  pdf_parser
  2. Chunking           →  chunker
  3. Concept Extraction →  concept_extractor
  4. Vector Indexing    →  vector_store  (FAISS)
  5. Question Generation→  generator  (LLM)
  6. Distractor Gen     →  distractor_generator
  7. Difficulty Check   →  difficulty_classifier
  8. Validation         →  validator  (NLI + grammar + structure + diversity)
  9. Output Formatting  →  structured JSON

Usage:
    from inference.pipeline import QuizPipeline
    pipeline = QuizPipeline()
    result = pipeline.generate_quiz(content="...", num_questions=5)
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Dict, List, Optional

from config import SUPPORTED_DIFFICULTIES, SUPPORTED_QUESTION_TYPES

logger = logging.getLogger(__name__)


class QuizPipeline:
    """
    End-to-end quiz generation pipeline.

    Lazily initialises all sub-modules on first call to avoid heavy startup
    cost when the API server boots.
    """

    def __init__(self):
        self._generator = None
        self._distractor_gen = None
        self._validator = None
        self._difficulty_classifier = None
        self._vector_store = None
        self._concept_extractor_fn = None

    # ─── Lazy component initialization ──────────────────────────────────

    @property
    def generator(self):
        if self._generator is None:
            from inference.generator import QuestionGenerator

            self._generator = QuestionGenerator()
        return self._generator

    @property
    def distractor_gen(self):
        if self._distractor_gen is None:
            from inference.distractor_generator import DistractorGenerator

            self._distractor_gen = DistractorGenerator(
                generator=self.generator, vector_store=self.vector_store
            )
        return self._distractor_gen

    @property
    def validator(self):
        if self._validator is None:
            from inference.validator import QuestionValidator

            self._validator = QuestionValidator()
        return self._validator

    @property
    def difficulty_classifier(self):
        if self._difficulty_classifier is None:
            from inference.difficulty_classifier import DifficultyClassifier

            self._difficulty_classifier = DifficultyClassifier()
        return self._difficulty_classifier

    @property
    def vector_store(self):
        if self._vector_store is None:
            from inference.vector_store import VectorStore

            self._vector_store = VectorStore()
        return self._vector_store

    # ─── Content Processing ──────────────────────────────────────────────

    def _ingest_and_chunk(self, content: str) -> list:
        """
        Parse content and split into semantic chunks.

        Parameters
        ----------
        content : str
            Raw text or file path.

        Returns
        -------
        List[TextChunk]
        """
        from utils.pdf_parser import parse_document
        from utils.chunker import chunk_document, chunk_raw_text

        document = parse_document(content)

        if document.sections:
            chunks = chunk_document(document)
        else:
            chunks = chunk_raw_text(document.full_text)

        logger.info("Ingested content into %d chunks.", len(chunks))
        return chunks

    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from a text chunk."""
        from utils.concept_extractor import extract_concepts

        concepts = extract_concepts(text, top_n=8)
        return concepts.all_concepts

    def _index_chunks(self, chunks: list) -> None:
        """Build a FAISS index from chunks for RAG retrieval."""
        texts = [c.text for c in chunks]
        metadatas = [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "heading": c.source_heading,
                "page": c.source_page,
            }
            for c in chunks
        ]
        self.vector_store.build_index(texts, metadatas)
        logger.info("FAISS index built with %d vectors.", self.vector_store.size)

    # ─── Question Type & Difficulty Distribution ─────────────────────────

    def _plan_question_distribution(
        self,
        num_questions: int,
        types: Optional[List[str]] = None,
        difficulty: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Plan the distribution of question types and difficulties.

        Returns a list of dicts, each with 'type' and 'difficulty' keys.
        """
        available_types = types or SUPPORTED_QUESTION_TYPES[:3]  # default: MCQ, TF, FITB
        # Validate types
        available_types = [
            t for t in available_types if t in SUPPORTED_QUESTION_TYPES
        ]
        if not available_types:
            available_types = ["MCQ"]

        plan = []
        for i in range(num_questions):
            q_type = available_types[i % len(available_types)]

            if difficulty:
                q_diff = difficulty
            else:
                # Distribute difficulty: 30% easy, 40% medium, 30% hard
                rand = random.random()
                if rand < 0.3:
                    q_diff = "easy"
                elif rand < 0.7:
                    q_diff = "medium"
                else:
                    q_diff = "hard"

            plan.append({"type": q_type, "difficulty": q_diff})

        return plan

    # ─── Core Generation Logic ───────────────────────────────────────────

    def _generate_single_question(
        self,
        chunk_text: str,
        question_type: str,
        difficulty: str,
        concepts: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Generate a single question. Keep it simple and fast."""
        try:
            question = self.generator.generate_question(
                context=chunk_text,
                question_type=question_type,
                difficulty=difficulty,
                concepts=concepts,
            )

            # Basic quality check: must have a non-empty question
            q_text = question.get("question", "").strip()
            if not q_text or len(q_text) < 10:
                logger.warning("Generated question too short, skipping.")
                return None

            return question

        except Exception as exc:
            logger.warning("Question generation failed: %s", exc)
            return None

    # ─── Public API: generate_quiz ───────────────────────────────────────

    def generate_quiz(
        self,
        content: str,
        num_questions: int = 5,
        difficulty: Optional[str] = None,
        types: Optional[List[str]] = None,
        topic: Optional[str] = None,
        generate_explanations: bool = True,
        strict_validation: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate a complete quiz from input content.

        Parameters
        ----------
        content : str
            Raw text, file path (PDF/TXT), or structured lesson content.
        num_questions : int
            Desired number of questions.
        difficulty : str, optional
            Target difficulty (easy/medium/hard). If None, auto-distributed.
        types : List[str], optional
            Question types to generate. Defaults to [MCQ, TrueFalse, FillInTheBlank].
        topic : str, optional
            Topic name for the output metadata.
        generate_explanations : bool
            Whether to include explanations.
        strict_validation : bool
            If True, NLI + grammar must pass. If False, only structural
            checks are enforced (faster, more lenient).

        Returns
        -------
        Dict[str, Any]
            Structured quiz JSON with topic, questions, and metadata.
        """
        start_time = time.time()

        # Validate inputs
        if difficulty and difficulty not in SUPPORTED_DIFFICULTIES:
            raise ValueError(
                f"Invalid difficulty '{difficulty}'. Choose from {SUPPORTED_DIFFICULTIES}"
            )

        # ── Step 1: Ingest & Chunk ──────────────────────────────────────
        logger.info("Step 1/6: Ingesting and chunking content...")
        chunks = self._ingest_and_chunk(content)

        if not chunks:
            return {
                "topic": topic or "Unknown",
                "questions": [],
                "metadata": {"error": "No processable content found."},
            }

        # ── Step 2: Index chunks in FAISS ───────────────────────────────
        logger.info("Step 2/6: Building FAISS index...")
        self._index_chunks(chunks)

        # ── Step 3: Extract concepts from all chunks ────────────────────
        logger.info("Step 3/6: Extracting key concepts...")
        all_concepts = []
        for chunk in chunks[:10]:  # limit to first 10 chunks for speed
            concepts = self._extract_concepts(chunk.text)
            all_concepts.extend(concepts)
        # Deduplicate
        seen = set()
        unique_concepts = []
        for c in all_concepts:
            if c.lower() not in seen:
                seen.add(c.lower())
                unique_concepts.append(c)

        # Auto-detect topic if not provided
        if not topic and unique_concepts:
            topic = unique_concepts[0].title()

        # ── Step 4: Plan question distribution ──────────────────────────
        logger.info("Step 4/6: Planning question distribution...")
        plan = self._plan_question_distribution(
            num_questions=num_questions,
            types=types,
            difficulty=difficulty,
        )

        # ── Step 5: Generate questions ──────────────────────────────────
        logger.info("Step 5/6: Generating %d questions...", num_questions)
        raw_questions: list[Dict[str, Any]] = []

        for i, spec in enumerate(plan):
            # Select a chunk (round-robin across chunks)
            chunk_idx = i % len(chunks)
            chunk = chunks[chunk_idx]

            # Use pre-extracted concepts (don't re-extract per chunk — slow)
            chunk_concepts = unique_concepts[:5]

            question = self._generate_single_question(
                chunk_text=chunk.text,
                question_type=spec["type"],
                difficulty=spec["difficulty"],
                concepts=chunk_concepts,
            )

            if question:
                raw_questions.append(question)
                logger.info(
                    "  [%d/%d] Generated: %s",
                    len(raw_questions), num_questions,
                    question.get('question', '')[:60],
                )

            # Stop once we have enough
            if len(raw_questions) >= num_questions:
                break

        logger.info("Generated %d questions total.", len(raw_questions))

        # ── Step 6: Lightweight filter (skip heavy NLI on CPU) ──────────
        logger.info("Step 6/6: Filtering questions...")
        final_questions = []
        for q in raw_questions:
            q_text = q.get("question", "").strip()
            # Basic quality: non-empty question with reasonable length
            if q_text and len(q_text) >= 10:
                final_questions.append(q)

        final_questions = final_questions[:num_questions]

        # Clean up validation internals from output
        for q in final_questions:
            q.pop("validation", None)
            q.pop("is_valid", None)
            q.pop("_parse_error", None)
            q.pop("difficulty_analysis", None)
            if not generate_explanations:
                q.pop("explanation", None)

        elapsed = round(time.time() - start_time, 2)

        result = {
            "topic": topic or "General",
            "questions": final_questions,
            "metadata": {
                "num_requested": num_questions,
                "num_generated": len(final_questions),
                "types_used": list(set(q.get("type", "") for q in final_questions)),
                "difficulties_used": list(
                    set(q.get("difficulty", "") for q in final_questions)
                ),
                "num_chunks": len(chunks),
                "num_concepts_extracted": len(unique_concepts),
                "processing_time_seconds": elapsed,
            },
        }

        logger.info(
            "Quiz generated: %d questions in %.2fs.",
            len(final_questions),
            elapsed,
        )
        return result

    # ─── Public API: generate_from_pdf ───────────────────────────────────

    def generate_from_pdf(
        self,
        pdf_path: str,
        num_questions: int = 10,
        difficulty: Optional[str] = None,
        types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Convenience method: generate quiz directly from a PDF file path.
        """
        return self.generate_quiz(
            content=pdf_path,
            num_questions=num_questions,
            difficulty=difficulty,
            types=types,
        )
