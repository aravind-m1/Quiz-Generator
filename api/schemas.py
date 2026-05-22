"""
Pydantic Schemas for the Quiz Generation API
==============================================
Defines request/response models with comprehensive validation.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


# ─── Enums ──────────────────────────────────────────────────────────────────────

class QuestionType(str, Enum):
    MCQ = "MCQ"
    TRUE_FALSE = "TrueFalse"
    FILL_IN_THE_BLANK = "FillInTheBlank"
    SHORT_ANSWER = "ShortAnswer"
    ASSERTION_REASON = "AssertionReason"


class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ─── Request Schemas ────────────────────────────────────────────────────────────

class QuizGenerationRequest(BaseModel):
    """Request body for quiz generation from raw text."""

    content: str = Field(
        ...,
        min_length=50,
        description="Raw text content (lesson, chapter, notes) to generate questions from. "
        "Minimum 50 characters.",
        examples=[
            "Photosynthesis is a process used by plants to convert light energy "
            "into chemical energy that can be later released to fuel the plant's "
            "activities."
        ],
    )
    num_questions: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of questions to generate (1-50).",
    )
    difficulty: Optional[DifficultyLevel] = Field(
        default=None,
        description="Target difficulty level. If not specified, difficulty is auto-distributed.",
    )
    types: Optional[List[QuestionType]] = Field(
        default=None,
        description="Question types to generate. If not specified, defaults to MCQ, TrueFalse, FillInTheBlank.",
    )
    topic: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Topic name for the quiz metadata.",
    )
    generate_explanations: bool = Field(
        default=True,
        description="Whether to include explanations for each question.",
    )
    strict_validation: bool = Field(
        default=False,
        description="If True, applies strict NLI + grammar validation (slower but higher quality).",
    )

    @validator("types", pre=True)
    def validate_types(cls, v):
        if v is not None and len(v) == 0:
            return None
        return v


class PDFUploadParams(BaseModel):
    """Metadata parameters for PDF upload endpoint."""

    num_questions: int = Field(default=10, ge=1, le=50)
    difficulty: Optional[DifficultyLevel] = None
    types: Optional[List[QuestionType]] = None
    topic: Optional[str] = None
    generate_explanations: bool = True


# ─── Response Schemas ───────────────────────────────────────────────────────────

class GeneratedQuestion(BaseModel):
    """A single generated quiz question."""

    type: str = Field(description="Question type: MCQ, TrueFalse, FillInTheBlank, ShortAnswer, AssertionReason")
    difficulty: str = Field(description="Difficulty level: easy, medium, hard")
    question: str = Field(description="The question text")
    options: List[str] = Field(
        default_factory=list,
        description="Answer options (for MCQ and AssertionReason). Empty list for other types.",
    )
    answer: str = Field(description="The correct answer")
    explanation: Optional[str] = Field(
        default=None,
        description="Explanation of why the answer is correct, referencing the source context.",
    )
    source_context: Optional[str] = Field(
        default=None,
        description="The source text chunk this question was generated from.",
    )


class QuizMetadata(BaseModel):
    """Metadata about the quiz generation process."""

    num_requested: int
    num_generated: int
    types_used: List[str]
    difficulties_used: List[str]
    num_chunks: int
    num_concepts_extracted: int
    processing_time_seconds: float


class QuizGenerationResponse(BaseModel):
    """Complete quiz generation response."""

    topic: str = Field(description="Quiz topic")
    questions: List[GeneratedQuestion] = Field(description="Generated questions")
    metadata: QuizMetadata = Field(description="Generation process metadata")


# ─── Health Check ───────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str = "1.0.0"
    models_loaded: Dict[str, bool] = Field(default_factory=dict)


# ─── Error Response ─────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Standardised error response."""

    error: str
    detail: Optional[str] = None
    status_code: int = 500
