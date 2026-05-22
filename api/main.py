"""
FastAPI Application – Quiz Question Generation API
====================================================
Production-ready API server with:
  - POST /api/v1/generate-quiz   (text input)
  - POST /api/v1/upload-pdf      (PDF file upload)
  - GET  /api/v1/health          (health check)
  - GET  /                       (API documentation redirect)

Features:
  - CORS middleware for frontend integration
  - Structured error handling
  - Request/response validation via Pydantic
  - Lazy model loading (models load on first request, not at startup)
  - Async-compatible endpoints for production use
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from api.schemas import (
    DifficultyLevel,
    ErrorResponse,
    HealthResponse,
    QuestionType,
    QuizGenerationRequest,
    QuizGenerationResponse,
)

# ─── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("quiz_api")

# ─── Singleton Pipeline ────────────────────────────────────────────────────────
_pipeline = None


def get_pipeline():
    """Lazy-initialise the QuizPipeline singleton."""
    global _pipeline
    if _pipeline is None:
        from inference.pipeline import QuizPipeline

        logger.info("Initialising QuizPipeline...")
        _pipeline = QuizPipeline()
    return _pipeline


# ─── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup / shutdown hooks."""
    logger.info("🚀 Quiz Generation API starting up...")
    yield
    logger.info("🛑 Quiz Generation API shutting down.")


# ─── App Factory ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Quiz Question Generation API",
    description=(
        "State-of-the-art AI-powered quiz question generation system.\n\n"
        "Accepts course content (text or PDF) and generates high-quality quiz "
        "questions with controlled difficulty, diverse question types, and "
        "source-backed explanations."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    # Swagger at /docs, website at /

)

# CORS
from config import API_CORS_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=API_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Static Files ──────────────────────────────────────────────────────────────
_static_dir = Path(__file__).resolve().parent.parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")
    logger.info("Serving static files from %s", _static_dir)


# ─── Exception Handlers ────────────────────────────────────────────────────────

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail, status_code=exc.status_code
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            status_code=500,
        ).model_dump(),
    )


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    """Serve the frontend website."""
    html_path = _static_dir / "index.html"
    if html_path.exists():
        return FileResponse(str(html_path))
    return RedirectResponse(url="/docs")


@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health Check",
)
async def health_check():
    """
    Check API health and model loading status.
    """
    pipeline = get_pipeline()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded={
            "generator": pipeline._generator is not None,
            "vector_store": pipeline._vector_store is not None,
            "validator": pipeline._validator is not None,
        },
    )


@app.post(
    "/api/v1/generate-quiz",
    response_model=QuizGenerationResponse,
    tags=["Quiz Generation"],
    summary="Generate Quiz from Text",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Generation failed"},
    },
)
async def generate_quiz(request: QuizGenerationRequest):
    """
    Generate quiz questions from raw text content.

    Accepts lesson/chapter text and produces structured quiz questions
    with controlled difficulty, diverse question types, and explanations.

    ### Example
    ```json
    {
      "content": "Natural language processing (NLP) is a subfield of linguistics...",
      "num_questions": 5,
      "difficulty": "medium",
      "types": ["MCQ", "TrueFalse"],
      "generate_explanations": true
    }
    ```
    """
    pipeline = get_pipeline()

    try:
        result = pipeline.generate_quiz(
            content=request.content,
            num_questions=request.num_questions,
            difficulty=request.difficulty.value if request.difficulty else None,
            types=[t.value for t in request.types] if request.types else None,
            topic=request.topic,
            generate_explanations=request.generate_explanations,
            strict_validation=request.strict_validation,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Quiz generation failed: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Quiz generation failed: {str(exc)}"
        )

    return result


@app.post(
    "/api/v1/upload-pdf",
    response_model=QuizGenerationResponse,
    tags=["Quiz Generation"],
    summary="Generate Quiz from PDF Upload",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file"},
        500: {"model": ErrorResponse, "description": "Generation failed"},
    },
)
async def upload_pdf(
    file: UploadFile = File(..., description="PDF file to generate questions from"),
    num_questions: int = Form(default=10, ge=1, le=50),
    difficulty: Optional[str] = Form(
        default=None, description="Difficulty level: easy, medium, hard"
    ),
    types: Optional[str] = Form(
        default=None,
        description="Comma-separated question types: MCQ,TrueFalse,FillInTheBlank,ShortAnswer,AssertionReason",
    ),
    topic: Optional[str] = Form(default=None),
    generate_explanations: bool = Form(default=True),
):
    """
    Upload a PDF file and generate quiz questions from its content.

    ### Supported formats
    - PDF files (.pdf)
    - Text files (.txt, .md)
    """
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    allowed_extensions = {".pdf", ".txt", ".md"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {allowed_extensions}",
        )

    # Save uploaded file temporarily
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=ext, dir=str(Path(__file__).parent.parent / "data" / "raw")
        ) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to save uploaded file: {exc}"
        )

    # Parse question types
    type_list = None
    if types:
        type_list = [t.strip() for t in types.split(",") if t.strip()]

    # Validate difficulty
    diff = None
    if difficulty and difficulty in ("easy", "medium", "hard"):
        diff = difficulty

    # Generate quiz
    pipeline = get_pipeline()
    try:
        result = pipeline.generate_quiz(
            content=tmp_path,
            num_questions=num_questions,
            difficulty=diff,
            types=type_list,
            topic=topic,
            generate_explanations=generate_explanations,
        )
    except Exception as exc:
        logger.exception("PDF quiz generation failed: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Quiz generation failed: {str(exc)}"
        )
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return result


@app.get(
    "/api/v1/supported-types",
    tags=["System"],
    summary="List Supported Question Types",
)
async def supported_types():
    """List all supported question types and difficulty levels."""
    from config import SUPPORTED_QUESTION_TYPES, SUPPORTED_DIFFICULTIES

    return {
        "question_types": SUPPORTED_QUESTION_TYPES,
        "difficulty_levels": SUPPORTED_DIFFICULTIES,
    }


# ─── Run directly ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    from config import API_HOST, API_PORT

    uvicorn.run(
        "api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info",
    )
