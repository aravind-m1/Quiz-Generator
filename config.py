"""
Global configuration for the Quiz Question Generation system.
All model names, paths, and hyperparameters are centralized here.
"""

import os
from pathlib import Path

import torch

# ─── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
FAISS_INDEX_DIR = BASE_DIR / "models" / "faiss_index"

# Ensure directories exist
for _dir in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, FAISS_INDEX_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ─── Model Configuration ────────────────────────────────────────────────────────

# Generator – Seq2Seq model for question generation
GENERATOR_MODEL_NAME = os.getenv(
    "QG_GENERATOR_MODEL", "google/flan-t5-base"
)

# Embedding model for FAISS vector store (retrieval)
EMBEDDING_MODEL_NAME = os.getenv(
    "QG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
)

# NLI cross-encoder for hallucination validation
NLI_MODEL_NAME = os.getenv(
    "QG_NLI_MODEL", "cross-encoder/nli-deberta-v3-small"
)

# Device selection – auto-detect CUDA availability
_default_device = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = os.getenv("QG_DEVICE", _default_device)

# ─── Chunking Configuration ─────────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("QG_CHUNK_SIZE", "400"))  # words per chunk
CHUNK_OVERLAP = int(os.getenv("QG_CHUNK_OVERLAP", "80"))  # overlapping words

# ─── Generation Hyperparameters ──────────────────────────────────────────────────
GENERATION_MAX_LENGTH = 512
GENERATION_TEMPERATURE = 0.7
GENERATION_TOP_P = 0.92
GENERATION_NUM_BEAMS = 1  # set >1 for beam search (deterministic)

# ─── Training Hyperparameters ────────────────────────────────────────────────────
TRAINING_EPOCHS = 3
TRAINING_BATCH_SIZE = 8
TRAINING_LEARNING_RATE = 2e-5
TRAINING_WEIGHT_DECAY = 0.01
TRAINING_MAX_INPUT_LENGTH = 512
TRAINING_MAX_TARGET_LENGTH = 256

# ─── Validation Thresholds ──────────────────────────────────────────────────────
NLI_ENTAILMENT_THRESHOLD = 0.65  # minimum entailment score to keep a question
DIVERSITY_SIMILARITY_THRESHOLD = 0.80  # max cosine sim between any two questions

# ─── Supported Question Types ───────────────────────────────────────────────────
SUPPORTED_QUESTION_TYPES = [
    "MCQ",
    "TrueFalse",
    "FillInTheBlank",
    "ShortAnswer",
    "AssertionReason",
]

SUPPORTED_DIFFICULTIES = ["easy", "medium", "hard"]

# ─── API Configuration ──────────────────────────────────────────────────────────
API_HOST = os.getenv("QG_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("QG_API_PORT", "8000"))
API_CORS_ORIGINS = ["*"]
