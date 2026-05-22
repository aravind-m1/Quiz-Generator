# QuizForge AI — Intelligent Quiz Question Generation System

An AI-powered system that automatically generates high-quality quiz questions from educational content. Built with a hybrid RAG + Transformer pipeline, it takes textbook chapters, lecture notes, or PDF documents as input and produces structured, pedagogically sound quiz questions with configurable difficulty and question types.

> **Note:** This system handles **question generation only** — it is not a quiz-taking platform.

---

## Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Training Pipeline](#training-pipeline)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Documentation](#documentation)
- [License](#license)

---

## Features

- **5 Question Types** — MCQ, True/False, Fill-in-the-Blank, Short Answer, and Assertion-Reason
- **Bloom's Taxonomy Difficulty Control** — Easy (Remember), Medium (Apply), Hard (Evaluate)
- **PDF & Text Input** — Upload PDF documents or paste raw text directly
- **RAG-Based Generation** — Questions grounded in source content via FAISS vector retrieval
- **Multi-Pass Generation** — 4-pass strategy (Question → Answer → Distractors → Explanation) for complete, accurate output
- **NLI Validation** — DeBERTa-v3 entailment checking to prevent hallucinated questions
- **Grammar & Diversity Filters** — LanguageTool grammar checking + cosine similarity deduplication
- **Web Interface** — Modern, responsive dark-mode frontend with real-time generation feedback
- **REST API** — FastAPI backend with auto-generated Swagger documentation
- **Fine-Tuning Support** — Full fine-tuning, LoRA, and QLoRA on domain-specific datasets (SQuAD, SciQ, RACE)
- **CPU/GPU Auto-Detection** — Runs seamlessly on both CPU and CUDA-enabled GPUs

---

## System Architecture

The system follows a **6-stage pipeline** architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                              │
│  PDF Parser (PyMuPDF/pdfplumber)  ←→  Raw Text Parser           │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                     PROCESSING LAYER                            │
│  Semantic Chunker (400w, 80w overlap)                           │
│  Concept Extractor (KeyBERT + SpaCy NER + TF-IDF)               │
│  FAISS Vector Index (BGE-small 384-dim embeddings)              │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    GENERATION LAYER                             │
│  FLAN-T5 Encoder-Decoder LLM                                   │
│  Multi-Pass Strategy:                                           │
│    Pass 1: Generate Question                                    │
│    Pass 2: Generate Answer                                      │
│    Pass 3: Generate Distractors (MCQ)                           │
│    Pass 4: Generate Explanation                                 │
│  Bloom's Taxonomy Prompt Engineering                            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    VALIDATION LAYER                              │
│  Structural Checks → NLI Entailment (DeBERTa-v3)               │
│  → Grammar (LanguageTool) → Diversity Filter (Cosine Sim)       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                      OUTPUT LAYER                               │
│  FastAPI REST API  →  Structured JSON  →  Web Frontend          │
└─────────────────────────────────────────────────────────────────┘
```

### Models Used

| Model | Parameters | Purpose |
|-------|-----------|---------|
| `google/flan-t5-base` | 248M | Question, answer, distractor, and explanation generation |
| `BAAI/bge-small-en-v1.5` | 33.4M | Sentence embeddings for FAISS vector indexing |
| `cross-encoder/nli-deberta-v3-small` | 22M | NLI entailment validation |
| `KeyBERT` | — | Keyphrase extraction from text chunks |
| `SpaCy en_core_web_sm` | 12M | Named entity recognition |

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.11+ |
| **Deep Learning** | PyTorch, Hugging Face Transformers, Sentence-Transformers |
| **NLP** | SpaCy, KeyBERT, LanguageTool |
| **Vector Search** | FAISS (faiss-cpu) |
| **API** | FastAPI, Uvicorn, Pydantic v2 |
| **Frontend** | HTML5, CSS3 (custom dark theme), Vanilla JavaScript |
| **Fine-Tuning** | PEFT (LoRA/QLoRA), BitsAndBytes |
| **PDF Parsing** | PyMuPDF (fitz), pdfplumber |
| **Deployment** | Docker, Render/Railway/HuggingFace Spaces |

---

## Project Structure

```
Quiz-Generator/
├── api/                          # FastAPI application layer
│   ├── __init__.py
│   ├── main.py                   # API routes, CORS, static serving
│   └── schemas.py                # Pydantic request/response models
├── inference/                    # Core AI inference modules
│   ├── __init__.py
│   ├── generator.py              # Multi-pass FLAN-T5 question generator
│   ├── pipeline.py               # End-to-end orchestration pipeline
│   ├── vector_store.py           # FAISS index management
│   ├── validator.py              # NLI + grammar + diversity validation
│   ├── distractor_generator.py   # MCQ distractor generation
│   └── difficulty_classifier.py  # Bloom's Taxonomy classification
├── utils/                        # Support utilities
│   ├── __init__.py
│   ├── pdf_parser.py             # PDF/text document parsing
│   ├── chunker.py                # Semantic text chunking
│   └── concept_extractor.py      # KeyBERT + SpaCy concept extraction
├── training/                     # Model fine-tuning pipeline
│   ├── __init__.py
│   ├── train.py                  # Training loop (Full/LoRA/QLoRA)
│   └── prepare_dataset.py        # SQuAD/SciQ/RACE dataset preparation
├── static/                       # Frontend web interface
│   ├── index.html                # Main HTML page
│   ├── style.css                 # Premium dark-mode stylesheet
│   └── app.js                    # Frontend application logic
├── models/                       # Model checkpoints & FAISS indices
├── data/
│   ├── raw/                      # Input documents
│   └── processed/                # Processed datasets
├── config.py                     # Centralized configuration
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Production container
├── QuizForge_AI_Documentation.pdf  # Full project documentation
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager
- 4 GB+ RAM (8 GB recommended for large PDFs)

### Setup

```bash
# Clone the repository
git clone https://github.com/aravind-m1/Quiz-Generator.git
cd Quiz-Generator

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download SpaCy language model
python -m spacy download en_core_web_sm
```

### Quick Verify

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

## Usage

### Start the Server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Then open your browser at **http://localhost:8000**

### Web Interface

1. **Paste content** or **upload a PDF** in the input area
2. Set the **number of questions** (1–20)
3. Choose a **difficulty level** (Easy / Medium / Hard / Auto)
4. Select **question types** (MCQ, True/False, Fill-in-Blank, etc.)
5. Click **Generate Quiz** and wait for the results

### Command Line (cURL)

```bash
# Generate from text
curl -X POST http://localhost:8000/api/v1/generate-quiz \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Machine learning is a subset of AI that enables systems to learn from data...",
    "num_questions": 3,
    "difficulty": "medium",
    "types": ["MCQ", "TrueFalse"]
  }'

# Upload a PDF
curl -X POST http://localhost:8000/api/v1/upload-pdf \
  -F "file=@textbook.pdf" \
  -F "num_questions=5"
```

### Sample Output

```json
{
  "topic": "Machine Learning",
  "questions": [
    {
      "type": "MCQ",
      "difficulty": "easy",
      "question": "What is the primary goal of machine learning?",
      "options": [
        "To manually program every decision",
        "To allow computers to learn from data automatically",
        "To replace all human workers",
        "To store large amounts of data"
      ],
      "answer": "To allow computers to learn from data automatically",
      "explanation": "Machine learning focuses on developing programs that access data and use it to learn for themselves, without explicit programming."
    }
  ],
  "metadata": {
    "num_requested": 3,
    "num_generated": 3,
    "num_chunks": 27,
    "num_concepts_extracted": 81,
    "processing_time_seconds": 86.28
  }
}
```

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/api/v1/generate-quiz` | POST | Generate quiz from text content |
| `/api/v1/upload-pdf` | POST | Generate quiz from uploaded PDF |
| `/api/v1/health` | GET | Server health check |
| `/docs` | GET | Interactive Swagger documentation |

### POST `/api/v1/generate-quiz`

**Request Body:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `content` | string | *required* | Input text (min 50 chars) |
| `num_questions` | int | 5 | Number of questions (1–20) |
| `difficulty` | string | null | `easy`, `medium`, `hard`, or null for auto |
| `types` | list | `["MCQ"]` | Question types to generate |
| `topic` | string | null | Optional topic label |
| `generate_explanations` | bool | true | Include explanations |

---

## Training Pipeline

Fine-tune the generator model for domain-specific question quality:

```bash
# Full fine-tuning on SQuAD + SciQ
python training/train.py --strategy full --epochs 3 --batch_size 8

# LoRA fine-tuning (97% fewer trainable parameters)
python training/train.py --strategy lora --lora_r 16 --lora_alpha 32

# QLoRA (4-bit quantized + LoRA — for large models on consumer GPUs)
python training/train.py --strategy qlora --model_name google/flan-t5-large
```

**Supported Datasets:** SQuAD 2.0, SciQ, RACE

---

## Configuration

All settings are centralized in `config.py` and can be overridden with environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `QG_GENERATOR_MODEL` | `google/flan-t5-base` | Generator model name/path |
| `QG_DEVICE` | Auto-detected | `cuda` or `cpu` |
| `QG_CHUNK_SIZE` | 400 | Words per text chunk |
| `QG_CHUNK_OVERLAP` | 80 | Overlap between chunks |
| `QG_GENERATION_MAX_LENGTH` | 256 | Max tokens per generation |
| `QG_NLI_THRESHOLD` | 0.65 | NLI entailment score threshold |
| `QG_DIVERSITY_THRESHOLD` | 0.80 | Max cosine similarity between questions |

---

## Deployment

### Docker

```bash
docker build -t quizforge-ai .
docker run -p 8000:8000 quizforge-ai
```

### Cloud Platforms

The application is ready for deployment on:
- **Render** — Connect repo, set build command to `pip install -r requirements.txt`, start command to `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
- **Railway** — Auto-detects Python, deploy directly from GitHub
- **HuggingFace Spaces** — Use the Dockerfile for GPU-enabled deployment

---

## Documentation

A comprehensive 20-page project documentation PDF is included in the repository:

**[QuizForge_AI_Documentation.pdf](QuizForge_AI_Documentation.pdf)**

It covers:
- Abstract & Introduction
- Objectives
- Model Architecture (with diagrams)
- Methodology
- Results & Performance Metrics
- Tools & Technologies
- Advantages & Limitations
- Future Work
- References (14 academic citations)

---

## Performance

| Metric | Value |
|--------|-------|
| Startup time | < 2 seconds (lazy model loading) |
| First request latency | ~30s (model download + load) |
| Generation time (CPU) | 40–130 seconds per quiz |
| Generation time (GPU) | 5–15 seconds per quiz |
| Max PDF size tested | 200 pages (2,954 chunks) |
| Supported question types | 5 |
| Difficulty levels | 3 (Easy / Medium / Hard) |

---

## License

This project is developed for educational and research purposes.

---

<p align="center">
  Built with PyTorch, Hugging Face Transformers, and FastAPI
</p>
