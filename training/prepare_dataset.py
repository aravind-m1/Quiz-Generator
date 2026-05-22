"""
Dataset Preparation for Question Generation Fine-Tuning
=========================================================
Downloads, processes, and unifies multiple QA datasets into a standardised
format for training the Seq2Seq question generation model.

Supported datasets:
  - SQuAD 2.0  (reading comprehension)
  - SciQ       (science MCQs with distractors)
  - RACE       (multi-level reading comprehension)

Output format (JSONL):
  {
    "context":  "...",
    "question": "...",
    "answer":   "...",
    "type":     "MCQ | ShortAnswer | ...",
    "difficulty": "easy | medium | hard",
    "distractors": ["...", "..."],  // if MCQ
    "source_dataset": "squad | sciq | race"
  }
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

logger = logging.getLogger(__name__)


# ─── SQuAD Processing ──────────────────────────────────────────────────────────

def process_squad(split: str = "train", max_samples: int = 20000) -> List[Dict]:
    """
    Download and process SQuAD 2.0 for question generation.
    Converts extractive QA pairs into generative training format.

    Parameters
    ----------
    split : str
        Dataset split to process (train/validation).
    max_samples : int
        Maximum number of samples to extract.
    """
    from datasets import load_dataset

    logger.info("Loading SQuAD dataset (split=%s)...", split)
    dataset = load_dataset("squad_v2", split=split)

    processed = []
    for item in tqdm(dataset, desc="Processing SQuAD", total=min(len(dataset), max_samples)):
        if len(processed) >= max_samples:
            break

        context = item["context"]
        question = item["question"]
        answers = item["answers"]["text"]

        if not answers:
            continue  # skip unanswerable questions

        answer = answers[0]

        # Build the training record
        record = {
            "context": context,
            "question": question,
            "answer": answer,
            "type": "ShortAnswer",
            "difficulty": _estimate_difficulty_from_length(question),
            "distractors": [],
            "source_dataset": "squad",
            "input_text": f"generate question: {context}",
            "target_text": json.dumps(
                {
                    "question": question,
                    "answer": answer,
                    "options": [],
                    "explanation": f"Based on the context: '{context[:100]}...'",
                }
            ),
        }
        processed.append(record)

    logger.info("Processed %d SQuAD samples.", len(processed))
    return processed


# ─── SciQ Processing ───────────────────────────────────────────────────────────

def process_sciq(split: str = "train", max_samples: int = 10000) -> List[Dict]:
    """
    Download and process SciQ dataset.
    SciQ has MCQ format with context, question, correct answer, and 3 distractors.
    """
    from datasets import load_dataset

    logger.info("Loading SciQ dataset (split=%s)...", split)
    dataset = load_dataset("allenai/sciq", split=split)

    processed = []
    for item in tqdm(dataset, desc="Processing SciQ", total=min(len(dataset), max_samples)):
        if len(processed) >= max_samples:
            break

        context = item.get("support", "")
        if not context or len(context) < 30:
            continue

        question = item["question"]
        correct_answer = item["correct_answer"]
        distractors = [
            item.get("distractor1", ""),
            item.get("distractor2", ""),
            item.get("distractor3", ""),
        ]
        distractors = [d for d in distractors if d]  # remove empty

        options = [correct_answer] + distractors
        random.shuffle(options)

        record = {
            "context": context,
            "question": question,
            "answer": correct_answer,
            "type": "MCQ",
            "difficulty": _estimate_difficulty_from_length(question),
            "distractors": distractors,
            "source_dataset": "sciq",
            "input_text": f"generate MCQ question: {context}",
            "target_text": json.dumps(
                {
                    "question": question,
                    "answer": correct_answer,
                    "options": options[:4],
                    "explanation": f"The correct answer is '{correct_answer}' as supported by the context.",
                }
            ),
        }
        processed.append(record)

    logger.info("Processed %d SciQ samples.", len(processed))
    return processed


# ─── RACE Processing ───────────────────────────────────────────────────────────

def process_race(split: str = "train", max_samples: int = 15000) -> List[Dict]:
    """
    Download and process RACE dataset (Reading Comprehension from English Exams).
    RACE has difficulty labels ('middle' and 'high') which map well to
    our medium/hard classification.
    """
    from datasets import load_dataset

    logger.info("Loading RACE dataset (split=%s)...", split)

    processed = []

    for difficulty_level in ["middle", "high"]:
        try:
            dataset = load_dataset("ehovy/race", difficulty_level, split=split)
        except Exception as exc:
            logger.warning("Could not load RACE/%s: %s", difficulty_level, exc)
            continue

        diff_label = "medium" if difficulty_level == "middle" else "hard"

        for item in tqdm(
            dataset,
            desc=f"Processing RACE/{difficulty_level}",
            total=min(len(dataset), max_samples // 2),
        ):
            if len(processed) >= max_samples:
                break

            context = item["article"]
            question = item["question"]
            options = item["options"]
            answer_key = item["answer"]  # "A", "B", "C", or "D"

            # Map letter to actual answer
            answer_idx = ord(answer_key) - ord("A")
            if answer_idx < 0 or answer_idx >= len(options):
                continue
            correct_answer = options[answer_idx]
            distractors = [o for i, o in enumerate(options) if i != answer_idx]

            record = {
                "context": context,
                "question": question,
                "answer": correct_answer,
                "type": "MCQ",
                "difficulty": diff_label,
                "distractors": distractors,
                "source_dataset": "race",
                "input_text": f"generate {diff_label} MCQ question: {context[:512]}",
                "target_text": json.dumps(
                    {
                        "question": question,
                        "answer": correct_answer,
                        "options": options,
                        "explanation": f"The correct answer is '{correct_answer}'.",
                    }
                ),
            }
            processed.append(record)

    logger.info("Processed %d RACE samples.", len(processed))
    return processed


# ─── Difficulty Estimation Heuristic ────────────────────────────────────────────

def _estimate_difficulty_from_length(question: str) -> str:
    """Simple heuristic: longer questions with more complex words → harder."""
    words = question.split()
    word_count = len(words)
    avg_word_len = sum(len(w) for w in words) / max(word_count, 1)

    if word_count <= 8 and avg_word_len < 5:
        return "easy"
    elif word_count <= 15 or avg_word_len < 6:
        return "medium"
    else:
        return "hard"


# ─── Unified Dataset Preparation ───────────────────────────────────────────────

def prepare_all_datasets(
    output_dir: str = None,
    squad_samples: int = 20000,
    sciq_samples: int = 10000,
    race_samples: int = 15000,
    val_ratio: float = 0.1,
) -> Dict[str, str]:
    """
    Download, process, and merge all datasets into train and validation splits.

    Parameters
    ----------
    output_dir : str
        Directory to save the processed datasets.
    squad_samples : int
        Max SQuAD samples.
    sciq_samples : int
        Max SciQ samples.
    race_samples : int
        Max RACE samples.
    val_ratio : float
        Fraction of data to use for validation.

    Returns
    -------
    Dict with keys 'train_path' and 'val_path'.
    """
    from config import DATA_PROCESSED_DIR

    out_dir = Path(output_dir or str(DATA_PROCESSED_DIR))
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []

    # Process each dataset
    try:
        all_records.extend(process_squad(max_samples=squad_samples))
    except Exception as exc:
        logger.error("SQuAD processing failed: %s", exc)

    try:
        all_records.extend(process_sciq(max_samples=sciq_samples))
    except Exception as exc:
        logger.error("SciQ processing failed: %s", exc)

    try:
        all_records.extend(process_race(max_samples=race_samples))
    except Exception as exc:
        logger.error("RACE processing failed: %s", exc)

    if not all_records:
        logger.error("No datasets could be processed!")
        return {}

    # Shuffle
    random.shuffle(all_records)

    # Split into train and validation
    val_count = max(1, int(len(all_records) * val_ratio))
    val_records = all_records[:val_count]
    train_records = all_records[val_count:]

    # Save as JSONL
    train_path = out_dir / "train.json"
    val_path = out_dir / "val.json"

    _save_jsonl(train_records, train_path)
    _save_jsonl(val_records, val_path)

    # Also save statistics
    stats = {
        "total_samples": len(all_records),
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "sources": {
            "squad": sum(1 for r in all_records if r["source_dataset"] == "squad"),
            "sciq": sum(1 for r in all_records if r["source_dataset"] == "sciq"),
            "race": sum(1 for r in all_records if r["source_dataset"] == "race"),
        },
        "types": {
            "MCQ": sum(1 for r in all_records if r["type"] == "MCQ"),
            "ShortAnswer": sum(
                1 for r in all_records if r["type"] == "ShortAnswer"
            ),
        },
        "difficulties": {
            "easy": sum(1 for r in all_records if r["difficulty"] == "easy"),
            "medium": sum(1 for r in all_records if r["difficulty"] == "medium"),
            "hard": sum(1 for r in all_records if r["difficulty"] == "hard"),
        },
    }

    stats_path = out_dir / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info(
        "Dataset preparation complete: %d train, %d val. Saved to %s",
        len(train_records),
        len(val_records),
        out_dir,
    )

    return {"train_path": str(train_path), "val_path": str(val_path)}


def _save_jsonl(records: list[dict], path: Path):
    """Save records as a JSON array file (HuggingFace compatible)."""
    # Save as a list (load_dataset("json") expects this)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    logger.info("Saved %d records to %s", len(records), path)


# ─── CLI Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = prepare_all_datasets()
    if result:
        print(f"\n✅ Dataset prepared successfully!")
        print(f"   Train: {result.get('train_path')}")
        print(f"   Val:   {result.get('val_path')}")
    else:
        print("❌ Dataset preparation failed.")
