"""
Fine-Tuning Script for Question Generation Models
====================================================
Supports two training strategies:

  1. Full Fine-Tuning (FLAN-T5-base / T5-base)
     - Standard HuggingFace Trainer with Seq2SeqTrainer
     - Suitable for T5-base/large on GPUs with 16+ GB VRAM

  2. LoRA / QLoRA Fine-Tuning (FLAN-T5-XL, Mistral, LLaMA)
     - Parameter-efficient fine-tuning using PEFT library
     - 4-bit quantisation via bitsandbytes for large models
     - Suitable for consumer GPUs (8-12 GB VRAM)

Usage:
    # Full fine-tuning on T5
    python -m training.train --model google/flan-t5-base --strategy full

    # LoRA fine-tuning on FLAN-T5-XL
    python -m training.train --model google/flan-t5-xl --strategy lora

    # QLoRA fine-tuning on Mistral
    python -m training.train --model mistralai/Mistral-7B-Instruct-v0.2 --strategy qlora
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# ─── Dataset Loading ────────────────────────────────────────────────────────────

def load_training_data(train_path: str, val_path: str):
    """Load preprocessed training data from JSONL files."""
    from datasets import load_dataset

    data_files = {"train": train_path}
    if val_path and Path(val_path).exists():
        data_files["validation"] = val_path

    dataset = load_dataset("json", data_files=data_files)
    logger.info(
        "Loaded dataset: %d train, %d validation",
        len(dataset["train"]),
        len(dataset.get("validation", [])),
    )
    return dataset


# ─── Tokenization ──────────────────────────────────────────────────────────────

def tokenize_seq2seq(examples, tokenizer, max_input_length=512, max_target_length=256):
    """
    Tokenize input-target pairs for Seq2Seq models (T5/BART).
    """
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
    )

    labels = tokenizer(
        examples["target_text"],
        max_length=max_target_length,
        truncation=True,
        padding="max_length",
    )

    # Replace pad token id with -100 so they are ignored in loss
    label_ids = labels["input_ids"]
    label_ids = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in label_ids
    ]
    model_inputs["labels"] = label_ids

    return model_inputs


def tokenize_causal(examples, tokenizer, max_length=768):
    """
    Tokenize for causal LLMs (Mistral, LLaMA).
    Combines input + target into a single sequence.
    """
    texts = [
        f"### Instruction:\n{inp}\n\n### Response:\n{tgt}"
        for inp, tgt in zip(examples["input_text"], examples["target_text"])
    ]

    tokenized = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


# ─── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics_factory(tokenizer):
    """Create a compute_metrics function for the Trainer."""

    def compute_metrics(eval_pred):
        import numpy as np
        from rouge_score import rouge_scorer

        predictions, labels = eval_pred
        # Decode predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Replace -100 in labels
        labels = [[l for l in label if l != -100] for label in labels]
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Calculate ROUGE
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []

        for pred, ref in zip(decoded_preds, decoded_labels):
            scores = scorer.score(ref, pred)
            rouge1_scores.append(scores["rouge1"].fmeasure)
            rouge2_scores.append(scores["rouge2"].fmeasure)
            rougeL_scores.append(scores["rougeL"].fmeasure)

        return {
            "rouge1": round(np.mean(rouge1_scores), 4),
            "rouge2": round(np.mean(rouge2_scores), 4),
            "rougeL": round(np.mean(rougeL_scores), 4),
        }

    return compute_metrics


# ─── Full Fine-Tuning (Seq2Seq) ────────────────────────────────────────────────

def train_full_seq2seq(
    model_name: str,
    train_path: str,
    val_path: Optional[str] = None,
    output_dir: str = None,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_input_length: int = 512,
    max_target_length: int = 256,
):
    """
    Full fine-tuning of a Seq2Seq model (T5/BART/FLAN-T5).
    """
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq,
    )
    from config import MODELS_DIR

    output_dir = output_dir or str(MODELS_DIR / "qg_finetuned")

    logger.info("Starting FULL fine-tuning: %s", model_name)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Load and tokenize dataset
    dataset = load_training_data(train_path, val_path)
    tokenized = dataset.map(
        lambda ex: tokenize_seq2seq(
            ex, tokenizer, max_input_length, max_target_length
        ),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding=True
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch" if val_path else "no",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=epochs,
        predict_with_generate=True,
        generation_max_length=max_target_length,
        fp16=torch.cuda.is_available(),
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        report_to="none",
        load_best_model_at_end=True if val_path else False,
        metric_for_best_model="rougeL" if val_path else None,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_factory(tokenizer) if val_path else None,
    )

    logger.info("Training started...")
    trainer.train()

    # Save final model
    final_dir = f"{output_dir}/final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info("Model saved to %s", final_dir)

    return final_dir


# ─── LoRA / QLoRA Fine-Tuning ──────────────────────────────────────────────────

def train_lora(
    model_name: str,
    train_path: str,
    val_path: Optional[str] = None,
    output_dir: str = None,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    use_4bit: bool = False,
    max_input_length: int = 512,
    max_target_length: int = 256,
):
    """
    LoRA/QLoRA fine-tuning for large models (FLAN-T5-XL, Mistral, LLaMA).
    """
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        AutoModelForCausalLM,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        Trainer,
        TrainingArguments,
        DataCollatorForSeq2Seq,
        DataCollatorForLanguageModeling,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    from config import MODELS_DIR

    output_dir = output_dir or str(MODELS_DIR / "qg_lora")

    logger.info("Starting LoRA fine-tuning: %s (4bit=%s)", model_name, use_4bit)

    # Determine if model is Seq2Seq or Causal
    is_seq2seq = any(
        tag in model_name.lower() for tag in ["t5", "bart", "pegasus"]
    )

    # Quantisation config (for QLoRA)
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    if is_seq2seq:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto" if use_4bit else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        task_type = TaskType.SEQ_2_SEQ_LM
        target_modules = ["q", "v"]  # T5 attention layers
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto" if use_4bit else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        task_type = TaskType.CAUSAL_LM
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=task_type,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and tokenize dataset
    dataset = load_training_data(train_path, val_path)

    if is_seq2seq:
        tokenized = dataset.map(
            lambda ex: tokenize_seq2seq(
                ex, tokenizer, max_input_length, max_target_length
            ),
            batched=True,
            remove_columns=dataset["train"].column_names,
        )
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch" if val_path else "no",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            save_total_limit=2,
            num_train_epochs=epochs,
            predict_with_generate=True,
            fp16=torch.cuda.is_available(),
            logging_steps=50,
            report_to="none",
            gradient_accumulation_steps=4,
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized.get("validation"),
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
    else:
        tokenized = dataset.map(
            lambda ex: tokenize_causal(ex, tokenizer),
            batched=True,
            remove_columns=dataset["train"].column_names,
        )
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False
        )

        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch" if val_path else "no",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            weight_decay=0.01,
            save_total_limit=2,
            num_train_epochs=epochs,
            fp16=torch.cuda.is_available(),
            logging_steps=50,
            report_to="none",
            gradient_accumulation_steps=4,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized.get("validation"),
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

    logger.info("LoRA training started...")
    trainer.train()

    # Save LoRA adapters
    adapter_dir = f"{output_dir}/lora_adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    logger.info("LoRA adapter saved to %s", adapter_dir)

    return adapter_dir


# ─── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_model(
    model_path: str,
    test_path: str,
    num_samples: int = 100,
) -> dict:
    """
    Evaluate a trained model on test samples.

    Returns metrics: ROUGE-1, ROUGE-2, ROUGE-L, and sample outputs.
    """
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from rouge_score import rouge_scorer
    import numpy as np

    logger.info("Evaluating model: %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Load test data
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    test_data = test_data[:num_samples]
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    all_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    sample_outputs = []

    for item in test_data[:num_samples]:
        input_text = item["input_text"]
        reference = item["target_text"]

        inputs = tokenizer(
            input_text, return_tensors="pt", max_length=512, truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=256)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

        scores = scorer.score(reference, prediction)
        for key in all_scores:
            all_scores[key].append(scores[key].fmeasure)

        if len(sample_outputs) < 5:
            sample_outputs.append(
                {
                    "input": input_text[:200],
                    "reference": reference[:200],
                    "prediction": prediction[:200],
                }
            )

    metrics = {
        k: round(np.mean(v), 4) for k, v in all_scores.items()
    }
    metrics["num_samples"] = len(test_data)
    metrics["sample_outputs"] = sample_outputs

    logger.info("Evaluation results: %s", {k: v for k, v in metrics.items() if k != "sample_outputs"})
    return metrics


# ─── CLI Entry Point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a Question Generation model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/flan-t5-base",
        help="HuggingFace model name or path.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["full", "lora", "qlora"],
        default="full",
        help="Training strategy: full, lora, or qlora.",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to training data JSON.",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to validation data JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save the trained model.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--evaluate", action="store_true", help="Evaluate after training.")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Resolve data paths
    from config import DATA_PROCESSED_DIR

    train_path = args.train_data or str(DATA_PROCESSED_DIR / "train.json")
    val_path = args.val_data or str(DATA_PROCESSED_DIR / "val.json")

    if not Path(train_path).exists():
        logger.error(
            "Training data not found at %s. Run `python -m training.prepare_dataset` first.",
            train_path,
        )
        return

    # Train
    if args.strategy == "full":
        model_dir = train_full_seq2seq(
            model_name=args.model,
            train_path=train_path,
            val_path=val_path if Path(val_path).exists() else None,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )
    elif args.strategy in ("lora", "qlora"):
        model_dir = train_lora(
            model_name=args.model,
            train_path=train_path,
            val_path=val_path if Path(val_path).exists() else None,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr if args.strategy == "lora" else 2e-4,
            lora_r=args.lora_r,
            use_4bit=(args.strategy == "qlora"),
        )
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")

    print(f"\n✅ Training complete! Model saved to: {model_dir}")

    # Optional evaluation
    if args.evaluate and Path(val_path).exists():
        print("\n📊 Running evaluation...")
        metrics = evaluate_model(model_dir, val_path)
        print(f"   ROUGE-1: {metrics['rouge1']}")
        print(f"   ROUGE-2: {metrics['rouge2']}")
        print(f"   ROUGE-L: {metrics['rougeL']}")


if __name__ == "__main__":
    main()
