"""
DPO Fine-tuning script for the embedding model.

Usage:
    python scripts/dpo_finetune.py \
        --api-url http://localhost:8000 \
        --base-model intfloat/multilingual-e5-large \
        --output-dir ./lora_adapter \
        --min-pairs 100

This script:
  1. Fetches preference pairs from /api/v1/feedback/export
  2. Fine-tunes the embedding model with DPO using HuggingFace TRL
  3. Saves the LoRA adapter
  4. Logs metrics to MLFlow
"""
import argparse
import json
import os
import sys

import requests
from loguru import logger


def fetch_pairs(api_url: str, min_pairs: int):
    resp = requests.get(f"{api_url}/api/v1/feedback/export")
    resp.raise_for_status()
    data = resp.json()
    pairs = data["pairs"]
    logger.info(f"Fetched {len(pairs)} preference pairs.")
    if len(pairs) < min_pairs:
        logger.warning(f"Only {len(pairs)} pairs available (need {min_pairs}). Skipping.")
        sys.exit(0)
    return pairs


def run_dpo(pairs, base_model: str, output_dir: str):
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, TaskType
        from transformers import AutoTokenizer, AutoModel
        from trl import DPOTrainer, DPOConfig
    except ImportError as e:
        logger.error(f"Missing dependency: {e}. Install requirements.txt.")
        sys.exit(1)

    logger.info(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model     = AutoModel.from_pretrained(base_model)

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["query", "key", "value"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Build dataset
    dataset = Dataset.from_list([
        {"prompt": p["query"], "chosen": p["chosen"], "rejected": p["rejected"]}
        for p in pairs
    ])

    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-7,
        beta=0.1,
        logging_steps=10,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        report_to="mlflow",
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    logger.info("Starting DPO training…")
    trainer.train()
    trainer.save_model(output_dir)
    logger.info(f"LoRA adapter saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url",     default="http://localhost:8000")
    parser.add_argument("--base-model",  default="intfloat/multilingual-e5-large")
    parser.add_argument("--output-dir",  default="./lora_adapter")
    parser.add_argument("--min-pairs",   type=int, default=100)
    args = parser.parse_args()

    pairs = fetch_pairs(args.api_url, args.min_pairs)
    run_dpo(pairs, args.base_model, args.output_dir)
