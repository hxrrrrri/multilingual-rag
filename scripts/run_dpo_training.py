"""
DPO (Direct Preference Optimization) training for the generation LLM.

Reads feedback data (thumbs up/down from the API) and constructs
preference pairs to align the model.

DPO Loss:
  L = -E[ log σ( β * log π_θ(y_w|x)/π_ref(y_w|x)
                - β * log π_θ(y_l|x)/π_ref(y_l|x) ) ]

Where y_w = preferred (thumbs up) response, y_l = rejected (thumbs down).

Usage:
  python scripts/run_dpo_training.py \\
      --feedback_file ./feedback_data.jsonl \\
      --queries_file  ./query_history.jsonl \\
      --output_dir    ./dpo_model \\
      --beta 0.1
"""

import argparse
import json
import os
from loguru import logger


def load_preference_pairs(feedback_file: str, queries_file: str) -> list:
    """
    Match thumbs-up and thumbs-down feedback to their query-answer pairs
    and construct (prompt, chosen, rejected) triplets for DPO.
    """
    # Load query history
    queries = {}
    if os.path.exists(queries_file):
        with open(queries_file) as f:
            for line in f:
                item = json.loads(line.strip())
                queries[item["query_id"]] = item

    # Load feedback
    positive, negative = {}, {}
    if os.path.exists(feedback_file):
        with open(feedback_file) as f:
            for line in f:
                item = json.loads(line.strip())
                qid = item["query_id"]
                if item["sentiment"] == "positive":
                    positive[qid] = item
                else:
                    negative[qid] = item

    # Build pairs: for each negative, find a positive on the same query or similar
    pairs = []
    for qid, neg in negative.items():
        if qid in queries and qid in positive:
            q = queries[qid]
            pairs.append({
                "prompt":   q["query"],
                "chosen":   positive[qid].get("answer", ""),
                "rejected": neg.get("answer", ""),
            })

    logger.info(f"Constructed {len(pairs)} DPO preference pairs")
    return pairs


def run_dpo(args):
    pairs = load_preference_pairs(args.feedback_file, args.queries_file)
    if len(pairs) < 10:
        logger.warning(f"Only {len(pairs)} pairs available — need at least 10 for meaningful DPO training")
        logger.info("Collect more feedback via the UI thumbs up/down buttons before rerunning DPO")
        return

    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    from trl import DPOTrainer, DPOConfig

    import mlflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("dpo-alignment")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"DPO training on {device} with {len(pairs)} pairs | beta={args.beta}")

    model_name = os.getenv("LLM_MODEL", "google/flan-t5-xl")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                              lora_dropout=0.05, bias="none")
    model = get_peft_model(model, lora_config)

    dataset = Dataset.from_list(pairs)
    train_size = int(0.9 * len(dataset))
    ds_split = dataset.train_test_split(test_size=len(dataset) - train_size)

    dpo_config = DPOConfig(
        beta=args.beta,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=5e-7,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=100,
        evaluation_strategy="steps",
        eval_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="mlflow",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=ds_split["train"],
        eval_dataset=ds_split["test"],
        tokenizer=tokenizer,
    )

    with mlflow.start_run():
        mlflow.log_params({"beta": args.beta, "epochs": args.epochs,
                           "pairs": len(pairs), "model": model_name})
        trainer.train()
        trainer.save_model(args.output_dir)
        mlflow.log_artifacts(args.output_dir, artifact_path="dpo_model")

    logger.info(f"DPO model saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback_file", default="./feedback_data.jsonl")
    parser.add_argument("--queries_file",  default="./query_history.jsonl")
    parser.add_argument("--output_dir",    default="./dpo_model")
    parser.add_argument("--beta",          type=float, default=0.1)
    parser.add_argument("--epochs",        type=int,   default=2)
    parser.add_argument("--batch_size",    type=int,   default=4)
    args = parser.parse_args()
    run_dpo(args)
