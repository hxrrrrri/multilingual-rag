"""
Fine-tune multilingual-e5-large with LoRA for domain adaptation.

Uses Multiple Negatives Ranking Loss (MNRL):
  For each (query, positive_passage) pair in a batch,
  all other passages act as hard negatives.

Usage:
  python scripts/finetune_embeddings.py \\
      --data_path ./data/train_pairs.jsonl \\
      --output_dir ./lora_adapter \\
      --epochs 3 \\
      --rank 16 \\
      --batch_size 32
"""

import argparse
import json
import os
import mlflow

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from loguru import logger


# ── Dataset ────────────────────────────────────────────────────────────────────

class PairDataset(Dataset):
    """
    Expects a JSONL file where each line is:
    {"query": "...", "positive": "..."}
    """
    def __init__(self, path: str):
        self.pairs = []
        with open(path) as f:
            for line in f:
                item = json.loads(line.strip())
                if "query" in item and "positive" in item:
                    self.pairs.append(item)
        logger.info(f"Loaded {len(self.pairs)} training pairs from {path}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]["query"], self.pairs[idx]["positive"]


# ── Encoding ───────────────────────────────────────────────────────────────────

def mean_pool(model_output, attention_mask):
    token_emb = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_emb.size()).float()
    return (token_emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def encode(tokenizer, model, texts: list, prefix: str, device: str) -> torch.Tensor:
    prefixed = [f"{prefix}{t}" for t in texts]
    enc = tokenizer(prefixed, padding=True, truncation=True, max_length=512, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc)
    emb = mean_pool(out, enc["attention_mask"])
    return torch.nn.functional.normalize(emb, p=2, dim=1)


# ── Loss ───────────────────────────────────────────────────────────────────────

def mnrl_loss(query_emb: torch.Tensor, pos_emb: torch.Tensor, temperature: float = 0.02) -> torch.Tensor:
    """
    Multiple Negatives Ranking Loss.
    Similarity matrix: (batch_size x batch_size).
    Diagonal = positive pairs. Off-diagonal = in-batch negatives.
    """
    sim = torch.matmul(query_emb, pos_emb.T) / temperature
    labels = torch.arange(sim.size(0), device=sim.device)
    return torch.nn.functional.cross_entropy(sim, labels)


# ── Training ───────────────────────────────────────────────────────────────────

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on {device}")

    # Tokenizer + base model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    base_model = AutoModel.from_pretrained(args.model_name)

    # LoRA config
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        target_modules=["query", "key", "value"],  # attention projections
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model.to(device)
    model.print_trainable_parameters()

    # Data
    dataset = PairDataset(args.data_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # Optimiser
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    # MLFlow tracking
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("embedding-finetune")

    with mlflow.start_run():
        mlflow.log_params({
            "model_name":  args.model_name,
            "lora_rank":   args.rank,
            "batch_size":  args.batch_size,
            "epochs":      args.epochs,
            "lr":          args.lr,
            "train_pairs": len(dataset),
        })

        global_step = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0

            for batch_queries, batch_positives in loader:
                optimizer.zero_grad()

                q_emb = encode(tokenizer, model, list(batch_queries), "query: ", device)
                p_emb = encode(tokenizer, model, list(batch_positives), "passage: ", device)

                loss = mnrl_loss(q_emb, p_emb)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                global_step += 1

                if global_step % 50 == 0:
                    logger.info(f"Epoch {epoch} | step {global_step} | loss={loss.item():.4f}")
                    mlflow.log_metric("train_loss", loss.item(), step=global_step)

            avg_loss = epoch_loss / len(loader)
            logger.info(f"Epoch {epoch} complete | avg_loss={avg_loss:.4f}")
            mlflow.log_metric("epoch_loss", avg_loss, step=epoch)

        # Save LoRA adapter
        os.makedirs(args.output_dir, exist_ok=True)
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        mlflow.log_artifacts(args.output_dir, artifact_path="lora_adapter")
        logger.info(f"LoRA adapter saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="intfloat/multilingual-e5-large")
    parser.add_argument("--data_path",  default="./data/train_pairs.jsonl")
    parser.add_argument("--output_dir", default="./lora_adapter")
    parser.add_argument("--epochs",     type=int, default=3)
    parser.add_argument("--rank",       type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr",         type=float, default=2e-4)
    args = parser.parse_args()
    train(args)
