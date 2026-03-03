"""
Evaluate the full RAG pipeline using RAGAS metrics.

Metrics computed:
  - Faithfulness       (answer supported by context?)
  - Answer Relevancy   (answer relevant to question?)
  - Context Recall     (context covers the ground truth?)
  - Context Precision  (retrieved context is relevant?)

Usage:
  python scripts/evaluate_pipeline.py \\
      --eval_file ./data/eval_dataset.jsonl \\
      --api_url   http://localhost:8000
"""

import argparse
import json
import os
import httpx
from loguru import logger
import mlflow


def load_eval_dataset(path: str) -> list:
    """
    Expects JSONL with lines:
    {"question": "...", "ground_truth": "...", "document_ids": [...]}
    """
    items = []
    with open(path) as f:
        for line in f:
            item = json.loads(line.strip())
            if "question" in item and "ground_truth" in item:
                items.append(item)
    logger.info(f"Loaded {len(items)} eval examples")
    return items


def query_api(api_url: str, question: str, doc_ids: list = None) -> dict:
    """Hit the live /api/query endpoint."""
    payload = {"query": question, "top_k": 5, "include_sources": True}
    if doc_ids:
        payload["document_ids"] = doc_ids
    r = httpx.post(f"{api_url}/api/query", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def evaluate(args):
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
    from datasets import Dataset

    eval_data = load_eval_dataset(args.eval_file)
    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    logger.info(f"Running {len(eval_data)} queries against {args.api_url}")
    for i, item in enumerate(eval_data):
        try:
            resp = query_api(args.api_url, item["question"], item.get("document_ids"))
            rows["question"].append(item["question"])
            rows["answer"].append(resp["answer"])
            rows["contexts"].append([s["text"] for s in resp.get("sources", [])])
            rows["ground_truth"].append(item["ground_truth"])
            logger.debug(f"[{i+1}/{len(eval_data)}] Query OK")
        except Exception as e:
            logger.warning(f"Query failed for '{item['question'][:40]}': {e}")

    if not rows["question"]:
        logger.error("No successful queries — check API is running and documents are indexed")
        return

    ds = Dataset.from_dict(rows)
    results = ragas_evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )

    logger.info("=" * 50)
    logger.info("RAGAS EVALUATION RESULTS")
    logger.info("=" * 50)
    for metric, score in results.items():
        logger.info(f"  {metric:30s}: {score:.4f}")
    logger.info("=" * 50)

    # Log to MLFlow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("rag-evaluation")
    with mlflow.start_run():
        mlflow.log_params({"eval_examples": len(rows["question"]), "api_url": args.api_url})
        for metric, score in results.items():
            mlflow.log_metric(metric, float(score))

    # Save results
    output = {"results": dict(results), "n_examples": len(rows["question"])}
    with open("eval_results.json", "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to eval_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", default="./data/eval_dataset.jsonl")
    parser.add_argument("--api_url",   default="http://localhost:8000")
    args = parser.parse_args()
    evaluate(args)
