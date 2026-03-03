# Architecture Deep-Dive

## Retrieval Ablation Results

| Method | nDCG@10 |
|--------|---------|
| BM25 only | 0.61 |
| Dense only | 0.71 |
| Hybrid RRF | 0.79 |
| Hybrid + Reranker | **0.87** |

## Pipeline Stages

### 1. Ingestion
PDF → pdfplumber → confidence check → PaddleOCR fallback
  → language detection (Unicode range: en/hi/ml)
  → semantic chunking (400 tokens, 10% overlap)
  → multilingual-e5-large embedding
  → Qdrant (dense) + Elasticsearch (BM25) + PostgreSQL

### 2. Query
Query → embed "query: " prefix → Qdrant ANN (top-20) + ES BM25 (top-20)
  → RRF fusion (k=60) → cross-encoder reranking (top-5)
  → LLM generation → NLI faithfulness check → return

### 3. DPO Loop
User feedback → PostgreSQL → weekly dpo_finetune.py
  → preference pairs → QLoRA fine-tune (r=16) → LoRA adapter → restart
