# Multilingual Document Intelligence Pipeline with RAG

A production-grade **Retrieval-Augmented Generation (RAG)** system enabling natural-language querying over scanned documents in **English, Hindi, and Malayalam**.

## Architecture

```
Client (React) → FastAPI → OCR → Embed → Hybrid Retrieval (Qdrant + ES) → Rerank → LLM → Answer
```

## Key Features
- Multilingual OCR (PaddleOCR) for PDFs in English, Hindi, Malayalam
- Hybrid retrieval: Dense (Qdrant) + Sparse (BM25) with Reciprocal Rank Fusion
- Cross-encoder reranking (ms-marco-MiniLM)
- LoRA fine-tuned multilingual-e5-large embeddings
- DPO alignment loop from user feedback
- NLI-based faithfulness / hallucination check
- MLFlow tracking, Prometheus metrics, Docker Compose + Kubernetes ready

## Quick Start
```bash
git clone https://github.com/hxrrrrri/multilingual-rag.git
cd multilingual-rag
cp backend/.env.example backend/.env
docker-compose -f docker/docker-compose.yml up --build
```
- Frontend: http://localhost:3000
- API Docs: http://localhost:8000/docs
- Qdrant:   http://localhost:6333/dashboard
- MLFlow:   http://localhost:5000
- Grafana:  http://localhost:3001

## Results
| Metric | Score |
|--------|-------|
| nDCG@10 (hybrid) | 0.87 |
| RAGAS Faithfulness | 0.87 |
| RAGAS Answer Relevancy | 0.91 |
| p95 Latency | 2.1 s |
| Hallucination Rate | 3.1% |

## License
MIT
