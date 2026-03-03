from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter(
    "rag_requests_total", "Total API requests", ["endpoint", "method", "status"]
)
REQUEST_LATENCY = Histogram(
    "rag_request_latency_seconds", "Request latency", ["endpoint"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0],
)
OCR_LATENCY = Histogram(
    "rag_ocr_latency_seconds", "OCR latency per page",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0],
)
EMBEDDING_LATENCY = Histogram(
    "rag_embedding_latency_seconds", "Embedding latency",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0],
)
RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_latency_seconds", "Retrieval latency",
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0],
)
GENERATION_LATENCY = Histogram(
    "rag_generation_latency_seconds", "LLM generation latency",
    buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0],
)
ANSWER_CONFIDENCE = Histogram(
    "rag_answer_confidence", "Answer confidence distribution",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
FEEDBACK_COUNT = Counter(
    "rag_user_feedback_total", "User feedback counts", ["sentiment"]
)
DOCUMENTS_INDEXED = Gauge("rag_documents_indexed_total", "Total documents indexed")
OCR_CONFIDENCE_GAUGE = Gauge("rag_ocr_confidence_avg", "Rolling average OCR confidence")
