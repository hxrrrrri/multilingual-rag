"""
Central configuration — reads from environment / .env file.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # App
    APP_NAME: str = "multilingual-rag"
    APP_ENV: str = "development"
    DEBUG: bool = True
    SECRET_KEY: str = "dev-secret-change-in-prod"

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://raguser:ragpassword@localhost:5432/ragdb"

    # Qdrant
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "multilingual_docs"

    # Elasticsearch
    ES_HOST: str = "http://localhost:9200"
    ES_INDEX: str = "multilingual_docs"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # LLM
    LLM_BASE_URL: str = "http://localhost:11434/v1"
    LLM_API_KEY: str = "ollama"
    LLM_MODEL: str = "llama3"

    # Embeddings
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"
    LORA_ADAPTER_PATH: str = ""

    # Reranker
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # NLI
    NLI_MODEL: str = "cross-encoder/nli-deberta-v3-small"

    # Retrieval
    RETRIEVAL_TOP_K: int = 20
    RERANK_TOP_N: int = 5
    RRF_K: int = 60

    # MLFlow
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT: str = "multilingual-rag"

    # Server
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # Files
    UPLOAD_DIR: str = "./uploads"
    MAX_FILE_SIZE_MB: int = 50


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
