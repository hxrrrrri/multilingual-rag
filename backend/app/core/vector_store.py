"""
Qdrant vector store + Elasticsearch BM25 initialisation.
"""
from loguru import logger
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams
from elasticsearch import AsyncElasticsearch

from app.core.config import settings

qdrant_client: AsyncQdrantClient = None
es_client: AsyncElasticsearch = None

VECTOR_SIZE = 1024  # multilingual-e5-large output dim


async def init_vector_stores():
    global qdrant_client, es_client

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_client = AsyncQdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
    )
    collections = await qdrant_client.get_collections()
    names = [c.name for c in collections.collections]
    if settings.QDRANT_COLLECTION not in names:
        await qdrant_client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info(f"Created Qdrant collection: {settings.QDRANT_COLLECTION}")
    else:
        logger.info(f"Qdrant collection exists: {settings.QDRANT_COLLECTION}")

    # ── Elasticsearch ─────────────────────────────────────────────────────────
    es_client = AsyncElasticsearch(settings.ES_HOST)
    if not await es_client.indices.exists(index=settings.ES_INDEX):
        await es_client.indices.create(
            index=settings.ES_INDEX,
            body={
                "mappings": {
                    "properties": {
                        "chunk_id":  {"type": "keyword"},
                        "doc_id":    {"type": "keyword"},
                        "text":      {"type": "text", "analyzer": "standard"},
                        "language":  {"type": "keyword"},
                        "page":      {"type": "integer"},
                    }
                }
            },
        )
        logger.info(f"Created ES index: {settings.ES_INDEX}")
    else:
        logger.info(f"ES index exists: {settings.ES_INDEX}")


def get_qdrant() -> AsyncQdrantClient:
    return qdrant_client


def get_es() -> AsyncElasticsearch:
    return es_client
