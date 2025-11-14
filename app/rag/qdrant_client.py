# app/rag/qdrant_client.py

import os
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# Environment variables (must be set in .env / Render)
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

GENERAL_COLLECTION = os.getenv("QDRANT_GENERAL_COLLECTION", "general_docs")
PERSONAL_COLLECTION = os.getenv("QDRANT_PERSONAL_COLLECTION", "personal_knowledge")

# Must match embedding model dimension (all-MiniLM-L6-v2 -> 384 dims)
EMBEDDING_DIM = 384


def get_qdrant_client() -> QdrantClient:
    """
    Create a Qdrant client for Cloud cluster.
    """
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise RuntimeError("QDRANT_URL and QDRANT_API_KEY must be set as environment variables.")

    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )


def ensure_collections_exist(collection_names: List[str] | None = None) -> None:
    """
    Ensure that required collections exist with the correct vector configuration.
    """
    client = get_qdrant_client()

    if collection_names is None:
        collection_names = [GENERAL_COLLECTION, PERSONAL_COLLECTION]

    for name in collection_names:
        if not client.collection_exists(name):
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )
