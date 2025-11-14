# app/rag/embeddings.py

import os
from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer

# Default to MiniLM; can be overridden via env
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Lazily load and cache the sentence-transformers model.

    Model: all-MiniLM-L6-v2 (384-dim embeddings).
    """
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of strings into dense vectors.
    Returns a list of lists (plain Python floats) for easy JSON serialization.
    """
    if not texts:
        return []

    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,  # recommended for cosine similarity
    )
    return embeddings.tolist()
