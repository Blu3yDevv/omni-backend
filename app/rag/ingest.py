# app/rag/ingest.py

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from qdrant_client.models import PointStruct

from .embeddings import embed_texts
from .qdrant_client import get_qdrant_client, GENERAL_COLLECTION, PERSONAL_COLLECTION


@dataclass
class Document:
    """
    A single document to be stored in Qdrant.

    - id: unique ID for the point. Must ultimately be either:
        * an integer, or
        * a string representing a valid UUID.

      If you pass a non-numeric, non-UUID string, we will convert it to a
      deterministic UUID (UUID5) under the hood.
    - text: the raw text content
    - metadata: free-form JSON metadata (source, tags, created_at, etc.)
    """
    id: Union[str, int]
    text: str
    metadata: Dict[str, Any]


def _normalize_point_id(raw_id: Union[str, int]) -> Union[str, int]:
    """
    Ensure the point ID is acceptable for Qdrant:
    - if it's an int -> use as-is
    - if it's a numeric string -> cast to int
    - if it's a UUID string -> return normalized UUID string
    - otherwise -> derive a deterministic UUID5 from the string
    """
    if isinstance(raw_id, int):
        return raw_id

    s = str(raw_id)

    # Numeric string: treat as integer ID
    if s.isdigit():
        return int(s)

    # Try to parse as UUID string
    try:
        u = uuid.UUID(s)
        return str(u)
    except ValueError:
        # Deterministic UUID5 based on the string content
        return str(uuid.uuid5(uuid.NAMESPACE_URL, s))


def _build_points(docs: List[Document]) -> List[PointStruct]:
    texts = [d.text for d in docs]
    vectors = embed_texts(texts)

    points: List[PointStruct] = []
    for doc, vec in zip(docs, vectors):
        payload = {
            "text": doc.text,
            "metadata": doc.metadata,
        }

        point_id = _normalize_point_id(doc.id)

        points.append(
            PointStruct(
                id=point_id,
                vector=vec,
                payload=payload,
            )
        )
    return points


def upsert_documents(
    docs: List[Document],
    collection_name: Optional[str] = None,
) -> int:
    """
    Upsert a batch of documents into a Qdrant collection.

    Returns the number of upserted docs.
    """
    if not docs:
        return 0

    client = get_qdrant_client()
    collection = collection_name or GENERAL_COLLECTION

    points = _build_points(docs)
    client.upsert(
        collection_name=collection,
        points=points,
        wait=True,
    )

    return len(docs)


def upsert_general_docs(docs: List[Document]) -> int:
    return upsert_documents(docs, collection_name=GENERAL_COLLECTION)


def upsert_personal_knowledge(docs: List[Document]) -> int:
    return upsert_documents(docs, collection_name=PERSONAL_COLLECTION)
