# app/rag/rag_pipeline.py

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from qdrant_client.models import Filter  # for future filters
from qdrant_client import models as qmodels

from .embeddings import embed_texts
from .qdrant_client import (
    get_qdrant_client,
    GENERAL_COLLECTION,
    PERSONAL_COLLECTION,
)


class RetrievedSource(TypedDict, total=False):
    id: str
    collection: str
    score: float
    text_preview: str
    metadata: Dict[str, Any]


class RagResult(TypedDict):
    research_summary: str
    sources: List[RetrievedSource]
    raw_context: str


def search_collection(
    collection_name: str,
    query_vector: List[float],
    limit: int = 5,
    qfilter: Optional[Filter] = None,
) -> List[qmodels.ScoredPoint]:
    """
    Low-level wrapper around Qdrant search.
    """
    client = get_qdrant_client()

    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        query_filter=qfilter,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return results


def run_rag(
    query: str,
    plan: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
    include_personal: bool = True,
) -> RagResult:
    """
    High-level RAG helper used by the Researcher agent later.

    - Embeds the query.
    - Searches general_docs (+ personal_knowledge if enabled).
    - Builds a simple research_summary and structured sources list.

    `plan` is currently unused, but later you can:
      - read plan["domains"] or plan["constraints"] to build Qdrant filters.
    """
    # 1) Embed query
    query_vec = embed_texts([query])[0]

    # 2) Query collections
    hits_general = search_collection(GENERAL_COLLECTION, query_vec, limit=top_k)

    hits_personal: List[qmodels.ScoredPoint] = []
    if include_personal:
        hits_personal = search_collection(PERSONAL_COLLECTION, query_vec, limit=top_k)

    # 3) Combine results and sort by score (Qdrant returns higher score = closer)
    all_hits: List[tuple[str, qmodels.ScoredPoint]] = [
        (GENERAL_COLLECTION, h) for h in hits_general
    ] + [
        (PERSONAL_COLLECTION, h) for h in hits_personal
    ]

    all_hits.sort(key=lambda t: (t[1].score or 0.0), reverse=True)

    # 4) Build sources + raw_context
    sources: List[RetrievedSource] = []
    context_chunks: List[str] = []

    for idx, (collection_name, hit) in enumerate(all_hits, start=1):
        payload = hit.payload or {}

        text = payload.get("text") or ""
        metadata = payload.get("metadata") or {}

        preview = text[:200].replace("\n", " ").strip()

        sources.append(
            RetrievedSource(
                id=str(hit.id),
                collection=collection_name,
                score=float(hit.score or 0.0),
                text_preview=preview,
                metadata=metadata,
            )
        )

        # Tag each chunk with an index so the LLM can reference it
        context_chunks.append(f"[{idx}] {text}")

    raw_context = "\n\n".join(context_chunks)

    # 5) Lightweight "summary" – this is intentionally simple.
    #    The Researcher / Implementer agents will do deeper summarization
    #    by reading raw_context + sources with the LLM.
    if not context_chunks:
        research_summary = "No relevant documents were retrieved from the knowledge base."
    else:
        research_summary = (
            "Retrieved the following context snippets from the knowledge base:\n\n"
            + "\n\n".join(context_chunks[:3])  # first few chunks only
        )

    return RagResult(
        research_summary=research_summary,
        sources=sources,
        raw_context=raw_context,
    )
