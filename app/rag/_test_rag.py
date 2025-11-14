# app/rag/_test_rag.py

import os
from datetime import datetime, timezone
from uuid import uuid4

from .ingest import Document, upsert_general_docs
from .rag_pipeline import run_rag
from .qdrant_client import ensure_collections_exist

"""
Usage (from omni-backend root):

    export QDRANT_URL="https://....cloud.qdrant.io:6333"
    export QDRANT_API_KEY="your-key"
    python -m app.rag._test_rag
"""


def main() -> None:
    ensure_collections_exist()

    # 1) Ingest a couple of test docs
    now_iso = datetime.now(timezone.utc).isoformat()

    docs = [
        Document(
            id=str(uuid4()),  # VALID UUID
            text=(
                "OmniAI is a multi-agent chatbot system using Llama 3.1 8B "
                "as a base model and QLoRA fine-tuning."
            ),
            metadata={
                "source": "omni-docs",
                "tags": ["omni", "architecture"],
                "created_at": now_iso,
            },
        ),
        Document(
            id=str(uuid4()),  # VALID UUID
            text=(
                "Qdrant is a vector database used to store embeddings "
                "for semantic search and RAG."
            ),
            metadata={
                "source": "qdrant-intro",
                "tags": ["qdrant", "database"],
                "created_at": now_iso,
            },
        ),
    ]

    inserted = upsert_general_docs(docs)
    print(f"Upserted {inserted} docs.")

    # 2) Run a test query
    query = "What is Qdrant and how does OmniAI use it?"
    rag_result = run_rag(query=query, plan=None, top_k=5)

    print("\n=== RESEARCH SUMMARY ===")
    print(rag_result["research_summary"])
    print("\n=== SOURCES ===")
    for src in rag_result["sources"]:
        print(src)


if __name__ == "__main__":
    main()
