# app/agents/researcher.py

from __future__ import annotations

from app.types import OmniState


def researcher_node(state: OmniState) -> OmniState:
    """
    TEMP DEBUG VERSION:
    Bypass real RAG and just attach a simple message.
    Later, we will re-enable Qdrant-based RAG here.
    """
    state.research = {
        "summary": "RAG is disabled in debug mode.",
        "sources": [],
    }
    return state
