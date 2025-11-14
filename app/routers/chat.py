# app/routers/chat.py

from __future__ import annotations

import time
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from app.graph.workflow import run_omni_graph
from app.models.api import ChatRequest, ChatResponse, AgentBreakdown, ChatMessage
from app.types import OmniState

router = APIRouter()


def _convert_history_to_internal(history: List[ChatMessage] | None) -> List[Dict[str, Any]]:
    if not history:
        return []
    return [msg.model_dump() for msg in history]


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    """
    Main OmniAI chat endpoint.
    """
    if not payload.message or not payload.message.strip():
        raise HTTPException(status_code=400, detail="Message must not be empty.")

    user_message = payload.message.strip()
    chat_history = _convert_history_to_internal(payload.chat_history)

    t0 = time.time()
    try:
        state: OmniState = run_omni_graph(
            user_message=user_message,
            chat_history=chat_history,
        )
    except Exception as e:
        # You can add more structured logging here later
        raise HTTPException(status_code=500, detail=f"Internal error in OmniAI pipeline: {e}")

    latency_ms = (time.time() - t0) * 1000.0

    # Build agent breakdown if caller wants it (for now, always include it)
    breakdown = AgentBreakdown(
        plan=state.plan or {},
        research=state.research or {},
        draft_answer=state.draft_answer or "",
        tester_issues=state.tester_issues or [],
        tester_fixes=state.tester_fixes or [],
        safety_flags=state.safety_flags or [],
    )

    return ChatResponse(
        session_id=payload.session_id,
        answer=state.final_answer or state.draft_answer or "",
        agent_breakdown=breakdown,
        latency_ms=latency_ms,
    )
