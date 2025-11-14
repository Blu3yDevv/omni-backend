# app/models/api.py

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    chat_history: Optional[List[ChatMessage]] = None
    settings: Optional[Dict[str, Any]] = None  # e.g. show_agent_breakdown, depth


class AgentBreakdown(BaseModel):
    plan: Dict[str, Any]
    research: Dict[str, Any]
    draft_answer: str
    tester_issues: List[str]
    tester_fixes: List[str]
    safety_flags: List[str]


class ChatResponse(BaseModel):
    session_id: Optional[str] = None
    answer: str
    agent_breakdown: Optional[AgentBreakdown] = None
    latency_ms: Optional[float] = None
