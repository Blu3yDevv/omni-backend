# app/types.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class OmniState:
    """
    Shared state for the OmniAI LangGraph workflow.

    This is the object that flows between nodes (planner, researcher,
    implementer, tester, finalizer).
    """

    # Core conversation
    user_message: str = ""
    chat_history: List[Dict[str, Any]] = field(default_factory=list)

    # Planner output
    plan: Dict[str, Any] = field(default_factory=dict)

    # Researcher output (RAG)
    research: Dict[str, Any] = field(default_factory=dict)

    # Implementer output
    draft_answer: str = ""

    # Tester output
    tester_issues: List[str] = field(default_factory=list)
    tester_fixes: List[str] = field(default_factory=list)

    # Safety / guardrails
    safety_flags: List[str] = field(default_factory=list)

    # Finalizer output
    final_answer: str = ""

    # Optional metadata
    session_id: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)
