# app/graph/state.py

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class OmniState(TypedDict, total=False):
    """
    Shared state flowing through the LangGraph workflow.

    Each node reads this and returns partial updates.
    """

    # Input / context
    user_message: str
    chat_history: List[Dict[str, Any]]  # list of {role, content}

    # Planner output
    plan: Dict[str, Any]

    # Researcher output
    research: Dict[str, Any]

    # Implementer output
    draft_answer: str

    # Tester output
    tester_issues: str
    tester_fixes: str

    # Guardrails (to be fleshed out in Section 7)
    safety_flags: Dict[str, Any]

    # Finalizer output
    final_answer: str
