# app/graph/workflow.py

from __future__ import annotations

from typing import List, Dict, Any

from app.types import OmniState
from app.agents.planner import planner_node
from app.agents.researcher import researcher_node
from app.agents.implementer import implementer_node
from app.agents.tester import tester_node
from app.agents.finalizer import finalizer_node


def run_omni_graph(user_message: str, chat_history: List[Dict[str, Any]]) -> OmniState:
    state = OmniState(
        user_message=user_message,
        chat_history=chat_history or [],
    )

    # 1) Planner
    state = planner_node(state)
    complexity = state.plan.get("complexity", "normal")
    needs_research = bool(state.plan.get("needs_research", False))

    # 2) Researcher (stubbed or real)
    if needs_research:
        state = researcher_node(state)
    else:
        state.research = {
            "summary": "Planner decided no external research is needed.",
            "sources": [],
        }

    # 3) Implementer
    state = implementer_node(state)

    # For now, if complexity is simple, skip tester/finalizer
    if complexity == "simple":
        state.final_answer = state.draft_answer
        return state

    # 4) Tester
    state = tester_node(state)

    # 5) Finalizer
    state = finalizer_node(state)

    return state
