# app/agents/planner.py

from __future__ import annotations

from typing import Any, Dict, List

from app.agents.base import call_llm_json
from app.types import OmniState

Plan = Dict[str, Any]

PLANNER_SYSTEM_PROMPT = """
You are the Planner Agent for OmniAI (Omni Nano).

Your job:
- Read the user's latest message and short chat history.
- Decide how complex the request is.
- Decide whether we need external research / RAG or not.
- Break the work into clear goals and steps.
- Surface any important constraints or warnings.

You MUST respond ONLY with valid JSON in this schema:

{
  "complexity": "simple" | "normal" | "complex",
  "needs_research": true or false,
  "goals": [ "goal 1", "goal 2", ... ],
  "steps": [ "step 1", "step 2", ... ],
  "constraints": [ "constraint 1", "constraint 2", ... ]
}

Guidelines:
- Mark coding, system design, or multi-part reasoning as "normal" or "complex".
- Set "needs_research": true if external factual info, up-to-date knowledge, or
  long-term context are important.
- Otherwise, "needs_research": false for self-contained logic, coding patterns,
  or explanations that do not need the internet / RAG.
- Keep goals and steps short, clear, and high-signal.
- Constraints should include anything important: safety, missing info, ambiguity,
  time/compute limits, etc.
"""


def _format_chat_history(chat_history: List[Dict[str, Any]]) -> str:
    """
    Convert prior chat messages into a lightweight text block for the planner.
    """
    if not chat_history:
        return "No prior messages."

    lines: List[str] = []
    for msg in chat_history[-5:]:  # last few turns only
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if not content:
            continue
        lines.append(f"{role}: {content}")

    return "\n".join(lines)


def _build_planner_user_prompt(state: OmniState) -> str:
    """
    Build the 'user_prompt' sent to the planner LLM.
    """
    user_message = state.user_message or ""
    history_block = _format_chat_history(state.chat_history or [])

    prompt = f"""
User's latest message:
{user_message}

Recent chat history:
{history_block}

Your task:
- Analyse the request.
- Decide complexity.
- Decide if research is needed.
- Produce goals, steps, and constraints in the required JSON schema.
""".strip()

    return prompt


def planner_node(state: OmniState) -> OmniState:
    """
    LangGraph node: Planner.

    Inputs (from OmniState):
    - state.user_message: str
    - state.chat_history: list of prior messages

    Output:
    - state.plan: Plan dict with keys:
        - complexity
        - needs_research
        - goals
        - steps
        - constraints
    """
    user_prompt = _build_planner_user_prompt(state)

    # Call Omni Nano via the shared helper (using light-mode token budget).
    data = call_llm_json(
        system_prompt=PLANNER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_new_tokens=128,
        temperature=0.2,
    )

    if not isinstance(data, dict):
        # Fallback if the model didn't return proper JSON
        data = {"raw": data}

    plan: Plan = {
        "complexity": data.get("complexity", "normal"),
        "needs_research": bool(data.get("needs_research", True)),
        "goals": data.get("goals", []),
        "steps": data.get("steps", []),
        "constraints": data.get("constraints", []),
    }

    state.plan = plan
    return state
