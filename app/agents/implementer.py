# app/agents/implementer.py

from __future__ import annotations

from typing import Any, Dict, List

from app.agents.base import call_llm_text
from app.types import OmniState

IMPLEMENTER_SYSTEM_PROMPT = """
You are the Implementer Agent for OmniAI (Omni Nano).

Your job:
- Take the user's request, the Planner's plan, and any research summary.
- Produce a first DRAFT ANSWER for the user.
- The draft should be:
    - Clear and structured.
    - Honest about uncertainty.
    - Focused on being actually useful.
- Leave room for the Tester and Finalizer to refine it later.

You are NOT the final step; this is just a solid first draft.

Return ONLY the draft answer text. Do NOT output JSON.
"""


def _build_implementer_user_prompt(state: OmniState) -> str:
    """
    Build the user_prompt for the Implementer LLM.

    Includes:
    - user_message
    - plan (complexity, goals, steps, constraints)
    - research summary
    """
    user_message: str = state.user_message or ""
    plan: Dict[str, Any] = state.plan or {}
    research: Dict[str, Any] = state.research or {}

    plan_str = repr(plan) if plan else "No explicit plan provided."
    research_summary = (
        research.get("summary", "") if isinstance(research, dict) else ""
    )
    research_sources = (
        research.get("sources", []) if isinstance(research, dict) else []
    )

    prompt = f"""
User's original request:
{user_message}

Planner Agent plan:
{plan_str}

Research summary (if any):
{research_summary}

Research sources (if any):
{research_sources}

Your task:
- Use the plan and research (if available) to write a helpful, structured draft answer.
- This is NOT the final answer; it's a first pass that will be reviewed by a Tester and Finalizer.
- Be direct and clear, but don't over-apologize or ramble.
- If you are missing important info, state that clearly and suggest how the user could clarify.
""".strip()

    return prompt


def implementer_node(state: OmniState) -> OmniState:
    """
    Lang-style node: Implementer.

    Inputs:
    - state.user_message
    - state.plan
    - state.research

    Output:
    - state.draft_answer
    """
    user_prompt = _build_implementer_user_prompt(state)

    draft = call_llm_text(
        system_prompt=IMPLEMENTER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_new_tokens=128,  # still light-mode; llm_client further hard-caps to 32
        temperature=0.3,
    )

    state.draft_answer = draft.strip()
    return state
