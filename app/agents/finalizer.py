# app/agents/finalizer.py

from __future__ import annotations

from typing import Any, Dict, List

from app.agents.base import call_llm_text
from app.types import OmniState

FINALIZER_SYSTEM_PROMPT = """
You are the Finalizer Agent for OmniAI (Omni Nano).

Your job:
- Take the user's request, a draft answer, and feedback from the Tester Agent.
- Produce a clear, structured, final answer in the OmniAI style:
    - Direct and honest.
    - Helpful and practical.
    - Not cringe, not overly formal, not rude.
- Apply the tester's fixes and address any issues they found.
- If there are safety flags, include brief disclaimers or adjustments to keep the answer safe.

You MUST return ONLY the final answer text. Do NOT output JSON.
"""


def _build_finalizer_user_prompt(state: OmniState) -> str:
    """
    Build the user_prompt for the finalizer LLM.

    Includes:
    - user_message
    - draft_answer
    - tester_issues
    - tester_fixes
    - safety_flags
    """
    user_message = state.user_message or ""
    draft = state.draft_answer or ""
    issues = state.tester_issues or []
    fixes = state.tester_fixes or []
    safety_flags = state.safety_flags or []

    prompt = f"""
User's original request:
{user_message}

Draft answer from the Implementer Agent:
{draft}

Tester Agent issues (if any):
{issues}

Tester Agent suggested fixes (if any):
{fixes}

Safety flags (if any):
{safety_flags}

Your task:
- Produce the best possible final answer for the user.
- Apply useful fixes and address issues.
- If safety flags exist, adjust the answer and/or add a brief disclaimer.
- Respond with ONLY the final answer text, no JSON, no additional meta commentary.
""".strip()

    return prompt


def finalizer_node(state: OmniState) -> OmniState:
    """
    LangGraph node: Finalizer.

    Inputs (from OmniState):
    - state.user_message
    - state.draft_answer
    - state.tester_issues
    - state.tester_fixes
    - state.safety_flags

    Output:
    - state.final_answer
    """
    user_prompt = _build_finalizer_user_prompt(state)

    final_text = call_llm_text(
        system_prompt=FINALIZER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_new_tokens=128,
        temperature=0.3,
    )

    state.final_answer = final_text.strip()
    return state
