# app/agents/tester.py

from __future__ import annotations

from typing import Any, Dict, List

from app.agents.base import call_llm_json
from app.types import OmniState

TesterReview = Dict[str, Any]

TESTER_SYSTEM_PROMPT = """
You are the Tester Agent for OmniAI (Omni Nano).

Your job:
- Critically review a draft answer produced by the Implementer Agent.
- Identify problems with clarity, correctness, structure, safety, and usefulness.
- Suggest concrete fixes and improvements.
- Flag any potential safety issues or policy violations.

You MUST respond ONLY with valid JSON in this schema:

{
  "issues": [
    "issue 1",
    "issue 2",
    ...
  ],
  "fixes": [
    "fix 1",
    "fix 2",
    ...
  ],
  "safety_flags": [
    "flag 1",
    "flag 2",
    ...
  ]
}

Guidelines:
- Keep "issues" focused and specific (e.g. "too verbose", "missing step X").
- "fixes" should be actionable suggestions (e.g. "shorten intro", "add warning about limitations").
- Use "safety_flags" for anything that might be harmful, misleading, or needs a disclaimer.
- If the draft is mostly fine, you can have an empty "issues" list and a few small "fixes".
"""


def _build_tester_user_prompt(state: OmniState) -> str:
    """
    Build the user_prompt for the tester LLM.

    Includes:
    - user_message
    - plan (if any)
    - research summary (if any)
    - draft_answer
    """
    user_message = state.user_message or ""
    plan = state.plan or {}
    research = state.research or {}
    draft = state.draft_answer or ""

    plan_str = "" if not plan else repr(plan)
    research_summary = research.get("summary", "") if isinstance(research, dict) else ""
    research_sources = research.get("sources", []) if isinstance(research, dict) else []

    prompt = f"""
User's original request:
{user_message}

High-level plan (if any):
{plan_str}

Research summary (if any):
{research_summary}

Research sources (if any):
{research_sources}

Draft answer from the Implementer Agent:
{draft}

Your task:
- Review the draft answer given the request, plan, and research.
- Find issues.
- Suggest fixes.
- Flag safety issues.
- Output JSON ONLY in the required schema.
""".strip()

    return prompt


def tester_node(state: OmniState) -> OmniState:
    """
    LangGraph node: Tester.

    Inputs (from OmniState):
    - state.user_message
    - state.plan
    - state.research
    - state.draft_answer

    Outputs:
    - state.tester_issues
    - state.tester_fixes
    - state.safety_flags
    """
    user_prompt = _build_tester_user_prompt(state)

    data: TesterReview = call_llm_json(
        system_prompt=TESTER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_new_tokens=128,
        temperature=0.2,
    )

    if not isinstance(data, dict):
        data = {"raw": data}

    issues = data.get("issues", [])
    fixes = data.get("fixes", [])
    safety_flags = data.get("safety_flags", [])

    # Normalize types
    if not isinstance(issues, list):
        issues = [str(issues)]
    if not isinstance(fixes, list):
        fixes = [str(fixes)]
    if not isinstance(safety_flags, list):
        safety_flags = [str(safety_flags)]

    state.tester_issues = [str(i) for i in issues]
    state.tester_fixes = [str(f) for f in fixes]
    state.safety_flags = [str(s) for s in safety_flags]

    return state
