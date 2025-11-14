# app/agents/base.py

from __future__ import annotations

from typing import Any, Dict, List

from app.services.llm_client import (
    generate_chat_completion,
    generate_structured_json,
)

# Global light-mode default for free hardware
DEFAULT_MAX_TOKENS = 128
DEFAULT_TEMPERATURE = 0.3


def _build_messages(system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()},
    ]


def call_llm_text(
    system_prompt: str,
    user_prompt: str,
    *,
    max_new_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    """
    Call Omni Nano and return a plain text completion.

    All agents that just need text should use this.
    """
    messages = _build_messages(system_prompt, user_prompt)
    text = generate_chat_completion(
        messages=messages,
        temperature=temperature,
        max_new_tokens=max_new_tokens,  # mapped inside llm_client
    )
    return text.strip()


def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    *,
    max_new_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Dict[str, Any]:
    """
    Call Omni Nano expecting a JSON-like response.

    - We still pass messages like in call_llm_text().
    - generate_structured_json() will:
        - try json.loads(...)
        - or fall back to {"raw": "..."} if the model didn't return valid JSON.
    """
    messages = _build_messages(system_prompt, user_prompt)
    data = generate_structured_json(
        messages=messages,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    return data
