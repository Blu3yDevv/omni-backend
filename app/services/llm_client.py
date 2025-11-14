# app/services/llm_client.py

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import httpx
from gradio_client import Client


class LLMClientError(Exception):
    """Custom exception for LLM client errors."""
    pass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Hugging Face Space ID for your Gradio app
LLM_SPACE_ID = os.getenv("LLM_SPACE_ID", "Blu3yDevv/omni-nano-inference")

# Optional HF token if the Space is private (not strictly needed if Space is public)
HF_API_TOKEN = (
    os.getenv("HF_API_TOKEN")
    or os.getenv("HF_TOKEN")
    or os.getenv("HF_API_KEY")
)

# Basic retry config
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_TIMEOUT_SEC = float(os.getenv("LLM_TIMEOUT_SEC", "60"))

# Hard cap to keep calls cheap on free hardware
HARD_MAX_NEW_TOKENS = 32


# Singleton client instance
_gradio_client: Optional[Client] = None


def _get_client() -> Client:
    """
    Lazily initialize and return a gradio_client.Client for the Space.
    """
    global _gradio_client
    if _gradio_client is not None:
        return _gradio_client

    client_kwargs: Dict[str, Any] = {}
    if HF_API_TOKEN:
        client_kwargs["hf_token"] = HF_API_TOKEN

    # Example: Client("username/space-name", hf_token="...")
    _gradio_client = Client(LLM_SPACE_ID, **client_kwargs)
    return _gradio_client


# ---------------------------------------------------------------------------
# Core call used by agents
# ---------------------------------------------------------------------------

def generate_chat_completion(
    messages: List[Dict[str, Any]],
    temperature: float = 0.3,
    max_new_tokens: int = 512,
    max_tokens: Optional[int] = None,
    **_: Any,
) -> str:
    """
    Call the Omni Nano Gradio Space and return a plain text completion.

    NOTE:
    - We hard-cap the token budget to HARD_MAX_NEW_TOKENS (currently 32)
      to keep requests fast on the free HF hardware.
    - `max_new_tokens` and `max_tokens` are accepted for compatibility but ignored.
    """
    client = _get_client()

    # Resolve effective token cap (ignore larger values)
    effective_max = HARD_MAX_NEW_TOKENS

    # Our Space expects:
    #   omni_chat(messages_json: str, max_new_tokens: int, temperature: float) -> str
    #
    # and is wired in Gradio as:
    #   client.predict(messages_json, max_new_tokens, temperature, api_name="/predict")
    #
    messages_json = json.dumps({"messages": messages})

    last_err: Optional[Exception] = None

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            result = client.predict(
                messages_json,
                int(effective_max),
                float(temperature),
                api_name="/predict",
            )

            if not isinstance(result, str):
                result = str(result)

            return result.strip()

        except httpx.RequestError as e:
            last_err = e
            print(f"[LLM] HTTP error calling Space (attempt {attempt}/{LLM_MAX_RETRIES}): {e}")
        except Exception as e:
            last_err = e
            print(f"[LLM] Error calling Space (attempt {attempt}/{LLM_MAX_RETRIES}): {e}")

        time.sleep(1.0 * attempt)

    raise LLMClientError(
        f"Failed to get completion from LLM Space after {LLM_MAX_RETRIES} attempts: {last_err}"
    )


def generate_structured_json(
    messages: List[Dict[str, Any]],
    temperature: float = 0.2,
    max_new_tokens: int = 512,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    For nodes that want the model to return JSON.

    It calls generate_chat_completion() (hard-capped tokens) and then json.loads().
    """
    raw = generate_chat_completion(
        messages=messages,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        max_tokens=max_tokens,
    )

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # If the model didn't return valid JSON, wrap raw text in a dict
        return {"raw": raw}

