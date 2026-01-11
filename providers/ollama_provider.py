import logging
from typing import Any, List, Optional

from providers.constants import PROVIDER_CONFIGS
import ollama
from providers.llm_core import (
    ActionPayload,
    MAX_LLM_RETRIES_DEFAULT,
    Message,
    _build_messages,
    _parse_action_from_text,
)

LOGGER = logging.getLogger("android_action_kernel")


def _extract_ollama_content(response: Any) -> Optional[str]:
    """Extract message content from varied Ollama response shapes.

    Example:
        content = _extract_ollama_content({"message": {"content": "hi"}})
    """
    if isinstance(response, dict):
        message = response.get("message") or {}
        return message.get("content")
    if isinstance(response, str):
        return response
    message = getattr(response, "message", None)
    if isinstance(message, dict):
        return message.get("content")
    return getattr(message, "content", None)


def get_ollama_decision(
    goal: str,
    screen_context: str,
    ollama_model: Optional[str] = None,
    max_retries: int = MAX_LLM_RETRIES_DEFAULT,
) -> ActionPayload:
    """Sends screen context to Ollama (local HTTP) and asks for the next move.

    Example:
        decision = get_ollama_decision("open phone", "screen json here", "gemma3:12b")
    """
    provider_config = PROVIDER_CONFIGS["ollama"]
    model = ollama_model or provider_config.get("model")
    if not model:
        raise ValueError("Ollama model is required. Set --ollama-model or OLLAMA_MODEL.")
    messages = _build_messages(goal, screen_context)
    try:
        response = ollama.chat(
            model=model,
            messages=messages,
            stream=False,
            format="json",
        )
    except TypeError as exc:
        raise RuntimeError(f"Ollama chat format error: {exc}") from exc

    LOGGER.debug("ollama raw response: %s", response)

    content = _extract_ollama_content(response)
    LOGGER.debug("ollama response content: %s", content)
    if content is None:
        raise ValueError("Ollama response did not include message content.")
    return _parse_action_from_text(content, "ollama")
