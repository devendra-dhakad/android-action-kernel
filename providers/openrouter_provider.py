import json
import logging
from typing import Optional

from providers.constants import PROVIDER_CONFIGS
from providers.llm_core import (
    ActionPayload,
    MAX_LLM_RETRIES_DEFAULT,
    SYSTEM_PROMPT,
)

LOGGER = logging.getLogger("android_action_kernel")


def get_openrouter_decision(
    goal: str,
    screen_context: str,
    openrouter_model: Optional[str] = None,
    openrouter_base_url: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    max_retries: int = MAX_LLM_RETRIES_DEFAULT,
) -> ActionPayload:
    """Sends screen context to OpenRouter and asks for the next move."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI SDK not installed. Install it or use --llm-provider ollama."
        ) from exc
    provider_config = PROVIDER_CONFIGS["openrouter"]
    api_key = openrouter_api_key or provider_config.get("api_key")
    if not api_key:
        raise RuntimeError("OpenRouter API key not set. Set OPENROUTER_API_KEY.")
    base_url = openrouter_base_url or provider_config.get("base_url")
    model = openrouter_model or provider_config.get("model")
    client_kwargs = {"api_key": api_key, "base_url": base_url}
    client = OpenAI(**client_kwargs)
    system_prompt = SYSTEM_PROMPT

    request_kwargs = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"GOAL: {goal}\n\nSCREEN_CONTEXT:\n{screen_context}"},
        ],
    }
    response = client.chat.completions.create(**request_kwargs)

    content = response.choices[0].message.content
    LOGGER.debug("openrouter response content: %s", content)
    return json.loads(content)
