import json
import logging

from providers.constants import PROVIDER_CONFIGS
from providers.llm_core import (
    ActionPayload,
    MAX_LLM_RETRIES_DEFAULT,
    SYSTEM_PROMPT,
)

LOGGER = logging.getLogger("android_action_kernel")


def get_openai_decision(
    goal: str,
    screen_context: str,
    max_retries: int = MAX_LLM_RETRIES_DEFAULT,
) -> ActionPayload:
    """Sends screen context to OpenAI and asks for the next move."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI SDK not installed. Install it or use --llm-provider ollama."
        ) from exc
    provider_config = PROVIDER_CONFIGS["openai"]
    client = OpenAI(api_key=provider_config.get("api_key"))
    model = provider_config["model"]
    system_prompt = SYSTEM_PROMPT

    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"GOAL: {goal}\n\nSCREEN_CONTEXT:\n{screen_context}"},
        ],
    )

    content = response.choices[0].message.content
    LOGGER.debug("openai response content: %s", content)
    return json.loads(content)
