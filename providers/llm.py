import logging
from typing import Optional

from providers.llm_core import ActionPayload
from providers.ollama_provider import get_ollama_decision
from providers.openai_provider import get_openai_decision
from providers.openrouter_provider import get_openrouter_decision

LOGGER = logging.getLogger("android_action_kernel")


def get_llm_decision(
    goal: str,
    screen_context: str,
    llm_provider: str,
) -> ActionPayload:
    """Sends screen context to the configured LLM provider and asks for the next move."""
    if llm_provider not in ("openai", "openrouter", "ollama"):
        raise ValueError(
            f"Unknown LLM provider '{llm_provider}'. Use 'openai', 'openrouter', or 'ollama'."
        )
    LOGGER.info("Using LLM provider: %s", llm_provider)
    if llm_provider == "openai":
        return get_openai_decision(goal, screen_context)
    if llm_provider == "openrouter":
        return get_openrouter_decision(goal, screen_context)
    if llm_provider == "ollama":
        return get_ollama_decision(goal, screen_context)
    raise ValueError(f"Unknown LLM provider '{llm_provider}'.")
