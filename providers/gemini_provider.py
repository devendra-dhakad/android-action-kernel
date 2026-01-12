import logging
from typing import Optional

from providers.constants import PROVIDER_CONFIGS
from providers.llm_core import (
    ActionPayload,
    MAX_LLM_RETRIES_DEFAULT,
    SYSTEM_PROMPT,
    _parse_action_from_text,
)

LOGGER = logging.getLogger("android_action_kernel")


def get_gemini_decision(
    goal: str,
    screen_context: str,
    gemini_model: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    max_retries: int = MAX_LLM_RETRIES_DEFAULT,
) -> ActionPayload:
    """Sends screen context to Google Gemini (AI Studio) and asks for the next move."""
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "Google GenAI SDK not installed. Install it via `pip install google-genai`."
        ) from exc

    provider_config = PROVIDER_CONFIGS["gemini"]
    api_key = gemini_api_key or provider_config.get("api_key")
    if not api_key:
        raise RuntimeError("Gemini API key not set. Set GEMINI_API_KEY.")

    client = genai.Client(api_key=api_key)

    model_name = gemini_model or provider_config.get("model")

    prompt = f"GOAL: {goal}\n\nSCREEN_CONTEXT:\n{screen_context}"

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )
        content = response.text
    except Exception as exc:
        raise RuntimeError(f"Gemini API request failed: {exc}") from exc

    LOGGER.debug("gemini response content: %s", content)
    return _parse_action_from_text(content, "gemini")
