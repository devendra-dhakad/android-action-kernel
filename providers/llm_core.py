import json
import logging
from typing import Any, Callable, Dict, List, Optional

LOGGER = logging.getLogger("android_action_kernel")

MAX_LLM_RETRIES_DEFAULT = 1
SYSTEM_PROMPT = """
You are an Android Driver Agent. Your job is to achieve the user's goal by navigating the UI.

You will receive:
1. The User's Goal.
2. A list of interactive UI elements (JSON) with their (x,y) center coordinates.

You must output ONLY a valid JSON object matching exactly one of the schemas below.
Do not add extra keys (for example, "target" or "contact").
Do not wrap the JSON in markdown or code fences.
If the target contact is not visible, tap a visible search bar or "New chat" button first.

Schemas (exact keys):
- {"action": "tap", "coordinates": [x, y], "reason": "Why you are tapping"}
- {"action": "type", "text": "Hello World", "reason": "Why you are typing"}
- {"action": "home", "reason": "Go to home screen"}
- {"action": "back", "reason": "Go back"}
- {"action": "wait", "reason": "Wait for loading"}
- {"action": "done", "reason": "Task complete"}

Example Output:
{"action": "tap", "coordinates": [540, 1200], "reason": "Clicking the 'Connect' button"}
"""

ALLOWED_ACTIONS = {"tap", "type", "home", "back", "wait", "done"}
ACTION_SCHEMA = {
    "tap": {"action", "coordinates", "reason"},
    "type": {"action", "text", "reason"},
    "home": {"action", "reason"},
    "back": {"action", "reason"},
    "wait": {"action", "reason"},
    "done": {"action", "reason"},
}

ActionPayload = Dict[str, Any]
Message = Dict[str, str]
RequestFn = Callable[[List[Message]], str]


def _try_parse_action_json(text: str) -> Optional[ActionPayload]:
    """Parse a JSON object from text, tolerating surrounding junk.

    Example:
        _try_parse_action_json('{"action": "tap", "coordinates": [1, 2], "reason": "ok"}')
    """
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    if isinstance(parsed, dict):
        return parsed
    return None


def _parse_wrapped_response(response: Any) -> Optional[ActionPayload]:
    """Parse a wrapped response that may be dict or JSON string.

    Example:
        _parse_wrapped_response('{"action": "wait", "reason": "loading"}')
    """
    if isinstance(response, dict):
        return response
    if isinstance(response, str):
        return _try_parse_action_json(response.strip())
    return None


def _validate_action_payload(decision: ActionPayload) -> None:
    """Validate an action payload against required schema and types.

    Example:
        _validate_action_payload({"action": "home", "reason": "reset state"})
    """
    action = decision.get("action")
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"Invalid action: {action!r}. Expected one of {sorted(ALLOWED_ACTIONS)}.")
    allowed_keys = ACTION_SCHEMA[action]
    decision_keys = set(decision.keys())
    missing_keys = allowed_keys - decision_keys
    if missing_keys:
        raise ValueError(
            f"Missing required keys for {action} action: {sorted(missing_keys)}."
        )
    extra_keys = decision_keys - allowed_keys
    if extra_keys:
        raise ValueError(
            f"Unexpected keys for {action} action: {sorted(extra_keys)}."
        )
    reason = decision.get("reason")
    if not isinstance(reason, str):
        raise ValueError("Invalid 'reason' for action; expected a string.")
    if action == "tap":
        coords = decision.get("coordinates")
        if (
            not isinstance(coords, list)
            or len(coords) != 2
            or not all(isinstance(value, (int, float)) for value in coords)
        ):
            raise ValueError("Invalid 'coordinates' for tap action; expected [x, y].")
    if action == "type":
        text = decision.get("text")
        if not isinstance(text, str):
            raise ValueError("Invalid 'text' for type action; expected a string.")

def _parse_action_from_text(raw_text: str, provider: str) -> ActionPayload:
    """Parse and validate an action payload from raw LLM text.

    Example:
        _parse_action_from_text('{"action": "tap", "coordinates": [612, 1684], "reason": "Tap on Poorna Pattem to call"}', "ollama")
    """
    if not raw_text or not raw_text.strip():
        raise ValueError(f"{provider} response was empty.")

    parsed = _try_parse_action_json(raw_text)
    if parsed is None:
        preview = raw_text.strip().replace("\n", " ")[:200]
        raise ValueError(f"{provider} did not return valid JSON. Preview: {preview!r}")
    if "action" not in parsed and "response" in parsed:
        parsed = _parse_wrapped_response(parsed.get("response"))
        if parsed is None:
            raise ValueError(f"{provider} response did not contain action JSON in 'response'.")
    _validate_action_payload(parsed)
    return parsed


def _build_messages(goal: str, screen_context: str) -> List[Message]:
    """Build system and user messages for the LLM request.

    Example:
        _build_messages("open phone app", "[]")
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"GOAL: {goal}\n\nSCREEN_CONTEXT:\n{screen_context}"},
    ]


def _log_request_attempt(
    provider: str,
    attempt: int,
    max_retries: int,
    log_context: Optional[Dict[str, Any]] = None,
) -> None:
    """Log a request attempt with optional context metadata.

    Example:
        _log_request_attempt("ollama", 0, 2, {"model": "gemma3:12b"})
    """
    if log_context:
        context = " ".join(f"{key}={value}" for key, value in log_context.items())
        LOGGER.debug(
            "%s request attempt %d/%d %s",
            provider,
            attempt + 1,
            max_retries + 1,
            context,
        )
    else:
        LOGGER.debug("%s request attempt %d/%d", provider, attempt + 1, max_retries + 1)