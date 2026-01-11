import argparse
import logging
import os
import time
import subprocess
import json
from typing import Any, Callable, Dict, List, Optional

import sanitizer

# --- CONFIGURATION ---
ADB_PATH = "adb"  # Ensure adb is in your PATH
OPENAI_MODEL_DEFAULT = os.environ.get("OPENAI_MODEL", "gpt-4o")  # Or "gpt-4-turbo" for faster/cheaper execution
SCREEN_DUMP_PATH = "/sdcard/window_dump.xml"
LOCAL_DUMP_PATH = "window_dump.xml"
UI_WAIT_SECONDS = 2
KEYEVENT_HOME = "KEYCODE_HOME"
KEYEVENT_BACK = "KEYCODE_BACK"
OLLAMA_MODEL_DEFAULT = os.environ.get("OLLAMA_MODEL", "llama3.1")
OLLAMA_HOST_DEFAULT = os.environ.get("OLLAMA_HOST")
LOG_FILE_DEFAULT = os.environ.get("ANDROID_ACTION_LOG_FILE", "android_action_kernel.log")
LOG_LEVEL_DEFAULT = os.environ.get("ANDROID_ACTION_LOG_LEVEL", "INFO")
MAX_LLM_RETRIES_DEFAULT = 1
OPENROUTER_BASE_URL_DEFAULT = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL_DEFAULT = os.environ.get("OPENROUTER_MODEL", "gpt-oss-120b")
OPENROUTER_API_KEY_DEFAULT = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_SITE_URL_DEFAULT = os.environ.get("OPENROUTER_SITE_URL")
OPENROUTER_SITE_NAME_DEFAULT = os.environ.get("OPENROUTER_SITE_NAME")
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
RETRY_PROMPT_TEMPLATE = (
    "Your previous response was invalid: {error}. "
    "Return ONLY one JSON object with exactly these keys: "
    "tap -> action, coordinates, reason; "
    "type -> action, text, reason; "
    "home/back/wait/done -> action, reason. "
    "No extra keys and no markdown."
)
LLM_PROVIDER_ALIASES = {
    "openai_api": "openai",
    "openai": "openai",
    "openrouter": "openrouter",
    "openrouter_api": "openrouter",
    "ollama": "ollama",
    "ollama_http": "ollama",
}

ActionPayload = Dict[str, Any]
Message = Dict[str, str]
ActionHandler = Callable[[ActionPayload], None]
RequestFn = Callable[[List[Message]], str]

LOGGER = logging.getLogger("android_action_kernel")

def resolve_log_level(level: str, default: int = logging.INFO) -> int:
    if not level:
        return default
    return getattr(logging, level.upper(), default)

def setup_logging(log_path: str, console_level: str = "INFO") -> logging.Logger:
    log_dir = os.path.dirname(os.path.abspath(log_path))
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logger = LOGGER
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    console_handler = logging.StreamHandler()
    console_handler.setLevel(resolve_log_level(console_level))
    console_formatter = logging.Formatter("%(levelname)s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s")
    console_handler.setFormatter(console_formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.debug("Logging initialized. console_level=%s log_file=%s", console_level, log_path)
    return logger

def format_command(command: List[str]) -> str:
    try:
        return subprocess.list2cmdline(command)
    except Exception:
        return " ".join(command)

def run_adb_command_result(command: List[str]) -> subprocess.CompletedProcess:
    """Executes a shell command via ADB and returns the completed process."""
    full_command = [ADB_PATH] + command
    LOGGER.debug("ADB command: %s", format_command(full_command))
    result = subprocess.run(full_command, capture_output=True, text=True)
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    LOGGER.debug("ADB return code: %s", result.returncode)
    if stdout:
        LOGGER.debug("ADB stdout: %s", stdout)
    if stderr:
        LOGGER.debug("ADB stderr: %s", stderr)
    if result.returncode != 0:
        LOGGER.error("ADB command failed: %s (code %s)", format_command(full_command), result.returncode)
    if stderr and "error" in stderr.lower():
        LOGGER.error("ADB error: %s", stderr)
    return result

def run_adb_command(command: List[str]) -> str:
    """Executes a shell command via ADB and returns stdout."""
    result = run_adb_command_result(command)
    return result.stdout.strip()

def ensure_adb_device_connected() -> List[str]:
    """Validates adb has at least one connected device in 'device' state."""
    result = run_adb_command_result(["devices"])
    if result.returncode != 0:
        message = result.stderr.strip() or "adb devices failed."
        raise RuntimeError(message)
    devices = []
    other_states = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("list of devices"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        serial, state = parts[0], parts[1]
        if state == "device":
            devices.append(serial)
        else:
            other_states.append(f"{serial} ({state})")
    if not devices:
        detail = ", ".join(other_states) if other_states else "none"
        raise RuntimeError(
            "No authorized ADB devices found. Connect a device or start an emulator. "
            f"Detected: {detail}."
        )
    LOGGER.info("ADB devices connected: %s", ", ".join(devices))
    return devices

def get_screen_state(
    screen_dump_path: str = SCREEN_DUMP_PATH,
    local_dump_path: str = LOCAL_DUMP_PATH,
) -> str:
    """Dumps the current UI XML and returns the sanitized JSON string."""
    # 1. Capture XML
    LOGGER.debug("Dumping UI to %s", screen_dump_path)
    if os.path.exists(local_dump_path):
        try:
            os.remove(local_dump_path)
        except OSError as exc:
            LOGGER.warning("Failed to remove stale UI dump %s: %s", local_dump_path, exc)
    result = run_adb_command_result(["shell", "uiautomator", "dump", screen_dump_path])
    if result.returncode != 0:
        stderr = result.stderr.strip() or "unknown error"
        raise RuntimeError(f"ADB uiautomator dump failed: {stderr}")

    # 2. Pull to local
    LOGGER.debug("Pulling UI dump to %s", local_dump_path)
    result = run_adb_command_result(["pull", screen_dump_path, local_dump_path])
    if result.returncode != 0:
        stderr = result.stderr.strip() or "unknown error"
        raise RuntimeError(f"ADB pull failed: {stderr}")

    # 3. Read & Sanitize
    if not os.path.exists(local_dump_path):
        raise RuntimeError(f"Could not capture screen. Missing {local_dump_path}")

    with open(local_dump_path, "r", encoding="utf-8") as f:
        xml_content = f.read()
        LOGGER.debug("UI dump size: %d bytes", len(xml_content))

    elements = sanitizer.get_interactive_elements(xml_content)
    LOGGER.debug("Interactive elements found: %d", len(elements))
    screen_context = json.dumps(elements, indent=2)
    LOGGER.debug("Screen context JSON:\n%s", screen_context)
    return screen_context

def _handle_tap(action: ActionPayload) -> None:
    coords = action.get("coordinates")
    if not isinstance(coords, list) or len(coords) != 2:
        LOGGER.warning("Invalid coordinates for tap action: %s", coords)
        return
    x, y = coords
    LOGGER.info("Tapping: (%s, %s)", x, y)
    run_adb_command(["shell", "input", "tap", str(x), str(y)])

def _handle_type(action: ActionPayload) -> None:
    raw_text = action.get("text")
    if not isinstance(raw_text, str):
        LOGGER.warning("Invalid text payload for type action: %s", raw_text)
        return
    text = raw_text.replace(" ", "%s")  # ADB requires %s for spaces
    LOGGER.info("Typing: %s", raw_text)
    LOGGER.debug("ADB type payload: %s", text)
    run_adb_command(["shell", "input", "text", text])

def _handle_home(action: ActionPayload) -> None:
    LOGGER.info("Going Home")
    run_adb_command(["shell", "input", "keyevent", KEYEVENT_HOME])

def _handle_back(action: ActionPayload) -> None:
    LOGGER.info("Going Back")
    run_adb_command(["shell", "input", "keyevent", KEYEVENT_BACK])

def _handle_wait(action: ActionPayload) -> None:
    LOGGER.info("Waiting...")
    time.sleep(UI_WAIT_SECONDS)

def _handle_done(action: ActionPayload) -> None:
    LOGGER.info("Goal achieved.")
    raise SystemExit(0)

ACTION_HANDLERS: Dict[str, ActionHandler] = {
    "tap": _handle_tap,
    "type": _handle_type,
    "home": _handle_home,
    "back": _handle_back,
    "wait": _handle_wait,
    "done": _handle_done,
}

def execute_action(action: ActionPayload):
    """Executes the action decided by the LLM."""
    act_type = action.get("action")
    LOGGER.info("Executing action: %s", action)
    handler = ACTION_HANDLERS.get(act_type)
    if not handler:
        LOGGER.warning("Unknown action type: %s", act_type)
        return
    handler(action)

def _try_parse_action_json(text: str) -> Optional[ActionPayload]:
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
    if isinstance(response, dict):
        return response
    if isinstance(response, str):
        return _try_parse_action_json(response.strip())
    return None

def _validate_action_payload(decision: ActionPayload) -> None:
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

def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    lines = stripped.splitlines()
    if len(lines) < 2:
        return stripped
    if not lines[0].startswith("```"):
        return stripped
    for i in range(len(lines) - 1, 0, -1):
        if lines[i].strip().startswith("```"):
            return "\n".join(lines[1:i]).strip()
    return stripped

def _build_retry_prompt(error: Exception) -> str:
    return RETRY_PROMPT_TEMPLATE.format(error=error)

def _build_openrouter_headers(site_url: Optional[str], site_name: Optional[str]) -> Dict[str, str]:
    headers = {}
    if site_url:
        headers["HTTP-Referer"] = site_url
    if site_name:
        headers["X-Title"] = site_name
    return headers

def _parse_action_from_text(raw_text: str, provider: str) -> ActionPayload:
    if not raw_text or not raw_text.strip():
        raise ValueError(f"{provider} response was empty.")
    cleaned = _strip_code_fences(raw_text)
    parsed = _try_parse_action_json(cleaned)
    if parsed is None and cleaned != raw_text:
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

def _request_with_retries(
    provider: str,
    messages: List[Message],
    max_retries: int,
    request_fn: RequestFn,
    log_context: Optional[Dict[str, Any]] = None,
    log_messages: bool = False,
) -> ActionPayload:
    last_error = None
    for attempt in range(max_retries + 1):
        _log_request_attempt(provider, attempt, max_retries, log_context)
        if log_messages:
            LOGGER.debug("%s messages=%s", provider, json.dumps(messages, indent=2))
        content = request_fn(messages)
        LOGGER.debug("%s response content: %s", provider, content)
        try:
            if not isinstance(content, str):
                raise ValueError(f"{provider} response did not include message content.")
            decision = _parse_action_from_text(content.strip(), provider)
        except ValueError as exc:
            last_error = exc
            LOGGER.warning(
                "%s response invalid (attempt %d/%d): %s",
                provider,
                attempt + 1,
                max_retries + 1,
                exc,
            )
            if attempt >= max_retries:
                break
            if isinstance(content, str):
                messages.append({"role": "assistant", "content": content})
            else:
                messages.append({"role": "assistant", "content": ""})
            messages.append({"role": "user", "content": _build_retry_prompt(exc)})
            continue
        LOGGER.debug("%s parsed decision: %s", provider, decision)
        return decision
    raise last_error or ValueError(f"{provider} response invalid.")

def _extract_ollama_content(response: Any) -> Optional[str]:
    if isinstance(response, dict):
        message = response.get("message") or {}
        return message.get("content")
    if isinstance(response, str):
        return response
    message = getattr(response, "message", None)
    if isinstance(message, dict):
        return message.get("content")
    return getattr(message, "content", None)

def _normalize_llm_provider(llm_provider: str) -> str:
    provider = LLM_PROVIDER_ALIASES.get(llm_provider)
    if not provider:
        raise ValueError(
            f"Unknown LLM provider '{llm_provider}'. Use 'openai_api', 'openrouter', or 'ollama'."
        )
    return provider

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
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    model = OPENAI_MODEL_DEFAULT
    messages = _build_messages(goal, screen_context)

    def _request(messages: List[Message]) -> str:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=messages,
        )
        return response.choices[0].message.content

    return _request_with_retries(
        "OpenAI",
        messages,
        max_retries,
        _request,
        log_context={"model": model},
        log_messages=True,
    )

def get_openrouter_decision(
    goal: str,
    screen_context: str,
    openrouter_model: Optional[str] = None,
    openrouter_base_url: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    openrouter_site_url: Optional[str] = None,
    openrouter_site_name: Optional[str] = None,
    max_retries: int = MAX_LLM_RETRIES_DEFAULT,
) -> ActionPayload:
    """Sends screen context to OpenRouter and asks for the next move."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI SDK not installed. Install it or use --llm-provider ollama."
        ) from exc
    api_key = openrouter_api_key or OPENROUTER_API_KEY_DEFAULT
    if not api_key:
        raise RuntimeError("OpenRouter API key not set. Set OPENROUTER_API_KEY.")
    base_url = openrouter_base_url or OPENROUTER_BASE_URL_DEFAULT
    model = openrouter_model or OPENROUTER_MODEL_DEFAULT
    headers = _build_openrouter_headers(
        openrouter_site_url or OPENROUTER_SITE_URL_DEFAULT,
        openrouter_site_name or OPENROUTER_SITE_NAME_DEFAULT,
    )
    client_kwargs = {"api_key": api_key, "base_url": base_url}
    if headers:
        client_kwargs["default_headers"] = headers
    client = OpenAI(**client_kwargs)
    messages = _build_messages(goal, screen_context)

    def _request(messages: List[Message]) -> str:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=messages,
        )
        return response.choices[0].message.content

    return _request_with_retries(
        "OpenRouter",
        messages,
        max_retries,
        _request,
        log_context={"model": model, "base_url": base_url},
    )

def get_ollama_decision(
    goal: str,
    screen_context: str,
    ollama_model: Optional[str] = None,
    ollama_host: Optional[str] = None,
    max_retries: int = MAX_LLM_RETRIES_DEFAULT,
) -> ActionPayload:
    """Sends screen context to Ollama (local HTTP) and asks for the next move."""
    try:
        import ollama
    except ImportError as exc:
        raise RuntimeError(
            "Ollama SDK not installed. Install it with `pip install -U ollama`."
        ) from exc
    model = ollama_model or OLLAMA_MODEL_DEFAULT
    if not model:
        raise ValueError("Ollama model is required. Set --ollama-model or OLLAMA_MODEL.")
    host = ollama_host or OLLAMA_HOST_DEFAULT
    messages = _build_messages(goal, screen_context)
    client = ollama.Client(host=host) if host else None
    use_format = True

    def _request(messages: List[Message]) -> str:
        nonlocal use_format
        try:
            if host:
                if use_format:
                    response = client.chat(
                        model=model,
                        messages=messages,
                        stream=False,
                        format="json",
                    )
                else:
                    response = client.chat(model=model, messages=messages, stream=False)
            else:
                if use_format:
                    response = ollama.chat(
                        model=model,
                        messages=messages,
                        stream=False,
                        format="json",
                    )
                else:
                    response = ollama.chat(model=model, messages=messages, stream=False)
        except TypeError:
            if not use_format:
                raise
            use_format = False
            if host:
                response = client.chat(model=model, messages=messages, stream=False)
            else:
                response = ollama.chat(model=model, messages=messages, stream=False)
        return _extract_ollama_content(response)

    return _request_with_retries(
        "Ollama",
        messages,
        max_retries,
        _request,
        log_context={"model": model, "host": host or "default"},
    )

def get_llm_decision(
    goal: str,
    screen_context: str,
    llm_provider: str,
    ollama_model: Optional[str] = None,
    ollama_host: Optional[str] = None,
) -> ActionPayload:
    """Sends screen context to the configured LLM provider and asks for the next move."""
    provider = _normalize_llm_provider(llm_provider)
    LOGGER.info("Using LLM provider: %s", provider)
    if provider == "openai":
        return get_openai_decision(goal, screen_context)
    if provider == "openrouter":
        return get_openrouter_decision(goal, screen_context)
    if provider == "ollama":
        return get_ollama_decision(goal, screen_context, ollama_model, ollama_host)
    raise ValueError(f"Unknown LLM provider '{llm_provider}'.")

def run_agent(
    goal: str,
    max_steps: int = 10,
    llm_provider: str = "openai_api",
    ollama_model: Optional[str] = None,
    ollama_host: Optional[str] = None,
):
    LOGGER.info("Android Use Agent Started. Goal: %s", goal)

    try:
        ensure_adb_device_connected()
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        return

    for step in range(max_steps):
        LOGGER.info("Step %d/%d", step + 1, max_steps)

        # 1. Perception
        LOGGER.info("Scanning screen...")
        try:
            screen_context = get_screen_state()
        except RuntimeError as exc:
            LOGGER.error("%s", exc)
            return

        # 2. Reasoning
        LOGGER.info("Thinking...")
        decision = get_llm_decision(
            goal,
            screen_context,
            llm_provider,
            ollama_model,
            ollama_host,
        )
        LOGGER.info("Decision: %s", decision.get("reason"))

        # 3. Action
        execute_action(decision)

        # Wait for UI to update
        LOGGER.debug("Waiting for UI to update...")
        time.sleep(UI_WAIT_SECONDS)

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Android UI automation agent.")
    parser.add_argument(
        "--llm-provider",
        default="openai_api",
        choices=["openai_api", "openai", "openrouter", "openrouter_api", "ollama", "ollama_http"],
        help="LLM backend to use.",
    )
    parser.add_argument(
        "--ollama-model",
        default=None,
        help="Ollama model name (defaults to OLLAMA_MODEL or 'llama3.1').",
    )
    parser.add_argument(
        "--ollama-host",
        default=None,
        help="Ollama host URL (defaults to OLLAMA_HOST).",
    )
    parser.add_argument(
        "--log-file",
        default=LOG_FILE_DEFAULT,
        help="Path to the log file (detailed debug logs).",
    )
    parser.add_argument(
        "--log-level",
        default=LOG_LEVEL_DEFAULT,
        help="Console log level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser

def main() -> int:
    args = build_arg_parser().parse_args()
    setup_logging(args.log_file, args.log_level)
    # Example Goal: "Open settings and turn on Wi-Fi"
    # Or your demo goal: "Find the 'Connect' button and tap it"
    goal = input("Enter your goal: ")
    LOGGER.info("User goal: %s", goal)
    run_agent(
        goal,
        llm_provider=args.llm_provider,
        ollama_model=args.ollama_model,
        ollama_host=args.ollama_host,
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
