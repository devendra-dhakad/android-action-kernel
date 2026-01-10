import argparse
import logging
import os
import time
import subprocess
import json
import shlex
import shutil
from typing import Dict, Any, List, Optional
import sanitizer

# --- CONFIGURATION ---
ADB_PATH = "adb"  # Ensure adb is in your PATH
MODEL = "gpt-4o"  # Or "gpt-4-turbo" for faster/cheaper execution
SCREEN_DUMP_PATH = "/sdcard/window_dump.xml"
LOCAL_DUMP_PATH = "window_dump.xml"
GEMINI_CLI_COMMAND = os.environ.get("GEMINI_CLI_COMMAND")
GEMINI_CLI_ARGS = os.environ.get("GEMINI_CLI_ARGS", "")
OLLAMA_MODEL_DEFAULT = os.environ.get("OLLAMA_MODEL", "llama3.1")
OLLAMA_HOST_DEFAULT = os.environ.get("OLLAMA_HOST")
LOG_FILE_DEFAULT = os.environ.get("ANDROID_ACTION_LOG_FILE", "android_action_kernel.log")
LOG_LEVEL_DEFAULT = os.environ.get("ANDROID_ACTION_LOG_LEVEL", "INFO")

SYSTEM_PROMPT = """
You are an Android Driver Agent. Your job is to achieve the user's goal by navigating the UI.

You will receive:
1. The User's Goal.
2. A list of interactive UI elements (JSON) with their (x,y) center coordinates.

You must output ONLY a valid JSON object with your next action.

Available Actions:
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

def get_screen_state() -> str:
    """Dumps the current UI XML and returns the sanitized JSON string."""
    # 1. Capture XML
    LOGGER.debug("Dumping UI to %s", SCREEN_DUMP_PATH)
    if os.path.exists(LOCAL_DUMP_PATH):
        try:
            os.remove(LOCAL_DUMP_PATH)
        except OSError as exc:
            LOGGER.warning("Failed to remove stale UI dump %s: %s", LOCAL_DUMP_PATH, exc)
    result = run_adb_command_result(["shell", "uiautomator", "dump", SCREEN_DUMP_PATH])
    if result.returncode != 0:
        stderr = result.stderr.strip() or "unknown error"
        raise RuntimeError(f"ADB uiautomator dump failed: {stderr}")

    # 2. Pull to local
    LOGGER.debug("Pulling UI dump to %s", LOCAL_DUMP_PATH)
    result = run_adb_command_result(["pull", SCREEN_DUMP_PATH, LOCAL_DUMP_PATH])
    if result.returncode != 0:
        stderr = result.stderr.strip() or "unknown error"
        raise RuntimeError(f"ADB pull failed: {stderr}")

    # 3. Read & Sanitize
    if not os.path.exists(LOCAL_DUMP_PATH):
        raise RuntimeError(f"Could not capture screen. Missing {LOCAL_DUMP_PATH}")

    with open(LOCAL_DUMP_PATH, "r", encoding="utf-8") as f:
        xml_content = f.read()
        LOGGER.debug("UI dump size: %d bytes", len(xml_content))

    elements = sanitizer.get_interactive_elements(xml_content)
    LOGGER.debug("Interactive elements found: %d", len(elements))
    screen_context = json.dumps(elements, indent=2)
    LOGGER.debug("Screen context JSON:\n%s", screen_context)
    return screen_context

def execute_action(action: Dict[str, Any]):
    """Executes the action decided by the LLM."""
    act_type = action.get("action")
    LOGGER.info("Executing action: %s", action)

    if act_type == "tap":
        x, y = action.get("coordinates")
        LOGGER.info("Tapping: (%s, %s)", x, y)
        run_adb_command(["shell", "input", "tap", str(x), str(y)])

    elif act_type == "type":
        text = action.get("text").replace(" ", "%s") # ADB requires %s for spaces
        LOGGER.info("Typing: %s", action.get("text"))
        LOGGER.debug("ADB type payload: %s", text)
        run_adb_command(["shell", "input", "text", text])

    elif act_type == "home":
        LOGGER.info("Going Home")
        run_adb_command(["shell", "input", "keyevent", "KEYWORDS_HOME"])

    elif act_type == "back":
        LOGGER.info("Going Back")
        run_adb_command(["shell", "input", "keyevent", "KEYWORDS_BACK"])

    elif act_type == "wait":
        LOGGER.info("Waiting...")
        time.sleep(2)

    elif act_type == "done":
        LOGGER.info("Goal achieved.")
        exit(0)
    else:
        LOGGER.warning("Unknown action type: %s", act_type)

def build_prompt(goal: str, screen_context: str) -> str:
    return f"{SYSTEM_PROMPT.strip()}\n\nGOAL: {goal}\n\nSCREEN_CONTEXT:\n{screen_context}"

def ensure_executable(command: str) -> str:
    if os.path.dirname(command):
        if os.path.exists(command):
            return command
        raise FileNotFoundError(f"Gemini CLI not found at '{command}'.")
    resolved = shutil.which(command)
    if resolved:
        return resolved
    raise FileNotFoundError(f"Gemini CLI '{command}' not found in PATH.")

def resolve_gemini_command(gemini_cli_command: Optional[str]) -> str:
    if gemini_cli_command:
        return ensure_executable(gemini_cli_command)
    if GEMINI_CLI_COMMAND:
        return ensure_executable(GEMINI_CLI_COMMAND)
    for candidate in ("gemini.cmd", "gemini"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise FileNotFoundError("Gemini CLI not found in PATH.")

def resolve_gemini_args(gemini_cli_args: Optional[str]) -> List[str]:
    raw_args = gemini_cli_args if gemini_cli_args is not None else GEMINI_CLI_ARGS
    if not raw_args:
        return []
    return shlex.split(raw_args, posix=os.name != "nt")

def _try_parse_action_json(text: str) -> Optional[Dict[str, Any]]:
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

def _parse_wrapped_response(response: Any) -> Optional[Dict[str, Any]]:
    if isinstance(response, dict):
        return response
    if isinstance(response, str):
        return _try_parse_action_json(response.strip())
    return None

def _validate_action_payload(decision: Dict[str, Any]) -> None:
    action = decision.get("action")
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"Invalid action: {action!r}. Expected one of {sorted(ALLOWED_ACTIONS)}.")
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

def _parse_action_from_text(raw_text: str, provider: str) -> Dict[str, Any]:
    parsed = _try_parse_action_json(raw_text)
    if parsed is None:
        raise ValueError(f"{provider} did not return valid JSON.")
    if "action" not in parsed and "response" in parsed:
        parsed = _parse_wrapped_response(parsed.get("response"))
        if parsed is None:
            raise ValueError(f"{provider} response did not contain action JSON in 'response'.")
    _validate_action_payload(parsed)
    return parsed

def get_openai_decision(goal: str, screen_context: str) -> Dict[str, Any]:
    """Sends screen context to OpenAI and asks for the next move."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI SDK not installed. Install it or use --llm-provider gemini_cli."
        ) from exc

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"GOAL: {goal}\n\nSCREEN_CONTEXT:\n{screen_context}"},
    ]
    LOGGER.debug("OpenAI request model=%s messages=%s", MODEL, json.dumps(messages, indent=2))
    response = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        messages=messages,
    )

    content = response.choices[0].message.content
    LOGGER.debug("OpenAI response content: %s", content)
    try:
        decision = json.loads(content)
    except json.JSONDecodeError:
        LOGGER.error("OpenAI returned invalid JSON: %s", content)
        raise
    LOGGER.debug("OpenAI parsed decision: %s", decision)
    return decision

def get_gemini_cli_decision(
    goal: str,
    screen_context: str,
    gemini_cli_command: Optional[str] = None,
    gemini_cli_args: Optional[str] = None,
    gemini_cli_prompt_arg: Optional[str] = None,
    gemini_cli_prompt_positional: bool = False,
) -> Dict[str, Any]:
    """Sends screen context to Gemini CLI and asks for the next move."""
    try:
        command = resolve_gemini_command(gemini_cli_command)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Gemini CLI executable not found. Set --gemini-cli-command or add it to PATH."
        ) from exc
    args = resolve_gemini_args(gemini_cli_args)
    command = [command] + args
    prompt = build_prompt(goal, screen_context)
    stdin_prompt = None
    if gemini_cli_prompt_arg:
        command = command + [gemini_cli_prompt_arg, prompt]
    elif gemini_cli_prompt_positional:
        command = command + [prompt]
    else:
        stdin_prompt = prompt
    LOGGER.debug("Gemini CLI command: %s", format_command(command))
    if stdin_prompt is not None:
        LOGGER.debug("Gemini CLI prompt (stdin):\n%s", stdin_prompt)
    else:
        LOGGER.debug("Gemini CLI prompt (args): %s", prompt)
    result = subprocess.run(command, input=stdin_prompt, capture_output=True, text=True)
    LOGGER.debug("Gemini CLI return code: %s", result.returncode)
    if result.stdout:
        LOGGER.debug("Gemini CLI stdout: %s", result.stdout.strip())
    if result.stderr:
        LOGGER.debug("Gemini CLI stderr: %s", result.stderr.strip())
    if result.returncode != 0:
        LOGGER.error("Gemini CLI failed with exit code %s", result.returncode)
        raise RuntimeError(
            f"Gemini CLI failed with exit code {result.returncode}: {result.stderr.strip()}"
        )
    output = result.stdout.strip()
    try:
        decision = json.loads(output)
    except json.JSONDecodeError as exc:
        stdout_preview = output[:400]
        stderr_preview = (result.stderr or "").strip()[:400]
        details = []
        if stdout_preview:
            details.append(f"stdout: {stdout_preview!r}")
        if stderr_preview:
            details.append(f"stderr: {stderr_preview!r}")
        detail_text = " " + " ".join(details) if details else " (stdout was empty)"
        LOGGER.error("Gemini CLI did not return valid JSON.%s", detail_text)
        raise ValueError(
            "Gemini CLI did not return valid JSON. Ensure the CLI is configured for JSON-only output."
            + detail_text
        ) from exc
    if not isinstance(decision, dict):
        LOGGER.error("Gemini CLI JSON root is not an object: %s", type(decision).__name__)
        raise ValueError("Gemini CLI did not return a JSON object.")
    if "action" not in decision and "response" in decision:
        LOGGER.warning("Gemini CLI returned a wrapper response; parsing action JSON from 'response'.")
        parsed = _parse_wrapped_response(decision.get("response"))
        if parsed is None:
            response_preview = str(decision.get("response"))[:400]
            LOGGER.error("Gemini CLI response did not contain action JSON. response=%r", response_preview)
            raise ValueError("Gemini CLI response did not contain action JSON in 'response'.")
        decision = parsed
    try:
        _validate_action_payload(decision)
    except ValueError as exc:
        LOGGER.error("Gemini CLI action validation failed: %s", exc)
        raise
    LOGGER.debug("Gemini CLI parsed decision: %s", decision)
    return decision

def get_ollama_decision(
    goal: str,
    screen_context: str,
    ollama_model: Optional[str] = None,
    ollama_host: Optional[str] = None,
) -> Dict[str, Any]:
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
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"GOAL: {goal}\n\nSCREEN_CONTEXT:\n{screen_context}"},
    ]
    LOGGER.debug("Ollama request model=%s host=%s", model, host or "default")
    if host:
        client = ollama.Client(host=host)
        response = client.chat(model=model, messages=messages, stream=False)
    else:
        response = ollama.chat(model=model, messages=messages, stream=False)
    content = None
    if isinstance(response, dict):
        message = response.get("message") or {}
        content = message.get("content")
    elif isinstance(response, str):
        content = response
    else:
        message = getattr(response, "message", None)
        if isinstance(message, dict):
            content = message.get("content")
        else:
            content = getattr(message, "content", None)
    if not isinstance(content, str):
        LOGGER.error("Ollama response missing message content: %r", response)
        raise ValueError("Ollama response did not include message content.")
    LOGGER.debug("Ollama response content: %s", content)
    decision = _parse_action_from_text(content.strip(), "Ollama")
    LOGGER.debug("Ollama parsed decision: %s", decision)
    return decision

def get_llm_decision(
    goal: str,
    screen_context: str,
    llm_provider: str,
    gemini_cli_command: Optional[str] = None,
    gemini_cli_args: Optional[str] = None,
    gemini_cli_prompt_arg: Optional[str] = None,
    gemini_cli_prompt_positional: bool = False,
    ollama_model: Optional[str] = None,
    ollama_host: Optional[str] = None,
) -> Dict[str, Any]:
    """Sends screen context to the configured LLM provider and asks for the next move."""
    LOGGER.info("Using LLM provider: %s", llm_provider)
    if llm_provider in {"openai_api", "openai"}:
        return get_openai_decision(goal, screen_context)
    if llm_provider in {"gemini_cli", "gemini"}:
        return get_gemini_cli_decision(
            goal,
            screen_context,
            gemini_cli_command,
            gemini_cli_args,
            gemini_cli_prompt_arg,
            gemini_cli_prompt_positional,
        )
    if llm_provider in {"ollama", "ollama_http"}:
        return get_ollama_decision(goal, screen_context, ollama_model, ollama_host)
    raise ValueError(
        f"Unknown LLM provider '{llm_provider}'. Use 'openai_api', 'gemini_cli', or 'ollama'."
    )

def run_agent(
    goal: str,
    max_steps=10,
    llm_provider: str = "openai_api",
    gemini_cli_command: Optional[str] = None,
    gemini_cli_args: Optional[str] = None,
    gemini_cli_prompt_arg: Optional[str] = None,
    gemini_cli_prompt_positional: bool = False,
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
            gemini_cli_command,
            gemini_cli_args,
            gemini_cli_prompt_arg,
            gemini_cli_prompt_positional,
            ollama_model,
            ollama_host,
        )
        LOGGER.info("Decision: %s", decision.get("reason"))

        # 3. Action
        execute_action(decision)

        # Wait for UI to update
        LOGGER.debug("Waiting for UI to update...")
        time.sleep(2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Android UI automation agent.")
    parser.add_argument(
        "--llm-provider",
        default="openai_api",
        choices=["openai_api", "openai", "gemini_cli", "gemini", "ollama", "ollama_http"],
        help="LLM backend to use.",
    )
    parser.add_argument(
        "--gemini-cli-command",
        default=None,
        help="Gemini CLI executable path or name (auto-detects if omitted).",
    )
    parser.add_argument(
        "--gemini-cli-args",
        default=None,
        help="Extra args to pass to Gemini CLI.",
    )
    parser.add_argument(
        "--gemini-cli-prompt-arg",
        default=None,
        help="Flag to pass the prompt as an argument (e.g. --prompt or -p).",
    )
    parser.add_argument(
        "--gemini-cli-prompt-positional",
        action="store_true",
        help="Pass the prompt as a positional argument instead of stdin.",
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
    args = parser.parse_args()
    setup_logging(args.log_file, args.log_level)
    # Example Goal: "Open settings and turn on Wi-Fi"
    # Or your demo goal: "Find the 'Connect' button and tap it"
    GOAL = input("Enter your goal: ")
    LOGGER.info("User goal: %s", GOAL)
    run_agent(
        GOAL,
        llm_provider=args.llm_provider,
        gemini_cli_command=args.gemini_cli_command,
        gemini_cli_args=args.gemini_cli_args,
        gemini_cli_prompt_arg=args.gemini_cli_prompt_arg,
        gemini_cli_prompt_positional=args.gemini_cli_prompt_positional,
        ollama_model=args.ollama_model,
        ollama_host=args.ollama_host,
    )
