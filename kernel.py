import argparse
import logging
import os
import time
import subprocess
import json
from typing import Callable, Dict, List, Optional

import sanitizer
from providers.llm import get_llm_decision
from providers.llm_core import ActionPayload

# --- CONFIGURATION ---
ADB_PATH = "adb"  # Ensure adb is in your PATH
SCREEN_DUMP_PATH = "/sdcard/window_dump.xml"
LOCAL_DUMP_PATH = "window_dump.xml"
UI_WAIT_SECONDS = 2
KEYEVENT_HOME = "KEYCODE_HOME"
KEYEVENT_BACK = "KEYCODE_BACK"
LOG_FILE_DEFAULT = os.environ.get("ANDROID_ACTION_LOG_FILE", "android_action_kernel.log")
LOG_LEVEL_DEFAULT = os.environ.get("ANDROID_ACTION_LOG_LEVEL", "INFO")
ActionHandler = Callable[[ActionPayload], None]

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

def run_agent(
    goal: str,
    max_steps: int = 10,
    llm_provider: str = "openai",
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
        default="openai",
        choices=["openai", "openrouter", "ollama", "gemini"],
        help="LLM backend to use.",
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
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
