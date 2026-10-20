"""
logger.py
---------
Dedicated logging module for the Stealth Speech-to-Text & LLM Fact-Checking Pipeline.
Writes clean, structured logs to a single 'vague.log' file located in the project root.

Captured Events:
  - STT: "speaker paused so appending data to buffer"
  - STT: "30sec over appending text to buffer"
  - LLM: "now the 20 seconds over so sending to llm"
  - LLM: "when the llm response comes log it in this file" -> "LLM response received"
  - System: Startup, shutdown, warnings, and errors.
"""

from __future__ import annotations

import os
import sys
import time
import datetime
import threading
from typing import List, Optional

# Lock for thread safety within the current process
_process_lock = threading.Lock()

# Custom log file override if set
_custom_log_path: Optional[str] = None
_console_echo: bool = os.environ.get("VAGUE_LOG_CONSOLE", "0") in ("1", "true", "True")


def configure(log_path: Optional[str] = None, console_echo: Optional[bool] = None) -> None:
    """Configure logger destination and options."""
    global _custom_log_path, _console_echo
    if log_path is not None:
        _custom_log_path = os.path.abspath(log_path)
    if console_echo is not None:
        _console_echo = console_echo


def get_log_path() -> str:
    """
    Return the single target 'vague.log' path in the project root.
    """
    if _custom_log_path:
        return _custom_log_path

    env_path = os.environ.get("VAGUE_LOG_FILE") or os.environ.get("VAGUE_LOG_PATH")
    if env_path:
        return os.path.abspath(env_path)

    # Project root directory (parent of whisper/ directory)
    module_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(module_dir) if os.path.basename(module_dir).lower() == "whisper" else module_dir
    return os.path.abspath(os.path.join(project_root, "vague.log"))


# Backward-compatible alias
def get_log_paths() -> List[str]:
    return [get_log_path()]


def _write_entry(prefix: str, message: str) -> None:
    """Internal helper to write formatted timestamped entry to the single vague.log file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    line = f"[{timestamp}] [{prefix}] {message}"

    if _console_echo:
        print(f"[vague.log] {line}", flush=True)

    log_file = get_log_path()
    with _process_lock:
        try:
            parent_dir = os.path.dirname(log_file)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
        except Exception as exc:
            sys.stderr.write(f"[logger.py] Failed to write log to {log_file}: {exc}\n")


# ---------------------------------------------------------------------------
# STT Pipeline Logging APIs
# ---------------------------------------------------------------------------
def log_stt_append(
    reason: str,
    speaker: str,
    text: str,
    duration_sec: Optional[float] = None,
) -> None:
    """
    Log when transcribed text is appended to the buffer.

    Parameters
    ----------
    reason : str
        Either "speaker paused" or "30sec over"
    speaker : str
        Speaker identifier, e.g. "SPEAKER_00", "UNKNOWN"
    text : str
        Transcribed utterance text
    duration_sec : float, optional
        Accumulated audio length in seconds before triggering
    """
    text_clean = text.strip().replace("\r", " ").replace("\n", " ")
    dur_str = f" | Audio: {duration_sec:.1f}s" if duration_sec is not None else ""

    if "pause" in reason.lower():
        msg = f"speaker paused so appending data to buffer | Speaker: {speaker}{dur_str} | Text: \"{text_clean}\""
    elif "30" in reason.lower() or "max" in reason.lower():
        msg = f"30sec over appending text to buffer | Speaker: {speaker}{dur_str} | Text: \"{text_clean}\""
    else:
        msg = f"{reason} -> appending text to buffer | Speaker: {speaker}{dur_str} | Text: \"{text_clean}\""

    _write_entry("STT", msg)


# ---------------------------------------------------------------------------
# LLM Pipeline Logging APIs
# ---------------------------------------------------------------------------
def log_llm_dispatch(
    interval_sec: float,
    speaker_block: str,
    speakers: Optional[List[str]] = None,
) -> None:
    """
    Log when the periodic timer fires and dispatches the buffer to the LLM.

    Parameters
    ----------
    interval_sec : float
        Interval duration, typically 20.0s
    speaker_block : str
        Formatted prompt with speaker-labelled utterances
    speakers : list of str, optional
        List of speaker names included in this dispatch
    """
    speakers_str = f" | Speakers: {', '.join(speakers)}" if speakers else ""
    int_str = f"{int(interval_sec)}" if (isinstance(interval_sec, (int, float)) and interval_sec == int(interval_sec)) else f"{interval_sec:.1f}"

    indented_block = "\n    " + speaker_block.replace("\n", "\n    ")
    msg = (
        f"now the {int_str} seconds over so sending to llm{speakers_str}\n"
        f"  --- Content Sent to LLM ---{indented_block}"
    )
    _write_entry("LLM", msg)


def log_llm_response(
    response: str,
    latency_sec: Optional[float] = None,
    speaker_block: Optional[str] = None,
) -> None:
    """
    Log when an LLM response is received.

    Parameters
    ----------
    response : str
        The textual response from LLM
    latency_sec : float, optional
        Time taken for LLM API call in seconds
    speaker_block : str, optional
        Input that was evaluated (for correlation if needed)
    """
    lat_str = f" | Latency: {latency_sec:.2f}s" if latency_sec is not None else ""
    indented_response = "\n    " + response.strip().replace("\n", "\n    ")
    msg = (
        f"LLM response received{lat_str}\n"
        f"  --- LLM Response ---{indented_response}"
    )
    _write_entry("LLM", msg)


# ---------------------------------------------------------------------------
# General Purpose System & Error Logging
# ---------------------------------------------------------------------------
def log_system_start(component: str, details: Optional[str] = None) -> None:
    """Log system startup."""
    det_str = f" ({details})" if details else ""
    _write_entry("SYSTEM", f"{component} started{det_str} | Log file: {get_log_path()}")


def log_system_stop(component: str, details: Optional[str] = None) -> None:
    """Log system shutdown."""
    det_str = f" ({details})" if details else ""
    _write_entry("SYSTEM", f"{component} stopped cleanly{det_str}")


def log_info(message: str) -> None:
    """Log informational message."""
    _write_entry("INFO", message)


def log_warning(message: str) -> None:
    """Log warning message."""
    _write_entry("WARNING", message)


def log_error(message: str, exc: Optional[Exception] = None) -> None:
    """Log error message with optional exception details."""
    exc_str = f" | Exception: {exc}" if exc else ""
    _write_entry("ERROR", f"{message}{exc_str}")
