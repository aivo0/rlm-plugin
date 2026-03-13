"""rlm_runtime — helper library for RLM agent scripts.

Provides context loading, metadata, LLM sub-queries (sync and async),
and per-session state persistence.
"""

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

# ── Per-session state directory ──────────────────────────────────────────────

_STATE_DIR = Path(tempfile.gettempdir()) / f"rlm_state_{os.getpid()}"

PREVIEW_LINES = 20


# ── Context helpers ──────────────────────────────────────────────────────────

def load_context(path: str) -> str:
    """Read a file into a string."""
    return Path(path).expanduser().read_text()


def context_meta(text: str) -> str:
    """Return a human-readable summary: char count, line count, head/tail preview."""
    lines = text.split("\n")
    char_count = len(text)
    line_count = len(lines)

    preview_start = "\n".join(lines[:PREVIEW_LINES])
    preview_end = "\n".join(lines[-PREVIEW_LINES:])

    return "\n".join([
        "Context statistics:",
        f"  - {char_count:,} characters",
        f"  - {line_count:,} lines",
        "",
        f"First {PREVIEW_LINES} lines:",
        preview_start,
        "",
        f"Last {PREVIEW_LINES} lines:",
        preview_end,
    ])


# ── LLM sub-queries ─────────────────────────────────────────────────────────

def llm_query(chunk: str, instruction: str) -> str:
    """Run a synchronous sub-query via claude -p. Returns stdout."""
    prompt = f"{instruction}\n\n---\n\n{chunk}"
    result = subprocess.run(
        ["claude", "-p"],
        input=prompt,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude -p failed (rc={result.returncode}): {result.stderr}")
    return result.stdout.strip()


async def async_llm_query(chunk: str, instruction: str) -> str:
    """Run an async sub-query via claude -p. Use with asyncio.gather() for parallelism."""
    prompt = f"{instruction}\n\n---\n\n{chunk}"
    proc = await asyncio.create_subprocess_exec(
        "claude", "-p",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate(input=prompt.encode())
    if proc.returncode != 0:
        raise RuntimeError(f"claude -p failed (rc={proc.returncode}): {stderr.decode()}")
    return stdout.decode().strip()


# ── State persistence ────────────────────────────────────────────────────────

def save_state(key: str, value) -> None:
    """Persist a value as JSON in the per-session temp directory."""
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    (_STATE_DIR / f"{key}.json").write_text(json.dumps(value))


def load_state(key: str, default=None):
    """Load a previously saved value. Returns default if not found."""
    path = _STATE_DIR / f"{key}.json"
    if not path.exists():
        return default
    return json.loads(path.read_text())


def cleanup_state() -> None:
    """Remove the per-session temp directory."""
    if _STATE_DIR.exists():
        shutil.rmtree(_STATE_DIR)
