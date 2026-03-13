"""Shared ANSI display helpers for benchmark scripts."""

import re
import sys
import threading
import time

# ANSI codes
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
CLEAR_LINE = "\033[2K\r"

try:
    import shutil
    BOX_W = min(shutil.get_terminal_size().columns, 96) - 6
except Exception:
    BOX_W = 74

MAX_CONTENT_W = BOX_W - 4

SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


def wrap_text(text: str, max_width: int) -> list[str]:
    if len(text) <= max_width:
        return [text]
    return [text[i:i + max_width] for i in range(0, len(text), max_width)]


def strip_ansi(text: str) -> str:
    return re.sub(r"\033\[[0-9;]*m", "", text)


def box_top(title: str, color: str) -> str:
    inner = BOX_W - 2
    t = f" {title} "
    right = max(0, inner - len(strip_ansi(t)))
    return f"    {color}╭{t}{'─' * right}╮{RESET}"


def box_bottom(color: str) -> str:
    return f"    {color}╰{'─' * (BOX_W - 2)}╯{RESET}"


def box_line(text: str, color: str) -> str:
    stripped = strip_ansi(text)
    pad = max(0, MAX_CONTENT_W - len(stripped))
    return f"    {color}│{RESET} {text}{' ' * pad} {color}│{RESET}"


def display_box(title: str, content: str, color: str) -> None:
    print(box_top(title, color))
    for line in content.split("\n"):
        for chunk in wrap_text(line, MAX_CONTENT_W):
            print(box_line(f"{color}{chunk}{RESET}", color))
    print(box_bottom(color))


def display_bar(title: str, subtitle: str, color: str) -> None:
    bar = "─" * (BOX_W - 2)
    print(f"  {color}{bar}{RESET}")
    print(f"  {color}{BOLD} {title}{RESET} {DIM}{subtitle}{RESET}")
    print(f"  {color}{bar}{RESET}")


def format_size(chars: int) -> str:
    return f"{chars / 1000:.1f}K" if chars >= 1000 else str(chars)


class Spinner:
    def __init__(self):
        self._thread = None
        self._stop_event = threading.Event()
        self._message = ""
        self._start_time = 0.0

    def start(self, message: str) -> None:
        self.stop()
        self._message = message
        self._start_time = time.time()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread:
            self._stop_event.set()
            self._thread.join()
            self._thread = None
            sys.stdout.write(CLEAR_LINE)
            sys.stdout.flush()

    def _run(self) -> None:
        idx = 0
        while not self._stop_event.is_set():
            frame = SPINNER_FRAMES[idx % len(SPINNER_FRAMES)]
            elapsed = f"{time.time() - self._start_time:.1f}"
            sys.stdout.write(
                f"{CLEAR_LINE}    {CYAN}{frame}{RESET} {self._message} {DIM}{elapsed}s{RESET}"
            )
            sys.stdout.flush()
            idx += 1
            self._stop_event.wait(0.08)
