"""Auto-setup venv with benchmark dependencies."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).parent
ROOT_DIR = BENCH_DIR.parent
VENV_DIR = ROOT_DIR / ".venv"
VENV_PYTHON = VENV_DIR / "bin" / "python3"
REQUIREMENTS = BENCH_DIR / "requirements.txt"


def ensure_venv() -> str:
    """Create venv and install deps if needed. Returns path to venv python."""
    if VENV_PYTHON.exists():
        return str(VENV_PYTHON)

    print("\n  Setting up Python environment...")
    candidates = ["python3.13", "python3.12", "python3.11", "python3"]
    python_bin = "python3"
    for candidate in candidates:
        if shutil.which(candidate):
            python_bin = candidate
            break

    try:
        subprocess.run([python_bin, "-m", "venv", str(VENV_DIR)], check=True)
        pip = str(VENV_DIR / "bin" / "pip")
        subprocess.run([pip, "install", "--upgrade", "pip"], check=True)
        subprocess.run([pip, "install", "-r", str(REQUIREMENTS)], check=True)
        print("  Python environment ready.\n")
    except subprocess.CalledProcessError:
        print("Failed to set up Python environment.", file=sys.stderr)
        print(f"  Try: python3 -m venv .venv && .venv/bin/pip install -r benchmarks/requirements.txt",
              file=sys.stderr)
        sys.exit(1)

    return str(VENV_PYTHON)
