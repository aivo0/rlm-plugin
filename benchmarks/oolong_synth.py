#!/usr/bin/env python3
"""
Oolong Synth Benchmark — oolongbench/oolong-synth

Synthetic long-context tasks: timeline ordering, user tracking, counting.
Compares direct LLM vs RLM on the same query.

Usage:
    python3 benchmarks/oolong_synth.py [--idx 4743]

Python deps are auto-installed into .venv on first run.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Auto-setup venv
sys.path.insert(0, str(Path(__file__).parent))
from _venv import ensure_venv

VENV_PYTHON = ensure_venv()

# Load dataset via venv Python (which has 'datasets' installed)
parser = argparse.ArgumentParser()
parser.add_argument("--idx", type=int, default=4743)
args = parser.parse_args()

print(f"\n  Loading dataset (idx={args.idx})...")

load_script = f"""
import json, sys
from datasets import load_dataset
ds = load_dataset("oolongbench/oolong-synth", split="test")
idx = {args.idx}
if idx < 0 or idx >= len(ds):
    print(f"Index {{idx}} out of range (dataset has {{len(ds)}} entries, use 0-{{len(ds)-1}})", file=sys.stderr)
    sys.exit(1)
example = ds[idx]
print(json.dumps({{
    "context": example["context_window_text_with_labels"],
    "question": example["question"],
    "answer": example["answer"],
    "task_group": example.get("task_group", "unknown"),
}}))
"""

try:
    output = subprocess.run(
        [VENV_PYTHON, "-"],
        input=load_script, capture_output=True, text=True,
        timeout=120,
    )
    if output.returncode != 0:
        raise RuntimeError(output.stderr)
    example = json.loads(output.stdout.strip())
except Exception as e:
    print(f"Failed to load dataset: {e}", file=sys.stderr)
    sys.exit(1)

print(f"  task={example['task_group']}, context={len(example['context']) / 1024:.1f}KB")

# Run benchmark
sys.path.insert(0, str(Path(__file__).parent.parent))
from benchmarks._rlm_bench import run_benchmark

run_benchmark(
    benchmark_name="Oolong Synth",
    dataset_label="oolongbench/oolong-synth",
    idx=args.idx,
    context=example["context"],
    question=example["question"],
    expected=example["answer"],
)
