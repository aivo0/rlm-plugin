#!/usr/bin/env python3
"""
LongBench NarrativeQA Benchmark — THUDM/LongBench (narrativeqa split)

Reading comprehension over long narratives.
Compares direct LLM vs RLM on the same query.

Usage:
    python3 benchmarks/longbench_narrativeqa.py [--idx 182]

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
parser.add_argument("--idx", type=int, default=182)
args = parser.parse_args()

print(f"\n  Loading dataset (idx={args.idx})...")

load_script = f"""
import json, sys
from datasets import load_dataset
ds = load_dataset("THUDM/LongBench", "narrativeqa", split="test", trust_remote_code=True)
idx = {args.idx}
if idx < 0 or idx >= len(ds):
    print(f"Index {{idx}} out of range (dataset has {{len(ds)}} entries, use 0-{{len(ds)-1}})", file=sys.stderr)
    sys.exit(1)
example = ds[idx]
print(json.dumps({{
    "input": example["input"],
    "context": example["context"],
    "answers": example["answers"],
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

print(f"  context={len(example['context']) / 1024:.1f}KB")

# Run benchmark
sys.path.insert(0, str(Path(__file__).parent.parent))
from benchmarks._rlm_bench import run_benchmark

run_benchmark(
    benchmark_name="LongBench NarrativeQA",
    dataset_label="THUDM/LongBench (narrativeqa)",
    idx=args.idx,
    context=example["context"],
    question=example["input"],
    expected=example["answers"],
)
