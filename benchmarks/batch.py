#!/usr/bin/env python3
"""
Batch Benchmark Runner — runs multiple indices and computes aggregate scores.

Usage:
    # LongBench NarrativeQA — 10 random samples
    python3 benchmarks/batch.py longbench -n 10

    # Oolong Synth — 10 random samples
    python3 benchmarks/batch.py oolong -n 10

    # Specific indices
    python3 benchmarks/batch.py longbench --indices 5,42,99,182

    # Full dataset (all indices)
    python3 benchmarks/batch.py oolong --all

    # Quiet mode (no per-example display, just summary)
    python3 benchmarks/batch.py longbench -n 20 -q

Python deps are auto-installed into .venv on first run.
"""

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from _venv import ensure_venv
from _display import (
    BOLD, BLUE, CYAN, DIM, GREEN, MAGENTA, RED, RESET, YELLOW,
    BOX_W, display_box, wrap_text,
)

VENV_PYTHON = ensure_venv()


# ── Dataset loaders ─────────────────────────────────────────────────────────

def load_dataset_examples(
    dataset: str,
    n: int | None = None,
    indices: list[int] | None = None,
    all_examples: bool = False,
    seed: int = 42,
) -> dict:
    """Load dataset examples in a single subprocess call.

    Handles sampling internally so we only load the dataset once.
    Returns {"size": int, "indices": list[int], "examples": list[dict]}.
    """
    if dataset == "longbench":
        ds_load = 'load_dataset("THUDM/LongBench", "narrativeqa", split="test", trust_remote_code=True)'
        append_block = (
            '    results.append({"idx": idx, "context": ex["context"],'
            ' "question": ex["input"], "expected": ex["answers"]})'
        )
    else:
        ds_load = 'load_dataset("oolongbench/oolong-synth", split="test")'
        append_block = (
            '    results.append({"idx": idx,'
            ' "context": ex["context_window_text_with_labels"],'
            ' "question": ex["question"], "expected": ex["answer"],'
            ' "task_group": ex.get("task_group", "unknown")})'
        )

    indices_json = json.dumps(indices) if indices else "None"

    script = (
        "import json, sys, random\n"
        "from datasets import load_dataset\n"
        "print('Loading dataset...', file=sys.stderr, flush=True)\n"
        f"ds = {ds_load}\n"
        "size = len(ds)\n"
        "print(f'Dataset loaded: {size} examples', file=sys.stderr, flush=True)\n"
        f"indices = {indices_json}\n"
        "if indices is None:\n"
        f"    if {all_examples!r}:\n"
        "        indices = list(range(size))\n"
        "    else:\n"
        f"        rng = random.Random({seed})\n"
        f"        n = min({n or 10}, size)\n"
        "        indices = sorted(rng.sample(range(size), n))\n"
        "results = []\n"
        "for idx in indices:\n"
        "    if idx < 0 or idx >= size:\n"
        "        print(f'WARN: index {idx} out of range (0-{size-1})', file=sys.stderr)\n"
        "        continue\n"
        "    ex = ds[idx]\n"
        f"    {append_block.lstrip()}\n"
        "print(json.dumps({'size': size, 'indices': indices, 'examples': results}))\n"
    )
    result = subprocess.run(
        [VENV_PYTHON, "-"], input=script,
        capture_output=True, text=True, timeout=600,
    )
    # Show any stderr progress/warnings
    if result.stderr:
        for line in result.stderr.strip().split("\n"):
            if line.strip():
                print(f"  {DIM}{line.strip()}{RESET}")
    if result.returncode != 0:
        print(f"{RED}Failed to load dataset{RESET}", file=sys.stderr)
        sys.exit(1)
    return json.loads(result.stdout.strip())


# ── Summary display ─────────────────────────────────────────────────────────

def print_summary(benchmark_name: str, results: list[dict]) -> dict:
    """Print aggregate score summary. Returns summary dict."""
    n = len(results)
    if n == 0:
        print(f"  {RED}No results to summarize.{RESET}")
        return {}

    # Aggregate scores
    direct_f1s = [r["directLlm"]["scores"]["f1"] for r in results]
    direct_cs = [r["directLlm"]["scores"]["contains"] for r in results]
    direct_judges = [r["directLlm"]["scores"]["judge"] for r in results
                     if r["directLlm"]["scores"].get("judge", -1) >= 0]
    direct_times = [r["directLlm"]["elapsedS"] for r in results]

    rlm_f1s = [r["rlm"]["scores"]["f1"] for r in results]
    rlm_cs = [r["rlm"]["scores"]["contains"] for r in results]
    rlm_judges = [r["rlm"]["scores"]["judge"] for r in results
                  if r["rlm"]["scores"].get("judge", -1) >= 0]
    rlm_times = [r["rlm"]["elapsedS"] for r in results]
    rlm_subs = [r["rlm"]["subQueries"] for r in results]

    def avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    bar = "═" * (BOX_W + 2)
    thin = "─" * (BOX_W + 2)

    print(f"\n  {BOLD}{bar}{RESET}")
    print(f"  {CYAN}{BOLD}  {benchmark_name} — Batch Results ({n} examples){RESET}")
    print(f"  {BOLD}{bar}{RESET}")
    print()

    # Table header
    print(f"  {'Metric':<24} {'Direct LLM':>12} {'RLM':>12} {'Delta':>12}")
    print(f"  {thin}")

    def row(label, d_val, r_val, fmt=".2f", higher_better=True):
        delta = r_val - d_val
        sign = "+" if delta > 0 else ""
        color = GREEN if (delta > 0) == higher_better else RED if delta != 0 else DIM
        print(f"  {label:<24} {d_val:>12{fmt}} {r_val:>12{fmt}} {color}{sign}{delta:>11{fmt}}{RESET}")

    if direct_judges and rlm_judges:
        row("Judge (avg)", avg(direct_judges), avg(rlm_judges))
    row("Contains (avg)", avg(direct_cs), avg(rlm_cs))
    row("F1 (avg)", avg(direct_f1s), avg(rlm_f1s))
    print(f"  {thin}")
    row("Time (avg s)", avg(direct_times), avg(rlm_times), higher_better=False)
    row("Time (total s)", sum(direct_times), sum(rlm_times), ".1f", higher_better=False)
    print(f"  {thin}")
    print(f"  {'Sub-queries (avg)':<24} {'—':>12} {avg(rlm_subs):>12.1f}")
    print(f"  {'Sub-queries (total)':<24} {'—':>12} {sum(rlm_subs):>12}")
    print()

    # Per-example breakdown — use judge as primary metric
    has_judge = all(r["directLlm"]["scores"].get("judge", -1) >= 0 for r in results)
    primary = "Judge" if has_judge else "F1"
    print(f"  {BOLD}Per-example scores:{RESET}")
    print(f"  {'idx':>6}  {'D.Judge':>8}  {'R.Judge':>8}  {'D.F1':>6}  {'R.F1':>6}  {'Winner':>8}  {'Context':>10}")
    print(f"  {thin}")
    for r in results:
        d_f1 = r["directLlm"]["scores"]["f1"]
        r_f1 = r["rlm"]["scores"]["f1"]
        d_j = r["directLlm"]["scores"].get("judge", -1)
        r_j = r["rlm"]["scores"].get("judge", -1)
        ctx_size = f"{r['contextChars'] / 1024:.0f}KB"

        d_j_str = f"{d_j:.1f}" if d_j >= 0 else "—"
        r_j_str = f"{r_j:.1f}" if r_j >= 0 else "—"

        # Winner based on judge if available, else F1
        if has_judge:
            if r_j > d_j + 0.01:
                winner = f"{GREEN}RLM{RESET}"
            elif d_j > r_j + 0.01:
                winner = f"{MAGENTA}Direct{RESET}"
            else:
                winner = f"{DIM}Tie{RESET}"
        else:
            if r_f1 > d_f1 + 0.01:
                winner = f"{GREEN}RLM{RESET}"
            elif d_f1 > r_f1 + 0.01:
                winner = f"{MAGENTA}Direct{RESET}"
            else:
                winner = f"{DIM}Tie{RESET}"
        print(f"  {r['idx']:>6}  {d_j_str:>8}  {r_j_str:>8}  {d_f1:>6.2f}  {r_f1:>6.2f}  {winner:>17}  {ctx_size:>10}")

    print(f"\n  {BOLD}{bar}{RESET}\n")

    # Summary dict
    summary = {
        "benchmark": benchmark_name,
        "n": n,
        "directLlm": {
            "judge_avg": avg(direct_judges) if direct_judges else None,
            "f1_avg": avg(direct_f1s),
            "contains_avg": avg(direct_cs),
            "time_avg_s": avg(direct_times),
            "time_total_s": sum(direct_times),
        },
        "rlm": {
            "judge_avg": avg(rlm_judges) if rlm_judges else None,
            "f1_avg": avg(rlm_f1s),
            "contains_avg": avg(rlm_cs),
            "time_avg_s": avg(rlm_times),
            "time_total_s": sum(rlm_times),
            "sub_queries_avg": avg(rlm_subs),
            "sub_queries_total": sum(rlm_subs),
        },
        "per_example": results,
    }
    return summary


# ── Main ────────────────────────────────────────────────────────────────────

def _partial_batch_path(dataset: str) -> Path:
    """Path for the in-progress partial batch file."""
    traj_dir = Path.cwd() / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    return traj_dir / f"batch-{dataset}-partial.json"


def _save_partial(path: Path, metadata: dict, results: list[dict]):
    """Atomically save partial batch progress."""
    data = {**metadata, "results": results}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.rename(path)


def _load_partial(path: Path) -> dict | None:
    """Load partial batch progress if it exists."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def main():
    parser = argparse.ArgumentParser(description="Batch benchmark runner")
    parser.add_argument("dataset", choices=["longbench", "oolong"],
                        help="Dataset to benchmark")
    parser.add_argument("-n", type=int, default=10,
                        help="Number of random samples (default: 10)")
    parser.add_argument("--indices", type=str, default=None,
                        help="Comma-separated list of specific indices")
    parser.add_argument("--all", action="store_true",
                        help="Run all examples in the dataset")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Quiet mode — skip per-example display")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sample selection (default: 42)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from partial batch progress")
    parser.add_argument("--no-resume", action="store_true",
                        help="Discard partial progress and start fresh")
    args = parser.parse_args()

    is_longbench = args.dataset == "longbench"
    bench_name = "LongBench NarrativeQA" if is_longbench else "Oolong Synth"
    dataset_label = "THUDM/LongBench (narrativeqa)" if is_longbench else "oolongbench/oolong-synth"

    print(f"\n  {CYAN}{BOLD}{bench_name} — Batch Benchmark{RESET}")

    # Parse explicit indices if given
    explicit_indices = None
    if args.indices:
        explicit_indices = [int(x.strip()) for x in args.indices.split(",")]
        print(f"  {DIM}Indices: {explicit_indices}{RESET}")

    # Check for partial progress
    partial_path = _partial_batch_path(args.dataset)
    partial = None
    completed_indices = set()
    results = []

    if not args.no_resume:
        partial = _load_partial(partial_path)

    if partial and partial.get("results"):
        completed_indices = {r["idx"] for r in partial["results"]}
        results = list(partial["results"])
        if args.resume or completed_indices:
            print(f"  {YELLOW}Found partial progress: {len(results)} examples already done{RESET}")
            print(f"  {DIM}Completed indices: {sorted(completed_indices)}{RESET}")
            if not args.resume:
                print(f"  {DIM}Use --resume to continue, --no-resume to start fresh{RESET}")

    if args.no_resume:
        results = []
        completed_indices = set()
        if partial_path.exists():
            partial_path.unlink()

    # Single subprocess call: load dataset, sample if needed, extract examples
    print(f"  {DIM}Loading dataset and selecting examples...{RESET}")
    data = load_dataset_examples(
        dataset=args.dataset,
        n=args.n,
        indices=explicit_indices,
        all_examples=args.all,
        seed=args.seed,
    )
    examples = data["examples"]
    indices = data["indices"]
    print(f"  {DIM}Dataset size: {data['size']} | Selected: {len(indices)} indices{RESET}")
    if not explicit_indices and not args.all:
        print(f"  {DIM}Sampled indices (seed={args.seed}): {indices}{RESET}")
    print(f"  {GREEN}✓{RESET} Loaded {len(examples)} examples\n")

    # Filter out already-completed examples
    remaining = [ex for ex in examples if ex["idx"] not in completed_indices]
    if completed_indices and remaining:
        print(f"  {GREEN}Skipping {len(completed_indices)} completed, {len(remaining)} remaining{RESET}\n")

    # Import bench runner
    from benchmarks._rlm_bench import run_benchmark

    # Metadata for partial saves
    partial_metadata = {
        "benchmark": bench_name,
        "dataset": args.dataset,
        "indices": indices,
        "n_target": len(examples),
        "seed": args.seed,
    }

    # Graceful shutdown on signals
    shutdown_requested = False

    def handle_signal(signum, frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            # Second signal — force exit
            sys.exit(1)
        shutdown_requested = True
        print(f"\n  {YELLOW}Shutdown requested — finishing current example...{RESET}")

    signal.signal(signal.SIGTERM, handle_signal)

    # Run each example
    total_start = time.time()
    total_count = len(examples)

    for i, ex in enumerate(remaining):
        if shutdown_requested:
            break

        idx = ex["idx"]
        done_count = len(completed_indices) + (i)
        progress = f"[{done_count + 1}/{total_count}]"
        print(f"  {BOLD}{progress}{RESET} Index {idx} ({len(ex['context']) / 1024:.0f}KB)")

        try:
            result = run_benchmark(
                benchmark_name=bench_name,
                dataset_label=dataset_label,
                idx=idx,
                context=ex["context"],
                question=ex["question"],
                expected=ex["expected"],
                verbose=not args.quiet,
            )
            results.append(result)
            completed_indices.add(idx)

            # Save partial progress after each example
            _save_partial(partial_path, partial_metadata, results)

            # Compact progress in quiet mode
            if args.quiet:
                d_j = result["directLlm"]["scores"].get("judge", -1)
                r_j = result["rlm"]["scores"].get("judge", -1)
                d_t = result["directLlm"]["elapsedS"]
                r_t = result["rlm"]["elapsedS"]
                d_str = f"J={d_j:.1f}" if d_j >= 0 else f"F1={result['directLlm']['scores']['f1']:.2f}"
                r_str = f"J={r_j:.1f}" if r_j >= 0 else f"F1={result['rlm']['scores']['f1']:.2f}"
                print(f"    Direct: {d_str} ({d_t:.1f}s)  |  RLM: {r_str} ({r_t:.1f}s)")
        except KeyboardInterrupt:
            print(f"\n  {YELLOW}Interrupted after {len(results)} examples.{RESET}")
            _save_partial(partial_path, partial_metadata, results)
            print(f"  {DIM}Progress saved. Run with --resume to continue.{RESET}")
            break
        except Exception as e:
            print(f"  {RED}Example {idx} failed: {e}{RESET}")
            continue

    total_time = time.time() - total_start
    print(f"\n  {DIM}Total wall time: {total_time:.1f}s{RESET}")

    # Print summary
    summary = print_summary(bench_name, results)

    # Save final summary and clean up partial file
    if results:
        traj_dir = Path.cwd() / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y-%m-%dT%H-%M-%S")
        slug = args.dataset
        summary_file = f"batch-{slug}-n{len(results)}-{ts}.json"
        (traj_dir / summary_file).write_text(json.dumps(summary, indent=2))
        print(f"  {DIM}Saved: {summary_file}{RESET}")

        # Clean up partial file only if all examples completed
        if len(results) >= len(examples):
            if partial_path.exists():
                partial_path.unlink()
                print(f"  {DIM}Cleaned up partial progress file{RESET}")
        else:
            print(f"  {DIM}Partial progress kept ({len(results)}/{len(examples)}). Use --resume to continue.{RESET}")
        print()


if __name__ == "__main__":
    main()
