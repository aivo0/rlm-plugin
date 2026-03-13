"""Shared RLM benchmark runner — loads context, runs Direct LLM vs RLM, compares."""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Add plugin root to path for rlm_runtime
sys.path.insert(0, str(Path(__file__).parent.parent))

import rlm_runtime
from benchmarks._display import (
    BOLD, BLUE, CYAN, DIM, GREEN, MAGENTA, RED, RESET, YELLOW,
    BOX_W, MAX_CONTENT_W, Spinner, display_bar, display_box,
    format_size, wrap_text, box_top, box_bottom, box_line,
)

CHUNK_SIZE = 6000  # characters per chunk


def run_direct_llm(context: str, question: str) -> tuple[str, float]:
    """Run direct LLM via claude -p. Returns (answer, elapsed_seconds)."""
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer the question based on the context above. Be concise."
    t = time.time()
    result = subprocess.run(
        ["claude", "-p"],
        input=prompt,
        capture_output=True, text=True,
    )
    elapsed = time.time() - t
    if result.returncode != 0:
        raise RuntimeError(f"claude -p failed: {result.stderr}")
    return result.stdout.strip(), elapsed


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """Split text into chunks, preferring line boundaries."""
    lines = text.split("\n")
    chunks = []
    current = []
    current_len = 0

    for line in lines:
        line_len = len(line) + 1  # +1 for newline
        if current_len + line_len > chunk_size and current:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
        current.append(line)
        current_len += line_len

    if current:
        chunks.append("\n".join(current))

    return chunks


async def run_rlm(context: str, question: str) -> tuple[str, float, int, int]:
    """Run RLM pattern using rlm_runtime. Returns (answer, elapsed_s, iterations, sub_queries)."""
    t = time.time()

    # Step 1: Save context to temp file and load via runtime
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(context)
        context_path = f.name

    try:
        ctx = rlm_runtime.load_context(context_path)
    finally:
        os.unlink(context_path)

    # Step 2: Inspect
    meta = rlm_runtime.context_meta(ctx)
    print(f"    {DIM}{meta.split(chr(10))[1].strip()}, {meta.split(chr(10))[2].strip()}{RESET}")

    # Step 3: Chunk
    chunks = chunk_text(ctx)
    n_chunks = len(chunks)
    print(f"    {DIM}Split into {n_chunks} chunks (~{CHUNK_SIZE} chars each){RESET}")

    # Step 4: Sub-query each chunk in parallel
    instruction = (
        f"Read the following text excerpt carefully and extract any information "
        f"relevant to this question: {question}\n\n"
        f"If the excerpt contains relevant information, summarize it concisely. "
        f"If it contains no relevant information, respond with: NO_RELEVANT_INFO"
    )

    print(f"    {CYAN}Running {n_chunks} parallel sub-queries...{RESET}")
    spinner = Spinner()
    spinner.start(f"Processing {n_chunks} chunks")

    tasks = [
        rlm_runtime.async_llm_query(chunk, instruction)
        for chunk in chunks
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    spinner.stop()

    # Filter out errors and empty results
    relevant = []
    errors = 0
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            errors += 1
            print(f"    {RED}Chunk {i + 1} failed: {r}{RESET}")
        elif "NO_RELEVANT_INFO" not in r and r.strip():
            relevant.append(r.strip())

    print(f"    {GREEN}{len(relevant)}{RESET}{DIM} relevant results, {errors} errors{RESET}")

    # Step 5: Synthesize
    if not relevant:
        answer = "Could not find relevant information to answer the question."
    else:
        combined = "\n\n---\n\n".join(relevant)

        # If combined results are small enough, synthesize in one call
        if len(combined) < 30000:
            synthesis_prompt = (
                f"Based on the following excerpts from a longer document, answer this question:\n\n"
                f"Question: {question}\n\n"
                f"Excerpts:\n{combined}\n\n"
                f"Provide a concise, direct answer."
            )
            spinner.start("Synthesizing final answer")
            answer = rlm_runtime.llm_query("", synthesis_prompt)
            spinner.stop()
        else:
            # Need another round of reduction
            print(f"    {CYAN}Combined results too large ({format_size(len(combined))}), reducing...{RESET}")
            sub_chunks = chunk_text(combined, chunk_size=8000)
            reduce_instruction = (
                f"Summarize the key facts relevant to this question: {question}\n\n"
                f"Be concise but preserve all relevant details."
            )
            spinner.start(f"Reducing {len(sub_chunks)} sub-results")
            reduce_tasks = [
                rlm_runtime.async_llm_query(sc, reduce_instruction)
                for sc in sub_chunks
            ]
            reduced = await asyncio.gather(*reduce_tasks, return_exceptions=True)
            spinner.stop()

            reduced_text = "\n\n".join(
                r for r in reduced
                if isinstance(r, str) and r.strip()
            )
            synthesis_prompt = (
                f"Based on the following information, answer this question:\n\n"
                f"Question: {question}\n\n"
                f"{reduced_text}\n\n"
                f"Provide a concise, direct answer."
            )
            spinner.start("Synthesizing final answer")
            answer = rlm_runtime.llm_query("", synthesis_prompt)
            spinner.stop()
            n_chunks += len(sub_chunks)

    elapsed = time.time() - t
    # iterations: 1 for chunk pass + optional reduce pass
    iterations = 2 if len("".join(relevant)) > 30000 else 1
    return answer, elapsed, iterations, n_chunks


def run_benchmark(
    benchmark_name: str,
    dataset_label: str,
    idx: int,
    context: str,
    question: str,
    expected,
) -> None:
    """Run a full benchmark: direct LLM vs RLM, display comparison, save trajectory."""

    print(f"\n  {CYAN}{BOLD}{benchmark_name}{RESET}")
    print(f"  {DIM}Dataset: {dataset_label} | Index: {idx}{RESET}")
    print(f"  {GREEN}✓{RESET} Loaded: context={format_size(len(context))} chars")
    print(f"  {DIM}Question: {question[:80]}{'...' if len(question) > 80 else ''}{RESET}")
    print(f"  {DIM}Expected: {json.dumps(expected) if isinstance(expected, list) else expected}{RESET}\n")

    # ── Direct LLM ──────────────────────────────────────────────────────────
    display_bar("Direct LLM", "(no RLM)", MAGENTA)

    spinner = Spinner()
    spinner.start("Generating response")
    try:
        direct_text, direct_time = run_direct_llm(context, question)
    except Exception as e:
        spinner.stop()
        print(f"  {RED}Direct LLM failed: {e}{RESET}")
        direct_text, direct_time = f"ERROR: {e}", 0.0
    spinner.stop()

    display_box(f"✔ Direct Result  {DIM}{direct_time:.1f}s", direct_text, MAGENTA)
    print()

    # ── RLM ──────────────────────────────────────────────────────────────────
    display_bar("RLM", "(rlm_runtime)", BLUE)

    try:
        rlm_text, rlm_time, iterations, sub_queries = asyncio.run(
            run_rlm(context, question)
        )
    except Exception as e:
        print(f"  {RED}RLM failed: {e}{RESET}")
        rlm_text, rlm_time, iterations, sub_queries = f"ERROR: {e}", 0.0, 0, 0

    stats = (
        f"{iterations} step{'s' if iterations != 1 else ''} · "
        f"{sub_queries} sub-quer{'ies' if sub_queries != 1 else 'y'} · "
        f"{rlm_time:.1f}s"
    )
    display_box(f"✔ RLM Result  {DIM}{stats}", rlm_text, GREEN)
    print()

    # ── Comparison summary ──────────────────────────────────────────────────
    sum_bar = "═" * (BOX_W + 2)
    print(f"  {BOLD}{sum_bar}{RESET}")
    expected_str = json.dumps(expected) if isinstance(expected, list) else str(expected)
    print(f"  {YELLOW}{BOLD}Expected:{RESET} {expected_str}")
    print(f"  {BOLD}{'─' * (BOX_W + 2)}{RESET}")

    sum_max_w = BOX_W + 2
    print(f"  {MAGENTA}{BOLD}Direct LLM{RESET} {DIM}({direct_time:.1f}s){RESET}")
    for line in direct_text.split("\n"):
        for chunk in wrap_text(line, sum_max_w):
            print(f"  {chunk}")
    print(f"  {BOLD}{'─' * (BOX_W + 2)}{RESET}")

    print(f"  {GREEN}{BOLD}RLM{RESET} {DIM}({rlm_time:.1f}s, {iterations} iters, {sub_queries} subs){RESET}")
    for line in rlm_text.split("\n"):
        for chunk in wrap_text(line, sum_max_w):
            print(f"  {chunk}")
    print(f"  {BOLD}{sum_bar}{RESET}")

    # ── Save trajectory ─────────────────────────────────────────────────────
    traj_dir = Path.cwd() / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H-%M-%S")
    bench_slug = benchmark_name.lower().replace(" ", "-")
    traj_file = f"benchmark-{bench_slug}-idx{idx}-{ts}.json"
    (traj_dir / traj_file).write_text(json.dumps({
        "benchmark": bench_slug,
        "idx": idx,
        "expected": expected,
        "directLlm": {"answer": direct_text, "elapsedS": direct_time},
        "rlm": {
            "answer": rlm_text,
            "iterations": iterations,
            "subQueries": sub_queries,
            "elapsedS": rlm_time,
        },
    }, indent=2))
    print(f"\n  {DIM}Saved: {traj_file}{RESET}\n")
