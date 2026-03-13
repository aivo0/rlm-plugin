# RLM Plugin Benchmarks

Evaluation scripts for testing the RLM plugin's runtime against standard long-context benchmarks.

## Prerequisites

- `claude` CLI installed and authenticated
- Python 3.11+ (venv with `datasets` is auto-created on first run)

## Single-Example Benchmarks

### Oolong Synth (oolongbench/oolong-synth)
Synthetic long-context tasks: timeline ordering, user tracking, counting.

```bash
python3 benchmarks/oolong_synth.py
python3 benchmarks/oolong_synth.py --idx 50
```

### LongBench NarrativeQA (THUDM/LongBench)
Reading comprehension over long narratives.

```bash
python3 benchmarks/longbench_narrativeqa.py
python3 benchmarks/longbench_narrativeqa.py --idx 99
```

## Batch Benchmark

Run multiple examples and compute aggregate scores (F1, exact match, contains match).

```bash
# 10 random samples from LongBench (default)
python3 benchmarks/batch.py longbench

# 20 random samples from Oolong
python3 benchmarks/batch.py oolong -n 20

# Specific indices
python3 benchmarks/batch.py longbench --indices 5,42,99,182

# Full dataset
python3 benchmarks/batch.py oolong --all

# Quiet mode (compact output, just scores)
python3 benchmarks/batch.py longbench -n 20 -q

# Custom random seed
python3 benchmarks/batch.py oolong -n 10 --seed 123
```

## Scoring Metrics

- **Judge** — LLM-as-judge via `claude -p`: scores whether the prediction conveys the same key facts as the reference (1.0 = correct, 0.5 = partial, 0.0 = wrong). Primary metric.
- **Contains** — 1.0 if all reference tokens appear in the prediction
- **F1** — Token-level F1 between predicted and reference answers (penalizes verbose answers)

## How benchmarks work

Each benchmark:
1. Loads a dataset from HuggingFace via the `datasets` library
2. Extracts context + question for a given index
3. Runs **Direct LLM** (single `claude -p` call with full context)
4. Runs **RLM** pattern using `rlm_runtime` (chunk → parallel sub-queries → synthesize)
5. Scores both results against the expected answer
6. Saves trajectory JSON to `trajectories/`

The batch runner aggregates scores across examples and prints a comparison table.
