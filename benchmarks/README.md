# RLM Plugin Benchmarks

Evaluation scripts for testing the RLM plugin's runtime against standard long-context benchmarks.

## Prerequisites

- `claude` CLI installed and authenticated
- Python 3.11+ (venv with `datasets` is auto-created on first run)

## Available Benchmarks

### Oolong Synth (oolongbench/oolong-synth)
Synthetic long-context tasks: timeline ordering, user tracking, counting.

```bash
python3 benchmarks/oolong_synth.py

# Custom index
python3 benchmarks/oolong_synth.py --idx 50
```

### LongBench NarrativeQA (THUDM/LongBench)
Reading comprehension over long narratives.

```bash
python3 benchmarks/longbench_narrativeqa.py

# Custom index
python3 benchmarks/longbench_narrativeqa.py --idx 200
```

## How benchmarks work

Each benchmark:
1. Loads a dataset from HuggingFace via the `datasets` library
2. Extracts context + question for a given index
3. Runs **Direct LLM** (single `claude -p` call with full context)
4. Runs **RLM** pattern using `rlm_runtime` (chunk → parallel sub-queries → synthesize)
5. Compares both results against the expected answer
6. Saves trajectory JSON to `trajectories/`
