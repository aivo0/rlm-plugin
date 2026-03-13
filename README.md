# rlm — Recursive LLM plugin for Claude Code

A Claude Code plugin that processes contexts too large for a single LLM call. It spawns a subagent that writes and executes Python scripts to chunk the input, sub-query each chunk via `claude -p`, and synthesize the results — iterating until the output fits.

## How it works

```
Large file / dataset
        │
        ▼
  ┌─────────────┐
  │  Load into  │  rlm_runtime.load_context()
  │  Python     │
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  Chunk      │  Split into 3–8K char pieces
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  Sub-query  │  Parallel claude -p calls via
  │  each chunk │  rlm_runtime.async_llm_query()
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  Synthesize │  Merge / reduce / re-chunk
  └──────┬──────┘  if still too large
         │
         ▼
      Result
```

The LLM never sees the full context. Only chunks go to sub-queries, and only the final synthesized output is returned.

## Installation

Requires [Claude Code](https://docs.anthropic.com/en/docs/claude-code) CLI.

### From the plugin directory

```bash
claude plugin add /path/to/rlm-plugin
```

### Manual install

Clone the repo and symlink it into your local plugins directory:

```bash
git clone https://github.com/aivo0/rlm-plugin.git
mkdir -p ~/.claude/plugins/local/plugins
ln -s "$(pwd)/rlm-plugin" ~/.claude/plugins/local/plugins/rlm
```

Then register it in `~/.claude/plugins/installed_plugins.json` — add to the `"plugins"` object:

```json
"rlm@local": [
  {
    "scope": "user",
    "installPath": "/absolute/path/to/rlm-plugin",
    "version": "0.1.0",
    "installedAt": "2026-01-01T00:00:00.000Z",
    "lastUpdated": "2026-01-01T00:00:00.000Z"
  }
]
```

Restart Claude Code after installing.

## Usage

The RLM agent activates automatically when Claude Code determines that a task involves large contexts. You can also invoke it explicitly:

```
Analyze the 200K-line log file at /tmp/server.log and find all error patterns
```

```
Summarize this 500-page PDF: /home/user/docs/report.pdf
```

The agent writes Python scripts that use the `rlm_runtime` library:

```python
import sys
sys.path.insert(0, "${CLAUDE_PLUGIN_ROOT}")
import rlm_runtime

text = rlm_runtime.load_context("/path/to/large_file.txt")
print(rlm_runtime.context_meta(text))  # size + preview

# Chunk and sub-query
chunks = [text[i:i+6000] for i in range(0, len(text), 6000)]
import asyncio
results = asyncio.run(asyncio.gather(*[
    rlm_runtime.async_llm_query(chunk, "Extract key facts")
    for chunk in chunks
]))

# Synthesize
combined = "\n".join(results)
final = rlm_runtime.llm_query(combined, "Summarize these facts")
print(final)
```

### Runtime API

| Function | Description |
|---|---|
| `load_context(path)` | Read a file into a string |
| `context_meta(text)` | Char/line counts + head/tail preview |
| `llm_query(chunk, instruction)` | Synchronous `claude -p` sub-query |
| `async_llm_query(chunk, instruction)` | Async sub-query for use with `asyncio.gather()` |
| `save_state(key, value)` | Persist intermediate results across script calls |
| `load_state(key, default=None)` | Retrieve persisted state |
| `cleanup_state()` | Remove temp state directory |

## Benchmarks

The `benchmarks/` directory contains evaluation scripts that compare direct LLM (single `claude -p` with full context) against the RLM chunking approach on standard long-context datasets.

### Prerequisites

- `claude` CLI installed and authenticated
- Python 3.11+

Dependencies (`datasets`, `numpy`) are auto-installed into a `.venv` on first run.

### Single example

```bash
# Oolong Synth — synthetic long-context tasks
python3 benchmarks/oolong_synth.py
python3 benchmarks/oolong_synth.py --idx 50

# LongBench NarrativeQA — reading comprehension over long narratives
python3 benchmarks/longbench_narrativeqa.py
python3 benchmarks/longbench_narrativeqa.py --idx 199
```

### Batch benchmark

Run multiple examples and compute aggregate scores (token F1, exact match, contains match).

```bash
# 10 random samples (default)
python3 benchmarks/batch.py longbench
python3 benchmarks/batch.py oolong -n 20

# Specific indices
python3 benchmarks/batch.py longbench --indices 5,42,99,182

# Full dataset
python3 benchmarks/batch.py oolong --all

# Quiet mode — compact output, just scores per example + summary table
python3 benchmarks/batch.py longbench -n 20 -q

# Custom random seed
python3 benchmarks/batch.py oolong -n 10 --seed 123
```

The batch runner prints an aggregate comparison table and per-example F1 breakdown, then saves full results to `trajectories/batch-{dataset}-n{N}-{timestamp}.json`.

### Scoring metrics

| Metric | Description |
|---|---|
| **Judge** | LLM-as-judge via `claude -p` — scores whether the prediction conveys the correct key facts (1.0/0.5/0.0). Primary metric. |
| **Contains** | 1.0 if all reference tokens appear in the prediction |
| **F1** | Token-level F1 between predicted and reference answers (penalizes verbose answers) |

## Project structure

```
rlm-plugin/
├── plugin.json          # Plugin manifest
├── rlm_runtime.py       # Runtime library used by agent scripts
├── agents/
│   └── rlm.md           # Agent definition (system prompt + tools)
└── benchmarks/
    ├── batch.py                 # Batch benchmark runner with scoring
    ├── oolong_synth.py          # Oolong Synth single-example benchmark
    ├── longbench_narrativeqa.py # LongBench NarrativeQA single-example benchmark
    ├── _rlm_bench.py            # Shared benchmark runner (Direct LLM vs RLM)
    ├── _scoring.py              # Token F1, exact match, contains match
    ├── _display.py              # Terminal display helpers
    ├── _venv.py                 # Auto venv setup
    └── requirements.txt         # Benchmark Python deps
```

## References

- **Recursive LLM (RLM)**: Lee, S., Kim, H., & Yun, S. (2025). *Can LLMs Handle Unlimited Context? Benchmark and Solution for LLMs with Recursive LLM*. [arXiv:2502.12820](https://arxiv.org/abs/2502.12820)
- **rlm-cli**: This plugin was inspired by [rlm-cli](https://github.com/viplismism/rlm-cli) by viplismism.

## License

MIT
