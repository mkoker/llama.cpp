# Agent SDK Comparison

Agent SDK Comparison is a Python benchmark harness for comparing OpenAI, Anthropic, Goose, and Gemma adapters across standardized agent tasks: web search, code generation, file I/O, data parsing, and multi-step reasoning. It runs the same prompts across selected SDK adapters, collects latency/usage/validation results, and can operate in offline mock mode before you spend API calls.

## Supported SDKs and adapters

- `openai` — OpenAI API adapter.
- `anthropic` — Anthropic API adapter.
- `goose` — Goose CLI adapter.
- `gemma` — OpenAI-compatible local Gemma endpoint adapter.
- `mock` — deterministic offline adapter for validating the harness.

## Setup

Clone the repository, enter the project directory, optionally create a virtual environment, then install dependencies.

```bash
git clone <repo-url> agent-sdk-comparison
cd agent-sdk-comparison

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

The project is Python 3.12-compatible. `requirements.txt` currently installs:

- `openai`
- `anthropic`
- `PyYAML`
- `pytest`

## Configuration

Edit `config.yaml` to choose benchmark defaults, enabled agents, models, and adapter-specific settings. The checked-in default uses `benchmark.mode: mock`, which is the safest first run because it does not make network API calls.

Compact example aligned with the default `config.yaml`:

```yaml
benchmark:
  iterations: 1
  mode: mock
  temperature: 0.0
  max_tokens: 1024

agents:
  openai:
    enabled: true
    model: gpt-4o-mini
    api_key_env: OPENAI_API_KEY
  anthropic:
    enabled: true
    model: claude-3-5-haiku-latest
    api_key_env: ANTHROPIC_API_KEY
  goose:
    enabled: true
    model: null
    model_env: GOOSE_MODEL
    binary: goose
  gemma:
    enabled: true
    model: gemma-4-31b-it
    base_url: http://192.168.1.169:8080/v1
    api_key_env: GEMMA_API_KEY
```

For real runs, change `benchmark.mode` or pass `--mode real` on the command line, ensure the target agents are enabled, and confirm their credentials/runtime requirements are present.

## Secrets and environment variables

Do not store secrets in `config.yaml`. Do not store secrets in source control. API keys and model/runtime overrides are read from environment variables named by the config file.

Example shell setup:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOSE_MODEL="gpt-4o-mini"
export GEMMA_API_KEY="not-needed-for-some-local-servers"
```

Notes:

- `OPENAI_API_KEY` is used by the OpenAI adapter.
- `ANTHROPIC_API_KEY` is used by the Anthropic adapter.
- `GOOSE_MODEL` can select the model used by the Goose runtime when supported by your Goose setup.
- `GEMMA_API_KEY` is read for the OpenAI-compatible Gemma client; some local servers ignore the value but the environment variable should still be set if the adapter expects it.

## Running benchmarks

Start with mock mode to validate the harness, task selection, summaries, and output formatting without external API calls:

```bash
python run_benchmark.py --mode mock --summary
```

Run a small real-agent comparison for one task and one iteration:

```bash
python run_benchmark.py --agents openai,anthropic --tasks code_gen_slugify --iterations 1
```

Write results to an output file:

```bash
python run_benchmark.py --mode mock --summary --output reports/mock-results.json
```

Useful options:

- `--mode mock` — use the deterministic offline adapter.
- `--mode real` — use configured real SDK adapters.
- `--agents openai,anthropic,goose,gemma` — comma-separated adapter list.
- `--tasks code_gen_slugify,reasoning_project_schedule` — comma-separated task IDs.
- `--iterations 3` — repeat each selected agent/task pair.
- `--summary` — print a concise summary.
- `--output <path>` — write exported benchmark results.

## Built-in tasks

Canonical task IDs are defined in `tasks/basic_tasks.py`:

- `web_search_current_ai_news` — current AI news lookup with citations.
- `code_gen_slugify` — deterministic Python slugify helper generation.
- `file_read_write_summary` — read JSONL-style events and write an aggregate summary.
- `data_parse_invoice_text` — extract structured invoice JSON from text.
- `reasoning_project_schedule` — dependency-aware scheduling reasoning task.

## Adding tasks

Tasks live in `tasks/basic_tasks.py`. To add one:

1. Create a new `BenchmarkTask` with a stable `id`, human-readable `name`, `category`, prompt, and either `expected_markers` or a custom validator.
2. Add a helper function if the task is more than a few lines.
3. Include the new task in `all_tasks()` so it becomes part of the canonical benchmark set.
4. Update `config.yaml` if you want the task to be configurable by default.

Keep task construction offline-safe: defining a task should not perform network calls, API calls, or filesystem writes.

## Adding SDK adapters

The shared adapter contract is in `sdks/interface.py`. New adapters should implement `AgentInterface` or match the same shape:

- Define a stable `name`.
- Implement `run(prompt, **kwargs)`.
- Return a mapping with at least an `output` field; include normalized metadata such as model, latency, usage, command, stderr, or base URL when available.

After creating the adapter class, register it in `run_benchmark.py` by adding it to `AGENT_FACTORIES`. Once registered, it can be selected with `--agents <name>` and used in the same benchmark matrix as the built-in adapters.

## Notes and troubleshooting

- Goose requires the `goose` CLI to be installed and available on `PATH`; `config.yaml` points to it with `binary: goose`.
- Gemma defaults to the local OpenAI-compatible endpoint `http://192.168.1.169:8080/v1`. Change `agents.gemma.base_url` in `config.yaml` for another inference server.
- Use `mock` mode first to validate harness behavior without API calls, credentials, Goose runtime setup, or a local Gemma server.
- If a real run fails immediately, confirm the matching environment variable is exported and that the agent is enabled in `config.yaml`.
- If an adapter returns output but validation fails, inspect the task's `expected_markers` or validator in `tasks/basic_tasks.py` before assuming the SDK call failed.
