# Healthcare AI Agent — Secure Data Pipeline

An AI agent that generates Python code via **Claude Haiku** and executes it inside **Monty** (Pydantic's secure Python interpreter written in Rust). Demonstrates how to safely run AI-generated code in a regulated healthcare environment.

## Why This Exists

In healthcare data engineering, you need to:
- Process sensitive patient data (PHI/PII) under HIPAA-like regulations
- Use AI agents to automate data pipelines
- **Guarantee** that AI-generated code cannot access the filesystem, network, or environment variables

Monty solves this: it's a sandboxed Python interpreter with microsecond startup, no containers needed.

<img width="1402" height="584" alt="mon" src="https://github.com/user-attachments/assets/6d53c9f1-b882-4cb6-a11d-32b33fb1636c" />

## What It Does

The app runs a 3-step data pipeline on fake patient records:

```
Step 1: Validate     → Check required fields, flag incomplete records
Step 2: De-identify  → Mask names, SSNs, DOBs, addresses (HIPAA)
Step 3: Aggregate    → Compute diagnosis counts, avg lab values
```

For each step:
1. Claude Model **generates** ~20 lines of Python code
2. Monty **sandboxes** and executes the code (no filesystem, no network)
3. Rich terminal dashboard **animates** the progress live

## Architecture

```
┌──────────────────────────────────────────────┐
│  Rich Live Dashboard (terminal UI)           │
│  ┌────────┐  ┌───────────┐  ┌────────────┐  │
│  │Validate│→ │De-identify│→ │ Aggregate  │  │
│  └───┬────┘  └─────┬─────┘  └─────┬──────┘  │
│      │              │              │         │
│  ┌───▼──────────────▼──────────────▼──────┐  │
│  │  Claude Haiku → generates Python code  │  │
│  │  Monty Sandbox → executes it safely    │  │
│  └────────────────────────────────────────┘  │
└──────────────────────────────────────────────┘
```

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
# Clone and install
git clone <repo-url>
cd py-sandbox-trial
uv sync --all-groups
```

## Run

```bash
# Set your Anthropic API key
export ANTHROPIC_API_KEY=your-key-here

# Optionally choose a model (default: claude-opus-4-6)
export CLAUDE_MODEL=claude-opus-4-6

# Run the pipeline
uv run python main.py
```

| Env Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | (required) | Your Anthropic API key |
| `CLAUDE_MODEL` | `claude-opus-4-6` | Model for code generation (e.g. `claude-sonnet-4-5-20250929`, `claude-haiku-4-5-20251001`) |

You'll see a live terminal dashboard showing each step animate through code generation, sandbox execution, and results.

## Run Tests

```bash
uv run pytest tests/ -v
```

Tests cover:
- **Sandbox security** (14 tests) — Monty blocks `open()`, `os.listdir()`, `exec()`, `eval()`, `__import__()`, etc.
- **Agent functions** (6 tests) — Code generation (mocked Claude API) and sandbox execution with real healthcare data
- **Pipeline integration** (2 tests) — End-to-end pipeline with mocked Claude API

## Project Structure

```
py-sandbox-trial/
├── main.py                         # Entry point
├── pyproject.toml                  # uv project config
├── src/healthcare_agent/
│   ├── agent.py                    # generate_code() + execute_in_sandbox()
│   ├── pipeline.py                 # Rich live dashboard + orchestration
│   ├── data.py                     # 5 fake patient records
│   └── prompts.py                  # Claude system + step prompts
└── tests/
    ├── test_sandbox.py             # Monty security boundary tests
    ├── test_agent.py               # Agent function tests
    └── test_pipeline.py            # End-to-end pipeline tests
```

## Security Model

Monty enforces these boundaries on all AI-generated code:

| Access | Status |
|---|---|
| Filesystem (`open()`, file I/O) | Blocked |
| Network (HTTP, sockets) | Blocked |
| Environment variables | Blocked |
| System calls (`os.*`, `subprocess`) | Blocked |
| Dangerous builtins (`exec`, `eval`, `__import__`) | Blocked |
| Basic Python (dicts, lists, functions, loops) | Allowed |
| JSON module | Allowed |

## Dependencies

| Package | Purpose |
|---|---|
| `pydantic-monty` | Secure Python sandbox (Rust-based) |
| `anthropic` | Claude API for code generation |
| `rich` | Terminal dashboard with live panels |

