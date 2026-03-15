# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RL-based creative writing training pipeline. Uses the **verifiers** library to define an RL environment that scores LLM-generated stories with the LitBench Bradley-Terry reward model (`SAA-Lab/Creative-Writing-Verifier`), trained via **prime-rl** (async GRPO with vLLM rollouts).

The stack: LitBench-Train dataset (prompts) → verifiers Environment (reward via BT RM) → prime-rl (GRPO training).

## Development

**Package manager:** `uv` (not pip)

```bash
# Install dependencies
uv sync

# Run the project
uv run python main.py

# Add a dependency
uv add <package>
```

**Python version:** 3.12+ (set in `.python-version`)

## Architecture

- `main.py` — Entry point
- `environments/` — verifiers Environment definitions (dataset + reward rubric)
- `prime-rl/` — Prime RL training framework (git submodule)

### verifiers Environment Pattern

Environments follow this structure:
1. Load a HuggingFace dataset with `question`/`answer` columns
2. Define async reward functions returning `float` scores
3. Compose into a `vf.Rubric` with weights
4. Wire into `vf.SingleTurnEnv` with a system prompt

Key rule: **never override `rollout()`** — only define reward functions and dataset loading.

### Training Workflow

```bash
# 1. Serve inference model
vllm serve <model> --port 8000 --enable-prefix-caching

# 2. Eval reward signal (want 0.3–0.7 avg reward)
uv run vf-eval <env_name> -m <model> -b "http://localhost:8000/v1" -n 50 -r 3

# 3. Train
prime train --env environments/<env_name> --model <model> --num-gpus 1 ...
```

## Code Style

Follow verifiers conventions: strict `ruff` formatting, explicit type annotations, snake_case for functions/variables, PascalCase for classes. "Fail fast, fail loud" — no silent error swallowing.
