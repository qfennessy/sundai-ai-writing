# my-writing-rl

*Created: 2026-03-15 | Updated: 2026-03-15*

Creative writing model training pipeline using preference learning on [LitBench-Train](https://huggingface.co/datasets/SAA-Lab/LitBench-Train) (43k r/WritingPrompts chosen/rejected story pairs).

Two training paths:
1. **DPO (primary)** — trains directly on preference pairs, no reward model needed
2. **Online RL (GRPO)** — uses a reward model + [verifiers](https://github.com/PrimeIntellect-ai/verifiers) environment + [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) for rollout-based training

## Setup

```bash
# Requires Python 3.12+
uv sync
```

## Path 1: DPO Training (recommended starting point)

DPO learns directly from the chosen/rejected pairs — no reward model, no rollouts, roughly SFT-level compute. The model learns to produce text more like the higher-upvoted story and less like the lower-upvoted one.

```bash
# Single GPU (requires CUDA)
uv run python train_dpo.py

# With custom settings
uv run python train_dpo.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --beta 0.1 \
    --learning_rate 5e-7 \
    --num_train_epochs 1 \
    --output_dir ./checkpoints/dpo-prose-v1

# Multi-GPU
uv run accelerate launch train_dpo.py
```

Key hyperparameter: `--beta` controls KL penalty strength. Lower (0.05) = more aggressive improvement but higher collapse risk. Higher (0.2) = safer but smaller gains.

### Quick test run

```bash
# Small subset to verify the pipeline works
uv run python train_dpo.py --max_examples 100 --num_train_epochs 1
```

## Path 2: Online RL with Reward Model

Graduate to this after DPO plateaus. Online RL can discover outputs not in the dataset, pushing past the DPO quality ceiling.

Requires a reward model — either use `SAA-Lab/Creative-Writing-Verifier` or train a BT RM from LitBench-Train.

```bash
# Install prime-rl
curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/scripts/install.sh | bash

# Serve inference
vllm serve Qwen/Qwen3-1.7B --port 8000 --enable-prefix-caching

# Eval reward signal first (target 0.3-0.7 avg reward)
uv run vf-eval creative_writing -m "Qwen/Qwen3-1.7B" -b "http://localhost:8000/v1" -n 50 -r 3

# Train
prime train \
  --env environments/creative_writing \
  --model Qwen/Qwen3-1.7B \
  --num-gpus 1 \
  --batch-size 16 \
  --num-rollouts 4 \
  --learning-rate 1e-6 \
  --kl-coeff 0.05 \
  --output-dir ./checkpoints/creative-writing-v1
```

## DPO vs Online RL Tradeoffs

| | DPO | Online RL (GRPO + RM) |
|---|---|---|
| Requires RM? | No | Yes |
| Requires rollouts? | No | Yes |
| Compute | ~SFT level | 3-5x more |
| Data | Static pairs | Generates new data during training |
| Exploration | None — learns from existing pairs only | Discovers outputs not in dataset |
| Best for | Strong baseline fast | Pushing past dataset ceiling |

## Domain Adaptation for Coco's Story

LitBench is r/WritingPrompts fiction. For oral/family history narrative (Coco's Story):

- Fine-tune the reward model on a small in-domain preference set, or
- Add a second reward component (LLM judge with a family-history-specific rubric) weighted alongside the LitBench RM

## LitBench-Train Dataset Schema

43,827 rows:

| Column | Type | Description |
|---|---|---|
| `prompt` | string | r/WritingPrompts prompt |
| `chosen_story` | string | Higher-rated story (more upvotes) |
| `rejected_story` | string | Lower-rated story |
| `chosen_upvotes` | int64 | Upvote count for chosen |
| `rejected_upvotes` | int64 | Upvote count for rejected |
