# my-writing-rl

*Created: 2026-03-15*

RL-based creative writing training pipeline. Uses the [verifiers](https://github.com/PrimeIntellect-ai/verifiers) library to define an RL environment that scores LLM-generated stories with the [LitBench](https://huggingface.co/SAA-Lab/Creative-Writing-Verifier) Bradley-Terry reward model, trained via [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) (async GRPO with vLLM rollouts).

## How the Stack Connects

```
LitBench-Train dataset          →  prompts fed to policy model
                                    (r/WritingPrompts writing prompts)

Creative-Writing-Verifier (BT RM) →  reward function inside a verifiers Environment
                                    (scores each generated story, returns 0.0–1.0)

verifiers library               →  wraps dataset + reward into an Environment
                                    that prime-rl knows how to consume

prime-rl                        →  runs async GRPO, calls vLLM for rollouts,
                                    calls the environment for rewards
```

## Setup

```bash
# Requires Python 3.12+
uv sync

# Install prime-rl (for training)
curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/scripts/install.sh | bash
```

## Next Steps

### 1. Calibrate the Reward Model

Run a small batch of prompts through the BT reward model to verify sigmoid-normalized scores land in a useful range. Target a mean of 0.3-0.7 across samples. If scores cluster near 0 or 1, the normalization needs adjusting.

```python
from environments.creative_writing.creative_writing import build_dataset, score_with_litbench_rm

ds = build_dataset(num_examples=20)
for row in ds:
    # Score a dummy story against each prompt to check the range
    score = score_with_litbench_rm(row["question"], "Once upon a time, a story happened.")
    print(f"score={score:.3f}")
```

### 2. Serve Inference and Run vf-eval

Spin up vLLM with a small model and run the eval loop to confirm the full pipeline works end-to-end. This requires a CUDA GPU.

```bash
# Terminal 1: serve inference
vllm serve Qwen/Qwen3-1.7B --port 8000

# Terminal 2: eval against the environment
uv run vf-eval creative_writing -m "Qwen/Qwen3-1.7B" -b "http://localhost:8000/v1" -n 50 -r 3
```

Watch the average reward. You want it landing in the 0.3-0.7 range — enough signal to learn from, not already saturated. If the model struggles to get non-zero rewards, consider SFT warmup first.

### 3. Run GRPO Training via prime-rl

Once the reward signal looks good:

```bash
# Terminal 1: inference worker
vllm serve Qwen/Qwen3-1.7B --port 8000 --enable-prefix-caching

# Terminal 2: training
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

### 4. Consider SFT Warmup

If the base model gets very low initial rewards, warm-start with supervised fine-tuning on the `chosen_story` column from LitBench-Train before running RL. Verifiers supports SFT warmup on filtered rollouts.

### 5. Domain Adaptation for Coco's Story

LitBench is r/WritingPrompts fiction. For oral/family history narrative (Coco's Story), you'll want to:

- Fine-tune the reward model on a small in-domain preference set, or
- Add a second reward component (LLM judge with a family-history-specific rubric) weighted alongside the LitBench RM

## LitBench-Train Dataset Schema

43,827 rows with these columns:

| Column | Type | Description |
|---|---|---|
| `prompt` | string | r/WritingPrompts prompt |
| `chosen_story` | string | Higher-rated story |
| `rejected_story` | string | Lower-rated story |
| `chosen_upvotes` | int64 | Upvote count for chosen |
| `rejected_upvotes` | int64 | Upvote count for rejected |
