#!/bin/bash
# Pod setup script for DPO training on Prime Intellect
# Usage: paste into pod after `prime pods ssh dpo-unslop`

set -euo pipefail

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

# Clone repo
git clone https://github.com/qfennessy/sundai-ai-writing.git
cd sundai-ai-writing

# Install dependencies
uv sync

# Run DPO training
uv run python train_dpo.py \
    --dataset unslop \
    --model_name Qwen/Qwen3-VL-4B-Instruct \
    --beta 0.1 \
    --learning_rate 5e-7 \
    --num_train_epochs 1 \
    --output_dir ./checkpoints/dpo-unslop-v1 \
    --wandb_entity qfennessy-sagacious-heritage \
    --wandb_project creative-writing-rl \
    --run_name dpo-unslop-v1
