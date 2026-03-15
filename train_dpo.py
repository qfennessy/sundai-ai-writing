"""DPO training on preference pairs.

Supports multiple datasets:
  - litbench:  SAA-Lab/LitBench-Train (43k r/WritingPrompts pairs)
  - unslop:    qfennessy/unslop-dpo (1k de-slop pairs)
  - Any HuggingFace dataset with prompt/chosen/rejected columns

Usage:
    # Train on unslop dataset
    uv run python train_dpo.py --dataset unslop

    # Train on LitBench
    uv run python train_dpo.py --dataset litbench

    # Train on any HF dataset with prompt/chosen/rejected columns
    uv run python train_dpo.py --dataset username/my-dataset

    # Multi-GPU
    uv run accelerate launch train_dpo.py --dataset unslop

    # Override defaults
    uv run python train_dpo.py \
        --dataset unslop \
        --model_name Qwen/Qwen3-VL-4B-Instruct \
        --beta 0.1 \
        --learning_rate 5e-7 \
        --num_train_epochs 1 \
        --output_dir ./checkpoints/dpo-unslop-v1
"""

import argparse
import os

from datasets import load_dataset
from trl import DPOConfig, DPOTrainer

DATASET_SHORTCUTS = {
    "litbench": "SAA-Lab/LitBench-Train",
    "unslop": "qfennessy/unslop-dpo",
}

SYSTEM_PROMPT = (
    "You are a skilled creative writer. "
    "Write vivid, emotionally resonant stories with strong voice and original detail. "
    "Avoid clichés, purple prose, and AI-sounding phrasing."
)


def parse_args():
    parser = argparse.ArgumentParser(description="DPO training on preference pairs")
    parser.add_argument(
        "--dataset",
        type=str,
        default="unslop",
        help="Dataset name: 'litbench', 'unslop', or any HF dataset ID with prompt/chosen/rejected columns",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints/dpo-v1",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="KL penalty strength. Lower (0.05) = more aggressive, higher (0.2) = safer",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-7,
        help="Learning rate (DPO is sensitive; keep lower than SFT)",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="Batch size per device",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps (effective batch = per_device * accum)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Max sequence length for prompt + response",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=0,
        help="Limit dataset size (0 = use all)",
    )
    parser.add_argument(
        "--eval_split",
        type=float,
        default=0.02,
        help="Fraction of data to hold out for eval",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity (team or username). Enables W&B logging when set.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="creative-writing-rl",
        help="W&B project name",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="W&B run name",
    )
    return parser.parse_args()


def prepare_litbench(max_examples: int = 0, eval_split: float = 0.02):
    """Load LitBench-Train and reshape for TRL's DPOTrainer."""
    ds = load_dataset("SAA-Lab/LitBench-Train", split="train")

    if max_examples > 0:
        ds = ds.shuffle(seed=42).select(range(min(max_examples, len(ds))))

    def reshape(example):
        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["prompt"]},
        ]
        return {
            "prompt": prompt_messages,
            "chosen": [{"role": "assistant", "content": example["chosen_story"]}],
            "rejected": [{"role": "assistant", "content": example["rejected_story"]}],
        }

    ds = ds.map(reshape, remove_columns=ds.column_names)
    return _maybe_split(ds, eval_split)


def prepare_generic(dataset_id: str, max_examples: int = 0, eval_split: float = 0.02):
    """Load any HF dataset with prompt/chosen/rejected columns."""
    ds = load_dataset(dataset_id, split="train")

    if max_examples > 0:
        ds = ds.shuffle(seed=42).select(range(min(max_examples, len(ds))))

    expected = {"prompt", "chosen", "rejected"}
    actual = set(ds.column_names)
    if not expected.issubset(actual):
        missing = expected - actual
        raise ValueError(
            f"Dataset {dataset_id} missing columns: {missing}. "
            f"Found: {actual}. Expected: {expected}"
        )

    # Drop extra columns
    extra = actual - expected
    if extra:
        ds = ds.remove_columns(list(extra))

    return _maybe_split(ds, eval_split)


def _maybe_split(ds, eval_split: float):
    if eval_split > 0:
        split = ds.train_test_split(test_size=eval_split, seed=42)
        return split["train"], split["test"]
    return ds, None


def main():
    args = parse_args()

    dataset_id = DATASET_SHORTCUTS.get(args.dataset, args.dataset)
    print(f"Dataset: {dataset_id}")

    if dataset_id == "SAA-Lab/LitBench-Train":
        train_ds, eval_ds = prepare_litbench(args.max_examples, args.eval_split)
    else:
        train_ds, eval_ds = prepare_generic(dataset_id, args.max_examples, args.eval_split)

    print(f"Train: {len(train_ds)} examples")
    if eval_ds:
        print(f"Eval:  {len(eval_ds)} examples")

    report_to = "none"
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity
        os.environ["WANDB_PROJECT"] = args.wandb_project
        report_to = "wandb"

    config = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        beta=args.beta,
        bf16=True,
        max_length=args.max_length,
        logging_steps=10,
        save_steps=500,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=500 if eval_ds else None,
        report_to=report_to,
        run_name=args.run_name or f"dpo-{args.dataset}",
        remove_unused_columns=True,
    )

    trainer = DPOTrainer(
        model=args.model_name,
        ref_model=None,  # TRL auto-creates frozen copy
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    trainer.train()
    trainer.save_model(args.output_dir + "/final")
    print(f"Model saved to {args.output_dir}/final")


if __name__ == "__main__":
    main()
