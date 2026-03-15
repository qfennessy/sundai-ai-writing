"""DPO training on LitBench-Train preference pairs.

Trains directly on (prompt, chosen_story, rejected_story) triples without
a separate reward model. The model learns to produce text more like the
higher-upvoted story and less like the lower-upvoted one.

Usage:
    # Single GPU
    uv run python train_dpo.py

    # Multi-GPU
    uv run accelerate launch train_dpo.py

    # Override defaults
    uv run python train_dpo.py \
        --model_name Qwen/Qwen2.5-3B-Instruct \
        --beta 0.1 \
        --learning_rate 5e-7 \
        --num_train_epochs 1 \
        --output_dir ./checkpoints/dpo-prose-v1
"""

import argparse

from datasets import load_dataset
from trl import DPOConfig, DPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="DPO training on LitBench-Train")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints/dpo-prose-v1",
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
        help="Limit dataset size (0 = use all 43k pairs)",
    )
    parser.add_argument(
        "--eval_split",
        type=float,
        default=0.02,
        help="Fraction of data to hold out for eval",
    )
    return parser.parse_args()


SYSTEM_PROMPT = (
    "You are a skilled creative writer. "
    "Write vivid, emotionally resonant stories with strong voice and original detail. "
    "Avoid clichés, purple prose, and AI-sounding phrasing."
)


def format_as_chat(prompt_text: str, story: str) -> list[dict]:
    """Format a prompt+story as chat messages for the tokenizer."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": story},
    ]


def prepare_dataset(max_examples: int = 0, eval_split: float = 0.02):
    """Load LitBench-Train and reshape for TRL's DPOTrainer.

    TRL expects columns: prompt, chosen, rejected
    where chosen/rejected are either strings or list[dict] chat messages.
    """
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

    if eval_split > 0:
        split = ds.train_test_split(test_size=eval_split, seed=42)
        return split["train"], split["test"]

    return ds, None


def main():
    args = parse_args()

    train_ds, eval_ds = prepare_dataset(
        max_examples=args.max_examples,
        eval_split=args.eval_split,
    )
    print(f"Train: {len(train_ds)} examples")
    if eval_ds:
        print(f"Eval:  {len(eval_ds)} examples")

    config = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        beta=args.beta,
        bf16=True,
        max_length=args.max_length,
        logging_steps=50,
        save_steps=500,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=500 if eval_ds else None,
        report_to="none",
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
