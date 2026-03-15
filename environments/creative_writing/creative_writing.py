"""Creative writing RL environment using LitBench reward model.

Feeds r/WritingPrompts prompts from LitBench-Train to a policy model,
scores generated stories with the SAA-Lab/Creative-Writing-Verifier
Bradley-Terry reward model, and returns normalized rewards for GRPO training.
"""

import verifiers as vf
from datasets import Dataset, load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# --- Load the LitBench reward model once at module level ---
RM_MODEL_NAME = "SAA-Lab/Creative-Writing-Verifier"


def _load_reward_model():
    """Lazy-load the reward model on first use."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(
        RM_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        num_labels=1,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(RM_MODEL_NAME)
    model.eval()
    return model, tokenizer, device


_rm_cache: dict = {}


def _get_reward_model():
    if "model" not in _rm_cache:
        model, tokenizer, device = _load_reward_model()
        _rm_cache["model"] = model
        _rm_cache["tokenizer"] = tokenizer
        _rm_cache["device"] = device
    return _rm_cache["model"], _rm_cache["tokenizer"], _rm_cache["device"]


def score_with_litbench_rm(prompt_text: str, story: str) -> float:
    """Score a story using the LitBench Bradley-Terry reward model.

    Returns a float in roughly [0, 1] via sigmoid normalization of the
    raw BT logit. Calibrate on a held-out set if the mean reward
    collapses to 0 or 1.
    """
    model, tokenizer, device = _get_reward_model()
    messages = [
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": story},
    ]
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        score = model(encoded).logits[0].item()
    return torch.sigmoid(torch.tensor(score)).item()


# --- Build dataset from LitBench-Train prompts ---
def build_dataset(num_examples: int = 5000, seed: int = 42) -> Dataset:
    """Extract unique writing prompts from LitBench-Train.

    LitBench-Train contains paired (chosen, rejected) stories per prompt.
    We extract unique prompts as RL training inputs — the policy generates
    stories during rollouts, scored by the BT reward model.
    """
    raw = load_dataset("SAA-Lab/LitBench-Train", split="train")
    raw = raw.shuffle(seed=seed)

    seen: set[str] = set()
    rows: list[dict] = []
    for ex in raw:
        prompt_text = ex["prompt"]
        if prompt_text not in seen:
            seen.add(prompt_text)
            rows.append({
                "question": (
                    "Write a short story (300-600 words) for this prompt:\n\n"
                    + prompt_text
                ),
                "answer": "",  # no ground truth — reward comes from RM
            })
        if len(rows) >= num_examples:
            break

    return Dataset.from_list(rows)


# --- Reward functions ---
def prose_quality_reward(completion, prompt, **kwargs) -> float:
    """Use the LitBench BT reward model to score the generated story."""
    # completion is Messages (list of chat dicts); extract last assistant turn
    if isinstance(completion, list):
        story = completion[-1]["content"]
    else:
        story = str(completion)

    # prompt is Messages; extract the user's writing prompt
    if isinstance(prompt, list):
        # Find last user message
        prompt_text = next(
            m["content"] for m in reversed(prompt) if m["role"] == "user"
        )
    else:
        prompt_text = str(prompt)

    return score_with_litbench_rm(prompt_text, story)


def length_reward(completion, **kwargs) -> float:
    """Penalize very short or excessively long outputs."""
    if isinstance(completion, list):
        story = completion[-1]["content"]
    else:
        story = str(completion)

    word_count = len(story.split())
    if word_count < 100:
        return 0.0
    elif word_count > 800:
        return 0.5
    return 1.0


# --- Wire it together ---
def load_environment(num_examples: int = 5000) -> vf.SingleTurnEnv:
    """Build the creative writing RL environment.

    Returns a SingleTurnEnv that:
    - Draws unique prompts from LitBench-Train
    - Scores stories with the BT reward model (weight 0.85)
    - Applies a length guard (weight 0.15)
    """
    rubric = vf.Rubric(
        funcs=[prose_quality_reward, length_reward],
        weights=[0.85, 0.15],
    )

    env = vf.SingleTurnEnv(
        dataset=lambda: build_dataset(num_examples),
        rubric=rubric,
        system_prompt=(
            "You are a skilled creative writer. "
            "Write vivid, emotionally resonant stories with strong voice "
            "and original detail. "
            "Avoid clichés, purple prose, and AI-sounding phrasing."
        ),
        env_id="creative_writing",
    )
    return env
