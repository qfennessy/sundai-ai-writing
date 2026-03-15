"""Extract good (human) and bad (AI-sloppy) samples from unslop-good-train.jsonl.

Each record has:
  - user message: AI-generated sloppy prose (preceded by an instruction like
    "Polish this AI passage to feel more human:")
  - assistant message: the original human-written text

Outputs:
  data/unslop-good.jsonl   — human-written passages (good samples)
  data/unslop-bad.jsonl    — AI-sloppy passages (bad samples)
  data/unslop-dpo.jsonl    — DPO-ready triples: {prompt, chosen, rejected}
"""

import json
import re
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_FILE = DATA_DIR / "unslop-good-train.jsonl"

# Instruction prefixes to strip from the user message to get the raw AI text
INSTRUCTION_PATTERN = re.compile(
    r"^(Polish|Rephrase|Rewrite|Make|Transform|Revise|Edit|Refine|Convert|Humanize)"
    r".*?:\n",
    re.IGNORECASE,
)


def split_instruction_and_text(user_content: str) -> tuple[str, str]:
    """Split user message into instruction and AI-generated passage.

    Returns (instruction, ai_text). The instruction is the first line
    (e.g. "Polish this AI passage to feel more human:"), and the AI text
    is everything after it.
    """
    parts = user_content.split("\n", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    # Fallback: try regex split
    match = INSTRUCTION_PATTERN.match(user_content)
    if match:
        return match.group(0).strip().rstrip(":"), user_content[match.end():].strip()
    return "Rewrite this passage to sound more natural:", user_content.strip()


def main():
    with open(INPUT_FILE) as f:
        records = [json.loads(line) for line in f]

    good_samples = []
    bad_samples = []
    dpo_triples = []

    for record in records:
        messages = record["messages"]
        user_content = messages[0]["content"]
        assistant_content = messages[1]["content"]

        instruction, bad_text = split_instruction_and_text(user_content)
        good_text = assistant_content

        good_samples.append({"text": good_text})
        bad_samples.append({"text": bad_text})

        # DPO triple: AI text as prompt, human version as chosen, AI slop as rejected
        dpo_triples.append({
            "prompt": bad_text,
            "chosen": good_text,
            "rejected": bad_text,
        })

    # Write outputs
    for filename, data in [
        ("unslop-good.jsonl", good_samples),
        ("unslop-bad.jsonl", bad_samples),
        ("unslop-dpo.jsonl", dpo_triples),
    ]:
        output_path = DATA_DIR / filename
        with open(output_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        print(f"Wrote {len(data)} records to {output_path}")

    # Print stats
    good_lens = [len(s["text"].split()) for s in good_samples]
    bad_lens = [len(s["text"].split()) for s in bad_samples]
    print(f"\nGood samples: avg {sum(good_lens)/len(good_lens):.0f} words, "
          f"min {min(good_lens)}, max {max(good_lens)}")
    print(f"Bad samples:  avg {sum(bad_lens)/len(bad_lens):.0f} words, "
          f"min {min(bad_lens)}, max {max(bad_lens)}")


if __name__ == "__main__":
    main()
