# creative-writing

### Overview
- **Environment ID**: `creative-writing`
- **Description**: Scores LLM-generated stories using the LitBench Bradley-Terry reward model (SAA-Lab/Creative-Writing-Verifier)
- **Tags**: creative-writing, prose, train, eval

### Datasets
- **Primary dataset**: [SAA-Lab/LitBench-Train](https://huggingface.co/datasets/SAA-Lab/LitBench-Train) — 43,827 r/WritingPrompts preference pairs
- **Split sizes**: Extracts ~5,000 unique prompts by default

### Task
- **Type**: single-turn
- **Output format**: Free-form prose (300-600 words)
- **Rubric**: Weighted combination of BT reward model score (0.85) and length guard (0.15)

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_examples` | int | `5000` | Number of unique prompts to extract from LitBench-Train |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `prose_quality_reward` | BT reward model score, sigmoid-normalized to [0, 1] |
| `length_reward` | 1.0 if 100-800 words, 0.5 if >800, 0.0 if <100 |
| `reward` | Weighted sum: 0.85 * quality + 0.15 * length |
