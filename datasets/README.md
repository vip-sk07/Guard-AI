# Guard AI — Core Datasets Directory

**Department of Artificial Intelligence and Data Science**  
**Mepco Schlenk Engineering College**

This directory hosts the benchmark datasets used for training, preference optimization (DPO), adversarial evaluation, and alignment tax validation in Guard AI.

---

## Directory Overview

| Directory | Dataset | Purpose | Format | Samples |
|-----------|---------|---------|--------|---------|
| `hh_rlhf/` | **Anthropic HH-RLHF** | Preference optimization (DPO), helpfulness vs harmlessness trade-offs | `.jsonl` | 3,000 pairs |
| `pku_saferlhf/` | **PKU-SafeRLHF** | SLM safety classifier training across multi-turn safety boundaries | `.jsonl` | 3,000 samples |
| `advbench/` | **AdvBench Suite** | Adversarial vectors & behavioral compliance red-teaming | `.jsonl` | Benchmark suite |
| `chatbot_safety/`| **Chatbot Safety** | Stealth jailbreak, DAN, and roleplay prompt patterns for Semantic Cache | `.jsonl` | Benchmark suite |
| `mmlu/` | **MMLU Benchmark** | Linguistic intelligence & reasoning alignment-tax validation | `.jsonl` | 500 questions |

---

## Dataset Schema

### 1. `hh_rlhf/train_dpo.jsonl`
```json
{
  "prompt": "Human query...",
  "chosen": "Harmless and helpful assistant response...",
  "rejected": "Unsafe or unhelpful assistant response...",
  "source": "Anthropic/hh-rlhf"
}
```

### 2. `pku_saferlhf/train_safety.jsonl`
```json
{
  "prompt": "User query...",
  "response": "Model response...",
  "is_safe": true,
  "label": 1,
  "source": "PKU-Alignment/PKU-SafeRLHF"
}
```

### 3. `mmlu/mmlu_eval.jsonl`
```json
{
  "subject": "computer_security",
  "question": "What is ...?",
  "choices": ["A", "B", "C", "D"],
  "answer": 2,
  "source": "cais/mmlu"
}
```
