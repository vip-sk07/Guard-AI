import json
import os
import re
from pathlib import Path
from datasets import load_dataset

DATASETS_DIR = Path(__file__).parent / "datasets"

def clean_text(t):
    if not isinstance(t, str):
        return ""
    return re.sub(r"\s+", " ", t).strip()

def prepare_hh_rlhf(limit=3000):
    print(f"[HH-RLHF] Streaming up to {limit} pairs from Anthropic/hh-rlhf...")
    target_file = DATASETS_DIR / "hh_rlhf" / "train_dpo.jsonl"
    count = 0
    try:
        ds = load_dataset("Anthropic/hh-rlhf", split="train", streaming=True)
        with open(target_file, "w", encoding="utf-8") as f:
            for row in ds:
                if count >= limit:
                    break
                chosen = clean_text(row.get("chosen", ""))
                rejected = clean_text(row.get("rejected", ""))

                m = re.search(r"Human:(.*?)Assistant:", chosen, re.DOTALL)
                prompt = m.group(1).strip() if m else chosen[:300]

                m_c = re.search(r"Assistant:(.*?)$", chosen, re.DOTALL)
                m_r = re.search(r"Assistant:(.*?)$", rejected, re.DOTALL)
                resp_c = m_c.group(1).strip() if m_c else chosen[-200:]
                resp_r = m_r.group(1).strip() if m_r else rejected[-200:]

                if len(prompt) > 10 and len(resp_c) > 10 and len(resp_r) > 10:
                    record = {
                        "prompt": prompt,
                        "chosen": resp_c,
                        "rejected": resp_r,
                        "source": "Anthropic/hh-rlhf"
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
        print(f"[HH-RLHF] Saved {count} samples to {target_file}")
    except Exception as e:
        print(f"[HH-RLHF] Error downloading: {e}")

def prepare_pku_saferlhf(limit=3000):
    print(f"[PKU-SafeRLHF] Streaming up to {limit} samples from PKU-Alignment/PKU-SafeRLHF...")
    target_file = DATASETS_DIR / "pku_saferlhf" / "train_safety.jsonl"
    count = 0
    try:
        ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train", streaming=True)
        with open(target_file, "w", encoding="utf-8") as f:
            for row in ds:
                if count >= limit:
                    break
                prompt = clean_text(row.get("prompt", ""))
                for i in range(2):
                    resp = clean_text(row.get(f"response_{i}", ""))
                    is_safe = row.get(f"is_response_{i}_safe", True)
                    if resp:
                        record = {
                            "prompt": prompt,
                            "response": resp,
                            "is_safe": bool(is_safe),
                            "label": 1 if is_safe else 0,
                            "source": "PKU-Alignment/PKU-SafeRLHF"
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        count += 1
                        if count >= limit:
                            break
        print(f"[PKU-SafeRLHF] Saved {count} samples to {target_file}")
    except Exception as e:
        print(f"[PKU-SafeRLHF] Error downloading: {e}")

def prepare_mmlu(limit=500):
    print(f"[MMLU] Downloading up to {limit} questions from cais/mmlu...")
    target_file = DATASETS_DIR / "mmlu" / "mmlu_eval.jsonl"
    count = 0
    subjects = [
        "computer_security", "abstract_algebra", "high_school_mathematics",
        "machine_learning", "logical_fallacies", "professional_law"
    ]
    try:
        with open(target_file, "w", encoding="utf-8") as f:
            for subj in subjects:
                if count >= limit:
                    break
                ds = load_dataset("cais/mmlu", subj, split="test")
                for row in ds:
                    if count >= limit:
                        break
                    record = {
                        "subject": subj,
                        "question": row["question"],
                        "choices": row["choices"],
                        "answer": row["answer"],
                        "source": "cais/mmlu"
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
        print(f"[MMLU] Saved {count} questions to {target_file}")
    except Exception as e:
        print(f"[MMLU] Error downloading: {e}")

def prepare_adversarial_datasets():
    print("[Adversarial Datasets] Compiling AdvBench and Chatbot Safety benchmarks...")
    from data.benchmark_loader import RED_TEAM_PROMPTS

    adv_file = DATASETS_DIR / "advbench" / "adversarial_behaviors.jsonl"
    cs_file = DATASETS_DIR / "chatbot_safety" / "stealth_jailbreak_prompts.jsonl"

    with open(adv_file, "w", encoding="utf-8") as f_adv, open(cs_file, "w", encoding="utf-8") as f_cs:
        for p in RED_TEAM_PROMPTS:
            rec = {
                "id": p.id,
                "text": p.text,
                "category": p.category.value if hasattr(p.category, "value") else str(p.category),
                "severity": p.severity.value if hasattr(p.severity, "value") else str(p.severity),
                "source": "RedTeamBenchmark"
            }
            if "jailbreak" in rec["category"].lower() or "roleplay" in rec["category"].lower() or "prompt_injection" in rec["category"].lower():
                f_cs.write(json.dumps(rec, ensure_ascii=False) + "\n")
            else:
                f_adv.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("[Adversarial Datasets] Written benchmark samples to advbench/ and chatbot_safety/")

if __name__ == "__main__":
    print("Preparing datasets in:", DATASETS_DIR)
    prepare_hh_rlhf(limit=3000)
    prepare_pku_saferlhf(limit=3000)
    prepare_mmlu(limit=500)
    prepare_adversarial_datasets()
    print("All datasets prepared successfully!")
