import json
import math
from typing import Dict, List

def compute_dpo_loss(
    pi_win_logp: float,
    pi_lose_logp: float,
    ref_win_logp: float,
    ref_lose_logp: float,
    beta: float = 0.1,
) -> float:
    pi_logr = pi_win_logp - pi_lose_logp
    ref_logr = ref_win_logp - ref_lose_logp

    implicit_reward = beta * (pi_logr - ref_logr)

    if implicit_reward > 0:
        loss = math.log(1.0 + math.exp(-implicit_reward))
    else:
        loss = -implicit_reward + math.log(1.0 + math.exp(implicit_reward))

    return round(float(loss), 6)

class DPODatasetManager:
    def __init__(self, export_path: str = "data/dpo_preferences.json"):
        self.export_path = export_path
        self.records: List[Dict] = []

    def add_preference(self, prompt: str, chosen: str, rejected: str, metadata: Dict = None):
        entry = {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "metadata": metadata or {},
        }
        self.records.append(entry)

    def save_to_disk(self):

        with open(self.export_path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2)
        return len(self.records)

    def generate_synthetic_logps(self, score_margin: float):
        base_logp = -12.50
        pi_win = base_logp + (score_margin * 1.8)
        pi_lose = base_logp - (score_margin * 1.5)
        ref_win = base_logp + 0.2
        ref_lose = base_logp - 0.2
        return pi_win, pi_lose, ref_win, ref_lose
