from __future__ import annotations

from dataclasses import dataclass
import numpy as np

@dataclass
class SteeringVector:
    layer_idx: int
    vector: np.ndarray
    strength: float = 1.0

class SteeringManager:
    def __init__(self, n_layers: int, hidden_size: int):
        self.n_layers = n_layers
        self.hidden_size = hidden_size

    def initialize_random_vector(self) -> SteeringVector:
        layer = np.random.randint(0, self.n_layers)
        vec = np.random.normal(0, 0.02, size=(self.hidden_size,))
        return SteeringVector(layer_idx=layer, vector=vec, strength=1.0)

    def mutate_vector(self, sv: SteeringVector, mutation_rate: float) -> SteeringVector:
        noise = np.random.normal(0, mutation_rate, size=sv.vector.shape)
        new_vec = sv.vector + noise
        new_vec = new_vec / (np.linalg.norm(new_vec) + 1e-9)
        return SteeringVector(layer_idx=sv.layer_idx, vector=new_vec, strength=sv.strength)

    def apply_to_model(self, sv: SteeringVector, model_instance: any):

        pass
