import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    eps = 1e-8
    norm = np.linalg.norm(v)
    return v / norm if norm > eps else v
