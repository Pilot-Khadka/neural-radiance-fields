import numpy as np
from dataclasses import dataclass

from core.math_utils import normalize


@dataclass
class Light:
    direction: np.ndarray
    color: np.ndarray

    def __post_init__(self):
        self.direction = normalize(self.direction)
