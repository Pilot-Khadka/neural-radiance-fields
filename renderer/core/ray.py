import numpy as np
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from materials.base import Material


@dataclass
class Ray:
    origin: np.ndarray
    direction: np.ndarray

    def at(self, t: float) -> np.ndarray:
        return self.origin + t * self.direction


@dataclass
class Intersection:
    t: float
    point: np.ndarray
    normal: np.ndarray
    material: "Material"

    def is_closer_than(self, other: Optional["Intersection"]) -> bool:
        return other is None or self.t < other.t
