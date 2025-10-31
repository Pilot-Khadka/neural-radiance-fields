import numpy as np
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.ray import Ray


class Camera(ABC):
    def __init__(
        self,
        position: np.ndarray,
        look_at: np.ndarray,
        up: np.ndarray,
        width: int,
        height: int,
    ):
        self.position = position
        self.look_at = look_at
        self.up = up
        self.width = width
        self.height = height
        self.aspect = width / height

    @abstractmethod
    def get_ray(self, u: float, v: float) -> "Ray":
        pass
