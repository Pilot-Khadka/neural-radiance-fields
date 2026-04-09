from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

import numpy as np


if TYPE_CHECKING:
    from ..core.ray import Intersection


class BaseMaterial(ABC):
    def __init__(self, color: np.ndarray):
        self.color = np.array(color)

    @abstractmethod
    def is_volume(self) -> bool:
        pass

    @abstractmethod
    def shade(
        self,
        intersection: Intersection,
        light_dir: np.ndarray,
        view_dir: np.ndarray,
        light_color: np.ndarray,
    ) -> np.ndarray:
        pass


class VolumeMaterial(BaseMaterial):
    def __init__(self, color, density, absorption, scattering):
        super().__init__(color)
        self.density = density
        self.absorption = np.array(absorption)
        self.scattering = scattering

    def is_volume(self) -> bool:
        return True

    def shade(self, intersection, light_dir, view_dir, light_color):
        """Simple placeholder."""
        return self.color * self.scattering
