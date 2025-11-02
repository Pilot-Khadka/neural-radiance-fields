import numpy as np
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.ray import Intersection


class Material(ABC):
    def __init__(self, color: np.ndarray):
        self.color = color

    @abstractmethod
    def shade(
        self,
        intersection: "Intersection",
        light_dir: np.ndarray,
        view_dir: np.ndarray,
        light_color: np.ndarray,
    ) -> np.ndarray:
        pass


class VolumeMaterial(ABC):
    def __init__(self, color, density, absorption, scattering):
        self.color = np.array(color)
        self.density = density
        self.absorption = np.array(absorption)
        self.scattering = scattering
