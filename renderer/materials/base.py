import numpy as np
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.ray import Intersection


class BaseMaterial(ABC):
    def __init__(self, color: np.ndarray):
        self.color = np.array(color)

    @abstractmethod
    def is_volume(self) -> bool:
        pass


class Material(BaseMaterial):
    def __init__(self, color: np.ndarray):
        super().__init__(color)

    def is_volume(self) -> bool:
        return False

    @abstractmethod
    def shade(
        self,
        intersection: "Intersection",
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
