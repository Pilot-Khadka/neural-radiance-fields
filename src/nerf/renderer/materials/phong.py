from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from ..materials.base import BaseMaterial
from ..core.math_utils import normalize

if TYPE_CHECKING:
    from ..core.ray import Intersection


class PhongMaterial(BaseMaterial):
    def __init__(
        self,
        color: np.ndarray,
        specular: float = 0.5,
        shininess: int = 32,
        ambient: float = 0.2,
    ):
        super().__init__(color)
        self.specular = specular
        self.shininess = shininess
        self.ambient = ambient

    def is_volume(self) -> bool:
        return False

    def shade(
        self,
        intersection: Intersection,
        light_dir: np.ndarray,
        view_dir: np.ndarray,
        light_color: np.ndarray,
    ) -> np.ndarray:
        ambient = self.color * self.ambient

        diffuse_strength = max(0, np.dot(intersection.normal, light_dir))
        diffuse = self.color * diffuse_strength

        reflect_dir = normalize(
            2 * np.dot(intersection.normal, light_dir) * intersection.normal - light_dir
        )
        spec_strength = max(0, np.dot(view_dir, reflect_dir)) ** self.shininess
        specular = light_color * self.specular * spec_strength

        return ambient + diffuse + specular
