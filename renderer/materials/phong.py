import numpy as np
from typing import TYPE_CHECKING

from materials.base import Material
from core.math_utils import normalize

if TYPE_CHECKING:
    from core.ray import Intersection


class PhongMaterial(Material):
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

    def shade(
        self,
        intersection: "Intersection",
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
