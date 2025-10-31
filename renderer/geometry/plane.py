import numpy as np
from typing import Optional

from geometry.base import Geometry
from core.ray import Ray, Intersection
from core.math_utils import normalize
from materials.base import Material


class Plane(Geometry):
    def __init__(self, point: np.ndarray, normal: np.ndarray, material: Material):
        super().__init__(material)
        self.point = point
        self.normal = normalize(normal)

    def intersect(self, ray: Ray) -> Optional[Intersection]:
        denom = np.dot(self.normal, ray.direction)
        if abs(denom) < 1e-6:
            return None

        t = np.dot(self.point - ray.origin, self.normal) / denom
        if t < 0.001:
            return None

        point = ray.at(t)
        return Intersection(t, point, self.normal, self.material)
