import numpy as np
from typing import Optional

from geometry.base import Geometry
from core.ray import Ray, Intersection
from materials.base import Material


class Sphere(Geometry):
    def __init__(self, center: np.ndarray, radius: float, material: Material):
        super().__init__(material)
        self.center = center
        self.radius = radius

    def intersect(self, ray: Ray) -> Optional[Intersection]:
        # vector from sphere center to ray origin
        oc = ray.origin - self.center

        # coeff of quadratic equation for ray-sphere intersection
        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius * self.radius

        # determine if the ray intersects the sphere
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            # no real roots -> ray misses the sphere
            return None

        # smallest root gives the nearest intersection along the ray
        t = (-b - np.sqrt(discriminant)) / (2.0 * a)
        if t < 0.001:
            # intersection behind the camera or too close to origin
            return None

        # intersection point and normal at that point
        point = ray.at(t)
        normal = (point - self.center) / self.radius

        return Intersection(t, point, normal, self.material)
