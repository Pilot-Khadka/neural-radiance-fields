import numpy as np
from typing import Optional

from geometry.base import Geometry
from core.ray import Ray, Intersection
from materials.base import BaseMaterial


class Sphere(Geometry):
    def __init__(self, center: np.ndarray, radius: float, material: BaseMaterial):
        super().__init__(material)
        self.center = center
        self.radius = radius

    def intersect(self, ray: Ray) -> Optional[Intersection]:
        oc = ray.origin - self.center

        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius * self.radius

        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            # no real roots -> ray misses the sphere
            return None

        # smallest root gives the nearest intersection along the ray
        t = (-b - np.sqrt(discriminant)) / (2.0 * a)
        if t < 0.001:
            # intersection behind the camera or too close to origin
            return None

        point = ray.at(t)
        normal = (point - self.center) / self.radius

        return Intersection(t, point, normal, self.material)

    def contains(self, point):
        return np.linalg.norm(point - self.center) <= self.radius

    def get_volume_bounds(self, ray: Ray):
        oc = ray.origin - self.center
        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius * self.radius
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            return float("inf"), float("-inf")

        sqrt_d = np.sqrt(discriminant)
        t1 = (-b - sqrt_d) / (2.0 * a)
        t2 = (-b + sqrt_d) / (2.0 * a)

        if t1 > t2:
            t1, t2 = t2, t1

        if t2 < 0:
            return float("inf"), float("-inf")

        t1 = max(t1, 0.0)
        return t1, t2

    def get_volume_bounds_array(self, origin, direction):
        oc = origin - self.center
        a = np.dot(direction, direction)
        b = 2.0 * np.dot(oc, direction)
        c = np.dot(oc, oc) - self.radius * self.radius
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            return float("inf"), float("-inf")

        sqrt_d = np.sqrt(discriminant)
        t1 = (-b - sqrt_d) / (2.0 * a)
        t2 = (-b + sqrt_d) / (2.0 * a)

        if t1 > t2:
            t1, t2 = t2, t1

        if t2 < 0:
            return float("inf"), float("-inf")

        t1 = max(t1, 0.0)
        return t1, t2
