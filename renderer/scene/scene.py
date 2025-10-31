import numpy as np
from typing import List, Optional

from core.ray import Ray, Intersection
from geometry.base import Geometry
from lighting.light import Light


class Scene:
    def __init__(self):
        self.objects: List[Geometry] = []
        self.lights: List["Light"] = []
        self.background_color = np.array([0.5, 0.7, 1.0])

    def add_object(self, obj: Geometry) -> "Scene":
        self.objects.append(obj)
        return self

    def add_light(self, light: "Light") -> "Scene":
        self.lights.append(light)
        return self

    def set_background(self, color: np.ndarray) -> "Scene":
        self.background_color = color
        return self

    def intersect(self, ray: Ray) -> Optional[Intersection]:
        """find closest intersection"""
        closest = None
        for obj in self.objects:
            hit = obj.intersect(ray)
            if hit and hit.is_closer_than(closest):
                closest = hit
        return closest
