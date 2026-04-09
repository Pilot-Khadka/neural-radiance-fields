from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from abc import ABC, abstractmethod


if TYPE_CHECKING:
    from ..materials.base import BaseMaterial
    from ..core.ray import Intersection, Ray


class Geometry(ABC):
    def __init__(self, material: BaseMaterial):
        self.material = material

    @abstractmethod
    def intersect(self, ray: Ray) -> Optional[Intersection]:
        pass
