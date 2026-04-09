from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

import numpy as np


if TYPE_CHECKING:
    from ..scene.scene import Scene
    from ..cameras.base import Camera


class Renderer(ABC):
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    @abstractmethod
    def render(self, scene: "Scene", camera: "Camera") -> np.ndarray:
        pass
