import numpy as np
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scene.scene import Scene
    from cameras.base import Camera


class Renderer(ABC):
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    @abstractmethod
    def render(self, scene: "Scene", camera: "Camera") -> np.ndarray:
        pass
