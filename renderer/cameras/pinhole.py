import numpy as np

from core.ray import Ray
from cameras.base import Camera
from core.math_utils import normalize


class PinholeCamera(Camera):
    def __init__(
        self,
        position: np.ndarray,
        look_at: np.ndarray,
        up: np.ndarray,
        fov: float,
        width: int,
        height: int,
    ):
        super().__init__(position, look_at, up, width, height)
        self.fov = fov
        self._setup_view()

    def _setup_view(self):
        theta = self.fov * np.pi / 180
        h = np.tan(theta / 2)
        viewport_height = 2.0 * h
        viewport_width = self.aspect * viewport_height

        w = normalize(self.position - self.look_at)
        u = normalize(np.cross(self.up, w))
        v = np.cross(w, u)

        self.horizontal = u * viewport_width
        self.vertical = v * viewport_height
        self.lower_left = self.position - self.horizontal / 2 - self.vertical / 2 - w

    def get_ray(self, u: float, v: float) -> "Ray":
        direction = (
            self.lower_left + u * self.horizontal + v * self.vertical - self.position
        )
        return Ray(self.position, normalize(direction))
