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
        theta = np.radians(self.fov)
        h = np.tan(theta / 2)
        viewport_height = 2.0 * h
        viewport_width = self.aspect * viewport_height

        forward = normalize(self.look_at - self.position)
        right = normalize(np.cross(forward, self.up))
        up = np.cross(right, forward)

        self.horizontal = right * viewport_width
        self.vertical = up * viewport_height
        self.lower_left = (
            self.position - self.horizontal / 2 - self.vertical / 2 + forward
        )

    def get_ray(self, u: float, v: float) -> "Ray":
        direction = (
            self.lower_left + u * self.horizontal + v * self.vertical - self.position
        )
        return Ray(self.position, normalize(direction))

    def get_rays_batch(self, width, height):
        u = (np.arange(width) + 0.5) / width
        v = (np.arange(height) + 0.5) / height

        u_grid, v_grid = np.meshgrid(u, v)

        rays_d = (
            self.lower_left[None, None, :]
            + u_grid[:, :, None] * self.horizontal[None, None, :]
            + v_grid[:, :, None] * self.vertical[None, None, :]
            - self.position[None, None, :]
        )

        rays_d = rays_d / np.linalg.norm(rays_d, axis=2, keepdims=True)

        rays_o = np.broadcast_to(
            self.position[None, None, :], (height, width, 3)
        ).copy()

        return rays_o, rays_d
