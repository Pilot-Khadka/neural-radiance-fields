import numpy as np
from tqdm import tqdm

from core.ray import Ray
from scene.scene import Scene
from cameras.base import Camera
from renderer.base import Renderer
from core.math_utils import normalize


class RayTracerRenderer(Renderer):
    def __init__(self, width: int, height: int, samples_per_pixel: int = 1):
        super().__init__(width, height)
        self.samples_per_pixel = samples_per_pixel

    def render(self, scene: Scene, camera: Camera) -> np.ndarray:
        image = np.zeros((self.height, self.width, 3))

        for j in tqdm(range(self.height), desc="Rendering"):
            for i in range(self.width):
                u = i / (self.width - 1)
                v = 1 - (j / (self.height - 1))

                ray = camera.get_ray(u, v)
                color = self._trace(ray, scene)
                image[j, i] = np.clip(color, 0, 1)

        return image

    def _trace(self, ray: Ray, scene: Scene) -> np.ndarray:
        hit = scene.intersect(ray)

        if hit is None:
            return scene.background_color

        color = np.zeros(3)
        view_dir = normalize(-ray.direction)

        for light in scene.lights:
            color += hit.material.shade(hit, light.direction, view_dir, light.color)

        return color
