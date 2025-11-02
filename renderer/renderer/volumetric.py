import numpy as np
from tqdm import tqdm
from core.ray import Ray


class VolumetricRenderer:
    def __init__(
        self, width, height, num_samples=64, step_size=0.05, background_color=None
    ):
        self.width = width
        self.height = height
        self.num_samples = num_samples
        self.step_size = step_size
        self.background_color = (
            background_color
            if background_color is not None
            else np.array([0.1, 0.1, 0.15])
        )

    def render(self, scene, camera):
        image = np.zeros((self.height, self.width, 3))

        for y in tqdm(range(self.height), desc="Rendering"):
            for x in range(self.width):
                u = (x + 0.5) / self.width
                v = (y + 0.5) / self.height
                ray = camera.get_ray(u, v)
                # ray = camera.get_ray(x, y)
                color = self.trace_volumetric_ray(ray, scene)
                image[y, x] = np.clip(color, 0, 1)

        return image

    def trace_volumetric_ray(self, ray: Ray, scene):
        color = np.zeros(3)
        transmittance = 1.0

        t_min, t_max = self.find_ray_bounds(ray, scene)

        if t_min >= t_max:
            return self.background_color

        t = t_min
        while t < t_max and transmittance > 0.01:
            sample_pos = ray.at(t)

            density = 0.0
            local_color = np.zeros(3)
            total_scattering = 0.0

            for obj in scene.objects:
                if hasattr(obj, "contains") and obj.contains(sample_pos):
                    obj_density = obj.material.density
                    density += obj_density
                    local_color += obj.material.color * obj_density
                    total_scattering += obj.material.scattering * obj_density

            if density > 0:
                local_color /= density
                avg_scattering = total_scattering / density

                # lighting = self.calculate_lighting(sample_pos, scene)
                lighting = 1.0

                sample_transmittance = np.exp(-density * self.step_size)
                alpha = 1.0 - sample_transmittance

                color += transmittance * alpha * local_color * lighting * avg_scattering
                transmittance *= sample_transmittance

            t += self.step_size

        color += transmittance * self.background_color

        return color

    def calculate_lighting(self, position, scene):
        total_light = np.zeros(3)

        for light in scene.lights:
            light_dir = -light.direction / np.linalg.norm(light.direction)

            shadow_transmittance = self.compute_shadow_transmittance(
                position, light_dir, scene
            )

            total_light += (
                light.color
                * shadow_transmittance
                * max(0.0, np.dot(light_dir, np.array([0, 1, 0])))
            )

        ambient = 0.3
        total_light += ambient

        return np.clip(total_light, 0, 2.0)

    def compute_shadow_transmittance(self, position, light_dir, scene, max_dist=10.0):
        transmittance = 1.0
        t = 0.05

        shadow_ray = Ray(position, light_dir)

        while t < max_dist and transmittance > 0.01:
            sample_pos = shadow_ray.at(t)

            # density = 0.0
            density = 0.2
            for obj in scene.objects:
                if hasattr(obj, "contains") and obj.contains(sample_pos):
                    density += obj.material.density

            transmittance *= np.exp(-density * self.step_size)
            t += self.step_size * 2

        return transmittance

    def find_ray_bounds(self, ray: Ray, scene):
        t_min = float("inf")
        t_max = 0.0
        # hit = False

        for obj in scene.objects:
            if hasattr(obj, "get_volume_bounds"):
                obj_t_min, obj_t_max = obj.get_volume_bounds(ray)
                if (
                    np.isfinite(obj_t_min)
                    and np.isfinite(obj_t_max)
                    and obj_t_min < obj_t_max
                ):
                    t_min = min(t_min, obj_t_min)
                    t_max = max(t_max, obj_t_max)
                    # hit = True
                    # print(f"hit {obj}: t_min={obj_t_min}, t_max={obj_t_max}")

        # if not hit:
        #     print("no intersections")

        if t_min == float("inf"):
            return 0.0, 0.0

        return max(t_min, 0.0), t_max
