import numpy as np
from numba import jit
from tqdm import tqdm


@jit(nopython=True)
def sample_density_at_point(point, centers, radii, densities):
    """check all volumes and sum their densities at this point."""
    total_density = 0.0

    for i in range(len(centers)):
        distance = np.linalg.norm(point - centers[i])
        if distance <= radii[i]:
            total_density += densities[i]

    return total_density


@jit(nopython=True)
def calculate_shadow(start_pos, light_direction, centers, radii, densities, step_size):
    """march toward light to see how much is blocked by volumes."""
    opacity = 0.0
    distance = 0.0
    max_distance = 10.0

    while distance < max_distance:
        sample_point = start_pos + light_direction * distance
        density = sample_density_at_point(sample_point, centers, radii, densities)
        opacity += density * step_size

        if opacity > 5.0:
            break

        distance += step_size * 2

    return np.exp(-opacity)


@jit(nopython=True)
def calculate_lighting(
    position, light_dirs, light_colors, centers, radii, densities, step_size
):
    """sum light from all sources, accounting for shadows."""
    total_light = np.zeros(3)

    for i in range(len(light_dirs)):
        shadow_factor = calculate_shadow(
            position, light_dirs[i], centers, radii, densities, step_size
        )

        brightness = max(0.0, np.dot(light_dirs[i], np.array([0.0, 1.0, 0.0])))
        total_light += light_colors[i] * shadow_factor * brightness

    total_light += 0.3
    return np.clip(total_light, 0.0, 2.0)


@jit(nopython=True)
def march_ray(
    origin,
    direction,
    t_values,
    centers,
    radii,
    densities,
    colors,
    scatterings,
    light_dirs,
    light_colors,
    background,
    step_size,
):
    """walk along the ray, accumulating color from volumes."""
    accumulated_color = np.zeros(3)
    remaining_transmittance = 1.0

    for i in range(len(t_values)):
        if remaining_transmittance < 0.01:
            break

        sample_point = origin + direction * t_values[i]

        density = 0.0
        weighted_color = np.zeros(3)
        weighted_scattering = 0.0

        for obj_idx in range(len(centers)):
            distance = np.linalg.norm(sample_point - centers[obj_idx])
            if distance <= radii[obj_idx]:
                obj_density = densities[obj_idx]
                density += obj_density
                weighted_color += colors[obj_idx] * obj_density
                weighted_scattering += scatterings[obj_idx] * obj_density

        if density > 0:
            volume_color = weighted_color / density
            scattering = weighted_scattering / density

            lighting = calculate_lighting(
                sample_point,
                light_dirs,
                light_colors,
                centers,
                radii,
                densities,
                step_size,
            )

            absorption = np.exp(-density * step_size)
            contribution = 1.0 - absorption

            accumulated_color += (
                remaining_transmittance
                * contribution
                * volume_color
                * lighting
                * scattering
            )
            remaining_transmittance *= absorption

    accumulated_color += remaining_transmittance * background
    return accumulated_color


class VolumetricRenderer:
    """renders 3D volumes (clouds, smoke, fog) by marching rays through space."""

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
        rays_origins, rays_directions = camera.get_rays_batch(self.width, self.height)

        volume_data = self._extract_volumes(scene)
        light_data = self._extract_lights(scene)

        image = self._render_all_rays(
            rays_origins, rays_directions, volume_data, light_data
        )

        return np.clip(image, 0, 1)

    def _extract_volumes(self, scene):
        objects = []
        centers = []
        radii = []
        densities = []
        colors = []
        scatterings = []

        for obj in scene.objects:
            if hasattr(obj, "contains"):
                objects.append(obj)
                centers.append(obj.center)
                radii.append(obj.radius)
                densities.append(obj.material.density)
                colors.append(obj.material.color)
                scatterings.append(obj.material.scattering)

        return {
            "objects": objects,
            "centers": np.array(centers, dtype=np.float64),
            "radii": np.array(radii, dtype=np.float64),
            "densities": np.array(densities, dtype=np.float64),
            "colors": np.array(colors, dtype=np.float64),
            "scatterings": np.array(scatterings, dtype=np.float64),
        }

    def _extract_lights(self, scene):
        directions = []
        colors = []

        for light in scene.lights:
            normalized = -light.direction / np.linalg.norm(light.direction)
            directions.append(normalized)
            colors.append(light.color)

        return {
            "directions": np.array(directions, dtype=np.float64),
            "colors": np.array(colors, dtype=np.float64),
        }

    def _render_all_rays(self, origins, directions, volumes, lights):
        height, width = origins.shape[:2]
        image = np.zeros((height, width, 3))

        for y in tqdm(range(height), desc="Rendering"):
            for x in range(width):
                color = self._trace_single_ray(
                    origins[y, x], directions[y, x], volumes, lights
                )
                image[y, x] = color

        return image

    def _trace_single_ray(self, origin, direction, volumes, lights):
        """march a single ray through the scene."""
        start_t, end_t = self._find_ray_bounds(origin, direction, volumes)

        if start_t >= end_t:
            return self.background_color

        num_steps = int((end_t - start_t) / self.step_size) + 1
        sample_positions = np.linspace(start_t, end_t, num_steps)

        return march_ray(
            origin,
            direction,
            sample_positions,
            volumes["centers"],
            volumes["radii"],
            volumes["densities"],
            volumes["colors"],
            volumes["scatterings"],
            lights["directions"],
            lights["colors"],
            self.background_color,
            self.step_size,
        )

    def _find_ray_bounds(self, origin, direction, volumes):
        """find where ray enters and exits all volumes."""
        earliest_entry = float("inf")
        latest_exit = 0.0

        for obj in volumes["objects"]:
            if hasattr(obj, "get_volume_bounds_array"):
                entry, exit = obj.get_volume_bounds_array(origin, direction)

                if np.isfinite(entry) and np.isfinite(exit) and entry < exit:
                    earliest_entry = min(earliest_entry, entry)
                    latest_exit = max(latest_exit, exit)

        if earliest_entry == float("inf"):
            return 0.0, 0.0

        return max(earliest_entry, 0.0), latest_exit
