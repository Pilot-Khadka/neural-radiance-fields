import torch
import matplotlib.pyplot as plt

from gs.camera import SimpleCamera
from gs.gaussian import GaussianModel
from gs.project_gaussian_to_2d import project_gaussians
from gs.project_gaussian_to_2d import calculate_covariance
from gs.rasterizer import rasterize_gaussians


def debug_render(mean2d, cov2d, model):
    colors = model.get_color()
    opacities = model.get_opacity()

    img = rasterize_gaussians(
        mean2d=mean2d,
        cov2d=cov2d,
        colors=colors,
        opacities=opacities,
        image_size=(512, 512),
    )

    plt.figure(figsize=(6, 6))
    plt.imshow(img.detach().numpy())
    plt.title("Rendered Gaussian Splatting")
    plt.axis("off")
    plt.show()


def main():
    N = 1
    model = GaussianModel(n_points=N)

    xyz = model.xyz
    scale = model.get_scale()
    rotation = model.get_rotation()

    cov3d = calculate_covariance(scale=scale, rotation=rotation)
    camera = SimpleCamera()
    mean2d = torch.tensor([[64.0, 64.0]])
    cov2d = torch.tensor([[[200.0, 0.0], [0.0, 200.0]]])
    colors = torch.tensor([[1.0, 0.0, 0.0]])
    opacities = torch.tensor([[1.0]])

    # mean2d, cov2d, depth, valid = project_gaussians(xyz, cov3d, camera)
    print("mean:", mean2d)
    # debug_render(mean2d[valid], cov2d[valid], model)
    debug_render(mean2d, cov2d, model)


if __name__ == "__main__":
    main()
