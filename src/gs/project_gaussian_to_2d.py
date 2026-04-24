import torch


# from gs.camera import Camera
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from gs.gaussian import GaussianModel


class Camera:
    def __init__(self):
        self.fx = 500.0
        self.fy = 500.0
        self.cx = 256.0
        self.cy = 256.0

        self.R = torch.eye(3)
        # because the gaussians are centered around the origin
        # if the camera is at the center, many gaussians will be behind the camera
        self.t = torch.tensor([0.0, 0.0, 3.0])


def quaternion_to_rot(rot: torch.Tensor) -> torch.Tensor:
    """
    Converts unit quaternions (w,x,y,z) to rotation matrices.

    inputs:
        rot: (N, 4)
    outputs:
        rot matrix: (N, 3, 3)
    """
    w, x, y, z = rot.unbind(-1)

    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(-1, 3, 3)


def calculate_covariance(scale: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """
    Given scale and rotation from the gaussian, built the covariance matrix
        sum = (R.S) (R.S)^T = R.diag(S^2).(R^T)

    inputs:
        scale: (N, 3)
        rotation: (N, 4)
    outputs:
        covariance: (N, 3, 3)
    """
    # (N, 4) -> (N, 3, 3)
    R = quaternion_to_rot(rotation)

    # scale: (N, 3) -> (N, 3, 1)
    sigma = R @ torch.diag_embed(scale**2) @ R.transpose(-1, -2)
    return sigma


def project_gaussians(
    xyz: torch.Tensor,
    cov3d: torch.Tensor,
    camera: Camera,
):

    R_cam = camera.R  # (3, 3)
    t_cam = camera.t  # (3,)

    # transfer keypoints from world space to cam space
    p_cam = xyz @ R_cam.T + t_cam  # (N, 3)
    X, Y, Z = p_cam.unbind(-1)

    valid = Z > 0.1
    u = camera.fx * X / Z + camera.cx
    v = camera.fy * Y / Z + camera.cy
    mean2d = torch.stack([u, v], dim=-1)  # (N, 2)

    # also rotate world covariance into camera frame.
    # Broadcasting: (3,3) @ (N,3,3) @ (3,3)
    cov_cam = R_cam @ cov3d @ R_cam.T  # (N, 3, 3)

    zeros = torch.zeros_like(Z)
    J = torch.stack(
        [
            camera.fx / Z,
            zeros,
            -camera.fx * X / Z**2,
            zeros,
            camera.fy / Z,
            -camera.fy * Y / Z**2,
        ],
        dim=-1,
    ).reshape(-1, 2, 3)  # (N, 2, 3)

    cov2d = J @ cov_cam @ J.transpose(-1, -2)  # (N, 2, 2)

    return mean2d, cov2d, Z, valid


def plot_gaussians_depth(mean2d, cov2d, depth, valid, max_points=500):
    """
    Visualize 2D Gaussians as ellipses colored by depth.

    mean2d : (N, 2)
    cov2d  : (N, 2, 2)
    depth  : (N,)
    valid  : (N,)
    """

    # Keep only valid points
    mean2d = mean2d[valid]
    cov2d = cov2d[valid]
    depth = depth[valid]

    # Subsample if too many points
    if mean2d.shape[0] > max_points:
        idx = torch.randperm(mean2d.shape[0])[:max_points]
        mean2d = mean2d[idx]
        cov2d = cov2d[idx]
        depth = depth[idx]

    dnorm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    cmap = cm.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(mean2d.shape[0]):
        mu = mean2d[i]
        cov = cov2d[i]

        # Eigen decomposition
        eigvals, eigvecs = torch.linalg.eigh(cov)
        eigvals = torch.clamp(eigvals, min=1e-8)

        width = 2 * torch.sqrt(eigvals[1])
        height = 2 * torch.sqrt(eigvals[0])

        angle = torch.atan2(eigvecs[1, 1], eigvecs[0, 1]) * 180.0 / torch.pi

        color = cmap(dnorm[i].item())

        ellipse = Ellipse(
            xy=mu.detach().cpu().numpy(),
            width=width.item(),
            height=height.item(),
            angle=angle.item(),
            fill=False,
            edgecolor=color,
            linewidth=1.5,
        )
        ax.add_patch(ellipse)

    ax.set_aspect("equal")
    ax.set_title("Projected 2D Gaussians (Colored by Depth)")
    ax.set_xlim(0, 512)
    ax.set_ylim(512, 0)  # invert Y to match image coordinates
    plt.show()


def main():
    N = 1000
    model = GaussianModel(n_points=N)

    xyz = model.xyz
    # scale shape: (N, 3)
    scale = model.get_scale()
    # rotation shape: (N, 4)
    rotation = model.get_rotation()

    cov3d = calculate_covariance(scale=scale, rotation=rotation)
    camera = Camera()

    mean2d, cov2d, depth, valid = project_gaussians(xyz, cov3d, camera)
    plot_gaussians_depth(mean2d, cov2d, depth, valid)

    print("2D means:", mean2d.shape)  # (N, 2)
    print("2D cov:", cov2d.shape)  # (N, 2, 2)
    print("Depth:", depth.shape)  # (N,)
    print("Valid:", valid.shape)  # (N,)

    eigvals = torch.linalg.eigvals(cov2d[valid])
    print("Cov2D eigenvalues (real part):", eigvals.real.mean())

    near = cov2d[(depth < 1.0) & valid]
    far = cov2d[(depth > 4.0) & valid]

    print(near.mean(), far.mean())

    i = 0
    print("cov2d:", cov2d[i])
    eigvals, eigvecs = torch.linalg.eigh(cov2d[i])
    print("major axis direction:", eigvecs[:, 1])

    print("min depth:", depth.min().item())
    print("max depth:", depth.max().item())


if __name__ == "__main__":
    main()
