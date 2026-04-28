import torch

from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


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
