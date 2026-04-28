"""
convert each projected gaussian into pixel
evaluate the 2d gaussian pdf at each pixel
alpha blend the result into a framebuffer


for a single blob, a correct single gaussian should look like a blob
"""

import torch
from tqdm import tqdm


def rasterize_gaussians(
    mean2d,
    cov2d,
    colors,
    opacities,
    image_size=(512, 512),
):
    """
    mean2d    : (N, 2)           Gaussian center in pixel coords
    cov2d     : (N, 2, 2)        2D covariance from projection
    colors    : (N, 3)
    opacities : (N, 1)
    """
    eps = 1e-6
    H, W = image_size
    # final rgb image
    framebuffer = torch.zeros((H, W, 3), dtype=torch.float32)
    # how much opacity each pixel has
    alpha_buf = torch.zeros((H, W), dtype=torch.float32)

    # defines the shape and orientation of the ellipse
    inv_cov = torch.linalg.inv(cov2d)  # (N, 2, 2)
    for i in tqdm(range(mean2d.shape[0])):
        # all gaussian params
        mu = mean2d[i]  # center in pixel space
        sigma_inv = inv_cov[i]  # ellipse shape
        color = colors[i]  # RGB
        alpha = opacities[i].item()  # max opacity

        # principal axis length, covers 3 * sigma of Gaussians
        eigvals, _ = torch.linalg.eigh(cov2d[i])
        radius = 3 * torch.sqrt(eigvals.max())

        # square patch arround gaussian
        xmin = max(int(mu[0] - radius), 0)
        xmax = min(int(mu[0] + radius), W - 1)
        ymin = max(int(mu[1] - radius), 0)
        ymax = min(int(mu[1] + radius), H - 1)

        if xmin >= xmax or ymin >= ymax:
            continue

        xs = torch.arange(xmin, xmax + 1)
        ys = torch.arange(ymin, ymax + 1)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")

        # offset from center
        # for every pixel:
        # dx = [x - mu x, y - my y]
        dx = torch.stack([xx - mu[0], yy - mu[1]], dim=-1)

        # Mahalanobis distance
        # converts gaussian into ellipse in pixel space
        d = torch.einsum("...i,ij,...j->...", dx, sigma_inv, dx)

        w = torch.exp(-0.5 * d)
        a = alpha * w

        a_prev = alpha_buf[ymin : ymax + 1, xmin : xmax + 1]
        fb_prev = framebuffer[ymin : ymax + 1, xmin : xmax + 1]

        a_new = a + a_prev * (1 - a)
        mask = a_new > eps
        fb_new = torch.zeros_like(fb_prev)
        fb_new[mask] = (
            fb_prev[mask] * (a_prev[mask] * (1 - a[mask])).unsqueeze(-1)
            + (a[mask].unsqueeze(-1) * color)
        ) / a_new[mask].unsqueeze(-1)

        framebuffer[ymin : ymax + 1, xmin : xmax + 1][mask] = fb_new[mask]
        alpha_buf[ymin : ymax + 1, xmin : xmax + 1] = a_new

    return framebuffer
