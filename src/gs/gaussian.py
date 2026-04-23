import torch
from torch import nn
import torch.nn.functional as F


class GaussianModel(nn.Module):
    """
    Blob in a space with shape, size and color (tiny 3d cloud)
    Each gaussian has:
        position: (x,y,z)
        covariance: (shape and size, can be stretched and rotated, ellipsoid)
        color: view dependent
        opacity/density
    """

    def __init__(self, n_points=2000):
        super().__init__()
        self.xyz = nn.Parameter(torch.rand(n_points, 3) * 2 - 1)

        # scale must always be positive
        # log(0.05) ~ -3.0, initial scale -> 0.05
        self.log_scale = nn.Parameter(torch.full((n_points, 3), -3.0))
        # in quaternion
        self.rotation = nn.Parameter(
            torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(n_points, 1)
        )
        # similar to scale, initial opacity 0.05
        self.raw_opacity = nn.Parameter(torch.full((n_points, 1), -3.0))
        self.raw_color = nn.Parameter(torch.zeros(n_points, 3))

    def get_scale(self):
        return torch.exp(self.log_scale)

    def get_rotation(self):
        # to unit quaternion
        return F.normalize(self.rotation, dim=-1)

    def get_opacity(self):
        return torch.sigmoid(self.raw_opacity)

    def get_color(self):
        return torch.sigmoid(self.raw_color)
