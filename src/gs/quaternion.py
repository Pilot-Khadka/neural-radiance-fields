import torch


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
