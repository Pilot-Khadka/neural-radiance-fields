import os
import json
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torch.utils.data import Dataset


class NeRFDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        json_file = os.path.join(root_dir, f"transforms_{split}.json")
        with open(json_file, "r") as f:
            meta = json.load(f)
        self.image_paths = [
            os.path.join(root_dir, frame["file_path"] + ".png")
            for frame in meta["frames"]
        ]
        self.poses = [
            torch.tensor(frame["transform_matrix"], dtype=torch.float32)
            for frame in meta["frames"]
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        pose = self.poses[idx]
        return img, pose


def draw_camera_frustum(ax, pose, scale=0.5, color="blue"):
    pose_np = pose.numpy() if torch.is_tensor(pose) else pose

    camera_pos = pose_np[:3, 3]
    R = pose_np[:3, :3]

    frustum_depth = scale
    frustum_width = scale * 0.6
    frustum_height = scale * 0.45

    corners = np.array(
        [
            [0, 0, 0],
            [-frustum_width, -frustum_height, -frustum_depth],
            [frustum_width, -frustum_height, -frustum_depth],
            [frustum_width, frustum_height, -frustum_depth],
            [-frustum_width, frustum_height, -frustum_depth],
        ]
    )

    corners_world = (R @ corners.T).T + camera_pos

    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]

    for start, end in edges:
        points = np.array([corners_world[start], corners_world[end]])
        ax.plot3D(*points.T, color=color, linewidth=1.5)

    return corners_world


def plot_camera_positions_3d(dataset, num_cameras=None, figsize=(15, 15)):
    if num_cameras is None:
        num_cameras = len(dataset)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    camera_positions = []

    for i in range(min(num_cameras, len(dataset))):
        _, pose = dataset[i]
        pose_np = pose.numpy()

        camera_pos = pose_np[:3, 3]
        camera_positions.append(camera_pos)

        draw_camera_frustum(ax, pose, scale=0.5, color="blue")

    camera_positions = np.array(camera_positions)

    ax.scatter(0, 0, 0, c="green", s=200, marker="*", label="Scene Center")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Camera Positions and Orientations")
    ax.legend()

    max_range = np.abs(camera_positions).max()
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    transform = T.Compose(
        [
            T.Resize((400, 400)),
            T.ToTensor(),
        ]
    )

    dataset = NeRFDataset(root_dir="lego", split="train", transform=transform)

    print(f"Loaded {len(dataset)} images with camera poses")

    fig1 = plot_camera_positions_3d(dataset, num_cameras=20)
    plt.savefig("camera_positions_3d.png", dpi=150, bbox_inches="tight")
    plt.show()
