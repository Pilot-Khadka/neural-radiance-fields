import matplotlib.pyplot as plt
import torchvision.transforms as T

from nerf.data.dataloader import NeRFDataset, plot_camera_positions_3d


if __name__ == "__main__":
    from nerf.utils import get_data_path

    transform = T.Compose(
        [
            T.Resize((400, 400)),
            T.ToTensor(),
        ]
    )

    data_root = get_data_path() / "lego"
    dataset = NeRFDataset(root_dir=data_root, split="train", transform=transform)

    print(f"Loaded {len(dataset)} images with camera poses")

    fig1 = plot_camera_positions_3d(dataset, num_cameras=20)
    plt.savefig("camera_positions_3d.png", dpi=150, bbox_inches="tight")
    plt.show()
