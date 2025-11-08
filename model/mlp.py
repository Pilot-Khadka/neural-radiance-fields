import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding:
    def __init__(self, num_freqs=10, include_input=True):
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.freq_bands = 2.0 ** torch.arange(num_freqs, dtype=torch.float32) * torch.pi

    def encode(self, x):
        out = [x] if self.include_input else []
        for freq in self.freq_bands:
            out.append(torch.sin(x * freq))
            out.append(torch.cos(x * freq))
        return torch.cat(out, dim=-1)


class NeRF(nn.Module):
    def __init__(self, pos_dim, dir_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(pos_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.sigma_out = nn.Linear(hidden_dim, 1)  # density

        self.feature = nn.Linear(hidden_dim, hidden_dim)
        self.fc_dir = nn.Linear(hidden_dim + dir_dim, hidden_dim)
        self.color_out = nn.Linear(hidden_dim, 3)  # RGB

    def forward(self, x, d):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        sigma = F.relu(self.sigma_out(h))

        feat = self.feature(h)
        h_dir = torch.cat([feat, d], dim=-1)
        h_dir = F.relu(self.fc_dir(h_dir))
        rgb = torch.sigmoid(self.color_out(h_dir))
        return rgb, sigma


if __name__ == "__main__":
    pos_enc = PositionalEncoding(num_freqs=10, include_input=True)
    dir_enc = PositionalEncoding(num_freqs=4, include_input=True)

    num_samples = 8
    positions = torch.rand(num_samples, 3) * 2 - 1  # range [-1, 1]
    directions = torch.randn(num_samples, 3)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)  # normalize

    x_encoded = pos_enc.encode(positions)
    d_encoded = dir_enc.encode(directions)

    model = NeRF(
        pos_dim=x_encoded.shape[-1], dir_dim=d_encoded.shape[-1], hidden_dim=128
    )

    rgb, sigma = model(x_encoded, d_encoded)

    print("RGB shape:", rgb.shape)  # [N, 3]
    print("Sigma shape:", sigma.shape)  # [N, 1]
    print("RGB sample:", rgb[0])
    print("Sigma sample:", sigma[0])
