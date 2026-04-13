import torch
from torch import nn


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


class MLP(nn.Module):
    def __init__(
        self,
        pos_dim: int,
        dir_dim: int,
        num_layers: int = 8,
        hidden_dim: int = 256,
        skip_layer: int = 4,
    ):
        super().__init__()
        self.skip_layer = skip_layer
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Linear(pos_dim, hidden_dim))
            elif i == skip_layer:
                self.layers.append(nn.Linear(hidden_dim + pos_dim, hidden_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.sigma_out = nn.Linear(hidden_dim, 1)
        self.feature = nn.Linear(hidden_dim, hidden_dim)
        self.fc_dir = nn.Linear(hidden_dim + dir_dim, hidden_dim // 2)
        self.color_out = nn.Linear(hidden_dim // 2, 3)

        self._init_weights()

    def _init_weights(self):
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
        for layer in [self.sigma_out, self.feature, self.fc_dir, self.color_out]:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, d):
        h = x
        for i, layer in enumerate(self.layers):
            if i == self.skip_layer:
                h = torch.cat([h, x], dim=-1)
            h = torch.relu(layer(h))

        sigma = torch.nn.functional.softplus(self.sigma_out(h))
        feat = self.feature(h)
        h_dir = torch.relu(self.fc_dir(torch.cat([feat, d], dim=-1)))
        rgb = torch.sigmoid(self.color_out(h_dir))
        return rgb, sigma


if __name__ == "__main__":
    pos_enc = PositionalEncoding(num_freqs=10, include_input=True)
    dir_enc = PositionalEncoding(num_freqs=4, include_input=True)

    num_samples = 8
    positions = torch.rand(num_samples, 3) * 2 - 1  # range [-1, 1]
    directions = torch.randn(num_samples, 3)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)

    x_encoded = pos_enc.encode(positions)
    d_encoded = dir_enc.encode(directions)

    model = MLP(
        pos_dim=x_encoded.shape[-1],
        dir_dim=d_encoded.shape[-1],
        hidden_dim=128,
    )

    rgb, sigma = model(x_encoded, d_encoded)

    print("RGB shape:", rgb.shape)  # [N, 3]
    print("Sigma shape:", sigma.shape)  # [N, 1]
    print("RGB sample:", rgb[0])
    print("Sigma sample:", sigma[0])
