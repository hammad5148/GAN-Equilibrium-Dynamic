"""DCGAN-style models used by the notebook experiments."""

import torch
from torch import nn
from torch.nn.utils import spectral_norm


class Generator(nn.Module):
    """Generator supporting 28x28 grayscale and 32x32 RGB images."""

    def __init__(self, latent_dim: int = 100, hidden_dim: int = 64,
                 out_channels: int = 1, img_size: int = 28) -> None:
        super().__init__()
        if img_size not in (28, 32):
            raise ValueError("Generator currently supports img_size 28 or 32")
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.init_size = img_size // 4
        projection_features = hidden_dim * 4 * self.init_size ** 2
        self.project = nn.Sequential(
            nn.Linear(latent_dim, projection_features),
            nn.BatchNorm1d(projection_features),
            nn.ReLU(inplace=True),
        )
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 3, 1, 1, bias=False),
            nn.Tanh(),
        )
        self.apply(_init_weights)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.project(z)
        x = x.view(x.size(0), -1, self.init_size, self.init_size)
        return self.conv_blocks(x)


class Discriminator(nn.Module):
    """DCGAN discriminator with sigmoid or raw critic output."""

    def __init__(self, in_channels: int = 1, hidden_dim: int = 64,
                 output_mode: str = "vanilla", use_spectral_norm: bool = False,
                 img_size: int = 28) -> None:
        super().__init__()
        if output_mode not in {"vanilla", "wgan"}:
            raise ValueError("output_mode must be 'vanilla' or 'wgan'")
        if img_size not in (28, 32):
            raise ValueError("Discriminator currently supports img_size 28 or 32")
        self.output_mode = output_mode
        final_size = img_size // 8

        def maybe_sn(layer: nn.Module) -> nn.Module:
            return spectral_norm(layer) if use_spectral_norm else layer

        self.conv_blocks = nn.Sequential(
            maybe_sn(nn.Conv2d(in_channels, hidden_dim, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            maybe_sn(nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1, bias=False)),
            nn.Identity() if use_spectral_norm else nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            maybe_sn(nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, 2, 1, bias=False)),
            nn.Identity() if use_spectral_norm else nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            maybe_sn(nn.Linear(hidden_dim * 4 * final_size * final_size, 1)),
        )
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.conv_blocks(x))
        return torch.sigmoid(logits) if self.output_mode == "vanilla" else logits


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)
