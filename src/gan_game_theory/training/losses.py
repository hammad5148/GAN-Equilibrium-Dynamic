"""Vanilla GAN and WGAN-GP objectives."""

import torch
from torch import autograd, nn


def gradient_penalty(discriminator, real, fake) -> torch.Tensor:
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=real.device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    output = discriminator(interpolated)
    gradients = autograd.grad(output, interpolated, torch.ones_like(output),
                              create_graph=True, retain_graph=True)[0]
    return ((gradients.view(batch_size, -1).norm(2, dim=1) - 1) ** 2).mean()


def vanilla_loss(discriminator, fake, real) -> torch.Tensor:
    criterion = nn.BCELoss()
    real_labels = torch.ones(real.size(0), 1, device=real.device)
    fake_labels = torch.zeros(real.size(0), 1, device=real.device)
    return (criterion(discriminator(real), real_labels) +
            criterion(discriminator(fake), fake_labels)) / 2


def wgan_loss(discriminator, fake, real) -> torch.Tensor:
    return discriminator(fake).mean() - discriminator(real).mean()
