import torch

from gan_game_theory.models import Discriminator, Generator
from gan_game_theory.training.losses import gradient_penalty


def test_models_support_fashion_mnist_shapes():
    generator = Generator(img_size=28)
    discriminator = Discriminator(img_size=28)
    fake = generator(torch.randn(4, 100))
    assert fake.shape == (4, 1, 28, 28)
    assert discriminator(fake).shape == (4, 1)


def test_models_support_cifar_shapes():
    generator = Generator(out_channels=3, img_size=32)
    discriminator = Discriminator(in_channels=3, img_size=32)
    fake = generator(torch.randn(4, 100))
    assert fake.shape == (4, 3, 32, 32)
    assert discriminator(fake).shape == (4, 1)


def test_gradient_penalty_is_scalar():
    generator = Generator(img_size=28)
    discriminator = Discriminator(img_size=28, output_mode="wgan")
    real = torch.randn(4, 1, 28, 28)
    fake = generator(torch.randn(4, 100)).detach()
    assert gradient_penalty(discriminator, real, fake).ndim == 0
