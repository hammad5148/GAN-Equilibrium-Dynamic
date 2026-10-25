from .losses import gradient_penalty, vanilla_loss, wgan_loss
from .trainer import GANTrainer

__all__ = ["GANTrainer", "gradient_penalty", "vanilla_loss", "wgan_loss"]
