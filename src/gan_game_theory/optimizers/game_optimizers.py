"""Optimizers used to compare game-solving dynamics."""

import torch
from torch import optim


class OptimisticAdam(optim.Adam):
    """Adam with the first-order optimistic correction from the notebook."""

    def __init__(self, params, lr=2e-4, betas=(0.5, 0.999), eps=1e-8):
        super().__init__(params, lr=lr, betas=betas, eps=eps)
        self._previous_gradients: dict[int, torch.Tensor] = {}

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                key = id(parameter)
                current = parameter.grad.detach().clone()
                if key in self._previous_gradients:
                    parameter.grad.copy_(2.0 * current - self._previous_gradients[key])
                self._previous_gradients[key] = current
        return super().step(closure)


class NegativeMomentumSGD(optim.SGD):
    """SGD whose velocity uses a negative momentum coefficient."""

    def __init__(self, params, lr=1e-3, momentum=-0.1):
        super().__init__(params, lr=lr, momentum=0.0)
        self.negative_beta = momentum
        self._velocity: dict[int, torch.Tensor] = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                key = id(parameter)
                velocity = self._velocity.setdefault(key, torch.zeros_like(parameter))
                velocity.mul_(self.negative_beta).add_(parameter.grad)
                parameter.add_(velocity, alpha=-group["lr"])
        return loss


def build_optimizers(generator, discriminator, name: str, config):
    name = name.lower()
    if name == "adam":
        factory = lambda params, lr: optim.Adam(
            params, lr=lr, betas=(config.beta1, config.beta2))
        return factory(generator.parameters(), config.generator_lr), factory(
            discriminator.parameters(), config.discriminator_lr)
    if name == "optimistic_adam":
        return (
            OptimisticAdam(generator.parameters(), config.generator_lr,
                           (config.beta1, config.beta2)),
            OptimisticAdam(discriminator.parameters(), config.discriminator_lr,
                           (config.beta1, config.beta2)),
        )
    if name == "negative_momentum":
        return (
            NegativeMomentumSGD(generator.parameters(), config.negative_momentum_lr,
                                config.negative_momentum_beta),
            NegativeMomentumSGD(discriminator.parameters(), config.negative_momentum_lr,
                                config.negative_momentum_beta),
        )
    raise ValueError(f"Unknown optimizer: {name}")
