"""Unified training loop for the notebook's six factorial configurations."""

from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torchvision.utils import make_grid, save_image

from ..diagnostics import (
    DiscriminatorBalanceTracker,
    EquilibriumHealthScore,
    GradientStabilityTracker,
    JacobianSpectrumTracker,
)
from .losses import gradient_penalty, vanilla_loss, wgan_loss


class GANTrainer:
    def __init__(self, generator, discriminator, opt_generator, opt_discriminator,
                 device, loss_type: str, model_config, training_config,
                 diagnostic_config, mode_tracker=None, output_dir: str | Path | None = None):
        if loss_type not in {"vanilla", "wgan_gp"}:
            raise ValueError("loss_type must be 'vanilla' or 'wgan_gp'")
        self.generator = generator
        self.discriminator = discriminator
        self.opt_generator = opt_generator
        self.opt_discriminator = opt_discriminator
        self.device = device
        self.loss_type = loss_type
        self.model_config = model_config
        self.training_config = training_config
        self.diagnostic_config = diagnostic_config
        self.mode_tracker = mode_tracker
        self.output_dir = Path(output_dir) if output_dir else None
        self.balance_tracker = DiscriminatorBalanceTracker()
        self.gradient_tracker = GradientStabilityTracker(diagnostic_config.gradient_window)
        self.jacobian_tracker = JacobianSpectrumTracker(
            diagnostic_config.jacobian_probes, diagnostic_config.spectral_tau)
        self.ehs_tracker = EquilibriumHealthScore(
            diagnostic_config.ehs_weights, diagnostic_config.spectral_tau)
        self.metrics_history = defaultdict(list)
        self._last_spectral = {"spectral_radius": 0.0,
                               "max_imag_real_ratio": 0.0,
                               "spectral_health": 1.0}

    def train_epoch(self, loader, epoch: int) -> dict:
        self.generator.train()
        self.discriminator.train()
        bce = nn.BCELoss()
        g_loss_total = d_loss_total = wasserstein_total = penalty_total = 0.0
        real_outputs, fake_outputs = [], []
        generator_steps = discriminator_steps = 0

        for batch_index, (real_images, _) in enumerate(loader):
            real = real_images.to(self.device, non_blocking=True)
            batch_size = real.size(0)

            self.opt_discriminator.zero_grad(set_to_none=True)
            noise = torch.randn(batch_size, self.model_config.latent_dim, device=self.device)
            fake = self.generator(noise).detach()
            real_output = self.discriminator(real)
            fake_output = self.discriminator(fake)
            penalty = torch.tensor(0.0, device=self.device)
            if self.loss_type == "vanilla":
                labels_real = torch.ones(batch_size, 1, device=self.device)
                labels_fake = torch.zeros(batch_size, 1, device=self.device)
                d_loss = (bce(real_output, labels_real) +
                          bce(fake_output, labels_fake)) / 2
            else:
                penalty = gradient_penalty(self.discriminator, real, fake)
                d_loss = (fake_output.mean() - real_output.mean() +
                          self.training_config.gp_lambda * penalty)
            d_loss.backward()
            self.opt_discriminator.step()
            real_outputs.append(real_output.detach().cpu())
            fake_outputs.append(fake_output.detach().cpu())
            d_loss_total += d_loss.item()
            penalty_total += penalty.item()
            if self.loss_type == "wgan_gp":
                wasserstein_total += (real_output.mean() - fake_output.mean()).item()
            discriminator_steps += 1

            is_generator_step = (self.loss_type == "vanilla" or
                                 batch_index % self.training_config.n_critic == 0)
            if is_generator_step:
                self.opt_generator.zero_grad(set_to_none=True)
                noise = torch.randn(batch_size, self.model_config.latent_dim,
                                    device=self.device)
                generated = self.generator(noise)
                if self.loss_type == "vanilla":
                    labels = torch.ones(batch_size, 1, device=self.device)
                    g_loss = bce(self.discriminator(generated), labels)
                else:
                    g_loss = -self.discriminator(generated).mean()
                g_loss.backward()
                self.opt_generator.step()
                g_loss_total += g_loss.item()
                generator_steps += 1

        self.gradient_tracker.record(self.generator, self.discriminator)
        balance = self.balance_tracker.compute(
            torch.cat(real_outputs).flatten(), torch.cat(fake_outputs).flatten(),
            raw_critic=self.loss_type == "wgan_gp")
        stability = self.gradient_tracker.compute()
        if self.mode_tracker:
            modes = self.mode_tracker.compute(self.generator)
        else:
            modes = {"mode_coverage": 0.0, "num_modes": 0, "entropy": 0.0,
                     "class_distribution": []}

        if epoch % self.diagnostic_config.jacobian_every == 0:
            real_probe = next(iter(loader))[0][:32].to(self.device)
            noise_probe = torch.randn(real_probe.size(0), self.model_config.latent_dim,
                                      device=self.device)
            loss_fn = vanilla_loss if self.loss_type == "vanilla" else wgan_loss
            self._last_spectral = self.jacobian_tracker.compute(
                self.generator, self.discriminator, loss_fn, real_probe, noise_probe)
        health = self.ehs_tracker.compute(
            balance["balance_score"], modes["mode_coverage"],
            stability["stability_score"], self._last_spectral["spectral_health"])
        metrics = {
            "epoch": epoch,
            "g_loss": g_loss_total / max(generator_steps, 1),
            "d_loss": d_loss_total / max(discriminator_steps, 1),
            "wasserstein_dist": wasserstein_total / max(discriminator_steps, 1),
            "gradient_penalty": penalty_total / max(discriminator_steps, 1),
            **balance, **stability, **modes, **self._last_spectral, **health,
        }
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.metrics_history[key].append(float(value))
        return metrics

    def save_sample_grid(self, epoch: int, run_name: str) -> Path | None:
        if self.output_dir is None:
            return None
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(self.training_config.sample_grid_size,
                                self.model_config.latent_dim, device=self.device)
            images = self.generator(noise).cpu()
        path = self.output_dir / "samples" / f"{run_name}_ep{epoch:04d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        grid = make_grid(images, nrow=8, normalize=True, value_range=(-1, 1))
        save_image(grid, path)
        self.generator.train()
        return path
