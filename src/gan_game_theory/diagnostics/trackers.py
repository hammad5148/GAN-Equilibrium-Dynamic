"""Game-theoretic metrics from the notebooks."""

from collections import defaultdict, deque

import numpy as np
import torch


class DiscriminatorBalanceTracker:
    """Scores how close real and generated outputs are to 0.5."""

    def __init__(self):
        self.history = defaultdict(list)

    @torch.no_grad()
    def compute(self, d_real, d_fake, raw_critic: bool = False) -> dict:
        if raw_critic:
            d_real, d_fake = torch.sigmoid(d_real), torch.sigmoid(d_fake)
        real_mean, fake_mean = d_real.mean().item(), d_fake.mean().item()
        values = {
            "d_real_mean": real_mean,
            "d_fake_mean": fake_mean,
            "d_real_std": d_real.std().item(),
            "d_fake_std": d_fake.std().item(),
            "balance_score": max(0.0, 1.0 - abs(real_mean - 0.5) - abs(fake_mean - 0.5)),
        }
        for key, value in values.items():
            self.history[key].append(value)
        return values


class GradientStabilityTracker:
    def __init__(self, window: int = 10):
        self.g_norms = deque(maxlen=window)
        self.d_norms = deque(maxlen=window)

    @staticmethod
    def _norm(model) -> float:
        total = sum(p.grad.detach().norm(2).item() ** 2
                    for p in model.parameters() if p.grad is not None)
        return total ** 0.5

    @staticmethod
    def _cov(values) -> float:
        if len(values) < 2:
            return 0.0
        array = np.asarray(values, dtype=float)
        mean = array.mean()
        return 0.0 if mean < 1e-10 else float(array.std() / mean)

    def record(self, generator, discriminator) -> None:
        self.g_norms.append(self._norm(generator))
        self.d_norms.append(self._norm(discriminator))

    def compute(self) -> dict:
        g_cov, d_cov = self._cov(self.g_norms), self._cov(self.d_norms)
        values = {
            "g_grad_cov": g_cov,
            "d_grad_cov": d_cov,
            "stability_score": max(0.0, 1.0 - (g_cov + d_cov) / 2.0),
        }
        return values


class ModeCoverageTracker:
    def __init__(self, classifier, num_classes: int = 10, device=None):
        self.classifier = classifier
        self.num_classes = num_classes
        self.device = device or next(classifier.parameters()).device

    @torch.no_grad()
    def compute(self, generator, num_samples: int = 1000) -> dict:
        was_training = generator.training
        generator.eval()
        predictions = []
        for start in range(0, num_samples, 256):
            size = min(256, num_samples - start)
            noise = torch.randn(size, generator.latent_dim, device=self.device)
            predictions.append(self.classifier.predict_classes(generator(noise)).cpu())
        generator.train(was_training)
        predictions = torch.cat(predictions)
        counts = torch.bincount(predictions, minlength=self.num_classes).float()
        probabilities = counts / counts.sum().clamp_min(1.0)
        entropy = -(probabilities[probabilities > 0] *
                    probabilities[probabilities > 0].log()).sum().item()
        values = {
            "mode_coverage": float(predictions.unique().numel() / self.num_classes),
            "num_modes": int(predictions.unique().numel()),
            "entropy": float(entropy / np.log(self.num_classes)),
            "class_distribution": probabilities.tolist(),
        }
        return values


class EquilibriumHealthScore:
    def __init__(self, weights: dict[str, float], spectral_tau: float = 10.0):
        self.weights = weights
        self.spectral_tau = spectral_tau
        self.last_spectral_health = 1.0

    def compute(self, discriminator_balance: float, mode_coverage: float,
                gradient_stability: float, spectral_health: float | None = None) -> dict:
        if spectral_health is not None:
            self.last_spectral_health = spectral_health
        components = {
            "discriminator_balance": float(np.clip(discriminator_balance, 0, 1)),
            "mode_coverage": float(np.clip(mode_coverage, 0, 1)),
            "gradient_stability": float(np.clip(gradient_stability, 0, 1)),
            "spectral_health": float(np.clip(self.last_spectral_health, 0, 1)),
        }
        ehs = sum(self.weights[key] * value for key, value in components.items())
        return {"ehs": float(ehs), **components}
