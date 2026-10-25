"""A tractable randomized proxy for the GAN gradient-field spectrum."""

from collections import defaultdict

import numpy as np
import torch
from scipy import linalg


class JacobianSpectrumTracker:
    """Estimate spectral health with randomized Jacobian-vector products.

    This is a diagnostic proxy, not an exact full parameter-space eigendecomposition.
    """

    def __init__(self, num_probes: int = 15, spectral_tau: float = 10.0):
        self.num_probes = num_probes
        self.spectral_tau = spectral_tau
        self.history = defaultdict(list)

    def compute(self, generator, discriminator, loss_fn, real_batch, noise_batch) -> dict:
        generator.eval()
        discriminator.eval()
        parameters = list(generator.parameters()) + list(discriminator.parameters())
        generator_parameter_count = len(list(generator.parameters()))
        rows = []
        for _ in range(self.num_probes):
            probe = self._random_probe(parameters)
            row = self._jvp(generator, discriminator, loss_fn, real_batch,
                            noise_batch, parameters, generator_parameter_count, probe)
            if row is not None:
                rows.append(row)
        generator.train()
        discriminator.train()

        if len(rows) < 2:
            return {"spectral_radius": 0.0, "max_imag_real_ratio": 0.0,
                    "spectral_health": 1.0}
        try:
            gram = np.asarray(rows) @ np.asarray(rows).T
            eigenvalues = linalg.eigvals(gram)
            eigenvalues = np.sqrt(np.abs(eigenvalues) + 1e-10) * np.sign(eigenvalues.real)
        except Exception:
            eigenvalues = np.asarray([0.0])
        ratios = [abs(value.imag) / abs(value.real) if abs(value.real) > 1e-10 else 100.0
                  for value in eigenvalues if abs(value.imag) > 1e-10 or abs(value.real) > 1e-10]
        max_ratio = float(max(ratios)) if ratios else 0.0
        return {
            "spectral_radius": float(np.max(np.abs(eigenvalues))),
            "max_imag_real_ratio": max_ratio,
            "spectral_health": float(1.0 - min(1.0, max_ratio / self.spectral_tau)),
        }

    @staticmethod
    def _random_probe(parameters):
        vector = torch.cat([torch.randn_like(parameter).flatten() for parameter in parameters])
        return vector / (vector.norm() + 1e-10)

    @staticmethod
    def _jvp(generator, discriminator, loss_fn, real, noise,
             parameters, generator_parameter_count, probe):
        try:
            fake = generator(noise)
            loss = loss_fn(discriminator, fake, real)
            gradients = torch.autograd.grad(loss, parameters, create_graph=True,
                                            allow_unused=True)
            vector = []
            for index, (gradient, parameter) in enumerate(zip(gradients, parameters)):
                sign = -1.0 if index < generator_parameter_count else 1.0
                vector.append((sign * gradient if gradient is not None
                               else torch.zeros_like(parameter)).flatten())
            directional = torch.dot(torch.cat(vector), probe)
            second = torch.autograd.grad(directional, parameters, allow_unused=True)
            return np.concatenate([
                (value.detach().cpu().numpy().flatten() if value is not None
                 else np.zeros(parameter.numel()))
                for value, parameter in zip(second, parameters)
            ])
        except RuntimeError:
            return None
