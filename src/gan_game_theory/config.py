"""Typed configuration for experiments and artifact paths."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PathConfig:
    root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    data_dir: Path | None = None
    results_dir: Path | None = None

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        self.data_dir = self.data_dir or self.root / "data"
        self.results_dir = self.results_dir or self.root / "outputs"

    @property
    def figures_dir(self) -> Path:
        return self.results_dir / "figures"

    @property
    def metrics_dir(self) -> Path:
        return self.results_dir / "metrics"

    @property
    def checkpoints_dir(self) -> Path:
        return self.results_dir / "checkpoints"

    @property
    def samples_dir(self) -> Path:
        return self.results_dir / "samples"

    def ensure(self) -> None:
        for path in (self.data_dir, self.results_dir, self.figures_dir,
                     self.metrics_dir, self.checkpoints_dir, self.samples_dir):
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    latent_dim: int = 100
    generator_hidden_dim: int = 64
    discriminator_hidden_dim: int = 64
    num_classes: int = 10


@dataclass
class TrainingConfig:
    batch_size: int = 128
    epochs: int = 100
    generator_lr: float = 2e-4
    discriminator_lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    gp_lambda: float = 10.0
    n_critic: int = 5
    negative_momentum_beta: float = -0.1
    negative_momentum_lr: float = 1e-3
    sample_every: int = 10
    sample_grid_size: int = 64


@dataclass
class DiagnosticConfig:
    jacobian_every: int = 10
    jacobian_probes: int = 15
    gradient_window: int = 10
    spectral_tau: float = 10.0
    ehs_weights: dict[str, float] = field(default_factory=lambda: {
        "discriminator_balance": 0.30,
        "mode_coverage": 0.30,
        "gradient_stability": 0.20,
        "spectral_health": 0.20,
    })


@dataclass
class ExperimentConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    diagnostics: DiagnosticConfig = field(default_factory=DiagnosticConfig)
    losses: tuple[str, ...] = ("vanilla", "wgan_gp")
    optimizers: tuple[str, ...] = ("adam", "optimistic_adam", "negative_momentum")
    seeds: tuple[int, ...] = (42, 123, 456)
    imbalance_decay: float = 0.5
    max_per_class: int = 5000
    classifier_epochs: int = 10
    classifier_lr: float = 1e-3
    num_workers: int = 0

    def ensure_dirs(self) -> None:
        self.paths.ensure()


def default_config(root: str | Path | None = None) -> ExperimentConfig:
    """Return the project defaults, optionally rooted at another directory."""
    paths = PathConfig(root=Path(root).resolve()) if root else PathConfig()
    return ExperimentConfig(paths=paths)
