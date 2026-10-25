"""Publication-friendly plots for the tracked experiments."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..analysis.results import load_results, mean_std


def _label(config: str) -> str:
    return config.replace("_", " ").title()


def _plot_metric(grouped, metric: str, title: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for config, by_seed in grouped.items():
        average, spread = mean_std(by_seed, metric)
        if len(average) == 0:
            continue
        epochs = np.arange(1, len(average) + 1)
        ax.plot(epochs, average, linewidth=2, label=_label(config))
        ax.fill_between(epochs, average - spread, average + spread, alpha=0.12)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_leaderboard(grouped, path: Path) -> None:
    names, values, errors = [], [], []
    for config, by_seed in grouped.items():
        average, spread = mean_std(by_seed, "ehs")
        if len(average):
            names.append(_label(config))
            values.append(average[-1])
            errors.append(spread[-1])
    order = np.argsort(values)[::-1]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(np.arange(len(order)), np.asarray(values)[order], yerr=np.asarray(errors)[order], capsize=4)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([names[i] for i in order], rotation=35, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Final EHS (mean ± std)")
    ax.set_title("Equilibrium Health Score leaderboard")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_all(metrics_dir: str | Path, figures_dir: str | Path) -> list[Path]:
    grouped = load_results(metrics_dir)
    output = Path(figures_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    plots = [
        ("g_loss", "Generator loss", "Loss", "01_generator_loss.png"),
        ("d_loss", "Discriminator/critic loss", "Loss", "02_discriminator_loss.png"),
        ("balance_score", "Discriminator balance", "Balance score", "03_balance_score.png"),
        ("mode_coverage", "Mode coverage", "Fraction of classes", "04_mode_coverage.png"),
        ("ehs", "Equilibrium Health Score", "EHS", "05_ehs_comparison.png"),
        ("stability_score", "Gradient stability", "Stability score", "06_gradient_stability.png"),
    ]
    for metric, title, ylabel, filename in plots:
        path = output / filename
        _plot_metric(grouped, metric, title, ylabel, path)
        paths.append(path)
    leaderboard = output / "07_ehs_leaderboard.png"
    _plot_leaderboard(grouped, leaderboard)
    paths.append(leaderboard)
    return paths
