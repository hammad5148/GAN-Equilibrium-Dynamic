"""Load and summarize metric JSON files produced by training."""

import re
from pathlib import Path

import numpy as np

from ..utils import load_json, save_json

RUN_PATTERN = re.compile(
    r"^(?P<loss>vanilla|wgan_gp)_(?P<optimizer>adam|optimistic_adam|negative_momentum)"
    r"_s(?P<seed>\d+)(?P<tag>.*)$"
)


def load_results(metrics_dir: str | Path) -> dict[str, dict[int, dict]]:
    grouped: dict[str, dict[int, dict]] = {}
    for path in sorted(Path(metrics_dir).glob("*.json")):
        if path.name == "summary_table.json":
            continue
        match = RUN_PATTERN.match(path.stem)
        if not match:
            continue
        config = f"{match['loss']}_{match['optimizer']}"
        tag = match["tag"]
        if tag:
            config += tag
        grouped.setdefault(config, {})[int(match["seed"])] = load_json(path)
    return grouped


def mean_std(results_by_seed: dict[int, dict], metric: str):
    arrays = [np.asarray(result[metric], dtype=float) for result in results_by_seed.values()
              if metric in result]
    if not arrays:
        return np.array([]), np.array([])
    length = min(len(array) for array in arrays)
    matrix = np.stack([array[:length] for array in arrays])
    return matrix.mean(axis=0), matrix.std(axis=0)


def _dataset_from_config(config: str) -> str:
    if "cifar10" in config:
        return "CIFAR10"
    if "imb_fmnist" in config or "imbalanced_fmnist" in config:
        return "FashionMNIST_imbalanced"
    return "FashionMNIST_balanced"


def final_summary(grouped_results: dict[str, dict[int, dict]]) -> list[dict]:
    rows = []
    for config, by_seed in sorted(grouped_results.items()):
        row = {"config": config, "dataset": _dataset_from_config(config)}
        for metric in ("ehs", "balance_score", "mode_coverage",
                       "stability_score", "spectral_health"):
            average, _ = mean_std(by_seed, metric)
            row[metric] = float(average[-1]) if len(average) else 0.0
        rows.append(row)
    return rows


def write_summary(metrics_dir: str | Path) -> list[dict]:
    summary = final_summary(load_results(metrics_dir))
    save_json(summary, Path(metrics_dir) / "summary_table.json")
    return summary
