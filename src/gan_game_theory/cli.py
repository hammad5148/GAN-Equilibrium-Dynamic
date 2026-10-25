"""Command-line entry points for training and analysis."""

import argparse

import torch

from .analysis.results import write_summary
from .config import default_config
from .experiments import run_grid
from .utils import resolve_device
from .visualization.plots import plot_all


def _common_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--root", default=None, help="Project root (defaults to the installed project)")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser


def _config_from_args(args):
    config = default_config(args.root)
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    return config


def train_main(argv=None):
    parser = _common_parser("Train one GAN configuration")
    parser.add_argument("--loss", choices=["vanilla", "wgan_gp"], default="vanilla")
    parser.add_argument("--optimizer", choices=["adam", "optimistic_adam", "negative_momentum"],
                        default="adam")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", choices=["MNIST", "FashionMNIST", "CIFAR10"],
                        default="FashionMNIST")
    parser.add_argument("--imbalanced", action="store_true")
    args = parser.parse_args(argv)
    config = _config_from_args(args)
    run_grid(config, args.dataset, args.imbalanced, [args.loss], [args.optimizer],
             [args.seed], resolve_device(args.device))


def grid_main(argv=None):
    parser = _common_parser("Run the GAN loss/optimizer factorial grid")
    parser.add_argument("--dataset", choices=["MNIST", "FashionMNIST", "CIFAR10"],
                        default="FashionMNIST")
    parser.add_argument("--imbalanced", action="store_true")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Use one seed and five epochs")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    args = parser.parse_args(argv)
    config = _config_from_args(args)
    seeds = args.seeds or list(config.seeds)
    if args.smoke_test:
        config.training.epochs = 5
        seeds = [seeds[0]]
    run_grid(config, args.dataset, args.imbalanced, seeds=seeds,
             device=resolve_device(args.device))


def plot_main(argv=None):
    parser = argparse.ArgumentParser(description="Plot tracked GAN results")
    parser.add_argument("--root", default=None)
    args = parser.parse_args(argv)
    config = default_config(args.root)
    summary = write_summary(config.paths.metrics_dir)
    paths = plot_all(config.paths.metrics_dir, config.paths.figures_dir)
    print(f"Wrote {len(summary)} summary rows and {len(paths)} plots to {config.paths.results_dir}")


if __name__ == "__main__":
    train_main()
