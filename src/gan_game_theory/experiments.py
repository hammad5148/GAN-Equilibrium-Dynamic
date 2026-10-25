"""Programmatic experiment orchestration used by the CLI."""

import time
from pathlib import Path

import torch
from torch import optim

from .data.datasets import (
    build_classifier_loaders,
    get_balanced_loader,
    get_imbalanced_loader,
)
from .models import Discriminator, Generator, ModeClassifier
from .optimizers import build_optimizers
from .training import GANTrainer
from .utils import save_json, scalar_history, set_seed
from .diagnostics import ModeCoverageTracker


def train_mode_probe(dataset_name, config, device):
    train_loader, validation_loader = build_classifier_loaders(
        dataset_name, config.paths.data_dir, num_workers=config.num_workers)
    channels = 1 if dataset_name in {"MNIST", "FashionMNIST"} else 3
    classifier = ModeClassifier(config.model.num_classes, channels).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=config.classifier_lr)
    from .models.classifier import train_classifier
    accuracy = train_classifier(classifier, train_loader, validation_loader, optimizer,
                                config.classifier_epochs, device)
    return classifier, accuracy


def train_one(loss_type, optimizer_name, seed, loader, image_size, channels,
              classifier, config, device, tag="") -> dict:
    set_seed(seed)
    generator = Generator(config.model.latent_dim, config.model.generator_hidden_dim,
                          channels, image_size).to(device)
    discriminator = Discriminator(
        channels, config.model.discriminator_hidden_dim,
        "vanilla" if loss_type == "vanilla" else "wgan", img_size=image_size).to(device)
    opt_generator, opt_discriminator = build_optimizers(
        generator, discriminator, optimizer_name, config.training)
    tracker = ModeCoverageTracker(classifier, config.model.num_classes, device)
    run_name = f"{loss_type}_{optimizer_name}_s{seed}{tag}"
    trainer = GANTrainer(
        generator, discriminator, opt_generator, opt_discriminator, device, loss_type,
        config.model, config.training, config.diagnostics, tracker, config.paths.results_dir)
    start = time.time()
    for epoch in range(1, config.training.epochs + 1):
        metrics = trainer.train_epoch(loader, epoch)
        if epoch == 1 or epoch % 10 == 0:
            print(f"{run_name} | epoch {epoch:03d} | EHS={metrics['ehs']:.3f} "
                  f"coverage={metrics['mode_coverage']:.2f}")
        if epoch % config.training.sample_every == 0:
            trainer.save_sample_grid(epoch, run_name)
    result = scalar_history(dict(trainer.metrics_history))
    save_json(result, config.paths.metrics_dir / f"{run_name}.json")
    print(f"Completed {run_name} in {(time.time() - start) / 60:.1f} min")
    return result


def run_grid(config, dataset_name="FashionMNIST", imbalanced=False,
             losses=None, optimizers=None, seeds=None, device=None):
    config.ensure_dirs()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    losses = list(losses or config.losses)
    optimizers = list(optimizers or config.optimizers)
    seeds = list(seeds or config.seeds)
    classifier, accuracy = train_mode_probe(dataset_name, config, device)
    print(f"Mode probe accuracy ({dataset_name}): {accuracy:.4f}")
    if imbalanced:
        loader, image_size, channels, _ = get_imbalanced_loader(
            dataset_name, config.paths.data_dir, config.training.batch_size,
            config.max_per_class, config.imbalance_decay, config.num_workers)
        tag = "_imb_fmnist" if dataset_name == "FashionMNIST" else f"_imb_{dataset_name.lower()}"
    else:
        loader, image_size, channels = get_balanced_loader(
            dataset_name, config.paths.data_dir, config.training.batch_size, config.num_workers)
        tag = "_cifar10" if dataset_name == "CIFAR10" else ""
    results = {}
    for loss_type in losses:
        for optimizer_name in optimizers:
            key = f"{loss_type}_{optimizer_name}{tag}"
            results[key] = {}
            for seed in seeds:
                results[key][seed] = train_one(
                    loss_type, optimizer_name, seed, loader, image_size, channels,
                    classifier, config, device, tag=tag)
    return results
