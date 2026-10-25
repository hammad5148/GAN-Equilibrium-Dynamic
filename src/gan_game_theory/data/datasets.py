"""Dataset and class-imbalance utilities for Fashion-MNIST and CIFAR-10."""

from collections import defaultdict

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms


DATASET_SPECS = {
    "MNIST": (28, 1),
    "FashionMNIST": (28, 1),
    "CIFAR10": (32, 3),
}


def _transform(dataset_name: str, classifier: bool = False):
    image_size, channels = DATASET_SPECS[dataset_name]
    operations = [transforms.Resize(image_size)]
    if channels == 1:
        operations.append(transforms.Grayscale(1))
    operations.append(transforms.ToTensor())
    if not classifier:
        operations.append(transforms.Normalize([0.5] * channels, [0.5] * channels))
    return transforms.Compose(operations)


def _dataset(dataset_name: str, root, train: bool, classifier: bool = False):
    if dataset_name not in DATASET_SPECS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    dataset_class = getattr(torchvision.datasets, dataset_name)
    return dataset_class(root, train=train, download=True,
                         transform=_transform(dataset_name, classifier=classifier))


def get_balanced_loader(dataset_name: str, root, batch_size: int = 128,
                        num_workers: int = 0):
    dataset = _dataset(dataset_name, root, train=True)
    image_size, channels = DATASET_SPECS[dataset_name]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                        drop_last=True)
    return loader, image_size, channels


def get_imbalanced_loader(dataset_name: str, root, batch_size: int = 128,
                          max_per_class: int = 5000, decay: float = 0.5,
                          num_workers: int = 0):
    dataset = _dataset(dataset_name, root, train=True)
    labels = np.asarray(dataset.targets)
    counts = {class_id: max(int(max_per_class * decay ** class_id), 10)
              for class_id in range(10)}
    class_to_indices: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        class_to_indices[int(label)].append(index)

    selected = []
    for class_id, count in counts.items():
        available = class_to_indices[class_id]
        selected.extend(np.random.choice(available, size=min(count, len(available)),
                                         replace=False).tolist())
    subset = Subset(dataset, selected)
    image_size, channels = DATASET_SPECS[dataset_name]
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                        drop_last=True)
    return loader, image_size, channels, counts


def build_classifier_loaders(dataset_name: str, root, batch_size: int = 256,
                             validation_batch_size: int = 512, num_workers: int = 0):
    train_dataset = _dataset(dataset_name, root, train=True, classifier=True)
    validation_dataset = _dataset(dataset_name, root, train=False, classifier=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers,
                              pin_memory=torch.cuda.is_available())
    validation_loader = DataLoader(validation_dataset, batch_size=validation_batch_size,
                                   shuffle=False, num_workers=num_workers,
                                   pin_memory=torch.cuda.is_available())
    return train_loader, validation_loader
