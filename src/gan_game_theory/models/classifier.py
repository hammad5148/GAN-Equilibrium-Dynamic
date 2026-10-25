"""Classifier used as a lightweight mode-coverage probe."""

import copy

import torch
from torch import nn


class ModeClassifier(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(64, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(images))

    @torch.no_grad()
    def predict_classes(self, images: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        predictions = self((images + 1.0) / 2.0).argmax(dim=1)
        self.train(was_training)
        return predictions


def train_classifier(model: ModeClassifier, train_loader, validation_loader,
                     optimizer, epochs: int, device: torch.device) -> float:
    """Train a classifier and restore the state with the best validation accuracy."""
    criterion = nn.CrossEntropyLoss()
    best_accuracy = 0.0
    best_state = copy.deepcopy(model.state_dict())
    for _ in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            criterion(model(images), labels).backward()
            optimizer.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                correct += (model(images).argmax(1) == labels).sum().item()
                total += labels.numel()
        accuracy = correct / max(total, 1)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return best_accuracy
