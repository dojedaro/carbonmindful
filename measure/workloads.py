"""Models, datasets and training steps for the controlled experiments.

Datasets are held in memory as normalised tensors. This removes data-loader workers as a
source of variation and as a bottleneck that could distort a CPU and GPU comparison.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn, optim

SPLIT_SEED = 42
VALIDATION_SIZE = 5000
EVALUATION_BATCH = 512
_STATISTICS = {
    "MNIST": ((0.1307,), (0.3081,)),
    "CIFAR-10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
}
DATASETS = tuple(_STATISTICS)
MODELS = ("mlp", "cnn")


@dataclass(frozen=True)
class Splits:
    """Tensors for one dataset. The test split is used only after training ends."""

    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor


@dataclass(frozen=True)
class EpochResult:
    loss: float
    accuracy: float
    seconds: float
    samples: int


def _normalise(images: torch.Tensor, dataset: str) -> torch.Tensor:
    mean, std = _STATISTICS[dataset]
    channels = len(mean)
    means = torch.tensor(mean).view(1, channels, 1, 1)
    stds = torch.tensor(std).view(1, channels, 1, 1)
    return (images.float() / 255.0 - means) / stds


def _raw_tensors(dataset: str, root: Path, train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    from torchvision import datasets  # noqa: PLC0415 - slow import, needed only for real data

    if dataset == "MNIST":
        mnist = datasets.MNIST(root=str(root), train=train, download=True)
        return mnist.data.unsqueeze(1), mnist.targets.long()
    if dataset == "CIFAR-10":
        cifar = datasets.CIFAR10(root=str(root), train=train, download=True)
        images = torch.from_numpy(cifar.data).permute(0, 3, 1, 2)
        return images, torch.tensor(cifar.targets, dtype=torch.long)
    raise ValueError(f"Unknown dataset: {dataset}")


def make_splits(
    images: torch.Tensor,
    labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    dataset: str,
    validation_size: int = VALIDATION_SIZE,
) -> Splits:
    """Hold out a validation set. The split is fixed, so seeds vary only the training."""
    order = torch.randperm(len(images), generator=torch.Generator().manual_seed(SPLIT_SEED))
    val, train = order[:validation_size], order[validation_size:]
    normal = _normalise(images, dataset)
    return Splits(
        train_x=normal[train],
        train_y=labels[train],
        val_x=normal[val],
        val_y=labels[val],
        test_x=_normalise(test_images, dataset),
        test_y=test_labels,
    )


def load_splits(dataset: str, root: Path) -> Splits:
    """Download if needed, then build the splits. torchvision verifies the downloads."""
    images, labels = _raw_tensors(dataset, root, train=True)
    test_images, test_labels = _raw_tensors(dataset, root, train=False)
    return make_splits(images, labels, test_images, test_labels, dataset)


class TinyMLP(nn.Module):
    """The fully connected network from the original submission: 784-128-64-10."""

    def __init__(self, inputs: int = 28 * 28) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor = self.net(x)
        return output


class SmallCNN(nn.Module):
    """A compact convolutional network with far higher compute per sample than the MLP."""

    def __init__(self, channels: int = 3) -> None:
        super().__init__()

        def block(inputs: int, outputs: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(inputs, outputs, 3, padding=1),
                nn.BatchNorm2d(outputs),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.net = nn.Sequential(
            block(channels, 32),
            block(32, 64),
            block(64, 128),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor = self.net(x)
        return output


def build_model(name: str, sample: torch.Tensor) -> nn.Module:
    """Create a model whose input size matches the data."""
    if name == "mlp":
        return TinyMLP(inputs=int(sample[0].numel()))
    if name == "cnn":
        return SmallCNN(channels=int(sample.shape[1]))
    raise ValueError(f"Unknown model: {name}")


def seed_everything(seed: int) -> torch.Generator:
    """Seed model initialisation and return the generator that orders the batches."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return torch.Generator().manual_seed(seed)


def _sync(device: str) -> None:
    """Wait for queued GPU work, so timings and energy cover the work actually done."""
    if device == "cuda":
        torch.cuda.synchronize()


def train_one_epoch(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: optim.Optimizer,
    device: str,
    batch_size: int,
    generator: torch.Generator,
) -> EpochResult:
    """Train over one shuffled pass of the data."""
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    order = torch.randperm(len(x), generator=generator)
    total_loss, correct = 0.0, 0
    start = time.perf_counter()
    for begin in range(0, len(x), batch_size):
        index = order[begin : begin + batch_size]
        inputs, targets = x[index].to(device), y[index].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(index)
        correct += int((logits.argmax(dim=1) == targets).sum().item())
    _sync(device)
    return EpochResult(total_loss / len(x), correct / len(x), time.perf_counter() - start, len(x))


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, device: str) -> EpochResult:
    """Return loss and accuracy without changing the model."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, correct = 0.0, 0
    start = time.perf_counter()
    for begin in range(0, len(x), EVALUATION_BATCH):
        inputs = x[begin : begin + EVALUATION_BATCH].to(device)
        targets = y[begin : begin + EVALUATION_BATCH].to(device)
        logits = model(inputs)
        total_loss += float(loss_fn(logits, targets).item())
        correct += int((logits.argmax(dim=1) == targets).sum().item())
    _sync(device)
    return EpochResult(total_loss / len(x), correct / len(x), time.perf_counter() - start, len(x))
