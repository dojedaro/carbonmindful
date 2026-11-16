"""Synthetic bundles that show the interface. These numbers are not results."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .bundle import build_bundle

_SPECS: list[tuple[str, str, str, float, float, float]] = [
    ("MLP", "MNIST", "cpu", 0.0004, 0.975, 180.0),
    ("CNN", "CIFAR-10", "cpu", 0.0080, 0.740, 1500.0),
    ("CNN", "CIFAR-10", "cuda", 0.0060, 0.745, 240.0),
]
_EPOCHS = 8


def _bundle(spec: tuple[str, str, str, float, float, float], seed: int) -> bytes:
    model, dataset, device, per_epoch, top, seconds = spec
    rng = np.random.default_rng([seed, len(model), len(device)])
    index = np.arange(1, _EPOCHS + 1)
    accuracy = top - 0.55 * top * np.exp(-0.7 * index) + rng.normal(0, 0.001, _EPOCHS)
    energy = per_epoch * rng.normal(1, 0.02, _EPOCHS)
    frame = pd.DataFrame(
        {
            "epoch": index,
            "duration_s": seconds / _EPOCHS * rng.normal(1, 0.02, _EPOCHS),
            "val_acc": accuracy.clip(0, 1),
            "energy_kwh": energy,
            "samples_per_s": 50000 / (seconds / _EPOCHS),
        }
    )
    points = int(seconds // 5)
    samples = pd.DataFrame(
        {
            "timestamp": 1_700_000_000 + 5.0 * np.arange(points),
            "cpu_util_pct": ((35 if device == "cuda" else 95) + rng.normal(0, 3, points)).clip(
                0, 100
            ),
        }
    )
    if device == "cuda":
        samples["gpu_util_pct"] = (80 + rng.normal(0, 5, points)).clip(0, 100)
    summary = {
        "schema_version": 1,
        "synthetic": True,
        "run_id": f"demo_{model}_{device}_seed{seed}".lower(),
        "model": model,
        "dataset": dataset,
        "device": device,
        "platform": "synthetic",
        "seed": seed,
        "duration_s": float(frame["duration_s"].sum()),
        "energy_kwh": float(energy.sum()),
        "tracker_pue": 1.0,
        "energy_source": "SYNTHETIC DEMO DATA",
        "test_accuracy": float(accuracy[-1]),
    }
    return build_bundle(summary, frame, samples)


def demo_bundles() -> list[bytes]:
    """Return nine synthetic bundles: three configurations with three seeds."""
    return [_bundle(spec, seed) for spec in _SPECS for seed in (1, 2, 3)]
