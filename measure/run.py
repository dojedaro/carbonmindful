"""Run one controlled training experiment and write its bundle.

What each run records:

* Energy per epoch from CodeCarbon tasks, covering training and validation. It is measured
  for that epoch and never split by time share.
* The tracker runs with PUE 1.0, so recorded energy is IT energy. PUE belongs to the
  scenarios in the dashboard and is applied only there.
* CPU, RAM and GPU energy separately, because they differ in quality: CPU may be
  counter-based while RAM is a constant-power estimate.
* Test accuracy, evaluated once after the final epoch and outside the energy measurement.

Example:
    python -m measure.run --model mlp --dataset MNIST --epochs 3 --seed 1 --platform laptop
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform as platform_module
import sys
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
import torch
from torch import nn, optim

from carbonmindful.bundle import build_bundle

from . import sensors, workloads

LEARNING_RATE = 1e-3
BATCH_SIZE = 128
DEFAULT_DATA_DIR = Path.home() / ".cache" / "carbonmindful"


class Tracker(Protocol):
    """The part of the CodeCarbon tracker these runs use."""

    def start_task(self, task_name: str | None = None) -> None: ...

    def stop_task(self, task_name: str | None = None) -> Any: ...


@dataclass(frozen=True)
class RunConfig:
    model: str
    dataset: str
    epochs: int
    seed: int
    device: str
    platform: str

    @property
    def run_id(self) -> str:
        return (
            f"{self.model}_{self.dataset}_{self.device}_{self.platform}_e{self.epochs}_s{self.seed}"
        )


def make_tracker(country: str) -> Any:
    """Create the tracker. PUE stays at 1.0, so PUE cannot be counted twice."""
    from codecarbon import OfflineEmissionsTracker  # noqa: PLC0415 - slow import

    return OfflineEmissionsTracker(
        country_iso_code=country,
        pue=1.0,
        measure_power_secs=1,
        tracking_mode="machine",
        save_to_file=False,
        log_level="error",
        allow_multiple_runs=True,
    )


def prepare_tracker(tracker: Tracker) -> None:
    """Finish the tracker's lazy hardware setup, which would otherwise fall in epoch 1."""
    tracker.start_task("setup")
    tracker.stop_task("setup")


def tracker_modes(tracker: Any) -> dict[str, str]:
    """Report how each component was tracked. Call this after :func:`prepare_tracker`."""
    labels = {"gpu": "nvml", "ram": "constant estimate"}
    modes: dict[str, str] = {}
    for hardware in getattr(tracker, "_hardware", []):
        name = type(hardware).__name__.lower()
        modes[name] = str(getattr(hardware, "_mode", None) or labels.get(name, "unknown"))
    return modes


def tracker_version() -> str:
    return metadata.version("codecarbon")


def _train_epochs(
    config: RunConfig,
    model: nn.Module,
    splits: workloads.Splits,
    tracker: Tracker,
) -> tuple[list[dict[str, float]], float]:
    generator = workloads.seed_everything(config.seed)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    rows: list[dict[str, float]] = []
    emissions = 0.0
    for epoch in range(1, config.epochs + 1):
        counter_start = sensors.read_package_counter_kwh()
        tracker.start_task(f"epoch_{epoch}")
        started = time.perf_counter()
        train = workloads.train_one_epoch(
            model,
            splits.train_x,
            splits.train_y,
            optimizer,
            config.device,
            BATCH_SIZE,
            generator,
        )
        val = workloads.evaluate(model, splits.val_x, splits.val_y, config.device)
        duration = time.perf_counter() - started
        measured = tracker.stop_task(f"epoch_{epoch}")
        counter_end = sensors.read_package_counter_kwh()
        emissions += float(measured.emissions)
        counter = (
            counter_end - counter_start
            if counter_start is not None and counter_end is not None
            else math.nan
        )
        rows.append(
            {
                "epoch": epoch,
                "duration_s": duration,
                "package_counter_kwh": counter,
                "train_s": train.seconds,
                "train_loss": train.loss,
                "train_acc": train.accuracy,
                "val_loss": val.loss,
                "val_acc": val.accuracy,
                "samples_per_s": train.samples / train.seconds,
                "energy_kwh": float(measured.energy_consumed),
                "cpu_energy_kwh": float(measured.cpu_energy),
                "ram_energy_kwh": float(measured.ram_energy),
                "gpu_energy_kwh": float(measured.gpu_energy),
            }
        )
    return rows, emissions


def energy_source(modes: dict[str, str], version: str) -> str:
    """Describe the energy data in plain words, for the bundle and the paper."""
    parts = [f"{name.upper()} {mode}" for name, mode in sorted(modes.items())]
    return f"CodeCarbon {version}, " + ", ".join(parts) if parts else f"CodeCarbon {version}"


def execute(
    config: RunConfig,
    splits: workloads.Splits,
    tracker: Tracker,
    tracker_country: str,
    modes: dict[str, str],
    version: str,
) -> bytes:
    """Train, evaluate and return the finished bundle."""
    workloads.seed_everything(config.seed)
    model = workloads.build_model(config.model, splits.train_x[:1]).to(config.device)
    with sensors.UsageSampler() as sampler:
        rows, emissions = _train_epochs(config, model, splits, tracker)
    test = workloads.evaluate(model, splits.test_x, splits.test_y, config.device)
    epochs = pd.DataFrame(rows)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "run_id": config.run_id,
        "model": config.model,
        "dataset": config.dataset,
        "device": config.device,
        "platform": config.platform,
        "seed": config.seed,
        "duration_s": float(epochs["duration_s"].sum()),
        "energy_kwh": float(epochs["energy_kwh"].sum()),
        "cpu_energy_kwh": float(epochs["cpu_energy_kwh"].sum()),
        "ram_energy_kwh": float(epochs["ram_energy_kwh"].sum()),
        "gpu_energy_kwh": float(epochs["gpu_energy_kwh"].sum()),
        "tracker_pue": 1.0,
        "tracker_emissions_kg": emissions,
        "tracker_country_iso": tracker_country,
        "energy_source": energy_source(modes, version),
        "test_accuracy": test.accuracy,
        "test_loss": test.loss,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "train_samples": len(splits.train_x),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "machine": sensors.describe_machine(config.device),
        "software": {
            "python": platform_module.python_version(),
            "torch": torch.__version__,
            "codecarbon": version,
        },
    }
    if epochs["package_counter_kwh"].notna().all():
        summary["package_counter_kwh"] = float(epochs["package_counter_kwh"].sum())
    return build_bundle(summary, epochs, sampler.frame())


def write_atomically(path: Path, data: bytes) -> None:
    """Write through a temporary file, so an interruption cannot leave half a bundle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def measure_idle(seconds: float, tracker: Tracker) -> dict[str, float]:
    """Measure the machine doing nothing, to show how much of a run is background draw."""
    tracker.start_task("idle")
    time.sleep(seconds)
    measured = tracker.stop_task("idle")
    hours = seconds / 3600
    return {
        "seconds": seconds,
        "energy_kwh": float(measured.energy_consumed),
        "cpu_energy_kwh": float(measured.cpu_energy),
        "ram_energy_kwh": float(measured.ram_energy),
        "gpu_energy_kwh": float(measured.gpu_energy),
        "mean_power_w": float(measured.energy_consumed) * 1000 / hours,
        "mean_cpu_power_w": float(measured.cpu_energy) * 1000 / hours,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=workloads.MODELS)
    parser.add_argument("--dataset", choices=workloads.DATASETS)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--platform", required=True, help="Machine label, for example laptop")
    parser.add_argument("--tracker-country", default="KOR")
    parser.add_argument("--out", type=Path, default=Path("../experiments/runs"))
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--force", action="store_true", help="Replace an existing bundle")
    parser.add_argument("--idle", type=float, help="Measure idle power for this many seconds")
    parser.add_argument("--idle-label", default="idle")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.idle:
        tracker = make_tracker(args.tracker_country)
        prepare_tracker(tracker)
        with sensors.keep_awake():
            result = measure_idle(args.idle, tracker)
        tracker.stop()
        result["energy_source"] = energy_source(tracker_modes(tracker), tracker_version())  # type: ignore[assignment]
        path = args.out / f"{args.idle_label}.json"
        write_atomically(path, json.dumps(result, indent=2).encode())
        print(f"Idle power {result['mean_power_w']:.2f} W. Wrote {path.name}")
        return 0
    if None in (args.model, args.dataset, args.epochs, args.seed):
        print("--model, --dataset, --epochs and --seed are required.", file=sys.stderr)
        return 2
    if args.device == "cuda" and not torch.cuda.is_available():
        print("--device cuda was requested, but no CUDA GPU is available.", file=sys.stderr)
        return 2
    config = RunConfig(args.model, args.dataset, args.epochs, args.seed, args.device, args.platform)
    path = args.out / f"{config.run_id}.zip"
    if path.exists() and not args.force:
        print(f"Skipping {config.run_id}: bundle exists.")
        return 0
    splits = workloads.load_splits(config.dataset, args.data_dir)
    tracker = make_tracker(args.tracker_country)
    prepare_tracker(tracker)
    with sensors.keep_awake():
        data = execute(
            config,
            splits,
            tracker,
            args.tracker_country,
            tracker_modes(tracker),
            tracker_version(),
        )
    tracker.stop()
    write_atomically(path, data)
    print(f"Wrote {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
