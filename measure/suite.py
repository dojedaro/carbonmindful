"""Run the planned experiments. Safe to stop and restart: finished runs are skipped.

Seeds are interleaved, so slow drift such as heat or background activity spreads across
configurations and does not fall on one. Each run gets a fresh process, and idle power is
measured before and after the suite.

Examples:
    python -m measure.suite --suite laptop --platform laptop
    python -m measure.suite --suite gpu --platform colab-t4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import psutil

from . import sensors
from .run import DEFAULT_DATA_DIR

BUSY_CPU_PERCENT = 20.0


@dataclass(frozen=True)
class Planned:
    model: str
    dataset: str
    epochs: int
    device: str = "cpu"


SUITES: dict[str, tuple[Planned, ...]] = {
    # E1: the original task at its original length, and long enough to study returns.
    # E2: a workload with different computational demands.
    "laptop": (
        Planned("mlp", "MNIST", 3),
        Planned("mlp", "MNIST", 20),
        Planned("cnn", "CIFAR-10", 15),
    ),
    # E3: the same workload and machine, so the comparison isolates the device.
    "gpu": (
        Planned("cnn", "CIFAR-10", 15, "cpu"),
        Planned("cnn", "CIFAR-10", 15, "cuda"),
    ),
}


def plan(suite: str, seeds: int) -> list[tuple[Planned, int]]:
    """Order runs by seed first, so every configuration's repeats are spread over time."""
    return [(item, seed) for seed in range(1, seeds + 1) for item in SUITES[suite]]


def preflight(allow_battery: bool, allow_busy: bool) -> list[str]:
    """Return the reasons this machine should not start measuring now."""
    problems = []
    if sensors.power_plugged() is False and not allow_battery:
        problems.append("The machine is on battery. Plug it in, or pass --allow-battery.")
    load = psutil.cpu_percent(interval=3)
    if load > BUSY_CPU_PERCENT and not allow_busy:
        problems.append(f"CPU load is {load:.0f}%. Close other work, or pass --allow-busy.")
    return problems


def _command(args: argparse.Namespace, *extra: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "measure.run",
        "--platform",
        args.platform,
        "--out",
        str(args.out),
        "--data-dir",
        str(args.data_dir),
        *extra,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", required=True, choices=sorted(SUITES))
    parser.add_argument("--platform", required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("../experiments/runs"))
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cooldown", type=float, default=20.0)
    parser.add_argument("--idle-seconds", type=float, default=120.0)
    parser.add_argument("--allow-battery", action="store_true")
    parser.add_argument("--allow-busy", action="store_true")
    args = parser.parse_args()

    problems = preflight(args.allow_battery, args.allow_busy)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 2

    failures = 0
    runs = plan(args.suite, args.seeds)
    with sensors.keep_awake():
        idle = ["--idle", str(args.idle_seconds), "--idle-label"]
        label = f"idle_{args.suite}_{args.platform}"
        subprocess.run(_command(args, *idle, f"{label}_before"), check=False)
        for index, (item, seed) in enumerate(runs, start=1):
            print(f"[{index}/{len(runs)}] {item} seed {seed}", flush=True)
            result = subprocess.run(
                _command(
                    args,
                    "--model",
                    item.model,
                    "--dataset",
                    item.dataset,
                    "--epochs",
                    str(item.epochs),
                    "--seed",
                    str(seed),
                    "--device",
                    item.device,
                ),
                check=False,
            )
            failures += int(result.returncode != 0)
            time.sleep(args.cooldown)
        subprocess.run(_command(args, *idle, f"{label}_after"), check=False)
    print(f"Finished {len(runs)} runs with {failures} failures.", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
