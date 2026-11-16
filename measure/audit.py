"""Check every run in a folder and summarise each configuration.

Usage: python -m measure.audit ../experiments/runs
Exits with status 1 if a run fails a check.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from carbonmindful.bundle import RunBundle, parse_bundle

RATIO_TOLERANCE = 0.01
# Limit fixed before the clean rerun: mean CPU use by other processes during a run,
# as a percentage of the whole machine. Whole-machine energy includes that work.
OUTSIDE_LOAD_LIMIT_PCT = 5.0


def outside_load_pct(run: RunBundle) -> float:
    """Mean CPU use by other processes during the run, or 0 if it was not recorded."""
    if "own_cpu_pct" not in run.samples:
        return 0.0
    return float((run.samples["cpu_util_pct"] - run.samples["own_cpu_pct"]).mean())


def check_run(run: RunBundle) -> list[str]:
    """Return the problems found in one run."""
    summary = run.summary
    problems = []
    if summary.get("machine", {}).get("on_mains_power") is False:
        problems.append("recorded on battery")
    if summary.get("tracker_pue") != 1.0:
        problems.append("tracker PUE is not 1.0")
    if run.warnings:
        problems.append("; ".join(run.warnings))
    counter = summary.get("package_counter_kwh")
    if counter and abs(summary["cpu_energy_kwh"] / counter - 1) > RATIO_TOLERANCE:
        problems.append(f"CPU/counter ratio {summary['cpu_energy_kwh'] / counter:.3f}")
    if run.epochs["energy_kwh"].le(0).any():
        problems.append("an epoch has no energy")
    outside = outside_load_pct(run)
    if outside > OUTSIDE_LOAD_LIMIT_PCT:
        problems.append(f"outside CPU load {outside:.1f}%")
    if len(run.samples) < 2:
        problems.append("too few utilisation samples")
    return problems


def row(run: RunBundle) -> dict[str, object]:
    """Flatten one run for the summary tables."""
    summary = run.summary
    seconds = float(summary["duration_s"])
    counter = summary.get("package_counter_kwh")
    return {
        "run_id": run.run_id,
        "model": summary["model"],
        "dataset": summary["dataset"],
        "device": summary["device"],
        "platform": run.platform,
        "epochs": len(run.epochs),
        "seed": summary["seed"],
        "source": summary.get("energy_source"),
        "duration_s": seconds,
        "energy_wh": float(summary["energy_kwh"]) * 1000,
        "cpu_wh": float(summary["cpu_energy_kwh"]) * 1000,
        "ram_wh": float(summary["ram_energy_kwh"]) * 1000,
        "gpu_wh": float(summary["gpu_energy_kwh"]) * 1000,
        "cpu_w": float(summary["cpu_energy_kwh"]) * 3.6e6 / seconds,
        "ram_w": float(summary["ram_energy_kwh"]) * 3.6e6 / seconds,
        "gpu_w": float(summary["gpu_energy_kwh"]) * 3.6e6 / seconds,
        "test_acc_pct": run.accuracy * 100,
        "samples_per_s": float(run.epochs["samples_per_s"].mean()),
        "cpu_util": float(run.samples["cpu_util_pct"].mean()),
        "outside_load": outside_load_pct(run),
        "ratio": float(summary["cpu_energy_kwh"]) / counter if counter else float("nan"),
    }


def main(folder: Path) -> int:
    paths = sorted(folder.glob("*.zip"))
    runs = [parse_bundle(path.read_bytes()) for path in paths]
    failed = 0
    for run in runs:
        for problem in check_run(run):
            print(f"PROBLEM {run.run_id}: {problem}")
            failed += 1
    table = pd.DataFrame([row(run) for run in runs])
    keys = ["model", "dataset", "device", "platform", "epochs"]
    columns = ["duration_s", "energy_wh", "cpu_wh", "ram_wh", "gpu_wh", "cpu_w", "ram_w", "gpu_w",
               "test_acc_pct", "samples_per_s", "cpu_util", "outside_load", "ratio"]  # fmt: skip
    grouped = table.groupby(keys)[columns]
    pd.set_option("display.width", 220, "display.max_columns", 40)
    print(f"{len(runs)} runs, {failed} problems")
    print("Sources:", table["source"].unique().tolist())
    print("Runs per configuration:\n", grouped.size().to_string())
    print("\nMean:\n", grouped.mean().round(3).T.to_string())
    print("\nSD:\n", grouped.std().round(3).T.to_string())
    print("\nRatio range:", table["ratio"].min(), table["ratio"].max())
    for path in sorted(folder.glob("idle*.json")):
        idle = json.loads(path.read_text(encoding="utf-8"))
        print(
            path.name,
            {key: round(value, 2) for key, value in idle.items() if key.startswith("mean")},
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
