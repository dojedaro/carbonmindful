"""Turn audited run bundles into the tables behind the paper's results.

Usage:
    python -m measure.analyze --runs ../experiments/runs ../experiments/runs_colab \
        --out ../experiments/results

Every number in the paper's results should trace to a CSV written here. Averages are taken
only within one configuration, and the CPU and GPU comparison pairs runs by seed on the same
platform. Results describe these runs and are not statistical inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from carbonmindful import calc
from carbonmindful.bundle import RunBundle, parse_bundle
from carbonmindful.factors import DEFAULT_PATH, load_factors

from .audit import check_run, row

KEYS = ["model", "dataset", "device", "platform", "epochs"]
PUE_PATH = DEFAULT_PATH.parent / "pue_scenarios.csv"


def load_runs(folders: list[Path]) -> list[RunBundle]:
    """Read every bundle. Unused attempts are kept in other folders, not passed here."""
    return [
        parse_bundle(path.read_bytes())
        for folder in folders
        for path in sorted(folder.glob("*.zip"))
    ]


def run_table(runs: list[RunBundle]) -> pd.DataFrame:
    """One row per run, with a column stating whether it passed the audit."""
    rows = []
    for run in runs:
        problems = check_run(run)
        rows.append({**row(run), "problems": "; ".join(problems)})
    return pd.DataFrame(rows)


def configuration_table(table: pd.DataFrame) -> pd.DataFrame:
    """Mean and sample SD per configuration, for energy by component and performance."""
    measures = [
        "duration_s", "energy_wh", "cpu_wh", "ram_wh", "gpu_wh",
        "test_acc_pct", "samples_per_s", "cpu_util", "outside_load",
    ]  # fmt: skip
    grouped = table.groupby(KEYS)[measures].agg(["mean", "std"])
    grouped.columns = pd.Index([f"{column[0]}_{column[1]}" for column in grouped.columns.to_list()])
    grouped.insert(0, "n", table.groupby(KEYS).size())
    return grouped.reset_index()


def device_comparison(table: pd.DataFrame) -> pd.DataFrame:
    """Compare CPU with GPU, seed by seed, on the same platform and workload.

    ``energy_ratio_without_idle_gpu`` removes the GPU energy from the CPU run, because a
    GPU that sits idle during a CPU run is a property of the machine, not of the training.
    """
    keys = ["model", "dataset", "platform", "epochs", "seed"]
    measures = [*keys, "energy_wh", "gpu_wh", "duration_s", "test_acc_pct"]
    cpu = table[table["device"] == "cpu"][measures]
    gpu = table[table["device"] == "cuda"][measures]
    pairs = cpu.merge(gpu, on=keys, suffixes=("_cpu", "_gpu"))
    pairs["energy_ratio"] = pairs["energy_wh_cpu"] / pairs["energy_wh_gpu"]
    pairs["energy_ratio_without_idle_gpu"] = (pairs["energy_wh_cpu"] - pairs["gpu_wh_cpu"]) / pairs[
        "energy_wh_gpu"
    ]
    pairs["time_ratio"] = pairs["duration_s_cpu"] / pairs["duration_s_gpu"]
    pairs["accuracy_difference_points"] = pairs["test_acc_pct_gpu"] - pairs["test_acc_pct_cpu"]
    return pairs


def idle_power(folders: list[Path]) -> pd.DataFrame:
    """Read the idle measurements taken before and after each suite."""
    rows = []
    for folder in folders:
        for path in sorted(folder.glob("idle*.json")):
            idle = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "platform": path.stem.split("_")[2],
                    "when": path.stem.split("_")[-1],
                    "mean_power_w": idle["mean_power_w"],
                    "mean_cpu_power_w": idle["mean_cpu_power_w"],
                }
            )
    return pd.DataFrame(rows)


def background_table(table: pd.DataFrame, idle: pd.DataFrame) -> pd.DataFrame:
    """Show total energy, energy above idle, and energy without the RAM estimate.

    Total energy stays the headline. The other two show how far the result depends on
    background draw and on RAM's constant-power estimate.
    """
    power = idle.groupby("platform")["mean_power_w"].mean()
    frame = table.assign(
        idle_w=table["platform"].map(power),
        above_idle_wh=lambda t: t["energy_wh"] - t["idle_w"] * t["duration_s"] / 3600,
        without_ram_wh=lambda t: t["energy_wh"] - t["ram_wh"],
        ram_share_pct=lambda t: 100 * t["ram_wh"] / t["energy_wh"],
    )
    measures = ["energy_wh", "above_idle_wh", "without_ram_wh", "ram_share_pct", "idle_w"]
    return frame.groupby(KEYS)[measures].mean().reset_index()


def scenario_table(table: pd.DataFrame) -> pd.DataFrame:
    """Estimate emissions for each configuration, country and PUE, with the seed spread."""
    factors = {iso: factor.kg_per_kwh for iso, factor in load_factors().items()}
    pues = pd.read_csv(PUE_PATH)["pue"].astype(float).tolist()
    energy = table.assign(energy_it_kwh=table["energy_wh"] / 1000)
    grouped = energy.groupby(KEYS)["energy_it_kwh"].agg(["mean", "std"]).reset_index()
    return calc.scenario_matrix(grouped, pues, factors, KEYS)


def returns_table(runs: list[RunBundle]) -> pd.DataFrame:
    """Average accuracy and measured energy by epoch, and the cost of each accuracy gain."""
    rows = []
    for run in runs:
        summary = run.summary
        frame = run.epochs.sort_values("epoch")
        rows.append(
            frame.assign(
                model=summary["model"],
                dataset=summary["dataset"],
                device=summary["device"],
                platform=run.platform,
                epochs=len(frame),
                seed=summary["seed"],
                val_acc_pct=frame["val_acc"] * 100,
                cumulative_wh=frame["energy_kwh"].cumsum() * 1000,
            )
        )
    long = pd.concat(rows, ignore_index=True)
    grouped = long.groupby([*KEYS, "epoch"]).agg(
        val_acc_mean=("val_acc_pct", "mean"),
        val_acc_std=("val_acc_pct", "std"),
        cumulative_wh_mean=("cumulative_wh", "mean"),
        epoch_wh_mean=("energy_kwh", lambda values: float(np.mean(values)) * 1000),
    )
    result = grouped.reset_index()
    gain = result.groupby(KEYS)["val_acc_mean"].diff()
    result["gain_points"] = gain
    result["wh_per_point_gained"] = (result["epoch_wh_mean"] / gain).where(gain > 0)
    return result


def early_stop_table(runs: list[RunBundle], tolerance_points: float = 1.0) -> pd.DataFrame:
    """Energy share used by the first epoch close to each run's best validation accuracy."""
    rows = []
    for run in runs:
        tradeoff = calc.epoch_tradeoff(run.epochs, 1.0, 1.0)
        point = calc.early_stop_point(tradeoff, tolerance_points)
        rows.append(
            {
                "model": run.summary["model"],
                "dataset": run.summary["dataset"],
                "device": run.summary["device"],
                "platform": run.platform,
                "epochs": len(run.epochs),
                "seed": run.summary["seed"],
                "epoch_reached": point.epoch,
                "energy_share_pct": point.energy_share_pct,
            }
        )
    frame = pd.DataFrame(rows)
    summary = frame.groupby(KEYS)[["epoch_reached", "energy_share_pct"]].agg(["mean", "std"])
    summary.columns = pd.Index([f"{column[0]}_{column[1]}" for column in summary.columns.to_list()])
    return summary.reset_index()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    runs = load_runs(args.runs)
    table = run_table(runs)
    failed = table[table["problems"] != ""]
    if not failed.empty:
        print("These runs failed the audit, so no results were written:", file=sys.stderr)
        print(failed[["run_id", "problems"]].to_string(index=False), file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    outputs = {
        "runs": table,
        "configurations": configuration_table(table),
        "cpu_vs_gpu": device_comparison(table),
        "background_energy": background_table(table, idle_power(args.runs)),
        "scenarios": scenario_table(table),
        "returns_by_epoch": returns_table(runs),
        "early_stop": early_stop_table(runs),
    }
    for name, frame in outputs.items():
        frame.to_csv(args.out / f"{name}.csv", index=False)
    print(f"Wrote {len(outputs)} tables from {len(runs)} runs to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
