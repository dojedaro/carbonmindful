"""The single calculation chain used everywhere in CarbonMindful.

    E_IT (kWh)  ->  E_facility = E_IT x PUE  ->  CO2e = E_facility x grid factor

PUE is applied exactly once, here. Regional values are scenario-based estimates:
the same recorded energy is re-priced with another grid factor. The workload is
not re-run in that country.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import pandas as pd

CONFIGURATION_KEYS = ["model", "dataset", "device", "platform", "epochs", "accuracy_type"]


def _require_finite(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return number


def facility_energy_kwh(energy_it_kwh: float, pue: float) -> float:
    """Scale IT energy to facility energy."""
    energy = _require_finite("energy_it_kwh", energy_it_kwh)
    ratio = _require_finite("pue", pue)
    if energy < 0:
        raise ValueError("energy_it_kwh cannot be negative")
    if ratio < 1:
        raise ValueError("pue cannot be below 1.0")
    return energy * ratio


def emissions_kg(energy_it_kwh: float, pue: float, factor_kg_per_kwh: float) -> float:
    """Estimate kg CO2e for IT energy under one PUE and grid factor."""
    factor = _require_finite("factor_kg_per_kwh", factor_kg_per_kwh)
    if factor < 0:
        raise ValueError("factor_kg_per_kwh cannot be negative")
    return facility_energy_kwh(energy_it_kwh, pue) * factor


def scenario_table(
    energy_it_kwh: float, pues: Iterable[float], factors: Mapping[str, float]
) -> pd.DataFrame:
    """Return one row per (region, PUE) scenario."""
    rows = [
        {
            "iso": iso,
            "pue": float(pue),
            "factor_kg_per_kwh": float(factor),
            "energy_it_kwh": float(energy_it_kwh),
            "energy_facility_kwh": facility_energy_kwh(energy_it_kwh, pue),
            "emissions_kg": emissions_kg(energy_it_kwh, pue, factor),
        }
        for iso, factor in factors.items()
        for pue in pues
    ]
    return pd.DataFrame(rows)


def saving_pct(high: float, low: float) -> float:
    """Percentage reduction when moving from ``high`` to ``low``."""
    if high <= 0:
        raise ValueError("high must be positive")
    return 100.0 * (1.0 - low / high)


def kg_per_accuracy_point(emissions: float, accuracy_fraction: float) -> float:
    """Total kg CO2e divided by final accuracy in percentage points.

    Descriptive only. It is meaningful for runs of the same task and dataset, and
    it is not linear in model quality, so it should not rank different tasks.
    """
    if not 0 < accuracy_fraction <= 1:
        raise ValueError("accuracy_fraction must be in (0, 1]")
    return emissions / (accuracy_fraction * 100.0)


def epoch_tradeoff(epochs: pd.DataFrame, pue: float, factor_kg_per_kwh: float) -> pd.DataFrame:
    """Per-epoch energy, emissions and marginal cost of each accuracy gain.

    ``marginal_kg_per_point`` is NaN for the first epoch (no earlier reference) and
    for any epoch without an accuracy gain, where the ratio has no meaning.
    """
    table = epochs.sort_values("epoch").reset_index(drop=True)
    energy = table["energy_it_kwh"]
    scale = pue * factor_kg_per_kwh
    gain = (table["val_acc"] * 100.0).diff()
    result = pd.DataFrame(
        {
            "epoch": table["epoch"],
            "val_acc_pct": table["val_acc"] * 100.0,
            "acc_gain_points": gain,
            "energy_it_kwh": energy,
            "cumulative_energy_it_kwh": energy.cumsum(),
            "emissions_kg": energy * scale,
            "cumulative_emissions_kg": energy.cumsum() * scale,
        }
    )
    result["marginal_kg_per_point"] = (energy * scale / gain).where(gain > 0)
    return result


@dataclass(frozen=True)
class EarlyStopPoint:
    """First epoch within ``tolerance_points`` of the best validation accuracy."""

    epoch: int
    val_acc_pct: float
    energy_share_pct: float
    tolerance_points: float


def early_stop_point(tradeoff: pd.DataFrame, tolerance_points: float = 1.0) -> EarlyStopPoint:
    """Locate the epoch and energy share needed to get close to the best accuracy."""
    best = tradeoff["val_acc_pct"].max()
    row = tradeoff[tradeoff["val_acc_pct"] >= best - tolerance_points].iloc[0]
    total = tradeoff["energy_it_kwh"].sum()
    return EarlyStopPoint(
        epoch=int(row["epoch"]),
        val_acc_pct=float(row["val_acc_pct"]),
        energy_share_pct=(
            100.0 * float(row["cumulative_energy_it_kwh"]) / float(total) if total > 0 else math.nan
        ),
        tolerance_points=tolerance_points,
    )


def aggregate_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean, sample standard deviation and count per comparable configuration.

    Only repeats of one configuration are averaged. Runs with a different epoch count
    or platform are different experiments, even with the same model and device, and test
    and validation accuracy are different measures.
    """
    grouped = runs.groupby(CONFIGURATION_KEYS, sort=True)
    summary = grouped.agg(
        n=("run_id", "count"),
        energy_it_kwh_mean=("energy_it_kwh", "mean"),
        energy_it_kwh_std=("energy_it_kwh", "std"),
        duration_s_mean=("duration_s", "mean"),
        duration_s_std=("duration_s", "std"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
    )
    return summary.reset_index()


def repeated_seed_configurations(runs: pd.DataFrame) -> list[str]:
    """List configurations in which a seed repeats, so its runs are not independent repeats."""
    repeated = runs[runs.duplicated([*CONFIGURATION_KEYS, "seed"], keep=False)]
    labels = {
        f"{row.model} / {row.dataset} / {row.device} / {row.platform} / {row.epochs} epochs"
        for row in repeated.itertuples()
    }
    return sorted(labels)


def scenario_matrix(
    grouped: pd.DataFrame,
    pues: Iterable[float],
    factors: Mapping[str, float],
    keys: list[str],
) -> pd.DataFrame:
    """Price each configuration under each country and PUE, with the spread across repeats.

    ``grouped`` holds ``keys`` plus ``mean`` and ``std`` of IT energy in kWh. The chain is
    linear, so the spread scales by the same multiplier as the mean.
    """
    rows = []
    for record in grouped.to_dict("records"):
        for iso, factor in factors.items():
            for pue in pues:
                rows.append(
                    {
                        **{key: record[key] for key in keys},
                        "iso": iso,
                        "pue": float(pue),
                        "emissions_kg_mean": emissions_kg(record["mean"], pue, factor),
                        "emissions_kg_std": float(record["std"]) * float(pue) * factor,
                    }
                )
    return pd.DataFrame(rows)
