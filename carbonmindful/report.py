"""Reproducibility text and exports: every reported number can be recalculated."""

from __future__ import annotations

import pandas as pd

from . import __version__, calc, fmt
from .bundle import RunBundle
from .factors import GridFactor

_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def safe_csv(frame: pd.DataFrame) -> bytes:
    """CSV in which text cells cannot run as spreadsheet formulas."""

    def neutralise(value: object) -> object:
        if isinstance(value, str) and value.startswith(_FORMULA_PREFIXES):
            return "'" + value
        return value

    clean = frame.copy()
    for column in clean.columns:
        if not pd.api.types.is_numeric_dtype(clean[column]):
            clean[column] = clean[column].map(neutralise)
    return clean.to_csv(index=False).encode("utf-8")


def scenario_export(scenarios: pd.DataFrame, factors: dict[str, GridFactor]) -> bytes:
    """Scenario CSV with the year and source of each factor, for reproducing the values."""
    table = scenarios.assign(
        factor_year=scenarios["iso"].map(lambda iso: factors[iso].year),
        factor_source=scenarios["iso"].map(
            lambda iso: f"{factors[iso].original_source}, {factors[iso].source}"
        ),
        factor_status=scenarios["iso"].map(lambda iso: factors[iso].status),
    )
    return safe_csv(table)


def worked_example(bundle: RunBundle, factor: GridFactor, pue: float) -> str:
    """Show the calculation with its inputs, so a reader can reproduce the result."""
    energy = bundle.energy_it_kwh
    result = calc.emissions_kg(energy, pue, factor.kg_per_kwh)
    return (
        f"{energy:.6g} kWh (IT) × {pue:g} (PUE) × {factor.kg_per_kwh:.5g} kg CO₂e/kWh "
        f"({factor.country}, {factor.year}) = {result:.6g} kg CO₂e"
    )


def energy_breakdown(bundle: RunBundle) -> str | None:
    """Split recorded energy by component, because their data quality can differ."""
    labels = (("cpu_energy_kwh", "CPU"), ("ram_energy_kwh", "RAM"), ("gpu_energy_kwh", "GPU"))
    parts = [
        f"{label} {fmt.energy(float(bundle.summary[key]))}"
        for key, label in labels
        if bundle.summary.get(key)
    ]
    if not parts:
        return None
    text = "Recorded energy by component: " + ", ".join(parts) + "."
    cpu = bundle.summary.get("cpu_energy_kwh")
    counter = bundle.summary.get("package_counter_kwh")
    if cpu and counter:
        text += f" CPU energy is {float(cpu) / float(counter):.3f} times the independent counter."
    return text


def tracker_reconciliation(bundle: RunBundle, factors: dict[str, GridFactor]) -> str | None:
    """Explain how the tracker's own emissions value relates to the chain used here."""
    native = bundle.summary.get("tracker_emissions_kg")
    iso = bundle.summary.get("tracker_country_iso")
    reported = float(bundle.summary["energy_kwh"])
    if native is None or reported <= 0:
        return None
    implied = float(native) / reported
    text = (
        f"The tracker reported {float(native):.6g} kg CO₂e for {reported:.6g} kWh, "
        f"an implied factor of {implied:.4f} kg CO₂e/kWh"
    )
    if iso in factors:
        factor = factors[iso]
        text += (
            f". The registry uses {factor.kg_per_kwh:.4f} kg CO₂e/kWh for {factor.country} "
            f"({factor.year}). The tracker value is a cross-check and is not used in any scenario."
        )
    else:
        text += ". The tracker value is a cross-check and is not used in any scenario."
    return text


def markdown_report(
    bundle: RunBundle,
    factors: dict[str, GridFactor],
    scenarios: pd.DataFrame,
) -> str:
    """Build a portable run report, including sources and the scenario table."""
    summary = bundle.summary
    lines = [
        f"# CarbonMindful run report: {bundle.run_id}",
        "",
    ]
    if bundle.synthetic:
        lines += ["**SYNTHETIC DEMO DATA. These values are not experimental results.**", ""]
    lines += [
        f"- Workload: {summary['model']} on {summary['dataset']}, device {summary['device']}, "
        f"seed {summary['seed']}",
        f"- IT energy: {bundle.energy_it_kwh:.6g} kWh ({fmt.energy(bundle.energy_it_kwh)})",
        f"- Energy source: {summary.get('energy_source', 'not recorded')}",
        f"- Duration: {fmt.duration(float(summary['duration_s']))}",
        f"- {bundle.accuracy_label}: {bundle.accuracy * 100:.2f}%",
        "",
        "## Scenario-based regional estimates",
        "",
        "The recorded energy is re-priced with each grid factor. "
        "The workload was not run in these countries.",
        "",
        "Formula: emissions = IT energy × PUE × grid factor. "
        f"Calculated with CarbonMindful {__version__}.",
        "",
        "| Scenario | PUE | kg CO₂e/kWh | Estimated kg CO₂e |",
        "|---|---:|---:|---:|",
    ]
    for row in scenarios.sort_values(["iso", "pue"]).itertuples():
        factor = factors[str(row.iso)]
        lines.append(
            f"| {factor.country} | {row.pue:g} | {row.factor_kg_per_kwh:.5g} "
            f"| {row.emissions_kg:.6g} |"
        )
    lines += ["", "## Grid factor sources", ""]
    for factor in factors.values():
        if factor.iso in set(scenarios["iso"]):
            lines.append(
                f"- {factor.country}: {factor.kg_per_kwh:.5g} kg CO₂e/kWh, {factor.year}. "
                f"{factor.metric}. {factor.original_source}, {factor.source}. "
                f"Retrieved {factor.retrieved}. Status: {factor.status}. {factor.note}"
            )
    if bundle.warnings:
        lines += ["", "## Notes", ""]
        lines += [f"- {message}" for message in bundle.warnings]
    return "\n".join(lines) + "\n"
