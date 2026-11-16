"""CarbonMindful dashboard. Run with: streamlit run app.py"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st

from carbonmindful import __version__, calc, charts, fmt, report
from carbonmindful.bundle import BundleError, RunBundle, parse_bundle
from carbonmindful.demo import demo_bundles
from carbonmindful.factors import GridFactor, load_factors

SCENARIO_NOTE = (
    "Scenario-based estimates: the recorded energy is re-priced with each country's grid "
    "factor. The workload was not run in these countries."
)
STYLE = """
<style>
.block-container {padding-top: 2.5rem;}
[data-testid="stMetricValue"] {font-size: 1.55rem;}
</style>
"""
Factors = dict[str, GridFactor]


@dataclass(frozen=True)
class Controls:
    """The scenario chosen in the sidebar."""

    sources: list[tuple[str, bytes]]
    regions: list[str]
    reference_iso: str
    reference_pue: float
    comparison_pue: float


@st.cache_data(show_spinner=False, max_entries=128)
def _parse(data: bytes) -> RunBundle:
    return parse_bundle(data)


@st.cache_data(show_spinner=False)
def _factors() -> Factors:
    return load_factors()


def _safe_name(name: str) -> str:
    """File names reach Markdown-rendered messages, so only plain characters are kept."""
    return re.sub(r"[^\w .-]", "_", name)


def _load_runs(sources: list[tuple[str, bytes]]) -> dict[str, RunBundle]:
    runs: dict[str, RunBundle] = {}
    for name, data in sources:
        label = _safe_name(name)
        try:
            bundle = _parse(data)
        except BundleError as exc:
            st.error(f"{label}: {exc}")
            continue
        if bundle.run_id in runs:
            st.warning(f"{label}: run ID {bundle.run_id} is already loaded, so it was skipped.")
            continue
        runs[bundle.run_id] = bundle
    return runs


def _run_table(runs: dict[str, RunBundle]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_id": run.run_id,
                "model": run.summary["model"],
                "dataset": run.summary["dataset"],
                "device": run.summary["device"],
                "platform": run.platform,
                "epochs": len(run.epochs),
                "seed": run.summary["seed"],
                "energy_it_kwh": run.energy_it_kwh,
                "duration_s": float(run.summary["duration_s"]),
                "accuracy": run.accuracy,
                "accuracy_type": run.accuracy_label.split()[0],
            }
            for run in runs.values()
        ]
    )


def _choose_sources(uploaded: list[tuple[str, bytes]], use_demo: bool) -> list[tuple[str, bytes]]:
    """Uploaded runs always win, so synthetic values never mix with real ones."""
    if uploaded or not use_demo:
        return uploaded
    return [(f"demo {i}", data) for i, data in enumerate(demo_bundles())]


def _sidebar(factors: Factors) -> Controls:
    with st.sidebar:
        st.header("Run data")
        uploads = st.file_uploader("Run bundles (.zip)", type="zip", accept_multiple_files=True)
        # Uploaded and synthetic runs are never loaded together, so fake values
        # cannot enter an average of real repeats.
        use_demo = st.toggle(
            "Use synthetic demo data",
            value=True,
            disabled=bool(uploads),
            help="Ignored once a bundle is uploaded.",
        )
        st.header("Scenario")
        names = {iso: factor.country for iso, factor in factors.items()}
        regions = st.multiselect(
            "Countries", list(names), default=list(names), format_func=names.__getitem__
        )
        reference_iso = st.selectbox(
            "Reference country",
            regions or list(names),
            format_func=names.__getitem__,
            help="Used for the Overview and Accuracy trade-off tabs.",
        )
        reference_pue = st.number_input(
            "Reference PUE",
            1.0,
            3.0,
            1.54,
            0.01,
            format="%.2f",
            help="Default: Uptime Institute 2025 survey average, as cited by Google.",
        )
        comparison_pue = st.number_input(
            "Comparison PUE",
            1.0,
            3.0,
            1.09,
            0.01,
            format="%.2f",
            help="Default: Google fleet-wide PUE, Q2 2026.",
        )
        st.caption(f"CarbonMindful {__version__}")
    return Controls(
        _choose_sources([(upload.name, upload.getvalue()) for upload in uploads or []], use_demo),
        regions,
        str(reference_iso),
        round(float(reference_pue), 2),
        round(float(comparison_pue), 2),
    )


def _notices(runs: dict[str, RunBundle], factors: Factors, regions: list[str]) -> None:
    notes = []
    if any(run.synthetic for run in runs.values()):
        notes.append("Synthetic demo data is loaded. These values are not experimental results.")
    if any(factors[iso].provisional for iso in regions):
        notes.append("Some grid factors are provisional and must be verified before publication.")
    if notes:
        st.warning("  \n".join(notes))


def _scenarios(
    run: RunBundle, factors: Factors, regions: list[str], pues: list[float]
) -> pd.DataFrame:
    selected = {iso: factors[iso].kg_per_kwh for iso in regions}
    table = calc.scenario_table(run.energy_it_kwh, pues, selected)
    table.insert(1, "country", table["iso"].map(lambda iso: factors[iso].country))
    return table


def _scenario_columns() -> dict[str, Any]:
    return {
        "iso": None,
        "country": "Scenario",
        "pue": st.column_config.NumberColumn("PUE", format="%.2f"),
        "factor_kg_per_kwh": st.column_config.NumberColumn("kg CO₂e/kWh", format="%.5f"),
        "energy_it_kwh": st.column_config.NumberColumn("IT energy (kWh)", format="%.6g"),
        "energy_facility_kwh": st.column_config.NumberColumn(
            "Facility energy (kWh)", format="%.6g"
        ),
        "emissions_kg": st.column_config.NumberColumn("Estimated kg CO₂e", format="%.6g"),
    }


def _overview(run: RunBundle, factors: Factors, controls: Controls) -> None:
    factor = factors[controls.reference_iso]
    pue = controls.reference_pue
    emissions = calc.emissions_kg(run.energy_it_kwh, pue, factor.kg_per_kwh)
    columns = st.columns(4)
    columns[0].metric("IT energy", fmt.energy(run.energy_it_kwh), border=True)
    columns[1].metric(
        f"Estimate, {factor.country} scenario",
        fmt.co2e(emissions),
        help=f"At PUE {pue:.2f}. The workload was not run in this country.",
        border=True,
    )
    columns[2].metric(run.accuracy_label, f"{run.accuracy * 100:.2f}%", border=True)
    columns[3].metric("Duration", fmt.duration(float(run.summary["duration_s"])), border=True)
    st.subheader("How this estimate is calculated")
    st.code(report.worked_example(run, factor, pue), language=None, wrap_lines=True)
    st.caption(f"Energy source: {run.summary.get('energy_source', 'not recorded')}.")
    breakdown = report.energy_breakdown(run)
    if breakdown:
        st.caption(breakdown)
    reconciliation = report.tracker_reconciliation(run, factors)
    if reconciliation:
        st.caption(reconciliation)
    for message in run.warnings:
        st.caption(message)


def _regional(run: RunBundle, factors: Factors, controls: Controls) -> None:
    st.info(SCENARIO_NOTE)
    if not controls.regions:
        st.warning("Select at least one country.")
        return
    table = _scenarios(run, factors, controls.regions, [controls.reference_pue])
    st.altair_chart(charts.regional_bars(table), width="stretch")
    ranked = table.sort_values("emissions_kg")
    low, high = ranked.iloc[0], ranked.iloc[-1]
    if len(table) > 1 and high["emissions_kg"] > 0:
        saving = calc.saving_pct(high["emissions_kg"], low["emissions_kg"])
        st.markdown(
            f"The {low['country']} scenario is **{saving:.1f}% lower** than the "
            f"{high['country']} scenario. This ratio depends only on the grid factors."
        )
    with st.expander("Data table"):
        st.dataframe(table, hide_index=True, width="stretch", column_config=_scenario_columns())
    st.subheader("Grid factors")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Country": factors[iso].country,
                    "kg CO₂e/kWh": factors[iso].kg_per_kwh,
                    "Year": factors[iso].year,
                    "Metric": factors[iso].metric,
                    "Source": f"{factors[iso].original_source}, {factors[iso].source}",
                    "Retrieved": factors[iso].retrieved,
                    "Status": factors[iso].status,
                    "Note": factors[iso].note,
                }
                for iso in controls.regions
            ]
        ),
        hide_index=True,
        width="stretch",
    )


def _pue(run: RunBundle, factors: Factors, controls: Controls) -> None:
    st.info(SCENARIO_NOTE)
    if not controls.regions:
        st.warning("Select at least one country.")
        return
    reference, comparison = controls.reference_pue, controls.comparison_pue
    table = _scenarios(run, factors, controls.regions, sorted({reference, comparison}))
    chart_data = table.assign(pue_label=table["pue"].map(lambda value: f"PUE {value:.2f}"))
    st.altair_chart(charts.pue_bars(chart_data), width="stretch")
    if reference != comparison:
        change = (comparison / reference - 1) * 100
        st.markdown(
            f"Moving from PUE {reference:.2f} to {comparison:.2f} changes every estimate by "
            f"**{change:+.1f}%**. The change is the same in all countries because PUE is a "
            "multiplier in the formula."
        )
    st.caption(
        "PUE describes data-centre overhead. Applying it to a run recorded on other hardware "
        "is a hypothetical scenario."
    )
    with st.expander("Data table"):
        st.dataframe(table, hide_index=True, width="stretch", column_config=_scenario_columns())


def _tradeoff(run: RunBundle, factors: Factors, controls: Controls) -> None:
    if "energy_it_kwh" not in run.epochs:
        st.warning("This bundle has no per-epoch energy, so the trade-off cannot be shown.")
        return
    factor = factors[controls.reference_iso]
    pue = controls.reference_pue
    table = calc.epoch_tradeoff(run.epochs, pue, factor.kg_per_kwh)
    left, right = st.columns(2)
    with left:
        st.subheader("Accuracy against energy")
        st.altair_chart(charts.tradeoff_curve(table), width="stretch")
    with right:
        st.subheader("Marginal cost of accuracy")
        if table["marginal_kg_per_point"].notna().any():
            st.altair_chart(charts.marginal_bars(table), width="stretch")
        else:
            st.info("No epoch after the first improved validation accuracy.")
    total = calc.emissions_kg(run.energy_it_kwh, pue, factor.kg_per_kwh)
    per_point = (
        calc.kg_per_accuracy_point(total, run.accuracy) if run.accuracy > 0 else float("nan")
    )
    stop = calc.early_stop_point(table, 1.0)
    first, second = st.columns(2)
    first.metric(
        f"CO₂e per accuracy point, {factor.country} scenario",
        fmt.co2e(per_point),
        border=True,
    )
    second.metric(
        "Energy used to get within 1 point of best",
        "n/a" if math.isnan(stop.energy_share_pct) else f"{stop.energy_share_pct:.0f}%",
        help=f"Reached at epoch {stop.epoch} with {stop.val_acc_pct:.2f}% validation accuracy.",
        border=True,
    )
    st.caption(
        "CO₂e per accuracy point compares runs of the same task and dataset. It does not rank "
        "different tasks, because one point of accuracy does not mean the same thing across them."
    )
    with st.expander("Data table"):
        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            column_config={
                "epoch": "Epoch",
                "val_acc_pct": st.column_config.NumberColumn(
                    "Validation accuracy (%)", format="%.2f"
                ),
                "acc_gain_points": st.column_config.NumberColumn("Points gained", format="%.2f"),
                "energy_it_kwh": st.column_config.NumberColumn("IT energy (kWh)", format="%.6g"),
                "cumulative_energy_it_kwh": st.column_config.NumberColumn(
                    "Cumulative IT energy (kWh)", format="%.6g"
                ),
                "emissions_kg": st.column_config.NumberColumn("Estimated kg CO₂e", format="%.6g"),
                "cumulative_emissions_kg": st.column_config.NumberColumn(
                    "Cumulative kg CO₂e", format="%.6g"
                ),
                "marginal_kg_per_point": st.column_config.NumberColumn(
                    "kg CO₂e per point gained", format="%.6g"
                ),
            },
        )


def _hardware(run: RunBundle) -> None:
    if run.samples.empty:
        st.info("This bundle has no utilisation samples.")
    else:
        st.altair_chart(charts.utilisation_lines(run.samples), width="stretch")
    tiles: list[tuple[str, str]] = []
    for column, label in (
        ("cpu_util_pct", "Mean CPU utilisation"),
        ("gpu_util_pct", "Mean GPU utilisation"),
    ):
        if column in run.samples and run.samples[column].notna().any():
            tiles.append((label, f"{run.samples[column].mean():.0f}%"))
    if "samples_per_s" in run.epochs and run.epochs["samples_per_s"].notna().any():
        tiles.append(("Mean throughput", f"{run.epochs['samples_per_s'].mean():,.0f} samples/s"))
    if tiles:
        for slot, (label, value) in zip(st.columns(3), tiles, strict=False):
            slot.metric(label, value, border=True)
    st.caption(
        "High utilisation shows that a device was busy. It does not show efficient use, and on "
        "a CPU it may indicate a bottleneck. Read it together with throughput."
    )


def _compare(runs: dict[str, RunBundle], factors: Factors, controls: Controls) -> None:
    table = _run_table(runs)
    aggregate = calc.aggregate_runs(table)
    repeated = calc.repeated_seed_configurations(table)
    if repeated:
        st.warning(
            "A seed appears more than once in: "
            + "; ".join(repeated)
            + ". Those runs are not independent repeats, so their spread may be understated."
        )
    st.altair_chart(charts.comparison_points(aggregate), width="stretch")
    st.caption(
        "Points show the mean and lines show one sample standard deviation. Only runs with the "
        "same model, dataset, device, platform and epoch count are averaged."
    )
    st.dataframe(
        aggregate,
        hide_index=True,
        width="stretch",
        column_config={
            "model": "Model",
            "dataset": "Dataset",
            "device": "Device",
            "platform": "Platform",
            "epochs": "Epochs",
            "accuracy_type": "Accuracy type",
            "n": "Runs",
            "energy_it_kwh_mean": st.column_config.NumberColumn(
                "Mean IT energy (kWh)", format="%.4g"
            ),
            "energy_it_kwh_std": st.column_config.NumberColumn("SD (kWh)", format="%.2g"),
            "duration_s_mean": st.column_config.NumberColumn("Mean duration (s)", format="%.1f"),
            "duration_s_std": st.column_config.NumberColumn("SD (s)", format="%.1f"),
            "accuracy_mean": st.column_config.NumberColumn("Mean accuracy", format="%.4f"),
            "accuracy_std": st.column_config.NumberColumn("SD", format="%.4f"),
        },
    )
    st.subheader("Estimated emissions by configuration and scenario")
    st.caption(SCENARIO_NOTE + " The range is one sample standard deviation across repeats.")
    regions = controls.regions or list(factors)
    grouped = aggregate.rename(columns={"energy_it_kwh_mean": "mean", "energy_it_kwh_std": "std"})
    matrix = calc.scenario_matrix(
        grouped,
        sorted({controls.reference_pue, controls.comparison_pue}),
        {iso: factors[iso].kg_per_kwh for iso in regions},
        ["model", "dataset", "device", "platform", "epochs"],
    )
    matrix.insert(0, "country", matrix["iso"].map(lambda iso: factors[iso].country))
    st.dataframe(
        matrix,
        hide_index=True,
        width="stretch",
        column_config={
            "iso": None,
            "country": "Scenario",
            "model": "Model",
            "dataset": "Dataset",
            "device": "Device",
            "platform": "Platform",
            "epochs": "Epochs",
            "pue": st.column_config.NumberColumn("PUE", format="%.2f"),
            "emissions_kg_mean": st.column_config.NumberColumn("Estimated kg CO₂e", format="%.4g"),
            "emissions_kg_std": st.column_config.NumberColumn("SD (kg)", format="%.2g"),
        },
    )
    with st.expander("All runs"):
        st.dataframe(table, hide_index=True, width="stretch")


def _export(run: RunBundle, factors: Factors, controls: Controls) -> None:
    regions = controls.regions or list(factors)
    pues = sorted({controls.reference_pue, controls.comparison_pue})
    table = _scenarios(run, factors, regions, pues)
    first, second = st.columns(2)
    first.download_button(
        "Download scenarios (CSV)",
        report.scenario_export(table, factors),
        f"{run.run_id}_scenarios.csv",
        "text/csv",
        width="stretch",
    )
    text = report.markdown_report(run, {iso: factors[iso] for iso in regions}, table)
    second.download_button(
        "Download run report (Markdown)",
        text,
        f"{run.run_id}_report.md",
        "text/markdown",
        width="stretch",
    )
    with st.expander("Report preview", expanded=True):
        st.markdown(text)


def main() -> None:
    st.set_page_config(page_title="CarbonMindful", layout="wide")
    st.markdown(STYLE, unsafe_allow_html=True)
    st.title("CarbonMindful")
    st.caption("Carbon footprint calculation for AI model training")
    factors = _factors()
    controls = _sidebar(factors)
    runs = _load_runs(controls.sources)
    if not runs:
        st.info("Upload a run bundle, or turn on the synthetic demo data, to begin.")
        return
    _notices(runs, factors, controls.regions)
    run = runs[st.selectbox("Run", list(runs))]
    tabs = st.tabs(
        [
            "Overview",
            "Regional scenarios",
            "PUE sensitivity",
            "Accuracy trade-off",
            "Hardware",
            "Compare runs",
            "Export",
        ]
    )
    with tabs[0]:
        _overview(run, factors, controls)
    with tabs[1]:
        _regional(run, factors, controls)
    with tabs[2]:
        _pue(run, factors, controls)
    with tabs[3]:
        _tradeoff(run, factors, controls)
    with tabs[4]:
        _hardware(run)
    with tabs[5]:
        _compare(runs, factors, controls)
    with tabs[6]:
        _export(run, factors, controls)


if __name__ == "__main__":
    main()
