import filecmp
import math
from pathlib import Path

import pandas as pd
import pytest

from carbonmindful import calc, charts, fmt, report
from carbonmindful.bundle import parse_bundle
from carbonmindful.demo import demo_bundles
from carbonmindful.factors import DEFAULT_PATH, load_factors


@pytest.mark.parametrize(
    "kg,text",
    [(2.5, "2.5 kg CO₂e"), (0.000269, "269 mg CO₂e"), (0.0123, "12.3 g CO₂e"), (0.0, "0 mg CO₂e")],
)
def test_co2e_units(kg, text):
    assert fmt.co2e(kg) == text


def test_tiny_value_is_not_displayed_as_zero():
    assert fmt.co2e(2.8e-6) == "2.8 mg CO₂e"
    assert fmt.co2e(2.8e-9) == "0.0028 mg CO₂e"


def test_non_finite_values_show_na():
    assert fmt.co2e(math.nan) == "n/a"
    assert fmt.energy(math.inf) == "n/a"
    assert fmt.duration(math.nan) == "n/a"


@pytest.mark.parametrize("seconds,text", [(5, "5.0 s"), (120, "2.0 min"), (7200, "2.00 h")])
def test_duration(seconds, text):
    assert fmt.duration(seconds) == text


def test_registry_has_all_fields_and_matches_experiment_copy():
    factors = load_factors()
    assert set(factors) == {"CAN", "MEX", "KOR", "MNG"}
    for factor in factors.values():
        assert factor.year == 2025 and factor.kg_per_kwh > 0 and factor.source
    experiment_copy = Path(__file__).resolve().parents[2] / "experiments" / "grid_factors.csv"
    if experiment_copy.exists():
        assert filecmp.cmp(DEFAULT_PATH, experiment_copy, shallow=False)


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda t: t.drop(columns="source"), "missing"),
        (lambda t: pd.concat([t, t.iloc[[0]]]), "duplicate"),
        (lambda t: t.assign(source=None), "empty"),
        (lambda t: t.assign(factor_kg_co2e_per_kwh=0), "non-positive"),
    ],
)
def test_registry_rejects_bad_files(tmp_path, mutate, message):
    path = tmp_path / "factors.csv"
    mutate(pd.read_csv(DEFAULT_PATH)).to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        load_factors(path)


@pytest.fixture
def run():
    return parse_bundle(demo_bundles()[0])


def _spec(chart):
    return chart.to_dict()


def test_every_chart_builds_a_valid_spec(run):
    factors = {iso: f.kg_per_kwh for iso, f in load_factors().items()}
    table = calc.scenario_table(run.energy_it_kwh, [1.2, 1.6], factors)
    table["country"] = table["iso"]
    table["pue_label"] = table["pue"].map(lambda value: f"PUE {value:.2f}")
    tradeoff = calc.epoch_tradeoff(run.epochs, 1.2, 0.4)
    runs = pd.DataFrame(
        {
            "run_id": ["a", "b"],
            "model": ["m", "m"],
            "dataset": ["d", "d"],
            "device": ["cpu", "cpu"],
            "platform": ["p", "p"],
            "epochs": [3, 3],
            "energy_it_kwh": [1e-3, 2e-3],
            "duration_s": [1.0, 2.0],
            "accuracy": [0.5, 0.6],
            "accuracy_type": ["Test", "Test"],
        }
    )
    specs = [
        charts.regional_bars(table[table.pue == 1.2]),
        charts.pue_bars(table),
        charts.tradeoff_curve(tradeoff),
        charts.marginal_bars(tradeoff),
        charts.utilisation_lines(run.samples),
        charts.comparison_points(calc.aggregate_runs(runs)),
    ]
    assert all(_spec(chart) for chart in specs)


def test_long_series_is_thinned_for_the_browser():
    samples = pd.DataFrame({"timestamp": range(100_000), "cpu_util_pct": [50.0] * 100_000})
    spec = charts.utilisation_lines(samples).to_dict()
    rows = next(iter(spec["datasets"].values()))
    assert len(rows) <= 2 * charts.MAX_LINE_POINTS


def test_utilisation_ignores_empty_gpu_column():
    samples = pd.DataFrame(
        {"timestamp": [1, 2], "cpu_util_pct": [10.0, 20.0], "gpu_util_pct": [None, None]}
    )
    rows = next(iter(charts.utilisation_lines(samples).to_dict()["datasets"].values()))
    assert {row["device"] for row in rows} == {"CPU"}


def test_worked_example_reproduces_the_chain(run):
    factor = load_factors()["KOR"]
    text = report.worked_example(run, factor, 1.2)
    expected = run.energy_it_kwh * 1.2 * factor.kg_per_kwh
    assert f"{expected:.6g} kg CO₂e" in text and "South Korea, 2025" in text


def test_reconciliation_states_the_tracker_is_a_cross_check(run):
    assert report.tracker_reconciliation(run, load_factors()) is None
    summary = {**run.summary, "tracker_emissions_kg": 0.5, "tracker_country_iso": "KOR"}
    bundle = parse_bundle_with(summary, run)
    text = report.tracker_reconciliation(bundle, load_factors())
    assert "cross-check" in text and "South Korea" in text


def parse_bundle_with(summary, run):
    from carbonmindful.bundle import build_bundle

    return parse_bundle(
        build_bundle(summary, run.epochs.drop(columns="energy_it_kwh"), run.samples)
    )


def test_report_flags_synthetic_and_scenarios(run):
    factors = load_factors()
    table = calc.scenario_table(
        run.energy_it_kwh, [1.2], {iso: f.kg_per_kwh for iso, f in factors.items()}
    )
    text = report.markdown_report(run, factors, table)
    assert "SYNTHETIC DEMO DATA" in text
    assert "not run in these countries" in text
    assert "Ember (2026)" in text


def test_csv_export_neutralises_formulas():
    frame = pd.DataFrame({"name": ["=2+2", "@x", "-y", "ok"], "value": [-1.0, 2.0, 3.0, 4.0]})
    lines = report.safe_csv(frame).decode().splitlines()
    assert lines[1] == "'=2+2,-1.0" and lines[2].startswith("'@x") and lines[4] == "ok,4.0"


def test_report_records_version_notes_and_factor_year(run):
    factors = load_factors()
    table = calc.scenario_table(run.energy_it_kwh, [1.2], {"CAN": factors["CAN"].kg_per_kwh})
    text = report.markdown_report(run, {"CAN": factors["CAN"]}, table)
    assert "CarbonMindful 1.0.0" in text and "2025" in text


def test_export_table_carries_factor_provenance(run):
    factors = load_factors()
    table = calc.scenario_table(run.energy_it_kwh, [1.2], {"CAN": factors["CAN"].kg_per_kwh})
    lines = report.scenario_export(table, factors).decode().splitlines()
    assert "factor_year" in lines[0] and "factor_source" in lines[0] and "2025" in lines[1]


def test_registry_uses_verified_ember_values():
    factors = load_factors()
    assert factors["CAN"].kg_per_kwh == pytest.approx(0.191587)
    assert factors["MEX"].kg_per_kwh == pytest.approx(0.481442)
    assert factors["KOR"].kg_per_kwh == pytest.approx(0.417659)
    assert factors["MNG"].kg_per_kwh == pytest.approx(0.82)
    assert not any(factor.provisional for factor in factors.values())
    assert "assumed coal" in factors["MNG"].note
