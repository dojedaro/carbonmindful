import pandas as pd
import pytest

from carbonmindful import calc
from measure import analyze


def _table():
    return pd.DataFrame(
        {
            "run_id": list("abcd"),
            "model": ["cnn"] * 4,
            "dataset": ["CIFAR-10"] * 4,
            "device": ["cpu", "cuda", "cpu", "cuda"],
            "platform": ["colab"] * 4,
            "epochs": [15] * 4,
            "seed": [1, 1, 2, 2],
            "energy_wh": [8.0, 1.0, 12.0, 1.0],
            "gpu_wh": [3.0, 0.7, 4.0, 0.7],
            "ram_wh": [3.0, 0.1, 4.0, 0.1],
            "duration_s": [1200.0, 40.0, 1200.0, 40.0],
            "test_acc_pct": [70.0, 71.0, 72.0, 72.0],
        }
    )


def test_device_comparison_pairs_by_seed():
    pairs = analyze.device_comparison(_table())
    assert len(pairs) == 2
    first = pairs[pairs.seed == 1].iloc[0]
    assert first.energy_ratio == pytest.approx(8.0)
    assert first.energy_ratio_without_idle_gpu == pytest.approx(5.0)
    assert first.time_ratio == pytest.approx(30.0)
    assert first.accuracy_difference_points == pytest.approx(1.0)


def test_device_comparison_never_pairs_across_platforms():
    table = _table()
    table.loc[table.device == "cuda", "platform"] = "other"
    assert analyze.device_comparison(table).empty


def test_background_table_subtracts_idle_power():
    idle = pd.DataFrame(
        {"platform": ["colab", "colab"], "mean_power_w": [18.0, 22.0], "mean_cpu_power_w": [1, 1]}
    )
    result = analyze.background_table(_table(), idle)
    cpu = result[result.device == "cpu"].iloc[0]
    assert cpu.idle_w == pytest.approx(20.0)
    assert cpu.above_idle_wh == pytest.approx(10.0 - 20.0 * 1200 / 3600)
    assert cpu.without_ram_wh == pytest.approx(10.0 - 3.5)


def test_scenario_matrix_applies_the_chain_and_scales_the_spread():
    grouped = pd.DataFrame({"model": ["m"], "mean": [0.01], "std": [0.002]})
    result = calc.scenario_matrix(grouped, [1.5], {"AAA": 0.5}, ["model"])
    row = result.iloc[0]
    assert row.emissions_kg_mean == pytest.approx(0.0075)
    assert row.emissions_kg_std == pytest.approx(0.0015)
    assert (row.iso, row.pue, row.model) == ("AAA", 1.5, "m")


def test_scenario_matrix_size_is_configurations_by_countries_by_pues():
    grouped = pd.DataFrame({"model": ["a", "b"], "mean": [1.0, 2.0], "std": [0.1, 0.2]})
    result = calc.scenario_matrix(grouped, [1.09, 1.54], {"X": 0.2, "Y": 0.8}, ["model"])
    assert len(result) == 8


def test_pue_sources_are_recorded():
    table = pd.read_csv(analyze.PUE_PATH)
    assert list(table["pue"]) == [1.09, 1.54]
    assert table[["source", "period", "retrieved"]].notna().all().all()


def test_early_stop_table_has_single_row_headers():
    from carbonmindful.bundle import parse_bundle
    from carbonmindful.demo import demo_bundles

    runs = [parse_bundle(data) for data in demo_bundles()[:3]]
    table = analyze.early_stop_table(runs)
    assert "energy_share_pct_mean" in table.columns
    assert len(table) == 1 and table.iloc[0].model == "MLP"
