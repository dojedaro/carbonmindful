import math

import pandas as pd
import pytest

from carbonmindful import calc


def test_chain_applies_pue_once():
    assert calc.emissions_kg(0.001, 1.2, 0.5) == pytest.approx(0.0006)


@pytest.mark.parametrize(
    "energy,pue,factor",
    [(-1, 1.2, 0.5), (1, 0.9, 0.5), (1, 1.2, -0.1), (math.nan, 1.2, 0.5), (1, math.inf, 0.5)],
)
def test_invalid_inputs_rejected(energy, pue, factor):
    with pytest.raises(ValueError):
        calc.emissions_kg(energy, pue, factor)


def test_scenario_table_is_proportional_to_pue():
    table = calc.scenario_table(0.001, [1.2, 1.6], {"AAA": 0.1, "BBB": 0.8})
    assert len(table) == 4
    low = table[(table.iso == "BBB") & (table.pue == 1.2)].emissions_kg.item()
    high = table[(table.iso == "BBB") & (table.pue == 1.6)].emissions_kg.item()
    assert high / low == pytest.approx(1.6 / 1.2)


def test_saving_uses_grid_ratio_only():
    assert calc.saving_pct(0.81631, 0.19072) == pytest.approx(76.64, abs=0.01)


def test_accuracy_metric_uses_percentage_points():
    assert calc.kg_per_accuracy_point(0.000269, 0.9514) == pytest.approx(2.827e-6, rel=1e-3)


def _epochs():
    return pd.DataFrame(
        {
            "epoch": [1, 2, 3, 4],
            "val_acc": [0.90, 0.95, 0.95, 0.94],
            "energy_it_kwh": [1.0, 1.0, 1.0, 1.0],
        }
    )


def test_marginal_is_undefined_without_gain():
    table = calc.epoch_tradeoff(_epochs(), 1.0, 1.0)
    assert math.isnan(table.marginal_kg_per_point[0])
    assert table.marginal_kg_per_point[1] == pytest.approx(0.2)
    assert math.isnan(table.marginal_kg_per_point[2])
    assert math.isnan(table.marginal_kg_per_point[3])


def test_early_stop_point():
    point = calc.early_stop_point(calc.epoch_tradeoff(_epochs(), 1.0, 1.0), 1.0)
    assert point.epoch == 2
    assert point.energy_share_pct == pytest.approx(50.0)


def _runs(**overrides):
    base = {
        "run_id": list("abc"),
        "model": ["m"] * 3,
        "dataset": ["d"] * 3,
        "device": ["cpu", "cpu", "cuda"],
        "platform": ["laptop"] * 3,
        "epochs": [3] * 3,
        "energy_it_kwh": [1.0, 3.0, 5.0],
        "duration_s": [1.0, 1.0, 1.0],
        "accuracy": [0.5, 0.7, 0.9],
        "accuracy_type": ["Test"] * 3,
        "seed": [1, 2, 3],
    }
    return pd.DataFrame({**base, **overrides})


def test_aggregate_groups_and_counts():
    table = calc.aggregate_runs(_runs())
    cpu = table[table.device == "cpu"].iloc[0]
    assert cpu.n == 2 and cpu.energy_it_kwh_mean == 2.0
    assert cpu.energy_it_kwh_std == pytest.approx(math.sqrt(2))


@pytest.mark.parametrize(
    "overrides",
    [{"epochs": [3, 20, 3]}, {"platform": ["laptop", "colab", "laptop"]}],
)
def test_aggregate_never_averages_different_experiments(overrides):
    table = calc.aggregate_runs(_runs(**overrides))
    assert len(table) == 3 and (table.n == 1).all()


def test_epoch_emissions_apply_pue_and_factor():
    table = calc.epoch_tradeoff(_epochs(), 1.5, 0.4)
    assert table.emissions_kg[0] == pytest.approx(1.0 * 1.5 * 0.4)
    assert table.cumulative_emissions_kg[3] == pytest.approx(4 * 1.5 * 0.4)
    assert table.marginal_kg_per_point[1] == pytest.approx(0.6 / 5.0)


def test_epoch_totals_agree_with_the_run_level_chain():
    table = calc.epoch_tradeoff(_epochs(), 1.5, 0.4)
    assert table.emissions_kg.sum() == pytest.approx(calc.emissions_kg(4.0, 1.5, 0.4))


def test_zero_energy_run_does_not_crash():
    epochs = _epochs().assign(energy_it_kwh=0.0)
    point = calc.early_stop_point(calc.epoch_tradeoff(epochs, 1.0, 1.0))
    assert math.isnan(point.energy_share_pct)


def test_validation_and_test_accuracy_are_not_averaged():
    table = calc.aggregate_runs(_runs(accuracy_type=["Test", "Validation", "Test"]))
    assert len(table) == 3


def test_repeated_seeds_are_reported():
    runs = _runs(seed=[1, 1, 2])
    assert calc.repeated_seed_configurations(runs) == ["m / d / cpu / laptop / 3 epochs"]
    assert calc.repeated_seed_configurations(_runs(seed=[1, 2, 3])) == []
