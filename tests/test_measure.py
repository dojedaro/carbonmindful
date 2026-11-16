import json
import zipfile
from types import SimpleNamespace

import pytest
import torch

from carbonmindful.bundle import parse_bundle
from measure import run, sensors, suite, workloads


class FakeTracker:
    """Reports fixed energy for each task, and records the order of calls."""

    def __init__(self):
        self.calls = []

    def start_task(self, name=None):
        self.calls.append(("start", name))

    def stop_task(self, name=None):
        self.calls.append(("stop", name))
        return SimpleNamespace(
            energy_consumed=0.003,
            cpu_energy=0.002,
            ram_energy=0.001,
            gpu_energy=0.0,
            emissions=0.0012,
        )


def _splits(channels=1, size=8, n=256):
    images = torch.randint(0, 255, (n, channels, size, size), dtype=torch.uint8)
    labels = torch.randint(0, 10, (n,))
    dataset = "MNIST" if channels == 1 else "CIFAR-10"
    return workloads.make_splits(images, labels, images[:64], labels[:64], dataset, 32)


def _config(**changes):
    values = dict(model="mlp", dataset="MNIST", epochs=2, seed=1, device="cpu", platform="laptop")
    return run.RunConfig(**{**values, **changes})


def _execute(config=None, tracker=None, splits=None):
    return run.execute(
        config or _config(),
        splits or _splits(),
        tracker or FakeTracker(),
        "KOR",
        {"cpu": "windows_emi", "ram": "constant estimate"},
        "3.3.2",
    )


@pytest.fixture(autouse=True)
def no_counter(monkeypatch):
    monkeypatch.setattr(sensors, "read_package_counter_kwh", lambda: None)


def test_bundle_is_readable_and_energy_is_the_sum_of_measured_epochs():
    bundle = parse_bundle(_execute())
    assert bundle.energy_it_kwh == pytest.approx(0.006)
    assert list(bundle.epochs["energy_kwh"]) == pytest.approx([0.003, 0.003])
    assert not any("sums to" in warning for warning in bundle.warnings)


def test_components_are_recorded_separately():
    summary = parse_bundle(_execute()).summary
    assert summary["cpu_energy_kwh"] == pytest.approx(0.004)
    assert summary["ram_energy_kwh"] == pytest.approx(0.002)
    assert "CPU windows_emi" in summary["energy_source"]
    assert "RAM constant estimate" in summary["energy_source"]


def test_pue_is_never_given_to_the_tracker():
    assert parse_bundle(_execute()).summary["tracker_pue"] == 1.0
    assert not any("counted twice" in w for w in parse_bundle(_execute()).warnings)


def test_each_epoch_is_one_task_in_order():
    tracker = FakeTracker()
    _execute(tracker=tracker)
    assert tracker.calls == [
        ("start", "epoch_1"),
        ("stop", "epoch_1"),
        ("start", "epoch_2"),
        ("stop", "epoch_2"),
    ]


def test_test_set_is_evaluated_outside_the_energy_measurement(monkeypatch):
    events = []
    tracker = FakeTracker()
    original = workloads.evaluate

    def spy(model, x, y, device):
        events.append(("evaluate", len(x), len(tracker.calls)))
        return original(model, x, y, device)

    monkeypatch.setattr(workloads, "evaluate", spy)
    _execute(tracker=tracker)
    test_call = events[-1]
    assert test_call[1] == 64 and test_call[2] == len(tracker.calls)
    assert tracker.calls[-1][0] == "stop"


def test_summary_identifies_no_person_or_machine():
    text = json.dumps(parse_bundle(_execute()).summary)
    import getpass
    import socket

    assert getpass.getuser() not in text
    assert socket.gethostname() not in text
    assert "Users" not in text


def test_same_seed_reproduces_and_a_different_seed_differs():
    splits = _splits()
    first = parse_bundle(_execute(splits=splits)).epochs["train_loss"]
    second = parse_bundle(_execute(splits=splits)).epochs["train_loss"]
    other = parse_bundle(_execute(config=_config(seed=2), splits=splits)).epochs["train_loss"]
    assert list(first) == pytest.approx(list(second))
    assert list(first) != pytest.approx(list(other))


def test_validation_split_is_fixed_and_separate_from_training():
    images = torch.arange(256).view(256, 1, 1, 1).repeat(1, 1, 2, 2).to(torch.uint8)
    labels = torch.arange(256) % 10
    first = workloads.make_splits(images, labels, images[:8], labels[:8], "MNIST", 32)
    second = workloads.make_splits(images, labels, images[:8], labels[:8], "MNIST", 32)
    assert torch.equal(first.val_x, second.val_x)
    assert len(first.train_x) == 224 and len(first.val_x) == 32
    train = {tuple(row.flatten().tolist()) for row in first.train_x}
    val = {tuple(row.flatten().tolist()) for row in first.val_x}
    assert not train & val


def test_normalisation_uses_the_dataset_statistics():
    images = torch.full((1, 1, 2, 2), 255, dtype=torch.uint8)
    splits = workloads.make_splits(images, torch.tensor([1]), images, torch.tensor([1]), "MNIST", 0)
    assert splits.test_x[0, 0, 0, 0].item() == pytest.approx((1 - 0.1307) / 0.3081, rel=1e-5)


def test_mlp_matches_the_submitted_architecture():
    model = workloads.build_model("mlp", torch.zeros(1, 1, 28, 28))
    sizes = [layer.out_features for layer in model.net if isinstance(layer, torch.nn.Linear)]
    assert sizes == [128, 64, 10]
    assert sum(p.numel() for p in model.parameters()) == 109386


def test_cnn_accepts_cifar_shaped_input_and_trains():
    splits = _splits(channels=3, size=32)
    bundle = parse_bundle(
        _execute(config=_config(model="cnn", dataset="CIFAR-10", epochs=1), splits=splits)
    )
    assert bundle.summary["parameters"] == 94986
    assert 0 <= bundle.accuracy <= 1


def test_unknown_names_are_rejected():
    with pytest.raises(ValueError):
        workloads.build_model("transformer", torch.zeros(1, 1, 2, 2))


def test_plan_interleaves_seeds_and_every_configuration_repeats():
    runs = suite.plan("laptop", 3)
    assert [seed for _, seed in runs[:3]] == [1, 1, 1] and runs[3][1] == 2
    assert len(runs) == 9
    for item in suite.SUITES["laptop"]:
        assert [seed for planned, seed in runs if planned == item] == [1, 2, 3]


def test_gpu_suite_compares_devices_on_one_workload():
    items = suite.SUITES["gpu"]
    assert {item.device for item in items} == {"cpu", "cuda"}
    assert len({(item.model, item.dataset, item.epochs) for item in items}) == 1


def test_original_three_epoch_task_is_in_the_plan():
    assert suite.Planned("mlp", "MNIST", 3) in suite.SUITES["laptop"]


def test_preflight_blocks_battery_and_busy_machine(monkeypatch):
    monkeypatch.setattr(sensors, "power_plugged", lambda: False)
    monkeypatch.setattr(suite.psutil, "cpu_percent", lambda interval: 90.0)
    assert len(suite.preflight(False, False)) == 2
    assert suite.preflight(True, True) == []


def test_atomic_write_leaves_only_the_finished_file(tmp_path):
    path = tmp_path / "a" / "run.zip"
    run.write_atomically(path, b"data")
    assert path.read_bytes() == b"data"
    assert [item.name for item in path.parent.iterdir()] == ["run.zip"]


def test_idle_measurement_reports_mean_power(monkeypatch):
    monkeypatch.setattr(run.time, "sleep", lambda seconds: None)
    result = run.measure_idle(3600, FakeTracker())
    assert result["mean_power_w"] == pytest.approx(3.0)
    assert result["mean_cpu_power_w"] == pytest.approx(2.0)


def test_bundle_contains_the_expected_files():
    import io

    names = zipfile.ZipFile(io.BytesIO(_execute())).namelist()
    assert set(names) == {"summary.json", "epochs.csv", "samples.csv"}


class SlowSetupTracker(FakeTracker):
    """Behaves like CodeCarbon: the first start_task performs slow hardware setup."""

    def __init__(self):
        super().__init__()
        self.ready = False

    def start_task(self, name=None):
        if not self.ready:
            import time

            time.sleep(0.3)
            self.ready = True
        super().start_task(name)


def test_tracker_setup_is_not_timed_as_training():
    tracker = SlowSetupTracker()
    run.prepare_tracker(tracker)
    epochs = parse_bundle(_execute(tracker=tracker)).epochs
    assert epochs["duration_s"].iloc[0] < 0.25


def test_modes_label_each_component():
    class Cpu:
        _mode = "windows_emi"

    class Ram:
        pass

    class Gpu:
        pass

    tracker = SimpleNamespace(_hardware=[Cpu(), Ram(), Gpu()])
    assert run.tracker_modes(tracker) == {
        "cpu": "windows_emi",
        "ram": "constant estimate",
        "gpu": "nvml",
    }


def test_counter_cross_check_is_recorded_per_epoch_and_in_total(monkeypatch):
    readings = iter([1.0, 1.5, 2.0, 2.25])
    monkeypatch.setattr(sensors, "read_package_counter_kwh", lambda: next(readings))
    bundle = parse_bundle(_execute())
    assert list(bundle.epochs["package_counter_kwh"]) == pytest.approx([0.5, 0.25])
    assert bundle.summary["package_counter_kwh"] == pytest.approx(0.75)


def test_missing_counter_leaves_no_summary_value():
    assert "package_counter_kwh" not in parse_bundle(_execute()).summary


def test_breakdown_reports_components_and_the_cross_check(monkeypatch):
    from carbonmindful import report

    readings = iter([0.0, 0.002, 0.002, 0.004])
    monkeypatch.setattr(sensors, "read_package_counter_kwh", lambda: next(readings))
    text = report.energy_breakdown(parse_bundle(_execute()))
    assert "CPU 4 Wh" in text and "RAM 2 Wh" in text and "1.000 times" in text


@pytest.mark.skipif(__import__("sys").platform != "win32", reason="Windows counter")
def test_real_package_counter_advances(monkeypatch):
    monkeypatch.undo()
    import time

    sensors._package_counter.cache_clear()
    first = sensors.read_package_counter_kwh()
    if first is None:
        pytest.skip("Energy counter is not available on this machine.")
    time.sleep(0.3)
    assert sensors.read_package_counter_kwh() > first


def test_os_label_is_not_misreported_on_windows_11():
    import sys

    if sys.platform == "win32" and sys.getwindowsversion().build >= 22000:
        assert sensors.describe_machine("cpu")["os"] == "Windows 11"


@pytest.mark.parametrize("error", [FileNotFoundError, AttributeError])
def test_machine_without_power_data_is_treated_as_no_battery(monkeypatch, error):
    def missing():
        raise error("no power supply information")

    monkeypatch.setattr(sensors.psutil, "sensors_battery", missing)
    assert sensors.power_plugged() is None
    assert sensors.describe_machine("cpu")["on_mains_power"] is None


def _with_load(total, own):
    from carbonmindful.bundle import build_bundle

    run = parse_bundle(_execute())
    samples = run.samples.iloc[:0].reindex(range(3))
    samples = samples.assign(timestamp=[1.0, 2.0, 3.0], cpu_util_pct=total, own_cpu_pct=own)
    return parse_bundle(
        build_bundle(run.summary, run.epochs.drop(columns="energy_it_kwh"), samples)
    )


def test_audit_flags_outside_load_over_the_limit():
    from measure import audit

    busy = _with_load(90.0, 70.0)
    quiet = _with_load(90.0, 88.0)
    assert audit.outside_load_pct(busy) == pytest.approx(20.0)
    assert any("outside CPU load" in problem for problem in audit.check_run(busy))
    assert not any("outside CPU load" in problem for problem in audit.check_run(quiet))


def test_sampler_records_own_cpu_share():
    import time

    with sensors.UsageSampler(period_s=0.05) as sampler:
        time.sleep(0.3)
    assert "own_cpu_pct" in sampler.frame().columns
