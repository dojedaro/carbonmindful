import io
import json
import zipfile

import pytest

from carbonmindful.bundle import BundleError, parse_bundle
from carbonmindful.demo import demo_bundles

SUMMARY = {
    "schema_version": 1,
    "run_id": "r1",
    "model": "m",
    "dataset": "d",
    "device": "cpu",
    "seed": 1,
    "duration_s": 10.0,
    "energy_kwh": 0.012,
    "tracker_pue": 1.2,
    "test_accuracy": 0.9,
}
EPOCHS = "epoch,duration_s,val_acc,energy_kwh\n1,5,0.8,0.006\n2,5,0.9,0.006\n"
SAMPLES = "timestamp,cpu_util_pct\n1,50\n2,60\n"


def make(summary=SUMMARY, epochs=EPOCHS, samples=SAMPLES, prefix=""):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        if summary is not None:
            text = summary if isinstance(summary, str) else json.dumps(summary)
            archive.writestr(prefix + "summary.json", text)
        if epochs is not None:
            archive.writestr(prefix + "epochs.csv", epochs)
        if samples is not None:
            archive.writestr(prefix + "samples.csv", samples)
    return buffer.getvalue()


def test_tracker_pue_is_removed_from_total_and_epochs():
    bundle = parse_bundle(make())
    assert bundle.energy_it_kwh == pytest.approx(0.01)
    assert bundle.epochs.energy_it_kwh.sum() == pytest.approx(0.01)
    assert any("not counted twice" in w for w in bundle.warnings)


def test_epoch_energy_gap_is_flagged():
    gap = "epoch,duration_s,val_acc,energy_kwh\n1,5,0.8,0.002\n2,5,0.9,0.002\n"
    assert any("sums to" in w for w in parse_bundle(make(epochs=gap)).warnings)
    assert not any("sums to" in w for w in parse_bundle(make()).warnings)


def test_nested_folder_is_accepted():
    assert parse_bundle(make(prefix="run/x/")).run_id == "r1"


def test_samples_are_optional():
    assert parse_bundle(make(samples=None)).samples.empty


def test_test_accuracy_preferred_over_validation():
    assert parse_bundle(make()).accuracy_label == "Test accuracy"
    without = {k: v for k, v in SUMMARY.items() if k != "test_accuracy"}
    bundle = parse_bundle(make(summary=without))
    assert bundle.accuracy == 0.9 and bundle.accuracy_label == "Validation accuracy"


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"summary": None}, "no summary.json"),
        ({"epochs": None}, "no epochs.csv"),
        ({"summary": "{bad"}, "not valid JSON"),
        ({"summary": "[1]"}, "JSON object"),
        ({"summary": {**SUMMARY, "tracker_pue": 0.9}}, "tracker_pue"),
        ({"summary": {**SUMMARY, "energy_kwh": -1}}, "energy_kwh"),
        ({"summary": {**SUMMARY, "schema_version": 9}}, "schema_version"),
        ({"summary": {**SUMMARY, "seed": True}}, "seed"),
        ({"summary": {**SUMMARY, "test_accuracy": 95.1}}, "test_accuracy"),
        ({"epochs": "epoch,duration_s\n1,2\n"}, "missing columns"),
        ({"epochs": "epoch,duration_s,val_acc\n"}, "no rows"),
        ({"epochs": "epoch,duration_s,val_acc\n1,2,95\n"}, "fraction"),
        ({"epochs": "epoch,duration_s,val_acc\n1,2,abc\n"}, "non-numeric"),
    ],
)
def test_bad_bundles_give_clear_errors(kwargs, message):
    with pytest.raises(BundleError, match=message):
        parse_bundle(make(**kwargs))


def test_not_a_zip():
    with pytest.raises(BundleError, match="not a valid zip"):
        parse_bundle(b"nope")


def test_demo_bundles_are_marked_synthetic():
    bundles = [parse_bundle(b) for b in demo_bundles()]
    assert len(bundles) == 9 and all(b.synthetic for b in bundles)


def corrupt_zip():
    data = bytearray(make())
    index = data.find(b"schema_version")
    if index >= 0:
        data[index] ^= 0xFF
    return bytes(data)


def test_corrupt_member_gives_clear_error():
    with pytest.raises(BundleError):
        parse_bundle(corrupt_zip())


def test_encrypted_member_gives_clear_error():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("summary.json", json.dumps(SUMMARY))
        archive.writestr("epochs.csv", EPOCHS)
    data = bytearray(buffer.getvalue())
    start = data.find(b"PK\x03\x04")
    data[start + 6] |= 0x01
    central = data.find(b"PK\x01\x02")
    data[central + 8] |= 0x01
    with pytest.raises(BundleError):
        parse_bundle(bytes(data))


@pytest.mark.parametrize(
    "summary,message",
    [
        ({**SUMMARY, "duration_s": -1}, "duration_s"),
        ({**SUMMARY, "tracker_emissions_kg": "x"}, "tracker_emissions_kg"),
        ({**SUMMARY, "tracker_emissions_kg": -1}, "tracker_emissions_kg"),
        ({**SUMMARY, "tracker_country_iso": ["KOR"]}, "tracker_country_iso"),
        ({**SUMMARY, "cpu_energy_kwh": "x"}, "cpu_energy_kwh"),
        ({**SUMMARY, "package_counter_kwh": -1}, "package_counter_kwh"),
        ({**SUMMARY, "run_id": "![x](http://evil/a.png)"}, "run_id"),
        ({**SUMMARY, "run_id": "../../secret"}, "run_id"),
        ({**SUMMARY, "model": "<script>"}, "model"),
        ({**SUMMARY, "energy_source": "a|b"}, "energy_source"),
        ({**SUMMARY, "platform": "x" * 500}, "platform"),
    ],
)
def test_unsafe_or_invalid_summary_fields_rejected(summary, message):
    with pytest.raises(BundleError, match=message):
        parse_bundle(make(summary=summary))


def test_non_finite_json_rejected():
    text = json.dumps(SUMMARY).replace('"duration_s": 10.0', '"duration_s": NaN')
    with pytest.raises(BundleError, match="duration_s"):
        parse_bundle(make(summary=text))


@pytest.mark.parametrize(
    "epochs,message",
    [
        ("epoch,duration_s,val_acc\n1,2,0.5\n1,2,0.6\n", "duplicate"),
        ("epoch,duration_s,val_acc\n1,inf,0.5\n", "finite"),
        ("epoch,duration_s,val_acc\n1,-2,0.5\n", "negative"),
        ("epoch,duration_s,val_acc,energy_kwh\n1,2,0.5,inf\n", "energy_kwh"),
    ],
)
def test_bad_epoch_values_rejected(epochs, message):
    with pytest.raises(BundleError, match=message):
        parse_bundle(make(epochs=epochs))


def test_optional_columns_with_junk_become_missing():
    samples = "timestamp,cpu_util_pct,gpu_util_pct\n1,50,oops\n2,60,70\n"
    epochs = "epoch,duration_s,val_acc,samples_per_s\n1,5,0.8,abc\n2,5,0.9,100\n"
    bundle = parse_bundle(make(samples=samples, epochs=epochs))
    assert bundle.samples.gpu_util_pct.isna().sum() == 1
    assert bundle.epochs.samples_per_s.isna().sum() == 1


def test_build_bundle_round_trip():
    import pandas as pd

    from carbonmindful.bundle import build_bundle

    epochs = pd.DataFrame(
        {"epoch": [1], "duration_s": [1.0], "val_acc": [0.5], "energy_kwh": [0.01]}
    )
    samples = pd.DataFrame({"timestamp": [1.0], "cpu_util_pct": [50.0]})
    summary = {**SUMMARY, "energy_kwh": 0.01, "tracker_pue": 1.0}
    bundle = parse_bundle(build_bundle(summary, epochs, samples))
    assert bundle.energy_it_kwh == 0.01 and len(bundle.epochs) == 1


def test_test_accuracy_value_is_used_when_it_differs_from_validation():
    bundle = parse_bundle(make(summary={**SUMMARY, "test_accuracy": 0.77}))
    assert bundle.accuracy == 0.77


def test_enormous_integer_is_rejected_not_a_crash():
    text = json.dumps(SUMMARY).replace('"energy_kwh": 0.012', '"energy_kwh": 1' + "0" * 400)
    with pytest.raises(BundleError, match="energy_kwh"):
        parse_bundle(make(summary=text))


def test_deeply_nested_json_is_rejected_not_a_crash():
    with pytest.raises(BundleError):
        parse_bundle(make(summary="[" * 200_000))


@pytest.mark.parametrize("seed", [2**80, -1])
def test_seed_outside_range_is_rejected(seed):
    with pytest.raises(BundleError, match="seed"):
        parse_bundle(make(summary={**SUMMARY, "seed": seed}))


def test_fractional_epoch_is_rejected():
    with pytest.raises(BundleError, match="whole numbers"):
        parse_bundle(make(epochs="epoch,duration_s,val_acc\n1.5,2,0.5\n"))
