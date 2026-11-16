from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture
def app():
    test = AppTest.from_file(APP, default_timeout=60)
    test.run()
    return test


def test_demo_renders_without_exceptions(app):
    assert not app.exception
    assert len(app.tabs) == 7


def test_synthetic_warning_is_visible(app):
    assert any("not experimental results" in w.value for w in app.warning)


def test_scenario_wording_is_explicit(app):
    assert any("not run in these countries" in i.value for i in app.info)


def test_no_country_selected_does_not_crash(app):
    app.multiselect[0].set_value([]).run()
    assert not app.exception
    assert any("at least one country" in w.value for w in app.warning)


def test_equal_pues_do_not_crash(app):
    app.number_input[1].set_value(1.2).run()
    assert not app.exception


def test_demo_off_shows_prompt(app):
    app.toggle[0].set_value(False).run()
    assert not app.exception
    assert any("Upload a run bundle" in i.value for i in app.info)


def test_uploaded_runs_never_mix_with_synthetic_runs():
    from app import _choose_sources

    uploaded = [("real.zip", b"real")]
    assert _choose_sources(uploaded, use_demo=True) == uploaded
    assert _choose_sources([], use_demo=False) == []
    assert len(_choose_sources([], use_demo=True)) == 9


def test_overview_value_matches_hand_calculation():
    from carbonmindful import calc, fmt
    from carbonmindful.bundle import parse_bundle
    from carbonmindful.demo import demo_bundles

    run = parse_bundle(demo_bundles()[0])
    expected = run.energy_it_kwh * 1.54 * 0.191587
    assert calc.emissions_kg(run.energy_it_kwh, 1.54, 0.191587) == pytest.approx(expected)
    test = AppTest.from_file(APP, default_timeout=60)
    test.run()
    assert any(metric.value == fmt.co2e(expected) for metric in test.metric)


def test_load_runs_skips_bad_files_and_duplicate_ids(monkeypatch):
    import app as module
    from carbonmindful.demo import demo_bundles

    messages = []
    monkeypatch.setattr(module.st, "error", messages.append)
    monkeypatch.setattr(module.st, "warning", messages.append)
    good = demo_bundles()[0]
    runs = module._load_runs([("a.zip", good), ("b.zip", good), ("c.zip", b"bad")])
    assert len(runs) == 1
    assert any("already loaded" in m for m in messages)
    assert any("not a valid zip" in m for m in messages)


def test_uploaded_file_names_cannot_inject_markup():
    from app import _safe_name

    assert _safe_name("run_1.zip") == "run_1.zip"
    assert "[" not in _safe_name("![x](http://evil/a.png).zip")
