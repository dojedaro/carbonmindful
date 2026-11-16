"""Upload a known bundle through the real browser UI and check the displayed values.

The bundle records tracker PUE 1.2 and 0.012 kWh, so IT energy must display as 10 Wh and
the Mongolia scenario at the default PUE 1.54 must be 0.01 x 1.54 x 0.82 kg = 12.6 g CO2e.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import pandas as pd
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from carbonmindful.bundle import build_bundle

PORT = 8598


def bundle_bytes() -> bytes:
    summary = {
        "schema_version": 1,
        "run_id": "e2e_run",
        "model": "m",
        "dataset": "d",
        "device": "cpu",
        "platform": "test",
        "seed": 1,
        "duration_s": 10.0,
        "energy_kwh": 0.012,
        "tracker_pue": 1.2,
        "test_accuracy": 0.9,
    }
    epochs = pd.DataFrame(
        {
            "epoch": [1, 2],
            "duration_s": [5.0, 5.0],
            "val_acc": [0.8, 0.9],
            "energy_kwh": [0.006] * 2,
        }
    )
    samples = pd.DataFrame({"timestamp": [1.0, 2.0], "cpu_util_pct": [50.0, 60.0]})
    return build_bundle(summary, epochs, samples)


def main() -> int:
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app.py",
            "--server.headless=true",
            f"--server.port={PORT}",
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://localhost:{PORT}/_stcore/health", timeout=1)
                break
            except OSError:
                time.sleep(1)
        with tempfile.TemporaryDirectory() as folder, sync_playwright() as playwright:
            path = Path(folder) / "e2e.zip"
            path.write_bytes(bundle_bytes())
            bad = Path(folder) / "bad.zip"
            bad.write_bytes(b"not a zip")
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1400, "height": 1200})
            page.goto(f"http://localhost:{PORT}")
            page.get_by_role("tab", name="Overview").wait_for(timeout=60000)
            page.locator('input[type="file"]').set_input_files([str(path), str(bad)])
            expect(page.get_by_text("bad.zip: The file is not a valid zip archive.")).to_be_visible(
                timeout=30000
            )
            expect(page.get_by_text("10 Wh", exact=True)).to_be_visible(timeout=30000)
            expect(page.get_by_text("Synthetic demo data is loaded")).to_have_count(0)
            expect(page.get_by_text("not counted twice").first).to_be_visible()
            page.get_by_test_id("stSidebar").get_by_text("Canada", exact=True).last.wait_for()
            box = page.get_by_test_id("stSelectbox").first
            box.click()
            page.get_by_role("option", name="Mongolia").click()
            expect(page.get_by_text("12.6 g CO₂e", exact=True)).to_be_visible(timeout=30000)
            page.get_by_role("tab", name="Compare runs", exact=True).click()
            expect(page.get_by_text("synthetic", exact=True)).to_have_count(0)
            expect(page.get_by_text("Mean IT energy (kWh)")).to_have_count(0)
            browser.close()
    finally:
        server.terminate()
    print("End-to-end upload check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
