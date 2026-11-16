"""Upload the real run bundles through the browser and screenshot the comparison tab.

Usage: python tests/real_runs_check.py OUTPUT_DIR RUNS_DIR [RUNS_DIR ...]
"""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
PORT = 8597


def main(output: Path, folders: list[Path]) -> int:
    output.mkdir(parents=True, exist_ok=True)
    files = sorted(str(path) for folder in folders for path in folder.glob("*.zip"))
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
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1500, "height": 1500})
            page.goto(f"http://localhost:{PORT}")
            page.get_by_role("tab", name="Overview").wait_for(timeout=60000)
            page.locator('input[type="file"]').set_input_files(files)
            expect(page.get_by_text("Synthetic demo data is loaded")).to_have_count(
                0, timeout=60000
            )
            expect(page.get_by_text("Recorded energy by component").first).to_be_visible(
                timeout=60000
            )
            for name in ("Overview", "Hardware", "Accuracy trade-off", "Compare runs"):
                page.get_by_role("tab", name=name, exact=True).click()
                page.wait_for_timeout(3000)
                if page.locator('[data-testid="stException"]').count():
                    print(f"Exception on tab {name}")
                    return 1
                page.screenshot(path=output / f"real_{name.replace(' ', '_')}.png", full_page=True)
            browser.close()
    finally:
        server.terminate()
    print(f"Loaded {len(files)} real bundles with no page exceptions.")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), [Path(item) for item in sys.argv[2:]]))
