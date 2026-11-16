"""Start the real app, open each tab in Chromium, save screenshots, report problems.

Usage: python tests/visual_check.py OUTPUT_DIR
Exits with status 1 if the page shows an exception or logs a console error.
"""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import ConsoleMessage, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
PORT = 8599
TABS = [
    "Overview",
    "Regional scenarios",
    "PUE sensitivity",
    "Accuracy trade-off",
    "Hardware",
    "Compare runs",
    "Export",
]


def wait_for_server() -> None:
    for _ in range(60):
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/_stcore/health", timeout=1)
            return
        except OSError:
            time.sleep(1)
    raise RuntimeError("The server did not start.")


def main(output: Path) -> int:
    output.mkdir(parents=True, exist_ok=True)
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
    problems: list[str] = []
    try:
        wait_for_server()
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1400, "height": 1300})

            def record(message: ConsoleMessage) -> None:
                if message.type == "error":
                    problems.append(f"console: {message.text[:200]}")

            page.on("console", record)
            page.goto(f"http://localhost:{PORT}")
            page.get_by_role("tab", name="Overview").wait_for(timeout=60000)
            for index, name in enumerate(TABS):
                page.get_by_role("tab", name=name, exact=True).click()
                page.locator('[data-testid="stSkeleton"]').first.wait_for(
                    state="detached", timeout=30000
                )
                page.wait_for_timeout(2000)
                if page.locator('[data-testid="stException"]').count():
                    problems.append(f"exception on tab {name}")
                target = output / f"{index:02d}_{name.replace(' ', '_')}.png"
                page.screenshot(path=target, full_page=True)
            browser.close()
    finally:
        server.terminate()
    print("\n".join(problems) or "No page exceptions or console errors.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
