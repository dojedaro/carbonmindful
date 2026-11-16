"""Break the code on purpose and confirm the test suite notices each break."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MUTANTS = [
    (
        "carbonmindful/calc.py",
        "return energy * ratio",
        "return energy * ratio * ratio",
        "PUE applied twice",
    ),
    (
        "carbonmindful/calc.py",
        "scale = pue * factor_kg_per_kwh",
        "scale = factor_kg_per_kwh",
        "PUE dropped from per-epoch values",
    ),
    (
        "carbonmindful/calc.py",
        ".where(gain > 0)",
        ".where(gain >= 0)",
        "marginal cost at zero gain",
    ),
    (
        "carbonmindful/calc.py",
        "(accuracy_fraction * 100.0)",
        "accuracy_fraction",
        "metric uses a fraction",
    ),
    (
        "carbonmindful/calc.py",
        '"epochs", "accuracy_type"]',
        '"accuracy_type"]',
        "different epoch counts averaged",
    ),
    ("carbonmindful/calc.py", '"device", "platform",', '"device",', "different platforms averaged"),
    (
        "carbonmindful/bundle.py",
        'energy_it_kwh=float(summary["energy_kwh"]) / tracker_pue',
        'energy_it_kwh=float(summary["energy_kwh"])',
        "tracker PUE kept in total",
    ),
    (
        "carbonmindful/bundle.py",
        "energy_it_kwh=energy / tracker_pue",
        "energy_it_kwh=energy",
        "tracker PUE kept per epoch",
    ),
    (
        "carbonmindful/bundle.py",
        "if test is not None:\n            return float(test)",
        "if False:\n            return float(test)",
        "test accuracy ignored",
    ),
    ("carbonmindful/report.py", 'return "\'" + value', "return value", "formula injection allowed"),
    ("carbonmindful/fmt.py", '(1e-3, "g CO₂e")', '(1e-3, "mg CO₂e")', "wrong unit label"),
    ("app.py", "if uploaded or not use_demo:", "if not use_demo:", "demo data mixed with uploads"),
]


def main() -> int:
    survivors = 0
    for path, old, new, label in MUTANTS:
        with tempfile.TemporaryDirectory() as folder:
            copy = Path(folder) / "code"
            shutil.copytree(
                ROOT,
                copy,
                ignore=shutil.ignore_patterns(
                    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"
                ),
            )
            target = copy / path
            text = target.read_text(encoding="utf-8")
            if old not in text:
                print(f"NOT APPLIED  {label}")
                survivors += 1
                continue
            target.write_text(text.replace(old, new, 1), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "-x", "--no-header", "tests"],
                cwd=copy,
                capture_output=True,
                check=False,
            )
            caught = result.returncode != 0
            print(f"{'caught  ' if caught else 'SURVIVED'}  {label}")
            survivors += 0 if caught else 1
    print(f"{len(MUTANTS) - survivors} of {len(MUTANTS)} caught.")
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
