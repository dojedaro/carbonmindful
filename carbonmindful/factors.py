"""Grid emission factors, each with its value, unit, source and reference year."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path(__file__).parent / "data" / "grid_factors.csv"
_COLUMNS = {
    "iso",
    "country",
    "year",
    "factor_kg_co2e_per_kwh",
    "metric",
    "source",
    "original_source",
    "retrieved",
    "status",
}


@dataclass(frozen=True)
class GridFactor:
    iso: str
    country: str
    year: int
    kg_per_kwh: float
    metric: str
    source: str
    original_source: str
    retrieved: str
    status: str
    note: str = ""

    @property
    def provisional(self) -> bool:
        return self.status.upper().startswith("PROVISIONAL")


def load_factors(path: Path = DEFAULT_PATH) -> dict[str, GridFactor]:
    """Load the registry, keyed by ISO code. Rejects incomplete or invalid rows."""
    table = pd.read_csv(path)
    missing = sorted(_COLUMNS - set(table.columns))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {', '.join(missing)}")
    if table["iso"].duplicated().any():
        raise ValueError(f"{path.name} has duplicate ISO codes")
    if table[sorted(_COLUMNS)].isna().any().any():
        raise ValueError(f"{path.name} has empty values")
    if not (table["factor_kg_co2e_per_kwh"] > 0).all():
        raise ValueError(f"{path.name} has non-positive factors")
    return {
        str(row["iso"]): GridFactor(
            iso=str(row["iso"]),
            country=str(row["country"]),
            year=int(row["year"]),
            kg_per_kwh=float(row["factor_kg_co2e_per_kwh"]),
            metric=str(row["metric"]),
            source=str(row["source"]),
            original_source=str(row["original_source"]),
            retrieved=str(row["retrieved"]),
            status=str(row["status"]),
            note=str(row["note"]) if "note" in row and pd.notna(row["note"]) else "",
        )
        for _, row in table.iterrows()
    }
