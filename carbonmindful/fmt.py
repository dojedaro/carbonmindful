"""Readable units for very small quantities. Small values never display as zero."""

from __future__ import annotations

import math

_CO2E = ((1.0, "kg CO₂e"), (1e-3, "g CO₂e"), (1e-6, "mg CO₂e"))
_ENERGY = ((1.0, "kWh"), (1e-3, "Wh"), (1e-6, "mWh"))


def _pick(value: float, units: tuple[tuple[float, str], ...]) -> tuple[float, str]:
    magnitude = abs(value)
    for scale, unit in units:
        if magnitude >= scale:
            return scale, unit
    return units[-1]


def co2e_scale(kg: float) -> tuple[float, str]:
    """Return the divisor and unit that suit a mass given in kg CO2e."""
    return _pick(kg, _CO2E)


def energy_scale(kwh: float) -> tuple[float, str]:
    """Return the divisor and unit that suit an energy given in kWh."""
    return _pick(kwh, _ENERGY)


def co2e(kg: float) -> str:
    """Format kg CO2e using kg, g or mg, with three significant figures."""
    if not math.isfinite(kg):
        return "n/a"
    scale, unit = co2e_scale(kg)
    return f"{kg / scale:.3g} {unit}"


def energy(kwh: float) -> str:
    """Format energy using kWh, Wh or mWh, with three significant figures."""
    if not math.isfinite(kwh):
        return "n/a"
    scale, unit = energy_scale(kwh)
    return f"{kwh / scale:.3g} {unit}"


def duration(seconds: float) -> str:
    """Format a duration using seconds, minutes or hours."""
    if not math.isfinite(seconds):
        return "n/a"
    if seconds < 60:
        return f"{seconds:.1f} s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} h"
