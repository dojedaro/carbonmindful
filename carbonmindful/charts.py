"""Altair charts. One axis per chart; colour follows the entity, never the rank.

Each chart picks the unit that suits its largest value, because training runs span
several orders of magnitude and raw kg values would read as zero.
"""

from __future__ import annotations

from typing import cast

import altair as alt
import pandas as pd

from . import fmt

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#52514e"
GRID = "#e6e5e0"
NUMBER = ",.3~r"
MAX_LINE_POINTS = 1500

Chart = alt.Chart | alt.LayerChart


def _style(chart: Chart, height: int = 320) -> Chart:
    styled = (
        chart.properties(height=height)
        .configure_axis(
            gridColor=GRID,
            domainColor=GRID,
            tickColor=GRID,
            labelColor=INK,
            titleColor=INK,
            labelFontSize=12,
            titleFontSize=12,
            titleFontWeight="normal",
            labelLimit=320,
            tickCount=6,
        )
        .configure_legend(labelColor=INK, titleColor=INK, orient="top", labelFontSize=12)
        .configure_view(strokeWidth=0)
    )
    return cast(Chart, styled)


def regional_bars(table: pd.DataFrame) -> Chart:
    """Ranked bars of scenario emissions for one PUE, labelled at the bar end."""
    scale, unit = fmt.co2e_scale(float(table["emissions_kg"].max()))
    frame = table.assign(value=table["emissions_kg"] / scale)
    order = frame.sort_values("value")["country"].tolist()
    base = alt.Chart(frame).encode(
        y=alt.Y("country:N", sort=order, title=None),
        x=alt.X("value:Q", title=f"Estimated emissions ({unit})"),
        tooltip=[
            alt.Tooltip("country:N", title="Scenario"),
            alt.Tooltip("value:Q", title=unit, format=NUMBER),
            alt.Tooltip("factor_kg_per_kwh:Q", title="kg CO₂e/kWh", format=".3f"),
        ],
    )
    bars = base.mark_bar(color=BLUE, cornerRadiusEnd=4, size=26)
    labels = base.mark_text(align="left", dx=6, color=INK).encode(
        text=alt.Text("value:Q", format=NUMBER)
    )
    return _style(bars + labels, 90 + 44 * len(frame))


def pue_bars(table: pd.DataFrame) -> Chart:
    """Grouped bars for up to two PUE values across the region scenarios."""
    scale, unit = fmt.co2e_scale(float(table["emissions_kg"].max()))
    frame = table.assign(value=table["emissions_kg"] / scale)
    order = frame.groupby("country")["value"].max().sort_values().index.tolist()
    labels = sorted(frame["pue_label"].unique())
    chart = (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=4, size=16)
        .encode(
            y=alt.Y("country:N", sort=order, title=None),
            yOffset=alt.YOffset("pue_label:N", sort=labels),
            x=alt.X("value:Q", title=f"Estimated emissions ({unit})"),
            color=alt.Color(
                "pue_label:N",
                title=None,
                sort=labels,
                scale=alt.Scale(domain=labels, range=[BLUE, ORANGE]),
            ),
            tooltip=[
                alt.Tooltip("country:N", title="Scenario"),
                alt.Tooltip("pue_label:N", title="PUE"),
                alt.Tooltip("value:Q", title=unit, format=NUMBER),
            ],
        )
    )
    return _style(chart, 110 + 56 * frame["country"].nunique())


def tradeoff_curve(tradeoff: pd.DataFrame) -> Chart:
    """Validation accuracy against cumulative IT energy, on one pair of axes."""
    scale, unit = fmt.energy_scale(float(tradeoff["cumulative_energy_it_kwh"].max()))
    frame = tradeoff.assign(cumulative=tradeoff["cumulative_energy_it_kwh"] / scale)
    low = max(0.0, float(frame["val_acc_pct"].min()) - 1)
    high = min(100.0, float(frame["val_acc_pct"].max()) + 1)
    base = alt.Chart(frame).encode(
        x=alt.X("cumulative:Q", title=f"Cumulative IT energy ({unit})"),
        y=alt.Y(
            "val_acc_pct:Q", title="Validation accuracy (%)", scale=alt.Scale(domain=[low, high])
        ),
        tooltip=[
            alt.Tooltip("epoch:O", title="Epoch"),
            alt.Tooltip("val_acc_pct:Q", title="Accuracy (%)", format=".2f"),
            alt.Tooltip("cumulative:Q", title=f"Cumulative {unit}", format=NUMBER),
        ],
    )
    line = base.mark_line(color=BLUE, strokeWidth=2)
    points = base.mark_point(color=BLUE, filled=True, size=70, opacity=1)
    return _style(line + points)


def marginal_bars(tradeoff: pd.DataFrame) -> Chart:
    """Marginal emissions per accuracy point gained, for epochs with a gain."""
    frame = tradeoff.dropna(subset=["marginal_kg_per_point"])
    scale, unit = fmt.co2e_scale(float(frame["marginal_kg_per_point"].max()))
    frame = frame.assign(value=frame["marginal_kg_per_point"] / scale)
    chart = (
        alt.Chart(frame)
        .mark_bar(color=BLUE, cornerRadiusEnd=4, size=22)
        .encode(
            x=alt.X("epoch:O", title="Epoch", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("value:Q", title=f"{unit} per accuracy point gained"),
            tooltip=[
                alt.Tooltip("epoch:O", title="Epoch"),
                alt.Tooltip("value:Q", title=f"{unit} per point", format=NUMBER),
                alt.Tooltip("acc_gain_points:Q", title="Points gained", format=".2f"),
            ],
        )
    )
    return _style(chart)


def utilisation_lines(samples: pd.DataFrame) -> Chart:
    """CPU and GPU utilisation against elapsed time on one percentage scale."""
    ordered = samples.sort_values("timestamp")
    step = max(1, len(ordered) // MAX_LINE_POINTS)
    frame = ordered.iloc[::step].assign(
        minutes=lambda t: (t["timestamp"] - ordered["timestamp"].iloc[0]) / 60
    )
    series = {"cpu_util_pct": "CPU", "gpu_util_pct": "GPU"}
    columns = [c for c in series if c in frame and frame[c].notna().any()]
    long = frame.melt(
        id_vars="minutes", value_vars=columns, var_name="metric", value_name="utilisation"
    ).dropna()
    long["device"] = long["metric"].map(series)
    names = [series[c] for c in columns]
    colors = [BLUE if name == "CPU" else ORANGE for name in names]
    chart = (
        alt.Chart(long)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("minutes:Q", title="Elapsed time (min)"),
            y=alt.Y("utilisation:Q", title="Utilisation (%)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color(
                "device:N",
                title=None,
                scale=alt.Scale(domain=names, range=colors),
                legend=alt.Legend() if len(names) > 1 else None,
            ),
            tooltip=[
                alt.Tooltip("device:N", title="Device"),
                alt.Tooltip("minutes:Q", title="Minutes", format=".1f"),
                alt.Tooltip("utilisation:Q", title="Utilisation (%)", format=".0f"),
            ],
        )
    )
    return _style(chart)


def comparison_points(aggregate: pd.DataFrame) -> Chart:
    """Mean IT energy per configuration, with one sample standard deviation."""
    scale, unit = fmt.energy_scale(float(aggregate["energy_it_kwh_mean"].max()))
    deviation = aggregate["energy_it_kwh_std"].fillna(0)
    frame = aggregate.assign(
        label=aggregate["model"]
        + " / "
        + aggregate["dataset"]
        + " / "
        + aggregate["device"]
        + " / "
        + aggregate["platform"]
        + " / "
        + aggregate["epochs"].astype(str)
        + " epochs",
        mean=aggregate["energy_it_kwh_mean"] / scale,
        low=(aggregate["energy_it_kwh_mean"] - deviation) / scale,
        high=(aggregate["energy_it_kwh_mean"] + deviation) / scale,
    )
    order = frame.sort_values("mean")["label"].tolist()
    y = alt.Y("label:N", title=None, sort=order)
    rules = (
        alt.Chart(frame)
        .mark_rule(color=INK, strokeWidth=2)
        .encode(y=y, x=alt.X("low:Q", title=f"Mean IT energy ({unit})"), x2="high:Q")
    )
    points = (
        alt.Chart(frame)
        .mark_point(color=BLUE, filled=True, size=90, opacity=1)
        .encode(
            y=y,
            x="mean:Q",
            tooltip=[
                alt.Tooltip("label:N", title="Configuration"),
                alt.Tooltip("mean:Q", title=f"Mean {unit}", format=NUMBER),
                alt.Tooltip("n:Q", title="Runs"),
            ],
        )
    )
    return _style(rules + points, 90 + 44 * len(frame))
