"""Deterministic Plotly evidence charts built from validated AI4I analytical inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from industrial_copilot.analytics.common import (
    apply_filters,
    prepare_analysis_frame,
    resolve_filters,
)
from industrial_copilot.analytics.comparisons import compare_product_types
from industrial_copilot.analytics.descriptive import failure_rate_by_range
from industrial_copilot.analytics.failure_analysis import failure_mode_breakdown
from industrial_copilot.analytics.models import (
    AnalysisFilters,
    NumericRange,
    SimilarConditionsResult,
)
from industrial_copilot.data.schema import (
    MACHINE_FAILURE,
    ROTATIONAL_SPEED,
    TOOL_WEAR,
    TORQUE,
)
from industrial_copilot.features.engineering import (
    HDF_RULE_CONDITION,
    MECHANICAL_POWER,
    TEMPERATURE_DELTA,
)

RPM_BAND_RANGES = (
    NumericRange(column=ROTATIONAL_SPEED, maximum=1379),
    NumericRange(column=ROTATIONAL_SPEED, minimum=1380, maximum=1600),
    NumericRange(column=ROTATIONAL_SPEED, minimum=1601),
)

FAILURE_COLORS = {0: "#4C78A8", 1: "#E45756"}
PRODUCT_COLORS = {"L": "#4C78A8", "M": "#F2CF5B", "H": "#54A24B"}


def failure_rate_by_rpm_bands_chart(
    frame: pd.DataFrame, filters: AnalysisFilters | None = None
) -> go.Figure:
    """Create a bar chart of empirical failure rate in engineering-relevant RPM bands."""

    selected = _selected_engineered_frame(frame, filters)
    analysis = failure_rate_by_range(selected, ROTATIONAL_SPEED, list(RPM_BAND_RANGES))
    labels = [item.label for item in analysis.ranges]
    rates = [item.failure_rate * 100 if item.failure_rate is not None else None for item in analysis.ranges]
    hover_text = [
        f"{item.failed_observation_count} failures / {item.observation_count} observations"
        for item in analysis.ranges
    ]
    figure = go.Figure(
        go.Bar(
            x=labels,
            y=rates,
            marker_color="#4C78A8",
            text=[f"{rate:.1f}%" if rate is not None else "No data" for rate in rates],
            textposition="outside",
            customdata=hover_text,
            hovertemplate="RPM band: %{x}<br>Failure rate: %{y:.2f}%<br>%{customdata}<extra></extra>",
        )
    )
    return _apply_layout(
        figure,
        title="Failure rate by RPM band",
        xaxis_title="Rotational speed band (rpm)",
        yaxis_title="Machine failure rate (%)",
        yaxis_tickformat=".0f",
    )


def failed_vs_healthy_distribution_chart(
    frame: pd.DataFrame,
    variable: str,
    filters: AnalysisFilters | None = None,
) -> go.Figure:
    """Compare a numeric operating variable with overlaid box/violin distributions."""

    selected = _selected_engineered_frame(frame, filters)
    _require_numeric_column(selected, variable)
    figure = go.Figure()
    for failure_value, label in ((0, "Healthy"), (1, "Failed")):
        values = selected.loc[selected[MACHINE_FAILURE].eq(failure_value), variable]
        figure.add_trace(
            go.Violin(
                y=values,
                name=label,
                box_visible=True,
                meanline_visible=True,
                points="outliers",
                line_color=FAILURE_COLORS[failure_value],
                fillcolor=FAILURE_COLORS[failure_value],
                opacity=0.58,
                hovertemplate=f"{label}<br>{variable}: %{{y:.2f}}<extra></extra>",
            )
        )
    return _apply_layout(
        figure,
        title=f"{variable}: failed versus healthy observations",
        xaxis_title="Machine status",
        yaxis_title=variable,
        violinmode="group",
    )


def failed_vs_healthy_torque_chart(
    frame: pd.DataFrame, filters: AnalysisFilters | None = None
) -> go.Figure:
    """Convenience chart for the required failed-versus-healthy torque distribution."""

    return failed_vs_healthy_distribution_chart(frame, TORQUE, filters)


def failed_vs_healthy_tool_wear_chart(
    frame: pd.DataFrame, filters: AnalysisFilters | None = None
) -> go.Figure:
    """Convenience chart for the required failed-versus-healthy tool-wear distribution."""

    return failed_vs_healthy_distribution_chart(frame, TOOL_WEAR, filters)


def failure_mode_frequency_chart(
    frame: pd.DataFrame, filters: AnalysisFilters | None = None
) -> go.Figure:
    """Create a mode-frequency bar chart while retaining the fact that modes may overlap."""

    selected = _selected_engineered_frame(frame, filters)
    breakdown = failure_mode_breakdown(selected)
    modes = [item.failure_mode for item in breakdown.modes]
    counts = [item.flagged_observation_count for item in breakdown.modes]
    rates = [item.rate_among_selected * 100 for item in breakdown.modes]
    figure = go.Figure(
        go.Bar(
            x=modes,
            y=counts,
            marker_color="#F58518",
            customdata=rates,
            text=counts,
            textposition="outside",
            hovertemplate=(
                "Failure mode: %{x}<br>Flagged observations: %{y}<br>"
                "Rate among selected: %{customdata:.2f}%<extra></extra>"
            ),
        )
    )
    figure = _apply_layout(
        figure,
        title="Failure-mode frequency",
        xaxis_title="Failure-mode flag",
        yaxis_title="Flagged observations",
    )
    figure.add_annotation(
        text="Mode flags may overlap; they are not mutually exclusive causes.",
        xref="paper",
        yref="paper",
        x=0,
        y=-0.23,
        showarrow=False,
        align="left",
        font={"size": 11},
    )
    return figure


def rpm_torque_failure_map_chart(
    frame: pd.DataFrame, filters: AnalysisFilters | None = None
) -> go.Figure:
    """Map operating points in RPM-torque space and distinguish observed failure status."""

    selected = _selected_engineered_frame(frame, filters)
    figure = go.Figure()
    for failure_value, label in ((0, "Healthy"), (1, "Machine failure")):
        subset = selected.loc[selected[MACHINE_FAILURE].eq(failure_value)]
        figure.add_trace(
            go.Scattergl(
                x=subset[ROTATIONAL_SPEED],
                y=subset[TORQUE],
                mode="markers",
                name=label,
                marker={"color": FAILURE_COLORS[failure_value], "size": 6, "opacity": 0.55},
                customdata=np.column_stack((subset.index, subset[MECHANICAL_POWER])),
                hovertemplate=(
                    "RPM: %{x:.0f}<br>Torque: %{y:.1f} Nm<br>"
                    "Mechanical power: %{customdata[1]:.0f} W<extra>" + label + "</extra>"
                ),
            )
        )
    return _apply_layout(
        figure,
        title="RPM × torque operating map",
        xaxis_title=ROTATIONAL_SPEED,
        yaxis_title=TORQUE,
        legend_title="Observed status",
    )


def hdf_operating_envelope_chart(
    frame: pd.DataFrame, filters: AnalysisFilters | None = None
) -> go.Figure:
    """Show the documented HDF RPM-temperature-delta envelope against real observations."""

    selected = _selected_engineered_frame(frame, filters)
    y_min = float(selected[TEMPERATURE_DELTA].min())
    y_max = float(selected[TEMPERATURE_DELTA].max())
    x_min = float(selected[ROTATIONAL_SPEED].min())
    x_max = float(selected[ROTATIONAL_SPEED].max())
    figure = go.Figure()
    for condition, label, color in (
        (False, "Outside documented HDF envelope", "#4C78A8"),
        (True, "Inside documented HDF envelope", "#E45756"),
    ):
        subset = selected.loc[selected[HDF_RULE_CONDITION].eq(condition)]
        figure.add_trace(
            go.Scattergl(
                x=subset[ROTATIONAL_SPEED],
                y=subset[TEMPERATURE_DELTA],
                mode="markers",
                name=label,
                marker={"color": color, "size": 6, "opacity": 0.6},
                customdata=np.column_stack((subset[MACHINE_FAILURE], subset["HDF"])),
                hovertemplate=(
                    "RPM: %{x:.0f}<br>Temperature delta: %{y:.2f} K<br>"
                    "Machine failure: %{customdata[0]}<br>HDF flag: %{customdata[1]}"
                    "<extra>" + label + "</extra>"
                ),
            )
        )
    envelope_x_max = min(1380, x_max)
    envelope_y_max = min(8.6, y_max)
    if x_min < envelope_x_max and y_min < envelope_y_max:
        figure.add_shape(
            type="rect",
            x0=x_min,
            x1=envelope_x_max,
            y0=y_min,
            y1=envelope_y_max,
            fillcolor="rgba(228,87,86,0.16)",
            line={"width": 0},
            layer="below",
        )
    figure.add_vline(x=1380, line_dash="dash", line_color="#E45756")
    figure.add_hline(y=8.6, line_dash="dash", line_color="#E45756")
    figure.add_annotation(
        x=(x_min + envelope_x_max) / 2,
        y=(y_min + envelope_y_max) / 2,
        text="Documented HDF envelope",
        showarrow=False,
        font={"size": 11, "color": "#E45756"},
    )
    return _apply_layout(
        figure,
        title="HDF operating envelope: temperature delta × RPM",
        xaxis_title=ROTATIONAL_SPEED,
        yaxis_title=TEMPERATURE_DELTA,
        legend_title="Rule condition",
    )


def torque_tool_wear_risk_heatmap_chart(
    frame: pd.DataFrame,
    filters: AnalysisFilters | None = None,
    bins: int = 8,
    minimum_cell_count: int = 10,
) -> go.Figure:
    """Plot observed failure rate in torque-tool-wear cells with OSF threshold boundaries."""

    if bins < 2:
        raise ValueError("bins must be at least 2.")
    if minimum_cell_count < 1:
        raise ValueError("minimum_cell_count must be at least 1.")
    selected = _selected_engineered_frame(frame, filters)
    if selected[TORQUE].nunique() < 2 or selected[TOOL_WEAR].nunique() < 2:
        raise ValueError("Risk heatmap requires at least two distinct torque and tool-wear values.")
    x_edges = np.linspace(selected[TORQUE].min(), selected[TORQUE].max(), bins + 1)
    y_edges = np.linspace(selected[TOOL_WEAR].min(), selected[TOOL_WEAR].max(), bins + 1)
    x_bin = pd.cut(selected[TORQUE], bins=x_edges, include_lowest=True, labels=False)
    y_bin = pd.cut(selected[TOOL_WEAR], bins=y_edges, include_lowest=True, labels=False)
    binned = selected.assign(_x_bin=x_bin, _y_bin=y_bin).dropna(subset=["_x_bin", "_y_bin"])
    grouped = binned.groupby(["_y_bin", "_x_bin"], observed=True)[MACHINE_FAILURE].agg(["count", "mean"])
    rate_matrix = np.full((bins, bins), np.nan)
    count_matrix = np.zeros((bins, bins), dtype=int)
    for (y_index, x_index), values in grouped.iterrows():
        y_position, x_position = int(y_index), int(x_index)
        count_matrix[y_position, x_position] = int(values["count"])
        if values["count"] >= minimum_cell_count:
            rate_matrix[y_position, x_position] = float(values["mean"] * 100)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    figure = go.Figure(
        go.Heatmap(
            x=x_centers,
            y=y_centers,
            z=rate_matrix,
            customdata=count_matrix,
            colorscale="YlOrRd",
            colorbar={"title": "Failure rate (%)"},
            hovertemplate=(
                "Torque: %{x:.1f} Nm<br>Tool wear: %{y:.0f} min<br>"
                "Failure rate: %{z:.2f}%<br>Observations: %{customdata}<extra></extra>"
            ),
            zmin=0,
        )
    )
    torque_line = np.linspace(max(float(selected[TORQUE].min()), 0.1), float(selected[TORQUE].max()), 250)
    for product_type, threshold, color in (
        ("L", 11000, PRODUCT_COLORS["L"]),
        ("M", 12000, PRODUCT_COLORS["M"]),
        ("H", 13000, PRODUCT_COLORS["H"]),
    ):
        wear_line = threshold / torque_line
        valid = (wear_line >= selected[TOOL_WEAR].min()) & (wear_line <= selected[TOOL_WEAR].max())
        figure.add_trace(
            go.Scatter(
                x=torque_line[valid],
                y=wear_line[valid],
                mode="lines",
                name=f"{product_type} OSF threshold",
                line={"color": color, "dash": "dash", "width": 2},
                hovertemplate=(
                    f"{product_type} OSF boundary<br>Torque: %{{x:.1f}} Nm<br>"
                    "Tool wear: %{y:.0f} min<extra></extra>"
                ),
            )
        )
    return _apply_layout(
        figure,
        title="Torque × tool-wear observed failure-risk surface",
        xaxis_title=TORQUE,
        yaxis_title=TOOL_WEAR,
        legend_title="Documented boundary",
    )


def mechanical_power_operating_map_chart(
    frame: pd.DataFrame, filters: AnalysisFilters | None = None
) -> go.Figure:
    """Show RPM-torque power regions and overlay observed PWF flags."""

    selected = _selected_engineered_frame(frame, filters)
    figure = go.Figure()
    regions = (
        (selected[MECHANICAL_POWER] < 3500, "Power < 3500 W", "#4C78A8"),
        (
            selected[MECHANICAL_POWER].between(3500, 9000, inclusive="both"),
            "3500–9000 W",
            "#54A24B",
        ),
        (selected[MECHANICAL_POWER] > 9000, "Power > 9000 W", "#F58518"),
    )
    for condition, label, color in regions:
        subset = selected.loc[condition]
        figure.add_trace(
            go.Scattergl(
                x=subset[ROTATIONAL_SPEED],
                y=subset[TORQUE],
                mode="markers",
                name=label,
                marker={"color": color, "size": 6, "opacity": 0.42},
                customdata=subset[MECHANICAL_POWER],
                hovertemplate=(
                    "RPM: %{x:.0f}<br>Torque: %{y:.1f} Nm<br>"
                    "Mechanical power: %{customdata:.0f} W<extra>" + label + "</extra>"
                ),
            )
        )
    pwf = selected.loc[selected["PWF"].eq(1)]
    figure.add_trace(
        go.Scattergl(
            x=pwf[ROTATIONAL_SPEED],
            y=pwf[TORQUE],
            mode="markers",
            name="Observed PWF",
            marker={"symbol": "x", "color": "#E45756", "size": 10, "line": {"width": 1}},
            customdata=pwf[MECHANICAL_POWER],
            hovertemplate=(
                "RPM: %{x:.0f}<br>Torque: %{y:.1f} Nm<br>"
                "Mechanical power: %{customdata:.0f} W<extra>Observed PWF</extra>"
            ),
        )
    )
    rpm_line = np.linspace(float(selected[ROTATIONAL_SPEED].min()), float(selected[ROTATIONAL_SPEED].max()), 250)
    for power, label in ((3500, "3500 W boundary"), (9000, "9000 W boundary")):
        torque_line = power * 60 / (rpm_line * 2 * np.pi)
        figure.add_trace(
            go.Scatter(
                x=rpm_line,
                y=torque_line,
                mode="lines",
                name=label,
                line={"color": "#2F4B7C", "dash": "dash", "width": 2},
                hovertemplate=f"{label}<br>RPM: %{{x:.0f}}<br>Torque: %{{y:.1f}} Nm<extra></extra>",
            )
        )
    return _apply_layout(
        figure,
        title="Mechanical-power operating map",
        xaxis_title=ROTATIONAL_SPEED,
        yaxis_title=TORQUE,
        legend_title="Power region / evidence",
    )


def product_type_failure_comparison_chart(
    frame: pd.DataFrame, filters: AnalysisFilters | None = None
) -> go.Figure:
    """Create a failure-rate comparison for L, M, and H product types."""

    selected = _selected_engineered_frame(frame, filters)
    comparison = compare_product_types(selected)
    labels = [item.product_type for item in comparison.groups]
    rates = [item.failure_rate * 100 for item in comparison.groups]
    counts = [item.observation_count for item in comparison.groups]
    failures = [item.failed_observation_count for item in comparison.groups]
    figure = go.Figure(
        go.Bar(
            x=labels,
            y=rates,
            marker_color=[PRODUCT_COLORS[label] for label in labels],
            text=[f"{rate:.2f}%" for rate in rates],
            textposition="outside",
            customdata=np.column_stack((counts, failures)),
            hovertemplate=(
                "Type: %{x}<br>Failure rate: %{y:.2f}%<br>"
                "Failures: %{customdata[1]} / %{customdata[0]}<extra></extra>"
            ),
        )
    )
    return _apply_layout(
        figure,
        title="Failure rate by product type",
        xaxis_title="Product type",
        yaxis_title="Machine failure rate (%)",
    )


def similar_condition_outcome_chart(result: SimilarConditionsResult) -> go.Figure:
    """Show the status of retrieved nearest operating conditions for one target observation."""

    if not result.observations:
        raise ValueError("Similar-condition chart requires at least one retrieved observation.")
    labels = [str(item.uid) for item in result.observations]
    outcomes = [item.machine_failure for item in result.observations]
    distances = [item.distance for item in result.observations]
    modes = [", ".join(item.active_failure_modes) or "None" for item in result.observations]
    figure = go.Figure(
        go.Scatter(
            x=labels,
            y=outcomes,
            mode="markers+text",
            marker={
                "color": [FAILURE_COLORS[value] for value in outcomes],
                "size": 18,
                "symbol": ["circle" if value == 0 else "x" for value in outcomes],
                "line": {"color": "white", "width": 1},
            },
            text=["Healthy" if value == 0 else "Failed" for value in outcomes],
            textposition="top center",
            customdata=np.column_stack((distances, modes)),
            hovertemplate=(
                "Similar UID: %{x}<br>Machine failure: %{y}<br>Distance: %{customdata[0]:.3f}<br>"
                "Failure modes: %{customdata[1]}<extra></extra>"
            ),
        )
    )
    return _apply_layout(
        figure,
        title=f"Similar-condition outcomes for UID {result.target_uid}",
        xaxis_title="Retrieved similar observation UID",
        yaxis_title="Machine failure flag",
        yaxis={
            "tickmode": "array",
            "tickvals": [0, 1],
            "ticktext": ["Healthy", "Failed"],
            "range": [-0.25, 1.25],
        },
    )


def _selected_engineered_frame(frame: pd.DataFrame, filters: AnalysisFilters | None) -> pd.DataFrame:
    selected = apply_filters(prepare_analysis_frame(frame), resolve_filters(filters))
    if selected.empty:
        raise ValueError("Chart input is empty after applying filters.")
    return selected


def _require_numeric_column(frame: pd.DataFrame, column: str) -> None:
    if column not in frame.columns:
        raise ValueError(f"Unknown chart variable: {column}")
    if not pd.api.types.is_numeric_dtype(frame[column]):
        raise ValueError(f"Chart variable must be numeric: {column}")


def _apply_layout(figure: go.Figure, title: str, **layout: object) -> go.Figure:
    """Apply a clear, print-like evidence canvas that remains readable in the Streamlit UI."""

    figure.update_layout(
        template="plotly_white",
        title={"text": title, "x": 0.01, "xanchor": "left"},
        margin={"l": 78, "r": 36, "t": 78, "b": 72},
        hovermode="closest",
        font={"family": "Arial, sans-serif", "size": 13, "color": "#1F2937"},
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        title_font={"size": 18, "color": "#102A43"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        hoverlabel={"bgcolor": "#102A43", "font": {"color": "#FFFFFF"}},
        bargap=0.38,
        **layout,
    )
    figure.update_xaxes(showgrid=False, zeroline=False, linecolor="#CBD5E1", tickfont={"color": "#334E68"})
    figure.update_yaxes(
        gridcolor="rgba(148, 163, 184, 0.28)",
        zeroline=False,
        linecolor="#CBD5E1",
        tickfont={"color": "#334E68"},
    )
    return figure
