"""
Chart engine — produces static PNG charts (matplotlib) for embedding in
DOCX reports, plus Plotly chart factories for interactive use in Streamlit.

Two chart types:
- Competitor bar charts (target vs N peers across one metric)
- 10-year financial line charts (Revenue, Net Income, FCF over time)
"""
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless backend — required for Streamlit / server use
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Brand palette (mirrors the DOCX's heading colour for visual consistency)
BRAND_NAVY = "#003366"
BRAND_LIGHT = "#5189c8"
PEER_GREY = "#9aa5b1"


def _format_pct(x, _pos=None):
    return f"{x:.0f}%"


def _format_money(x, _pos=None):
    if abs(x) >= 1_000:
        return f"${x/1_000:.1f}B"
    return f"${x:.0f}M"


# ============================================================================
# Competitor bar charts (matplotlib → PNG bytes)
# ============================================================================

def make_competitor_bar_chart_png(
    metric_label: str,
    metrics_df: pd.DataFrame,
    metric_column: str,
    target_symbol: str,
    is_percentage: bool = True,
    width_in: float = 6.4,
    height_in: float = 3.4,
    dpi: int = 150,
) -> Optional[BytesIO]:
    """
    Render a horizontal bar chart of one metric across target + peers.

    Args:
        metric_label: human-readable label for the chart title (e.g. "Return on Equity")
        metrics_df: DataFrame from peer_finder.build_peer_metrics_frame
                    (index = ticker, columns = metrics)
        metric_column: which column to plot
        target_symbol: ticker of the target — highlighted in brand colour
        is_percentage: if True, format axis as percentages
        width_in / height_in / dpi: matplotlib figure sizing

    Returns:
        BytesIO of PNG bytes, ready to embed in DOCX with `add_picture`.
        None if metric_column is missing or all values are NaN.
    """
    if metric_column not in metrics_df.columns:
        return None

    series = metrics_df[metric_column].dropna()
    if series.empty:
        return None

    # Sort so target sits at the top, peers below ranked by metric value
    target_val = series.get(target_symbol)
    peers = series.drop(target_symbol, errors="ignore").sort_values(ascending=True)
    ordered = peers.copy()
    if target_val is not None:
        ordered.loc[target_symbol] = target_val
    # Re-order so target is on top of the chart
    sym_order = list(peers.index) + [target_symbol] if target_val is not None else list(peers.index)
    ordered = ordered.loc[sym_order]

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Georgia", "DejaVu Serif", "Times New Roman", "serif"],
    })

    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    colors = [BRAND_NAVY if s == target_symbol else PEER_GREY for s in ordered.index]
    bars = ax.barh(ordered.index, ordered.values, color=colors, edgecolor="white", height=0.55)

    # Value labels at the end of each bar
    for bar, val in zip(bars, ordered.values):
        ax.text(
            bar.get_width() + (0.01 * abs(ordered.values.max() - ordered.values.min() or 1)),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}{'%' if is_percentage else ''}",
            va="center",
            fontsize=12, weight="bold", color="#222222",
        )

    ax.set_title(
        f"{metric_label} — {target_symbol} vs peers",
        fontsize=15, weight="bold", pad=14, color="#111111", family="serif",
    )
    if is_percentage:
        ax.xaxis.set_major_formatter(FuncFormatter(_format_pct))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(axis="y", labelsize=12, colors="#222222")
    ax.tick_params(axis="x", labelsize=11, colors="#666666")
    ax.grid(axis="x", linestyle=":", alpha=0.55, color="#cccccc")
    ax.set_axisbelow(True)
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


# ============================================================================
# 10-year line charts (matplotlib → PNG bytes)
# ============================================================================

def make_ten_year_line_chart_png(
    title: str,
    series_by_label: Dict[str, List[Tuple[str, float]]],
    y_format: str = "money",
    width_in: float = 6.5,
    height_in: float = 3.5,
    dpi: int = 150,
) -> Optional[BytesIO]:
    """
    Render a multi-series line chart of historical financials.

    Args:
        title: chart title
        series_by_label: dict mapping series name → list of (year, value) tuples
                         e.g. {"Revenue": [("2015", 100), ("2016", 120), ...]}
        y_format: 'money' for $M / $B formatting, 'pct' for percentage, else raw
        width_in / height_in / dpi: figure sizing

    Returns:
        BytesIO of PNG bytes; None if no data.
    """
    if not series_by_label or all(not pts for pts in series_by_label.values()):
        return None

    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)

    palette = [BRAND_NAVY, BRAND_LIGHT, "#d97706", "#059669"]  # extend as needed
    for i, (label, points) in enumerate(series_by_label.items()):
        if not points:
            continue
        years = [p[0] for p in points]
        values = [p[1] for p in points]
        ax.plot(years, values, marker="o", linewidth=2.0,
                color=palette[i % len(palette)], label=label)

    ax.set_title(title, fontsize=11, weight="bold", pad=10)
    if y_format == "money":
        ax.yaxis.set_major_formatter(FuncFormatter(_format_money))
    elif y_format == "pct":
        ax.yaxis.set_major_formatter(FuncFormatter(_format_pct))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="best", fontsize=9, frameon=False)
    ax.tick_params(axis="both", labelsize=8)
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


# ============================================================================
# VIA ATLAS-style chart (mirrors the client's standard report template)
# Single-metric line chart with: trend line, "Initial Published Date" marker,
# boxed latest-value annotation, and red end-date label.
# ============================================================================

def make_via_atlas_chart_png(
    chart_number: int,
    section_title: str,
    metric_label: str,
    points: List[Tuple[str, float]],
    initial_published_date: Optional[str] = None,
    color: str = "#ee6c1f",
    y_format: str = "money",
    width_in: float = 6.4,
    height_in: float = 4.4,
    dpi: int = 150,
) -> Optional[BytesIO]:
    """
    Render a VIA ATLAS-style chart: a single metric over time with a dashed
    linear trend line, a red vertical "Initial Published Date" marker, a green
    circle highlighting the data point at that date, a boxed annotation
    showing the latest value, and a red end-date label tile.

    Args:
        chart_number: number prefix shown in the title (e.g. 5 -> "5. ...")
        section_title: bold heading (e.g. "Cash Flow From Operations")
        metric_label: subtitle (e.g. "Operating Cash Flow")
        points: list of (date_str, value) tuples ordered chronologically.
                date_str format e.g. "2024-10" or "2024".
        initial_published_date: date string matching one of the point labels;
                if None, defaults to ~70% of the way through the series.
        color: line color (orange / green / black / purple ... rotate per chart).
        y_format: "money" | "pct" | "ratio"

    Returns:
        BytesIO of PNG bytes; None if no data.
    """
    if not points or len(points) < 2:
        return None

    import numpy as np

    dates = [p[0] for p in points]
    values = [p[1] for p in points]

    # Numeric x-axis for trend line fitting
    x = np.arange(len(dates))
    y = np.array(values, dtype=float)
    valid = ~np.isnan(y)
    if valid.sum() < 2:
        return None

    # Use a clean serif family for the VIA template aesthetic
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Georgia", "DejaVu Serif", "Times New Roman", "serif"],
    })

    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Faint "VIA" watermark behind the plot area
    fig.text(
        0.5, 0.45, "VIA",
        ha="center", va="center",
        fontsize=170, color="#d8d8d8", alpha=0.35,
        weight="bold", family="serif",
        zorder=0,
    )

    # Main data line with open circle markers
    ax.plot(
        x, y,
        color=color, linewidth=2.4,
        marker="o", markersize=7, markerfacecolor="white",
        markeredgecolor=color, markeredgewidth=1.8,
        zorder=3,
    )

    # Dashed mint-green linear trend line
    if valid.sum() >= 2:
        coeffs = np.polyfit(x[valid], y[valid], 1)
        trend = np.poly1d(coeffs)(x)
        ax.plot(
            x, trend,
            color="#34d399", linewidth=1.8, linestyle="--",
            alpha=0.9, zorder=2,
        )

    # Initial Published Date: vertical red line + green circle on intersection
    if initial_published_date is None:
        ipd_idx = int(len(dates) * 0.7)  # default ~70% through
    else:
        try:
            ipd_idx = dates.index(initial_published_date)
        except ValueError:
            # Try a less strict match (year prefix)
            ipd_year = initial_published_date.split("-")[0]
            ipd_idx = next(
                (i for i, d in enumerate(dates) if str(d).startswith(ipd_year)),
                int(len(dates) * 0.7),
            )

    if 0 <= ipd_idx < len(dates):
        ipd_value = values[ipd_idx]
        ax.axvline(
            x=ipd_idx,
            color="#dc2626", linewidth=1.6, alpha=0.85, zorder=4,
        )
        # Green highlight circle on the data point at IPD
        ax.scatter(
            [ipd_idx], [ipd_value],
            s=180, facecolor="none", edgecolor="#10b981",
            linewidths=2.4, zorder=5,
        )
        # Red label "Initial Published Date" near IPD
        y_range = max(y[valid]) - min(y[valid]) if valid.sum() > 1 else 1.0
        ax.annotate(
            "Initial Published Date",
            xy=(ipd_idx, ipd_value),
            xytext=(ipd_idx - 1.8, ipd_value + y_range * 0.18),
            fontsize=12, color="#dc2626", style="italic",
            ha="left", zorder=6,
        )

    # Boxed annotation at the right end showing the latest value
    last_x = len(dates) - 1
    last_y = values[-1]
    if y_format == "money":
        last_label = f"{last_y:,.2f}"
    elif y_format == "pct":
        last_label = f"{last_y:.2f}"
    else:
        last_label = f"{last_y:.2f}"
    ax.annotate(
        last_label,
        xy=(last_x, last_y),
        xytext=(10, 0),
        textcoords="offset points",
        fontsize=11, weight="bold",
        bbox=dict(boxstyle="round,pad=0.4", fc="white",
                  ec="#444444", linewidth=0.9),
        va="center",
        zorder=6,
    )

    # Red end-date label tile at the very right edge along the x-axis
    end_date_label = dates[-1]
    ax.annotate(
        end_date_label,
        xy=(last_x, ax.get_ylim()[0]),
        xytext=(0, -28),
        textcoords="offset points",
        fontsize=11, color="white", weight="bold",
        ha="center", va="top",
        bbox=dict(boxstyle="square,pad=0.45", fc="#dc2626",
                  ec="#dc2626"),
        zorder=6,
    )

    # Y-axis formatter
    if y_format == "money":
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:,.0f}"))
    elif y_format == "pct":
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.0f}"))

    # Title block: "N. Section Title" / metric subtitle (serif, VIA-style)
    title_text = f"{chart_number}. {section_title}"
    fig.text(
        0.05, 0.94, title_text,
        ha="left", fontsize=26, weight="bold",
        color="#111111", family="serif",
    )
    fig.text(
        0.5, 0.86, metric_label,
        ha="center", fontsize=17, style="italic",
        color="#333333", family="serif",
    )

    # X-axis labels — show every ~Nth so they don't crowd
    step = max(1, len(dates) // 9)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([dates[i] for i in range(0, len(dates), step)],
                        fontsize=11, rotation=0)

    # Grid + spine cleanup
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.grid(axis="y", linestyle=":", alpha=0.55, color="#cccccc")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=11, colors="#666666")

    # Pad y-axis so annotations don't get clipped
    if valid.sum() > 1:
        y_min, y_max = min(y[valid]), max(y[valid])
        y_range = y_max - y_min
        ax.set_ylim(y_min - y_range * 0.15, y_max + y_range * 0.25)

    fig.tight_layout(rect=[0.04, 0.04, 0.98, 0.83])
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


# ============================================================================
# Plotly chart factories (for Streamlit interactive views)
# ============================================================================

def make_competitor_bar_chart_plotly(
    metric_label: str,
    metrics_df: pd.DataFrame,
    metric_column: str,
    target_symbol: str,
    is_percentage: bool = True,
):
    """Plotly version of the competitor bar chart for Streamlit."""
    import plotly.graph_objects as go

    if metric_column not in metrics_df.columns:
        return None
    series = metrics_df[metric_column].dropna()
    if series.empty:
        return None

    target_val = series.get(target_symbol)
    peers = series.drop(target_symbol, errors="ignore").sort_values(ascending=True)
    sym_order = list(peers.index) + ([target_symbol] if target_val is not None else [])
    values = [series[s] for s in sym_order]
    colors = [BRAND_NAVY if s == target_symbol else PEER_GREY for s in sym_order]

    suffix = "%" if is_percentage else ""
    text = [f"{v:.1f}{suffix}" for v in values]

    fig = go.Figure(data=[
        go.Bar(
            x=values, y=sym_order, orientation="h",
            marker_color=colors, text=text, textposition="outside",
        )
    ])
    fig.update_layout(
        title=f"{metric_label} — {target_symbol} vs peers",
        margin=dict(l=60, r=40, t=50, b=40),
        height=350,
        xaxis_title=metric_label + (" (%)" if is_percentage else ""),
        yaxis_title=None,
        showlegend=False,
    )
    return fig


def make_ten_year_line_chart_plotly(
    title: str,
    series_by_label: Dict[str, List[Tuple[str, float]]],
):
    """Plotly version of the 10-year line chart for Streamlit."""
    import plotly.graph_objects as go

    if not series_by_label:
        return None
    fig = go.Figure()
    for label, points in series_by_label.items():
        if not points:
            continue
        years = [p[0] for p in points]
        values = [p[1] for p in points]
        fig.add_trace(go.Scatter(
            x=years, y=values, mode="lines+markers", name=label,
        ))
    fig.update_layout(
        title=title,
        margin=dict(l=60, r=40, t=50, b=40),
        height=350,
        xaxis_title="Year",
        yaxis_title="Value",
        legend=dict(orientation="h", y=-0.2),
    )
    return fig
