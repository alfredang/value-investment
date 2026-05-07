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
    width_in: float = 6.0,
    height_in: float = 3.5,
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

    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)

    colors = [BRAND_NAVY if s == target_symbol else PEER_GREY for s in ordered.index]
    bars = ax.barh(ordered.index, ordered.values, color=colors, edgecolor="white")

    # Value labels at the end of each bar
    for bar, val in zip(bars, ordered.values):
        ax.text(
            bar.get_width() + (0.01 * abs(ordered.values.max() - ordered.values.min() or 1)),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}{'%' if is_percentage else ''}",
            va="center",
            fontsize=9,
        )

    ax.set_title(f"{metric_label} — {target_symbol} vs peers", fontsize=11, weight="bold", pad=10)
    if is_percentage:
        ax.xaxis.set_major_formatter(FuncFormatter(_format_pct))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
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
