#!/usr/bin/env python3
"""
Create per-vertical scatter plots of MAU vs Annual Revenue with 20% quantile ticks.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "251201 oppty list.csv"
OUTPUT_PATH = ROOT_DIR / "vertical_mau_revenue_quantile_scatter.png"


def _format_value(value: float) -> str:
    """Format large numbers with unit suffixes."""
    if value is None or np.isnan(value):
        return "NaN"

    abs_value = abs(value)
    if abs_value >= 1e12:
        return f"{value / 1e12:.2f}T"
    if abs_value >= 1e9:
        return f"{value / 1e9:.2f}B"
    if abs_value >= 1e6:
        return f"{value / 1e6:.2f}M"
    if abs_value >= 1e3:
        return f"{value / 1e3:.2f}K"
    return f"{value:.0f}"


def _parse_numeric(series: pd.Series) -> pd.Series:
    """Convert comma-formatted numeric strings to floats."""
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"": np.nan, "-": np.nan, "nan": np.nan, "None": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _quantile_ticks(values: np.ndarray) -> tuple[list[float], list[str]]:
    """Return tick values and labels at every 20% quantile."""
    if values.size == 0:
        return [], []

    quantiles = np.linspace(0, 1, 6)
    ticks = np.quantile(values, quantiles, method="linear")
    labels = [f"{int(q * 100)}% ({_format_value(tick)})" for q, tick in zip(quantiles, ticks)]
    return ticks.tolist(), labels


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Annual Revenue"] = _parse_numeric(df["Annual Revenue"])
    df["MAU"] = _parse_numeric(df["MAU"])
    df = df.dropna(subset=["Annual Revenue", "MAU", "Vertical"])
    return df


def _display_name(vertical_value: str) -> str:
    """Use the English descriptor when available to avoid missing glyphs."""
    if "|" in vertical_value:
        return vertical_value.split("|")[-1].strip()
    return vertical_value


def plot_scatter(df: pd.DataFrame, output_path: Path) -> None:
    df = df.copy()
    df["Vertical Display"] = df["Vertical"].apply(_display_name)
    verticals = sorted(df["Vertical"].unique())
    if not verticals:
        raise ValueError("No vertical data found after filtering.")

    cols = 3
    rows = math.ceil(len(verticals) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.8, rows * 4.6), squeeze=False)

    for idx, vertical in enumerate(verticals):
        ax = axes[idx // cols][idx % cols]
        subset = df[df["Vertical"] == vertical]
        ax.scatter(
            subset["MAU"],
            subset["Annual Revenue"],
            s=36,
            alpha=0.7,
            edgecolors="black",
            linewidths=0.3,
        )

        xticks, xlabels = _quantile_ticks(subset["MAU"].to_numpy())
        yticks, ylabels = _quantile_ticks(subset["Annual Revenue"].to_numpy())

        if xticks:
            ax.set_xticks(xticks)
            ax.set_xticklabels(xlabels, rotation=35, ha="right", fontsize=8)
        if yticks:
            ax.set_yticks(yticks)
            ax.set_yticklabels(ylabels, fontsize=8)

        ax.set_title(_display_name(vertical), fontsize=11)
        ax.set_xlabel("MAU", fontsize=9)
        ax.set_ylabel("Annual Revenue (KRW)", fontsize=9)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    # Hide any unused axes.
    total_axes = rows * cols
    for idx in range(len(verticals), total_axes):
        axes[idx // cols][idx % cols].axis("off")

    fig.suptitle("MAU vs Annual Revenue by Vertical (20% Quantile Ticks)", fontsize=14, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved scatter grid to {output_path}")


def main() -> None:
    df = load_data(DATA_PATH)
    plot_scatter(df, OUTPUT_PATH)


if __name__ == "__main__":
    main()
