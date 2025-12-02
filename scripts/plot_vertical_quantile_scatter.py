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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "251201 oppty list.csv"
PLOT_PATH = ROOT_DIR / "vertical_mau_revenue_quantile_scatter.png"
QUANTILES_PATH = ROOT_DIR / "vertical_quantiles.csv"
QUANTILE_LEVELS = (0.2, 0.4, 0.6, 0.8, 1.0)
CLUSTER_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
TIER_COLORS = [
    "#27408b",  # Tier 1 - deep blue
    "#1c8c8c",
    "#5aa02c",
    "#ff8c00",
    "#c23b22",  # Tier 5 - warm red
]


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


def _quantile_axes(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return quantile-based positions plus evenly spaced ticks and labels."""
    if values.size == 0:
        return values, np.array([]), []

    sorted_idx = np.argsort(values)
    positions = np.empty_like(values, dtype=float)
    if values.size == 1:
        positions[sorted_idx] = 0.5
    else:
        positions[sorted_idx] = np.linspace(0.0, 1.0, values.size)

    quantiles = np.linspace(0.0, 1.0, 6)
    tick_labels = [
        f"{int(q * 100)}% ({_format_value(val)})"
        for q, val in zip(quantiles, np.quantile(values, quantiles, method="linear"))
    ]
    return positions, quantiles, tick_labels


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Annual Revenue"] = _parse_numeric(df["Annual Revenue"])
    df["MAU"] = _parse_numeric(df["MAU"])
    return df


def _display_name(vertical_value: str) -> str:
    """Use the English descriptor when available to avoid missing glyphs."""
    if "|" in vertical_value:
        return vertical_value.split("|")[-1].strip()
    return vertical_value


def assign_clusters(df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    """Assign KMeans clusters using log-scaled MAU and revenue features."""
    clean_df = df.dropna(subset=["MAU", "Annual Revenue"]).copy()
    if clean_df.empty:
        raise ValueError("Cannot cluster because MAU/Annual Revenue data is empty.")

    features = np.log1p(clean_df[["MAU", "Annual Revenue"]].to_numpy(dtype=float))
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    clean_df["Cluster"] = kmeans.fit_predict(scaled)

    center_scores = kmeans.cluster_centers_.sum(axis=1)
    tier_order = np.argsort(-center_scores)
    cluster_to_tier_rank = {cluster_idx: rank for rank, cluster_idx in enumerate(tier_order)}
    cluster_to_tier_name = {cluster_idx: f"Tier {rank + 1}" for rank, cluster_idx in enumerate(tier_order)}
    cluster_to_color = {
        cluster_idx: TIER_COLORS[rank % len(TIER_COLORS)]
        for cluster_idx, rank in cluster_to_tier_rank.items()
    }

    clean_df["TierRank"] = clean_df["Cluster"].map(cluster_to_tier_rank)
    clean_df["Tier"] = clean_df["Cluster"].map(cluster_to_tier_name)
    clean_df["ClusterColor"] = clean_df["Cluster"].map(cluster_to_color)

    merged = df.copy()
    merged = merged.merge(
        clean_df[["Account", "Cluster", "TierRank", "Tier", "ClusterColor"]],
        on="Account",
        how="left",
        suffixes=("", "_cluster"),
    )
    merged["Cluster"] = merged["Cluster"].astype("Int64")
    merged["TierRank"] = merged["TierRank"].astype("Int64")
    return merged


def compute_quantile_table(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | str]] = []
    for vertical, group in df.groupby("Vertical", sort=True):
        record: dict[str, float | str] = {
            "Vertical": vertical,
            "Vertical Display": _display_name(vertical),
        }
        for metric in ("MAU", "Annual Revenue"):
            values = group[metric].dropna().to_numpy()
            for q in QUANTILE_LEVELS:
                key = f"{metric}_q{int(q * 100)}"
                if values.size == 0:
                    record[key] = np.nan
                else:
                    quant_value = float(np.quantile(values, q, method="linear"))
                    record[key] = int(round(quant_value))
        records.append(record)

    return pd.DataFrame.from_records(records)


def export_dataset_with_tier(original_path: Path, enriched_df: pd.DataFrame, output_path: Path) -> None:
    """Write a copy of the original dataset with the Tier column appended."""
    original_df = pd.read_csv(original_path)
    tier_info = enriched_df[["Account", "Tier"]].copy()
    output_df = original_df.merge(tier_info, on="Account", how="left")
    output_df.to_csv(output_path, index=False)
    print(f"Saved dataset with Tier column to {output_path}")


def plot_scatter(df: pd.DataFrame, output_path: Path) -> None:
    verticals = sorted(df["Vertical"].dropna().unique())
    if not verticals:
        raise ValueError("No vertical data found after filtering.")

    cols = 3
    rows = math.ceil(len(verticals) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.8, rows * 4.6), squeeze=False)
    tier_ranks = sorted(rank for rank in df["TierRank"].dropna().unique())
    tier_colors = {rank: TIER_COLORS[rank % len(TIER_COLORS)] for rank in tier_ranks}
    tier_names = {rank: f"Tier {rank + 1}" for rank in tier_ranks}

    for idx, vertical in enumerate(verticals):
        ax = axes[idx // cols][idx % cols]
        subset = df[
            (df["Vertical"] == vertical)
            & df["MAU"].notna()
            & df["Annual Revenue"].notna()
            & df["TierRank"].notna()
        ]
        x_positions, xticks, xlabels = _quantile_axes(subset["MAU"].to_numpy())
        y_positions, yticks, ylabels = _quantile_axes(subset["Annual Revenue"].to_numpy())

        ax.scatter(
            x_positions,
            y_positions,
            s=36,
            alpha=0.7,
            edgecolors="black",
            linewidths=0.3,
            c=subset["TierRank"].map(tier_colors).to_numpy(),
        )

        if xticks.size:
            ax.set_xticks(xticks)
            ax.set_xticklabels(xlabels, rotation=35, ha="right", fontsize=8)
            ax.set_xlim(-0.05, 1.05)
            for ref in xticks:
                ax.axvline(ref, color="#d3d3d3", linestyle="--", linewidth=0.6, zorder=0)
        if yticks.size:
            ax.set_yticks(yticks)
            ax.set_yticklabels(ylabels, fontsize=8)
            ax.set_ylim(-0.05, 1.05)
            for ref in yticks:
                ax.axhline(ref, color="#d3d3d3", linestyle="--", linewidth=0.6, zorder=0)

        ax.set_title(_display_name(vertical), fontsize=11)
        ax.set_xlabel("MAU Quantile (equal 20% bins)", fontsize=9)
        ax.set_ylabel("Revenue Quantile (equal 20% bins)", fontsize=9)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    # Hide any unused axes.
    total_axes = rows * cols
    for idx in range(len(verticals), total_axes):
        axes[idx // cols][idx % cols].axis("off")

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            label=tier_names[rank],
            markerfacecolor=tier_colors[rank],
            markeredgecolor="black",
            markersize=7,
        )
        for rank in tier_ranks
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(len(legend_handles), 5),
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.suptitle("MAU vs Annual Revenue by Vertical (Quantile Axes, 5 Clusters)", fontsize=14, weight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved scatter grid to {output_path}")


def main() -> None:
    df = load_data(DATA_PATH)
    clustered_df = assign_clusters(df)
    plot_scatter(clustered_df, PLOT_PATH)
    quantile_table = compute_quantile_table(clustered_df)
    quantile_table.to_csv(QUANTILES_PATH, index=False)
    print(f"Saved quantile table to {QUANTILES_PATH}")
    export_dataset_with_tier(DATA_PATH, clustered_df, DATA_PATH)


if __name__ == "__main__":
    main()
