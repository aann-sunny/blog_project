import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

SRC_PATH = "/workspace/251203 sf lead.csv"
OUTPUT_DIR = "/workspace/outputs"
PLOT_BASE = os.path.join(OUTPUT_DIR, "vertical")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(r"[^0-9.\-]", "", regex=True)
        .replace("", np.nan)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def format_tick(value: float) -> str:
    if value >= 1e12:
        return f"{value / 1e12:.2f}T"
    if value >= 1e9:
        return f"{value / 1e9:.2f}B"
    if value >= 1e6:
        return f"{value / 1e6:.2f}M"
    if value >= 1e3:
        return f"{value / 1e3:.2f}K"
    return f"{value:.0f}"


def build_scatter(df_valid: pd.DataFrame, mau_q: pd.Series, rev_q: pd.Series, path: str):
    fig, ax = plt.subplots(figsize=(12, 8))
    vertical_groups = list(df_valid.groupby("Vertical Display"))
    cmap = plt.colormaps.get_cmap("tab20").resampled(len(vertical_groups))

    for idx, (vertical, subset) in enumerate(vertical_groups):
        ax.scatter(
            subset["MAU_numeric"],
            subset["Annual Revenue_numeric"],
            s=30,
            label=vertical,
            color=cmap(idx),
            alpha=0.8,
            edgecolor="none",
        )

    ax.set_xlabel("Monthly Active Users (MAU)")
    ax.set_ylabel("Annual Revenue")
    ax.set_xticks(mau_q.values)
    ax.set_xticklabels([format_tick(v) for v in mau_q.values])
    ax.set_yticks(rev_q.values)
    ax.set_yticklabels([format_tick(v) for v in rev_q.values])
    ax.set_title("Vertical Scatter: MAU vs Annual Revenue")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def build_cluster_plot(df_valid: pd.DataFrame, mau_q: pd.Series, rev_q: pd.Series, path: str):
    tier_order = [f"Tier {idx}" for idx in range(1, 6)]
    tier_palette = plt.colormaps.get_cmap("viridis").resampled(len(tier_order))
    fig, ax = plt.subplots(figsize=(12, 8))

    for idx, tier in enumerate(tier_order):
        tier_subset = df_valid[df_valid["Tier Cluster"] == tier]
        if tier_subset.empty:
            continue
        ax.scatter(
            tier_subset["MAU_numeric"],
            tier_subset["Annual Revenue_numeric"],
            s=30,
            color=tier_palette(idx),
            label=tier,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.2,
        )

    ax.set_xlabel("Monthly Active Users (MAU)")
    ax.set_ylabel("Annual Revenue")
    ax.set_xticks(mau_q.values)
    ax.set_xticklabels([format_tick(v) for v in mau_q.values])
    ax.set_yticks(rev_q.values)
    ax.set_yticklabels([format_tick(v) for v in rev_q.values])
    ax.set_title("Clustered Scatter: MAU vs Annual Revenue with Tiers")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(title="Tier", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    df_raw = pd.read_csv(SRC_PATH, dtype=str).fillna("")
    df_raw["Annual Revenue_numeric"] = clean_numeric(df_raw["Annual Revenue"])
    df_raw["MAU_numeric"] = clean_numeric(df_raw["MAU"])

    df_valid = df_raw.dropna(subset=["Annual Revenue_numeric", "MAU_numeric"]).copy()
    if df_valid.empty:
        raise ValueError("No rows contain both Annual Revenue and MAU numeric values.")

    df_valid["Vertical Display"] = (
        df_valid["Vertical (New)"]
        .fillna("Unknown")
        .apply(
            lambda value: value.split("|")[-1].strip()
            if isinstance(value, str) and "|" in value
            else (value if isinstance(value, str) and value else "Unknown")
        )
    )

    quantiles = np.linspace(0.0, 1.0, 6)
    mau_quantiles = df_valid["MAU_numeric"].quantile(quantiles)
    rev_quantiles = df_valid["Annual Revenue_numeric"].quantile(quantiles)

    build_scatter(
        df_valid,
        mau_quantiles,
        rev_quantiles,
        os.path.join(OUTPUT_DIR, "vertical_scatter.png"),
    )

    features = np.log1p(df_valid[["MAU_numeric", "Annual Revenue_numeric"]])
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=20)
    df_valid["cluster_id"] = kmeans.fit_predict(features)
    centers_log = kmeans.cluster_centers_
    centers = pd.DataFrame(
        {
            "cluster_id": range(len(centers_log)),
            "MAU_numeric": np.expm1(centers_log[:, 0]),
            "Annual Revenue_numeric": np.expm1(centers_log[:, 1]),
        }
    )
    centers = centers.sort_values(
        by=["Annual Revenue_numeric", "MAU_numeric"], ascending=False
    ).reset_index(drop=True)
    centers["Tier Cluster"] = [f"Tier {idx}" for idx in range(1, len(centers) + 1)]
    tier_map = {row.cluster_id: row["Tier Cluster"] for _, row in centers.iterrows()}
    df_valid["Tier Cluster"] = df_valid["cluster_id"].map(tier_map)

    build_cluster_plot(
        df_valid,
        mau_quantiles,
        rev_quantiles,
        os.path.join(OUTPUT_DIR, "clustered_scatter.png"),
    )

    quintiles = df_valid.groupby("Vertical (New)")[["MAU_numeric", "Annual Revenue_numeric"]].quantile(
        [0.2, 0.4, 0.6, 0.8, 1.0]
    )
    quintiles = quintiles.unstack()
    quintiles.columns = [
        f"{metric}_p{int(level * 100)}" for metric, level in quintiles.columns
    ]
    quintiles = quintiles.reset_index().sort_values("Vertical (New)")
    quintiles.to_csv(os.path.join(OUTPUT_DIR, "vertical_quintiles.csv"), index=False)

    df_raw["Tier (MAU-Revenue Cluster)"] = ""
    df_raw.loc[df_valid.index, "Tier (MAU-Revenue Cluster)"] = df_valid["Tier Cluster"].values
    df_raw = df_raw.drop(columns=["Annual Revenue_numeric", "MAU_numeric"])
    df_raw.to_csv(SRC_PATH, index=False)


if __name__ == "__main__":
    main()
