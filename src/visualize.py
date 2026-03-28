"""
Visualization module — generates analysis plots from the filtering results CSV.

Plots generated
---------------
1. metric_distributions.png  — per-metric histogram split by KEEP/DISCARD
2. quality_score_dist.png     — quality score distribution with threshold line
3. language_stats.png         — keep/discard counts + keep-rate per language
4. correlation_heatmap.png    — metric–metric correlation matrix
5. discard_reasons.png        — pie/bar chart of discard reasons
6. summary.json               — machine-readable filtering statistics
"""

import json
import logging
import os
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# ── Styling ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})

PALETTE = {"KEEP": "#2ecc71", "DISCARD": "#e74c3c"}


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ── Individual plot functions ─────────────────────────────────────────────────

def plot_metric_distributions(
    df: pd.DataFrame, output_dir: str = "outputs/plots"
) -> None:
    """Histogram of each metric, coloured by decision."""
    _ensure_dir(output_dir)

    metrics = [
        ("snr_db",            "SNR (dB)"),
        ("silence_ratio",     "Silence Ratio"),
        ("clipping_ratio",    "Clipping Ratio"),
        ("rms_energy",        "RMS Energy"),
        ("vad_ratio",         "VAD Ratio"),
        ("dnsmos_ovr",        "DNSMOS OVR"),
        ("spectral_flatness", "Spectral Flatness"),
        ("quality_score",     "Quality Score"),
    ]
    metrics = [(col, label) for col, label in metrics if col in df.columns]

    ncols = 4
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 4))
    axes = axes.flatten()

    for i, (col, label) in enumerate(metrics):
        ax = axes[i]
        for decision, grp in df.groupby("decision"):
            ax.hist(
                grp[col].dropna(),
                bins=40,
                alpha=0.65,
                label=decision,
                color=PALETTE.get(decision, "grey"),
                density=True,
                edgecolor="white",
                linewidth=0.3,
            )
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)

    # Hide unused subplot panels
    for j in range(len(metrics), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Audio Quality Metric Distributions", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = os.path.join(output_dir, "metric_distributions.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")


def plot_quality_score_distribution(
    df: pd.DataFrame,
    threshold: float = 0.60,
    output_dir: str = "outputs/plots",
) -> None:
    """Quality score histogram with threshold line."""
    _ensure_dir(output_dir)

    kept      = (df["decision"] == "KEEP").sum()
    discarded = (df["decision"] == "DISCARD").sum()
    total     = len(df)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        df["quality_score"].dropna(),
        bins=60,
        color="#3498db",
        alpha=0.75,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.axvline(
        threshold,
        color="#e74c3c",
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {threshold}",
    )

    ax.set_xlabel("Quality Score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        f"Quality Score Distribution\n"
        f"KEEP: {kept} ({kept/max(total,1)*100:.1f}%)  |  "
        f"DISCARD: {discarded} ({discarded/max(total,1)*100:.1f}%)",
        fontsize=12,
    )
    ax.legend(fontsize=11)
    plt.tight_layout()

    out = os.path.join(output_dir, "quality_score_dist.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")


def plot_language_stats(
    df: pd.DataFrame, output_dir: str = "outputs/plots"
) -> None:
    """Per-language keep/discard counts and keep-rate bar charts."""
    _ensure_dir(output_dir)

    lang_counts = (
        df.groupby(["language", "decision"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["KEEP", "DISCARD"], fill_value=0)
    )
    lang_counts["total"]     = lang_counts.sum(axis=1)
    lang_counts["keep_rate"] = lang_counts["KEEP"] / lang_counts["total"].clip(lower=1) * 100
    lang_counts = lang_counts.sort_values("total", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: stacked bar count
    lang_counts[["KEEP", "DISCARD"]].plot(
        kind="bar", ax=axes[0],
        color=["#2ecc71", "#e74c3c"],
        alpha=0.85,
        width=0.65,
    )
    axes[0].set_title("Sample Counts by Language", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Language")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=40)
    axes[0].legend()

    # Right: keep rate
    axes[1].bar(
        lang_counts.index,
        lang_counts["keep_rate"],
        color="#3498db",
        alpha=0.85,
        width=0.65,
    )
    axes[1].axhline(50, color="red", linestyle="--", alpha=0.5, linewidth=1.2)
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter())
    axes[1].set_title("Keep Rate by Language", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Language")
    axes[1].set_ylabel("Keep Rate (%)")
    axes[1].tick_params(axis="x", rotation=40)

    plt.suptitle("Per-Language Filtering Statistics", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out = os.path.join(output_dir, "language_stats.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")


def plot_correlation_heatmap(
    df: pd.DataFrame, output_dir: str = "outputs/plots"
) -> None:
    """Pearson correlation heatmap between numeric metrics."""
    _ensure_dir(output_dir)

    numeric_cols = [
        "snr_db", "silence_ratio", "clipping_ratio", "rms_energy",
        "vad_ratio", "dnsmos_ovr", "spectral_flatness", "zcr",
        "spectral_centroid", "quality_score",
    ]
    cols = [c for c in numeric_cols if c in df.columns]
    if len(cols) < 2:
        return

    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1, vmax=1,
        ax=ax,
        square=True,
        linewidths=0.4,
        annot_kws={"size": 9},
    )
    ax.set_title("Metric Correlation Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")


def plot_discard_reasons(
    df: pd.DataFrame, output_dir: str = "outputs/plots"
) -> None:
    """Bar chart of discard reason frequencies."""
    _ensure_dir(output_dir)

    discards = df[df["decision"] == "DISCARD"]["reason"]
    if discards.empty:
        return

    reason_counts = discards.value_counts().head(15)

    fig, ax = plt.subplots(figsize=(12, 5))
    reason_counts.plot(kind="barh", ax=ax, color="#e74c3c", alpha=0.8)
    ax.set_xlabel("Count")
    ax.set_title("Top Discard Reasons", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()

    out = os.path.join(output_dir, "discard_reasons.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")


# ── Summary report ─────────────────────────────────────────────────────────────

def generate_summary_report(
    df: pd.DataFrame, output_dir: str = "outputs/plots"
) -> dict:
    """Build and save a JSON summary of filtering statistics."""
    _ensure_dir(output_dir)

    total     = len(df)
    kept      = int((df["decision"] == "KEEP").sum())
    discarded = int((df["decision"] == "DISCARD").sum())

    kept_df     = df[df["decision"] == "KEEP"]
    discard_df  = df[df["decision"] == "DISCARD"]

    summary = {
        "total_samples": total,
        "kept": kept,
        "discarded": discarded,
        "keep_rate_pct": round(kept / max(total, 1) * 100, 2),
        "avg_quality_score": round(float(df["quality_score"].mean()), 4),
        "metrics_kept": {
            col: round(float(kept_df[col].mean()), 4)
            for col in ["snr_db", "vad_ratio", "dnsmos_ovr", "silence_ratio"]
            if col in kept_df.columns
        },
        "metrics_discarded": {
            col: round(float(discard_df[col].mean()), 4)
            for col in ["snr_db", "vad_ratio", "dnsmos_ovr", "silence_ratio"]
            if col in discard_df.columns and not discard_df[col].isna().all()
        },
        "top_discard_reasons": (
            df[df["decision"] == "DISCARD"]["reason"]
            .value_counts()
            .head(10)
            .to_dict()
        ),
        "per_language": (
            df.groupby("language")["decision"]
            .value_counts()
            .unstack(fill_value=0)
            .to_dict()
        ),
    }

    out = os.path.join(output_dir, "summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Summary saved: {out}")

    return summary


# ── Master function ───────────────────────────────────────────────────────────

def generate_all_plots(
    csv_path: str = "outputs/results.csv",
    output_dir: str = "outputs/plots",
    threshold: float = 0.60,
) -> Optional[dict]:
    """Generate all plots and the summary report from a results CSV."""
    if not os.path.exists(csv_path):
        logger.warning(f"Results CSV not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        logger.warning("Results CSV is empty — nothing to plot.")
        return None

    logger.info(f"Generating visualizations for {len(df)} samples …")

    plot_metric_distributions(df, output_dir)
    plot_quality_score_distribution(df, threshold, output_dir)
    plot_language_stats(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    plot_discard_reasons(df, output_dir)
    summary = generate_summary_report(df, output_dir)

    print("\n" + "=" * 50)
    print("  FILTERING SUMMARY")
    print("=" * 50)
    print(f"  Total samples : {summary['total_samples']}")
    print(f"  Kept          : {summary['kept']}  ({summary['keep_rate_pct']}%)")
    print(f"  Discarded     : {summary['discarded']}")
    print(f"  Avg quality   : {summary['avg_quality_score']}")
    print(f"\n  Plots saved to: {output_dir}/")
    print("=" * 50)

    return summary
