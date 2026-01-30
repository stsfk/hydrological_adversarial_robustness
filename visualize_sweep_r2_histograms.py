#!/usr/bin/env python3
"""
Visualise R² distributions of epsilon-vs-performance linear fits for LSTM and HBV FGSM sweeps.

Creates a single 2×2 figure with histograms for:
    (a) LSTM ΔKGE vs ε R²
    (b) LSTM ΔMSE vs ε R²
    (c) HBV ΔKGE vs ε R²
    (d) HBV ΔMSE vs ε R²
Each subplot highlights the median R² value.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lstm-summary",
        type=Path,
        default=Path("data") / "lstm_fgsm_effectiveness_summary.csv",
        help="Path to the LSTM FGSM effectiveness summary CSV.",
    )
    parser.add_argument(
        "--hbv-summary",
        type=Path,
        default=Path("data") / "hbv_fgsm_effectiveness_summary.csv",
        help="Path to the HBV FGSM effectiveness summary CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "figures" / "fgsm_r2_histograms.pdf",
        help="Output path for the generated figure (PDF).",
    )
    parser.add_argument("--bins", type=int, default=60, help="Number of histogram bins.")
    return parser.parse_args()


def load_r2_values(path: Path) -> Tuple[pd.Series, pd.Series]:
    df = pd.read_csv(path)
    kge_r2 = pd.to_numeric(df["kge_linear_r2"], errors="coerce")
    mse_r2 = pd.to_numeric(df["mse_linear_r2"], errors="coerce")
    return kge_r2.dropna(), mse_r2.dropna()


def add_histogram(
    ax: plt.Axes,
    values: pd.Series,
    bin_edges: np.ndarray,
    title: str,
    panel_tag: str,
    global_xlim: tuple[float, float],
) -> float:
    """Draw a histogram using shared bin edges and return the subplot's max count.

    Ensures all subplots share the same x-axis limits by using the provided
    bin_edges/global_xlim. Returns the maximum bin count for later y-axis sync.
    """
    finite_vals = values.replace([np.inf, -np.inf], np.nan).dropna()
    if finite_vals.empty:
        ax.text(0.5, 0.5, "No finite values", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{panel_tag} {title}")
        ax.set_xlabel("R² of linear model")
        ax.set_ylabel("Count")
        ax.set_xlim(global_xlim)
        ax.grid(alpha=0.2, linestyle="--", linewidth=0.6)
        return 0.0

    # Local stats for annotation, but global binning/limits for consistency.
    local_min = float(finite_vals.min())
    local_max = float(finite_vals.max())

    counts, _, _ = ax.hist(
        finite_vals,
        bins=bin_edges,
        color="#4C72B0",
        alpha=0.75,
        edgecolor="black",
    )
    def _fmt_natural(v: float) -> str:
        """Format with proper rounding.

        - For |v| >= 1e-2: show three decimals (e.g., 0.123 -> 0.123).
        - For very small values: round half-up to the first significant digit after
          the leading zeros (e.g., 0.00045 -> 0.0005; 0.002345 -> 0.002).
        - For exactly zero: 0.000.
        """
        try:
            if not np.isfinite(v):
                return str(v)
            if abs(v) >= 1e-2:
                return f"{v:.3f}"
            if v == 0:
                return "0.000"
            import numpy as _np
            from decimal import Decimal, ROUND_HALF_UP
            val = abs(float(v))
            e = int(_np.floor(-_np.log10(val)))
            decimals = max(1, min(e + 1, 12))
            rounded = Decimal(str(val)).quantize(Decimal(f"1e-{decimals}"), rounding=ROUND_HALF_UP)
            s = f"{rounded:.{decimals}f}"
            return ("-" if v < 0 else "") + s
        except Exception:
            return str(v)

    median_val = float(finite_vals.median())
    ax.axvline(
        median_val,
        color="#DD8452",
        linestyle="--",
        linewidth=1.5,
        label=f"Median = {_fmt_natural(median_val)}",
    )
    ax.set_xlim(global_xlim)
    ax.set_xlabel("R² of linear model")
    ax.set_ylabel("Count")
    ax.set_title(f"{panel_tag} {title}")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.2, linestyle="--", linewidth=0.6)

    def _fmt(v: float) -> str:
        return _fmt_natural(v)

    ax.text(
        0.02,
        0.9,
        f"Min = {_fmt(local_min)}\nMax = {_fmt(local_max)}",
        ha="left",
        va="top",
        transform=ax.transAxes,
        fontsize=8,
        color="0.35",
    )
    return float(np.max(counts)) if counts.size else 0.0


def main() -> None:
    args = parse_args()

    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    lstm_kge_r2, lstm_mse_r2 = load_r2_values(args.lstm_summary)
    hbv_kge_r2, hbv_mse_r2 = load_r2_values(args.hbv_summary)

    # Establish global x-limits and shared bin edges across all panels.
    all_vals = pd.concat([lstm_kge_r2, lstm_mse_r2, hbv_kge_r2, hbv_mse_r2], ignore_index=True)
    finite_all = all_vals.replace([np.inf, -np.inf], np.nan).dropna()
    if finite_all.empty:
        x_min, x_max = 0.0, 1.0
    else:
        x_min = float(finite_all.min())
        x_max = float(finite_all.max())
        if x_min == x_max:
            pad = 1e-6
            x_min -= pad
            x_max += pad
    global_xlim = (x_min, x_max)
    bin_edges = np.linspace(x_min, x_max, args.bins + 1)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=False, sharey=False)

    ymax_values = []
    ymax_values.append(
        add_histogram(
            axes[0, 0],
            lstm_kge_r2,
            bin_edges,
            "LSTM ΔKGE vs ε",
            "(a)",
            global_xlim,
        )
    )
    ymax_values.append(
        add_histogram(
            axes[0, 1],
            lstm_mse_r2,
            bin_edges,
            "LSTM ΔMSE vs ε",
            "(b)",
            global_xlim,
        )
    )
    ymax_values.append(
        add_histogram(
            axes[1, 0],
            hbv_kge_r2,
            bin_edges,
            "HBV ΔKGE vs ε",
            "(c)",
            global_xlim,
        )
    )
    ymax_values.append(
        add_histogram(
            axes[1, 1],
            hbv_mse_r2,
            bin_edges,
            "HBV ΔMSE vs ε",
            "(d)",
            global_xlim,
        )
    )

    # Synchronize y-axis across panels for consistent visual comparison.
    global_ymax = max(ymax_values) if ymax_values else 0.0
    if global_ymax > 0:
        upper = global_ymax * 1.05
        for ax in axes.ravel():
            ax.set_ylim(0, upper)

    fig.tight_layout()

    pdf_path = args.output.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved histogram figure to {pdf_path}")


if __name__ == "__main__":
    main()
