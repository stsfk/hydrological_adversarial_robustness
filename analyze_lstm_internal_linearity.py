#!/usr/bin/env python3
"""
Analyse linearity of LSTM hidden outputs across FGSM epsilon values.

For a chosen catchment, the script:
    * Generates adversarial forcings using a fixed FGSM gradient sign.
    * Sweeps epsilon values and records the LSTM hidden output sequence.
    * Fits per-channel linear models (mean activation vs ε) and summarises the
      resulting absolute slopes and R² values with histograms.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from lstm_train_final_model import (
    Forcing_Data,
    LSTM_decoder,
    TimeDistributed,
    mse_loss_with_nans,
)

# Keep explicit references so torch.load can safely unpickle these classes.
_ = LSTM_decoder
_ = TimeDistributed

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
FIGURE_DIR = DATA_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
SEQ_LEN = 365 * 2
TGT_LEN = 365
BASE_LEN = SEQ_LEN - TGT_LEN


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catchment", default="DE911520", help="Catchment identifier to analyse.")
    parser.add_argument("--eps-min", type=float, default=0.05, help="Smallest epsilon (exclusive of baseline).")
    parser.add_argument("--eps-max", type=float, default=0.5, help="Largest epsilon.")
    parser.add_argument("--eps-step", type=float, default=0.05, help="Step size between epsilon values.")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu", help="Torch device to use.")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Custom prefix stem for output files (defaults to catchment name).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=60,
        help="Number of histogram bins.",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")
    return torch.device("cpu")


def prepare_epsilons(min_eps: float, max_eps: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("eps-step must be positive.")
    eps_values = [0.0]
    current = min_eps
    while current <= max_eps + 1e-9:
        eps_values.append(round(float(current), 4))
        current += step
    return sorted(set(eps_values))


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def load_dataset(device: torch.device) -> Forcing_Data:
    return Forcing_Data(
        str(DATA_DIR / "data_test_CAMELS_DE1.00.csv"),
        record_length=4018,
        n_feature=3,
        storage_device="cpu",
        seq_length=SEQ_LEN,
        target_seq_length=TGT_LEN,
        base_length=BASE_LEN,
    )


def load_models(device: torch.device) -> Tuple[torch.nn.Module, torch.nn.Module]:
    embedding = torch.load(DATA_DIR / "embedding_full.pt", map_location=device, weights_only=False)
    decoder = torch.load(DATA_DIR / "decoder_full.pt", map_location=device, weights_only=False)
    embedding.eval()
    decoder.eval()
    for module in (embedding, decoder):
        for param in module.parameters():
            param.requires_grad_(False)
        module.to(device)
    return embedding, decoder


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------
def compute_fgsm_gradient_sign(
    decoder: LSTM_decoder,
    embedding: torch.nn.Module,
    catchment_idx: int,
    inputs: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    x_for_grad = inputs.clone().requires_grad_(True)
    code = embedding(torch.tensor([catchment_idx], device=inputs.device))
    pred = decoder.decode(code, x_for_grad)
    loss = mse_loss_with_nans(pred, target)
    loss.backward()
    grad = torch.nan_to_num(x_for_grad.grad, nan=0.0, posinf=0.0, neginf=0.0)
    return grad.sign().detach()


@torch.no_grad()
def collect_lstm_outputs(
    decoder: LSTM_decoder,
    embedding: torch.nn.Module,
    catchment_idx: int,
    inputs: torch.Tensor,
) -> np.ndarray:
    """
    Run the LSTM part of the decoder and return the hidden output sequence.

    Returns:
        np.ndarray with shape [time, hidden_units]
    """
    code = embedding(torch.tensor([catchment_idx], device=inputs.device))
    code_expanded = code.expand(inputs.size(1), -1, -1).transpose(0, 1)
    lstm_input = torch.cat([code_expanded, inputs], dim=2)

    lstm_out, _ = decoder.lstm(lstm_input)
    trimmed = lstm_out[:, decoder.base_length :, :]
    return trimmed.squeeze(0).detach().cpu().numpy()


def fit_linear_trends(eps_values: List[float], mean_matrix: np.ndarray) -> pd.DataFrame:
    """Fit y = a*eps + b for each channel of the mean activation matrix."""
    eps = np.asarray(eps_values, dtype=np.float64)
    if mean_matrix.ndim != 2:
        raise ValueError(f"Expected mean_matrix with shape [n_eps, n_units]; got {mean_matrix.shape}")
    n_eps, n_units = mean_matrix.shape

    slopes = np.full(n_units, np.nan, dtype=np.float64)
    intercepts = np.full(n_units, np.nan, dtype=np.float64)
    r2 = np.full(n_units, np.nan, dtype=np.float64)

    if n_eps < 2:
        return pd.DataFrame(
            {
                "layer": "lstm_output",
                "neuron_idx": np.arange(n_units),
                "slope": slopes,
                "intercept": intercepts,
                "r_squared": r2,
                "eps_min": np.nan,
                "eps_max": np.nan,
            }
        )

    finite_neurons = np.all(np.isfinite(mean_matrix), axis=0)
    if not np.any(finite_neurons):
        return pd.DataFrame(
            {
                "layer": "lstm_output",
                "neuron_idx": np.arange(n_units),
                "slope": slopes,
                "intercept": intercepts,
                "r_squared": r2,
                "eps_min": float(np.min(eps)),
                "eps_max": float(np.max(eps)),
            }
        )

    eps_mean = float(eps.mean())
    eps_centered = eps - eps_mean
    denom = float(np.sum(eps_centered**2))
    if denom == 0.0:
        return pd.DataFrame(
            {
                "layer": "lstm_output",
                "neuron_idx": np.arange(n_units),
                "slope": slopes,
                "intercept": intercepts,
                "r_squared": r2,
                "eps_min": float(np.min(eps)),
                "eps_max": float(np.max(eps)),
            }
        )

    m = mean_matrix[:, finite_neurons].astype(np.float64, copy=False)
    m_mean = m.mean(axis=0)
    m_centered = m - m_mean

    slope_vals = np.sum(eps_centered[:, None] * m_centered, axis=0) / denom
    intercept_vals = m_mean - slope_vals * eps_mean
    fitted = slope_vals[None, :] * eps[:, None] + intercept_vals[None, :]
    ss_res = np.sum((m - fitted) ** 2, axis=0)
    ss_tot = np.sum((m - m_mean) ** 2, axis=0)

    r2_vals = np.full(m.shape[1], np.nan, dtype=np.float64)
    nonzero = ss_tot > 0
    r2_vals[nonzero] = 1.0 - ss_res[nonzero] / ss_tot[nonzero]

    slopes[finite_neurons] = slope_vals
    intercepts[finite_neurons] = intercept_vals
    r2[finite_neurons] = r2_vals

    return pd.DataFrame(
        {
            "layer": "lstm_output",
            "neuron_idx": np.arange(n_units),
            "slope": slopes,
            "intercept": intercepts,
            "r_squared": r2,
            "eps_min": float(np.min(eps)),
            "eps_max": float(np.max(eps)),
        }
    )


def _format_stat(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    if abs(value) >= 1e-3:
        return f"{value:.3f}"
    return f"{value:.3g}"


def plot_linearity_histograms(
    trend_df: pd.DataFrame,
    catchment: str,
    output_path: Path,
    bins: int,
) -> None:
    r2_vals = pd.to_numeric(trend_df["r_squared"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    slope_vals = pd.to_numeric(trend_df["slope"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

    if r2_vals.empty or slope_vals.empty:
        raise ValueError("No finite slope/R² values available to plot.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    def _panel(ax: plt.Axes, values: pd.Series, panel_tag: str, xlabel: str) -> None:
        values_np = values.to_numpy(dtype=float)
        median_val = float(np.median(values_np))
        min_val = float(np.min(values_np))
        max_val = float(np.max(values_np))

        ax.hist(values_np, bins=bins, color="#4C72B0", alpha=0.8, edgecolor="black")
        ax.axvline(median_val, color="red", linestyle="--", linewidth=1.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)

        if panel_tag == "(a)":
            ax.set_xlim(min(0.0, min_val), max(1.0, max_val))
        else:
            ax.set_xlim(0.0, max_val if max_val > 0 else 1e-6)

        if panel_tag == "(b)":
            # Keep the annotation block in the top-left for consistency with (a),
            # but shift it right slightly to avoid overlapping the high-count bins
            # near zero.
            text_x = 0.12
            text_ha = "left"
        else:
            text_x = 0.02
            text_ha = "left"

        ax.text(
            text_x,
            0.92,
            panel_tag,
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            ha=text_ha,
        )
        ax.text(
            text_x,
            0.84,
            f"median: {_format_stat(median_val)}",
            transform=ax.transAxes,
            fontsize=11,
            color="red",
            ha=text_ha,
        )
        ax.text(
            text_x,
            0.74,
            f"min: {_format_stat(min_val)}\nmax: {_format_stat(max_val)}",
            transform=ax.transAxes,
            fontsize=11,
            color="black",
            ha=text_ha,
        )

    _panel(axes[0], r2_vals, "(a)", r"$R^2$ of linear model")
    slope_abs = slope_vals.abs()
    _panel(
        axes[1],
        slope_abs,
        "(b)",
        r"Absolute value of slope ($|\Delta|$ output value change per unit $\epsilon$)",
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    eps_values = prepare_epsilons(args.eps_min, args.eps_max, args.eps_step)

    dataset = load_dataset(device)
    try:
        catchment_idx = dataset.catchment_names.index(args.catchment)
    except ValueError as exc:
        raise ValueError(f"Catchment {args.catchment} not found in dataset.") from exc

    embedding, decoder = load_models(device)

    x_orig = dataset.x[catchment_idx].clone().unsqueeze(0).to(device)
    y_full = dataset.y[catchment_idx].clone().to(device)
    target = y_full[decoder.base_length :]
    if target.dim() == 1:
        target = target.unsqueeze(0)

    grad_sign = compute_fgsm_gradient_sign(decoder, embedding, catchment_idx, x_orig, target)

    mean_rows: List[np.ndarray] = []

    for eps in eps_values:
        if eps == 0.0:
            inputs_eps = x_orig.clone()
        else:
            inputs_eps = x_orig + eps * grad_sign
            inputs_eps[:, :, 0].clamp_(min=0.0)
            inputs_eps[:, :, 2].clamp_(min=0.0)

        lstm_out = collect_lstm_outputs(decoder, embedding, catchment_idx, inputs_eps)
        mean_rows.append(np.nanmean(lstm_out, axis=0))

    mean_matrix = np.stack(mean_rows, axis=0)
    trend_df = fit_linear_trends(eps_values, mean_matrix)

    stem = args.output_prefix or args.catchment
    trend_path = DATA_DIR / f"lstm_output_linear_trends_{stem}.csv"
    trend_df.to_csv(trend_path, index=False)

    figure_path = FIGURE_DIR / f"lstm_output_linearity_histograms_{stem}.pdf"
    plot_linearity_histograms(trend_df, args.catchment, figure_path, bins=args.bins)

    print(f"Catchment: {args.catchment}")
    print(f"Epsilons: {eps_values}")
    print(f"Saved per-channel linear trends to {trend_path}")
    print(f"Saved histogram figure to {figure_path}")


if __name__ == "__main__":
    main()
