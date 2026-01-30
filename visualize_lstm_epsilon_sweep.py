#!/usr/bin/env python3
"""
Analyse the sensitivity of the LSTM FGSM attack to different epsilon values
for a single catchment and create diagnostic plots.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import HydroErr as he
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

# Keep explicit references so torch.load can resolve these classes.
_ = LSTM_decoder
_ = TimeDistributed

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
TEST_CSV = DATA_DIR / "data_test_CAMELS_DE1.00.csv"
FIGURE_DIR = DATA_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 365 * 2
TGT_LEN = 365
BASE_LEN = SEQ_LEN - TGT_LEN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catchment", default="DE911520", help="Catchment name to analyse.")
    parser.add_argument("--eps-min", type=float, default=0.05, help="Smallest FGSM epsilon (exclusive of baseline).")
    parser.add_argument("--eps-max", type=float, default=0.5, help="Largest FGSM epsilon.")
    parser.add_argument("--eps-step", type=float, default=0.05, help="Step size for epsilon sweep.")
    parser.add_argument(
        "--hydro-step",
        type=float,
        default=0.1,
        help="Step used when selecting epsilon values for hydrograph comparisons.",
    )
    parser.add_argument(
        "--plot-days",
        type=int,
        default=365,
        help="Number of trailing days to display when no date window is specified.",
    )
    parser.add_argument(
        "--start",
        default="2016-01-01",
        help="Start date (YYYY-MM-DD) for hydrograph plots.",
    )
    parser.add_argument(
        "--end",
        default="2016-12-31",
        help="End date (YYYY-MM-DD) for hydrograph plots.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "mps", "cuda"],
        default="cpu",
        help="Torch device to run the evaluation on.",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return torch.device("mps")
    return torch.device("cpu")


def prepare_epsilons(args: argparse.Namespace) -> Tuple[List[float], List[float]]:
    eps_values = [0.0]
    if args.eps_step <= 0.0:
        raise ValueError("eps-step must be positive.")
    current = args.eps_min
    while current <= args.eps_max + 1e-9:
        eps_values.append(round(float(current), 2))
        current += args.eps_step

    hydro_eps = [0.0]
    current = args.hydro_step
    while current <= args.eps_max + 1e-9:
        hydro_eps.append(round(float(current), 2))
        current += args.hydro_step

    eps_values = sorted(set(round(eps, 2) for eps in eps_values))
    hydro_eps = [eps for eps in hydro_eps if eps in eps_values]
    return eps_values, hydro_eps


def load_dataset(device: torch.device) -> Forcing_Data:
    return Forcing_Data(
        str(TEST_CSV),
        record_length=4018,
        n_feature=3,
        storage_device="cpu",
        seq_length=SEQ_LEN,
        target_seq_length=TGT_LEN,
        base_length=BASE_LEN,
    )


def load_models(device: torch.device) -> Tuple[torch.nn.Module, torch.nn.Module]:
    # Import statements at module level ensure the classes are registered for torch.load.
    embedding = torch.load(DATA_DIR / "embedding_full.pt", map_location=device, weights_only=False)
    decoder = torch.load(DATA_DIR / "decoder_full.pt", map_location=device, weights_only=False)
    embedding.eval()
    decoder.eval()
    for module in (embedding, decoder):
        for param in module.parameters():
            param.requires_grad_(False)
    embedding = embedding.to(device)
    decoder = decoder.to(device)
    return embedding, decoder


def compute_metrics(pred: torch.Tensor, tgt: torch.Tensor) -> Tuple[float, float, np.ndarray]:
    mse_val = float(mse_loss_with_nans(pred, tgt).item())
    pred_np = pred.detach().cpu().numpy().ravel()
    tgt_np = tgt.detach().cpu().numpy().ravel()
    mask = np.isfinite(pred_np) & np.isfinite(tgt_np)
    if not np.any(mask):
        return mse_val, float("nan"), pred_np
    kge_val = float(he.kge_2009(pred_np[mask], tgt_np[mask]))
    return mse_val, kge_val, pred_np


def sweep_epsilons(
    dataset: Forcing_Data,
    embedding: torch.nn.Module,
    decoder: torch.nn.Module,
    idx: int,
    eps_values: List[float],
    device: torch.device,
) -> Tuple[pd.DataFrame, Dict[float, np.ndarray], np.ndarray]:
    x_orig = dataset.x[idx].unsqueeze(0).to(device)
    x_for_grad = x_orig.clone().requires_grad_(True)
    y = dataset.y[idx].clone().to(device)

    tgt = y[BASE_LEN:]
    if tgt.dim() == 1:
        tgt = tgt.unsqueeze(0)

    code = embedding(torch.tensor([idx], device=device))
    base_pred = decoder.decode(code, x_for_grad)
    base_loss = mse_loss_with_nans(base_pred, tgt)
    base_loss.backward()

    grad = torch.nan_to_num(x_for_grad.grad, nan=0.0, posinf=0.0, neginf=0.0)
    grad_sign = grad.sign()

    obs_np = tgt.detach().cpu().numpy().ravel()

    rows: List[Dict[str, float]] = []
    series_by_eps: Dict[float, np.ndarray] = {}

    for eps in eps_values:
        eps_key = float(round(eps, 2))
        if eps_key == 0.0:
            pred = base_pred.detach()
        else:
            adv = x_orig + eps_key * grad_sign
            adv[:, :, 0].clamp_(min=0.0)
            adv[:, :, 2].clamp_(min=0.0)
            with torch.no_grad():
                pred = decoder.decode(code, adv)

        mse_val, kge_val, pred_np = compute_metrics(pred, tgt)
        rows.append({"epsilon": eps_key, "mse": mse_val, "kge": kge_val})
        series_by_eps[eps_key] = pred_np

    results = pd.DataFrame(rows).sort_values("epsilon").reset_index(drop=True)
    base_row = results.loc[results["epsilon"] == 0.0].iloc[0]
    results["mse_delta"] = results["mse"] - base_row["mse"]
    results["kge_delta"] = results["kge"] - base_row["kge"]
    return results, series_by_eps, obs_np


def extract_dates(catchment: str) -> pd.DatetimeIndex:
    df = pd.read_csv(TEST_CSV)
    subset = df[df["catchment_name"] == catchment].copy()
    if subset.empty:
        raise ValueError(f"Catchment {catchment} not found in {TEST_CSV}.")
    dates = pd.to_datetime(subset["Date"])
    target_dates = dates.iloc[BASE_LEN:]
    return pd.DatetimeIndex(target_dates)


def plot_metrics(results: pd.DataFrame, output_base: Path) -> None:
    fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10, 4.5))

    def add_fit(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str) -> float:
        """Fit a simple linear model y = ax + b and draw it, returning R²."""
        mask = np.isfinite(x) & np.isfinite(y)
        x_fit = x[mask]
        y_fit = y[mask]
        if x_fit.size < 2:
            return float("nan")
        order = np.argsort(x_fit)
        x_fit = x_fit[order]
        y_fit = y_fit[order]
        coeffs = np.polyfit(x_fit, y_fit, 1)
        model = np.poly1d(coeffs)
        y_pred = model(x_fit)
        ss_res = float(np.sum((y_fit - y_pred) ** 2))
        ss_tot = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        line_x = np.array([x_fit.min(), x_fit.max()])
        ax.plot(line_x, model(line_x), color=color, linewidth=1.2, linestyle="-", label=f"Linear fit (R²={r_squared:.3f})")
        return r_squared

    x_eps = results["epsilon"].to_numpy()

    axes[0].plot(x_eps, results["kge_delta"], marker="o", color="#1b9e77", linestyle="", label="ΔKGE samples")
    r2_kge = add_fit(axes[0], x_eps, results["kge_delta"].to_numpy(), "#1b9e77")
    axes[0].axhline(0.0, color="0.4", linestyle="--", linewidth=1)
    axes[0].set_ylabel("ΔKGE (after - before)")
    baseline_kge = results.loc[results["epsilon"] == 0.0, "kge"].iloc[0]
    axes[0].text(
        0.02,
        0.5,
        f"Before KGE = {baseline_kge:.3f}\nLinear fit R² = {r2_kge:.3f}",
        transform=axes[0].transAxes,
        ha="left",
        va="center",
        fontsize=9,
        color="#1b9e77",
    )

    axes[0].legend(loc="lower left")
    axes[0].text(
        0.02,
        0.88,
        "(a)",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
    )

    axes[1].plot(x_eps, results["mse_delta"], marker="o", color="#d95f02", linestyle="", label="ΔMSE samples")
    r2_mse = add_fit(axes[1], x_eps, results["mse_delta"].to_numpy(), "#d95f02")
    axes[1].axhline(0.0, color="0.4", linestyle="--", linewidth=1)
    axes[1].set_ylabel("ΔMSE (after - before)")
    baseline_mse = results.loc[results["epsilon"] == 0.0, "mse"].iloc[0]
    axes[1].text(
        0.98,
        0.5,
        f"Before MSE = {baseline_mse:.4f}\nLinear fit R² = {r2_mse:.3f}",
        transform=axes[1].transAxes,
        ha="right",
        va="center",
        fontsize=9,
        color="#d95f02",
    )

    axes[1].set_xlabel("FGSM ε")
    axes[1].legend(loc="lower right")
    axes[1].text(
        0.02,
        0.88,
        "(b)",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()

    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def plot_hydrographs(
    dates: pd.DatetimeIndex,
    obs: np.ndarray,
    series_by_eps: Dict[float, np.ndarray],
    hydro_eps: List[float],
    catchment: str,
    plot_days: int,
    start_date: str | None,
    end_date: str | None,
    output_base: Path,
) -> None:
    date_series = dates.to_series().reset_index(drop=True)
    mask = pd.Series(True, index=date_series.index)
    if start_date is not None:
        start_ts = pd.to_datetime(start_date)
        mask &= date_series >= start_ts
    if end_date is not None:
        end_ts = pd.to_datetime(end_date)
        mask &= date_series <= end_ts

    if mask.sum() == 0:
        raise ValueError("Selected date range does not overlap with available data.")

    if (start_date is None and end_date is None) and plot_days is not None and plot_days > 0 and mask.sum() > plot_days:
        mask.iloc[:-plot_days] = False

    time_axis = date_series[mask].reset_index(drop=True)
    obs_slice = pd.Series(obs).iloc[mask.values].reset_index(drop=True)

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6.5))

    axes[0].plot(time_axis, obs_slice, color="black", linewidth=1.4, label="Observed")

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(hydro_eps)))
    last_series = None
    prev_eps = None
    for eps, color in zip(hydro_eps, colors):
        if eps not in series_by_eps:
            continue
        series_full = pd.Series(series_by_eps[eps])
        series = series_full.iloc[mask.values].reset_index(drop=True)
        axes[0].plot(time_axis, series, color=color, linewidth=1.1, label=f"ε = {eps:.2f}")
        if last_series is not None and prev_eps is not None:
            diff = series - last_series
            axes[1].plot(
                time_axis,
                diff,
                color=color,
                linewidth=1.1,
                label=f"{prev_eps:.2f} → {eps:.2f}",
            )
        prev_eps = eps
        last_series = series

    axes[0].set_ylabel("Discharge [mm/day]")
    axes[0].legend(loc="upper right", ncol=2, frameon=False)
    axes[0].text(
        0.02,
        0.9,
        "(a)",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
    )

    axes[1].axhline(0.0, color="0.4", linestyle="--", linewidth=1)
    axes[1].set_ylabel("ΔDischarge between successive ε [mm/day]")
    axes[1].set_xlabel("Date")
    if len(axes[1].lines) > 0:
        axes[1].legend(loc="upper right", ncol=2, frameon=False)
    axes[1].text(
        0.02,
        0.9,
        "(b)",
        transform=axes[1].transAxes,
        ha="left",
        va="top", 
        fontsize=12,
        fontweight="bold",
    )

    axes[1].tick_params(axis="x", rotation=0, labelrotation=0)
    fig.tight_layout()

    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    eps_values, hydro_eps = prepare_epsilons(args)

    dataset = load_dataset(device)
    try:
        catchment_index = dataset.catchment_names.index(args.catchment)
    except ValueError as exc:
        raise ValueError(f"Catchment {args.catchment} not found in dataset.") from exc

    embedding, decoder = load_models(device)

    results, series_by_eps, obs_np = sweep_epsilons(
        dataset, embedding, decoder, catchment_index, eps_values, device
    )
    results_path = DATA_DIR / f"lstm_fgsm_eps_sweep_{args.catchment}.csv"
    results.to_csv(results_path, index=False)

    dates = extract_dates(args.catchment)

    metrics_stem = FIGURE_DIR / f"lstm_fgsm_eps_metrics_{args.catchment}"
    hydro_stem = FIGURE_DIR / f"lstm_fgsm_eps_hydro_{args.catchment}"

    plot_metrics(results, metrics_stem)
    plot_hydrographs(
        dates,
        obs_np,
        series_by_eps,
        hydro_eps,
        args.catchment,
        args.plot_days,
        args.start,
        args.end,
        hydro_stem,
    )

    print(f"Saved metrics sweep to {results_path}")

if __name__ == "__main__":
    main()
