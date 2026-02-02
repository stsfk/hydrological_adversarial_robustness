#!/usr/bin/env python3
"""
Analyse the sensitivity of an HBV FGSM attack to different epsilon values for a
single catchment and create diagnostic plots.

The FGSM perturbation is applied to the meteorological forcing inputs
([P, PET, T]) and precipitation (P) + potential evapotranspiration (PET) are
clamped to remain non-negative after perturbation.

Metrics (MSE, KGE) are computed on the routed discharge after a warmup period.
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

import hbv
from lstm_train_final_model import mse_loss_with_nans


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
TEST_CSV = DATA_DIR / "data_test_CAMELS_DE1.00.csv"
PARAMETERS_CSV = DATA_DIR / "CAMELS_DE_parameters_hbv.csv"
SELECTED_CSV = DATA_DIR / "selected_catchments.csv"

FIGURE_DIR = DATA_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

WARMUP_DAYS = 365
PARAMETER_NAMES = [
    "BETA",
    "FC",
    "K0",
    "K1",
    "K2",
    "LP",
    "PERC",
    "UZL",
    "TT",
    "CFMAX",
    "CFR",
    "CWH",
]


# ---------------------------------------------------------------------------
# CLI
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
    if args.eps_step <= 0.0:
        raise ValueError("eps-step must be positive.")

    eps_values = [0.0]
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


# ---------------------------------------------------------------------------
# Data + model helpers
# ---------------------------------------------------------------------------
def load_parameter_table() -> pd.DataFrame:
    selected = pd.read_csv(SELECTED_CSV)
    params = pd.read_csv(PARAMETERS_CSV)
    merged = selected.merge(params, left_on="catchment_name", right_on="gauge_id")
    return merged.set_index("catchment_name")


def build_parameter_tensors(row: pd.Series, seq_len: int, device: torch.device) -> Dict[str, torch.Tensor]:
    values = torch.tensor([float(row[name]) for name in PARAMETER_NAMES], dtype=torch.float32, device=device)
    params: Dict[str, torch.Tensor] = {}
    for idx, name in enumerate(PARAMETER_NAMES):
        params[name] = values[idx].view(1, 1, 1).expand(1, seq_len, 1)
    return params


def load_catchment_timeseries(catchment: str) -> pd.DataFrame:
    df = pd.read_csv(TEST_CSV)
    subset = df[df["catchment_name"] == catchment].copy()
    if subset.empty:
        raise ValueError(f"Catchment {catchment} not found in {TEST_CSV}.")
    required = {"Date", "P", "T", "PET", "Q"}
    missing = sorted(required - set(subset.columns))
    if missing:
        raise ValueError(f"Missing columns for catchment {catchment}: {missing}")
    subset["Date"] = pd.to_datetime(subset["Date"])
    return subset.reset_index(drop=True)


def load_models(device: torch.device) -> Tuple[hbv.HBV, hbv.UH_routing]:
    model = hbv.HBV(n_models=1).to(device)
    routing = hbv.UH_routing(n_models=1).to(device)
    model.eval()
    routing.eval()
    for module in (model, routing):
        for param in module.parameters():
            param.requires_grad_(False)
    return model, routing


def routing_params(alpha: float, beta: float, device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "alpha": torch.tensor([[[alpha]]], dtype=torch.float32, device=device),
        "beta": torch.tensor([[[beta]]], dtype=torch.float32, device=device),
    }


def evaluate_routed_trimmed(
    model: hbv.HBV,
    routing: hbv.UH_routing,
    forcings: torch.Tensor,
    parameters: Dict[str, torch.Tensor],
    obs: torch.Tensor,
    alpha: float,
    beta: float,
) -> Tuple[Dict[str, float], np.ndarray]:
    with torch.no_grad():
        discharge = model(forcings, parameters)["y_hat"].squeeze(0).squeeze(-1)
        routed = routing(discharge.unsqueeze(0).unsqueeze(-1), routing_params(alpha, beta, forcings.device))
        routed_full = routed.squeeze(0).squeeze(-1)

    routed_trim = routed_full[WARMUP_DAYS:]
    obs_trim = obs[WARMUP_DAYS:]

    valid = torch.isfinite(routed_trim) & torch.isfinite(obs_trim)
    valid &= ~torch.isnan(obs_trim)

    metrics: Dict[str, float] = {"mse": float("nan"), "kge": float("nan")}
    if valid.any():
        routed_valid = routed_trim[valid].detach().cpu().numpy()
        obs_valid = obs_trim[valid].detach().cpu().numpy()
        metrics["mse"] = float(he.mse(routed_valid, obs_valid))
        metrics["kge"] = float(he.kge_2009(routed_valid, obs_valid))

    return metrics, routed_trim.detach().cpu().numpy()


def fgsm_grad_sign(
    model: hbv.HBV,
    routing: hbv.UH_routing,
    forcings_for_grad: torch.Tensor,
    parameters: Dict[str, torch.Tensor],
    obs: torch.Tensor,
    alpha: float,
    beta: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    forcings_for_grad.requires_grad_(True)
    discharge = model(forcings_for_grad, parameters)["y_hat"].squeeze(0).squeeze(-1)
    routed_full = routing(
        discharge.unsqueeze(0).unsqueeze(-1),
        routing_params(alpha, beta, forcings_for_grad.device),
    ).squeeze(0).squeeze(-1)
    loss = mse_loss_with_nans(routed_full[WARMUP_DAYS:], obs[WARMUP_DAYS:])
    loss.backward()

    grad = torch.nan_to_num(forcings_for_grad.grad, nan=0.0, posinf=0.0, neginf=0.0)
    return grad.sign().detach(), routed_full.detach()


def sweep_epsilons(
    model: hbv.HBV,
    routing: hbv.UH_routing,
    forcings: torch.Tensor,
    parameters: Dict[str, torch.Tensor],
    obs: torch.Tensor,
    alpha: float,
    beta: float,
    eps_values: List[float],
) -> Tuple[pd.DataFrame, Dict[float, np.ndarray], np.ndarray]:
    grad_sign, routed_baseline_full = fgsm_grad_sign(
        model, routing, forcings.clone(), parameters, obs, alpha, beta
    )

    obs_trim = obs[WARMUP_DAYS:].detach().cpu().numpy()
    series_by_eps: Dict[float, np.ndarray] = {0.0: routed_baseline_full[WARMUP_DAYS:].cpu().numpy()}

    base_metrics, _ = evaluate_routed_trimmed(
        model, routing, forcings.detach(), parameters, obs, alpha, beta
    )

    rows: List[Dict[str, float]] = [
        {"epsilon": 0.0, "mse": float(base_metrics["mse"]), "kge": float(base_metrics["kge"])}
    ]

    for eps in eps_values:
        eps_key = float(round(eps, 2))
        if eps_key == 0.0:
            continue
        adv = (forcings + eps_key * grad_sign).clone()
        adv[:, :, 0].clamp_(min=0.0)
        adv[:, :, 1].clamp_(min=0.0)

        metrics, routed_trim = evaluate_routed_trimmed(
            model, routing, adv.detach(), parameters, obs, alpha, beta
        )
        rows.append({"epsilon": eps_key, "mse": float(metrics["mse"]), "kge": float(metrics["kge"])})
        series_by_eps[eps_key] = routed_trim

    results = pd.DataFrame(rows).sort_values("epsilon").reset_index(drop=True)
    base_row = results.loc[np.isclose(results["epsilon"], 0.0)].iloc[0]
    results["mse_delta"] = results["mse"] - base_row["mse"]
    results["kge_delta"] = results["kge"] - base_row["kge"]
    return results, series_by_eps, obs_trim


def extract_dates(catchment_df: pd.DataFrame) -> pd.DatetimeIndex:
    dates = pd.to_datetime(catchment_df["Date"]).iloc[WARMUP_DAYS:]
    return pd.DatetimeIndex(dates)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_metrics(results: pd.DataFrame, output_base: Path) -> None:
    fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10, 4.5))

    def add_fit(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str) -> float:
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
        ax.plot(
            line_x,
            model(line_x),
            color=color,
            linewidth=1.2,
            linestyle="-",
            label=f"Linear fit (R²={r_squared:.3f})",
        )
        return r_squared

    x_eps = results["epsilon"].to_numpy()

    axes[0].plot(x_eps, results["kge_delta"], marker="o", color="#1b9e77", linestyle="", label="ΔKGE samples")
    r2_kge = add_fit(axes[0], x_eps, results["kge_delta"].to_numpy(), "#1b9e77")
    axes[0].axhline(0.0, color="0.4", linestyle="--", linewidth=1)
    axes[0].set_ylabel("ΔKGE (after - before)")
    baseline_kge = results.loc[np.isclose(results["epsilon"], 0.0), "kge"].iloc[0]
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
    baseline_mse = results.loc[np.isclose(results["epsilon"], 0.0), "mse"].iloc[0]
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
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_hydrographs(
    dates: pd.DatetimeIndex,
    obs: np.ndarray,
    series_by_eps: Dict[float, np.ndarray],
    hydro_eps: List[float],
    plot_days: int,
    start_date: str | None,
    end_date: str | None,
    output_base: Path,
) -> None:
    date_series = dates.to_series().reset_index(drop=True)
    mask = pd.Series(True, index=date_series.index)
    if start_date is not None:
        mask &= date_series >= pd.to_datetime(start_date)
    if end_date is not None:
        mask &= date_series <= pd.to_datetime(end_date)

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
            axes[1].plot(time_axis, diff, color=color, linewidth=1.1, label=f"{prev_eps:.2f} → {eps:.2f}")
        prev_eps = eps
        last_series = series

    axes[0].set_ylabel("Routed discharge [mm/day]")
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
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    eps_values, hydro_eps = prepare_epsilons(args)

    parameter_table = load_parameter_table()
    if args.catchment not in parameter_table.index:
        raise ValueError(f"Catchment {args.catchment} not found in HBV parameter table.")
    param_row = parameter_table.loc[args.catchment]

    catchment_df = load_catchment_timeseries(args.catchment)
    seq_len = len(catchment_df)

    forcings = torch.stack(
        [
            torch.tensor(catchment_df["P"].astype(float).to_numpy(), dtype=torch.float32),
            torch.tensor(catchment_df["PET"].astype(float).to_numpy(), dtype=torch.float32),
            torch.tensor(catchment_df["T"].astype(float).to_numpy(), dtype=torch.float32),
        ],
        dim=1,
    ).unsqueeze(0).to(device)

    obs_t = torch.tensor(catchment_df["Q"].astype(float).to_numpy(), dtype=torch.float32, device=device)
    parameters = build_parameter_tensors(param_row, seq_len, device)

    model, routing = load_models(device)

    results, series_by_eps, obs_trim = sweep_epsilons(
        model,
        routing,
        forcings,
        parameters,
        obs_t,
        float(param_row["alpha"]),
        float(param_row["beta"]),
        eps_values,
    )

    results_path = DATA_DIR / f"hbv_fgsm_eps_sweep_{args.catchment}.csv"
    results.to_csv(results_path, index=False)

    dates = extract_dates(catchment_df)

    metrics_stem = FIGURE_DIR / f"hbv_fgsm_eps_metrics_{args.catchment}"
    hydro_stem = FIGURE_DIR / f"hbv_fgsm_eps_hydro_{args.catchment}"

    plot_metrics(results, metrics_stem)
    plot_hydrographs(
        dates,
        obs_trim,
        series_by_eps,
        hydro_eps,
        args.plot_days,
        args.start,
        args.end,
        hydro_stem,
    )

    print(f"Saved metrics sweep to {results_path}")


if __name__ == "__main__":
    main()

