#!/usr/bin/env python3
"""
Compute FGSM attack effectiveness metrics for the HBV model across selected catchments.

For every catchment we sweep epsilon values from eps-min to eps-max, quantify how the
attack changes KGE and MSE, and fit linear models of the form:

    delta_performance = a * epsilon + b

where deltas are defined as (attacked − baseline).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import HydroErr as he
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
    parser.add_argument("--eps-min", type=float, default=0.05, help="Minimum epsilon to evaluate (exclusive of baseline).")
    parser.add_argument("--eps-max", type=float, default=0.5, help="Maximum epsilon to evaluate.")
    parser.add_argument("--eps-step", type=float, default=0.05, help="Step size for epsilon sweep.")
    parser.add_argument(
        "--device",
        choices=["cpu", "mps", "cuda"],
        default="cpu",
        help="Torch device to execute the sweep on.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory where result CSV files will be written.",
    )
    parser.add_argument(
        "--catchment-limit",
        type=int,
        default=None,
        help="Optional cap on the number of catchments to process (useful for quick tests).",
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


def prepare_epsilons(eps_min: float, eps_max: float, eps_step: float) -> List[float]:
    if eps_step <= 0.0:
        raise ValueError("eps-step must be positive.")
    eps_values = [0.0]
    current = eps_min
    while current <= eps_max + 1e-9:
        eps_values.append(round(float(current), 2))
        current += eps_step
    eps_values = sorted(set(round(eps, 2) for eps in eps_values))
    return eps_values


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


def load_models(device: torch.device) -> Tuple[hbv.HBV, hbv.UH_routing]:
    model = hbv.HBV(n_models=1).to(device)
    routing = hbv.UH_routing(n_models=1).to(device)
    model.eval()
    routing.eval()
    for module in (model, routing):
        for param in module.parameters():
            param.requires_grad_(False)
    return model, routing


def evaluate_routed(
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
        routed = routing(
            discharge.unsqueeze(0).unsqueeze(-1),
            {
                "alpha": torch.tensor([[[alpha]]], dtype=torch.float32, device=forcings.device),
                "beta": torch.tensor([[[beta]]], dtype=torch.float32, device=forcings.device),
            },
        )

    routed_eval = routed.squeeze(0).squeeze(-1)
    routed_trim = routed_eval[WARMUP_DAYS:]
    obs_trim = obs[WARMUP_DAYS:]

    valid_mask = torch.isfinite(routed_trim) & torch.isfinite(obs_trim)
    valid_mask &= ~torch.isnan(obs_trim)

    metrics: Dict[str, float] = {}
    if valid_mask.any():
        routed_valid = routed_trim[valid_mask].detach().cpu().numpy()
        obs_valid = obs_trim[valid_mask].detach().cpu().numpy()
        metrics["mse"] = float(he.mse(routed_valid, obs_valid))
        metrics["kge"] = float(he.kge_2009(routed_valid, obs_valid))
    else:
        metrics["mse"] = float("nan")
        metrics["kge"] = float("nan")

    return metrics, routed_trim.detach().cpu().numpy()


def fgsm_gradient(
    model: hbv.HBV,
    routing: hbv.UH_routing,
    forcings: torch.Tensor,
    parameters: Dict[str, torch.Tensor],
    obs: torch.Tensor,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    forcings.requires_grad_(True)
    discharge = model(forcings, parameters)["y_hat"].squeeze(0).squeeze(-1)
    routed_full = routing(
        discharge.unsqueeze(0).unsqueeze(-1),
        {
            "alpha": torch.tensor([[[alpha]]], dtype=torch.float32, device=forcings.device),
            "beta": torch.tensor([[[beta]]], dtype=torch.float32, device=forcings.device),
        },
    ).squeeze(0).squeeze(-1)
    loss = mse_loss_with_nans(routed_full[WARMUP_DAYS:], obs[WARMUP_DAYS:])
    loss.backward()
    grad = torch.nan_to_num(forcings.grad, nan=0.0, posinf=0.0, neginf=0.0)
    return grad.sign().detach()


def evaluate_catchment(
    catchment: str,
    catchment_df: pd.DataFrame,
    parameter_row: pd.Series,
    model: hbv.HBV,
    routing: hbv.UH_routing,
    eps_values: Iterable[float],
    device: torch.device,
) -> pd.DataFrame:
    P = catchment_df["P"].astype(float).to_numpy()
    T = catchment_df["T"].astype(float).to_numpy()
    PET = catchment_df["PET"].astype(float).to_numpy()
    Q_obs = catchment_df["Q"].astype(float).to_numpy()

    inputs = torch.stack(
        [
            torch.tensor(P, dtype=torch.float32),
            torch.tensor(PET, dtype=torch.float32),
            torch.tensor(T, dtype=torch.float32),
        ],
        dim=1,
    ).unsqueeze(0).to(device)
    obs_t = torch.tensor(Q_obs, dtype=torch.float32, device=device)

    parameters = build_parameter_tensors(parameter_row, inputs.shape[1], device)

    grad_sign = fgsm_gradient(
        model,
        routing,
        inputs.clone(),
        parameters,
        obs_t,
        float(parameter_row["alpha"]),
        float(parameter_row["beta"]),
    )

    rows = []
    for eps in eps_values:
        eps_key = float(round(eps, 2))
        if eps_key == 0.0:
            forcings_eps = inputs
        else:
            adv = (inputs + eps_key * grad_sign).clone()
            adv[:, :, 0].clamp_(min=0.0)
            adv[:, :, 1].clamp_(min=0.0)
            forcings_eps = adv

        metrics, _ = evaluate_routed(
            model,
            routing,
            forcings_eps.detach(),
            parameters,
            obs_t,
            float(parameter_row["alpha"]),
            float(parameter_row["beta"]),
        )

        rows.append({"epsilon": eps_key, "mse": metrics["mse"], "kge": metrics["kge"]})

    results = pd.DataFrame(rows).sort_values("epsilon").reset_index(drop=True)
    base_row = results.loc[np.isclose(results["epsilon"], 0.0)].iloc[0]
    results["mse_delta"] = results["mse"] - base_row["mse"]
    results["kge_delta"] = results["kge"] - base_row["kge"]

    return results


def fit_linear_model(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x_fit = x[mask]
    y_fit = y[mask]
    if x_fit.size < 2:
        return float("nan"), float("nan"), float("nan")
    order = np.argsort(x_fit)
    x_fit = x_fit[order]
    y_fit = y_fit[order]
    coeffs = np.polyfit(x_fit, y_fit, 1)
    model = np.poly1d(coeffs)
    y_pred = model(x_fit)
    ss_res = float(np.sum((y_fit - y_pred) ** 2))
    ss_tot = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(coeffs[0]), float(coeffs[1]), r_squared


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    eps_values = prepare_epsilons(args.eps_min, args.eps_max, args.eps_step)
    eps_max_key = float(round(args.eps_max, 2))

    parameters_table = load_parameter_table()
    forcings_df = pd.read_csv(TEST_CSV)
    grouped = forcings_df.groupby("catchment_name")

    available_catchments = [name for name in parameters_table.index if name in grouped.groups]
    if args.catchment_limit is not None:
        if args.catchment_limit <= 0:
            raise ValueError("catchment-limit must be positive when provided.")
        available_catchments = available_catchments[: args.catchment_limit]

    model, routing = load_models(device)

    per_epsilon_rows: List[dict] = []
    summary_rows: List[dict] = []

    total = len(available_catchments)
    for position, catchment in enumerate(available_catchments):
        catchment_df = grouped.get_group(catchment).copy()
        row = parameters_table.loc[catchment]
        results = evaluate_catchment(catchment, catchment_df, row, model, routing, eps_values, device)

        for _, metric_row in results.iterrows():
            per_epsilon_rows.append(
                {
                    "catchment": catchment,
                    "epsilon": float(metric_row["epsilon"]),
                    "mse": float(metric_row["mse"]),
                    "kge": float(metric_row["kge"]),
                    "mse_delta": float(metric_row["mse_delta"]),
                    "kge_delta": float(metric_row["kge_delta"]),
                }
            )

        baseline = results.loc[np.isclose(results["epsilon"], 0.0)].iloc[0]
        max_eps_row = results.loc[np.isclose(results["epsilon"], eps_max_key)].iloc[0]

        fit_mask = (results["epsilon"] >= args.eps_min - 1e-9) & (results["epsilon"] > 0.0)
        x_vals = results.loc[fit_mask, "epsilon"].to_numpy()
        kge_changes = results.loc[fit_mask, "kge_delta"].to_numpy()
        mse_changes = results.loc[fit_mask, "mse_delta"].to_numpy()

        kge_slope, kge_intercept, kge_r2 = fit_linear_model(x_vals, kge_changes)
        mse_slope, mse_intercept, mse_r2 = fit_linear_model(x_vals, mse_changes)

        kge_fit_points = int(np.sum(np.isfinite(x_vals) & np.isfinite(kge_changes)))
        mse_fit_points = int(np.sum(np.isfinite(x_vals) & np.isfinite(mse_changes)))

        summary_rows.append(
            {
                "catchment": catchment,
                "baseline_mse": float(baseline["mse"]),
                "baseline_kge": float(baseline["kge"]),
                "max_epsilon": args.eps_max,
                "mse_delta_at_max_eps": float(max_eps_row["mse_delta"]),
                "kge_delta_at_max_eps": float(max_eps_row["kge_delta"]),
                "kge_linear_slope": kge_slope,
                "kge_linear_intercept": kge_intercept,
                "kge_linear_r2": kge_r2,
                "kge_fit_points": kge_fit_points,
                "mse_linear_slope": mse_slope,
                "mse_linear_intercept": mse_intercept,
                "mse_linear_r2": mse_r2,
                "mse_fit_points": mse_fit_points,
            }
        )

        if (position + 1) % 10 == 0 or position + 1 == total:
            print(f"Processed {position + 1}/{total} catchments.")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    per_epsilon_df = pd.DataFrame(per_epsilon_rows)
    summary_df = pd.DataFrame(summary_rows)

    detail_csv = output_dir / "hbv_fgsm_effectiveness_per_epsilon.csv"
    summary_csv = output_dir / "hbv_fgsm_effectiveness_summary.csv"

    per_epsilon_df.to_csv(detail_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    print(f"Wrote per-epsilon metrics to {detail_csv}")
    print(f"Wrote per-catchment summary to {summary_csv}")


if __name__ == "__main__":
    main()
