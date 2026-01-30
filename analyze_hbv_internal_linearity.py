#!/usr/bin/env python3
"""
Analyse HBV internal state response to FGSM attacks across epsilon values.

For a selected catchment, compute a single FGSM gradient direction at
baseline, sweep epsilon values, run HBV to obtain internal states, and
quantify linearity (slope and R^2) of state means vs epsilon.

Outputs
  - data/hbv_internal_response_<CATCHMENT>.csv (per-epsilon state means)
  - data/hbv_internal_linearity_<CATCHMENT>.csv (per-state slope/R^2)
  - data/figures/hbv_internal_response_grid_<CATCHMENT>.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import hbv
from lstm_train_final_model import mse_loss_with_nans

DATA_DIR = Path("data")
FIG_DIR = DATA_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

WARMUP_DAYS = 365


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catchment", default="DE911520", help="Catchment to analyse")
    p.add_argument(
        "--epsilons",
        type=float,
        nargs="*",
        default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        help="Epsilon values for the sweep",
    )
    p.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    return p.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_inputs_and_params(catchment: str) -> Tuple[pd.Series, pd.DataFrame]:
    params = pd.read_csv(DATA_DIR / "CAMELS_DE_parameters_hbv.csv")
    selected = pd.read_csv(DATA_DIR / "selected_catchments.csv")
    merged = selected.merge(params, left_on="catchment_name", right_on="gauge_id")
    row = merged.loc[merged["catchment_name"] == catchment]
    if row.empty:
        raise SystemExit(f"Catchment {catchment} not found in parameters list")
    row = row.iloc[0]
    forc = pd.read_csv(DATA_DIR / "data_test_CAMELS_DE1.00.csv")
    forc = forc[forc["catchment_name"] == catchment].copy()
    if forc.empty:
        raise SystemExit(f"Forcings for {catchment} not found in test CSV")
    return row, forc


def build_parameter_tensors(row: pd.Series, seq_len: int, device: torch.device) -> Dict[str, torch.Tensor]:
    names = [
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
    values = torch.tensor([float(row[n]) for n in names], dtype=torch.float32, device=device)
    tensors = {}
    for i, n in enumerate(names):
        tensors[n] = values[i].view(1, 1, 1).expand(1, seq_len, 1)
    return tensors


def compute_fgsm_grad_sign(
    model: hbv.HBV,
    routing: hbv.UH_routing,
    inputs: torch.Tensor,
    params: Dict[str, torch.Tensor],
    obs: torch.Tensor,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    x = inputs.clone().requires_grad_(True)
    discharge = model(x, params)["y_hat"].squeeze(0).squeeze(-1)
    routed = routing(
        discharge.unsqueeze(0).unsqueeze(-1),
        {
            "alpha": torch.tensor([[[alpha]]], dtype=torch.float32, device=x.device),
            "beta": torch.tensor([[[beta]]], dtype=torch.float32, device=x.device),
        },
    ).squeeze(0).squeeze(-1)
    loss = mse_loss_with_nans(routed[WARMUP_DAYS:], obs[WARMUP_DAYS:])
    loss.backward()
    grad = torch.nan_to_num(x.grad, nan=0.0, posinf=0.0, neginf=0.0)
    return grad.sign().detach()


@torch.no_grad()
def run_hbv(
    model: hbv.HBV,
    inputs: torch.Tensor,
    params: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    out = model(inputs, params)
    return out


def fit_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Return slope, intercept, R^2 for y ~ a x + b (finite values only)."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan"), float("nan"), float("nan")
    coeffs = np.polyfit(x[mask], y[mask], 1)
    a, b = float(coeffs[0]), float(coeffs[1])
    yhat = np.poly1d(coeffs)(x[mask])
    ss_res = float(np.sum((y[mask] - yhat) ** 2))
    ss_tot = float(np.sum((y[mask] - np.mean(y[mask])) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return a, b, r2


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    row, forc = load_inputs_and_params(args.catchment)
    P = torch.tensor(forc["P"].to_numpy(), dtype=torch.float32)
    T = torch.tensor(forc["T"].to_numpy(), dtype=torch.float32)
    PET = torch.tensor(forc["PET"].to_numpy(), dtype=torch.float32)
    Q = torch.tensor(forc["Q"].to_numpy(), dtype=torch.float32)

    inputs = torch.stack([P, PET, T], dim=1).unsqueeze(0).to(device)
    obs = Q.to(device)
    seq_len = inputs.shape[1]

    model = hbv.HBV(n_models=1).to(device)
    model.eval()
    routing = hbv.UH_routing(n_models=1).to(device)
    routing.eval()
    params = build_parameter_tensors(row, seq_len, device)

    # Compute single FGSM gradient direction at baseline (ε=0 reference)
    grad_sign = compute_fgsm_grad_sign(
        model, routing, inputs, params, obs, float(row["alpha"]), float(row["beta"])
    )

    # Sweep epsilons: compute internal state means after warmup
    state_names = ["SNOWPACK", "MELTWATER", "SM", "SUZ", "SLZ"]
    records: List[Dict[str, float]] = []

    for eps in args.epsilons:
        eps = float(eps)
        if eps == 0.0:
            adv = inputs.clone()
        else:
            adv = inputs + eps * grad_sign
            adv[:, :, 0].clamp_(min=0.0)
            adv[:, :, 1].clamp_(min=0.0)

        out = run_hbv(model, adv, params)
        states = out["internal_states"]  # dict of [1, T, 1]
        rec = {"epsilon": eps}
        for name in state_names:
            s = states[name].squeeze(0).squeeze(-1)
            mean_val = float(s[WARMUP_DAYS:].mean().detach().cpu())
            rec[f"{name}_mean"] = mean_val
        records.append(rec)

    df = pd.DataFrame(records).sort_values("epsilon").reset_index(drop=True)
    resp_path = DATA_DIR / f"hbv_internal_response_{args.catchment}.csv"
    df.to_csv(resp_path, index=False)

    # Fit linear models mean(state) ~ epsilon
    linear_rows = []
    x = df["epsilon"].to_numpy()
    for name in state_names:
        y = df[f"{name}_mean"].to_numpy()
        a, b, r2 = fit_line(x, y)
        linear_rows.append({"state": name, "slope": a, "intercept": b, "r2": r2})
    linear_df = pd.DataFrame(linear_rows)
    lin_path = DATA_DIR / f"hbv_internal_linearity_{args.catchment}.csv"
    linear_df.to_csv(lin_path, index=False)

    print(f"Saved per-epsilon state means to {resp_path}")
    print(f"Saved per-state linearity to {lin_path}")

    # Figure: one subplot per internal state (no log scale), with fit and metrics
    import matplotlib.gridspec as gridspec
    nrows, ncols = 2, 3
    gs_fig = plt.figure(figsize=(12.0, 6.8))
    gs = gridspec.GridSpec(nrows, ncols, figure=gs_fig, wspace=0.28, hspace=0.32)

    # Order states into 3 + 2 layout
    ordered_states = ["SNOWPACK", "MELTWATER", "SM", "SUZ", "SLZ"]
    axes_grid: List[plt.Axes] = []
    for idx in range(nrows * ncols):
        axes_grid.append(gs_fig.add_subplot(gs[idx // ncols, idx % ncols]))

    for i, name in enumerate(ordered_states):
        ax = axes_grid[i]
        y = df[f"{name}_mean"].to_numpy(dtype=float)
        # Plot raw points (no connecting lines)
        ax.scatter(
            df["epsilon"], y, s=28, color="#4C78A8", edgecolors="white", linewidths=0.3, zorder=3
        )
        # Show the fitted linear model (dashed line)
        a = float(linear_df.loc[linear_df["state"] == name, "slope"].iloc[0])
        b = float(linear_df.loc[linear_df["state"] == name, "intercept"].iloc[0])
        r2 = float(linear_df.loc[linear_df["state"] == name, "r2"].iloc[0])
        xx = np.linspace(df["epsilon"].min(), df["epsilon"].max(), 100)
        ax.plot(xx, a * xx + b, color="#E45756", linestyle="--", linewidth=1.6)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("FGSM ε")
        ax.set_ylabel("Mean value [mm]")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
        # Metrics annotation (middle-left to avoid blocking labels)
        ax.text(
            0.02,
            0.55,
            f"slope: {a:.3g}\n$R^2$: {r2:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
        # subplot labels removed per request

        # Rotate y-axis tick labels for MELTWATER to avoid clutter with many zeros
        if name.upper() == "MELTWATER":
            for tick in ax.get_yticklabels():
                tick.set_rotation(45)
                tick.set_ha("right")

    # Hide the unused 6th subplot
    axes_grid[-1].axis("off")

    grid_path = FIG_DIR / f"hbv_internal_response_grid_{args.catchment}"
    gs_fig.savefig(grid_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(gs_fig)
    print(f"Saved grid figure to {grid_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
