#!/usr/bin/env python3
"""
Compute FGSM attack effectiveness metrics for the LSTM model across all catchments.

For each catchment we sweep epsilon values from eps-min to eps-max and record how the
attack changes KGE and MSE. We then fit linear models of the form:

    delta_performance = a * epsilon + b

where deltas are defined as (attacked − baseline).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import HydroErr as he
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

SEQ_LEN = 365 * 2
TGT_LEN = 365
BASE_LEN = SEQ_LEN - TGT_LEN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eps-min", type=float, default=0.05, help="Smallest FGSM epsilon (exclusive of baseline).")
    parser.add_argument("--eps-max", type=float, default=0.5, help="Largest FGSM epsilon to evaluate.")
    parser.add_argument("--eps-step", type=float, default=0.05, help="Step size for epsilon sweep.")
    parser.add_argument(
        "--device",
        choices=["cpu", "mps", "cuda"],
        default="cpu",
        help="Torch device to run the evaluation on.",
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
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
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


def compute_metrics(pred: torch.Tensor, tgt: torch.Tensor) -> Tuple[float, float]:
    mse_val = float(mse_loss_with_nans(pred, tgt).item())
    pred_np = pred.detach().cpu().numpy().ravel()
    tgt_np = tgt.detach().cpu().numpy().ravel()
    mask = np.isfinite(pred_np) & np.isfinite(tgt_np)
    if not np.any(mask):
        return mse_val, float("nan")
    kge_val = float(he.kge_2009(pred_np[mask], tgt_np[mask]))
    return mse_val, kge_val


def evaluate_catchment(
    dataset: Forcing_Data,
    embedding: torch.nn.Module,
    decoder: torch.nn.Module,
    idx: int,
    eps_values: Iterable[float],
    device: torch.device,
) -> pd.DataFrame:
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

    rows = []
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

        mse_val, kge_val = compute_metrics(pred, tgt)
        rows.append({"epsilon": eps_key, "mse": mse_val, "kge": kge_val})

    results = pd.DataFrame(rows).sort_values("epsilon").reset_index(drop=True)
    base_row = results.loc[results["epsilon"] == 0.0].iloc[0]
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

    dataset = load_dataset(device)
    embedding, decoder = load_models(device)

    per_epsilon_rows: List[dict] = []
    summary_rows = []

    indices = list(range(dataset.n_catchment))
    if args.catchment_limit is not None:
        if args.catchment_limit <= 0:
            raise ValueError("catchment-limit must be positive when provided.")
        indices = indices[: args.catchment_limit]

    n_catchments = len(indices)
    for position, idx in enumerate(indices):
        catchment = dataset.catchment_names[idx]
        results = evaluate_catchment(dataset, embedding, decoder, idx, eps_values, device)

        for _, row in results.iterrows():
            per_epsilon_rows.append(
                {
                    "catchment": catchment,
                    "epsilon": float(row["epsilon"]),
                    "mse": float(row["mse"]),
                    "kge": float(row["kge"]),
                    "mse_delta": float(row["mse_delta"]),
                    "kge_delta": float(row["kge_delta"]),
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

        if (position + 1) % 25 == 0 or position + 1 == n_catchments:
            print(f"Processed {position + 1}/{n_catchments} catchments.")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    per_epsilon_df = pd.DataFrame(per_epsilon_rows)
    summary_df = pd.DataFrame(summary_rows)

    detail_csv = output_dir / "lstm_fgsm_effectiveness_per_epsilon.csv"
    summary_csv = output_dir / "lstm_fgsm_effectiveness_summary.csv"

    per_epsilon_df.to_csv(detail_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    print(f"Wrote per-epsilon metrics to {detail_csv}")
    print(f"Wrote per-catchment summary to {summary_csv}")


if __name__ == "__main__":
    main()
