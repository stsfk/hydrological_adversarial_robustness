#!/usr/bin/env python3
"""
Evaluate epsilon-linearity of discharge along the FGSM direction for LSTM and HBV.

For a single catchment and model (LSTM or HBV):
  1) Compute the FGSM gradient sign once at epsilon=0.
  2) Sweep user-specified epsilon values.
  3) For each time step, fit a linear model y_t ~ a_t * eps + b_t.
  4) Summarise per-catchment linearity via median/mean R^2 and the fraction of
     time steps with R^2 >= 0.95, plus median/mean |slope|.

Outputs
-------
  data/fgsm_linearity_<model>_<catchment>.csv  # summary stats
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

import hbv
from lstm_train_final_model import (
    Forcing_Data,
    LSTM_decoder,
    TimeDistributed,
    mse_loss_with_nans,
)

# Explicit references for torch.load safety
_ = LSTM_decoder
_ = TimeDistributed

DATA_DIR = Path("data")
TEST_CSV = DATA_DIR / "data_test_CAMELS_DE1.00.csv"
PARAMETERS_CSV = DATA_DIR / "CAMELS_DE_parameters_hbv.csv"
SELECTED_CSV = DATA_DIR / "selected_catchments.csv"

SEQ_LEN = 365 * 2
TGT_LEN = 365
BASE_LEN = SEQ_LEN - TGT_LEN
WARMUP_DAYS = 365


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["lstm", "hbv"], required=True, help="Model to evaluate.")
    p.add_argument("--catchment", default="DE911520", help="Catchment ID to analyse.")
    p.add_argument(
        "--epsilons",
        type=float,
        nargs="*",
        default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        help="Epsilon values to sweep (include 0.0 for baseline).",
    )
    p.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu", help="Torch device.")
    return p.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fit_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
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


# ------------------------------ LSTM ---------------------------------
def load_lstm_dataset() -> Forcing_Data:
    return Forcing_Data(
        str(TEST_CSV),
        record_length=4018,
        n_feature=3,
        storage_device="cpu",
        seq_length=SEQ_LEN,
        target_seq_length=TGT_LEN,
        base_length=BASE_LEN,
    )


def load_lstm_models(device: torch.device) -> Tuple[torch.nn.Module, torch.nn.Module]:
    embedding = torch.load(DATA_DIR / "embedding_full.pt", map_location=device, weights_only=False)
    decoder = torch.load(DATA_DIR / "decoder_full.pt", map_location=device, weights_only=False)
    embedding.eval()
    decoder.eval()
    for m in (embedding, decoder):
        for p in m.parameters():
            p.requires_grad_(False)
        m.to(device)
    return embedding, decoder


def lstm_grad_sign(embedding, decoder, x_orig: torch.Tensor, tgt: torch.Tensor, idx: int) -> torch.Tensor:
    x_for_grad = x_orig.clone().requires_grad_(True)
    code = embedding(torch.tensor([idx], device=x_for_grad.device))
    pred = decoder.decode(code, x_for_grad)
    loss = mse_loss_with_nans(pred, tgt)
    loss.backward()
    grad = torch.nan_to_num(x_for_grad.grad, nan=0.0, posinf=0.0, neginf=0.0)
    return grad.sign().detach()


def run_lstm(args: argparse.Namespace, device: torch.device) -> pd.DataFrame:
    dset = load_lstm_dataset()
    if args.catchment not in dset.catchment_names:
        raise SystemExit(f"Catchment {args.catchment} not in dataset")
    idx = dset.catchment_names.index(args.catchment)
    embedding, decoder = load_lstm_models(device)

    x_orig = dset.x[idx].clone().unsqueeze(0).to(device)
    tgt = dset.y[idx].clone().to(device)[BASE_LEN:]
    if tgt.dim() == 1:
        tgt = tgt.unsqueeze(0)

    grad_sign = lstm_grad_sign(embedding, decoder, x_orig, tgt, idx)

    eps_list = np.array([float(e) for e in args.epsilons], dtype=float)
    preds = []
    for eps in eps_list:
        if eps == 0.0:
            pred = decoder.decode(embedding(torch.tensor([idx], device=device)), x_orig)
        else:
            x_adv = x_orig + float(eps) * grad_sign
            x_adv[:, :, 0].clamp_(min=0.0)
            x_adv[:, :, 2].clamp_(min=0.0)
            pred = decoder.decode(embedding(torch.tensor([idx], device=device)), x_adv)
        preds.append(pred.squeeze(0).detach().cpu().numpy())
    Y = np.stack(preds, axis=0)  

    slopes, r2s = [], []
    for t in range(Y.shape[1]):
        a, b, r2 = fit_line(eps_list, Y[:, t])
        slopes.append(a)
        r2s.append(r2)

    slopes = np.asarray(slopes)
    r2s = np.asarray(r2s)

    return pd.DataFrame(
        [
            {
                "model": "lstm",
                "catchment": args.catchment,
                "median_r2": float(np.nanmedian(r2s)),
                "mean_r2": float(np.nanmean(r2s)),
                "frac_r2_ge_0.95": float(np.nanmean(r2s >= 0.95)),
                "median_abs_slope": float(np.nanmedian(np.abs(slopes))),
                "mean_abs_slope": float(np.nanmean(np.abs(slopes))),
            }
        ]
    )


# ------------------------------ HBV ---------------------------------
def load_hbv_inputs_and_params(catchment: str) -> Tuple[pd.Series, pd.DataFrame]:
    params = pd.read_csv(PARAMETERS_CSV)
    selected = pd.read_csv(SELECTED_CSV)
    row = selected.merge(params, left_on="catchment_name", right_on="gauge_id")
    row = row.loc[row["catchment_name"] == catchment]
    if row.empty:
        raise SystemExit(f"Catchment {catchment} not found in parameters list")
    row = row.iloc[0]
    forc = pd.read_csv(TEST_CSV)
    forc = forc[forc["catchment_name"] == catchment].copy()
    if forc.empty:
        raise SystemExit(f"Forcings for {catchment} not found in test CSV")
    return row, forc


def build_hbv_param_tensors(row: pd.Series, seq_len: int, device: torch.device) -> Dict[str, torch.Tensor]:
    names = ["BETA", "FC", "K0", "K1", "K2", "LP", "PERC", "UZL", "TT", "CFMAX", "CFR", "CWH"]
    vals = torch.tensor([float(row[n]) for n in names], dtype=torch.float32, device=device)
    params: Dict[str, torch.Tensor] = {}
    for i, n in enumerate(names):
        params[n] = vals[i].view(1, 1, 1).expand(1, seq_len, 1)
    return params


def hbv_grad_sign(
    model: hbv.HBV,
    routing: hbv.UH_routing,
    x_orig: torch.Tensor,
    params: Dict[str, torch.Tensor],
    obs: torch.Tensor,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    x_for_grad = x_orig.clone().requires_grad_(True)
    discharge = model(x_for_grad, params)["y_hat"].squeeze(0).squeeze(-1)
    routed = routing(
        discharge.unsqueeze(0).unsqueeze(-1),
        {
            "alpha": torch.tensor([[[alpha]]], dtype=torch.float32, device=x_for_grad.device),
            "beta": torch.tensor([[[beta]]], dtype=torch.float32, device=x_for_grad.device),
        },
    ).squeeze(0).squeeze(-1)
    loss = mse_loss_with_nans(routed[WARMUP_DAYS:], obs[WARMUP_DAYS:])
    loss.backward()
    grad = torch.nan_to_num(x_for_grad.grad, nan=0.0, posinf=0.0, neginf=0.0)
    return grad.sign().detach()


@torch.no_grad()
def hbv_routed_series(
    model: hbv.HBV,
    routing: hbv.UH_routing,
    x: torch.Tensor,
    params: Dict[str, torch.Tensor],
    alpha: float,
    beta: float,
) -> torch.Tensor:
    discharge = model(x, params)["y_hat"].squeeze(0).squeeze(-1)
    routed = routing(
        discharge.unsqueeze(0).unsqueeze(-1),
        {
            "alpha": torch.tensor([[[alpha]]], dtype=torch.float32, device=x.device),
            "beta": torch.tensor([[[beta]]], dtype=torch.float32, device=x.device),
        },
    ).squeeze(0).squeeze(-1)
    return routed


def run_hbv(args: argparse.Namespace, device: torch.device) -> pd.DataFrame:
    row, forc = load_hbv_inputs_and_params(args.catchment)

    P = torch.tensor(forc["P"].to_numpy(), dtype=torch.float32)
    T = torch.tensor(forc["T"].to_numpy(), dtype=torch.float32)
    PET = torch.tensor(forc["PET"].to_numpy(), dtype=torch.float32)
    Q = torch.tensor(forc["Q"].to_numpy(), dtype=torch.float32)

    x_orig = torch.stack([P, PET, T], dim=1).unsqueeze(0).to(device)
    obs = Q.to(device)
    Tlen = x_orig.shape[1]

    model = hbv.HBV(n_models=1).to(device)
    model.eval()
    routing = hbv.UH_routing(n_models=1).to(device)
    routing.eval()
    params = build_hbv_param_tensors(row, Tlen, device)

    grad_sign = hbv_grad_sign(
        model, routing, x_orig, params, obs, float(row["alpha"]), float(row["beta"])
    )

    eps_list = np.array([float(e) for e in args.epsilons], dtype=float)
    preds = []
    for eps in eps_list:
        if eps == 0.0:
            routed = hbv_routed_series(model, routing, x_orig, params, float(row["alpha"]), float(row["beta"]))
        else:
            x_adv = x_orig + float(eps) * grad_sign
            x_adv[:, :, 0].clamp_(min=0.0)
            x_adv[:, :, 1].clamp_(min=0.0)
            routed = hbv_routed_series(model, routing, x_adv, params, float(row["alpha"]), float(row["beta"]))
        preds.append(routed[WARMUP_DAYS:].detach().cpu().numpy())
    Y = np.stack(preds, axis=0)  # [E, T']

    slopes, r2s = [], []
    for t in range(Y.shape[1]):
        a, b, r2 = fit_line(eps_list, Y[:, t])
        slopes.append(a)
        r2s.append(r2)

    slopes = np.asarray(slopes)
    r2s = np.asarray(r2s)

    return pd.DataFrame(
        [
            {
                "model": "hbv",
                "catchment": args.catchment,
                "median_r2": float(np.nanmedian(r2s)),
                "mean_r2": float(np.nanmean(r2s)),
                "frac_r2_ge_0.95": float(np.nanmean(r2s >= 0.95)),
                "median_abs_slope": float(np.nanmedian(np.abs(slopes))),
                "mean_abs_slope": float(np.nanmean(np.abs(slopes))),
            }
        ]
    )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    if args.model == "lstm":
        df = run_lstm(args, device)
    else:
        df = run_hbv(args, device)

    out = DATA_DIR / f"fgsm_linearity_{args.model}_{args.catchment}.csv"
    df.to_csv(out, index=False)
    print(f"Saved summary to {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

