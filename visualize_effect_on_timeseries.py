#!/usr/bin/env python3
"""
Plot example time series showing that adversarial input changes are visually
small for both the LSTM and HBV models. For a chosen catchment and date range,
this script overlays baseline vs. FGSM-perturbed precipitation (P) and
potential evapotranspiration (PET) inputs.

Usage (defaults shown):
    python visualize_input_change_timeseries.py \
        --catchment DE911520 \
        --epsilon 0.20 \
        --start 2016-01-01 --end 2016-12-31

Outputs:
    data/figures/input_change_timeseries_<catchment>_<start>_<end>.{png,pdf}
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import hbv
from lstm_train_final_model import Forcing_Data, LSTM_decoder, TimeDistributed, mse_loss_with_nans

# Explicit references for torch.load safety
_ = LSTM_decoder
_ = TimeDistributed


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
FIGURE_DIR = DATA_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 365 * 2
TGT_LEN = 365
BASE_LEN = SEQ_LEN - TGT_LEN
WARMUP_DAYS_HBV = 365


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catchment", default="DE911520", help="Catchment (e.g., DE911520)")
    p.add_argument("--epsilon", type=float, default=0.20, help="FGSM epsilon")
    p.add_argument("--start", default="2016-01-01", help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", default="2016-12-31", help="End date (YYYY-MM-DD)")
    p.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    return p.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")
    return torch.device("cpu")


def load_test_df() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "data_test_CAMELS_DE1.00.csv")


def load_lstm_components(device: torch.device):
    embedding = torch.load(DATA_DIR / "embedding_full.pt", map_location=device, weights_only=False)
    decoder = torch.load(DATA_DIR / "decoder_full.pt", map_location=device, weights_only=False)
    embedding.eval(); decoder.eval()
    for m in (embedding, decoder):
        for p in m.parameters():
            p.requires_grad_(False)
        m.to(device)
    return embedding, decoder


def load_lstm_dataset() -> Forcing_Data:
    return Forcing_Data(
        str(DATA_DIR / "data_test_CAMELS_DE1.00.csv"),
        record_length=4018,
        n_feature=3,
        storage_device="cpu",
        seq_length=SEQ_LEN,
        target_seq_length=TGT_LEN,
        base_length=BASE_LEN,
    )


def lstm_compute_adv_inputs(
    catchment: str, epsilon: float, device: torch.device
) -> Tuple[
    pd.Series,  # dates (inputs)
    pd.Series, pd.Series,  # P_base, P_adv
    pd.Series, pd.Series,  # T_base, T_adv
    pd.Series, pd.Series,  # PET_base, PET_adv
    pd.Series,  # dates_pred (for discharge predictions; starts after first year)
    pd.Series, pd.Series, pd.Series,  # Q_base, Q_adv, Q_obs
]:
    """
    Return baseline and FGSM-perturbed input time series and discharge for LSTM:
    (dates_inputs, P_base, P_adv, T_base, T_adv, PET_base, PET_adv,
     dates_pred, Q_base, Q_adv, Q_obs)
    """
    df = load_test_df()
    sub = df[df["catchment_name"] == catchment].copy()
    if sub.empty:
        raise SystemExit(f"Catchment {catchment} not found in test data")
    dates = pd.to_datetime(sub["Date"]).reset_index(drop=True)

    dataset = load_lstm_dataset()
    if catchment not in dataset.catchment_names:
        raise SystemExit(f"Catchment {catchment} not present in LSTM dataset")
    idx = dataset.catchment_names.index(catchment)

    embedding, decoder = load_lstm_components(device)

    # Build tensors
    x_full = dataset.x[idx].clone().unsqueeze(0).to(device)  # [1, T, 3], feature order: [P, T, PET]
    y_full = dataset.y[idx].clone().to(device)
    tgt = y_full[BASE_LEN:]
    if tgt.dim() == 1:
        tgt = tgt.unsqueeze(0)  # [1, TGT_LEN]

    # Compute gradients
    x = x_full.clone().requires_grad_(True)
    code = embedding(torch.tensor([idx], device=device))
    pred = decoder.decode(code, x)
    loss = mse_loss_with_nans(pred, tgt)
    loss.backward()
    grad_sign = torch.nan_to_num(x.grad, nan=0.0, posinf=0.0, neginf=0.0).sign()
    x_adv = x.detach() + epsilon * grad_sign
    # Clamp P and PET non-negative (feature order [P, T, PET])
    x_adv[:, :, 0].clamp_(min=0.0)
    x_adv[:, :, 2].clamp_(min=0.0)

    # Extract input series
    P_base = pd.Series(x_full[0, :, 0].cpu().numpy())
    T_base = pd.Series(x_full[0, :, 1].cpu().numpy())
    PET_base = pd.Series(x_full[0, :, 2].cpu().numpy())
    P_adv = pd.Series(x_adv[0, :, 0].cpu().numpy())
    T_adv = pd.Series(x_adv[0, :, 1].cpu().numpy())
    PET_adv = pd.Series(x_adv[0, :, 2].cpu().numpy())
    
    # Discharge predictions (baseline vs adversarial) and observed
    Q_base = pd.Series(decoder.decode(code, x_full).squeeze(0).detach().cpu().numpy())
    Q_adv = pd.Series(decoder.decode(code, x_adv).squeeze(0).detach().cpu().numpy())
    Q_obs = pd.Series(tgt.squeeze(0).detach().cpu().numpy())
    dates_pred = dates[BASE_LEN:].reset_index(drop=True)

    return (
        dates,
        P_base, P_adv,
        T_base, T_adv,
        PET_base, PET_adv,
        dates_pred,
        Q_base, Q_adv, Q_obs,
    )


def hbv_compute_adv_inputs(
    catchment: str, epsilon: float, device: torch.device
) -> Tuple[
    pd.Series,  # dates (inputs, and for routed since same length)
    pd.Series, pd.Series,  # P_base, P_adv
    pd.Series, pd.Series,  # T_base, T_adv
    pd.Series, pd.Series,  # PET_base, PET_adv
    pd.Series, pd.Series, pd.Series,  # Q_base, Q_adv, Q_obs
]:
    """
    Return baseline and FGSM-perturbed inputs and routed discharge for HBV:
    (dates, P_base, P_adv, T_base, T_adv, PET_base, PET_adv, Q_base, Q_adv, Q_obs)
    """
    df = load_test_df()
    sub = df[df["catchment_name"] == catchment].copy()
    if sub.empty:
        raise SystemExit(f"Catchment {catchment} not found in test data")
    dates = pd.to_datetime(sub["Date"]).reset_index(drop=True)

    # Prepare inputs (feature order for HBV FGSM: [P, PET, T])
    P = torch.tensor(sub["P"].astype(float).values, dtype=torch.float32, device=device)
    T = torch.tensor(sub["T"].astype(float).values, dtype=torch.float32, device=device)
    PET = torch.tensor(sub["PET"].astype(float).values, dtype=torch.float32, device=device)
    Q_obs = torch.tensor(sub["Q"].astype(float).values, dtype=torch.float32, device=device)
    x_in = torch.stack([P, PET, T], dim=1).unsqueeze(0)  # [1, T, 3]
    x_in.requires_grad_(True)

    # HBV model + routing
    model = hbv.HBV(n_models=1).to(device)
    routing = hbv.UH_routing(n_models=1).to(device)
    model.eval(); routing.eval()

    # Parameters
    params_df = pd.read_csv(DATA_DIR / "CAMELS_DE_parameters_hbv.csv")
    row = params_df[params_df["gauge_id"] == catchment]
    if row.empty:
        raise SystemExit(f"Catchment {catchment} not found in HBV parameter table")
    r0 = row.iloc[0]
    # Expand parameter scalars to [1, T, 1]
    def expand_param(val: float) -> torch.Tensor:
        return torch.tensor(val, dtype=torch.float32, device=device).view(1, 1, 1).expand(1, x_in.shape[1], 1)
    param_tensors = {name: expand_param(float(r0[name])) for name in [
        "BETA","FC","K0","K1","K2","LP","PERC","UZL","TT","CFMAX","CFR","CWH"
    ]}

    discharge = model(x_in, param_tensors)["y_hat"].squeeze(0).squeeze(-1)
    routed = routing(
        discharge.unsqueeze(0).unsqueeze(-1),
        {
            "alpha": torch.tensor([[[float(r0['alpha'])]]], dtype=torch.float32, device=device),
            "beta": torch.tensor([[[float(r0['beta'])]]], dtype=torch.float32, device=device),
        },
    )
    routed_eval = routed.squeeze(0).squeeze(-1)
    routed_trim = routed_eval[WARMUP_DAYS_HBV:]
    obs_trim = Q_obs[WARMUP_DAYS_HBV:]

    # Loss and gradients
    loss = mse_loss_with_nans(routed_trim, obs_trim)
    loss.backward()
    grad = torch.nan_to_num(x_in.grad, nan=0.0, posinf=0.0, neginf=0.0)
    x_adv = (x_in.detach() + epsilon * grad.sign())
    # Clamp P and PET non-negative
    x_adv[:, :, 0].clamp_(min=0.0)
    x_adv[:, :, 1].clamp_(min=0.0)

    # Inputs (baseline and adversarial)
    P_base = pd.Series(x_in[0, :, 0].detach().cpu().numpy())
    PET_base = pd.Series(x_in[0, :, 1].detach().cpu().numpy())
    T_base = pd.Series(x_in[0, :, 2].detach().cpu().numpy())
    P_adv = pd.Series(x_adv[0, :, 0].detach().cpu().numpy())
    PET_adv = pd.Series(x_adv[0, :, 1].detach().cpu().numpy())
    T_adv = pd.Series(x_adv[0, :, 2].detach().cpu().numpy())
    
    # Routed discharge (baseline and adversarial)
    discharge_adv = model(x_adv, param_tensors)["y_hat"].squeeze(0).squeeze(-1)
    routed_adv = routing(
        discharge_adv.unsqueeze(0).unsqueeze(-1),
        {
            "alpha": torch.tensor([[[float(r0['alpha'])]]], dtype=torch.float32, device=device),
            "beta": torch.tensor([[[float(r0['beta'])]]], dtype=torch.float32, device=device),
        },
    ).squeeze(0).squeeze(-1)

    Q_base = pd.Series(routed_eval.detach().cpu().numpy())
    Q_adv = pd.Series(routed_adv.detach().cpu().numpy())
    Q_obs = pd.Series(Q_obs.detach().cpu().numpy())

    return dates, P_base, P_adv, T_base, T_adv, PET_base, PET_adv, Q_base, Q_adv, Q_obs


def subset_by_date(dates: pd.Series, *series: pd.Series, start: str, end: str):
    idx = (dates >= pd.to_datetime(start)) & (dates <= pd.to_datetime(end))
    mask = idx.to_numpy()
    out_dates = dates[mask].reset_index(drop=True)
    out_series = []
    for s in series:
        if isinstance(s, pd.Series):
            out_series.append(s.iloc[mask].reset_index(drop=True))
        else:
            out_series.append(s[mask])
    return (out_dates, *out_series)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    # LSTM inputs + discharge
    (
        d_lstm_all,
        P_b_lstm, P_a_lstm,
        T_b_lstm, T_a_lstm,
        PET_b_lstm, PET_a_lstm,
        d_lstm_pred_all,
        Q_b_lstm, Q_a_lstm, Q_obs_lstm,
    ) = lstm_compute_adv_inputs(
        args.catchment, args.epsilon, device
    )
    d_lstm, P_b_lstm, P_a_lstm, T_b_lstm, T_a_lstm, PET_b_lstm, PET_a_lstm = subset_by_date(
        d_lstm_all, P_b_lstm, P_a_lstm, T_b_lstm, T_a_lstm, PET_b_lstm, PET_a_lstm,
        start=args.start, end=args.end
    )
    d_lstm_pred, Q_b_lstm, Q_a_lstm, Q_obs_lstm = subset_by_date(
        d_lstm_pred_all, Q_b_lstm, Q_a_lstm, Q_obs_lstm, start=args.start, end=args.end
    )

    # HBV inputs + discharge
    d_hbv_all, P_b_hbv, P_a_hbv, T_b_hbv, T_a_hbv, PET_b_hbv, PET_a_hbv, Q_b_hbv, Q_a_hbv, Q_obs_hbv = hbv_compute_adv_inputs(
        args.catchment, args.epsilon, device
    )
    d_hbv, P_b_hbv, P_a_hbv, T_b_hbv, T_a_hbv, PET_b_hbv, PET_a_hbv = subset_by_date(
        d_hbv_all, P_b_hbv, P_a_hbv, T_b_hbv, T_a_hbv, PET_b_hbv, PET_a_hbv,
        start=args.start, end=args.end
    )
    d_hbv_q, Q_b_hbv, Q_a_hbv, Q_obs_hbv = subset_by_date(
        d_hbv_all, Q_b_hbv, Q_a_hbv, Q_obs_hbv, start=args.start, end=args.end
    )

    # Plot 4x2: rows = P, T, PET, Q; cols = LSTM | HBV
    fig, axes = plt.subplots(4, 2, figsize=(12, 11.5), sharex=True)
    # LSTM P
    ax = axes[0, 0]
    ax.plot(d_lstm, P_b_lstm, color="#1f78b4", lw=1.3, label="Baseline")
    ax.plot(
        d_lstm,
        P_a_lstm,
        color="#e31a1c",
        lw=1.0,
        alpha=0.9,
        linestyle="--",
        label=f"FGSM ε={args.epsilon:.2f}",
    )
    ax.set_title(f"LSTM — Precipitation (P)")
    ax.set_ylabel("mm/day")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", frameon=False)
    # LSTM T
    ax = axes[1, 0]
    ax.plot(d_lstm, T_b_lstm, color="#1f78b4", lw=1.3, label="Baseline")
    ax.plot(
        d_lstm,
        T_a_lstm,
        color="#e31a1c",
        lw=1.0,
        alpha=0.9,
        linestyle="--",
        label=f"FGSM ε={args.epsilon:.2f}",
    )
    ax.set_title("LSTM — Temperature (T)")
    ax.set_ylabel("°C")
    ax.grid(True, linestyle=":", alpha=0.5)
    # LSTM PET
    ax = axes[2, 0]
    ax.plot(d_lstm, PET_b_lstm, color="#1f78b4", lw=1.3, label="Baseline")
    ax.plot(
        d_lstm,
        PET_a_lstm,
        color="#e31a1c",
        lw=1.0,
        alpha=0.9,
        linestyle="--",
        label=f"FGSM ε={args.epsilon:.2f}",
    )
    ax.set_title("LSTM — Potential Evapotranspiration (PET)")
    ax.set_ylabel("mm/day")
    ax.grid(True, linestyle=":", alpha=0.5)
    # HBV P
    ax = axes[0, 1]
    ax.plot(d_hbv, P_b_hbv, color="#33a02c", lw=1.3, label="Baseline")
    ax.plot(
        d_hbv,
        P_a_hbv,
        color="#fb9a99",
        lw=1.0,
        alpha=0.9,
        linestyle="--",
        label=f"FGSM ε={args.epsilon:.2f}",
    )
    ax.set_title("HBV — Precipitation (P)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", frameon=False)
    # HBV T
    ax = axes[1, 1]
    ax.plot(d_hbv, T_b_hbv, color="#33a02c", lw=1.3, label="Baseline")
    ax.plot(
        d_hbv,
        T_a_hbv,
        color="#fb9a99",
        lw=1.0,
        alpha=0.9,
        linestyle="--",
        label=f"FGSM ε={args.epsilon:.2f}",
    )
    ax.set_title("HBV — Temperature (T)")
    ax.grid(True, linestyle=":", alpha=0.5)
    # HBV PET
    ax = axes[2, 1]
    ax.plot(d_hbv, PET_b_hbv, color="#33a02c", lw=1.3, label="Baseline")
    ax.plot(
        d_hbv,
        PET_a_hbv,
        color="#fb9a99",
        lw=1.0,
        alpha=0.9,
        linestyle="--",
        label=f"FGSM ε={args.epsilon:.2f}",
    )
    ax.set_title("HBV — Potential Evapotranspiration (PET)")
    ax.grid(True, linestyle=":", alpha=0.5)

    # LSTM Q (predictions start after first year of record)
    ax = axes[3, 0]
    ax.plot(d_lstm_pred, Q_obs_lstm, color="black", lw=1.2, label="Observed")
    ax.plot(d_lstm_pred, Q_b_lstm, color="#1f78b4", lw=1.2, label="Baseline")
    ax.plot(
        d_lstm_pred,
        Q_a_lstm,
        color="#e31a1c",
        lw=1.0,
        alpha=0.9,
        linestyle="--",
        label=f"FGSM ε={args.epsilon:.2f}",
    )
    ax.set_title("LSTM — Discharge (Q)")
    ax.set_ylabel("mm/day")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", frameon=False)

    # HBV Q (routed discharge)
    ax = axes[3, 1]
    ax.plot(d_hbv_q, Q_obs_hbv, color="black", lw=1.2, label="Observed")
    ax.plot(d_hbv_q, Q_b_hbv, color="#33a02c", lw=1.2, label="Baseline")
    ax.plot(
        d_hbv_q,
        Q_a_hbv,
        color="#fb9a99",
        lw=1.0,
        alpha=0.9,
        linestyle="--",
        label=f"FGSM ε={args.epsilon:.2f}",
    )
    ax.set_title("HBV — Discharge (Q)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", frameon=False)

    for ax in axes[3, :]:
        ax.set_xlabel("Date")

    fig.tight_layout()

    stem = FIGURE_DIR / f"input_change_timeseries_{args.catchment}_{args.start}_{args.end}"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    print("Saved figure →", stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
