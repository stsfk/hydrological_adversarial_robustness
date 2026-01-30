#!/usr/bin/env python3
"""
Evaluate epsilon-linearity along random input directions for LSTM and HBV.

For a single catchment, generate N random sign directions in {-1, 0, 1}
(per time step and feature), sweep epsilons, and measure linearity of the
resulting simulated discharge time series via per-timepoint linear fits
(y_t ~ a_t * eps + b_t). Report summary stats per random direction.

Outputs:
  - data/random_dir_linearity_lstm_<catchment>.csv
  - data/random_dir_linearity_hbv_<catchment>.csv
  - data/figures/random_dir_linearity_hist_<catchment>.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# LSTM imports
from lstm_train_final_model import (
    Forcing_Data,
    LSTM_decoder,
    TimeDistributed,
)

# HBV imports
import hbv

DATA_DIR = Path("data")
FIG_DIR = DATA_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 365 * 2
TGT_LEN = 365
BASE_LEN = SEQ_LEN - TGT_LEN
WARMUP_DAYS = 365

# Keep explicit references so torch.load can safely unpickle these classes.
_ = LSTM_decoder
_ = TimeDistributed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catchment", default="DE911520")
    p.add_argument("--n-random", type=int, default=100, help="Number of random directions")
    p.add_argument(
        "--epsilons",
        type=float,
        nargs="*",
        default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
    )
    p.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--bins", type=int, default=40, help="Number of bins for histogram figure")
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


# ------------------------------ Figure ---------------------------------

def fmt_value(v: float) -> str:
    try:
        if not np.isfinite(v):
            return "nan"
        av = abs(v)
        if av < 1e-3:
            return f"{v:.1g}"
        if av < 1.0:
            return f"{v:.3f}"
        return f"{v:.2f}"
    except Exception:
        return "nan"


def uniform_bins_from_range(x: np.ndarray, nbins: int) -> np.ndarray | int:
    """Return monotonically increasing bin edges spanning [min, max] of finite x."""
    if x.size == 0:
        return nbins
    x_f = x[np.isfinite(x)]
    if x_f.size == 0:
        return nbins
    mn = float(np.min(x_f))
    mx = float(np.max(x_f))
    if not np.isfinite(mn) or not np.isfinite(mx):
        return nbins
    if mn == mx:
        delta = 1e-6 if mn == 0.0 else 1e-6 * abs(mn)
        mn -= delta
        mx += delta
    return np.linspace(mn, mx, nbins + 1)


def load_random_dir_csvs(catchment: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    lstm_path = DATA_DIR / f"random_dir_linearity_lstm_{catchment}.csv"
    hbv_path = DATA_DIR / f"random_dir_linearity_hbv_{catchment}.csv"
    if not lstm_path.exists() or not hbv_path.exists():
        raise FileNotFoundError(
            "Missing input CSVs for the combined figure. "
            f"Expected {lstm_path} and {hbv_path}."
        )
    df_lstm = pd.read_csv(lstm_path)
    df_hbv = pd.read_csv(hbv_path)

    # Normalize column names if needed
    for df in (df_lstm, df_hbv):
        if "frac_r2_ge_0.95" not in df.columns and "frac_r2_0.95" in df.columns:
            df.rename(columns={"frac_r2_0.95": "frac_r2_ge_0.95"}, inplace=True)
    return df_lstm, df_hbv


def plot_random_dir_histograms(catchment: str, bins: int) -> Path:
    """Create the 2×3 histogram figure and return the PDF path."""
    df_lstm, df_hbv = load_random_dir_csvs(catchment)

    metrics = [
        ("median_r2", r"$R^2$ (median)"),
        ("median_abs_slope", r"|slope| (median)"),
        ("frac_r2_ge_0.95", r"Fraction $R^2\geq 0.95$"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.0))
    color = "#4C78A8"
    edgecolor = "none"

    def _plot_row(ax, x: np.ndarray, label: str, ylabel: str) -> None:
        x_f = x[np.isfinite(x)]
        bins_edges = uniform_bins_from_range(x_f, max(10, bins))
        ax.hist(x_f, bins=bins_edges, color=color, edgecolor=edgecolor, alpha=1.0)
        if x_f.size:
            ax.set_xlim(float(np.min(x_f)), float(np.max(x_f)))

        med = float(np.median(x_f)) if x_f.size else np.nan
        if np.isfinite(med):
            ax.axvline(med, color="#E45756", linestyle="--", linewidth=1.4)
            ax.text(
                0.02,
                0.90,
                f"median: {fmt_value(med)}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                color="#E45756",
                fontsize=9,
            )

        mn = float(np.min(x_f)) if x_f.size else np.nan
        mx = float(np.max(x_f)) if x_f.size else np.nan
        ytxt = 0.84
        if np.isfinite(mn):
            ax.text(0.02, ytxt, f"min: {fmt_value(mn)}", transform=ax.transAxes, ha="left", va="top", fontsize=9)
            ytxt -= 0.06
        if np.isfinite(mx):
            ax.text(0.02, ytxt, f"max: {fmt_value(mx)}", transform=ax.transAxes, ha="left", va="top", fontsize=9)

        ax.set_xlabel(label)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

    for col, (key, label) in enumerate(metrics):
        _plot_row(
            axes[0, col],
            pd.to_numeric(df_lstm.get(key, pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float),
            label,
            "Count (LSTM)" if col == 0 else "Count",
        )
        _plot_row(
            axes[1, col],
            pd.to_numeric(df_hbv.get(key, pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float),
            label,
            "Count (HBV)" if col == 0 else "Count",
        )

    fig.tight_layout()
    out = FIG_DIR / f"random_dir_linearity_hist_{catchment}"
    pdf_path = out.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return pdf_path


# ------------------------------ LSTM ---------------------------------

def load_lstm_dataset(device: torch.device) -> Forcing_Data:
    return Forcing_Data(
        str(DATA_DIR / "data_test_CAMELS_DE1.00.csv"),
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
    embedding.eval(); decoder.eval()
    for m in (embedding, decoder):
        for p in m.parameters():
            p.requires_grad_(False)
        m.to(device)
    return embedding, decoder


@torch.no_grad()
def lstm_predict(embedding, decoder, x: torch.Tensor, idx: int) -> torch.Tensor:
    code = embedding(torch.tensor([idx], device=x.device))
    y = decoder.decode(code, x)
    return y.squeeze(0)  # [TGT_LEN]


def run_lstm_random(args: argparse.Namespace, device: torch.device) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed)
    dset = load_lstm_dataset(device)
    if args.catchment not in dset.catchment_names:
        raise SystemExit(f"Catchment {args.catchment} not in dataset")
    idx = dset.catchment_names.index(args.catchment)
    embedding, decoder = load_lstm_models(device)
    x_orig = dset.x[idx].clone().unsqueeze(0).to(device)
    T = x_orig.shape[1]
    F = x_orig.shape[2]  # features: [P, T, PET]

    eps_list = np.array([float(e) for e in args.epsilons], dtype=float)
    results: List[Dict[str, float]] = []

    for rid in range(args.n_random):
        # Random sign in {-1, 0, 1} per time step and feature
        dir_np = rng.integers(-1, 2, size=(1, T, F)).astype(np.int8)
        dir_t = torch.from_numpy(dir_np).to(device=device, dtype=x_orig.dtype)

        # Collect predictions across epsilons
        preds = []
        for eps in eps_list:
            x_adv = x_orig + float(eps) * dir_t
            # Clamp P and PET to be non-negative (channels 0 and 2)
            x_adv[:, :, 0].clamp_(min=0.0)
            x_adv[:, :, 2].clamp_(min=0.0)
            y = lstm_predict(embedding, decoder, x_adv, idx).detach().cpu().numpy()
            preds.append(y)
        Y = np.stack(preds, axis=0)  # [E, TGT_LEN]

        # Fit per time step
        slopes, r2s = [], []
        for t in range(Y.shape[1]):
            a, b, r2 = fit_line(eps_list, Y[:, t])
            slopes.append(a); r2s.append(r2)

        slopes = np.asarray(slopes); r2s = np.asarray(r2s)
        results.append(
            {
                "model": "lstm",
                "catchment": args.catchment,
                "rand_id": rid,
                "median_r2": float(np.nanmedian(r2s)),
                "mean_r2": float(np.nanmean(r2s)),
                "frac_r2_ge_0.95": float(np.nanmean(r2s >= 0.95)),
                "median_abs_slope": float(np.nanmedian(np.abs(slopes))),
                "mean_abs_slope": float(np.nanmean(np.abs(slopes))),
            }
        )

    return pd.DataFrame(results)


# ------------------------------ HBV ---------------------------------

def load_hbv_inputs_and_params(catchment: str) -> Tuple[pd.Series, pd.DataFrame]:
    params = pd.read_csv(DATA_DIR / "CAMELS_DE_parameters_hbv.csv")
    selected = pd.read_csv(DATA_DIR / "selected_catchments.csv")
    row = selected.merge(params, left_on="catchment_name", right_on="gauge_id")
    row = row.loc[row["catchment_name"] == catchment]
    if row.empty:
        raise SystemExit(f"Catchment {catchment} not found in parameters list")
    row = row.iloc[0]
    forc = pd.read_csv(DATA_DIR / "data_test_CAMELS_DE1.00.csv")
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


@torch.no_grad()
def hbv_routed_series(model: hbv.HBV, routing: hbv.UH_routing, x: torch.Tensor, params: Dict[str, torch.Tensor], alpha: float, beta: float) -> torch.Tensor:
    discharge = model(x, params)["y_hat"].squeeze(0).squeeze(-1)
    routed = routing(
        discharge.unsqueeze(0).unsqueeze(-1),
        {
            "alpha": torch.tensor([[[alpha]]], dtype=torch.float32, device=x.device),
            "beta": torch.tensor([[[beta]]], dtype=torch.float32, device=x.device),
        },
    ).squeeze(0).squeeze(-1)
    return routed


def run_hbv_random(args: argparse.Namespace, device: torch.device) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed)
    row, forc = load_hbv_inputs_and_params(args.catchment)

    P = torch.tensor(forc["P"].to_numpy(), dtype=torch.float32)
    T = torch.tensor(forc["T"].to_numpy(), dtype=torch.float32)
    PET = torch.tensor(forc["PET"].to_numpy(), dtype=torch.float32)

    # HBV input order used in other scripts: [P, PET, T]
    x_orig = torch.stack([P, PET, T], dim=1).unsqueeze(0).to(device)
    Tlen = x_orig.shape[1]
    F = x_orig.shape[2]  # 3

    model = hbv.HBV(n_models=1).to(device); model.eval()
    routing = hbv.UH_routing(n_models=1).to(device); routing.eval()
    params = build_hbv_param_tensors(row, Tlen, device)

    eps_list = np.array([float(e) for e in args.epsilons], dtype=float)
    results: List[Dict[str, float]] = []

    for rid in range(args.n_random):
        dir_np = rng.integers(-1, 2, size=(1, Tlen, F)).astype(np.int8)
        dir_t = torch.from_numpy(dir_np).to(device=device, dtype=x_orig.dtype)

        preds = []
        for eps in eps_list:
            x_adv = x_orig + float(eps) * dir_t
            # clamp P and PET >= 0 (channels 0 and 1 for this HBV order)
            x_adv[:, :, 0].clamp_(min=0.0)
            x_adv[:, :, 1].clamp_(min=0.0)
            routed = hbv_routed_series(model, routing, x_adv, params, float(row["alpha"]), float(row["beta"]))
            y = routed[WARMUP_DAYS:].detach().cpu().numpy()
            preds.append(y)
        Y = np.stack(preds, axis=0)  # [E, T']

        # Fit per time step (post-warmup)
        slopes, r2s = [], []
        for t in range(Y.shape[1]):
            a, b, r2 = fit_line(eps_list, Y[:, t])
            slopes.append(a); r2s.append(r2)
        slopes = np.asarray(slopes); r2s = np.asarray(r2s)

        results.append(
            {
                "model": "hbv",
                "catchment": args.catchment,
                "rand_id": rid,
                "median_r2": float(np.nanmedian(r2s)),
                "mean_r2": float(np.nanmean(r2s)),
                "frac_r2_ge_0.95": float(np.nanmean(r2s >= 0.95)),
                "median_abs_slope": float(np.nanmedian(np.abs(slopes))),
                "mean_abs_slope": float(np.nanmean(np.abs(slopes))),
            }
        )

    return pd.DataFrame(results)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    df_lstm = run_lstm_random(args, device)
    out_lstm = DATA_DIR / f"random_dir_linearity_lstm_{args.catchment}.csv"
    df_lstm.to_csv(out_lstm, index=False)

    df_hbv = run_hbv_random(args, device)
    out_hbv = DATA_DIR / f"random_dir_linearity_hbv_{args.catchment}.csv"
    df_hbv.to_csv(out_hbv, index=False)

    print(f"Saved summary to {out_lstm}")
    print(f"Saved summary to {out_hbv}")

    pdf_path = plot_random_dir_histograms(args.catchment, bins=args.bins)
    print(f"Saved figure to {pdf_path}")


if __name__ == "__main__":
    main()
