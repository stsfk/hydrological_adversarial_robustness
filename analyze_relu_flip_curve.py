#!/usr/bin/env python3
"""
Plot ReLU sign-flip proportion vs epsilon for the LSTM head.

For a selected catchment, we compute, for each ReLU layer in the
TimeDistributed head, the fraction of time steps whose pre-activation
sign flips between baseline (ε=0) and FGSM with ε in {0.1,0.2,0.3,0.4,0.5}.

Outputs:
  - CSV with per-epsilon, per-layer summary stats (mean/median/max flip frac)
  - Line plot of mean flip fraction vs ε per layer
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from lstm_train_final_model import (
    Forcing_Data,
    LSTM_decoder,
    TimeDistributed,
    mse_loss_with_nans,
)

# explicit references for torch.load safety
_ = LSTM_decoder
_ = TimeDistributed

DATA_DIR = Path("data")
FIGURE_DIR = DATA_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 365 * 2
TGT_LEN = 365
BASE_LEN = SEQ_LEN - TGT_LEN


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catchment", default="DE911520", help="Catchment to analyse")
    p.add_argument(
        "--epsilons",
        type=float,
        nargs="*",
        default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        help="List of epsilon values to evaluate (default: 0.05 steps from 0.05 to 0.50)",
    )
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


def load_dataset() -> Forcing_Data:
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
    embedding.eval(); decoder.eval()
    for m in (embedding, decoder):
        for p in m.parameters():
            p.requires_grad_(False)
        m.to(device)
    return embedding, decoder


def compute_fgsm_grad_sign(
    x_orig: torch.Tensor, target: torch.Tensor, embedding, decoder, idx: int
) -> torch.Tensor:
    x = x_orig.clone().requires_grad_(True)
    code = embedding(torch.tensor([idx], device=x.device))
    pred = decoder.decode(code, x)
    loss = mse_loss_with_nans(pred, target)
    loss.backward()
    grad = torch.nan_to_num(x.grad, nan=0.0, posinf=0.0, neginf=0.0)
    return grad.sign().detach()


@torch.no_grad()
def lstm_hidden(decoder: LSTM_decoder, embedding, x: torch.Tensor, idx: int) -> torch.Tensor:
    code = embedding(torch.tensor([idx], device=x.device))
    code_exp = code.expand(x.size(1), -1, -1).transpose(0, 1)
    lstm_in = torch.cat([code_exp, x], dim=2)
    out, _ = decoder.lstm(lstm_in)
    return out[:, decoder.base_length :, :]


@torch.no_grad()
def relu_preacts(decoder: LSTM_decoder, hidden_seq: torch.Tensor) -> Dict[str, np.ndarray]:
    batch, time_steps, feat_dim = hidden_seq.shape
    current = hidden_seq.reshape(batch * time_steps, feat_dim)
    preacts: Dict[str, np.ndarray] = {}
    relu_count = 0
    for layer in decoder.fc.m:
        if isinstance(layer, nn.ReLU):
            relu_count += 1
            z = current.detach().cpu().numpy().reshape(batch, time_steps, -1).squeeze(0)
            units = z.shape[1]
            preacts[f"MLP hidden layer {relu_count} (n={units})"] = z
        current = layer(current)
    return preacts


def sign_array(a: np.ndarray) -> np.ndarray:
    s = np.zeros_like(a, dtype=np.int8)
    s[a > 0] = 1
    s[a < 0] = -1
    return s


def flip_stats_for_eps(
    decoder: LSTM_decoder,
    embedding,
    x_orig: torch.Tensor,
    grad_sign: torch.Tensor,
    zb: Dict[str, np.ndarray],
    idx: int,
    eps: float,
) -> List[Dict[str, float]]:
    x_adv = (x_orig + eps * grad_sign).detach()
    x_adv[:, :, 0].clamp_(min=0.0)
    x_adv[:, :, 2].clamp_(min=0.0)
    h_adv = lstm_hidden(decoder, embedding, x_adv, idx)
    za = relu_preacts(decoder, h_adv)
    rows: List[Dict[str, float]] = []
    for layer_name in zb.keys():
        sb = sign_array(zb[layer_name])
        sa = sign_array(za[layer_name])
        flips = (sb != sa) & ~((sb == 0) & (sa == 0))
        flips_per_unit = flips.sum(axis=0)
        T = sb.shape[0]
        for j, cnt in enumerate(flips_per_unit.tolist()):
            rows.append(
                {
                    "epsilon": eps,
                    "layer": layer_name,
                    "neuron_idx": j,
                    "flip_count": int(cnt),
                    "flip_frac": float(cnt) / float(T),
                    "time_steps": int(T),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    dataset = load_dataset()
    if args.catchment not in dataset.catchment_names:
        raise SystemExit(f"Catchment {args.catchment} not found in dataset")
    idx = dataset.catchment_names.index(args.catchment)

    embedding, decoder = load_models(device)

    x_orig = dataset.x[idx].clone().unsqueeze(0).to(device)
    y_full = dataset.y[idx].clone().to(device)
    target = y_full[decoder.base_length :]
    if target.dim() == 1:
        target = target.unsqueeze(0)

    grad_sign = compute_fgsm_grad_sign(x_orig, target, embedding, decoder, idx)
    h_base = lstm_hidden(decoder, embedding, x_orig, idx)
    zb = relu_preacts(decoder, h_base)

    all_rows: List[Dict[str, float]] = []
    for eps in args.epsilons:
        all_rows.extend(
            flip_stats_for_eps(
                decoder, embedding, x_orig, grad_sign, zb, idx, float(eps)
            )
        )

    df = pd.DataFrame(all_rows)
    out_csv = DATA_DIR / f"lstm_relu_flip_curve_{args.catchment}.csv"
    df.to_csv(out_csv, index=False)

    # Aggregate per epsilon, per layer
    summ = (
        df.groupby(["epsilon", "layer"])  # type: ignore
        .agg(mean_flip_frac=("flip_frac", "mean"), median_flip_frac=("flip_frac", "median"), max_flip_frac=("flip_frac", "max"), n_units=("flip_frac", "count"))
        .reset_index()
        .sort_values(["layer", "epsilon"])
    )
    out_csv2 = DATA_DIR / f"lstm_relu_flip_curve_{args.catchment}_summary.csv"
    summ.to_csv(out_csv2, index=False)

    # Plot mean flip fraction vs epsilon per layer
    layers = summ["layer"].unique().tolist()
    colors = plt.cm.Set2(np.linspace(0, 1, len(layers)))
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for c, layer in zip(colors, layers):
        sub = summ[summ["layer"] == layer]
        # Layer name already contains unit count, avoid duplicating "(n=xx)" in label
        ax.plot(sub["epsilon"], sub["mean_flip_frac"], marker="o", color=c, label=str(layer))
    ax.set_xlabel("FGSM ε")
    ax.set_ylabel("Mean ReLU sign-flip fraction")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    fig.tight_layout()

    stem = FIGURE_DIR / f"lstm_relu_flip_curve_{args.catchment}"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print("Saved per-unit flip CSV:", out_csv)
    print("Saved summary CSV:", out_csv2)
    print("Saved plot:", stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
