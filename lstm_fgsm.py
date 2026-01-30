#!/usr/bin/env python3
"""
Evaluate the vulnerability of the pretrained CAMELS-DE LSTM model to an FGSM attack.

This script loads the pretrained catchment embedding + LSTM decoder, then applies a
Fast Gradient Sign Method (FGSM) perturbation of magnitude ε to the forcing inputs
([P, T, PET]) for each catchment in the test split. Precipitation (P) and potential
evapotranspiration (PET) are clamped to be non-negative after perturbation.

The implementation follows the method in `lstm_fgsm.ipynb`, with the key difference
that the work is wrapped in a `main()` so it does not execute on import.

Outputs (under `data/` by default):
    - `lstm_fgsm_metrics_eps_<eps>.csv`
    - `lstm_summary_eps_<eps>.csv`
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import HydroErr as he
import pandas as pd
import torch
from tqdm import tqdm

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
DEFAULT_TEST_CSV = DATA_DIR / "data_test_CAMELS_DE1.00.csv"

SEQ_LEN = 365 * 2
TGT_LEN = 365
BASE_LEN = SEQ_LEN - TGT_LEN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epsilon", type=float, default=0.2, help="FGSM epsilon.")
    p.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV, help="Path to test CSV.")
    p.add_argument("--record-length", type=int, default=4018, help="Rows per catchment in the test CSV.")
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Computation device (auto = MPS if available else CPU).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of catchments to evaluate (for quick debugging).",
    )
    return p.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but is not available.")
        return torch.device("cuda")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but is not available.")
        return torch.device("mps")
    return torch.device("cpu")


def load_dataset(test_csv: Path, record_length: int, device: torch.device) -> Forcing_Data:
    return Forcing_Data(
        str(test_csv),
        record_length=record_length,
        n_feature=3,
        storage_device=device,
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


@torch.no_grad()
def catchment_code(embedding: torch.nn.Module, idx: int, device: torch.device) -> torch.Tensor:
    return embedding(torch.tensor([idx], device=device))


def fgsm_one(
    idx: int,
    eps: float,
    dataset: Forcing_Data,
    embedding: torch.nn.Module,
    decoder: torch.nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    x_orig = dataset.x[idx].clone().unsqueeze(0).to(device).requires_grad_(True)
    y = dataset.y[idx].clone().to(device)

    tgt = y[BASE_LEN:]
    if tgt.dim() == 1:
        tgt = tgt.unsqueeze(0)

    pred = decoder.decode(catchment_code(embedding, idx, device), x_orig)
    loss = mse_loss_with_nans(pred, tgt)
    loss.backward()

    grad_sign = x_orig.grad.sign()

    x_adv = (x_orig + eps * grad_sign).detach()
    x_adv[:, :, 0].clamp_(min=0.0)  # P >= 0
    x_adv[:, :, 2].clamp_(min=0.0)  # PET >= 0

    with torch.no_grad():
        pred_adv = decoder.decode(catchment_code(embedding, idx, device), x_adv)

    mse_b = float(loss.item())
    mse_a = float(mse_loss_with_nans(pred_adv.detach(), tgt).item())

    obs = tgt.detach().cpu().numpy().ravel()
    sim_b = pred.detach().cpu().numpy().ravel()
    sim_a = pred_adv.detach().cpu().numpy().ravel()

    r_b, a_b, b_b, kge_b = he.kge_2009(simulated_array=sim_b, observed_array=obs, return_all=True)
    r_a, a_a, b_a, kge_a = he.kge_2009(simulated_array=sim_a, observed_array=obs, return_all=True)
    nse_b = he.nse(simulated_array=sim_b, observed_array=obs)
    nse_a = he.nse(simulated_array=sim_a, observed_array=obs)

    return {
        "catchment_id": idx,
        "catchment": dataset.catchment_names[idx],
        "epsilon": eps,
        "mse_before": mse_b,
        "mse_after": mse_a,
        "mse_delta": mse_a - mse_b,
        "kge_before": kge_b,
        "kge_after": kge_a,
        "kge_delta": kge_a - kge_b,
        "r_before": r_b,
        "r_after": r_a,
        "r_delta": r_a - r_b,
        "alpha_before": a_b,
        "alpha_after": a_a,
        "alpha_delta": a_a - a_b,
        "beta_before": b_b,
        "beta_after": b_a,
        "beta_delta": b_a - b_b,
        "nse_before": nse_b,
        "nse_after": nse_a,
        "nse_delta": nse_a - nse_b,
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    if device.type == "cuda":
        print(
            "Warning: CUDA may fail for LSTM backward (cuDNN). "
            "If you hit errors, try `--device mps` or `--device cpu`."
        )

    print("Running on", device)
    dataset = load_dataset(args.test_csv, args.record_length, device)
    embedding, decoder = load_models(device)

    n_catch = dataset.n_catchment
    if args.limit is not None:
        n_catch = min(n_catch, args.limit)

    results: List[Dict[str, Any]] = []
    for idx in tqdm(range(n_catch), total=n_catch, desc=f"LSTM FGSM (ε={args.epsilon:.2f})"):
        results.append(fgsm_one(idx, args.epsilon, dataset, embedding, decoder, device))

    results_df = pd.DataFrame(results)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = DATA_DIR / f"lstm_fgsm_metrics_eps_{args.epsilon:.2f}.csv"
    results_df.to_csv(metrics_path, index=False)

    summary_df = results_df[["mse_delta", "kge_delta", "nse_delta"]].agg(["mean", "median"])
    summary_path = DATA_DIR / f"lstm_summary_eps_{args.epsilon:.2f}.csv"
    summary_df.to_csv(summary_path)

    print(f"Metrics saved to {metrics_path}")
    print(f"Summary saved to {summary_path}")
    print(summary_df)


if __name__ == "__main__":
    main()


