#!/usr/bin/env python3
"""
FGSM perturbation study for the conceptual HBV model.

An FGSM attack with ε = 0.2 is applied to the meteorological forcings while
clamping precipitation (P) and potential evapotranspiration (PET) to remain
non-negative.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import HydroErr as he
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import hbv
from lstm_train_final_model import mse_loss_with_nans

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EPSILON: float = 0.2
WARMUP_DAYS: int = 365
DATA_DIR = Path("data")
OUTPUT_METRICS = DATA_DIR / f"hbv_fgsm_metrics_eps_{EPSILON:.2f}.csv"

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
# Helpers
# ---------------------------------------------------------------------------
def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps") # note running on mps may be slower than CPU
    return torch.device("cpu")


def build_parameter_tensors(row: pd.Series, seq_len: int, device: torch.device) -> Dict[str, torch.Tensor]:
    values = torch.tensor([float(row[name]) for name in PARAMETER_NAMES], dtype=torch.float32, device=device)
    parameters = {}
    for idx, name in enumerate(PARAMETER_NAMES):
        parameters[name] = values[idx].view(1, 1, 1).expand(1, seq_len, 1)
    return parameters


def routed_metrics(routed_eval: torch.Tensor, obs: torch.Tensor) -> Dict[str, float]:
    routed_trim = routed_eval[WARMUP_DAYS:].detach().cpu().numpy()
    obs_trim = obs[WARMUP_DAYS:].detach().cpu().numpy() 

    r_routed, alpha_routed, beta_routed, kge_routed = he.kge_2009(
        routed_trim, obs_trim, return_all=True
    )

    return {
        "mse_routed": he.mse(routed_trim, obs_trim),
        "nse_routed": he.nse(routed_trim, obs_trim),
        "r_routed": r_routed,
        "alpha_routed": alpha_routed,
        "beta_routed": beta_routed,
        "kge_routed": kge_routed,
    }


def fgsm_for_catchment(
    row: pd.Series,
    catchment_data: Dict[str, Dict[str, np.ndarray]],
    model: hbv.HBV,
    routing: hbv.UH_routing,
    device: torch.device,
) -> Dict[str, float]:
    cid = row["catchment_name"]
    series = catchment_data.get(cid)
    if series is None:
        return {"catchment": cid, "epsilon": EPSILON}

    P = series["P"]
    T = series["T"]
    PET = series["PET"]
    Q_obs = series["Q"]
    seq_len = len(P)

    inputs = torch.stack(
        [
            torch.tensor(P, dtype=torch.float32),
            torch.tensor(PET, dtype=torch.float32),
            torch.tensor(T, dtype=torch.float32),
        ],
        dim=1,
    ).unsqueeze(0).to(device)
    inputs.requires_grad_(True)

    obs_t = torch.tensor(Q_obs, dtype=torch.float32, device=device)

    parameters = build_parameter_tensors(row, seq_len, device)
    model.zero_grad(set_to_none=True)

    discharge = model(inputs, parameters)["y_hat"].squeeze(0).squeeze(-1)
    routing_params = {
        "alpha": torch.tensor([[[float(row["alpha"])]]], dtype=torch.float32, device=device),
        "beta": torch.tensor([[[float(row["beta"])]]], dtype=torch.float32, device=device),
    }
    routed_full = routing(
        discharge.unsqueeze(0).unsqueeze(-1),
        routing_params,
    ).squeeze(0).squeeze(-1)
    loss = mse_loss_with_nans(routed_full[WARMUP_DAYS:], obs_t[WARMUP_DAYS:])
    loss.backward()

    grad = torch.nan_to_num(inputs.grad, nan=0.0, posinf=0.0, neginf=0.0)
    grad_sign = grad.sign()
    adv_inputs = (inputs + EPSILON * grad_sign).detach()
    adv_inputs[:, :, 0].clamp_(min=0.0)  # precipitation must remain non-negative
    adv_inputs[:, :, 1].clamp_(min=0.0)  # PET must remain non-negative

    base_metrics = routed_metrics(routed_full.detach(), obs_t)
    with torch.no_grad():
        discharge_adv = model(adv_inputs, parameters)["y_hat"].squeeze(0).squeeze(-1)
        routed_adv = routing(discharge_adv.unsqueeze(0).unsqueeze(-1), routing_params).squeeze(0).squeeze(-1)
    adv_metrics = routed_metrics(routed_adv, obs_t)

    result = {"catchment": cid, "epsilon": EPSILON}
    for key, value in base_metrics.items():
        result[f"{key}_before"] = value
    for key, value in adv_metrics.items():
        result[f"{key}_after"] = value
        if key in base_metrics and pd.notna(base_metrics[key]) and pd.notna(value):
            result[f"{key}_delta"] = value - base_metrics[key]
        else:
            result[f"{key}_delta"] = float("nan")

    return result


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------
def main() -> None:
    device = resolve_device()
    print(f"Running HBV FGSM evaluation on {device}")

    parameters_df = pd.read_csv(DATA_DIR / "CAMELS_DE_parameters_hbv.csv")
    catchment_list = pd.read_csv(DATA_DIR / "selected_catchments.csv")
    catchment_params = catchment_list.merge(
        parameters_df, left_on="catchment_name", right_on="gauge_id"
    ).drop(columns=["gauge_id"])
    catchment_data_df = pd.read_csv(DATA_DIR / "data_test_CAMELS_DE1.00.csv")

    required_columns = ["catchment_name", "P", "T", "PET", "Q"]
    missing_columns = sorted(set(required_columns) - set(catchment_data_df.columns))
    if missing_columns:
        raise ValueError(f"Missing columns in test data: {missing_columns}")

    catchment_data: Dict[str, Dict[str, np.ndarray]] = {}
    for cid, grp in catchment_data_df[required_columns].groupby("catchment_name", sort=False):
        catchment_data[str(cid)] = {
            "P": grp["P"].to_numpy(dtype=np.float32, copy=True),
            "T": grp["T"].to_numpy(dtype=np.float32, copy=True),
            "PET": grp["PET"].to_numpy(dtype=np.float32, copy=True),
            "Q": grp["Q"].to_numpy(dtype=np.float32, copy=True),
        }

    model = hbv.HBV(n_models=1).to(device)
    model.eval()
    routing = hbv.UH_routing(n_models=1).to(device)
    routing.eval()

    results = []
    for _, row in tqdm(
        catchment_params.iterrows(),
        total=len(catchment_params),
        desc="FGSM attack on HBV",
    ):
        result = fgsm_for_catchment(row, catchment_data, model, routing, device)
        results.append(result)

    results_df = pd.DataFrame(results)
    OUTPUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_METRICS, index=False)
    print(f"Saved FGSM metrics to {OUTPUT_METRICS}")


if __name__ == "__main__":
    main()
