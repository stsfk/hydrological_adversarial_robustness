"""
Hyperparameter tuning for a catchment-embedding + LSTM decoder model on CAMELS-DE.

This script loads preprocessed train/validation CSVs from `./data/`, samples random
input/target sequences, and uses Optuna to minimize validation MSE (ignoring NaNs).
When run as a script, it saves the Optuna study to `data/hyperparameter_optimization_results.pkl`.
"""

# %%
import math, random, pathlib, joblib, optuna, numpy as np, torch
from torch import nn, optim
from torch.utils.data import Dataset
from contextlib import nullcontext

# %%
# 1. Device handling ---------------------------------------------
def resolve_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = resolve_device()


# %%
# 2. Data loader --------------------------------------------------
class Forcing_Data(Dataset):
    def __init__(self,
                 fpath               : str,
                 record_length       : int,
                 n_feature           : int = 3,
                 seq_length          : int = 730,
                 target_seq_length   : int = 365,
                 base_length         : int = 365,
                 storage_device      : str | torch.device = "cpu"):
        data_raw = np.genfromtxt(fpath, delimiter=",", skip_header=1)

        x = torch.from_numpy(data_raw[:, 2:2+n_feature]).float() # features start from col 2
        x = x.view(-1, record_length, n_feature).contiguous()
        self.x = x.to(storage_device)

        y = torch.from_numpy(data_raw[:, 2+n_feature]).float()
        y = y.view(-1, record_length).contiguous()
        self.y = y.to(storage_device)

        self.n_catchment       = self.y.shape[0]
        self.n_feature         = n_feature
        self.record_length     = record_length
        self.seq_length        = seq_length
        self.target_seq_length = target_seq_length
        self.base_length       = base_length
        self.storage_device    = storage_device

    def __len__(self):              return self.n_catchment
    def __getitem__(self, idx):     return self.x[idx], self.y[idx]

    def get_random_batch(self, batch_size: int = 64):
        idx_cat = random.sample(range(self.n_catchment), batch_size)
        idx_cat = torch.tensor(idx_cat, device=self.storage_device)

        x_sub = self.x.index_select(0, idx_cat)
        y_sub = self.y.index_select(0, idx_cat)

        idx_start = torch.randint(
            0, self.record_length - self.seq_length + 1,
            (batch_size,), device=self.storage_device)

        arange_seq = torch.arange(self.seq_length, device=self.storage_device)
        gather_y   = idx_start.unsqueeze(-1) + arange_seq          # [bs, seq]
        gather_x   = gather_y.unsqueeze(-1).repeat(1,1,self.n_feature)

        x_batch = x_sub.gather(1, gather_x)
        y_batch = y_sub.gather(1, gather_y)

        return x_batch, y_batch[:, self.base_length:], idx_cat

    def get_full_sequence(self):
        return self.x, self.y


# %%
# 3. Model definitions -------------------------------------------
class TimeDistributed(nn.Module):
    def __init__(self, module: nn.Module, batch_first=False):
        super().__init__()
        self.module, self.batch_first = module, batch_first
    def forward(self, x):
        if x.dim() <= 2:
            return self.module(x)
        b, t = (x.shape[0], x.shape[1]) if self.batch_first else (x.shape[1], x.shape[0])
        y = self.module(x.contiguous().view(-1, x.shape[-1]))
        return y.view(b, t, -1) if self.batch_first else y.view(t, b, -1)


# %%
class LSTM_decoder(nn.Module):
    def __init__(self,
                 latent_dim, feature_dim,
                 lstm_hidden_dim, fc_hidden_dims,
                 num_lstm_layers=1, output_dim=1, p=0.2, base_length=365):
        super().__init__()
        self.base_length = base_length
        self.lstm = nn.LSTM(feature_dim + latent_dim,
                            lstm_hidden_dim,
                            num_layers=num_lstm_layers,
                            batch_first=True)
        layers, in_dim = [], lstm_hidden_dim
        for h in fc_hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(p)]
            in_dim = h
        layers += [nn.Linear(in_dim, output_dim)]
        self.fc = TimeDistributed(nn.Sequential(*layers), batch_first=True)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, self.base_length:]).squeeze(-1)

    def decode(self, code, x):
        code = code.expand(x.size(1), -1, -1).transpose(0,1)
        return self.forward(torch.cat([code, x], dim=2))


# %%
# 4. Training / validation utils ---------------------------------
def mse_loss_with_nans(pred, target):
    mask = torch.isnan(target)
    return ((pred[~mask] - target[~mask])**2).mean()

class EarlyStopper:
    def __init__(self, patience=20, min_delta=0.):
        self.patience, self.min_delta = patience, min_delta
        self.counter, self.best = 0, math.inf
    def __call__(self, metric):
        if metric < self.best - self.min_delta:
            self.best, self.counter = metric, 0
        else:
            self.counter += 1
        return self.counter >= self.patience


@torch.no_grad()
def val_model_continuous(embedding, decoder, dataset, batch_size: int = 64):
    """
    Evaluate on the entire record and return the *global* MSE
    (sum of squared errors divided by number of valid points).
    Works no matter whether the last batch is smaller than `batch_size`.
    """
    x_cpu, y_cpu = dataset.get_full_sequence()          # master copy on CPU
    n_catch      = x_cpu.size(0)

    sse_total = 0.0     # sum of squared errors
    n_total   = 0       # number of valid points

    base_length = dataset.base_length                          # usually 365

    for start in range(0, n_catch, batch_size):
        end  = min(start + batch_size, n_catch)

        xb   = x_cpu[start:end].to(DEVICE, non_blocking=True)
        yb   = y_cpu[start:end].to(DEVICE, non_blocking=True)
        cats = torch.arange(start, end, device=DEVICE)

        code = embedding(cats)
        pred = decoder.decode(code, xb)

        # --- accumulate SSE and count ---------------------------------------
        target_slice = yb[:, base_length:]               # shape [b, tgt_len]
        mask   = torch.isnan(target_slice)
        diff2  = (pred - target_slice).pow(2)
        diff2  = diff2.masked_fill(mask, 0.0)           # ignore NaNs

        sse_total += diff2.sum().item()
        n_total   += (~mask).sum().item()

        # free GPU memory early
        del xb, yb, pred, code
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # Guard against division by zero (all-NaN edge case)
    return float("nan") if n_total == 0 else sse_total / n_total



# %%
# 5. Model-builder for Optuna ------------------------------------
class LSTM_model_builder:
    def __init__(self, n_catchments, forcing_dim):
        self.n_catchments, self.forcing_dim = n_catchments, forcing_dim
    def define_model(self, trial):
        latent_dim      = 2 ** trial.suggest_int("latent_dim_pow", 1, 6)
        lstm_hidden_dim = trial.suggest_int("lstm_hidden_dim", 4, 256)
        n_lstm_layers   = trial.suggest_int("n_lstm_layers", 1, 2)
        n_fc_layers     = trial.suggest_int("n_fc_layers", 1, 2)
        p               = trial.suggest_categorical("dropout_rate", [0.0, 0.25, 0.5])

        fc_hidden = [trial.suggest_int(f"fc_dim{i}", 2, 64)
                     for i in range(n_fc_layers)]

        decoder   = LSTM_decoder(latent_dim, self.forcing_dim,
                                 lstm_hidden_dim, fc_hidden,
                                 num_lstm_layers=n_lstm_layers, p=p)
        embedding = nn.Embedding(self.n_catchments, latent_dim)
        return embedding, decoder

# %%
# 6. Optuna objective --------------------------------------------
class Objective:
    def __init__(self, model_builder, dtrain, dval,
                 epochs=500, train_year=20, patience=20, val_batch_size=100):
        self.mb, self.dtrain, self.dval = model_builder, dtrain, dval
        self.epochs, self.train_year, self.patience, self.val_batch_size = epochs, train_year, patience, val_batch_size

    def __call__(self, trial):
        emb, dec = self.mb.define_model(trial)
        emb, dec = emb.to(DEVICE), dec.to(DEVICE)

        opt_e = optim.Adam(emb.parameters(),
                           lr=trial.suggest_float("lr_e",5e-5,1e-2,log=True))
        opt_d = optim.Adam(dec.parameters(),
                           lr=trial.suggest_float("lr_d",5e-5,1e-2,log=True))

        batch_size       = 2 ** trial.suggest_int("batch_pow", 4, 8)
        steps_per_epoch  = round(self.dtrain.n_catchment * self.train_year / batch_size)
        stopper          = EarlyStopper(patience=self.patience)

        for epoch in range(self.epochs):
            emb.train(); dec.train()
            for _ in range(steps_per_epoch):
                xb, yb, idx = self.dtrain.get_random_batch(batch_size)
                xb, yb, idx = xb.to(DEVICE), yb.to(DEVICE), idx.to(DEVICE)

                opt_e.zero_grad(); opt_d.zero_grad()
                code = emb(idx)
                pred = dec.decode(code, xb)
                loss = mse_loss_with_nans(pred, yb)
                loss.backward()
                opt_e.step(); opt_d.step()

            emb.eval(); dec.eval()
            val_loss = val_model_continuous(
                emb, dec, self.dval, batch_size=self.val_batch_size)

            trial.report(val_loss, epoch)
            print(f"Epoch {epoch:4d} | val_loss = {val_loss:.5f}")
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            if stopper(val_loss):
                break
        return stopper.best


# %%
# 7. Driver -------------------------------------------------------
def main():
    print(f"Using device {DEVICE}")

    SEQ_LENGTH        = 365*2
    TARGET_SEQ_LENGTH = 365
    BASE_LENGTH       = SEQ_LENGTH - TARGET_SEQ_LENGTH
    FORCING_DIM       = 3
    N_CATCHMENTS      = 1347

    data_root = pathlib.Path("./data/")
    dtrain = Forcing_Data(str(data_root/"data_train_CAMELS_DE1.00.csv"),
                            record_length=7670, n_feature=FORCING_DIM,
                            seq_length=SEQ_LENGTH, target_seq_length=TARGET_SEQ_LENGTH,
                            base_length=BASE_LENGTH)
    dval   = Forcing_Data(str(data_root/"data_val_CAMELS_DE1.00.csv"),
                            record_length=4017, n_feature=FORCING_DIM,
                            seq_length=SEQ_LENGTH, target_seq_length=TARGET_SEQ_LENGTH,
                            base_length=BASE_LENGTH)

    builder   = LSTM_model_builder(N_CATCHMENTS, FORCING_DIM)
    objective = Objective(builder, dtrain, dval)

    study = optuna.create_study(direction="minimize",
                                pruner=optuna.pruners.NopPruner())
    study.optimize(objective, n_trials=200)

    joblib.dump(study, "data/hyperparameter_optimization_results.pkl")
    print("Study stored to hyperparameter_optimization_results.pkl")


if __name__ == "__main__":
    main()
