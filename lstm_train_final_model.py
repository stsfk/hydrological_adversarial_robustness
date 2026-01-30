"""
Train and evaluate the final catchment-embedding + LSTM decoder model for CAMELS-DE.

Loads the best Optuna hyperparameters from `data/hyperparameter_optimization_results.pkl`, trains on
the combined train+validation split (`data_train_val_CAMELS_DE1.00.csv`) for a fixed number of epochs,
evaluates per-catchment metrics on the test split (`data_test_CAMELS_DE1.00.csv`), writes
`data/lstm_test_metrics.csv`, and saves the trained embedding/decoder (state_dict + full modules).
"""

import math, random, csv, pathlib, joblib, numpy as np, torch, HydroErr as he
from torch import nn, optim
from torch.utils.data import Dataset

# ╭──────────────── device ─────────────────╮
def resolve_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = resolve_device()

# ╭──────────────── dataset ────────────────╮
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
        names_raw = np.genfromtxt(fpath, delimiter=",", skip_header=1, dtype=str, usecols=0)
        if names_raw.ndim == 0:
            names_raw = np.array([names_raw])
        self.catchment_names   = names_raw[::record_length].tolist()

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

# ╭──────────────── model ──────────────────╮
class TimeDistributed(nn.Module):
    def __init__(self, module: nn.Module, batch_first=False):
        super().__init__(); self.m, self.bf = module, batch_first
    def forward(self, x):
        if x.dim()<=2: return self.m(x)
        b,t = (x.shape[0], x.shape[1]) if self.bf else (x.shape[1], x.shape[0])
        y = self.m(x.contiguous().view(-1, x.shape[-1]))
        return y.view(b,t,-1) if self.bf else y.view(t,b,-1)

class LSTM_decoder(nn.Module):
    def __init__(self, latent_dim, feature_dim,
                 lstm_hidden_dim, fc_hidden_dims,
                 num_lstm_layers=1, p=0.2, output_dim=1, base_length=365):
        super().__init__()
        self.base_length = base_length
        self.lstm = nn.LSTM(feature_dim+latent_dim, lstm_hidden_dim,
                            num_layers=num_lstm_layers, batch_first=True)
        layers, in_dim = [], lstm_hidden_dim
        for h in fc_hidden_dims:
            layers += [nn.Linear(in_dim,h), nn.ReLU(), nn.Dropout(p)]; in_dim=h
        layers += [nn.Linear(in_dim, output_dim)]
        self.fc = TimeDistributed(nn.Sequential(*layers), batch_first=True)
    def forward(self, x):
        out,_ = self.lstm(x)
        return self.fc(out[:, self.base_length:]).squeeze(-1)
    def decode(self, code, x):
        code = code.expand(x.size(1), -1, -1).transpose(0,1)
        return self.forward(torch.cat([code,x], dim=2))

# ╭──────────────── loss & metrics ─────────╮
def mse_loss_with_nans(pred, target):
    mask = torch.isnan(target)
    return ((pred[~mask]-target[~mask])**2).mean()

@torch.no_grad()
def metrics_per_catchment(emb, dec, dataset, batch_size=256):
    emb.eval(); dec.eval()
    x_all, y_all = dataset.get_full_sequence()
    base = dataset.base_length
    n_catch = x_all.shape[0]
    rows = []
    for start in range(0, n_catch, batch_size):
        end = min(start+batch_size, n_catch)
        xb = x_all[start:end].to(DEVICE, non_blocking=True)
        yb = y_all[start:end].to(DEVICE, non_blocking=True)
        ids = torch.arange(start,end, device=DEVICE)

        pred = dec.decode(emb(ids), xb)
        tgt  = yb[:, base:]

        pred_np, tgt_np = pred.cpu().numpy(), tgt.cpu().numpy()
        for i,(p_row,t_row) in enumerate(zip(pred_np,tgt_np)):
            mse = he.mse(p_row, t_row)
            
            r, alpha, beta, kge = he.kge_2009(p_row, t_row, return_all=True)
            nse = he.nse(p_row, t_row)
            
            rows.append({
                "catchment_id": int(start + i),
                "catchment": dataset.catchment_names[start + i],
                "MSE": mse,
                "KGE_2009": kge,
                "r": r,
                "alpha": alpha,
                "beta": beta,
                "NSE": nse
            })

        del xb,yb,pred
        if DEVICE.type=="cuda": torch.cuda.empty_cache()
    return rows

# ╭──────────────── helper: build model ────╮
def build_best_model(study_pkl, n_catch, feat_dim):
    study = joblib.load(study_pkl)
    p     = study.best_trial.params
    latent_dim      = 2 ** p["latent_dim_pow"]
    lstm_hidden_dim = p["lstm_hidden_dim"]
    n_lstm_layers   = p["n_lstm_layers"]
    n_fc_layers     = p["n_fc_layers"]
    dropout_rate    = p["dropout_rate"]
    fc_sizes        = [p[f"fc_dim{i}"] for i in range(n_fc_layers)]

    emb = nn.Embedding(n_catch, latent_dim)
    dec = LSTM_decoder(latent_dim, feat_dim,
                       lstm_hidden_dim, fc_sizes,
                       num_lstm_layers=n_lstm_layers, p=dropout_rate)
    return emb, dec, p, study

# ╭──────────────── helper: optimal epochs ─╮
def get_optimal_epochs(study):
    stats = study.best_trials[0].intermediate_values
    return min(stats, key=lambda k: stats[k]) + 1  # +1 because epochs are 0-based

# ╭──────────────── training (fixed epochs) ─╮
def train_fixed_epochs(emb, dec, dataset, params,
                       epochs, train_year=30):
    bs = 2 ** params["batch_pow"]
    opt_e = optim.Adam(emb.parameters(), lr=params["lr_e"])
    opt_d = optim.Adam(dec.parameters(), lr=params["lr_d"])

    steps_per_epoch = round(dataset.n_catchment * train_year / bs)
    emb, dec = emb.to(DEVICE), dec.to(DEVICE)

    for ep in range(epochs):
        emb.train(); dec.train()
        for _ in range(steps_per_epoch):
            xb,yb,ids = dataset.get_random_batch(bs)
            xb,yb,ids = xb.to(DEVICE), yb.to(DEVICE), ids.to(DEVICE)

            opt_e.zero_grad(); opt_d.zero_grad()
            loss = mse_loss_with_nans(dec.decode(emb(ids), xb), yb)
            loss.backward()
            opt_e.step(); opt_d.step()

        print(f"[epoch {ep+1:3d}/{epochs}]  last mini-batch loss {loss.item():.6f}")

    return emb.cpu(), dec.cpu()

# ╭──────────────── main ───────────────────╮

if __name__ == "__main__":
    print("Using", DEVICE)
    random.seed(12345); np.random.seed(12345); torch.manual_seed(12345)
    
    SEQ_LEN = 365*2;  TGT_LEN = 365;  BASE_LEN = SEQ_LEN - TGT_LEN
    FEAT_DIM = 3;     N_CATCH = 1347
    
    root = pathlib.Path("data/")
    study_pkl = root / "hyperparameter_optimization_results.pkl"
    
    d_train_val = Forcing_Data(str(root/"data_train_val_CAMELS_DE1.00.csv"),
                               record_length=11322, n_feature=FEAT_DIM,
                               seq_length=SEQ_LEN, target_seq_length=TGT_LEN,
                               base_length=BASE_LEN)
    
    d_test = Forcing_Data(str(root/"data_test_CAMELS_DE1.00.csv"),
                          record_length=4018, n_feature=FEAT_DIM,
                          seq_length=SEQ_LEN, target_seq_length=TGT_LEN,
                          base_length=BASE_LEN)
    
    emb, dec, best_hp, study = build_best_model(str(study_pkl),
                                                N_CATCH, FEAT_DIM)
    
    # ── scale epochs (20→30 yrs ⇒ × 2/3) ───────────────────────────────────
    opt_epochs_original = get_optimal_epochs(study)
    opt_epochs = max(1, round(opt_epochs_original * 2/3))
    print(f"Best epochs during tuning  : {opt_epochs_original}")
    print(f"Scaled epochs for 30 years : {opt_epochs}")
    print("Best hyper-parameters:", best_hp)
    
    # training
    emb, dec = train_fixed_epochs(emb, dec, d_train_val,
                                  best_hp, epochs=opt_epochs)
    
    # per-catchment metrics
    rows = metrics_per_catchment(emb.to(DEVICE), dec.to(DEVICE), d_test)
    
    # CSV
    with open("data/lstm_test_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["catchment_id","catchment","MSE","KGE_2009","r","alpha","beta","NSE"]
        )
        writer.writeheader(); writer.writerows(rows)
    print("Saved metrics → lstm_test_metrics.csv")
    
    # median KGE and histogram
    kges = np.array([r["KGE_2009"] for r in rows])
    median_kge = float(np.nanmedian(kges))
    print(f"Median KGE-2009 on test set = {median_kge:.4f}")
    
    # ── save models ────────────────────────────────────────────────────────
    torch.save(emb.state_dict(), "data/embedding_state.pt")
    torch.save(dec.state_dict(), "data/decoder_state.pt")
    torch.save(emb,              "data/embedding_full.pt")   # <-- entire Module
    torch.save(dec,              "data/decoder_full.pt")     # <-- entire Module
    print("Saved state-dicts (embedding_state.pt, decoder_state.pt)")
    print("Saved full models (embedding_full.pt,  decoder_full.pt)")
