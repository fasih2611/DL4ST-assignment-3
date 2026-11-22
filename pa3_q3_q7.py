"""AI651 PA3, Questions 3 to 7. LargeST San Diego, 716 sensors, one long night.

    python pa3_q3_q7.py                  # load the trained checkpoints from ./, redo every figure and number
    python pa3_q3_q7.py --retrain        # train all four models again first (about 13 min on an RTX 5090)

Data goes in data/: either the Kaggle files (ca_his_raw_2020.h5, ca_his_raw_2021.h5) or the San Diego extracts
written by fetch_sd.py (sd_his_5min_2020.npz, sd_his_5min_2021.npz), plus ca_meta.csv and ca_rn_adj.npy in both cases.
"""
import argparse
import copy
import json
import os
import platform
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
import matplotlib.dates as mdates

warnings.filterwarnings("ignore", category=RuntimeWarning)    # nanmean of a dead sensor complains. It is allowed to.

ROOT = Path(__file__).resolve().parent
RAW = PROC = ROOT / "data"                                   # inputs and cached arrays; everything else lands beside this file
OUT = FIG = ROOT
for _p in (PROC, OUT, FIG):
    _p.mkdir(parents=True, exist_ok=True)

SEED, T_IN, T_OUT = 42, 8, 4
SPLITS = {"train": ("2020-01-01", "2020-12-31"), "val": ("2021-01-01", "2021-03-31"), "test": ("2021-04-01", "2021-12-31")}
CFG = dict(seed=SEED, d=32, batch_size=64, lr=1e-3, weight_decay=0.0, max_epochs=50, patience=8, d_e=10)
I_STAR = 67
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WORKERS = 0 if os.name == "nt" else 4                         # Windows and worker processes have an understanding: none.

# What the original run (RTX 5090) took: seconds, epochs run, best epoch. Used unless --retrain measures it afresh.
TRAIN_LOG = {"Temporal-only": (165, 46, 38), "STGNN-fixed": (194, 50, 49), "STGNN-adaptive": (218, 50, 43), "STGNN-hybrid": (204, 50, 43)}


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def haversine_km(lat1, lng1, lat2, lng2):
    p1, p2, dl = np.radians(lat1), np.radians(lat2), np.radians(lng2 - lng1)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


# ==================================================================================================== Q3(a)
def load_sd_5min(sd_ids):
    """Both years, SD columns only, in metadata order. Seven gigabytes in, a fraction of one out."""
    frames = []
    for y in (2020, 2021):
        h5, npz = RAW / f"ca_his_raw_{y}.h5", RAW / f"sd_his_5min_{y}.npz"
        if h5.exists():
            with pd.HDFStore(h5, "r") as s:
                frames.append(s["/t"].loc[:, sd_ids])
        else:
            z = np.load(npz)
            assert list(z["ids"].astype(str)) == sd_ids, "column order drifted from the metadata"
            frames.append(pd.DataFrame(z["values"], index=pd.DatetimeIndex(z["times"]), columns=sd_ids))
    his5 = pd.concat(frames)
    assert his5.index.is_monotonic_increasing and his5.index.is_unique
    return his5


def q3a(meta, sd_meta, sd_ids, R):
    his5 = load_sd_5min(sd_ids)
    v5 = his5.to_numpy()
    obs = np.isfinite(v5)
    R["q3a"] = dict(rows_5min=len(his5), zeros_frac=float((v5[obs] == 0).mean()),
                    sensors_gt5pct_zeros=int((np.nanmean(np.where(obs, v5 == 0, np.nan), 0) > .05).sum()),
                    capped_999=int((v5 == 999).sum()), capped_sensors=int((v5 == 999).any(0).sum()))
    del v5, obs

    # PeMS counts vehicles per 5 minutes. The mean keeps that unit and survives a missing value; the sum would not.
    his15 = his5.resample("15min").mean()
    del his5
    mask = his15.notna().to_numpy()
    miss = ~mask
    sensor_miss = miss.mean(0)
    ts_miss = pd.Series(miss.mean(1), index=his15.index)
    R["q3a"].update(rows_15min=len(his15), missing=float(miss.mean()), missing_count=int(miss.sum()),
                    missing_by_split={k: float(ts_miss.loc[s:e].mean()) for k, (s, e) in SPLITS.items()},
                    never_observed=int((sensor_miss == 1).sum()),
                    dead_2020=int(np.isnan(his15.loc["2020"].to_numpy()).all(0).sum()),
                    dead_2021=int(np.isnan(his15.loc["2021"].to_numpy()).all(0).sum()),
                    gt50pct=int((sensor_miss > .5).sum()), lt5pct=int((sensor_miss < .05).sum()), gap_free=int((sensor_miss == 0).sum()),
                    all_missing_ts=int(miss.all(1).sum()), all_missing_days=int(ts_miss[miss.all(1)].index.normalize().nunique()))

    day_miss = ts_miss.resample("D").mean()
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6), constrained_layout=True, gridspec_kw={"width_ratios": [1, 2.2]})
    ax[0].hist(sensor_miss * 100, bins=50, color="tab:blue"); ax[0].set_yscale("log")
    ax[0].set_xlabel("missing (% of timestamps)"); ax[0].set_ylabel("number of sensors"); ax[0].set_title("Per-sensor missing fraction")
    ax[1].plot(day_miss.index, day_miss * 100, lw=0.8, color="tab:red")
    for d, lab in [("2021-01-01", "val"), ("2021-04-01", "test")]:
        ax[1].axvline(pd.Timestamp(d), color="k", ls="--", lw=0.8)
        ax[1].text(pd.Timestamp(d), 0.97, f" {lab}", transform=ax[1].get_xaxis_transform(), va="top", fontsize=8)
    ax[1].set_ylim(0, max(10, 115 * day_miss.max())); ax[1].set_ylabel("sensors missing (%)"); ax[1].set_title("Daily fraction of sensors missing")
    fig.savefig(FIG / "q3_missing.pdf"); plt.close(fig)

    # The quietest complete week of 2020, then a busy sensor, a quiet one, and one that blinks.
    wk = ts_miss.loc["2020"].resample("W-MON", label="left", closed="left")
    weekly = wk.mean()[wk.size() == 7 * 96]
    w0 = weekly.idxmin()
    wdf = his15.loc[w0:w0 + pd.Timedelta(days=7) - pd.Timedelta(minutes=15)]
    wmiss = wdf.isna().mean()
    wmean = wdf[wmiss[wmiss == 0].index].mean()
    partial = wmiss[(wmiss > 0) & (wmiss < 0.5)]
    picks = [("busy", wmean.idxmax()), ("quiet", wmean.idxmin())] + ([("with gap", partial.idxmax())] if len(partial) else [])
    info = sd_meta.set_index(sd_meta["ID"].astype(str))
    fig, axes = plt.subplots(len(picks), 1, figsize=(11, 2.4 * len(picks)), sharex=True, constrained_layout=True)
    for ax, (role, sid) in zip(np.atleast_1d(axes), picks):
        s, r = wdf[sid], info.loc[sid]
        ax.plot(s.index, s.values, lw=0.9)
        for t in s.index[s.isna()]:
            ax.axvspan(t, t + pd.Timedelta(minutes=15), color="tab:red", alpha=0.25, lw=0)
        ax.set_title(f"{role}: sensor {sid} ({r.Fwy}, {r.Lanes} lanes, {r.County}), weekly mean {s.mean():.0f}, "
                     f"missing {s.isna().mean():.1%}", fontsize=9, loc="left")
        ax.set_ylabel("veh / 5 min\n(15-min mean)"); ax.grid(alpha=0.3)
    np.atleast_1d(axes)[-1].set_xlabel("time")
    fig.savefig(FIG / "q3_week.pdf"); plt.close(fig)
    R["q3a"].update(week=str(w0.date()), week_missing=float(weekly.min()), week_sensors=dict(picks))

    X = his15.to_numpy(dtype=np.float32)[..., None]            # NaN stays NaN. Pretending otherwise comes later, and only for inputs.
    np.savez_compressed(PROC / "sd_15min.npz", X=X, mask=mask, times=his15.index.values.astype("datetime64[m]"),
                        ids=np.array(sd_ids), rows=sd_meta.index.to_numpy())
    print("Q3(a):", json.dumps(R["q3a"]))


# ==================================================================================================== Q3(b), Q3(c)
def q3bc(sd_meta, sd_rows, A_sd, R):
    N = len(A_sd)
    off = ~np.eye(N, dtype=bool)
    nz = A_sd != 0
    out_deg, in_deg = (nz & off).sum(1), (nz & off).sum(0)
    R["q3b"] = dict(nodes=N, nnz_total=int(nz.sum()), diag_nonzero=int(np.diag(nz).sum()),
                    diag_values=sorted(set(np.diag(A_sd).round(6).tolist())), nnz_offdiag=int((nz & off).sum()),
                    undirected_pairs=int(((nz | nz.T) & off).sum() // 2), symmetric=bool(np.allclose(A_sd, A_sd.T)),
                    struct_asym_entries=int((nz != nz.T).sum()), weight_asym_entries=int(((A_sd != A_sd.T) & off).sum()),
                    weight_min=float(A_sd[nz & off].min()), weight_median=float(np.median(A_sd[nz & off])),
                    out_deg_mean=float(out_deg.mean()), out_deg_median=float(np.median(out_deg)), out_deg_max=int(out_deg.max()),
                    in_deg_max=int(in_deg.max()), no_out_edges=int((out_deg == 0).sum()),
                    isolated=int(((out_deg == 0) & (in_deg == 0)).sum()))
    print("Q3(b):", json.dumps(R["q3b"]))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(A_sd, cmap="Blues", interpolation="nearest")
    fig.colorbar(im, label="adjacency weight"); ax.set_title("SD adjacency matrix heatmap")
    ax.set_xlabel("sensor index"); ax.set_ylabel("sensor index")
    fig.savefig(FIG / "q3_adj.pdf"); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4), constrained_layout=True)
    bins = np.arange(0, max(out_deg.max(), in_deg.max()) + 2) - 0.5
    ax[0].hist(out_deg, bins=bins, color="tab:blue"); ax[0].set_title("Out-degree (non-zeros per row, no self-loop)")
    ax[1].hist(in_deg, bins=bins, color="tab:orange"); ax[1].set_title("In-degree (non-zeros per column, no self-loop)")
    for a in ax:
        a.set_xlabel("number of neighbours"); a.set_ylabel("number of sensors"); a.grid(alpha=.3)
    fig.savefig(FIG / "q3_degree.pdf"); plt.close(fig)

    # Out-edges, i.e. rows: where traffic leaving sensor 67 goes next.
    M = (nz & off).astype(int)
    one_hop = np.flatnonzero(M[I_STAR])
    reach2 = np.flatnonzero((M + M @ M)[I_STAR]); reach2 = reach2[reach2 != I_STAR]
    two_hop = np.setdiff1d(reach2, one_hop)
    r = sd_meta.iloc[I_STAR]
    R["q3c"] = dict(index=I_STAR, id=str(r.ID), fwy=r.Fwy, one_hop=one_hop.tolist(), two_hop=two_hop.tolist(),
                    one_hop_fwys=sd_meta.iloc[one_hop]["Fwy"].value_counts().to_dict(),
                    two_hop_fwys=sd_meta.iloc[two_hop]["Fwy"].value_counts().to_dict())
    print("Q3(c):", R["q3c"])
    lat, lng = sd_meta["Lat"].to_numpy(), sd_meta["Lng"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 5.6), constrained_layout=True)
    ax.scatter(lng, lat, s=4, c="lightgray", label="other sensors")
    ax.scatter(lng[two_hop], lat[two_hop], s=22, c="tab:orange", label=f"2-hop ({len(two_hop)})")
    ax.scatter(lng[one_hop], lat[one_hop], s=22, c="tab:blue", label=f"1-hop ({len(one_hop)})")
    ax.scatter(lng[I_STAR], lat[I_STAR], s=90, c="red", marker="*", label=f"sensor {I_STAR} ({r.Fwy})")
    sel = np.r_[one_hop, two_hop, I_STAR]
    ax.set_xlim(lng[sel].min() - .06, lng[sel].max() + .06); ax.set_ylim(lat[sel].min() - .06, lat[sel].max() + .06)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.legend(fontsize=8); ax.set_title(f"Out-neighbourhood of sensor {I_STAR}")
    fig.savefig(FIG / "q3_neigh.pdf"); plt.close(fig)


# ==================================================================================================== Q4(a)
class WindowDataset(torch.utils.data.Dataset):
    """Item k is the window ending at t = k + 7: eight inputs up to t, four targets after it. Stored once, sliced on demand."""
    def __init__(self, x_norm, y_raw, m):
        self.x, self.y, self.m = x_norm, y_raw, m
    def __len__(self):
        return len(self.x) - T_IN - T_OUT + 1
    def __getitem__(self, k):
        return self.x[k:k + T_IN], self.y[k + T_IN:k + T_IN + T_OUT], self.m[k + T_IN:k + T_IN + T_OUT]


def build_splits(X, mask, times):
    """Cut by date first, window second, so no window ever straddles two splits."""
    blocks = {}
    for k, (s, e) in SPLITS.items():
        sel = (times >= pd.Timestamp(s)) & (times < pd.Timestamp(e) + pd.Timedelta(days=1))
        blocks[k] = (X[sel], mask[sel], times[sel])
    xtr = blocks["train"][0][..., 0]
    mu, sigma = np.nanmean(xtr, 0), np.nanstd(xtr, 0)          # 2020, observed entries, nothing else
    dead = ~np.isfinite(mu)
    mu[dead], sigma[dead] = np.nanmean(xtr), np.nanstd(xtr)     # sensors with no 2020 at all borrow the crowd's numbers
    sigma[sigma < 1e-6] = 1.0
    mu, sigma = mu.astype(np.float32), sigma.astype(np.float32)
    ds = {}
    for k, (x, m, _) in blocks.items():
        xn = np.nan_to_num((x - mu[None, :, None]) / sigma[None, :, None], nan=0.0).astype(np.float32)
        ds[k] = WindowDataset(xn, x[..., 0], m)                 # targets keep their NaNs; the mask deals with them
    return blocks, ds, mu, sigma, int(dead.sum())


# ==================================================================================================== Q4(b), Q5
class TemporalEncoder(nn.Module):
    """Same two convolutions for every sensor, then the remaining four steps are flattened into one vector."""
    def __init__(self, t_in=T_IN, d=32):
        super().__init__()
        self.convs = nn.Sequential(nn.Conv1d(1, d, 3), nn.ReLU(), nn.Conv1d(d, d, 3), nn.ReLU())
        self.proj = nn.Linear(d * (t_in - 4), d)
    def forward(self, x):                                       # (B, T, N, 1) -> (B, N, d)
        B, T, N, _ = x.shape
        z = self.convs(x.permute(0, 2, 3, 1).reshape(B * N, 1, T)).flatten(1)
        return torch.relu(self.proj(z)).reshape(B, N, -1)


class GraphConv(nn.Module):
    """H = Z + ReLU(A Z W). Without the Z in front a sensor keeps a seventh of itself and listens to the rest;
    test MAE 31.0 that way, worse than copying the last value. The skip lets it remember who it is."""
    def __init__(self, d=32):
        super().__init__()
        self.W = nn.Linear(d, d, bias=False)
    def forward(self, z, a_hat):
        return z + torch.relu(torch.einsum("nm,bmd->bnd", a_hat, self.W(z)))


class Head(nn.Module):
    def __init__(self, d=32, t_out=T_OUT):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, t_out))
    def forward(self, h):                                       # (B, N, d) -> (B, T_out, N)
        return self.mlp(h).permute(0, 2, 1)


class STGNN(nn.Module):
    """graph=None is the temporal-only baseline: same encoder, same head, no neighbours to consult."""
    def __init__(self, graph=None, d=32):
        super().__init__()
        self.graph, self.encoder, self.head = graph, TemporalEncoder(T_IN, d), Head(d, T_OUT)
        self.gconvs = nn.ModuleList([GraphConv(d)]) if graph is not None else nn.ModuleList()
    def forward(self, x):
        z = self.encoder(x)
        if self.graph is not None:
            a_hat = self.graph()                                # asked afresh every step; the adaptive one keeps changing its mind
            for gc in self.gconvs:
                z = gc(z, a_hat)
        return self.head(z)


def normalize_adj(A, kind):
    """Diagonal set to 1, not incremented: the file already has its self-loops and does not need a second helping."""
    A = np.array(A, dtype=np.float64, copy=True)
    np.fill_diagonal(A, 1.0)
    if kind == "sym":
        d = A.sum(1)
        return (A / np.sqrt(np.outer(d, d))).astype(np.float32)
    if kind == "row":                                           # D_out^-1 A: direction kept, rows sum to one
        return (A / A.sum(1, keepdims=True)).astype(np.float32)
    raise ValueError(kind)


class FixedGraph(nn.Module):
    def __init__(self, A_hat):
        super().__init__()
        self.register_buffer("A_hat", torch.as_tensor(A_hat, dtype=torch.float32))   # a buffer: it travels, it never learns
    def forward(self):
        return self.A_hat


class AdaptiveGraph(nn.Module):
    """softmax_row(ReLU(E1 E2^T)). Two embeddings, so i may listen to j without j returning the favour.
    topk, if given, masks the losing logits to -inf before the softmax: rows still sum to one and the survivors get
    the whole gradient. Not used in the reported runs; 716 x 716 is not big enough to need it."""
    def __init__(self, n, d_e=10, topk=None):
        super().__init__()
        self.E1, self.E2 = nn.Parameter(torch.randn(n, d_e)), nn.Parameter(torch.randn(n, d_e))
        self.topk = topk
    def forward(self):
        logits = torch.relu(self.E1 @ self.E2.T)
        if self.topk is not None:
            keep = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, logits.topk(self.topk, 1).indices, True)
            logits = logits.masked_fill(~keep, float("-inf"))
        return torch.softmax(logits, 1)


class HybridGraph(nn.Module):
    """alpha * road + (1 - alpha) * learned. Both row-stochastic, so the blend is too. alpha = sigmoid(a), learned."""
    def __init__(self, A_fixed_hat, adaptive, alpha=0.5, learn_alpha=True):
        super().__init__()
        self.register_buffer("A_fixed", torch.as_tensor(A_fixed_hat, dtype=torch.float32))
        self.adaptive = adaptive
        a0 = torch.logit(torch.tensor(float(alpha)))
        if learn_alpha:
            self.a = nn.Parameter(a0)
        else:
            self.register_buffer("a", a0)
    @property
    def alpha(self):
        return torch.sigmoid(self.a)
    def forward(self):
        return self.alpha * self.A_fixed + (1 - self.alpha) * self.adaptive()


# ==================================================================================================== training
class Trainer:
    def __init__(self, ds, mu, sigma):
        self.ds = ds
        self.mu, self.sigma = torch.as_tensor(mu, device=DEVICE), torch.as_tensor(sigma, device=DEVICE)

    def loader(self, split, shuffle=False, bs=None):
        g = torch.Generator().manual_seed(SEED)                 # every model sees the batches in the same order
        return torch.utils.data.DataLoader(self.ds[split], batch_size=bs or CFG["batch_size"], shuffle=shuffle,
                                           generator=g, num_workers=WORKERS, pin_memory=DEVICE.type == "cuda")

    @torch.no_grad()
    def evaluate(self, model, split, keep_nodes=None):
        """Original units, observed targets only. Indexing with the mask, never multiplying: NaN * 0 is still NaN."""
        model.eval()
        N = self.ds[split].y.shape[1]
        a = torch.zeros(T_OUT, N, device=DEVICE, dtype=torch.float64); s = torch.zeros_like(a); c = torch.zeros_like(a)
        kept = []
        for x, y, m in self.loader(split, bs=256):
            x, y, m = x.to(DEVICE), y.to(DEVICE), m.to(DEVICE)
            p = model(x) * self.sigma + self.mu
            e = torch.where(m, p - y, torch.zeros_like(p)).double()
            a += e.abs().sum(0); s += (e ** 2).sum(0); c += m.sum(0)
            if keep_nodes is not None:
                kept.append(p[:, :, keep_nodes].cpu().numpy())
        return dict(mae=(a.sum() / c.sum()).item(), rmse=(s.sum() / c.sum()).sqrt().item(),
                    mae_h=(a.sum(1) / c.sum(1)).tolist(), rmse_h=(s.sum(1) / c.sum(1)).sqrt().tolist(),
                    mae_node=(a.sum(0) / c.sum(0).clamp(min=1)).cpu().numpy(), n_node=c.sum(0).cpu().numpy(),
                    pred=np.concatenate(kept) if kept else None)

    def train(self, model):
        """Adam on masked MAE in z-units; early stopping on validation MAE; the best epoch is the one that gets kept."""
        model.to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
        best, best_state, bad, hist = float("inf"), None, 0, []
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        for ep in range(1, CFG["max_epochs"] + 1):
            model.train(); losses = []
            for x, y, m in self.loader("train", shuffle=True):
                x, y, m = x.to(DEVICE), y.to(DEVICE), m.to(DEVICE)
                loss = (model(x) - (y - self.mu) / self.sigma).abs()[m].mean()
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                losses.append(loss.item())
            v = self.evaluate(model, "val")["mae"]
            hist.append(v)
            if v < best:
                best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
            print(f"  epoch {ep:3d}  train {np.mean(losses):.4f}  val MAE {v:.3f}{'  *' if bad == 0 else ''}", flush=True)
            if bad >= CFG["patience"]:
                break
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        model.load_state_dict(best_state)
        return time.perf_counter() - t0, len(hist), int(np.argmin(hist) + 1)


class _LastValue(nn.Module):
    def forward(self, x):                                       # tomorrow will be like today, only more so
        return x[:, -1:, :, 0].expand(-1, T_OUT, -1)


# ==================================================================================================== main
def main(retrain=False):
    set_seed()
    R = {}
    meta = pd.read_csv(RAW / "ca_meta.csv")
    sd_meta = meta[meta["District"] == 11]                      # Caltrans District 11: San Diego, as LargeST defines it
    sd_rows, sd_ids = sd_meta.index.to_numpy(), sd_meta["ID"].astype(str).to_list()
    assert len(sd_ids) == 716 and (np.diff(sd_rows) > 0).all()

    if retrain or not (PROC / "sd_15min.npz").exists():
        q3a(meta, sd_meta, sd_ids, R)
    else:
        print("Q3(a): using data/sd_15min.npz (delete it to redo the extraction and its figures)")

    A_ca = np.load(RAW / "ca_rn_adj.npy", mmap_mode="r")
    A_sd = np.asarray(A_ca[np.ix_(sd_rows, sd_rows)])           # same positions for rows, columns and traffic columns
    np.save(PROC / "sd_adj.npy", A_sd)
    q3bc(sd_meta, sd_rows, A_sd, R)

    d = np.load(PROC / "sd_15min.npz")
    X, MASK, TIMES = d["X"], d["mask"], pd.DatetimeIndex(d["times"])
    N = X.shape[1]
    blocks, ds, mu, sigma, n_dead = build_splits(X, MASK, TIMES)
    R["q4a"] = dict(no_train_data_sensors=n_dead, **{k: dict(rows=len(blocks[k][0]), windows=len(ds[k]),
                    missing=float(1 - blocks[k][1].mean()), mean_flow=float(np.nanmean(blocks[k][0]))) for k in SPLITS})
    print("Q4(a):", R["q4a"])
    tr = Trainer(ds, mu, sigma)

    A_FIXED = normalize_adj(A_sd, "row")                        # directed graph, so no D^-1/2 A D^-1/2 here
    specs = {"Temporal-only": ("None", lambda: None),
             "STGNN-fixed": ("Fixed", lambda: FixedGraph(A_FIXED)),
             "STGNN-adaptive": ("Adaptive", lambda: AdaptiveGraph(N, CFG["d_e"])),
             "STGNN-hybrid": ("Hybrid", lambda: HybridGraph(A_FIXED, AdaptiveGraph(N, CFG["d_e"]), 0.5, True))}
    models, train_info = {}, {}
    for name, (adj, make_graph) in specs.items():
        set_seed(); graph = make_graph()                        # same seed before graph and model: only the graph differs
        set_seed(); model = STGNN(graph, CFG["d"])
        ckpt = OUT / f"{name}.pt"
        if retrain or not ckpt.exists():
            print(f"== training {name}")
            train_info[name] = tr.train(model)
            torch.save(model.state_dict(), ckpt)
        else:
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            train_info[name] = TRAIN_LOG[name]
        models[name] = model.to(DEVICE).eval()

    # ------------------------------------------------------------------------------------------ Q4(c), Q6(a)
    q3_sensors = {"busy": "1120611", "quiet": "1125610", "with gap": "1122479"}
    plot_nodes = [sd_ids.index(s) for s in q3_sensors.values()]
    EV = {}
    for name, m in list(models.items()) + [("Last value", _LastValue())]:
        EV[name] = {k: tr.evaluate(m, k, plot_nodes if k == "test" else None) for k in ("val", "test")}
        e = EV[name]
        print(f"{name:15s} val {e['val']['mae']:.3f}/{e['val']['rmse']:.3f}  test {e['test']['mae']:.3f}/{e['test']['rmse']:.3f}  "
              f"MAE/h {' '.join(f'{v:.2f}' for v in e['test']['mae_h'])}  RMSE/h {' '.join(f'{v:.2f}' for v in e['test']['rmse_h'])}")

    # Historical average: the 2020 mean for the same weekday and quarter-hour. The laziest forecast with a calendar.
    slot = lambda t: (t.dayofweek * 96 + t.hour * 4 + t.minute // 15).to_numpy()
    tx, tm, tt = blocks["train"]
    ha, cnt = np.zeros((7 * 96, N)), np.zeros((7 * 96, N))
    np.add.at(ha, slot(tt), np.nan_to_num(tx[..., 0])); np.add.at(cnt, slot(tt), tm)
    ha = np.where(cnt > 0, ha / np.maximum(cnt, 1), mu[None, :])
    res = {}
    for k in ("val", "test"):
        x, m, t = blocks[k]
        err = (ha[slot(t)] - x[..., 0])[m]
        res[k] = dict(mae=float(np.abs(err).mean()), rmse=float(np.sqrt((err ** 2).mean())))
    EV["Historical avg"] = res
    print("Historical avg", res)

    off = ~np.eye(N, dtype=bool); nz = A_sd != 0
    out_deg = (nz & off).sum(1)
    mt, mf = EV["Temporal-only"]["test"]["mae_node"], EV["STGNN-fixed"]["test"]["mae_node"]
    ok = EV["Temporal-only"]["test"]["n_node"].sum(0) > 0
    gain = (mt - mf)[ok]
    R["q4c"] = dict(sensors_improved=float((gain > 0).mean()), median_gain=float(np.median(gain)),
                    median_gain_no_out_edges=float(np.median((mt - mf)[ok & (out_deg == 0)])),
                    corr_gain_outdegree=float(np.corrcoef(gain, out_deg[ok])[0, 1]))
    print("Q4(c):", R["q4c"])

    test_t = blocks["test"][2]
    ts_miss = pd.Series(1 - blocks["test"][1].mean(1), index=test_t)
    wk = ts_miss.resample("W-MON", label="left", closed="left")
    w0 = wk.mean()[wk.size() == 7 * 96].idxmin(); w1 = w0 + pd.Timedelta(days=7)
    colors = {"Temporal-only": "tab:gray", "STGNN-fixed": "tab:blue", "STGNN-adaptive": "tab:orange", "STGNN-hybrid": "tab:green"}
    fig, axes = plt.subplots(3, 2, figsize=(14, 8.4), sharex=True, constrained_layout=True)
    for row, ((role, sid), ni) in enumerate(zip(q3_sensors.items(), plot_nodes)):
        actual = pd.Series(blocks["test"][0][:, ni, 0], index=test_t).loc[w0:w1]
        for col, h in enumerate((0, 3)):
            ax = axes[row, col]
            ax.plot(actual.index, actual.values, c="k", lw=1.3, label="actual")
            for name, cc in colors.items():
                p = EV[name]["test"]["pred"][:, h, row]
                when = test_t[T_IN + h: T_IN + h + len(p)]      # a forecast made at t for step h+1 belongs at t+h+1
                ps = pd.Series(p, index=when).loc[w0:w1]
                ax.plot(ps.index, ps.values, c=cc, lw=0.8, alpha=0.9, label=name)
            ax.set_title(f"{role}: sensor {sid}, {(h + 1) * 15}-min-ahead forecast", fontsize=9, loc="left"); ax.grid(alpha=.3)
            if col == 0:
                ax.set_ylabel("veh / 5 min (15-min mean)")
    axes[0, 0].legend(fontsize=7, ncol=5, loc="upper left")
    for ax in axes[-1]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    fig.savefig(FIG / "q6_pred.pdf"); plt.close(fig)
    R["q6a"] = dict(week=str(w0.date()), week_missing=float(wk.mean()[wk.size() == 7 * 96].min()),
                    sensor_mae={name: {sid: float(EV[name]["test"]["mae_node"][ni]) for sid, ni in zip(q3_sensors.values(), plot_nodes)}
                                for name in colors})

    # ------------------------------------------------------------------------------------------ Q5 figure, Q6(b)
    with torch.no_grad():
        A_ad = models["STGNN-adaptive"].graph().cpu().numpy()
        A_hy = models["STGNN-hybrid"].graph.adaptive().cpu().numpy()
        alpha = models["STGNN-hybrid"].graph.alpha.item()
    np.save(OUT / "A_adaptive.npy", A_ad); np.save(OUT / "A_hybrid_adaptive_part.npy", A_hy)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for ax, Mx, title in [(axes[0], A_FIXED, "Fixed (row-normalised road graph)"), (axes[1], A_ad, "Learned: adaptive model"),
                          (axes[2], A_hy, f"Learned: adaptive part of hybrid (alpha={alpha:.3f})")]:
        im = ax.imshow(np.clip(Mx, 1e-5, None), cmap="viridis", norm=LogNorm(1e-5, 1), interpolation="nearest")
        ax.set_title(title, fontsize=10); ax.set_xlabel("sensor j (source)"); ax.set_ylabel("sensor i (receives)")
    fig.colorbar(im, ax=axes, shrink=0.85, label="weight A[i, j] (log scale)")
    fig.savefig(FIG / "q5_adjacency.pdf"); plt.close(fig)

    lat, lng = sd_meta["Lat"].to_numpy(), sd_meta["Lng"].to_numpy()
    fwy = sd_meta["Fwy"].to_numpy()
    D = haversine_km(lat[:, None], lng[:, None], lat[None], lng[None])
    M1 = (nz & off).astype(int)
    und = (nz | nz.T) & off
    M2 = ((M1 + M1 @ M1) > 0) & off
    xtr = blocks["train"][0][..., 0]
    C = np.corrcoef(np.where(np.isnan(xtr), mu[None], xtr).T)   # 2020 only; the graph is judged on what it was allowed to see
    C[~np.isfinite(C)] = 0

    def topk_pairs(A, k):
        A = np.where(off, A, -np.inf)
        return np.repeat(np.arange(N), k), np.argsort(-A, 1)[:, :k].ravel()

    rng = np.random.default_rng(SEED)
    ri, rj = rng.integers(0, N, 20000), rng.integers(0, N, 20000); keep = ri != rj
    pairs = {"road graph edges": np.nonzero(nz & off), "random pairs": (ri[keep], rj[keep]), "learned top-10": topk_pairs(A_ad, 10)}
    Q6B = {"hybrid_alpha": alpha}
    for gname, A in [("adaptive", A_ad), ("hybrid-adaptive", A_hy)]:
        ent = -(A * np.log(A + 1e-12)).sum(1); col = A.sum(0)
        st = dict(median_row_max=float(np.median(A.max(1))), median_entropy=float(np.median(ent)), uniform_entropy=float(np.log(N)),
                  median_self_weight=float(np.median(np.diag(A))), median_weights_gt_001=float(np.median((A > 0.01).sum(1))),
                  max_col_sum=float(col.max()), top10_col_share=float(np.sort(col)[::-1][:10].sum() / N),
                  sensors_in_anyones_top10=int(len(np.unique(topk_pairs(A, 10)[1]))))
        for k in (1, 5, 10, 20):
            i, j = topk_pairs(A, k)
            st[f"top{k}"] = dict(in_road_out=float(nz[i, j].mean()), in_road_either=float(und[i, j].mean()),
                                 within_2hop=float(M2[i, j].mean()), km_median=float(np.median(D[i, j])),
                                 same_fwy=float((fwy[i] == fwy[j]).mean()), corr_median=float(np.median(C[i, j])))
        Ao = np.where(off, A, -np.inf); gi, gj = np.nonzero(Ao >= np.quantile(A[off], 0.999))
        st["strongest_0.1pct"] = dict(edges=int(len(gi)), in_road_either=float(und[gi, gj].mean()), km_median=float(np.median(D[gi, gj])))
        Q6B[gname] = st
    Ao = np.where(off & ~und, A_ad, -1).ravel()
    Q6B["strongest_absent_from_road"] = [dict(i=int(f // N), j=int(f % N), w=float(Ao[f]), km=float(D.ravel()[f]),
                                              fwy_i=str(fwy[f // N]), fwy_j=str(fwy[f % N]), corr=float(C.ravel()[f]))
                                         for f in np.argsort(-Ao)[:8]]
    Q6B["reference"] = dict(chance_in_road_out=float((nz & off).sum() / (N * (N - 1))), chance_in_road_either=float(und.sum() / (N * (N - 1))),
                            chance_within_2hop=float(M2.sum() / (N * (N - 1))),
                            **{nm: dict(km_median=float(np.median(D[i, j])), same_fwy=float((fwy[i] == fwy[j]).mean()),
                                        corr_median=float(np.median(C[i, j]))) for nm, (i, j) in pairs.items()})
    R["q6b"] = Q6B
    print("Q6(b):", json.dumps(Q6B, indent=1))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True, gridspec_kw={"width_ratios": [1.15, 1, 1]})
    ri_, rj_ = pairs["road graph edges"]
    ti, tj = topk_pairs(A_ad, 1)
    ax[0].add_collection(LineCollection(np.stack([np.c_[lng[ti], lat[ti]], np.c_[lng[tj], lat[tj]]], 1), colors="tab:red", lw=0.5, alpha=0.4,
                                        label="learned top-1 per sensor"))
    ax[0].add_collection(LineCollection(np.stack([np.c_[lng[ri_], lat[ri_]], np.c_[lng[rj_], lat[rj_]]], 1), colors="tab:blue", lw=1.2,
                                        label="road graph", zorder=2))
    ax[0].scatter(lng, lat, s=2, c="k", zorder=3); ax[0].autoscale(); ax[0].legend(fontsize=8, loc="lower left")
    ax[0].set_title("Road edges vs strongest learned edge of each sensor"); ax[0].set_xlabel("longitude"); ax[0].set_ylabel("latitude")
    for nm, cc in [("road graph edges", "tab:blue"), ("learned top-10", "tab:red"), ("random pairs", "tab:gray")]:
        i, j = pairs[nm]
        ax[1].hist(D[i, j], bins=np.linspace(0, D.max(), 50), density=True, histtype="step", lw=1.5, color=cc, label=nm)
        ax[2].hist(C[i, j], bins=np.linspace(-1, 1, 50), density=True, histtype="step", lw=1.5, color=cc, label=nm)
    ax[1].set_xlabel("great-circle distance (km)"); ax[1].set_ylabel("density"); ax[1].legend(fontsize=8); ax[1].set_title("Distance between connected sensors")
    ax[2].set_xlabel("Pearson correlation of 2020 (train) flow"); ax[2].legend(fontsize=8); ax[2].set_title("Correlation between connected sensors")
    fig.savefig(FIG / "q6_graph.pdf"); plt.close(fig)

    # ------------------------------------------------------------------------------------------ Q7
    config = dict(CFG, optimizer="Adam", loss="masked MAE on z-scored targets", early_stopping="val MAE, best weights restored",
                  topk=None, hybrid_alpha_init=0.5, hybrid_alpha_learned=True, graph_layers=1, graph_residual=True,
                  fixed_normalisation="row: D_out^-1 A, diagonal set to 1", resample="15-min mean of 5-min counts",
                  missing="inputs z-scored with train stats then NaN->0; targets and metrics masked", T_in=T_IN, T_out=T_OUT)
    json.dump(config, open(OUT / "config.json", "w"), indent=2)
    here = dict(python=platform.python_version(), torch=torch.__version__, cpu=platform.processor() or platform.machine(),
                cpu_count=os.cpu_count(), gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    hw = {"this_run": here}
    if not retrain:
        hw["training_run"] = dict(python="3.12.14", torch="2.14.0+cu130", cpu="x86_64", cpu_count=24, gpu="NVIDIA GeForce RTX 5090", ram_gb=33.3)
    json.dump(hw, open(OUT / "hardware.json", "w"), indent=2)
    strip = lambda e: {k: v for k, v in e.items() if k in ("mae", "rmse", "mae_h", "rmse_h")}
    results = {name: dict(adjacency=specs[name][0], train_time_s=float(train_info[name][0]), epochs=train_info[name][1],
                          best_epoch=train_info[name][2], params=sum(p.numel() for p in models[name].parameters()),
                          val=strip(EV[name]["val"]), test=strip(EV[name]["test"])) for name in specs}
    results["Last value"] = {k: strip(EV["Last value"][k]) for k in ("val", "test")}
    results["Historical avg"] = EV["Historical avg"]
    R["results"] = results
    json.dump(R, open(OUT / "results.json", "w"), indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("\nwritten: ./config.json, hardware.json, results.json; figures in this folder")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true", help="train all four models instead of loading *.pt beside this file")
    main(ap.parse_args().retrain)
