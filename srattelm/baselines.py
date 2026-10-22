# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Oleksii Fesenko
"""Adapted reproductions of GRU-D, BRITS, SAITS and GRIN under the two-time causal protocol of the paper.

Each model receives, for decision bin t, a window of the last W bins of causally available information of all
12 stations: the observed partial-bin mean q^o (zero where nothing arrived), the observation mask, the received
fraction rho and the information gap; it predicts the deviation of the verified state from the local causal estimate
(LOCF) for all stations and components at bin t. Training on the delay levels of the preliminary stage (the same
realizations as the technique), early stopping on the stationary validation part, evaluation on the test regimes.
The reproductions follow the main architectural principles of the published methods (see the paper) and do not
reproduce the authors' benchmarks.
"""
import os, sys, time, math
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import protocol as PR
import adapt_experiment as AE

torch.set_num_threads(int(os.environ.get('SRATTELM_THREADS', '8')))
X, NET = AE.X, AE.NET
W = 8                      # window of 6-h bins (48 h)
N, P = PR.N, PR.P
FIN = 3 * P + 2            # per-station features: q^o (P), mask (P), q^L (P), rho, gap


def tensors(v):
    """Per-bin station features (nb, N, FIN), target deviation (nb, N, P), loss mask (nb, N, P)."""
    qo = np.nan_to_num(v['qo'], nan=0.0); mask = np.isfinite(v['qo']).astype(np.float32)
    rho = np.nan_to_num(v['ratio'], nan=0.0).mean(2, keepdims=True)
    gap = np.log1p(v['gap'].mean(2, keepdims=True) * PR.BIN) / np.log1p(24.0)
    f = np.concatenate([qo, mask, v['locf'], rho, gap], 2).astype(np.float32)
    r = np.nan_to_num(v['qt'] - v['locf']).astype(np.float32)
    m = (np.isfinite(v['qt']) & (np.nan_to_num(v['ratio'], nan=1.0) < .999)).astype(np.float32)
    return torch.tensor(f), torch.tensor(r), torch.tensor(m)


def windows(f, idx):
    """(len(idx), W, N, FIN) windows ending at the bins idx (zero-padded at the start)."""
    pad = torch.cat([torch.zeros(W - 1, *f.shape[1:]), f])
    return torch.stack([pad[i:i + W] for i in idx])


class GRUD(nn.Module):
    """GRU-D [Che et al.]: input decay towards the last value by the gap, hidden-state decay, mask as input."""
    def __init__(self, H=48):
        super().__init__()
        self.gx = nn.Linear(1, P); self.gh = nn.Linear(1, H)
        self.cell = nn.GRUCell(FIN + P, H); self.emb = nn.Parameter(torch.randn(N, 8) * .1)
        self.head = nn.Sequential(nn.Linear(H + 8, 64), nn.Tanh(), nn.Linear(64, P))
    def forward(self, x):                               # x: (B, W, N, FIN)
        B = x.shape[0]; x = x.permute(0, 2, 1, 3).reshape(B * N, W, FIN)
        h = torch.zeros(B * N, self.cell.hidden_size); last = torch.zeros(B * N, P)
        for t in range(W):
            z = x[:, t]; qo, m, gap = z[:, :P], z[:, P:2 * P], z[:, -1:]
            dx = torch.exp(-torch.relu(self.gx(gap))); dh = torch.exp(-torch.relu(self.gh(gap)))
            xhat = m * qo + (1 - m) * dx * last
            h = self.cell(torch.cat([z, xhat], -1), dh * h); last = m * qo + (1 - m) * last
        e = self.emb.repeat(B, 1)
        return self.head(torch.cat([h, e], -1)).reshape(B, N, P)


class BRITS(nn.Module):
    """BRITS [Cao et al.]: recurrent imputation with temporal decay, history and feature regression, bidirectional
    passes over the causal window with a consistency loss."""
    def __init__(self, H=48):
        super().__init__()
        self.f = _RITS(H); self.b = _RITS(H)
    def forward(self, x):
        yf, imp_f = self.f(x); yb, imp_b = self.b(torch.flip(x, [1]))
        self.consistency = (imp_f - torch.flip(imp_b, [1])).abs().mean()
        return 0.5 * (yf + yb)


class _RITS(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.cell = nn.GRUCell(2 * P, H); self.hist = nn.Linear(H, P); self.feat = nn.Linear(P, P, bias=False)
        self.decay = nn.Linear(1, H); self.beta = nn.Linear(2 * P, P)
        self.emb = nn.Parameter(torch.randn(N, 8) * .1)
        self.head = nn.Sequential(nn.Linear(H + 8, 64), nn.Tanh(), nn.Linear(64, P))
    def forward(self, x):
        B = x.shape[0]; x = x.permute(0, 2, 1, 3).reshape(B * N, W, FIN)
        h = torch.zeros(B * N, self.cell.hidden_size); imps = []
        for t in range(W):
            z = x[:, t]; qo, m, gap = z[:, :P], z[:, P:2 * P], z[:, -1:]
            h = h * torch.exp(-torch.relu(self.decay(gap)))
            xh = self.hist(h); xc = m * qo + (1 - m) * xh
            zf = self.feat(xc) - torch.diagonal(self.feat.weight) * xc
            beta = torch.sigmoid(self.beta(torch.cat([m, gap.expand(-1, P)], -1)))
            ch = beta * zf + (1 - beta) * xh; cc = m * qo + (1 - m) * ch; imps.append(cc)
            h = self.cell(torch.cat([cc, m], -1), h)
        e = self.emb.repeat(B, 1)
        return self.head(torch.cat([h, e], -1)).reshape(B, N, P), torch.stack(imps, 1).reshape(B, N, W, P)


class SAITS(nn.Module):
    """SAITS [Du et al.]: two diagonally-masked self-attention blocks over the window with learned weighted
    combination; tokens are time steps carrying the features of all stations."""
    def __init__(self, D=96, heads=4):
        super().__init__()
        din = N * FIN
        self.in1 = nn.Linear(din, D); self.in2 = nn.Linear(din + N * P, D)
        self.pos = nn.Parameter(torch.randn(W, D) * .02)
        self.att1 = nn.MultiheadAttention(D, heads, batch_first=True); self.att2 = nn.MultiheadAttention(D, heads, batch_first=True)
        self.ff1 = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D)); self.ff2 = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D))
        self.out1 = nn.Linear(D, N * P); self.out2 = nn.Linear(D, N * P); self.comb = nn.Linear(2 * N * P + N * P, N * P)
        diag = torch.eye(W, dtype=torch.bool); causal = torch.triu(torch.ones(W, W, dtype=torch.bool), 1)
        mask = diag | causal; mask[0, 0] = False           # diagonal masked, no look-ahead; first token keeps itself
        self.register_buffer('mask', mask)
    def block(self, x, att, ff):
        a, _ = att(x, x, x, attn_mask=self.mask); x = x + a
        return x + ff(x)
    def forward(self, x):
        B = x.shape[0]; z = x.reshape(B, W, N * FIN)
        h1 = self.block(self.in1(z) + self.pos, self.att1, self.ff1); y1 = self.out1(h1)
        h2 = self.block(self.in2(torch.cat([z, y1], -1)) + self.pos, self.att2, self.ff2); y2 = self.out2(h2)
        m = x[..., P:2 * P].reshape(B, W, N * P)
        y = self.comb(torch.cat([y1, y2, m], -1))
        return y[:, -1].reshape(B, N, P)


class GRIN(nn.Module):
    """GRIN [Cini et al.]: spatial message passing over the station graph (learned adjacency) combined with a
    recurrent update per station; readout at the decision bin."""
    def __init__(self, H=48):
        super().__init__()
        self.emb = nn.Parameter(torch.randn(N, 8) * .1)
        self.inp = nn.Linear(FIN, H); self.mp = nn.Linear(H, H); self.cell = nn.GRUCell(2 * H, H)
        self.head = nn.Sequential(nn.Linear(H + 8, 64), nn.Tanh(), nn.Linear(64, P))
    def forward(self, x):
        B = x.shape[0]
        A = torch.softmax(torch.relu(self.emb @ self.emb.T), -1)
        h = torch.zeros(B, N, self.cell.hidden_size)
        for t in range(W):
            u = torch.tanh(self.inp(x[:, t]))
            msg = torch.einsum('ij,bjh->bih', A, self.mp(h))
            h = self.cell(torch.cat([u, msg], -1).reshape(B * N, -1), h.reshape(B * N, -1)).reshape(B, N, -1)
        return self.head(torch.cat([h, self.emb[None].expand(B, -1, -1)], -1))


MODELS = {'GRU-D': GRUD, 'BRITS': BRITS, 'SAITS': SAITS, 'GRIN': GRIN}


def fit(name, train_views, val_view, fold, seed=0, epochs=50, lr=1e-3, batch=64):
    torch.manual_seed(seed); np.random.seed(seed)
    btr, bva = PR.bidx(PR.FOLDS[fold]['train_end']), PR.bidx(PR.FOLDS[fold]['validation_end'])
    model = MODELS[name]()
    tr = [tensors(v) for v in train_views]; va = tensors(val_view)
    idx_tr = np.arange(1, btr); idx_va = np.arange(btr, bva)
    Xtr = torch.cat([windows(f, idx_tr) for f, _, _ in tr]); Rtr = torch.cat([r[idx_tr] for _, r, _ in tr]); Mtr = torch.cat([m[idx_tr] for _, _, m in tr])
    Xva, Rva, Mva = windows(va[0], idx_va), va[1][idx_va], va[2][idx_va]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, state, bad = float('inf'), None, 0
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Xtr))
        for k in range(0, len(perm), batch):
            i = perm[k:k + batch]
            y = model(Xtr[i]); loss = (((y - Rtr[i]) ** 2) * Mtr[i]).sum() / Mtr[i].sum().clamp_min(1)
            if name == 'BRITS':
                loss = loss + 0.1 * model.consistency
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(); loss.backward(); gn = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isfinite(gn):
                opt.step()
        model.eval()
        with torch.no_grad():
            y = torch.cat([model(Xva[k:k + 256]) for k in range(0, len(Xva), 256)])
            vl = float((((y - Rva) ** 2) * Mva).sum() / Mva.sum().clamp_min(1))
        if math.isfinite(vl) and vl < best - 1e-5:
            best, state, bad = vl, {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 8:
                break
    if state is None:                                   # diverged: restart once with a smaller learning rate
        if lr < 1e-4:
            raise RuntimeError(f'{name} did not train')
        print('  restart', name, 'with lr', lr / 3, flush=True)
        return fit(name, train_views, val_view, fold, seed=seed + 1, epochs=epochs, lr=lr / 3, batch=batch)
    model.load_state_dict(state)
    return model, math.sqrt(best), ep + 1


def predict(model, v, lo, hi):
    f, _, _ = tensors(v); idx = np.arange(max(1, lo), hi)
    Xt = windows(f, idx); model.eval()
    with torch.no_grad():
        y = torch.cat([model(Xt[k:k + 256]) for k in range(0, len(Xt), 256)]).numpy()
    q = np.array(v['locf'], dtype=float); q[idx] = v['locf'][idx] + y
    return q


if __name__ == '__main__':
    t0 = time.time(); rows = []
    out_csv = HERE / 'results_baselines.csv'
    done = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv); rows = prev.to_dict('records'); done = set(zip(prev.fold, prev.family, prev.method))
    for fold in 'AB':
        mean, std = PR.train_stats(X, fold)
        bva, bte = PR.bidx(PR.FOLDS[fold]['validation_end']), PR.bidx(PR.FOLDS[fold]['test_end'])
        tests = {'geo_x4': [PR.make_view(X, AE.geo(s, fold, 4), mean, std) for s in PR.SEEDS],
                 'geo_x8': [PR.make_view(X, AE.geo(s, fold, 8), mean, std) for s in PR.SEEDS],
                 'network': [PR.make_view(X, PR.network_delays(NET, s, 'shifted', fold), mean, std) for s in PR.SEEDS]}
        for family in ('geometric', 'network'):
            if family == 'geometric':
                trv = [PR.make_view(X, PR.geometric_delays(PR.TRAIN_SEED[fold], 'stationary', mult=m), mean, std) for m in (1, 2, 4)]
                regimes = ('geo_x4', 'geo_x8', 'network')
            else:
                trv = [PR.make_view(X, PR.network_delays(NET, PR.TRAIN_SEED[fold], f'level{m}'), mean, std) for m in (1, 2, 4)]
                regimes = ('network',)
            for name in MODELS:
                if (fold, family, name) in done:
                    continue
                t1 = time.time(); model, vr, ep = fit(name, trv, trv[0], fold)
                npar = sum(p.numel() for p in model.parameters())
                print(fold, family, name, 'val', round(vr, 4), 'epochs', ep, 'params', npar, 'train s', round(time.time() - t1), flush=True)
                for regime in regimes:
                    for s, v in zip(PR.SEEDS, tests[regime]):
                        q = predict(model, v, bva, bte)
                        rows.append(dict(fold=fold, family=family, regime=regime, seed=s, method=name, val_rmse=vr, epochs=ep,
                                         params=npar, train_s=round(time.time() - t1, 1), **PR.metrics(q, v, PR.omega(v, bva, bte))))
                pd.DataFrame(rows).to_csv(out_csv, index=False)
    print('total s', round(time.time() - t0))
