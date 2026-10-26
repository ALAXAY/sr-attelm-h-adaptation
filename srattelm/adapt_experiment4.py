# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Oleksii Fesenko
"""Follow-up 4: fuzzy (Takagi-Sugeno) partition of the discrepancy plane instead of the hard sector operator (16),
with the same folds, seeds, regimes and lambda-selection rule ('stat', stationary validation part) as adapt_experiment.

Memberships of the S = 8 rules (s = 4(l-1) + q, the same indexing as technique.sector_map):
  radial    mu_in = 1 - sigmoid(g_r (|d| - r0) / r0),  mu_out = sigmoid(g_r (|d| - r0) / r0)
  angular   von Mises kernels exp(kappa cos(psi - psi_q)), psi_q = -3pi/4 + q pi/2, normalized over q
  rule      mu_s = mu_rad * mu_ang (sums to one); g_r, kappa -> inf gives the hard operator (16)
Output    delta = sum_s mu_s B_s^T h (+ per-source offset b_i)
Update    fuzzy-weighted RLS of every rule with mu_s > MU_MIN:
          k = mu_s P_s h / (lam_s + mu_s h^T P_s h),  lam_s = lam ** mu_s  (a rule forgets in proportion to its
          activation; mu_s in {0, 1} gives exactly technique.rls_update of the active sector)
Initial   B_s0 = (H^T W_s H + gamma I)^{-1} H^T W_s E,  W_s = diag(mu_s)  (hard limit = technique.init_sectors)
Modes     two_hard   - hard sectors + offset lambda_b = 0.99 (control; reproduces 'sector_two', rule 'stat')
          global_two - one output matrix + offset lambda_b = 0.99 (control: are sectors needed at all?)
          fuzzy_rls  - fuzzy rules, no offset
          fuzzy_two  - fuzzy rules + offset lambda_b = 0.99
lambda_s from LAMS and (g_r, kappa) from GRID are selected jointly on the stationary validation part.
Usage     python adapt_experiment4.py                       -> all eight configurations, one after another
          python adapt_experiment4.py FOLD FAMILY COVERAGE   -> one configuration (for parallel runs)
          Output: adaptation4/results_adaptation4_FOLD_FAMILY_COVERAGE.csv (+ .json with the selected parameters);
          a configuration whose CSV exists is skipped (resume).
"""
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import protocol as PR
import technique as TQ
import adapt_experiment as AE
import adapt_experiment2 as AE2

LAMS = AE.LAMS
X, NET = AE.X, AE.NET
LAM_B = 0.99
MU_MIN = 1e-3
GRID = [(2.0, 1.0), (2.0, 4.0), (2.0, 16.0), (6.0, 1.0), (6.0, 4.0), (6.0, 16.0),
        (20.0, 1.0), (20.0, 4.0), (20.0, 16.0), (np.inf, np.inf)]
CENTERS = -0.75 * np.pi + 0.5 * np.pi * np.arange(4)


def memberships(d, u1, u2, r0, g_r, kappa):
    psi = np.arctan2(d @ u2, d @ u1)
    x = (np.linalg.norm(d, axis=-1) - r0) / r0
    if np.isinf(g_r):
        mo = (x > 0).astype(float)
    else:
        mo = 1.0 / (1.0 + np.exp(-np.clip(g_r * x, -60, 60)))
    if np.isinf(kappa):
        q = np.minimum(np.floor(2 * (psi + np.pi) / np.pi), 3).astype(int)
        a = np.eye(4)[q]
    else:
        a = np.exp(kappa * (np.cos(psi[..., None] - CENTERS) - 1.0))
        a = a / a.sum(-1, keepdims=True)
    return np.concatenate([(1.0 - mo)[..., None] * a, mo[..., None] * a], axis=-1)


def init_fuzzy(H, E, Mk, mu, gamma=1.0):
    S, Dh = mu.shape[1], H.shape[1]
    B = np.zeros((S, Dh, PR.P)); Pm = np.zeros((S, Dh, Dh))
    for s in range(S):
        w = mu[:, s] * Mk
        Hw = H * w[:, None]
        Pm[s] = np.linalg.inv(H.T @ Hw + gamma * np.eye(Dh))
        B[s] = Pm[s] @ Hw.T @ E
    return B, Pm


def train_arrays(M, train_views):
    """Hidden states, targets, masks and discrepancy vectors of the training part (as in adapt_experiment.prepare)."""
    btr = M['btr']; Hs, Es, Ms, Ds = [], [], [], []
    for v in train_views:
        xi, r_att, d = AE.prep(M, v)
        h = M['res'].run(xi)
        tgt = v['qt'] - v['locf'] - r_att
        m = np.isfinite(v['qt']).all(2) & (np.nan_to_num(v['ratio'], nan=1.0) < .999).any(2)
        Hs.append(h[1:btr].reshape(-1, TQ.H_DIM)); Es.append(np.nan_to_num(tgt[1:btr]).reshape(-1, PR.P))
        Ms.append(m[1:btr].reshape(-1).astype(float)); Ds.append(d[1:btr].reshape(-1, PR.P))
    return tuple(map(np.concatenate, (Hs, Es, Ms, Ds)))


def stream_fuzzy(v, lo, hi, h, r_att, mu, B, Pm, lam, lam_b=None):
    B, Pm = B.copy(), Pm.copy()
    q_hat = np.array(v['locf'], dtype=float)
    bias = np.zeros((PR.N, PR.P)); queue = []
    target = v['qt'] - v['locf'] - r_att
    delayed = np.isfinite(v['qt']).all(2) & (np.nan_to_num(v['ratio'], nan=1.0) < .999).any(2)
    for t in range(max(1, lo), hi):
        T = (t + 1) * PR.BIN
        keep = []
        for (ca, tt, i) in queue:
            if ca <= T:
                hh, mm = h[tt, i], mu[tt, i]
                act = np.nonzero(mm > MU_MIN)[0]
                if lam_b is not None:
                    pred = sum(mm[s] * (B[s].T @ hh) for s in act)
                    bias[i] = bias[i] + (1.0 - lam_b) * (target[tt, i] - pred - bias[i])
                e = target[tt, i] - bias[i]
                for s in act:
                    ls = lam ** mm[s]
                    Ph = Pm[s] @ hh
                    k = mm[s] * Ph / (ls + mm[s] * (hh @ Ph))
                    B[s] = B[s] + np.outer(k, e - B[s].T @ hh)
                    Pm[s] = (Pm[s] - np.outer(k, hh @ Pm[s])) / ls
            else:
                keep.append((ca, tt, i))
        queue = keep
        for i in range(PR.N):
            mm = mu[t, i]; act = np.nonzero(mm > MU_MIN)[0]
            q_hat[t, i] = v['locf'][t, i] + r_att[t, i] + sum(mm[s] * (B[s].T @ h[t, i]) for s in act) + bias[i]
            if delayed[t, i]:
                queue.append((v['complete_at'][t, i], t, i))
    return q_hat


class Runner:
    def __init__(self, M, train_views):
        self.M = M
        self.H, self.E, self.Mk, self.D = train_arrays(M, train_views)
        self.init_cache = {}
        self.view_cache = {}

    def view(self, v, key):
        if key not in self.view_cache:
            xi, r_att, d = AE.prep(self.M, v)
            self.view_cache[key] = (self.M['res'].run(xi), r_att, d)
        return self.view_cache[key]

    def init(self, g, kap):
        if (g, kap) not in self.init_cache:
            mu = memberships(self.D, self.M['u1'], self.M['u2'], self.M['r0'], g, kap)
            self.init_cache[(g, kap)] = init_fuzzy(self.H, self.E, self.Mk, mu)
        return self.init_cache[(g, kap)]

    def run(self, v, key, lo, hi, mode, lam, g=np.inf, kap=np.inf):
        h, r_att, d = self.view(v, key)
        M = self.M
        if mode == 'two_hard':
            sect = TQ.sector_map(d, M['u1'], M['u2'], M['r0'])
            return AE2.stream2(v, lo, hi, h, r_att, sect, M['B'], M['P'], lam, lam_b=LAM_B)
        if mode == 'global_two':
            return AE2.stream2(v, lo, hi, h, r_att, np.zeros(d.shape[:2], int), M['Bg'], M['Pg'], lam, lam_b=LAM_B)
        B0, P0 = self.init(g, kap)
        mu = memberships(d, M['u1'], M['u2'], M['r0'], g, kap)
        return stream_fuzzy(v, lo, hi, h, r_att, mu, B0, P0, lam, lam_b=LAM_B if mode == 'fuzzy_two' else None)

    def score(self, mode, lam, g=np.inf, kap=np.inf):
        v = self.M['val_view']; M = self.M
        qh = self.run(v, 'val', M['btr'], M['bva'], mode, lam, g, kap)
        return PR.metrics(qh, v, PR.omega(v, M['btr'], M['bva']))['RMSE']


OUT = HERE / 'adaptation4'
CONFIGS = [(f, fam, c) for fam in ('geometric', 'network') for f in 'AB' for c in ('single', 'multi')]


def main(fold, family, coverage):
    tag = f'{fold}_{family}_{coverage}'
    OUT.mkdir(exist_ok=True)
    if (OUT / f'results_adaptation4_{tag}.csv').exists():
        print(tag, 'exists, skipped', flush=True)
        return
    t0 = time.time()
    levels = (1,) if coverage == 'single' else (1, 2, 4)
    mean, std = PR.train_stats(X, fold)
    bva, bte = PR.bidx(PR.FOLDS[fold]['validation_end']), PR.bidx(PR.FOLDS[fold]['test_end'])
    if family == 'geometric':
        trv = [PR.make_view(X, PR.geometric_delays(PR.TRAIN_SEED[fold], 'stationary', mult=m), mean, std) for m in levels]
        regimes = ('geo_x4', 'geo_x8', 'network')
    else:
        trv = [PR.make_view(X, PR.network_delays(NET, PR.TRAIN_SEED[fold], f'level{m}'), mean, std) for m in levels]
        regimes = ('network',)
    M = AE.prepare(trv, fold)
    R = Runner(M, trv)
    print('prepared', round(time.time() - t0), flush=True)
    sel, val_scores = {}, {}
    for mode in ('two_hard', 'global_two'):
        sc = {lam: R.score(mode, lam) for lam in LAMS}
        sel[mode] = dict(lam=min(sc, key=sc.get), g_r=None, kappa=None); val_scores[mode] = {str(k): v for k, v in sc.items()}
        print(mode, sel[mode], round(time.time() - t0), flush=True)
    for mode in ('fuzzy_rls', 'fuzzy_two'):
        sc = {(lam, g, k): R.score(mode, lam, g, k) for (g, k) in GRID for lam in LAMS}
        best = min(sc, key=sc.get)
        sel[mode] = dict(lam=best[0], g_r=best[1], kappa=best[2]); val_scores[mode] = {str(k): v for k, v in sc.items()}
        print(mode, sel[mode], round(time.time() - t0), flush=True)
    tests = {'geo_x4': lambda s: AE.geo(s, fold, 4), 'geo_x8': lambda s: AE.geo(s, fold, 8),
             'network': lambda s: PR.network_delays(NET, s, 'shifted', fold)}
    rows = []
    for regime in regimes:
        for s in PR.SEEDS:
            v = PR.make_view(X, tests[regime](s), mean, std)
            m = PR.omega(v, bva, bte)
            key = f'{regime}_{s}'
            for mode, p in sel.items():
                g = p['g_r'] if p['g_r'] is not None else np.inf
                k = p['kappa'] if p['kappa'] is not None else np.inf
                qh = R.run(v, key, bva, bte, mode, p['lam'], g, k)
                rows.append(dict(fold=fold, family=family, coverage=coverage, regime=regime, seed=s, mode=mode,
                                 lam_rule='stat', lam=p['lam'], g_r=p['g_r'], kappa=p['kappa'], **PR.metrics(qh, v, m)))
            R.view_cache.pop(key, None)
        print(regime, 'done', round(time.time() - t0), flush=True)
    pd.DataFrame(rows).to_csv(OUT / f'results_adaptation4_{tag}.csv', index=False)
    (OUT / f'results_adaptation4_{tag}.json').write_text(json.dumps(dict(selected=sel, val_rmse=val_scores,
                                                                       seconds=round(time.time() - t0)), indent=1),
                                                        encoding='utf-8')
    print('total, s', round(time.time() - t0), flush=True)


if __name__ == '__main__':
    for cfg in ([tuple(sys.argv[1:4])] if len(sys.argv) >= 4 else CONFIGS):
        main(*cfg)
