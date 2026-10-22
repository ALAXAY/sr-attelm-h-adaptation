# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Oleksii Fesenko
"""Follow-up: output layer with an intercept (unit component appended to the hidden state) and a two-time-scale
refinement (sector RLS + exponentially weighted offset of the residual). Same folds, seeds, regimes as adapt_experiment."""
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import protocol as PR
import technique as TQ
import adapt_experiment as AE

LAMS = AE.LAMS
X, NET = AE.X, AE.NET
H1 = TQ.H_DIM + 1


def prepare_ic(train_views, fold, seed=0, gamma=1.0):
    """Preliminary stage as in adapt_experiment.prepare, plus initial matrices for the augmented hidden state."""
    M = AE.prepare(train_views, fold, seed, gamma)
    btr = M['btr']
    Hs, Es, Ms, Ss = [], [], [], []
    for v in train_views:
        xi, r_att, d = AE.prep(M, v)
        h = aug(M['res'].run(xi)); sect = TQ.sector_map(d, M['u1'], M['u2'], M['r0'])
        tgt = v['qt'] - v['locf'] - r_att
        m = np.isfinite(v['qt']).all(2) & (np.nan_to_num(v['ratio'], nan=1.0) < .999).any(2)
        Hs.append(h[1:btr].reshape(-1, H1)); Es.append(np.nan_to_num(tgt[1:btr]).reshape(-1, PR.P))
        Ms.append(m[1:btr].reshape(-1)); Ss.append(sect[1:btr].reshape(-1))
    H_, E_, M_, S_ = map(np.concatenate, (Hs, Es, Ms, Ss))
    M['Bi'], M['Pi'] = init_sectors(H_, E_, M_, S_, gamma)
    M['Bgi'], M['Pgi'] = init_sectors(H_, E_, M_, np.zeros_like(S_), gamma)
    return M


def aug(h):
    return np.concatenate([h, np.ones(h.shape[:-1] + (1,))], axis=-1)


def init_sectors(Hs, Es, Ms, sect, gamma=1.0):
    B = np.zeros((TQ.S_SECT, H1, PR.P)); Pm = np.zeros((TQ.S_SECT, H1, H1))
    for s in range(TQ.S_SECT):
        sel = (sect == s) & Ms
        Hm, Em = Hs[sel], Es[sel]
        Pm[s] = np.linalg.inv(Hm.T @ Hm + gamma * np.eye(H1))
        B[s] = Pm[s] @ Hm.T @ Em
    return B, Pm


def stream2(v, lo, hi, h, r_att, sect, B, Pm, lam, lam_b=None, online=True):
    """Sector RLS (optionally with an exponentially weighted offset of the residual, forgetting lam_b)."""
    B, Pm = B.copy(), Pm.copy()
    q_hat = np.array(v['locf'], dtype=float)
    bias = np.zeros((PR.N, PR.P)); queue = []
    target = v['qt'] - v['locf'] - r_att
    delayed = np.isfinite(v['qt']).all(2) & (np.nan_to_num(v['ratio'], nan=1.0) < .999).any(2)
    for t in range(max(1, lo), hi):
        T = (t + 1) * PR.BIN
        if online:
            keep = []
            for (ca, tt, i) in queue:
                if ca <= T:
                    s = sect[tt, i]
                    if lam_b is not None:
                        resid = target[tt, i] - B[s].T @ h[tt, i] - bias[i]
                        bias[i] = bias[i] + (1.0 - lam_b) * resid
                        TQ.rls_update(B, Pm, s, h[tt, i], target[tt, i] - bias[i], lam)
                    else:
                        TQ.rls_update(B, Pm, s, h[tt, i], target[tt, i], lam)
                else:
                    keep.append((ca, tt, i))
            queue = keep
        for i in range(PR.N):
            q_hat[t, i] = v['locf'][t, i] + r_att[t, i] + B[sect[t, i]].T @ h[t, i] + bias[i]
            if online and delayed[t, i]:
                queue.append((v['complete_at'][t, i], t, i))
    return q_hat


def run(M, v, lo, hi, mode, lam):
    xi, r_att, d = AE.prep(M, v)
    h = M['res'].run(xi)
    sect = TQ.sector_map(d, M['u1'], M['u2'], M['r0'])
    if mode == 'sector_ic':
        return stream2(v, lo, hi, aug(h), r_att, sect, M['Bi'], M['Pi'], lam)
    if mode == 'frozen_ic':
        return stream2(v, lo, hi, aug(h), r_att, sect, M['Bi'], M['Pi'], 1.0, online=False)
    if mode == 'global_ic':
        return stream2(v, lo, hi, aug(h), r_att, np.zeros(d.shape[:2], int), M['Bgi'], M['Pgi'], lam)
    if mode == 'sector_two':                       # RLS lambda + fast offset 0.99
        return stream2(v, lo, hi, h, r_att, sect, M['B'], M['P'], lam, lam_b=0.99)
    if mode == 'sector_ic_two':
        return stream2(v, lo, hi, aug(h), r_att, sect, M['Bi'], M['Pi'], lam, lam_b=0.99)
    raise ValueError(mode)


def select_lam(M, mode, view):
    scores = {}
    for lam in LAMS:
        qh = run(M, view, M['btr'], M['bva'], mode, lam)
        scores[lam] = PR.metrics(qh, view, PR.omega(view, M['btr'], M['bva']))['RMSE']
    return min(scores, key=scores.get), scores


if __name__ == '__main__':
    t0 = time.time(); rows = []
    out_csv = HERE / 'results_adaptation2.csv'
    done = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv); rows = prev.to_dict('records')
        done = set(zip(prev['fold'], prev['family'], prev['coverage']))
    for fold in 'AB':
        mean, std = PR.train_stats(X, fold)
        bva, bte = PR.bidx(PR.FOLDS[fold]['validation_end']), PR.bidx(PR.FOLDS[fold]['test_end'])
        tests = {'geo_x4': [PR.make_view(X, AE.geo(s, fold, 4), mean, std) for s in PR.SEEDS],
                 'geo_x8': [PR.make_view(X, AE.geo(s, fold, 8), mean, std) for s in PR.SEEDS],
                 'network': [PR.make_view(X, PR.network_delays(NET, s, 'shifted', fold), mean, std) for s in PR.SEEDS]}
        for family in ('geometric', 'network'):
            for coverage, levels in (('single', (1,)), ('multi', (1, 2, 4))):
                if (fold, family, coverage) in done:
                    continue
                if family == 'geometric':
                    trv = [PR.make_view(X, PR.geometric_delays(PR.TRAIN_SEED[fold], 'stationary', mult=m), mean, std) for m in levels]
                    val_shift = PR.make_view(X, PR.geometric_delays(PR.TRAIN_SEED[fold] + 1, 'stationary', mult=2), mean, std)
                    regimes = ('geo_x4', 'geo_x8', 'network')
                else:
                    trv = [PR.make_view(X, PR.network_delays(NET, PR.TRAIN_SEED[fold], f'level{m}'), mean, std) for m in levels]
                    val_shift = PR.make_view(X, PR.network_delays(NET, PR.TRAIN_SEED[fold], 'level2'), mean, std)
                    regimes = ('network',)
                M = prepare_ic(trv, fold)
                key = f'{fold}_{family}_{coverage}'
                rules = {}
                for mode in ('sector_ic', 'global_ic', 'sector_two', 'sector_ic_two'):
                    rules[mode] = dict(stat=select_lam(M, mode, M['val_view'])[0], shift=select_lam(M, mode, val_shift)[0])
                print(key, rules, 'elapsed', round(time.time() - t0), flush=True)
                for regime in regimes:
                    for s, v in zip(PR.SEEDS, tests[regime]):
                        m = PR.omega(v, bva, bte)
                        def add(mode, rule, lam):
                            qh = run(M, v, bva, bte, mode, lam)
                            rows.append(dict(fold=fold, family=family, coverage=coverage, regime=regime, seed=s,
                                             mode=mode, lam_rule=rule, lam=lam, **PR.metrics(qh, v, m)))
                        add('frozen_ic', '-', np.nan)
                        for lam in LAMS:
                            add('sector_ic', 'grid', lam)
                        for mode in rules:
                            for rule in ('stat', 'shift'):
                                add(mode, rule, rules[mode][rule])
                pd.DataFrame(rows).to_csv(out_csv, index=False)
                print(key, 'done; rows', len(rows), 'elapsed', round(time.time() - t0), flush=True)
    print('total time, s', round(time.time() - t0))
