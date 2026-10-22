# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Oleksii Fesenko
"""Experiment on online adaptation of the output layer under delay regimes not covered by the preliminary stage.

Factors
  coverage of the preliminary stage: single level (nu = 1) / multi-level (nu = 1, 2, 4)
  delay family of the preliminary stage: geometric / network-derived
  test regime: geometric x4 (inside the multi-level range), geometric x8 (outside), network degraded
               (for geometric-trained models this is a delay-model mismatch; for network-trained models the target regime)
  adaptation of the output layer during streaming:
      frozen      - B_s fixed after the preliminary stage
      sector      - RLS refinement of the active sector (proposed), lambda from a rule
      global      - RLS refinement of one output matrix for all regimes
      bias        - frozen B_s + exponentially weighted running mean of the verified error per source (simple tracker)
  lambda rule: 'stat' - chosen on the stationary validation part (as in the submitted manuscript);
               'shift' - chosen on a validation part whose mean delay is doubled relative to level 1 (separate realization)
  lambda grid for sensitivity: 0.98, 0.99, 0.995, 0.999, 1.0 (sector mode)
Outputs: results_adaptation.csv (one row per fold x seed x configuration).
"""
import os, sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import protocol as PR
import technique as TQ

torch.set_num_threads(int(os.environ.get('SRATTELM_THREADS', '8')))
LAMS = (0.98, 0.99, 0.995, 0.999, 1.0)
X, _ = PR.load_raw()
NET = {}
for name in ('delays_network.npz', 'delays_network_train.npz'):
    with np.load(HERE.parent / 'netsim' / name) as z:
        NET.update({k: z[k] for k in z.files if k.startswith('delay_')})


def geo(seed, fold, shift, mult=1.0, shift_at='validation_end'):
    shape = (len(PR.INDEX), PR.N, 2)
    rng = np.random.default_rng(seed + 8100)
    u = np.clip(rng.random(shape), 1e-12, 1 - 1e-12)
    mean = np.asarray([6.0, 2.0])[None, None, :] * np.linspace(.75, 1.25, PR.N)[None, :, None] * mult
    mean = np.broadcast_to(mean, shape).copy()
    mean[PR.INDEX >= pd.Timestamp(PR.FOLDS[fold][shift_at])] *= shift
    return np.floor(np.log1p(-u) / np.log(mean / (mean + 1)))[:, :, 0]


def prepare(train_views, fold, seed=0, gamma=1.0):
    """Preliminary stage of the proposed technique (identical to technique.fit_and_run, variant 'proposed')."""
    btr, bva = PR.bidx(PR.FOLDS[fold]['train_end']), PR.bidx(PR.FOLDS[fold]['validation_end'])
    val_view = train_views[0]
    for v in train_views:
        v['f'], v['rho'], v['tau'], v['dt'] = TQ.features(v)
    allf = np.concatenate([v['f'][1:btr].reshape(-1, TQ.F_DIM) for v in train_views])
    fmean, fstd = allf.mean(0), allf.std(0) + 1e-6
    fmean[8:], fstd[8:] = 0.0, 1.0
    model, val_rmse = TQ.train_attention(train_views, [val_view], 1, btr, btr, bva, use_availability=True,
                                         seed=seed, fmean=fmean, fstd=fstd, qk_availability=True)
    res = TQ.Reservoir(TQ.F_DIM + PR.P, seed=seed + 7)
    M = dict(model=model, res=res, fmean=fmean, fstd=fstd, btr=btr, bva=bva, val_view=val_view)
    preps = [prep(M, v) for v in train_views]
    M['u1'], M['u2'], M['r0'] = TQ.sector_fit(np.concatenate([p[2][1:btr] for p in preps]))
    Hs, Es, Ms, Ss = [], [], [], []
    for v, (xi, r_att, d) in zip(train_views, preps):
        h = res.run(xi); sect = TQ.sector_map(d, M['u1'], M['u2'], M['r0'])
        tgt = v['qt'] - v['locf'] - r_att
        m = np.isfinite(v['qt']).all(2) & (np.nan_to_num(v['ratio'], nan=1.0) < .999).any(2)
        Hs.append(h[1:btr].reshape(-1, TQ.H_DIM)); Es.append(np.nan_to_num(tgt[1:btr]).reshape(-1, PR.P))
        Ms.append(m[1:btr].reshape(-1)); Ss.append(sect[1:btr].reshape(-1))
    H_, E_, M_, S_ = map(np.concatenate, (Hs, Es, Ms, Ss))
    M['B'], M['P'] = TQ.init_sectors(H_, E_, M_, S_, gamma)
    M['Bg'], M['Pg'] = TQ.init_sectors(H_, E_, M_, np.zeros_like(S_), gamma)
    M['beta'] = [round(b, 4) for b in nn.functional.softplus(model.beta).tolist()]
    M['val_rmse_att'] = val_rmse
    return M


def prep(M, v):
    if 'f' not in v:
        v['f'], v['rho'], v['tau'], v['dt'] = TQ.features(v)
    f = (v['f'] - M['fmean']) / M['fstd']
    r_att, c, a = TQ.attention_outputs(M['model'], v, M['fmean'], M['fstd'])
    return np.concatenate([f, c], axis=2), r_att, c - v['locf']


def stream_bias(v, lo, hi, h, r_att, sect, B, lam):
    """Frozen output layer plus an exponentially weighted running mean of the verified error of every source."""
    q_hat = np.array(v['locf'], dtype=float)
    bias = np.zeros((PR.N, PR.P)); queue = []
    target = v['qt'] - v['locf'] - r_att
    delayed = np.isfinite(v['qt']).all(2) & (np.nan_to_num(v['ratio'], nan=1.0) < .999).any(2)
    for t in range(max(1, lo), hi):
        T = (t + 1) * PR.BIN
        keep = []
        for (ca, tt, i) in queue:
            if ca <= T:
                err = target[tt, i] - B[sect[tt, i]].T @ h[tt, i]
                bias[i] = lam * bias[i] + (1.0 - lam) * err
            else:
                keep.append((ca, tt, i))
        queue = keep
        for i in range(PR.N):
            q_hat[t, i] = v['locf'][t, i] + r_att[t, i] + B[sect[t, i]].T @ h[t, i] + bias[i]
            if delayed[t, i]:
                queue.append((v['complete_at'][t, i], t, i))
    return q_hat


def run(M, v, lo, hi, mode, lam):
    xi, r_att, d = prep(M, v)
    h = M['res'].run(xi)
    sect = TQ.sector_map(d, M['u1'], M['u2'], M['r0'])
    if mode == 'frozen':
        return TQ.stream(v, lo, hi, h, r_att, sect, M['B'], M['P'], 1.0, online=False)
    if mode == 'sector':
        return TQ.stream(v, lo, hi, h, r_att, sect, M['B'], M['P'], lam, online=True)
    if mode == 'global':
        z = np.zeros(d.shape[:2], int)
        return TQ.stream(v, lo, hi, h, r_att, z, M['Bg'], M['Pg'], lam, online=True)
    if mode == 'bias':
        return stream_bias(v, lo, hi, h, r_att, sect, M['B'], lam)
    raise ValueError(mode)


def select_lam(M, mode, view):
    scores = {}
    for lam in LAMS:
        qh = run(M, view, M['btr'], M['bva'], mode, lam)
        scores[lam] = PR.metrics(qh, view, PR.omega(view, M['btr'], M['bva']))['RMSE']
    best = min(scores, key=scores.get)
    return best, scores


if __name__ == '__main__':
    t0 = time.time(); rows = []; info = {}
    out_csv = HERE / 'results_adaptation.csv'
    info_json = HERE / 'results_adaptation_info.json'
    done = set()
    if out_csv.exists():                                            # resume after an interrupted run
        prev = pd.read_csv(out_csv)
        prev = prev[prev['mode'] != 'LOCF']
        rows = prev.to_dict('records')
        done = set(zip(prev['fold'], prev['family'], prev['coverage']))
        if info_json.exists():
            info = json.loads(info_json.read_text(encoding='utf-8'))
        print('resuming; configurations done:', sorted(done), flush=True)
    for fold in 'AB':
        mean, std = PR.train_stats(X, fold)
        bva, bte = PR.bidx(PR.FOLDS[fold]['validation_end']), PR.bidx(PR.FOLDS[fold]['test_end'])
        tests = {'geo_x4': [PR.make_view(X, geo(s, fold, 4), mean, std) for s in PR.SEEDS],
                 'geo_x8': [PR.make_view(X, geo(s, fold, 8), mean, std) for s in PR.SEEDS],
                 'network': [PR.make_view(X, PR.network_delays(NET, s, 'shifted', fold), mean, std) for s in PR.SEEDS]}
        for regime, tv in tests.items():
            for s, v in zip(PR.SEEDS, tv):
                rows.append(dict(fold=fold, family='-', coverage='-', regime=regime, seed=s, mode='LOCF',
                                 lam_rule='-', lam=np.nan, **PR.metrics(v['locf'], v, PR.omega(v, bva, bte))))
        for family in ('geometric', 'network'):
            for coverage, levels in (('single', (1,)), ('multi', (1, 2, 4))):
                if family == 'geometric':
                    trv = [PR.make_view(X, PR.geometric_delays(PR.TRAIN_SEED[fold], 'stationary', mult=m), mean, std)
                           for m in levels]
                    val_shift = PR.make_view(X, PR.geometric_delays(PR.TRAIN_SEED[fold] + 1, 'stationary', mult=2), mean, std)
                    regimes = ('geo_x4', 'geo_x8', 'network')
                else:
                    trv = [PR.make_view(X, PR.network_delays(NET, PR.TRAIN_SEED[fold], f'level{m}'), mean, std)
                           for m in levels]
                    val_shift = PR.make_view(X, PR.network_delays(NET, PR.TRAIN_SEED[fold], 'level2'), mean, std)
                    regimes = ('network',)
                key = f'{fold}_{family}_{coverage}'
                if (fold, family, coverage) in done:
                    continue
                M = prepare(trv, fold)
                info[key] = dict(beta=M['beta'], val_rmse_att=M['val_rmse_att'], train_s=round(time.time() - t0, 1))
                lam_rules = {}
                for mode in ('sector', 'global', 'bias'):
                    ls, sc_s = select_lam(M, mode, M['val_view'])
                    lh, sc_h = select_lam(M, mode, val_shift)
                    lam_rules[mode] = dict(stat=ls, shift=lh)
                    info[key][f'val_scores_{mode}'] = dict(stat=sc_s, shift=sc_h)
                print(key, 'lambda rules', lam_rules, 'elapsed', round(time.time() - t0), flush=True)
                for regime in regimes:
                    for s, v in zip(PR.SEEDS, tests[regime]):
                        m = PR.omega(v, bva, bte)
                        def add(mode, rule, lam):
                            qh = run(M, v, bva, bte, mode, lam)
                            rows.append(dict(fold=fold, family=family, coverage=coverage, regime=regime, seed=s,
                                             mode=mode, lam_rule=rule, lam=lam, beta=str(M['beta']),
                                             **PR.metrics(qh, v, m)))
                        add('frozen', '-', np.nan)
                        for lam in LAMS:
                            add('sector', 'grid', lam)
                        for mode in ('sector', 'global', 'bias'):
                            for rule in ('stat', 'shift'):
                                add(mode, rule, lam_rules[mode][rule])
                pd.DataFrame(rows).to_csv(out_csv, index=False)
                info_json.write_text(json.dumps(info, indent=1), encoding='utf-8')
                print(key, 'done; rows', len(rows), 'elapsed', round(time.time() - t0), flush=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    (HERE / 'results_adaptation_info.json').write_text(json.dumps(info, indent=1), encoding='utf-8')
    df = pd.DataFrame(rows)
    df['n_r2'] = df['n'] * df['RMSE'] ** 2
    g = df.groupby(['family', 'coverage', 'regime', 'mode', 'lam_rule', 'lam'], dropna=False)
    pooled = np.sqrt(g['n_r2'].sum() / g['n'].sum()).round(4)
    print(pooled.to_string())
    print('total time, s', round(time.time() - t0))
