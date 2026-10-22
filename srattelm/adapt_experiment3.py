# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Oleksii Fesenko
"""Sensitivity of the two-time-scale refinement to the offset forgetting factor lam_b (RLS lambda fixed at 0.999)."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import protocol as PR, technique as TQ, adapt_experiment as AE, adapt_experiment2 as AE2
X, NET = AE.X, AE.NET
LAMB = (0.95, 0.98, 0.995, 0.999)

def run_two(M, v, lo, hi, lam_b, lam=0.999):
    xi, r_att, d = AE.prep(M, v); h = M['res'].run(xi)
    sect = TQ.sector_map(d, M['u1'], M['u2'], M['r0'])
    return AE2.stream2(v, lo, hi, h, r_att, sect, M['B'], M['P'], lam, lam_b=lam_b)

if __name__ == '__main__':
    t0 = time.time(); rows = []
    out_csv = HERE / 'results_adaptation3.csv'
    done = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv); rows = prev.to_dict('records'); done = set(zip(prev['fold'], prev['family'], prev['coverage']))
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
                    regimes = ('geo_x4', 'geo_x8', 'network')
                else:
                    trv = [PR.make_view(X, PR.network_delays(NET, PR.TRAIN_SEED[fold], f'level{m}'), mean, std) for m in levels]
                    regimes = ('network',)
                M = AE.prepare(trv, fold)
                # validation score of lam_b on the stationary validation part (for an a-priori rule)
                vsc = {lb: PR.metrics(run_two(M, M['val_view'], M['btr'], M['bva'], lb), M['val_view'],
                                      PR.omega(M['val_view'], M['btr'], M['bva']))['RMSE'] for lb in LAMB + (0.99,)}
                print(fold, family, coverage, 'val', {k: round(v, 4) for k, v in vsc.items()}, 'elapsed', round(time.time() - t0), flush=True)
                for regime in regimes:
                    for s, v in zip(PR.SEEDS, tests[regime]):
                        m = PR.omega(v, bva, bte)
                        for lb in LAMB:
                            qh = run_two(M, v, bva, bte, lb)
                            rows.append(dict(fold=fold, family=family, coverage=coverage, regime=regime, seed=s, mode='sector_two',
                                             lam_rule='grid_b', lam=0.999, lam_b=lb, val_rmse=vsc[lb], **PR.metrics(qh, v, m)))
                pd.DataFrame(rows).to_csv(out_csv, index=False)
    print('total time, s', round(time.time() - t0))
