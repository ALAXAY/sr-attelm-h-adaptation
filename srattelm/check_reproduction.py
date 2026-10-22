# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Oleksii Fesenko
"""Quick reproduction check (about 3-5 min on a CPU): split A, preliminary stage on the geometric delay level 1,
test seed 11, regimes x4 and x8. The frozen output layer and the two-time-scale refinement are recomputed and
compared with the shipped rows of results_adaptation.csv and results_adaptation2.csv.

Usage:  python check_reproduction.py
"""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import protocol as PR
import adapt_experiment as AE
import adapt_experiment2 as AE2

TOL = 2e-4      # agreement in the fourth decimal; CPU/BLAS differences may change the last digit

if __name__ == '__main__':
    X = AE.X
    fold, seed = 'A', 11
    D1 = pd.read_csv(HERE / 'results_adaptation.csv'); D2 = pd.read_csv(HERE / 'results_adaptation2.csv')
    mean, std = PR.train_stats(X, fold)
    bva, bte = PR.bidx(PR.FOLDS[fold]['validation_end']), PR.bidx(PR.FOLDS[fold]['test_end'])
    trv = [PR.make_view(X, PR.geometric_delays(PR.TRAIN_SEED[fold], 'stationary', mult=1), mean, std)]
    M = AE2.prepare_ic(trv, fold)
    ok = True
    for regime, mult in (('geo_x4', 4), ('geo_x8', 8)):
        v = PR.make_view(X, AE.geo(seed, fold, mult), mean, std)
        m = PR.omega(v, bva, bte)
        sel = lambda D, mode, rule: D[(D.fold == fold) & (D.family == 'geometric') & (D.coverage == 'single')
                                      & (D.regime == regime) & (D.seed == seed) & (D['mode'] == mode)
                                      & (D.lam_rule == rule)].iloc[0]
        r_lc = D1[(D1.fold == fold) & (D1['mode'] == 'LOCF') & (D1.regime == regime) & (D1.seed == seed)].iloc[0]
        r_fr = sel(D1, 'frozen', '-')
        r_two = sel(D2, 'sector_two', 'stat')
        got = {'LOCF': PR.metrics(v['locf'], v, m)['RMSE'],
               'frozen output layer': PR.metrics(AE.run(M, v, bva, bte, 'frozen', np.nan), v, m)['RMSE'],
               'two-time-scale refinement': PR.metrics(AE2.run(M, v, bva, bte, 'sector_two', float(r_two.lam)), v, m)['RMSE']}
        ref = {'LOCF': r_lc.RMSE, 'frozen output layer': r_fr.RMSE, 'two-time-scale refinement': r_two.RMSE}
        for k in got:
            d = abs(got[k] - ref[k]); ok &= d <= TOL
            print(f'{regime:7s} {k:27s} recomputed {got[k]:.4f}  shipped {ref[k]:.4f}  |diff| {d:.1e}', flush=True)
    print('REPRODUCTION CHECK', 'PASSED' if ok else 'FAILED')
    sys.exit(0 if ok else 1)
