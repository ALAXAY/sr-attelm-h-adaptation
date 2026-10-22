# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Oleksii Fesenko
"""Experiment 1 of the article: pooled RMSE of the technique and of the reference methods in three test regimes
(Table 6) and paired comparison over six runs (Table 7).

Inputs (all in srattelm/): results_adaptation.csv (LOCF, technique with sector RLS alone), results_adaptation2.csv
(technique with the two-time-scale refinement), results_baselines.csv (GRU-D, BRITS, SAITS, GRIN),
results_official_agg.csv (official AGG core; regimes x4 and network-derived only).
Outputs: summary_methods.csv, paired_methods.csv.
Models of every method are trained on the delay levels 1, 2, 4 of the delay model of the test regime
(geometric for x4 and x8, network-derived for the network regime).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
REGIMES = ('geo_x4', 'geo_x8', 'network')
FAMILY = {'geo_x4': 'geometric', 'geo_x8': 'geometric', 'network': 'network'}
METHODS = ('two_scale', 'sector', 'GRU-D', 'BRITS', 'SAITS', 'GRIN', 'AGG', 'LOCF')
LABEL = {'two_scale': 'Technique, two-time-scale refinement', 'sector': 'Technique, sector RLS alone',
         'GRU-D': 'GRU-D (adapted reproduction)', 'BRITS': 'BRITS (adapted reproduction)',
         'SAITS': 'SAITS (adapted reproduction)', 'GRIN': 'GRIN (adapted reproduction)',
         'AGG': 'Official AGG core', 'LOCF': 'Last observation carried forward'}

D1 = pd.read_csv(HERE / 'results_adaptation.csv')
D2 = pd.read_csv(HERE / 'results_adaptation2.csv')
BL = pd.read_csv(HERE / 'results_baselines.csv')
AGG = pd.read_csv(HERE / 'results_official_agg.csv')


def runs(method, regime):
    """Per-run results (index: fold, seed) of a method in a test regime."""
    fam = FAMILY[regime]
    if method == 'LOCF':
        q = D1[(D1['mode'] == 'LOCF') & (D1.regime == regime)]
    elif method == 'AGG':
        if regime == 'geo_x8':
            return pd.DataFrame(columns=['RMSE', 'n'])
        q = AGG[AGG.delays == fam].rename(columns={'RMSE_agg': 'RMSE', 'n_agg': 'n'})
    elif method == 'two_scale':
        q = D2[(D2.family == fam) & (D2.coverage == 'multi') & (D2.regime == regime) & (D2['mode'] == 'sector_two')
               & (D2.lam_rule == 'stat')]
    elif method == 'sector':
        q = D1[(D1.family == fam) & (D1.coverage == 'multi') & (D1.regime == regime) & (D1['mode'] == 'sector')
               & (D1.lam_rule == 'stat')]
    else:
        q = BL[(BL.family == fam) & (BL.regime == regime) & (BL.method == method)]
    cols = ['fold', 'seed', 'RMSE', 'n'] + [c for c in ('MAE', 'PM25_RMSE') if c in q.columns]
    return q[cols].set_index(['fold', 'seed']).sort_index()


def pooled(q, col):
    if len(q) == 0 or col not in q.columns:
        return np.nan
    if col == 'MAE':
        return float((q['n'] * q[col]).sum() / q['n'].sum())
    return float(np.sqrt((q['n'] * q[col] ** 2).sum() / q['n'].sum()))


def paired(a, b, regime, seed=0):
    A, B = runs(a, regime), runs(b, regime)
    if len(A) != 6 or len(B) != 6:
        return None
    diff = (A.RMSE - B.RMSE).values
    rel = 100 * (1 - A.RMSE / B.RMSE).values
    rng = np.random.default_rng(seed)
    bs = [rng.choice(diff, 6, replace=True).mean() for _ in range(10000)]
    t = stats.ttest_rel(A.RMSE.values, B.RMSE.values)
    w = stats.wilcoxon(diff)
    return dict(comparison=f'{a} vs {b}', regime=regime, mean_diff=diff.mean(), mean_rel_pct=rel.mean(),
                rel_min_pct=rel.min(), rel_max_pct=rel.max(), ci95_lo=np.percentile(bs, 2.5),
                ci95_hi=np.percentile(bs, 97.5), wins=int((diff < 0).sum()), runs=6, t=t.statistic, p_t=t.pvalue,
                p_wilcoxon=w.pvalue)


if __name__ == '__main__':
    rows = []
    for m in METHODS:
        row = dict(method=m, label=LABEL[m])
        for r in REGIMES:
            q = runs(m, r)
            row[f'RMSE_{r}'] = pooled(q, 'RMSE')
            row[f'runs_{r}'] = len(q)
        q4 = runs(m, 'geo_x4')
        row['MAE_geo_x4'] = pooled(q4, 'MAE')
        row['PM25_RMSE_geo_x4'] = pooled(q4, 'PM25_RMSE')
        rows.append(row)
    S = pd.DataFrame(rows)
    for r in REGIMES:
        S[f'reduction_pct_{r}'] = 100 * (1 - S.loc[S.method == 'two_scale', f'RMSE_{r}'].iloc[0] / S[f'RMSE_{r}'])
    S.to_csv(HERE / 'summary_methods.csv', index=False, float_format='%.8g')
    P = pd.DataFrame([p for m in METHODS[1:] for r in REGIMES if (p := paired('two_scale', m, r)) is not None])
    P.to_csv(HERE / 'paired_methods.csv', index=False, float_format='%.8g')
    pd.set_option('display.width', 200)
    print(S[['label'] + [f'RMSE_{r}' for r in REGIMES]].round(4).to_string(index=False))
    print(P[['comparison', 'regime', 'mean_rel_pct', 'wins', 'p_t']].round(4).to_string(index=False))
