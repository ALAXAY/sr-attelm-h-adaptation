# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Oleksii Fesenko
"""Builds summary_adaptation.csv and paired_adaptation.csv from results_adaptation*.csv (experiment 3 of the paper)."""
import numpy as np, pandas as pd
from scipy import stats
d1 = pd.read_csv('results_adaptation.csv'); d2 = pd.read_csv('results_adaptation2.csv')
for d in (d1, d2): d['n_r2'] = d['n'] * d['RMSE'] ** 2
def pooled(g): return float(np.sqrt(g['n_r2'].sum() / g['n'].sum()))
keys = ['family', 'coverage', 'regime', 'mode', 'lam_rule']
p1 = d1.groupby(keys, dropna=False).apply(pooled).rename('RMSE').reset_index()
p2 = d2.groupby(keys).apply(pooled).rename('RMSE').reset_index()
def pick(p, mode, rule): return p[(p['mode'] == mode) & (p.lam_rule == rule)].set_index(['family', 'coverage', 'regime'])['RMSE']
idx = [(f, c, r) for f in ('geometric', 'network') for c in ('multi', 'single') for r in (('geo_x4', 'geo_x8', 'network') if f == 'geometric' else ('network',))]
S = pd.DataFrame(index=pd.MultiIndex.from_tuples(idx, names=['family', 'coverage', 'regime']))
locf = p1[p1['mode'] == 'LOCF'].set_index('regime')['RMSE']
S['LOCF'] = [locf[r] for (_, _, r) in idx]
for col, (m, ru) in {'frozen': ('frozen', '-'), 'sector': ('sector', 'shift'), 'global': ('global', 'shift'), 'offset': ('bias', 'shift'),
                     'sector_stat': ('sector', 'stat')}.items(): S[col] = pick(p1, m, ru)
for col, (m, ru) in {'two_scale': ('sector_two', 'stat'), 'two_scale_shift': ('sector_two', 'shift'), 'sector_ic': ('sector_ic', 'shift')}.items(): S[col] = pick(p2, m, ru)
S.to_csv('summary_adaptation.csv')
allr = pd.concat([d1, d2], ignore_index=True)
def runs(fam, cov, reg, mode, rule):
    q = allr[(allr.family == fam) & (allr.coverage == cov) & (allr.regime == reg) & (allr['mode'] == mode) & (allr.lam_rule == rule)]
    return q.set_index(['fold', 'seed']).sort_index()
def paired(fam, cov, reg, a, b, seed=0):
    A, B = runs(fam, cov, reg, *a), runs(fam, cov, reg, *b)
    diff = (A.RMSE - B.RMSE).values; rel = 100 * (1 - A.RMSE / B.RMSE).values
    rng = np.random.default_rng(seed); bs = [rng.choice(diff, 6, replace=True).mean() for _ in range(10000)]
    t = stats.ttest_rel(A.RMSE.values, B.RMSE.values); w = stats.wilcoxon(A.RMSE.values, B.RMSE.values)
    return dict(pooled_a=pooled(A), pooled_b=pooled(B), mean_diff=diff.mean(), mean_rel=rel.mean(), rel_min=rel.min(), rel_max=rel.max(),
                ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5), wins=int((diff < 0).sum()), t=t.statistic, p=t.pvalue, p_w=w.pvalue)
comps = [(('sector_two', 'stat'), ('frozen', '-'), 'two-scale vs frozen'), (('sector_two', 'stat'), ('sector', 'stat'), 'two-scale vs sector RLS'),
         (('sector_two', 'stat'), ('bias', 'stat'), 'two-scale vs offset only'), (('sector', 'stat'), ('frozen', '-'), 'sector RLS vs frozen'),
         (('bias', 'stat'), ('frozen', '-'), 'offset only vs frozen'), (('sector', 'stat'), ('global', 'stat'), 'sector RLS vs global RLS'),
         (('sector_two', 'stat'), ('global', 'stat'), 'two-scale vs global RLS')]
rows = []
for fam, cov, reg in idx:
    for a, b, lab in comps:
        r = paired(fam, cov, reg, a, b); r.update(family=fam, coverage=cov, regime=reg, comparison=lab); rows.append(r)
for reg in ('geo_x4', 'geo_x8', 'network'):
    A, B = runs('geometric', 'multi', reg, 'frozen', '-'), runs('geometric', 'single', reg, 'frozen', '-')
    diff = (A.RMSE - B.RMSE).values; rel = 100 * (1 - A.RMSE / B.RMSE).values
    rows.append(dict(family='geometric', coverage='multi vs single', regime=reg, comparison='frozen: multi-level vs single-level', pooled_a=pooled(A), pooled_b=pooled(B),
                     mean_diff=diff.mean(), mean_rel=rel.mean(), rel_min=rel.min(), rel_max=rel.max(), wins=int((diff < 0).sum()), t=np.nan,
                     p=stats.ttest_rel(A.RMSE, B.RMSE).pvalue, p_w=np.nan, ci_lo=np.nan, ci_hi=np.nan))
pd.DataFrame(rows).to_csv('paired_adaptation.csv', index=False)
print(S.round(4))
