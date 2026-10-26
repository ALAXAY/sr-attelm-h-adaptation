# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Oleksii Fesenko
"""Builds summary_adaptation4.csv and paired_adaptation4.csv from results_adaptation4_*.csv (fuzzy sector weights)
and results_adaptation.csv (hard sector RLS without offset, rule 'stat')."""
import glob, json, os
import numpy as np, pandas as pd
from scipy import stats
d4 = pd.concat([pd.read_csv(f) for f in sorted(glob.glob('adaptation4/results_adaptation4_*.csv'))], ignore_index=True)
d1 = pd.read_csv('results_adaptation.csv')
d1 = d1[(d1['mode'] == 'sector') & (d1.lam_rule == 'stat')].assign(mode='sector_hard')
allr = pd.concat([d4, d1], ignore_index=True); allr['n_r2'] = allr['n'] * allr['RMSE'] ** 2
def pooled(g): return float(np.sqrt(g['n_r2'].sum() / g['n'].sum()))
S = allr.groupby(['family', 'coverage', 'regime', 'mode']).apply(pooled).unstack('mode')
S = S[['sector_hard', 'fuzzy_rls', 'global_two', 'two_hard', 'fuzzy_two']]
S.to_csv('summary_adaptation4.csv')
def runs(fam, cov, reg, mode):
    q = allr[(allr.family == fam) & (allr.coverage == cov) & (allr.regime == reg) & (allr['mode'] == mode)]
    return q.set_index(['fold', 'seed']).sort_index()
rows = []
comps = [('fuzzy_two', 'two_hard', 'fuzzy two-scale vs hard two-scale'),
         ('fuzzy_two', 'global_two', 'fuzzy two-scale vs global two-scale'),
         ('two_hard', 'global_two', 'hard two-scale vs global two-scale'),
         ('fuzzy_rls', 'sector_hard', 'fuzzy RLS vs hard sector RLS')]
for (fam, cov, reg) in S.index:
    for a, b, lab in comps:
        A, B = runs(fam, cov, reg, a), runs(fam, cov, reg, b)
        diff = (A.RMSE - B.RMSE).values; rel = 100 * (1 - A.RMSE / B.RMSE).values
        rng = np.random.default_rng(0); bs = [rng.choice(diff, len(diff), replace=True).mean() for _ in range(10000)]
        t = stats.ttest_rel(A.RMSE.values, B.RMSE.values)
        rows.append(dict(family=fam, coverage=cov, regime=reg, comparison=lab, pooled_a=pooled(A), pooled_b=pooled(B),
                         mean_diff=diff.mean(), mean_rel=rel.mean(), rel_min=rel.min(), rel_max=rel.max(),
                         ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5), wins=int((diff < 0).sum()),
                         n_runs=len(diff), t=t.statistic, p=t.pvalue))
pd.DataFrame(rows).to_csv('paired_adaptation4.csv', index=False)
sel = {os.path.basename(f)[len('results_adaptation4_'):-5]: json.load(open(f))['selected'] for f in sorted(glob.glob('adaptation4/results_adaptation4_*.json'))}
json.dump(sel, open('selected_adaptation4.json', 'w'), indent=1)
print(S.round(4))
