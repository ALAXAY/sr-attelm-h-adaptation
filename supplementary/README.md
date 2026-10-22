# Formal check of expressions (1)–(36)

`SRAttELMH_verification.lean` states and proves 45 theorems (namespace `SRAttELMH`) on the expressions of the article
in Lean 4 with Mathlib. Comments in the file are in Ukrainian.

| Expressions | What is proved |
|---|---|
| (1), (2), (32) | monotonicity of the causally available set O(T); an available component was generated not later than T; the delay model (32) preserves condition (1) |
| (3), (33) | RMSE ≥ 0; RMSE = 0 if and only if the reconstruction is exact; MAE ≤ RMSE |
| (5)–(7) | 0 ≤ ρ ≤ 1 and ρ is non-decreasing in T; τ ≥ 0; Δt ≥ 0 |
| (9), (19)–(21) | the residual (21) equals the error of estimate (19) formed with B_{s,n-1}, b_{i,n-1} |
| (10)–(12) | attention weights are positive, sum to one and do not exceed one; invariance to a common shift of the link scores; monotonicity of a weight in its own score (in particular in β_ρ ρ_j); the context vector lies between the minimum and the maximum of the source values |
| (16) | the sector index belongs to {1, …, 8} for ψ ≥ −π; without min{·, 3} the direction ψ = π gives the argument 4 of the floor function |
| (17) | −1 < tanh x < 1 |
| (22) | b_n − v̄ = λ_b^n (b_0 − v̄); convergence to v̄ for 0 ≤ λ_b < 1 |
| (23)–(25) | scalar RLS (D_h = 1): P_n = P / (λ_s + h²P), information form 1/P_n = λ_s/P + h², g = P_n h, P_n > 0, a smaller λ_s gives a larger P_n |
| (28)–(31) | C_sector ≤ C_index ≤ C_global for the polynomial parts; M_hist · (S D_h) = M_stat · N, hence M_hist > M_stat if and only if N > S D_h |
| (34)–(36) | the mean relative reduction is positive when every run improves; 0 ≤ H ≤ ln|N_i|; κ = 0 for the frozen layer, κ = 100 for full recovery, 0 ≤ κ ≤ 100 in between; numerical checks of Table 9 (κ = 935/18 ≈ 51.9% in regime ×4) and of the reduction relative to SAITS in regime ×4 |

Expressions (4), (8), (13), (26), (27) are definitions or empirical procedures and are not formalized; the matrix
form of (23)–(25) is represented by its scalar case.

## How to check

1. Open https://live.lean-lang.org and select the latest Mathlib.
2. Paste the whole file. The check starts automatically.
3. The result is correct if the panel *All Messages* shows no `error` and every `#print axioms` line lists only
   `propext`, `Classical.choice` and `Quot.sound` (no `sorryAx`).

The file was checked in this way on 22 October 2026. The check of expression (16) led to the correction of the
expression in the manuscript: the minimum with 3 assigns the boundary direction ψ = π to the fourth angular sector,
as in `srattelm/technique.py` (`sector_map`).
