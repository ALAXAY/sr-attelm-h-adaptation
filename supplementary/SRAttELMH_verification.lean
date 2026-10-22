/-
  SR-AttELM-H (рукопис v2, Neurocomputing): формальна перевірка математичних виразів (1)–(36)
  Lean 4 + Mathlib (live.lean-lang.org, Latest Mathlib).

  Що перевіряється (теореми):
   (1),(2),(32) – причинність: монотонність доступної множини O(T), t^e ≤ T для доступної компоненти,
                  узгодженість моделі затримок (32) з умовою (1);
   (3),(33)     – RMSE ≥ 0, RMSE = 0 ⇔ точне відновлення, MAE ≤ RMSE;
   (5)          – 0 ≤ ρ ≤ 1, ρ не спадає з T;
   (6),(7)      – τ ≥ 0, Δt ≥ 0;
   (9),(19)–(21)– залишок (21) дорівнює похибці оцінки (19), сформованої з B_{s,n-1}, b_{i,n-1};
   (10),(11)    – ваги уваги: додатні, сума 1, кожна ≤ 1, інваріантність до зсуву оцінок зв'язку,
                  монотонність ваги за власною оцінкою зв'язку (зокрема за β_ρ ρ_j);
   (12)         – контекстний вектор не виходить за межі значень джерел (опукла комбінація);
   (16)         – індекс сектора належить {1,…,8} для всіх ψ ≥ −π (завдяки min{·,3} верхня межа ψ не потрібна);
                  без min напрямок ψ = π дає 2(ψ+π)/π = 4, тобто п'ятий кутовий сектор;
   (17)         – обмеженість активацій: −1 < tanh x < 1;
   (22)         – рекурсія зміщення: b_n − v̄ = λ_b^n (b_0 − v̄), збіжність до v̄ при 0 ≤ λ_b < 1;
   (23)–(25)    – RLS (скалярна форма, D_h = 1): P_n = P/(λ_s + h²P), інформаційна форма 1/P_n = λ_s/P + h²,
                  g = P_n h, P_n > 0, менше λ_s – більше P_n (швидше забування);
   (28)–(30)    – поліноміальні частини вартостей: C_sector ≤ C_index ≤ C_global;
   (31)         – M_hist·(S D_h) = M_stat·N, тобто M_hist > M_stat ⇔ N > S D_h;
   (34)         – середнє відносне зниження додатне, якщо кожен прогін кращий;
   (35)         – 0 ≤ H ≤ ln|N_i|;
   (36)         – κ = 0 для замороженого шару, κ = 100 при повній компенсації, 0 ≤ κ ≤ 100 між ними;
                  числова перевірка κ для табл. 9 (0,3305; 0,2945; 0,3118 → 935/18 ≈ 51,9 %).
  Не формалізується (означення або емпіричні процедури): (4), (8), (13), (26), (27) та матрична форма (23)–(25).
-/
import Mathlib

open Finset Real

namespace SRAttELMH

/-! ### (1), (2), (32): повідомлення з двома часовими координатами -/

/-- Повідомлення (1): вектор вимірювання, момент вимірювання, моменти надходження D компонент. -/
structure Message (D : ℕ) where
  x  : Fin D → ℝ
  te : ℝ
  ta : Fin D → ℝ
  arrival_after_generation : ∀ p, te ≤ ta p

/-- Причинно доступна множина (2) для одного повідомлення: компоненти, що надійшли до моменту T. -/
def available {D : ℕ} (z : Message D) (T : ℝ) : Set (Fin D) := {p | z.ta p ≤ T}

/-- Монотонність (2): пізніший момент рішення не втрачає доступних компонент. -/
theorem available_mono {D : ℕ} (z : Message D) {T₁ T₂ : ℝ} (h : T₁ ≤ T₂) :
    available z T₁ ⊆ available z T₂ :=
  fun p hp => show z.ta p ≤ T₂ from le_trans (show z.ta p ≤ T₁ from hp) h

/-- Доступна в момент T компонента виміряна не пізніше T. -/
theorem generated_before_decision {D : ℕ} (z : Message D) {T : ℝ} {p : Fin D}
    (hp : p ∈ available z T) : z.te ≤ T :=
  le_trans (z.arrival_after_generation p) (show z.ta p ≤ T from hp)

/-- Модель затримок (32): масштабування базової затримки множником ν ≥ 0 зберігає умову (1). -/
theorem delay_model_consistent (te ν dt0 : ℝ) (hν : 0 ≤ ν) (hd : 0 ≤ dt0) :
    te ≤ te + ν * dt0 := by
  have := mul_nonneg hν hd
  linarith

/-! ### (3), (33): RMSE і MAE над скінченною множиною Ω -/

/-- Цільова функція (3). -/
noncomputable def rmse {ι : Type*} (Ω : Finset ι) (qhat q : ι → ℝ) : ℝ :=
  Real.sqrt ((1 / (Ω.card : ℝ)) * ∑ i ∈ Ω, (qhat i - q i) ^ 2)

/-- Показник (33). -/
noncomputable def mae {ι : Type*} (Ω : Finset ι) (qhat q : ι → ℝ) : ℝ :=
  (1 / (Ω.card : ℝ)) * ∑ i ∈ Ω, |qhat i - q i|

theorem rmse_nonneg {ι : Type*} (Ω : Finset ι) (qhat q : ι → ℝ) : 0 ≤ rmse Ω qhat q :=
  Real.sqrt_nonneg _

theorem rmse_eq_zero_of_exact {ι : Type*} (Ω : Finset ι) (qhat q : ι → ℝ)
    (h : ∀ i ∈ Ω, qhat i = q i) : rmse Ω qhat q = 0 := by
  unfold rmse
  have hs : ∑ i ∈ Ω, (qhat i - q i) ^ 2 = 0 :=
    Finset.sum_eq_zero (fun i hi => by rw [h i hi]; ring)
  rw [hs]
  simp

theorem exact_of_rmse_eq_zero {ι : Type*} (Ω : Finset ι) (qhat q : ι → ℝ)
    (hΩ : Ω.Nonempty) (h : rmse Ω qhat q = 0) : ∀ i ∈ Ω, qhat i = q i := by
  unfold rmse at h
  have hcard : (0 : ℝ) < Ω.card := by exact_mod_cast hΩ.card_pos
  have hnn : 0 ≤ ∑ i ∈ Ω, (qhat i - q i) ^ 2 := Finset.sum_nonneg (fun i _ => sq_nonneg _)
  have h0 := Real.sqrt_eq_zero'.mp h
  have h1 : (1 / (Ω.card : ℝ)) * ∑ i ∈ Ω, (qhat i - q i) ^ 2 = 0 :=
    le_antisymm h0 (mul_nonneg (by positivity) hnn)
  have hsum : ∑ i ∈ Ω, (qhat i - q i) ^ 2 = 0 := by
    rcases mul_eq_zero.mp h1 with h2 | h2
    · exact absurd h2 (one_div_ne_zero hcard.ne')
    · exact h2
  intro i hi
  have h3 := (Finset.sum_eq_zero_iff_of_nonneg (fun j _ => sq_nonneg (qhat j - q j))).mp hsum i hi
  have h4 : qhat i - q i = 0 := pow_eq_zero_iff two_ne_zero |>.mp h3
  linarith

/-- MAE ≤ RMSE (нерівність Коші–Буняковського). -/
theorem mae_le_rmse {ι : Type*} (Ω : Finset ι) (qhat q : ι → ℝ) (hΩ : Ω.Nonempty) :
    mae Ω qhat q ≤ rmse Ω qhat q := by
  have hcard : (0 : ℝ) < Ω.card := by exact_mod_cast hΩ.card_pos
  have hmae : 0 ≤ mae Ω qhat q := by
    unfold mae
    exact mul_nonneg (by positivity) (Finset.sum_nonneg (fun i _ => abs_nonneg _))
  have key : (mae Ω qhat q) ^ 2 ≤ (1 / (Ω.card : ℝ)) * ∑ i ∈ Ω, (qhat i - q i) ^ 2 := by
    unfold mae
    have hcs := sq_sum_le_card_mul_sum_sq (s := Ω) (f := fun i => |qhat i - q i|)
    simp only [sq_abs] at hcs
    rw [mul_pow]
    calc (1 / (Ω.card : ℝ)) ^ 2 * (∑ i ∈ Ω, |qhat i - q i|) ^ 2
        ≤ (1 / (Ω.card : ℝ)) ^ 2 * ((Ω.card : ℝ) * ∑ i ∈ Ω, (qhat i - q i) ^ 2) := by gcongr
      _ = (1 / (Ω.card : ℝ)) * ∑ i ∈ Ω, (qhat i - q i) ^ 2 := by
          have hc := mul_inv_cancel₀ hcard.ne'
          linear_combination ((∑ i ∈ Ω, (qhat i - q i) ^ 2) * (1 / (Ω.card : ℝ))) * hc
  calc mae Ω qhat q = Real.sqrt ((mae Ω qhat q) ^ 2) := (Real.sqrt_sq hmae).symm
    _ ≤ rmse Ω qhat q := Real.sqrt_le_sqrt key

/-! ### (5)–(7): параметри доступності -/

/-- Частка отриманих компонент (5). -/
noncomputable def receivedShare {D : ℕ} (z : Message D) (T : ℝ) : ℝ :=
  (1 / (D : ℝ)) * ∑ p : Fin D, (if z.ta p ≤ T then (1 : ℝ) else 0)

theorem receivedShare_nonneg {D : ℕ} (z : Message D) (T : ℝ) : 0 ≤ receivedShare z T := by
  unfold receivedShare
  apply mul_nonneg (by positivity)
  exact Finset.sum_nonneg (fun p _ => by split_ifs <;> norm_num)

theorem receivedShare_le_one {D : ℕ} (hD : 0 < D) (z : Message D) (T : ℝ) :
    receivedShare z T ≤ 1 := by
  unfold receivedShare
  have hD' : (0 : ℝ) < D := by exact_mod_cast hD
  have hsum : ∑ p : Fin D, (if z.ta p ≤ T then (1 : ℝ) else 0) ≤ ∑ _p : Fin D, (1 : ℝ) :=
    Finset.sum_le_sum (fun p _ => by split_ifs <;> norm_num)
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one] at hsum
  calc (1 / (D : ℝ)) * ∑ p : Fin D, (if z.ta p ≤ T then (1 : ℝ) else 0)
      ≤ (1 / (D : ℝ)) * D := by gcongr
    _ = 1 := one_div_mul_cancel hD'.ne'

/-- ρ не спадає з моментом рішення T. -/
theorem receivedShare_mono {D : ℕ} (z : Message D) {T₁ T₂ : ℝ} (h : T₁ ≤ T₂) :
    receivedShare z T₁ ≤ receivedShare z T₂ := by
  unfold receivedShare
  apply mul_le_mul_of_nonneg_left _ (by positivity)
  apply Finset.sum_le_sum
  intro p _
  by_cases h1 : z.ta p ≤ T₁
  · have h2 : z.ta p ≤ T₂ := le_trans h1 h
    simp [h1, h2]
  · have h0 : (if z.ta p ≤ T₁ then (1 : ℝ) else 0) = 0 := by simp [h1]
    rw [h0]
    split_ifs <;> norm_num

/-- Тривалість інформаційного розриву (6) невід'ємна для джерела з доступною компонентою. -/
theorem age_nonneg {D : ℕ} (z : Message D) {T : ℝ} {p : Fin D} (hp : p ∈ available z T) :
    0 ≤ T - z.te := by
  have := generated_before_decision z hp
  linarith

/-- Середня затримка надходження (7) невід'ємна. -/
theorem mean_delay_nonneg {ι : Type*} (W : Finset ι) (ta te : ι → ℝ)
    (h : ∀ i ∈ W, te i ≤ ta i) :
    0 ≤ (1 / (W.card : ℝ)) * ∑ i ∈ W, (ta i - te i) := by
  apply mul_nonneg (by positivity)
  exact Finset.sum_nonneg (fun i hi => by linarith [h i hi])

/-! ### (9), (19)–(21): структура оцінки і залишку (покомпонентно) -/

/-- Залишок (21) дорівнює похибці оцінки (19), сформованої з попередніх B_{s,n-1} і b_{i,n-1}. -/
theorem residual_eq_estimate_error (q qL ratt δprev bprev : ℝ) :
    let r := q - qL                          -- (9)
    let ε := r - ratt                        -- (20)
    let qhat := qL + ratt + δprev + bprev    -- (19)
    let v := ε - δprev - bprev               -- (21)
    q - qhat = v := by
  intro r ε qhat v
  simp only [r, ε, qhat, v]
  ring

/-! ### (10), (11): ваги уваги -/

/-- Вага уваги (11) для оцінок зв'язку e. -/
noncomputable def attnWeight {ι : Type*} (N : Finset ι) (e : ι → ℝ) (j : ι) : ℝ :=
  Real.exp (e j) / ∑ j' ∈ N, Real.exp (e j')

theorem attnWeight_pos {ι : Type*} (N : Finset ι) (e : ι → ℝ) (hN : N.Nonempty) (j : ι) :
    0 < attnWeight N e j := by
  unfold attnWeight
  exact div_pos (Real.exp_pos _) (Finset.sum_pos (fun j' _ => Real.exp_pos _) hN)

theorem attnWeight_sum_one {ι : Type*} (N : Finset ι) (e : ι → ℝ) (hN : N.Nonempty) :
    ∑ j ∈ N, attnWeight N e j = 1 := by
  unfold attnWeight
  rw [← Finset.sum_div]
  exact div_self (Finset.sum_pos (fun j' _ => Real.exp_pos _) hN).ne'

theorem attnWeight_le_one {ι : Type*} (N : Finset ι) (e : ι → ℝ) {j : ι} (hj : j ∈ N) :
    attnWeight N e j ≤ 1 := by
  unfold attnWeight
  have hpos : 0 < ∑ j' ∈ N, Real.exp (e j') := Finset.sum_pos (fun j' _ => Real.exp_pos _) ⟨j, hj⟩
  rw [div_le_one hpos]
  exact Finset.single_le_sum (fun j' _ => (Real.exp_pos (e j')).le) hj

/-- Зсув усіх оцінок зв'язку на сталу c не змінює ваг (11). -/
theorem attnWeight_shift {ι : Type*} (N : Finset ι) (e : ι → ℝ) (c : ℝ) (j : ι) :
    attnWeight N (fun j' => e j' + c) j = attnWeight N e j := by
  unfold attnWeight
  simp only [Real.exp_add, ← Finset.sum_mul]
  rw [mul_div_mul_right _ _ (Real.exp_pos c).ne']

/-- Оцінка зв'язку (10) зростає з ρ_j при β_ρ ≥ 0 (решта складових фіксована). -/
theorem linkScore_mono_rho (base βρ βτ βΔ τ Δt : ℝ) (hβ : 0 ≤ βρ) {ρ₁ ρ₂ : ℝ} (h : ρ₁ ≤ ρ₂) :
    base + βρ * ρ₁ - βτ * τ - βΔ * Δt ≤ base + βρ * ρ₂ - βτ * τ - βΔ * Δt := by
  have := mul_le_mul_of_nonneg_left h hβ
  linarith

/-- Вага (11) зростає зі своєю оцінкою зв'язку, якщо оцінки інших джерел не змінюються. -/
theorem attnWeight_mono {ι : Type*} [DecidableEq ι] (N : Finset ι) {j : ι} (hj : j ∈ N) {e e' : ι → ℝ}
    (hle : e j ≤ e' j) (hother : ∀ j' ∈ N, j' ≠ j → e' j' = e j') :
    attnWeight N e j ≤ attnWeight N e' j := by
  unfold attnWeight
  have hR : ∑ j' ∈ N.erase j, Real.exp (e' j') = ∑ j' ∈ N.erase j, Real.exp (e j') :=
    Finset.sum_congr rfl (fun j' hj' => by
      rw [hother j' (Finset.mem_of_mem_erase hj') (Finset.ne_of_mem_erase hj')])
  rw [← Finset.add_sum_erase N _ hj, ← Finset.add_sum_erase N _ hj, hR]
  have hR0 : 0 ≤ ∑ j' ∈ N.erase j, Real.exp (e j') :=
    Finset.sum_nonneg (fun _ _ => (Real.exp_pos _).le)
  have ha : 0 < Real.exp (e j) := Real.exp_pos _
  have hb : Real.exp (e j) ≤ Real.exp (e' j) := Real.exp_le_exp.mpr hle
  have hd1 : 0 < Real.exp (e j) + ∑ j' ∈ N.erase j, Real.exp (e j') := by linarith
  have hd2 : 0 < Real.exp (e' j) + ∑ j' ∈ N.erase j, Real.exp (e j') := by linarith
  rw [div_le_iff₀ hd1, div_mul_eq_mul_div, le_div_iff₀ hd2]
  nlinarith [mul_le_mul_of_nonneg_right hb hR0]

/-! ### (12): контекстний вектор як опукла комбінація (покомпонентно) -/

theorem context_le_bound {ι : Type*} (N : Finset ι) (e : ι → ℝ) (hN : N.Nonempty)
    (y : ι → ℝ) (M : ℝ) (hM : ∀ j ∈ N, y j ≤ M) :
    ∑ j ∈ N, attnWeight N e j * y j ≤ M := by
  calc ∑ j ∈ N, attnWeight N e j * y j
      ≤ ∑ j ∈ N, attnWeight N e j * M :=
        Finset.sum_le_sum (fun j hj =>
          mul_le_mul_of_nonneg_left (hM j hj) (attnWeight_pos N e hN j).le)
    _ = M := by rw [← Finset.sum_mul, attnWeight_sum_one N e hN, one_mul]

theorem context_ge_bound {ι : Type*} (N : Finset ι) (e : ι → ℝ) (hN : N.Nonempty)
    (y : ι → ℝ) (m : ℝ) (hm : ∀ j ∈ N, m ≤ y j) :
    m ≤ ∑ j ∈ N, attnWeight N e j * y j := by
  calc m = ∑ j ∈ N, attnWeight N e j * m := by
        rw [← Finset.sum_mul, attnWeight_sum_one N e hN, one_mul]
    _ ≤ ∑ j ∈ N, attnWeight N e j * y j :=
        Finset.sum_le_sum (fun j hj =>
          mul_le_mul_of_nonneg_left (hm j hj) (attnWeight_pos N e hN j).le)

/-! ### (14)–(16): відображення в сектор -/

/-- Індекс сектора (16): два радіальні рівні × чотири кутові сектори, S = 8. -/
noncomputable def sectorIndex (dnorm d0 ψ : ℝ) : ℤ :=
  4 * (if d0 < dnorm then 1 else 0) + min ⌊2 * (ψ + π) / π⌋ 3 + 1

theorem sectorIndex_mem (dnorm d0 ψ : ℝ) (hψ₁ : -π ≤ ψ) :
    1 ≤ sectorIndex dnorm d0 ψ ∧ sectorIndex dnorm d0 ψ ≤ 8 := by
  unfold sectorIndex
  have hπ : 0 < π := Real.pi_pos
  have h0 : (0 : ℝ) ≤ 2 * (ψ + π) / π := by
    apply div_nonneg <;> linarith
  have hfl : 0 ≤ ⌊2 * (ψ + π) / π⌋ := Int.floor_nonneg.mpr h0
  have hmin1 : 0 ≤ min ⌊2 * (ψ + π) / π⌋ 3 := le_min hfl (by norm_num)
  have hmin2 : min ⌊2 * (ψ + π) / π⌋ 3 ≤ 3 := min_le_right _ _
  constructor
  · split_ifs <;> linarith
  · split_ifs <;> linarith

/-- Без min{·,3} граничний напрямок ψ = π дає аргумент підлоги 4, тобто п'ятий кутовий сектор. -/
theorem boundary_direction_needs_clamp : (2 * (π + π) / π : ℝ) = 4 := by
  have hπ : π ≠ 0 := Real.pi_pos.ne'
  rw [show (2 : ℝ) * (π + π) = 4 * π by ring, mul_div_assoc, div_self hπ, mul_one]

/-! ### (17): обмеженість активацій прихованого шару (σ = tanh) -/

theorem tanh_lt_one' (x : ℝ) : Real.tanh x < 1 := by
  rw [Real.tanh_eq_sinh_div_cosh, div_lt_one (Real.cosh_pos x)]
  exact Real.sinh_lt_cosh x

theorem neg_one_lt_tanh' (x : ℝ) : -1 < Real.tanh x := by
  have h := tanh_lt_one' (-x)
  rw [Real.tanh_neg] at h
  linarith

/-! ### (22): рекурсія зміщення по джерелу -/

/-- Рекурсія (21)–(22) при сталому значенні ε − Bᵀh ≡ v̄: b_n = b_{n-1} + (1 − λ_b)(v̄ − b_{n-1}). -/
def offsetSeq (lamb b0 vbar : ℝ) : ℕ → ℝ
  | 0 => b0
  | n + 1 => offsetSeq lamb b0 vbar n + (1 - lamb) * (vbar - offsetSeq lamb b0 vbar n)

theorem offset_tracks_constant (lamb b0 vbar : ℝ) (n : ℕ) :
    offsetSeq lamb b0 vbar n - vbar = lamb ^ n * (b0 - vbar) := by
  induction n with
  | zero => simp [offsetSeq]
  | succ n ih =>
    have h : offsetSeq lamb b0 vbar n = vbar + lamb ^ n * (b0 - vbar) := by linarith
    simp only [offsetSeq, pow_succ]
    rw [h]
    ring

/-- При 0 ≤ λ_b < 1 зміщення збігається до сталого зсуву v̄. -/
theorem offset_converges (lamb b0 vbar : ℝ) (hl0 : 0 ≤ lamb) (hl1 : lamb < 1) :
    Filter.Tendsto (offsetSeq lamb b0 vbar) Filter.atTop (nhds vbar) := by
  have h : Filter.Tendsto (fun n : ℕ => lamb ^ n * (b0 - vbar) + vbar) Filter.atTop
      (nhds (0 * (b0 - vbar) + vbar)) :=
    ((tendsto_pow_atTop_nhds_zero_of_lt_one hl0 hl1).mul_const _).add_const _
  simp only [zero_mul, zero_add] at h
  refine h.congr (fun n => ?_)
  have := offset_tracks_constant lamb b0 vbar n
  linarith

/-! ### (23)–(25): секторний RLS, скалярна форма (D_h = 1) -/

/-- Коефіцієнт підсилення (23). -/
noncomputable def rlsGain (lams P h : ℝ) : ℝ := P * h / (lams + h * P * h)

/-- Оновлення матриці достатніх статистик (25). -/
noncomputable def rlsCovNext (lams P h : ℝ) : ℝ := lams⁻¹ * (P - rlsGain lams P h * h * P)

theorem rls_denom_pos (lams P h : ℝ) (hl : 0 < lams) (hP : 0 < P) : 0 < lams + h * P * h := by
  have : 0 ≤ h * P * h := by
    have h1 : h * P * h = (h * h) * P := by ring
    rw [h1]
    exact mul_nonneg (mul_self_nonneg h) hP.le
  linarith

theorem rlsCovNext_eq (lams P h : ℝ) (hl : 0 < lams) (hP : 0 < P) :
    rlsCovNext lams P h = P / (lams + h * P * h) := by
  have h1 := mul_inv_cancel₀ (rls_denom_pos lams P h hl hP).ne'
  have h2 := mul_inv_cancel₀ hl.ne'
  unfold rlsCovNext rlsGain
  linear_combination (-(lams⁻¹ * P)) * h1 + (P * (lams + h * P * h)⁻¹) * h2

/-- Інформаційна форма (25): 1/P_n = λ_s/P + h². -/
theorem rls_information_form (lams P h : ℝ) (hl : 0 < lams) (hP : 0 < P) :
    1 / rlsCovNext lams P h = lams / P + h ^ 2 := by
  rw [rlsCovNext_eq lams P h hl hP, one_div_div]
  have hP' := mul_inv_cancel₀ hP.ne'
  linear_combination (h ^ 2) * hP'

/-- Коефіцієнт підсилення (23) дорівнює P_n h. -/
theorem rlsGain_eq (lams P h : ℝ) (hl : 0 < lams) (hP : 0 < P) :
    rlsGain lams P h = rlsCovNext lams P h * h := by
  rw [rlsCovNext_eq lams P h hl hP]
  unfold rlsGain
  ring

theorem rlsCovNext_pos (lams P h : ℝ) (hl : 0 < lams) (hP : 0 < P) : 0 < rlsCovNext lams P h := by
  rw [rlsCovNext_eq lams P h hl hP]
  exact div_pos hP (rls_denom_pos lams P h hl hP)

/-- Менший коефіцієнт забування дає більше P_n (швидше забування статистик). -/
theorem rlsCovNext_anti (lam1 lam2 P h : ℝ) (h₁ : 0 < lam1) (h₁₂ : lam1 < lam2) (hP : 0 < P) :
    rlsCovNext lam2 P h < rlsCovNext lam1 P h := by
  have h₂ : 0 < lam2 := lt_trans h₁ h₁₂
  rw [rlsCovNext_eq lam1 P h h₁ hP, rlsCovNext_eq lam2 P h h₂ hP]
  apply div_lt_div_of_pos_left hP (rls_denom_pos lam1 P h h₁ hP)
  linarith

/-! ### (28)–(30): поліноміальні частини обчислювальних вартостей -/

theorem cost_sector_le_index (Nloc Dh D : ℝ) (hN : 1 ≤ Nloc) (hDh : 0 ≤ Dh) (hD : 0 ≤ D) :
    Dh ^ 2 + Dh * D ≤ Nloc * Dh ^ 2 + Nloc * Dh * D + Dh ^ 3 := by
  have h1 : Dh ^ 2 ≤ Nloc * Dh ^ 2 := by nlinarith [sq_nonneg Dh]
  have h2 : Dh * D ≤ Nloc * Dh * D := by nlinarith [mul_nonneg hDh hD]
  have h3 : 0 ≤ Dh ^ 3 := pow_nonneg hDh 3
  linarith

theorem cost_index_le_global (N Df Nloc Dh D : ℝ) (hN : 0 ≤ N) (hDf : 0 ≤ Df) :
    Nloc * Dh ^ 2 + Nloc * Dh * D + Dh ^ 3 ≤ N * Df + Nloc * Dh ^ 2 + Nloc * Dh * D + Dh ^ 3 := by
  have := mul_nonneg hN hDf
  linarith

/-! ### (31): пам'ять під історію та під достатні статистики -/

noncomputable def memHist (N Dh D : ℝ) : ℝ := 8 * N * (Dh + D) / 2 ^ 20
noncomputable def memStat (S Dh D : ℝ) : ℝ := 8 * S * (Dh ^ 2 + Dh * D) / 2 ^ 20

theorem memory_cross (N S Dh D : ℝ) : memHist N Dh D * (S * Dh) = memStat S Dh D * N := by
  unfold memHist memStat
  ring

/-- M_hist > M_stat тоді й лише тоді, коли N > S·D_h. -/
theorem hist_gt_stat_iff (N S Dh D : ℝ) (hDh : 0 < Dh) (hD : 0 ≤ D) :
    memStat S Dh D < memHist N Dh D ↔ S * Dh < N := by
  unfold memHist memStat
  have hsum : 0 < Dh + D := by linarith
  constructor
  · intro h
    by_contra hc
    have hc' : N ≤ S * Dh := not_lt.mp hc
    have := mul_le_mul_of_nonneg_right hc' hsum.le
    nlinarith
  · intro h
    have := mul_lt_mul_of_pos_right h hsum
    nlinarith

/-! ### (34): середнє відносне зниження RMSE -/

theorem meanRelRed_pos (Nrun : ℕ) (Jprop Jcomp : ℕ → ℝ) (hn : 0 < Nrun)
    (hcomp : ∀ l ∈ Finset.range Nrun, 0 < Jcomp l)
    (hlt : ∀ l ∈ Finset.range Nrun, Jprop l < Jcomp l) :
    0 < (100 / (Nrun : ℝ)) * ∑ l ∈ Finset.range Nrun, (1 - Jprop l / Jcomp l) := by
  apply mul_pos (by positivity)
  apply Finset.sum_pos _ (Finset.nonempty_range_iff.mpr hn.ne')
  intro l hl
  have := (div_lt_one (hcomp l hl)).mpr (hlt l hl)
  linarith

/-! ### (35): ентропія ваг уваги -/

noncomputable def entropy {ι : Type*} (N : Finset ι) (α : ι → ℝ) : ℝ :=
  -∑ j ∈ N, α j * Real.log (α j)

theorem entropy_nonneg {ι : Type*} (N : Finset ι) (α : ι → ℝ)
    (hα : ∀ j ∈ N, 0 < α j) (hsum : ∑ j ∈ N, α j = 1) : 0 ≤ entropy N α := by
  unfold entropy
  rw [neg_nonneg]
  apply Finset.sum_nonpos
  intro j hj
  have hle : α j ≤ 1 := by
    have := Finset.single_le_sum (fun j' hj' => (hα j' hj').le) hj
    linarith
  have hlog : Real.log (α j) ≤ 0 := Real.log_nonpos (hα j hj).le hle
  have h2 := mul_le_mul_of_nonneg_left hlog (hα j hj).le
  rw [mul_zero] at h2
  exact h2

theorem entropy_le_log_card {ι : Type*} (N : Finset ι) (α : ι → ℝ) (hN : N.Nonempty)
    (hα : ∀ j ∈ N, 0 < α j) (hsum : ∑ j ∈ N, α j = 1) :
    entropy N α ≤ Real.log (N.card : ℝ) := by
  unfold entropy
  have hn : (0 : ℝ) < N.card := by exact_mod_cast hN.card_pos
  have hterm : ∀ j ∈ N,
      -(α j * Real.log (α j)) ≤ α j * Real.log (N.card : ℝ) + (1 / (N.card : ℝ) - α j) := by
    intro j hj
    have hpos := hα j hj
    have hx : 0 < 1 / ((N.card : ℝ) * α j) := by positivity
    have hlog := Real.log_le_sub_one_of_pos hx
    rw [Real.log_div one_ne_zero (by positivity), Real.log_one,
      Real.log_mul hn.ne' hpos.ne', zero_sub] at hlog
    have h2 := mul_le_mul_of_nonneg_left hlog hpos.le
    have h3 : α j * (1 / ((N.card : ℝ) * α j) - 1) = 1 / (N.card : ℝ) - α j := by
      have hα' := mul_inv_cancel₀ hpos.ne'
      linear_combination (1 / (N.card : ℝ)) * hα'
    nlinarith [h2, h3]
  calc -∑ j ∈ N, α j * Real.log (α j)
      = ∑ j ∈ N, -(α j * Real.log (α j)) := by rw [Finset.sum_neg_distrib]
    _ ≤ ∑ j ∈ N, (α j * Real.log (N.card : ℝ) + (1 / (N.card : ℝ) - α j)) :=
        Finset.sum_le_sum hterm
    _ = Real.log (N.card : ℝ) := by
        rw [Finset.sum_add_distrib, ← Finset.sum_mul, Finset.sum_sub_distrib, hsum, one_mul,
          Finset.sum_const, nsmul_eq_mul, mul_one_div_cancel hn.ne']
        ring

/-! ### (36): частка компенсованої втрати точності -/

noncomputable def kappa (f1 fm a : ℝ) : ℝ := (f1 - a) / (f1 - fm) * 100

theorem kappa_frozen (f1 fm : ℝ) : kappa f1 fm f1 = 0 := by
  unfold kappa
  simp

theorem kappa_full (f1 fm : ℝ) (h : fm < f1) : kappa f1 fm fm = 100 := by
  unfold kappa
  rw [div_self (sub_pos.mpr h).ne', one_mul]

theorem kappa_between (f1 fm a : ℝ) (h : fm < f1) (ha1 : fm ≤ a) (ha2 : a ≤ f1) :
    0 ≤ kappa f1 fm a ∧ kappa f1 fm a ≤ 100 := by
  unfold kappa
  have hpos : 0 < f1 - fm := sub_pos.mpr h
  constructor
  · apply mul_nonneg _ (by norm_num)
    exact div_nonneg (by linarith) hpos.le
  · have : (f1 - a) / (f1 - fm) ≤ 1 := (div_le_one hpos).mpr (by linarith)
    nlinarith

/-- Числова перевірка табл. 9 (один рівень навчання, режим ×4):
    J_frozen^(1) = 0,3305; J_frozen^(1,2,4) = 0,2945; J_adapt^(1) = 0,3118 → κ = 935/18 ≈ 51,94 %. -/
theorem kappa_table9_x4 : kappa 0.3305 0.2945 0.3118 = 935 / 18 := by
  unfold kappa
  norm_num

/-- Числова перевірка анотації: зниження RMSE проти SAITS у режимі ×4 (0,3039 → 0,2866) лежить у межах 5,69–5,70 %
    (десяткові частки записано дробами 2866/3039, бо norm_num не розкриває ділення десяткових літералів). -/
theorem reduction_vs_saits_x4 :
    (569 : ℝ) / 100 < (1 - 2866 / 3039) * 100 ∧ ((1 : ℝ) - 2866 / 3039) * 100 < 570 / 100 := by
  constructor <;> norm_num

end SRAttELMH

#print axioms SRAttELMH.exact_of_rmse_eq_zero
#print axioms SRAttELMH.mae_le_rmse
#print axioms SRAttELMH.receivedShare_le_one
#print axioms SRAttELMH.residual_eq_estimate_error
#print axioms SRAttELMH.attnWeight_sum_one
#print axioms SRAttELMH.attnWeight_mono
#print axioms SRAttELMH.sectorIndex_mem
#print axioms SRAttELMH.tanh_lt_one'
#print axioms SRAttELMH.offset_converges
#print axioms SRAttELMH.rls_information_form
#print axioms SRAttELMH.hist_gt_stat_iff
#print axioms SRAttELMH.entropy_le_log_card
#print axioms SRAttELMH.kappa_table9_x4

-- кінець файлу (цей рядок захищає останню команду від обрізання при копіюванні)
