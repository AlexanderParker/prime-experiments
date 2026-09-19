/-
LadderShortInterval (2026-09-20): the cube ladder from a short-interval twin hypothesis.

The cube ladder (window exponent `e ≥ 3`) follows from a twin pair in every interval
`(x, x + x^θ]` with `θ ≤ 1 - 1/e`; hence chains of every length and infinitely many twin primes.

The mechanism.  A rung with exponent `e` from a twin centre `s` is a twin centre `s'` with both
members inside the window `((s-1)^e, (s+1)^e)`.  Apply the short-interval hypothesis at
`x = (s-1)^e`: it gives a twin pair in `(x, x + x^θ]`.  The pair sits above `(s-1)^e` by
construction, and below `(s+1)^e` because

  `x^θ = (s-1)^(eθ) ≤ (s-1)^(e-1)`            (`eθ ≤ e - 1`, base `s - 1 ≥ 1`)

while the window is wider than that by the binomial theorem's second term:

  `(s+1)^e = ((s-1) + 2)^e ≥ (s-1)^e + 2e (s-1)^(e-1) > (s-1)^e + (s-1)^(e-1)`.

So every large twin centre has a rung; dependent choice then strings the rungs into a chain of
any length, and `twins_unbounded_of_chainsPow` (LadderDepth) turns chains into infinitely many
twin primes.  Nothing here uses `θ > 0` beyond the statement; the whole weight is `θ ≤ 1 - 1/e`.
-/
import LadderDepth
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace TwinLadder

/-- **The short-interval twin hypothesis with exponent `θ`**, in lower-bound form: for all
sufficiently large `x` there is a twin pair with both members in `(x, x + x^θ]`, i.e. a twin
centre `s` with `x < s - 1` and `s + 1 ≤ x + x^θ`. -/
def ShortInterval (θ : ℝ) : Prop :=
  ∀ᶠ x : ℕ in Filter.atTop, ∃ s : ℕ, TwinCentre s ∧ x < s - 1 ∧ (s + 1 : ℝ) ≤ x + (x : ℝ) ^ θ

/-! ### Two elementary inequalities: the window's width and the interval's length -/

/-- The binomial theorem's first two terms: `(a + 2)^(d+1) ≥ a^(d+1) + 2(d+1) a^d`. -/
theorem add_two_pow_ge (a d : ℕ) : a ^ (d + 1) + 2 * (d + 1) * a ^ d ≤ (a + 2) ^ (d + 1) := by
  induction d with
  | zero => norm_num
  | succ d ih =>
    have key := Nat.mul_le_mul_right (a + 2) ih
    have h1 : (a + 2) ^ (d + 2) = (a + 2) ^ (d + 1) * (a + 2) := pow_succ _ _
    have h2 : a ^ (d + 2) = a ^ (d + 1) * a := pow_succ _ _
    have h3 : a ^ (d + 1) = a ^ d * a := pow_succ _ _
    rw [h1, h2]
    nlinarith [key, h3, Nat.zero_le (a ^ d), Nat.zero_le d, Nat.zero_le a]

/-- **The window is wide**: `(a + 2)^e ≥ a^e + 2e a^(e-1)` for `e ≥ 1`. -/
theorem window_width (a e : ℕ) (he : 1 ≤ e) :
    a ^ e + 2 * e * a ^ (e - 1) ≤ (a + 2) ^ e := by
  obtain ⟨d, rfl⟩ : ∃ d, e = d + 1 := ⟨e - 1, by omega⟩
  simpa using add_two_pow_ge a d

/-- **The interval is short**: `(a^e)^θ ≤ a^(e-1)` for `a ≥ 1`, `e ≥ 1` and `θ ≤ 1 - 1/e`, since
`(a^e)^θ = a^(eθ)` and `eθ ≤ e - 1`. -/
theorem interval_length (a e : ℕ) (ha : 1 ≤ a) (he : 1 ≤ e) (θ : ℝ) (hθ : θ ≤ 1 - 1 / e) :
    ((a ^ e : ℕ) : ℝ) ^ θ ≤ ((a ^ (e - 1) : ℕ) : ℝ) := by
  have ha' : (1 : ℝ) ≤ a := by exact_mod_cast ha
  have he' : (1 : ℝ) ≤ e := by exact_mod_cast he
  have he0 : (0 : ℝ) < e := by linarith
  push_cast
  rw [← Real.rpow_natCast (a : ℝ) e, ← Real.rpow_mul (by linarith),
    ← Real.rpow_natCast (a : ℝ) (e - 1)]
  apply Real.rpow_le_rpow_of_exponent_le ha'
  rw [Nat.cast_sub he, Nat.cast_one]
  have hmul : (e : ℝ) * θ ≤ e * (1 - 1 / e) := mul_le_mul_of_nonneg_left hθ he0.le
  rw [mul_sub, mul_one, mul_one_div_cancel he0.ne'] at hmul
  exact hmul

/-! ### The rung: every large twin centre has one -/

/-- A rung with exponent `e ≥ 1` climbs: `s' > (s-1)^e ≥ s - 1`, so `s' > s`. -/
theorem rungPow_gt {e s s' : ℕ} (he : 1 ≤ e) (hs : TwinCentre s) (h : RungPow e s s') :
    s < s' := by
  have h6 := twinCentre_ge_six hs
  have h1 : (s - 1) ^ e < s' - 1 := h.2.1
  have h2 : s - 1 ≤ (s - 1) ^ e := Nat.le_self_pow (by omega) _
  omega

/-- **The rung from the short interval.**  For `e ≥ 3` and `θ ≤ 1 - 1/e`, every twin centre
`s` beyond some bound `S` has a rung with exponent `e`: the hypothesis at `x = (s-1)^e` gives a
twin pair in `((s-1)^e, (s-1)^e + ((s-1)^e)^θ]`, and that interval lies inside the window
`((s-1)^e, (s+1)^e)` by `interval_length` and `window_width`. -/
theorem rungPow_of_shortInterval (e : ℕ) (he : 3 ≤ e) (θ : ℝ) (hθ0 : 0 < θ)
    (hθ : θ ≤ 1 - 1 / e) (h : ShortInterval θ) :
    ∃ S : ℕ, ∀ s, TwinCentre s → S ≤ s → ∃ s', RungPow e s s' := by
  obtain ⟨X, hX⟩ := Filter.eventually_atTop.mp h
  refine ⟨X + 1, fun s hs hSs => ?_⟩
  have h6 := twinCentre_ge_six hs
  obtain ⟨a, rfl⟩ : ∃ a, s = a + 1 := ⟨s - 1, by omega⟩
  have ha1 : 1 ≤ a := by omega
  have he1 : 1 ≤ e := by omega
  -- the hypothesis applies at `x = a^e ≥ a ≥ X`
  have hxX : X ≤ a ^ e := le_trans (by omega) (Nat.le_self_pow (by omega) a)
  obtain ⟨s', hs', hlt, hle⟩ := hX (a ^ e) hxX
  refine ⟨s', hs', by simpa using hlt, ?_⟩
  -- the upper end: `s' + 1 ≤ a^e + (a^e)^θ ≤ a^e + a^(e-1) < a^e + 2e a^(e-1) ≤ (a+2)^e`
  have hshort := interval_length a e ha1 he1 θ hθ
  have hwide := window_width a e he1
  have hpos : 0 < a ^ (e - 1) := Nat.pow_pos (by omega)
  have hgap : a ^ (e - 1) < 2 * e * a ^ (e - 1) := by nlinarith
  have key : (s' : ℝ) + 1 < ((a : ℝ) + 2) ^ e := by
    have h1 : ((s' : ℕ) : ℝ) + 1 ≤ ((a ^ e : ℕ) : ℝ) + ((a ^ e : ℕ) : ℝ) ^ θ := hle
    have h2 : ((a ^ (e - 1) : ℕ) : ℝ) < ((2 * e * a ^ (e - 1) : ℕ) : ℝ) := by exact_mod_cast hgap
    have h3 : ((a ^ e + 2 * e * a ^ (e - 1) : ℕ) : ℝ) ≤ (((a + 2) ^ e : ℕ) : ℝ) := by
      exact_mod_cast hwide
    push_cast at h1 h2 h3 hshort
    linarith
  have hnat : s' + 1 < (a + 2) ^ e := by exact_mod_cast key
  have e2 : a + 1 + 1 = a + 2 := by omega
  rw [e2]
  exact hnat

/-! ### Chains of every length, and twins unbounded -/

/-- The short-interval hypothesis alone supplies a twin centre above any bound (apply it at
`x = X + S`). -/
theorem exists_twinCentre_ge_of_shortInterval {θ : ℝ} (h : ShortInterval θ) (S : ℕ) :
    ∃ s : ℕ, TwinCentre s ∧ S ≤ s := by
  obtain ⟨X, hX⟩ := Filter.eventually_atTop.mp h
  obtain ⟨s, hs, hlt, _⟩ := hX (X + S) (by omega)
  exact ⟨s, hs, by omega⟩

/-- **A path of `e`-rungs from a rung above a bound**, by dependent choice (the shape of
`path_of_ladderHyp_above`): if every twin centre at least `S₀` has an `e`-rung, any twin centre
`s₀ ≥ S₀` starts an infinite path, each node staying above `S₀` because rungs climb. -/
theorem pathPow_of_rungs_above (e S₀ s₀ : ℕ) (he : 1 ≤ e) (hs₀ : TwinCentre s₀) (hS : S₀ ≤ s₀)
    (hL : ∀ s : ℕ, TwinCentre s → S₀ ≤ s → ∃ s' : ℕ, RungPow e s s') :
    ∃ f : ℕ → ℕ, f 0 = s₀ ∧ ∀ k : ℕ, RungPow e (f k) (f (k + 1)) := by
  classical
  -- the state carries the twin-centre and bound facts along
  let next : {s : ℕ // TwinCentre s ∧ S₀ ≤ s} → {s : ℕ // TwinCentre s ∧ S₀ ≤ s} :=
    fun p =>
      let h := hL p.1 p.2.1 p.2.2
      ⟨Classical.choose h, (Classical.choose_spec h).1,
        by have := rungPow_gt he p.2.1 (Classical.choose_spec h); omega⟩
  let g : ℕ → {s : ℕ // TwinCentre s ∧ S₀ ≤ s} :=
    fun k => Nat.rec ⟨s₀, hs₀, hS⟩ (fun _ p => next p) k
  refine ⟨fun k => (g k).1, rfl, ?_⟩
  intro k
  show RungPow e (g k).1 (next (g k)).1
  exact Classical.choose_spec (hL (g k).1 (g k).2.1 (g k).2.2)

/-- **Chains of every length from the short interval.**  A twin centre above the rung bound
exists by the hypothesis itself; the path from it, cut at length `n`, is a chain. -/
theorem chainsPow_of_shortInterval (e : ℕ) (he : 3 ≤ e) (θ : ℝ) (hθ0 : 0 < θ)
    (hθ : θ ≤ 1 - 1 / e) (h : ShortInterval θ) :
    ∀ n : ℕ, ∃ f : ℕ → ℕ, ChainPow e n f := by
  intro n
  obtain ⟨S, hS⟩ := rungPow_of_shortInterval e he θ hθ0 hθ h
  obtain ⟨s₀, hs₀, hSs₀⟩ := exists_twinCentre_ge_of_shortInterval h S
  obtain ⟨f, hf0, hf⟩ := pathPow_of_rungs_above e S s₀ (by omega) hs₀ hSs₀ hS
  exact ⟨f, hf0 ▸ hs₀, fun k _ => hf k⟩

/-- **Twins unbounded from the short interval.**  A twin pair in every `(x, x + x^θ]` with
`θ ≤ 1 - 1/e` for some `e ≥ 3` gives infinitely many twin primes, through the `e`-ladder. -/
theorem twins_unbounded_of_shortInterval (e : ℕ) (he : 3 ≤ e) (θ : ℝ) (hθ0 : 0 < θ)
    (hθ : θ ≤ 1 - 1 / e) (h : ShortInterval θ) :
    ∀ N : ℕ, ∃ m : ℕ, N < 6 * m - 1 ∧ (6 * m - 1).Prime ∧ (6 * m + 1).Prime :=
  twins_unbounded_of_chainsPow e (by omega) (chainsPow_of_shortInterval e he θ hθ0 hθ h)


end TwinLadder
