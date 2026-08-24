/-
The two-teeth kill spacing law (found live 2026-08-24, round 21).

A gear `q` kills the slot line at exactly two residues - its teeth - at
`u` and `q - u` (`6u = q -+ 1`, `u = 6^{-1} mod q` up to sign). Between
consecutive kills there are therefore only TWO possible spacings, and they
alternate:

    from a tooth at `u`     : the next kill is `q - 2u` slots on;
    from a tooth at `q - u` : the next kill is `2u` slots on;

so consecutive kill spacings lie in `{2u, q - 2u}`, two consecutive
spacings always sum to exactly `q` (the alternation), and the minimum
spacing is `2u ~ q/3`. This is the closed form behind Constructor's fuel
bound `fuel <= ~3L/q'`: along a stretch of length `L`, gear `q`'s kills
number at most `~L/(2u) ~ 3L/q`.

Elementary modular arithmetic, fully abstract in `q` and `u`; the gear
side conditions are discharged by `6u = q -+ 1` in the `_gear` forms.
-/

import MergeLaw
import Mathlib.Data.Nat.ModEq

namespace TwoTeeth

/-- A kill of gear `q` (teeth `{u, q - u}`) at slot `x`. -/
def Kill (q u x : ℕ) : Prop := x % q = u ∨ x % q = q - u

/-- **From a low tooth** (`x = u mod q`): the next kill is exactly
`q - 2u` slots on, and lands on the high tooth. -/
theorem next_kill_of_lo {q u x y : ℕ} (hu : 0 < u) (h4u : 4 * u < q)
    (hx : x % q = u) (hy : Kill q u y) (hxy : x < y)
    (hmin : ∀ z, x < z → z < y → ¬ Kill q u z) :
    y - x = q - 2 * u ∧ y % q = q - u := by
  -- the witness kill `x + (q - 2u)` bounds `y` from above
  have hw : (x + (q - 2 * u)) % q = q - u := by
    rw [Nat.add_mod, hx, Nat.mod_eq_of_lt (show q - 2 * u < q by omega)]
    have e : u + (q - 2 * u) = q - u := by omega
    rw [e]
    exact Nat.mod_eq_of_lt (by omega)
  have hylew : y ≤ x + (q - 2 * u) := by
    by_contra hc
    exact hmin _ (by omega) (by omega) (Or.inr hw)
  rcases hy with hyu | hyu
  · -- same residue: q divides y - x, impossible below q - 2u
    have hdvd : q ∣ y - x := (Nat.modEq_iff_dvd' (le_of_lt hxy)).mp
      (show x % q = y % q by rw [hx, hyu])
    have := Nat.le_of_dvd (by omega) hdvd
    omega
  · have hs := MergeLaw.sub_mod_eq (le_of_lt hxy) (show u < q by omega)
      (show q - u < q by omega) hx hyu
    have e : q + (q - u) - u = 2 * q - 2 * u := by omega
    rw [e] at hs
    have e2 : (2 * q - 2 * u) % q = q - 2 * u := by
      rw [Nat.mod_eq_sub_mod (by omega)]
      have e3 : 2 * q - 2 * u - q = q - 2 * u := by omega
      rw [e3]
      exact Nat.mod_eq_of_lt (by omega)
    rw [e2, Nat.mod_eq_of_lt (show y - x < q by omega)] at hs
    exact ⟨hs, hyu⟩

/-- **From a high tooth** (`x = q - u mod q`): the next kill is exactly
`2u` slots on, and lands on the low tooth. -/
theorem next_kill_of_hi {q u x y : ℕ} (hu : 0 < u) (h4u : 4 * u < q)
    (hx : x % q = q - u) (hy : Kill q u y) (hxy : x < y)
    (hmin : ∀ z, x < z → z < y → ¬ Kill q u z) :
    y - x = 2 * u ∧ y % q = u := by
  have hw : (x + 2 * u) % q = u := by
    rw [Nat.add_mod, hx, Nat.mod_eq_of_lt (show 2 * u < q by omega)]
    have e : q - u + 2 * u = q + u := by omega
    rw [e, Nat.add_mod_left]
    exact Nat.mod_eq_of_lt (by omega)
  have hylew : y ≤ x + 2 * u := by
    by_contra hc
    exact hmin _ (by omega) (by omega) (Or.inl hw)
  rcases hy with hyu | hyu
  · have hs := MergeLaw.sub_mod_eq (le_of_lt hxy) (show q - u < q by omega)
      (show u < q by omega) hx hyu
    have e : q + u - (q - u) = 2 * u := by omega
    rw [e, Nat.mod_eq_of_lt (show 2 * u < q by omega),
      Nat.mod_eq_of_lt (show y - x < q by omega)] at hs
    exact ⟨hs, hyu⟩
  · have hdvd : q ∣ y - x := (Nat.modEq_iff_dvd' (le_of_lt hxy)).mp
      (show x % q = y % q by rw [hx, hyu])
    have := Nat.le_of_dvd (by omega) hdvd
    omega

/-- **The two-teeth kill spacing law**: consecutive kills of gear `q` are
spaced by `2u` or `q - 2u` - nothing else, ever. -/
theorem kill_spacing {q u x y : ℕ} (hu : 0 < u) (h4u : 4 * u < q)
    (hx : Kill q u x) (hy : Kill q u y) (hxy : x < y)
    (hmin : ∀ z, x < z → z < y → ¬ Kill q u z) :
    y - x = 2 * u ∨ y - x = q - 2 * u := by
  rcases hx with hx | hx
  · exact Or.inr (next_kill_of_lo hu h4u hx hy hxy hmin).1
  · exact Or.inl (next_kill_of_hi hu h4u hx hy hxy hmin).1

/-- **The minimum spacing is `2u ~ q/3`** - the closed form behind the
fuel bound `fuel <= ~3L/q`. -/
theorem kill_spacing_min {q u x y : ℕ} (hu : 0 < u) (h4u : 4 * u < q)
    (hx : Kill q u x) (hy : Kill q u y) (hxy : x < y)
    (hmin : ∀ z, x < z → z < y → ¬ Kill q u z) :
    2 * u ≤ y - x := by
  rcases kill_spacing hu h4u hx hy hxy hmin with h | h <;> omega

/-- **The alternation**: two consecutive kill spacings sum to exactly `q` -
every second kill is a full period on. -/
theorem kill_period {q u x y z : ℕ} (hu : 0 < u) (h4u : 4 * u < q)
    (hx : Kill q u x) (hy : Kill q u y) (hz : Kill q u z)
    (hxy : x < y) (hyz : y < z)
    (hmin1 : ∀ w, x < w → w < y → ¬ Kill q u w)
    (hmin2 : ∀ w, y < w → w < z → ¬ Kill q u w) :
    z - x = q := by
  rcases hx with hx | hx
  · obtain ⟨h1, hyres⟩ := next_kill_of_lo hu h4u hx hy hxy hmin1
    obtain ⟨h2, _⟩ := next_kill_of_hi hu h4u hyres hz hyz hmin2
    omega
  · obtain ⟨h1, hyres⟩ := next_kill_of_hi hu h4u hx hy hxy hmin1
    obtain ⟨h2, _⟩ := next_kill_of_lo hu h4u hyres hz hyz hmin2
    omega

/-! ## Gear forms: `6u = q -+ 1` discharges every side condition -/

/-- Any gear of the machine (`6u' = q' -+ 1`, `q' >= 5`) satisfies the
side conditions of the spacing law. -/
theorem gear_side {q u : ℕ} (hu6 : 6 * u + 1 = q ∨ 6 * u = q + 1)
    (hq : 5 ≤ q) : 0 < u ∧ 4 * u < q := by omega

/-- The spacing law at a machine gear: spacings lie in `{2u', q' - 2u'}`. -/
theorem kill_spacing_gear {q u x y : ℕ}
    (hu6 : 6 * u + 1 = q ∨ 6 * u = q + 1) (hq : 5 ≤ q)
    (hx : Kill q u x) (hy : Kill q u y) (hxy : x < y)
    (hmin : ∀ z, x < z → z < y → ¬ Kill q u z) :
    y - x = 2 * u ∨ y - x = q - 2 * u :=
  kill_spacing (gear_side hu6 hq).1 (gear_side hu6 hq).2 hx hy hxy hmin

/-- The minimum spacing at a machine gear: at least `2u' ~ q'/3`. -/
theorem kill_spacing_min_gear {q u x y : ℕ}
    (hu6 : 6 * u + 1 = q ∨ 6 * u = q + 1) (hq : 5 ≤ q)
    (hx : Kill q u x) (hy : Kill q u y) (hxy : x < y)
    (hmin : ∀ z, x < z → z < y → ¬ Kill q u z) :
    2 * u ≤ y - x :=
  kill_spacing_min (gear_side hu6 hq).1 (gear_side hu6 hq).2 hx hy hxy hmin

/-! ## T1-T5 in Constructor's frame (docs/novel/two-teeth-kill-spacing.md) -/

/-- **T1, the letters identity**: the two spacing letters sum to the
modulus, and the teeth are the 6-inverses up to sign (`6u' = -+1 mod q'`). -/
theorem teeth_letters {q u : ℕ} (hu6 : 6 * u + 1 = q ∨ 6 * u = q + 1)
    (hq : 5 ≤ q) : 2 * u + (q - 2 * u) = q ∧
      ((6 * u) % q = q - 1 ∨ (6 * u) % q = 1) := by
  obtain ⟨hu, h4u⟩ := gear_side hu6 hq
  constructor
  · omega
  · rcases hu6 with h | h
    · left
      have e : 6 * u = q - 1 := by omega
      rw [e]
      exact Nat.mod_eq_of_lt (by omega)
    · right
      have e : 6 * u = q + 1 := by omega
      rw [e, Nat.add_mod_left]
      exact Nat.mod_eq_of_lt (by omega)

/-- **T2 + T3 from a low tooth** (padded links transparent): any later kill
sits either a multiple of `q` on (still the low tooth) or in the `q - 2u`
class (now the high tooth). A `2u`-class spacing from a low tooth is
impossible - the sign alternation is forced. -/
theorem spacing_from_lo {q u x y : ℕ} (hu : 0 < u) (h4u : 4 * u < q)
    (hx : x % q = u) (hy : Kill q u y) (hxy : x ≤ y) :
    ((y - x) % q = 0 ∧ y % q = u) ∨
      ((y - x) % q = q - 2 * u ∧ y % q = q - u) := by
  rcases hy with hyu | hyu
  · left
    refine ⟨?_, hyu⟩
    have hs := MergeLaw.sub_mod_eq hxy (show u < q by omega)
      (show u < q by omega) hx hyu
    have e : q + u - u = q := by omega
    rw [e, Nat.mod_self] at hs
    exact hs
  · right
    refine ⟨?_, hyu⟩
    have hs := MergeLaw.sub_mod_eq hxy (show u < q by omega)
      (show q - u < q by omega) hx hyu
    have e : q + (q - u) - u = 2 * q - 2 * u := by omega
    rw [e] at hs
    have e2 : (2 * q - 2 * u) % q = q - 2 * u := by
      rw [Nat.mod_eq_sub_mod (by omega)]
      have e3 : 2 * q - 2 * u - q = q - 2 * u := by omega
      rw [e3]
      exact Nat.mod_eq_of_lt (by omega)
    rw [e2] at hs
    exact hs

/-- **T2 + T3 from a high tooth**: only the `0` class (stay) or the `2u`
class (move to the low tooth). -/
theorem spacing_from_hi {q u x y : ℕ} (hu : 0 < u) (h4u : 4 * u < q)
    (hx : x % q = q - u) (hy : Kill q u y) (hxy : x ≤ y) :
    ((y - x) % q = 0 ∧ y % q = q - u) ∨
      ((y - x) % q = 2 * u ∧ y % q = u) := by
  rcases hy with hyu | hyu
  · right
    refine ⟨?_, hyu⟩
    have hs := MergeLaw.sub_mod_eq hxy (show q - u < q by omega)
      (show u < q by omega) hx hyu
    have e : q + u - (q - u) = 2 * u := by omega
    rw [e, show (2 * u) % q = 2 * u from Nat.mod_eq_of_lt (by omega)] at hs
    exact hs
  · left
    refine ⟨?_, hyu⟩
    have hs := MergeLaw.sub_mod_eq hxy (show q - u < q by omega)
      (show q - u < q by omega) hx hyu
    have e : q + (q - u) - (q - u) = q := by omega
    rw [e, Nat.mod_self] at hs
    exact hs

/-- **T4, general form**: ANY two distinct kills of gear `q` are at least
`2u` apart - consecutive or not, padded or not. -/
theorem kills_gap_ge {q u x y : ℕ} (hu : 0 < u) (h4u : 4 * u < q)
    (hx : Kill q u x) (hy : Kill q u y) (hxy : x < y) : 2 * u ≤ y - x := by
  have hmod : (y - x) % q = 0 ∨ (y - x) % q = 2 * u ∨
      (y - x) % q = q - 2 * u := by
    rcases hx with hxu | hxu
    · rcases spacing_from_lo hu h4u hxu hy (le_of_lt hxy) with ⟨h, _⟩ | ⟨h, _⟩
      · exact Or.inl h
      · exact Or.inr (Or.inr h)
    · rcases spacing_from_hi hu h4u hxu hy (le_of_lt hxy) with ⟨h, _⟩ | ⟨h, _⟩
      · exact Or.inl h
      · exact Or.inr (Or.inl h)
  exact MergeLaw.floor_of_mod (by omega) (by omega) hmod

/-- A strictly increasing chain starts at its first element. -/
theorem chain_mono {x : ℕ → ℕ} {k : ℕ}
    (hmono : ∀ i, i + 1 < k → x i < x (i + 1)) :
    ∀ j, j < k → x 0 ≤ x j := by
  intro j
  induction j with
  | zero => intro _; exact le_refl _
  | succ j ihj =>
    intro hj
    have h3 := hmono j (by omega)
    have h4 := ihj (by omega)
    omega

/-- **T5, the fuel-span law**: `k` chained kills span at least
`2u * (k - 1)` slots. -/
theorem fuel_span_cap {q u : ℕ} (x : ℕ → ℕ) (hu : 0 < u) (h4u : 4 * u < q) :
    ∀ k, 1 ≤ k → (∀ i, i + 1 < k → x i < x (i + 1)) →
      (∀ i, i < k → Kill q u (x i)) →
      2 * u * (k - 1) ≤ x (k - 1) - x 0 := by
  intro k
  induction k with
  | zero => intro h; omega
  | succ k ih =>
    intro _ hmono hkill
    by_cases hk1 : 1 ≤ k
    · obtain ⟨m, rfl⟩ : ∃ m, k = m + 1 := ⟨k - 1, by omega⟩
      have ihk := ih (by omega) (fun i hi => hmono i (by omega))
        (fun i hi => hkill i (by omega))
      rw [Nat.add_sub_cancel] at ihk ⊢
      have hm : x m < x (m + 1) := hmono m (by omega)
      have hstep : 2 * u ≤ x (m + 1) - x m :=
        kills_gap_ge hu h4u (hkill m (by omega)) (hkill (m + 1) (by omega)) hm
      have h0m : x 0 ≤ x m := chain_mono hmono m (by omega)
      have hmul : 2 * u * (m + 1) = 2 * u * m + 2 * u := Nat.mul_succ (2 * u) m
      omega
    · have hk0 : k = 0 := by omega
      subst hk0
      simp

/-- **T5, count form**: at most `1 + span / 2u'` kills fit in a chain of
span `L` - the closed-form fuel cap `~3L/q'`. -/
theorem fuel_le {q u k : ℕ} (x : ℕ → ℕ) (hu : 0 < u) (h4u : 4 * u < q)
    (hk : 1 ≤ k) (hmono : ∀ i, i + 1 < k → x i < x (i + 1))
    (hkill : ∀ i, i < k → Kill q u (x i)) :
    k ≤ 1 + (x (k - 1) - x 0) / (2 * u) := by
  have h := fuel_span_cap x hu h4u k hk hmono hkill
  have e : (k - 1) * (2 * u) = 2 * u * (k - 1) := Nat.mul_comm _ _
  have hc : (k - 1) * (2 * u) ≤ x (k - 1) - x 0 := by omega
  have h2 : k - 1 ≤ (x (k - 1) - x 0) / (2 * u) :=
    (Nat.le_div_iff_mul_le (by omega)).mpr hc
  omega

end TwoTeeth
