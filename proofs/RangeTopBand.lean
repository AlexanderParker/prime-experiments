import Mathlib.Tactic

/-!
# The top-band row law, monotonicity of the `d0` test, and the window form

Ported from the scratch file `c4_0926/TopBand.lean` (k = 1 row law of the live-row test).

Dormancy (range_two_paths A1.8, derived_machine 2.2): write `g = k h + ρ` with `|ρ| < h/2`.
Row `h` is dormant on `g` iff `ρ² < h - B`, or (`ρ` even and `ρ² < 2h - B'`).
Here `ρ` is read off `r = g % h`: `ρ = r` if `2r < h`, else `ρ = r - h` (`balRho g h`).

* `rho_top_band`: in the top band `h < g`, `2g < 3h`, the balanced residue is `g - h`.
* `top_band_dormant_iff`: for every odd `g`, odd `h` with `h < g`, `2g < 3h` (the band
  `2g/3 < h < g`) and `B' - B ≤ h`, row `h` is dormant on `g` iff `(g - h + 1)² < 2g + 1 - B'`.
* `top_band_live_iff`: the live test in the band, with the non-inert condition kept abstract.
* `pass_mono`: the test `T ≤ (d + 1)²` is up-closed in `d ≥ 0`, so the set of `d` passing it is
  `{d : d ≥ d0}` with `d0` its least member.
* `window_form`: `(x - h + 1)² ≥ 2x + 1 - B'` iff `(x - h)² ≥ 2h - B'`, for all integers.
* `top_band_constants`: the constants of case 1 `(B, B') = (30, 30)` and case 19
  `(B, B') = (12, 10)` satisfy `B' - B ≤ h` for every `h ≥ 0`.

No size of `g` or `h` is assumed beyond the stated inequalities; all variables range over `ℤ`.

Naming: the scratch `rho` is renamed `balRho`, since `RangeLine.rho` is already declared in
`RangeGen1` (`primorial q / 2 + 1`).
-/

namespace RangeLine

/-- Balanced residue of `g` mod `h`: `g % h` if `2 (g % h) < h`, else `g % h - h`
(the scratch `rho`, renamed to avoid `RangeGen1.rho`). -/
def balRho (g h : ℤ) : ℤ := if 2 * (g % h) < h then g % h else g % h - h

/-- The dormancy test of the record: row `h` is dormant on `g` with constants `B`, `B'` iff
`ρ² < h - B`, or `ρ` is even and `ρ² < 2h - B'`, where `ρ = balRho g h`. -/
def Dormant (g h B B' : ℤ) : Prop :=
  balRho g h ^ 2 < h - B ∨ (Even (balRho g h) ∧ balRho g h ^ 2 < 2 * h - B')

/-- In the top band: for all integers `g`, `h` with `h < g` and `2g < 3h`, the balanced residue of
`g` mod `h` is `g - h`. -/
theorem rho_top_band (g h : ℤ) (hlt : h < g) (h23 : 2 * g < 3 * h) :
    balRho g h = g - h := by
  have h0 : 0 < h := by linarith
  have hm : g % h = g - h := by
    have e1 : (g - h + h * 1) % h = (g - h) % h := Int.add_mul_emod_self_left _ _ _
    have e2 : g - h + h * 1 = g := by ring
    rw [e2] at e1
    rw [e1]
    exact Int.emod_eq_of_lt (by linarith) (by linarith)
  unfold balRho
  rw [hm]
  split_ifs with hc
  · rfl
  · exfalso; exact hc (by linarith)

/-- Top-band row law: for all integers `g`, `h`, `B`, `B'` with `g` odd, `h` odd, `h < g`,
`2g < 3h` and `B' - B ≤ h`, row `h` is dormant on `g` iff `(g - h + 1)² < 2g + 1 - B'`. -/
theorem top_band_dormant_iff (g h B B' : ℤ) (hg : Odd g) (hh : Odd h)
    (hlt : h < g) (h23 : 2 * g < 3 * h) (hB : B' - B ≤ h) :
    Dormant g h B B' ↔ (g - h + 1) ^ 2 < 2 * g + 1 - B' := by
  have hr := rho_top_band g h hlt h23
  have hev : Even (g - h) := Odd.sub_odd hg hh
  have e : (g - h + 1) ^ 2 = (g - h) ^ 2 + 2 * (g - h) + 1 := by ring
  unfold Dormant
  rw [hr, e]
  constructor
  · rintro (h1 | ⟨_, h2⟩)
    · linarith
    · linarith
  · intro h1
    exact Or.inr ⟨hev, by linarith⟩

/-- The live test in the top band, with the non-inert condition `e_h > 0` kept abstract as an
arbitrary proposition `ni`: for all integers `g`, `h`, `B`, `B'` with `g` odd, `h` odd, `h < g`,
`2g < 3h` and `B' - B ≤ h`, `ni ∧ ¬ Dormant g h B B'` iff `ni ∧ 2g + 1 - B' ≤ (g - h + 1)²`. -/
theorem top_band_live_iff (g h B B' : ℤ) (ni : Prop) (hg : Odd g) (hh : Odd h)
    (hlt : h < g) (h23 : 2 * g < 3 * h) (hB : B' - B ≤ h) :
    (ni ∧ ¬ Dormant g h B B') ↔ (ni ∧ 2 * g + 1 - B' ≤ (g - h + 1) ^ 2) := by
  rw [top_band_dormant_iff g h B B' hg hh hlt h23 hB, not_lt]

/-- Monotonicity of the `d0` test in the distance: for all integers `T`, `d`, `d'` with
`0 ≤ d ≤ d'`, if `T ≤ (d + 1)²` then `T ≤ (d' + 1)²`. So the set of `d ≥ 0` (in particular of
even `d`) passing the test is up-closed, i.e. `{d : d ≥ d0}` with `d0` its least member. -/
theorem pass_mono (T d d' : ℤ) (hd : 0 ≤ d) (hdd : d ≤ d') (h : T ≤ (d + 1) ^ 2) :
    T ≤ (d' + 1) ^ 2 := by
  nlinarith

/-- Window form: the window test of `W` is the same inequality written at the row. For all
integers `x`, `h`, `B'`: `2x + 1 - B' ≤ (x - h + 1)²` iff `2h - B' ≤ (x - h)²`. -/
theorem window_form (x h B' : ℤ) :
    2 * x + 1 - B' ≤ (x - h + 1) ^ 2 ↔ 2 * h - B' ≤ (x - h) ^ 2 := by
  have e : (x - h + 1) ^ 2 = (x - h) ^ 2 + 2 * (x - h) + 1 := by ring
  rw [e]
  constructor <;> intro h1 <;> linarith

/-- The two cases' constants satisfy the band hypothesis `B' - B ≤ h` for every integer `h ≥ 0`:
case 1 `(B, B') = (30, 30)` and case 19 `(B, B') = (12, 10)`. -/
theorem top_band_constants (h : ℤ) (h0 : 0 ≤ h) : (30 : ℤ) - 30 ≤ h ∧ (10 : ℤ) - 12 ≤ h := by
  constructor <;> linarith

end RangeLine
