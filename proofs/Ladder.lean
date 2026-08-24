/-
THE (D) LADDER: four consecutive machine steps, hypothesis-free (round 22).

`Machine23.lean` (round 21) proved (D) at `alpha = 3` for the 19->23 step
end to end, and observed that the per-step recipe is mechanical: scan the OLD
machine's ladder in the kernel, then instantiate R39. This file turns the
recipe on the three steps BELOW it, so the ladder is now contiguous from the
bottom of the machine sequence:

    11 -> 13   g13 n <= F(11) + 13 = 7 + 13 = 20     (`D_at_11_13`)
    13 -> 17   g17 n <= F(13) + 17 = 11 + 17 = 28    (`D_at_13_17`)
    17 -> 19   g19 n <= F(17) + 19 = 18 + 19 = 37    (`D_at_17_19`)
    19 -> 23   g23 n <= F(19) + 23 = 25 + 23 = 48    (`Machine23.D_at_19_23`)

Every one of them is a theorem about the machine's OWN gap sequence, with no
hypotheses at all: the merge law (`MergeLaw.newgap_le_step`) plus the old
machine's kernel-scanned `F_2` and qualifying spectrum. Nothing empirical is
assumed anywhere in the chain; the only inputs are the four period scans
(385, 5005, 85085, 1616615 residues) that live in `Machine11`, `Machine13Q`,
`Machine17Q`, `Machine19Q`.

R39's own form `F(M + q') <= max (F2, max_j qualmax_j)` is recorded per rung
as `g13_le`, `g17_le`, `g19_le_of_17` - the bound the criterion actually
produces, which at each step is strictly inside the (D) budget:

    step      max(F2, max_j Q_j)      budget F + q'     margin
    11->13    max(11, 20) = 20        20                 0   (TIGHT)
    13->17    max(16, 26) = 26        28                 2
    17->19    max(25, 35) = 35        37                 2
    19->23    max(31, 47) = 47        48                 1

The two steps ABOVE the scannable range are recorded as hypothesis-explicit
instantiations - `D_at_23_29` and `D_at_37_41` - carrying their census inputs
as named hypotheses, so exactly what is assumed is visible in the statement.
See formalist.md round 22 for the will-not-close verdict on making 23->29
hypothesis-free (it needs a 37.2M-tuple period scan; the merge law cannot
supply its own input at the next rung).
-/

import Machine11
import Machine13Q
import Machine17Q
import Machine23
import MergeLaw

namespace Ladder

/-! ## Rung 1: 11 -> 13 -/

/-- Gear 13 kills slot `k`: teeth at `u' = 2` and `13 - u' = 11`. -/
def Killed13 (k : ℕ) : Prop := k % 13 = 2 ∨ k % 13 = 11

/-- The teeth ARE the divisibility conditions. -/
theorem killed13_iff {k : ℕ} (hk : 1 ≤ k) :
    Killed13 k ↔ (13 ∣ Census.lo k ∨ 13 ∣ Census.hi k) := by
  simp only [Killed13, Census.lo, Census.hi]
  omega

theorem not_killed13 {k : ℕ} (hk : 1 ≤ k) (h : Machine13.Exposed13 k) :
    ¬ Killed13 k :=
  fun hK => ((killed13_iff hk).mp hK).elim h.2.2.2.2.2.2.1 h.2.2.2.2.2.2.2

theorem exposed13_of {k : ℕ} (hk : 1 ≤ k) (h : Machine11.Exposed11 k)
    (hnk : ¬ Killed13 k) : Machine13.Exposed13 k :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2,
    fun hd => hnk ((killed13_iff hk).mpr (Or.inl hd)),
    fun hd => hnk ((killed13_iff hk).mpr (Or.inr hd))⟩

theorem exposed11_of_13 {k : ℕ} (h : Machine13.Exposed13 k) :
    Machine11.Exposed11 k :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1⟩

/-- **R39 at the 11->13 step**: every gap of machine 13 is at most
`max (F_2(11), max_j Q_j(11; 4)) = max (11, 20) = 20`. -/
theorem g13_le (n : ℕ) : Machine13.g13 n ≤ 20 :=
  MergeLaw.newgap_le_step
    (ExO := Machine11.Exposed11) (ExN := Machine13.Exposed13) (Kap := Killed13)
    (posO := Machine11.opSeq) (posN := Machine13.opSeq) (g := Machine11.g11)
    (q := 13) (u := 2) (B := 20) (F2 := 11) (Q := fun _ => 20)
    (fun _ => rfl) Machine11.opSeq_lt_succ Machine11.opSeq_pos
    Machine11.opSeq_exposed (fun _ hx hE => Machine11.opSeq_surj hx hE)
    Machine13.opSeq_pos Machine13.opSeq_lt_succ Machine13.opSeq_exposed
    (fun m x => Machine13.opSeq_gap_empty m x)
    (fun _ hx hE => not_killed13 hx hE)
    (fun _ hx hE hnk => exposed13_of hx hE hnk)
    (fun _ h => exposed11_of_13 h)
    (fun _ hk => by rcases hk with h | h <;> omega)
    (by omega) (by omega)
    Machine11.spectrum_two (by omega)
    Machine11.qual_bound_all (fun _ => le_refl 20) n

/-- **(D) at `alpha = 3` at the 11->13 step, hypothesis-free**: every gap of
machine 13 is at most `F(11) + q' = 7 + 13 = 20`. The rung is TIGHT - the
criterion value equals the budget exactly, because `Q_5(11; 4) = 20`. -/
theorem D_at_11_13 (n : ℕ) : Machine13.g13 n ≤ 7 + 13 :=
  le_trans (g13_le n) (by omega)

/-! ## Rung 2: 13 -> 17 -/

/-- Gear 17 kills slot `k`: teeth at `u' = 3` and `17 - u' = 14`. -/
def Killed17 (k : ℕ) : Prop := k % 17 = 3 ∨ k % 17 = 14

theorem killed17_iff {k : ℕ} (hk : 1 ≤ k) :
    Killed17 k ↔ (17 ∣ Census.lo k ∨ 17 ∣ Census.hi k) := by
  simp only [Killed17, Census.lo, Census.hi]
  omega

theorem not_killed17 {k : ℕ} (hk : 1 ≤ k) (h : Machine17.Exposed17 k) :
    ¬ Killed17 k :=
  fun hK => ((killed17_iff hk).mp hK).elim h.2.2.2.2.2.2.2.2.1
    h.2.2.2.2.2.2.2.2.2

theorem exposed17_of {k : ℕ} (hk : 1 ≤ k) (h : Machine13.Exposed13 k)
    (hnk : ¬ Killed17 k) : Machine17.Exposed17 k :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1,
    h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2,
    fun hd => hnk ((killed17_iff hk).mpr (Or.inl hd)),
    fun hd => hnk ((killed17_iff hk).mpr (Or.inr hd))⟩

theorem exposed13_of_17 {k : ℕ} (h : Machine17.Exposed17 k) :
    Machine13.Exposed13 k :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1,
    h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1⟩

/-- **R39 at the 13->17 step**: every gap of machine 17 is at most
`max (F_2(13), max_j Q_j(13; 6)) = max (16, 26) = 26`. (Census
cross-check: `F(17) = 18`, so 26 is a true, untight bound.) -/
theorem g17_le (n : ℕ) : Machine17.g17 n ≤ 26 :=
  MergeLaw.newgap_le_step
    (ExO := Machine13.Exposed13) (ExN := Machine17.Exposed17) (Kap := Killed17)
    (posO := Machine13.opSeq) (posN := Machine17.opSeq) (g := Machine13.g13)
    (q := 17) (u := 3) (B := 26) (F2 := 16) (Q := fun _ => 26)
    (fun _ => rfl) Machine13.opSeq_lt_succ Machine13.opSeq_pos
    Machine13.opSeq_exposed (fun _ hx hE => Machine13.opSeq_surj hx hE)
    Machine17.opSeq_pos Machine17.opSeq_lt_succ Machine17.opSeq_exposed
    (fun m x => Machine17.opSeq_gap_empty m x)
    (fun _ hx hE => not_killed17 hx hE)
    (fun _ hx hE hnk => exposed17_of hx hE hnk)
    (fun _ h => exposed13_of_17 h)
    (fun _ hk => by rcases hk with h | h <;> omega)
    (by omega) (by omega)
    Machine13.spectrum_two (by omega)
    Machine13.qual_bound_all (fun _ => le_refl 26) n

/-- **(D) at `alpha = 3` at the 13->17 step, hypothesis-free**: every gap of
machine 17 is at most `F(13) + q' = 11 + 17 = 28`. -/
theorem D_at_13_17 (n : ℕ) : Machine17.g17 n ≤ 11 + 17 :=
  le_trans (g17_le n) (by omega)

/-! ## Rung 3: 17 -> 19 -/

/-- Gear 19 kills slot `k`: teeth at `u' = 3` and `19 - u' = 16`. -/
def Killed19 (k : ℕ) : Prop := k % 19 = 3 ∨ k % 19 = 16

theorem killed19_iff {k : ℕ} (hk : 1 ≤ k) :
    Killed19 k ↔ (19 ∣ Census.lo k ∨ 19 ∣ Census.hi k) := by
  simp only [Killed19, Census.lo, Census.hi]
  omega

theorem not_killed19 {k : ℕ} (hk : 1 ≤ k) (h : Machine19.Exposed19 k) :
    ¬ Killed19 k :=
  fun hK => ((killed19_iff hk).mp hK).elim
    (fun hd => h.2.2.2.2.2.2.2.2.2.2.1 hd) (fun hd => h.2.2.2.2.2.2.2.2.2.2.2 hd)

theorem exposed19_of {k : ℕ} (hk : 1 ≤ k) (h : Machine17.Exposed17 k)
    (hnk : ¬ Killed19 k) : Machine19.Exposed19 k :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1,
    h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1,
    h.2.2.2.2.2.2.2.2.2,
    fun hd => hnk ((killed19_iff hk).mpr (Or.inl hd)),
    fun hd => hnk ((killed19_iff hk).mpr (Or.inr hd))⟩

theorem exposed17_of_19 {k : ℕ} (h : Machine19.Exposed19 k) :
    Machine17.Exposed17 k :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.1,
    h.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.1, h.2.2.2.2.2.2.2.2.1,
    h.2.2.2.2.2.2.2.2.2.1⟩

/-- **R39 at the 17->19 step**: every gap of machine 19 is at most
`max (F_2(17), max_j Q_j(17; 6)) = max (25, 35) = 35`, derived from machine
17's period scan ALONE. (Machine 19's own scan gives the sharp 25; the point
of the rung is that the merge law reaches the next machine without seeing
it.) -/
theorem g19_le_of_17 (n : ℕ) : Machine19.g19 n ≤ 35 :=
  MergeLaw.newgap_le_step
    (ExO := Machine17.Exposed17) (ExN := Machine19.Exposed19) (Kap := Killed19)
    (posO := Machine17.opSeq) (posN := Machine19.opSeq) (g := Machine17.g17)
    (q := 19) (u := 3) (B := 35) (F2 := 25) (Q := fun _ => 35)
    (fun _ => rfl) Machine17.opSeq_lt_succ Machine17.opSeq_pos
    Machine17.opSeq_exposed (fun _ hx hE => Machine17.opSeq_surj hx hE)
    Machine19.opSeq_pos Machine19.opSeq_lt_succ Machine19.opSeq_exposed
    (fun m x => Machine19.opSeq_gap_empty m x)
    (fun _ hx hE => not_killed19 hx hE)
    (fun _ hx hE hnk => exposed19_of hx hE hnk)
    (fun _ h => exposed17_of_19 h)
    (fun _ hk => by rcases hk with h | h <;> omega)
    (by omega) (by omega)
    Machine17.spectrum_two (by omega)
    Machine17.qual_bound_all (fun _ => le_refl 35) n

/-- **(D) at `alpha = 3` at the 17->19 step, hypothesis-free**: every gap of
machine 19 is at most `F(17) + q' = 18 + 19 = 37`. -/
theorem D_at_17_19 (n : ℕ) : Machine19.g19 n ≤ 18 + 19 :=
  le_trans (g19_le_of_17 n) (by omega)

/-! ## The ladder -/

/-- **THE LADDER: (D) at `alpha = 3` at four consecutive machine steps, with
no hypotheses.** Each conjunct is a statement about the corresponding
machine's own gap sequence; each is produced by the same mechanical recipe
(kernel-scan the old machine's `F_2` and qualifying spectrum, then
instantiate the merge law). -/
theorem D_ladder :
    (∀ n, Machine13.g13 n ≤ 7 + 13) ∧ (∀ n, Machine17.g17 n ≤ 11 + 17) ∧
      (∀ n, Machine19.g19 n ≤ 18 + 19) ∧ (∀ n, Machine23.g23 n ≤ 25 + 23) :=
  ⟨D_at_11_13, D_at_13_17, D_at_17_19, Machine23.D_at_19_23⟩

/-! ## Above the scannable range: hypothesis-explicit instantiations

The next two steps cannot be made hypothesis-free by this route: the merge
law consumes an `F_2` and a qualifying spectrum, and produces neither, so
each rung needs its own period scan of the OLD machine, and machine 23's
period is 37,182,145 CRT tuples (23x machine 19's - see formalist.md round 22
for the measured cost). What IS kernel-checkable now is the criterion
arithmetic and the merge-law step itself, with the census values named as
hypotheses. -/

/-- **R39 instantiated at the 23->29 step.** Gear 29 has teeth at `u' = 5`
and `29 - u' = 24`, so the qualifying floor is `2u' = 10`. With Mechanic's
corrected machine-23 ladder - `F_2(23) = 39` and
`Q_j(23; 10) = 43, 50, 55, 60, 0` for `j = 3..7`, so `max_j Q_j = 60` (the
pre-2026-08-24 table's `50/50/49/0/0` row was corrupt; both values here were
re-derived independently over the full 37,182,145-slot period) - every gap of
machine 29 is at most `F(23) + 29 = 34 + 29 = 63`, with margin 3. -/
theorem D_at_23_29 {ExO ExN Kap : ℕ → Prop} {posO posN g : ℕ → ℕ} {Q : ℕ → ℕ}
    (hg : ∀ m, g m = posO (m + 1) - posO m)
    (hOmono : ∀ m, posO m < posO (m + 1)) (hOpos : ∀ m, 1 ≤ posO m)
    (hOex : ∀ m, ExO (posO m))
    (hOsurj : ∀ x, 1 ≤ x → ExO x → ∃ m, posO m = x)
    (hNpos : ∀ m, 1 ≤ posN m) (hNmono : ∀ m, posN m < posN (m + 1))
    (hNex : ∀ m, ExN (posN m))
    (hNempty : ∀ m x, posN m < x → x < posN (m + 1) → ¬ ExN x)
    (hnk : ∀ x, 1 ≤ x → ExN x → ¬ Kap x)
    (hkn : ∀ x, 1 ≤ x → ExO x → ¬ Kap x → ExN x)
    (hsub : ∀ x, ExN x → ExO x)
    (hteeth : ∀ x, Kap x → x % 29 = 5 ∨ x % 29 = 24)
    (hF2 : Spectrum.SpectrumBound g 2 39)
    (hQ : ∀ j, 3 ≤ j → Spectrum.QualBound g 5 j (Q j))
    (hQm : ∀ j, Q j ≤ 60)
    (n : ℕ) : posN (n + 1) - posN n ≤ 34 + 29 :=
  MergeLaw.newgap_le_step (q := 29) (u := 5) (B := 34 + 29) (F2 := 39) (Q := Q)
    hg hOmono hOpos hOex hOsurj hNpos hNmono hNex hNempty hnk hkn hsub
    (fun x hk => (hteeth x hk).imp id (fun h => h)) (by omega) (by omega)
    hF2 (by omega) hQ (fun j => le_trans (hQm j) (by omega)) n

/-- **R39 instantiated at the 37->41 step** - the first step beyond any
period scan, decided in round 21 by Mechanic's exact `F_3(37) = 97` (witness
`k = 990,209,189,833`, gaps `[37, 23, 37]`, descent closed at both ends) and
Constructor's R44 qualmax census `max_j qualmax_j(37; 41) = 91`, which equals
`F(41) = 91` EXACTLY. Gear 41 has teeth at `u' = 7` and `41 - u' = 34`
(`6 * 7 = 42 = 41 + 1`), qualifying floor `2u' = 14`. With `F_2(37) = 90`,
the criterion value is `max (90, 91) = 91` against the budget
`F(37) + 41 = 88 + 41 = 129` - margin 38 = 0.93 q'. -/
theorem D_at_37_41 {ExO ExN Kap : ℕ → Prop} {posO posN g : ℕ → ℕ} {Q : ℕ → ℕ}
    (hg : ∀ m, g m = posO (m + 1) - posO m)
    (hOmono : ∀ m, posO m < posO (m + 1)) (hOpos : ∀ m, 1 ≤ posO m)
    (hOex : ∀ m, ExO (posO m))
    (hOsurj : ∀ x, 1 ≤ x → ExO x → ∃ m, posO m = x)
    (hNpos : ∀ m, 1 ≤ posN m) (hNmono : ∀ m, posN m < posN (m + 1))
    (hNex : ∀ m, ExN (posN m))
    (hNempty : ∀ m x, posN m < x → x < posN (m + 1) → ¬ ExN x)
    (hnk : ∀ x, 1 ≤ x → ExN x → ¬ Kap x)
    (hkn : ∀ x, 1 ≤ x → ExO x → ¬ Kap x → ExN x)
    (hsub : ∀ x, ExN x → ExO x)
    (hteeth : ∀ x, Kap x → x % 41 = 7 ∨ x % 41 = 34)
    (hF2 : Spectrum.SpectrumBound g 2 90)
    (hQ : ∀ j, 3 ≤ j → Spectrum.QualBound g 7 j (Q j))
    (hQm : ∀ j, Q j ≤ 91)
    (n : ℕ) : posN (n + 1) - posN n ≤ 88 + 41 :=
  MergeLaw.newgap_le_step (q := 41) (u := 7) (B := 88 + 41) (F2 := 90) (Q := Q)
    hg hOmono hOpos hOex hOsurj hNpos hNmono hNex hNempty hnk hkn hsub
    (fun x hk => (hteeth x hk).imp id (fun h => h)) (by omega) (by omega)
    hF2 (by omega) hQ (fun j => le_trans (hQm j) (by omega)) n

/-- The criterion arithmetic of the two steps above the scannable range,
kernel-checked on its own: `max (F_2, max_j Q_j) <= F + q'`. -/
theorem criterion_arith :
    max 39 60 ≤ 34 + 29 ∧ max 90 91 ≤ 88 + 41 := by
  constructor <;> decide

end Ladder
