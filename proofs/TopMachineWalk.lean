/-
THE WALK OF THE TOP MACHINE (Formalist, round 34).

Branch document: `research/proof/top_machine_3.md`, section 4, laws L30, L31,
L34, L35, L44, L45.  Scripts `research/topmachine/r3/*.py`.

THE OBJECT.  `TopMachine.lean` fixes the pair view: a gear `g` strikes the
pair `n` iff `g | n` or `g | n + 2` (`Strikes`), and a pair struck by nobody
is open (`IsOpen`).  This file adds the WALK: from a position `x`, how far
forward is the next open pair?  The branch's answer is a closed form in the
residues - the least natural number missing from the `2m` numbers
`(-x) mod g` and `(-x - 2) mod g` - so the walk costs `O(m)` arithmetic and
never looks at the wheel.

It also adds the SINGLE-NUMBER (twin-candidate) view: `g` strikes the number
`n` iff `g | n`, and a run of three consecutive unstruck numbers starting at
`n` is a twin candidate (`IsStart`).  The same closed form holds with three
teeth per gear instead of two, and - the point of L35 - the record is `3m`
on the nose, with no parity defect, because the piece is a SOLID triomino
while the pair machine's piece is the gapped domino `{x, x + 2}`.

Nothing here assumes primality.  Each theorem carries the exact hypothesis it
needs (`0 < g`, `2m < g`, `3m < g`, `3m + 3 <= g`, oddness, coprimality).
-/

import TopMachineCrt

namespace TopMachine

/-! ## 0. The forward offset of a gear

`off g y` is the least `j >= 0` with `g | y + j`, i.e. `(-y) mod g`.  Every
residue in this file is one of these: gear `g` strikes the pair `x + j` iff
`j = off g x` or `j = off g (x + 2)` (for `j < g`), and strikes the number
`x + j` iff `j = off g x`. -/

/-- The forward offset from `y` to the next multiple of `g`: `(-y) mod g`. -/
def off (g : ℕ) (y : ℤ) : ℕ := ((-y) % (g : ℤ)).toNat

theorem off_lt {g : ℕ} (hg : 0 < g) (y : ℤ) : off g y < g := by
  have hgz : (0 : ℤ) < (g : ℤ) := by exact_mod_cast hg
  have h1 : (-y) % (g : ℤ) < (g : ℤ) := Int.emod_lt_of_pos _ hgz
  have h2 : (0 : ℤ) ≤ (-y) % (g : ℤ) := Int.emod_nonneg _ (by omega)
  unfold off
  omega

theorem off_cast {g : ℕ} (hg : 0 < g) (y : ℤ) :
    ((off g y : ℕ) : ℤ) = (-y) % (g : ℤ) := by
  have hgz : (0 : ℤ) < (g : ℤ) := by exact_mod_cast hg
  have h2 : (0 : ℤ) ≤ (-y) % (g : ℤ) := Int.emod_nonneg _ (by omega)
  unfold off
  omega

/-- The defining property: `g` divides `y + off g y`. -/
theorem dvd_add_off {g : ℕ} (hg : 0 < g) (y : ℤ) :
    (g : ℤ) ∣ y + (off g y : ℕ) := by
  rw [off_cast hg]
  have h : (g : ℤ) ∣ (-y) - (-y) % (g : ℤ) :=
    ⟨(-y) / (g : ℤ), by rw [Int.emod_def]; ring⟩
  have he : y + (-y) % (g : ℤ) = -((-y) - (-y) % (g : ℤ)) := by ring
  rw [he]
  exact dvd_neg.mpr h

/-- The minimality: any `j < g` with `g | y + j` IS the offset. -/
theorem off_eq_of_dvd {g j : ℕ} (hg : 0 < g) (hj : j < g) {y : ℤ}
    (h : (g : ℤ) ∣ y + (j : ℕ)) : off g y = j := by
  have hgz : (0 : ℤ) < (g : ℤ) := by exact_mod_cast hg
  have hjz : (j : ℤ) < (g : ℤ) := by exact_mod_cast hj
  have hd : ((-y) - (j : ℤ)) % (g : ℤ) = 0 := by
    refine Int.emod_eq_zero_of_dvd ?_
    have he : (-y) - (j : ℤ) = -(y + (j : ℤ)) := by ring
    rw [he]
    exact dvd_neg.mpr h
  have he : (-y) % (g : ℤ) = (j : ℤ) % (g : ℤ) :=
    Int.emod_eq_emod_iff_emod_sub_eq_zero.mpr hd
  rw [Int.emod_eq_of_lt (by exact_mod_cast Nat.zero_le j) hjz] at he
  unfold off
  rw [he]
  simp

/-- Two offsets coincide exactly when the gear divides the difference. -/
theorem off_eq_iff {g : ℕ} (hg : 0 < g) (y z : ℤ) :
    off g y = off g z ↔ (g : ℤ) ∣ y - z := by
  constructor
  · intro h
    have h1 := dvd_add_off hg y
    have h2 := dvd_add_off hg z
    rw [h] at h1
    have := dvd_sub h1 h2
    rwa [show y + (off g z : ℕ) - (z + (off g z : ℕ)) = y - z by ring] at this
  · intro h
    refine off_eq_of_dvd hg (off_lt hg z) ?_
    have h2 := dvd_add_off hg z
    have := dvd_add h h2
    rwa [show y - z + (z + (off g z : ℕ)) = y + (off g z : ℕ) by ring] at this

/-- `off g 0 = 0`. -/
theorem off_zero {g : ℕ} (hg : 0 < g) : off g 0 = 0 :=
  off_eq_of_dvd hg hg (by simp)

/-- `off g 2 = g - 2`: the second tooth of the pair machine. -/
theorem off_two {g : ℕ} (hg : 3 ≤ g) : off g 2 = g - 2 := by
  refine off_eq_of_dvd (by omega) (by omega) ?_
  have : (2 : ℤ) + ((g - 2 : ℕ) : ℤ) = (g : ℤ) := by
    have : ((g - 2 : ℕ) : ℤ) = (g : ℤ) - 2 := by
      have : (2 : ℕ) ≤ g := by omega
      push_cast [Nat.cast_sub this]
      ring
    rw [this]; ring
  rw [this]

/-! ## 1. L30/L31 - THE NEXT OPEN PAIR, in closed form

`Res G x` is the branch's list `{ (-x) mod g , (-x-2) mod g : g in G }`, and
`mexS G x` is its mex.  L30: if every gear exceeds `2m` then the next open
pair after `x` is exactly `x + mexS G x`.  L31: with odd gears above `2m + 1`
the walk is at most `2m - (m mod 2)`. -/

/-- Every finite set of naturals misses a natural (the mex exists). -/
theorem exists_not_mem_nat (s : Finset ℕ) : ∃ n : ℕ, n ∉ s := by
  refine ⟨s.sup id + 1, fun h => ?_⟩
  have hle := Finset.le_sup (f := (id : ℕ → ℕ)) h
  simp only [id] at hle
  omega

/-- **The residue list of L30**: the `2m` numbers `(-x) mod g`, `(-x-2) mod g`. -/
def Res (G : Finset ℕ) (x : ℤ) : Finset ℕ :=
  G.biUnion (fun g => ({off g x, off g (x + 2)} : Finset ℕ))

/-- **The walk, in closed form**: the mex of the residue list. -/
def mexS (G : Finset ℕ) (x : ℤ) : ℕ := Nat.find (exists_not_mem_nat (Res G x))

theorem mexS_not_mem (G : Finset ℕ) (x : ℤ) : mexS G x ∉ Res G x :=
  Nat.find_spec (exists_not_mem_nat (Res G x))

theorem mem_of_lt_mexS {G : Finset ℕ} {x : ℤ} {j : ℕ} (h : j < mexS G x) :
    j ∈ Res G x := by
  have := Nat.find_min (exists_not_mem_nat (Res G x)) h
  simpa using this

theorem res_card_le (G : Finset ℕ) (x : ℤ) : (Res G x).card ≤ 2 * G.card := by
  classical
  refine le_trans (Finset.card_biUnion_le) ?_
  have hstep : ∀ g ∈ G, ({off g x, off g (x + 2)} : Finset ℕ).card ≤ 2 := by
    intro g _
    have h := Finset.card_insert_le (off g x) ({off g (x + 2)} : Finset ℕ)
    simpa using h
  calc ∑ g ∈ G, ({off g x, off g (x + 2)} : Finset ℕ).card
      ≤ ∑ _g ∈ G, 2 := Finset.sum_le_sum hstep
    _ = 2 * G.card := by rw [Finset.sum_const, smul_eq_mul, mul_comm]

/-- **L30, the location bound.**  The mex of `2m` numbers is at most `2m`. -/
theorem mexS_le (G : Finset ℕ) (x : ℤ) : mexS G x ≤ 2 * G.card := by
  by_contra hcon
  have hsub : Finset.range (2 * G.card + 1) ⊆ Res G x := by
    intro j hj
    exact mem_of_lt_mexS (by have := Finset.mem_range.mp hj; omega)
  have h1 := Finset.card_le_card hsub
  rw [Finset.card_range] at h1
  have h2 := res_card_le G x
  omega

/-- Below the mex every position is struck: some gear's residue names it. -/
theorem not_open_of_lt_mexS {G : Finset ℕ} (hpos : ∀ g ∈ G, 0 < g) (x : ℤ) {j : ℕ}
    (hj : j < mexS G x) : ¬ IsOpen G (x + (j : ℤ)) := by
  have hmem := mem_of_lt_mexS hj
  rw [Res, Finset.mem_biUnion] at hmem
  obtain ⟨g, hgG, hgj⟩ := hmem
  simp only [Finset.mem_insert, Finset.mem_singleton] at hgj
  have hg : 0 < g := hpos g hgG
  intro hopen
  refine hopen g hgG ?_
  rcases hgj with e | e
  · exact Or.inl (by rw [e]; exact dvd_add_off hg x)
  · refine Or.inr ?_
    rw [show x + (j : ℤ) + 2 = (x + 2) + (j : ℤ) by ring, e]
    exact dvd_add_off hg (x + 2)

/-- At the mex the position is open: the mex is smaller than every gear, so a
gear's only strikes in `[x, x + g)` are the two the residue list records. -/
theorem open_mexS {G : Finset ℕ} (hbig : ∀ g ∈ G, 2 * G.card < g) (x : ℤ) :
    IsOpen G (x + (mexS G x : ℤ)) := by
  intro g hgG hs
  have hgb := hbig g hgG
  have hg : 0 < g := by omega
  have hjg : mexS G x < g := lt_of_le_of_lt (mexS_le G x) hgb
  refine mexS_not_mem G x ?_
  rw [Res, Finset.mem_biUnion]
  refine ⟨g, hgG, ?_⟩
  rcases hs with h | h
  · exact Finset.mem_insert.mpr (Or.inl (off_eq_of_dvd hg hjg h).symm)
  · refine Finset.mem_insert.mpr (Or.inr (Finset.mem_singleton.mpr ?_))
    refine (off_eq_of_dvd hg hjg ?_).symm
    rw [show (x + 2) + ((mexS G x : ℕ) : ℤ) = x + (mexS G x : ℤ) + 2 by ring]
    exact h

/-- **L30, THE MEX FORM.**  If every gear exceeds `2m` (`m = |G|`), the next
open pair at or after `x` is exactly `x + mexS G x`: `mexS G x` is the LEAST
`j` with `x + j` open, and every smaller `j` is struck.

The hypothesis `2m < g` is needed only for the openness half - the mex is at
most `2m`, so it is below every gear, and a residue `r < g` of gear `g`
strikes `x + r` iff `r` is one of the gear's two listed residues.  The
struckness half below the mex needs nothing. -/
theorem mex_form {G : Finset ℕ} (hbig : ∀ g ∈ G, 2 * G.card < g) (x : ℤ) :
    IsLeast {j : ℕ | IsOpen G (x + (j : ℤ))} (mexS G x) := by
  refine ⟨open_mexS hbig x, ?_⟩
  intro j hj
  by_contra hlt
  exact not_open_of_lt_mexS (fun g hg => by have := hbig g hg; omega) x
    (by omega : j < mexS G x) hj

/-- **L31, the location bound in the sharp form.**  With odd gears above
`2m + 1` the walk never exceeds `2m - (m mod 2)`.  Proof: every position
below the mex is struck, so the mex is a run of consecutive struck pairs, and
`parity_upper` (L17) bounds such a run. -/
theorem mexS_le_parity {G : Finset ℕ} (hodd : ∀ g ∈ G, g % 2 = 1)
    (hbig : ∀ g ∈ G, 2 * G.card + 1 < g) (x : ℤ) :
    mexS G x ≤ 2 * G.card - G.card % 2 := by
  have h := parity_upper (G := G) (n := x) (L := mexS G x) hodd hbig
    (fun i hi => not_open_of_lt_mexS (fun g hg => by have := hbig g hg; omega) x hi)
  omega

/-! ## 2. L34/L35 - THE NEXT TWIN CANDIDATE

The single-number view.  A gear strikes the NUMBER `n` iff `g | n`; a TWIN
CANDIDATE (run-of-three start) at `n` is `n`, `n + 1`, `n + 2` all unstruck.
Three teeth per gear, so the residue list has `3m` entries; the record is
`3m` exactly, with no parity defect. -/

/-- Gear `g` strikes the NUMBER `n` iff it divides it. -/
def StrikesN (g : ℕ) (n : ℤ) : Prop := (g : ℤ) ∣ n

/-- A number struck by no gear. -/
def OpenNum (G : Finset ℕ) (n : ℤ) : Prop := ∀ g ∈ G, ¬ StrikesN g n

/-- `n` starts a RUN OF THREE - a twin candidate - iff `n`, `n + 1`, `n + 2`
are all unstruck. -/
def IsStart (G : Finset ℕ) (n : ℤ) : Prop :=
  OpenNum G n ∧ OpenNum G (n + 1) ∧ OpenNum G (n + 2)

/-- **The three-residue list of L34**: `(-x) mod g`, `(-x-1) mod g`,
`(-x-2) mod g` over the gears. -/
def Res3 (G : Finset ℕ) (x : ℤ) : Finset ℕ :=
  G.biUnion (fun g => ({off g x, off g (x + 1), off g (x + 2)} : Finset ℕ))

/-- **The twin-candidate walk**, in closed form. -/
def mexT (G : Finset ℕ) (x : ℤ) : ℕ := Nat.find (exists_not_mem_nat (Res3 G x))

theorem mexT_not_mem (G : Finset ℕ) (x : ℤ) : mexT G x ∉ Res3 G x :=
  Nat.find_spec (exists_not_mem_nat (Res3 G x))

theorem mem_of_lt_mexT {G : Finset ℕ} {x : ℤ} {j : ℕ} (h : j < mexT G x) :
    j ∈ Res3 G x := by
  have := Nat.find_min (exists_not_mem_nat (Res3 G x)) h
  simpa using this

theorem res3_card_le (G : Finset ℕ) (x : ℤ) : (Res3 G x).card ≤ 3 * G.card := by
  classical
  refine le_trans (Finset.card_biUnion_le) ?_
  have hstep : ∀ g ∈ G,
      ({off g x, off g (x + 1), off g (x + 2)} : Finset ℕ).card ≤ 3 := by
    intro g _
    have h1 := Finset.card_insert_le (off g x)
      ({off g (x + 1), off g (x + 2)} : Finset ℕ)
    have h2 := Finset.card_insert_le (off g (x + 1)) ({off g (x + 2)} : Finset ℕ)
    simp only [Finset.card_singleton] at h2
    omega
  calc ∑ g ∈ G, ({off g x, off g (x + 1), off g (x + 2)} : Finset ℕ).card
      ≤ ∑ _g ∈ G, 3 := Finset.sum_le_sum hstep
    _ = 3 * G.card := by rw [Finset.sum_const, smul_eq_mul, mul_comm]

/-- **L34, the location bound.**  The mex of `3m` numbers is at most `3m`. -/
theorem mexT_le (G : Finset ℕ) (x : ℤ) : mexT G x ≤ 3 * G.card := by
  by_contra hcon
  have hsub : Finset.range (3 * G.card + 1) ⊆ Res3 G x := by
    intro j hj
    exact mem_of_lt_mexT (by have := Finset.mem_range.mp hj; omega)
  have h1 := Finset.card_le_card hsub
  rw [Finset.card_range] at h1
  have h2 := res3_card_le G x
  omega

/-- A listed residue kills the run of three that would start there. -/
theorem not_start_of_mem_res3 {G : Finset ℕ} (hg : ∀ g ∈ G, 0 < g) {x : ℤ} {j : ℕ}
    (h : j ∈ Res3 G x) : ¬ IsStart G (x + (j : ℤ)) := by
  rw [Res3, Finset.mem_biUnion] at h
  obtain ⟨g, hgG, hgj⟩ := h
  simp only [Finset.mem_insert, Finset.mem_singleton] at hgj
  have hg0 := hg g hgG
  rintro ⟨h0, h1, h2⟩
  rcases hgj with e | e | e
  · exact h0 g hgG (by rw [e]; exact dvd_add_off hg0 x)
  · refine h1 g hgG ?_
    show (g : ℤ) ∣ x + (j : ℤ) + 1
    rw [show x + (j : ℤ) + 1 = (x + 1) + (j : ℤ) by ring, e]
    exact dvd_add_off hg0 (x + 1)
  · refine h2 g hgG ?_
    show (g : ℤ) ∣ x + (j : ℤ) + 2
    rw [show x + (j : ℤ) + 2 = (x + 2) + (j : ℤ) by ring, e]
    exact dvd_add_off hg0 (x + 2)

/-- An unlisted residue below every gear starts a run of three. -/
theorem start_of_not_mem_res3 {G : Finset ℕ} {x : ℤ} {j : ℕ}
    (hbig : ∀ g ∈ G, j < g) (h : j ∉ Res3 G x) : IsStart G (x + (j : ℤ)) := by
  have key : ∀ g ∈ G, j ≠ off g x ∧ j ≠ off g (x + 1) ∧ j ≠ off g (x + 2) := by
    intro g hgG
    refine ⟨?_, ?_, ?_⟩ <;>
      (intro e; exact h (by rw [Res3, Finset.mem_biUnion]; exact ⟨g, hgG, by simp [e]⟩))
  refine ⟨?_, ?_, ?_⟩ <;> intro g hgG hs
  · have hjg := hbig g hgG
    exact (key g hgG).1 (off_eq_of_dvd (by omega) hjg hs).symm
  · have hjg := hbig g hgG
    have hs' : (g : ℤ) ∣ (x + 1) + (j : ℤ) := by
      rw [show (x + 1) + (j : ℤ) = x + (j : ℤ) + 1 by ring]; exact hs
    exact (key g hgG).2.1 (off_eq_of_dvd (by omega) hjg hs').symm
  · have hjg := hbig g hgG
    have hs' : (g : ℤ) ∣ (x + 2) + (j : ℤ) := by
      rw [show (x + 2) + (j : ℤ) = x + (j : ℤ) + 2 by ring]; exact hs
    exact (key g hgG).2.2 (off_eq_of_dvd (by omega) hjg hs').symm

/-- **L34, THE TWIN-CANDIDATE MEX FORM.**  If every gear exceeds `3m`, the
next twin candidate at or after `x` starts exactly at `x + mexT G x`. -/
theorem triple_mex_form {G : Finset ℕ} (hbig : ∀ g ∈ G, 3 * G.card < g) (x : ℤ) :
    IsLeast {j : ℕ | IsStart G (x + (j : ℤ))} (mexT G x) := by
  have hg : ∀ g ∈ G, 0 < g := fun g hgG => by have := hbig g hgG; omega
  refine ⟨start_of_not_mem_res3 ?_ (mexT_not_mem G x), ?_⟩
  · intro g hgG
    exact lt_of_le_of_lt (mexT_le G x) (hbig g hgG)
  · intro j hj
    by_contra hlt
    exact not_start_of_mem_res3 hg (mem_of_lt_mexT (by omega : j < mexT G x)) hj

/-- **L35, the upper bound.**  No `3m + 1` consecutive positions can all fail
to start a run of three: the mex form finds a start within `3m`. -/
theorem triple_upper {G : Finset ℕ} (hbig : ∀ g ∈ G, 3 * G.card < g) {n : ℤ} {L : ℕ}
    (hL : ∀ i : ℕ, i < L → ¬ IsStart G (n + (i : ℤ))) : L ≤ 3 * G.card := by
  by_contra hcon
  exact hL (mexT G n) (by have := mexT_le G n; omega) ((triple_mex_form hbig n).1)

/-- **L35, attainment.**  Pairwise coprime gears admit `3m` consecutive
positions none of which starts a run of three: give gear `j` the SOLID
triomino `[3j, 3j + 3)` by asking, via CRT, that it divide `n + 3j + 2`.
The three cells of that block are killed by that one gear.

No size hypothesis: a gear always kills the three starts whose block contains
the multiple its residue names.  (Contrast `parity_attained`, where the piece
is the gapped domino `{x, x + 2}` and the tiling loses one cell when `m` is
odd - the parity defect is a property of the SEPARATION, not of the tooth
count.) -/
theorem triple_attained {G : Finset ℕ}
    (hcop : ∀ g ∈ G, ∀ h ∈ G, g ≠ h → Nat.Coprime g h) :
    ∃ n : ℤ, ∀ i : ℕ, i < 3 * G.card → ¬ IsStart G (n + (i : ℤ)) := by
  classical
  have hpick : ∀ h : ℕ, ∃ z : ℤ, ∀ hh : h ∈ G,
      z = -(((3 * (G.equivFin ⟨h, hh⟩ : Fin G.card).val + 2 : ℕ) : ℤ)) := by
    intro h
    by_cases hh : h ∈ G
    · exact ⟨-(((3 * (G.equivFin ⟨h, hh⟩ : Fin G.card).val + 2 : ℕ) : ℤ)), fun _ => rfl⟩
    · exact ⟨0, fun hh' => absurd hh' hh⟩
  choose r hr using hpick
  obtain ⟨n, hn⟩ := exists_crt G hcop r
  refine ⟨n, fun i hi hstart => ?_⟩
  have hj : i / 3 < G.card := by omega
  set j := i / 3 with hjdef
  set x := G.equivFin.symm ⟨j, hj⟩ with hxdef
  have hx : G.equivFin x = ⟨j, hj⟩ := Equiv.apply_symm_apply _ _
  have hgj : (x : ℕ) ∈ G := x.2
  have hidx : ((G.equivFin ⟨(x : ℕ), hgj⟩ : Fin G.card)).val = j := by
    have h1 : (⟨(x : ℕ), hgj⟩ : {y // y ∈ G}) = x := rfl
    rw [h1, hx]
  have hdvd := hn (x : ℕ) hgj
  rw [hr (x : ℕ) hgj, hidx] at hdvd
  have hA : ((x : ℕ) : ℤ) ∣ n + ((3 * j + 2 : ℕ) : ℤ) := by
    have hstep : n - -(((3 * j + 2 : ℕ) : ℤ)) = n + ((3 * j + 2 : ℕ) : ℤ) := by ring
    rwa [hstep] at hdvd
  obtain ⟨h0, h1, h2⟩ := hstart
  have h3 : i % 3 = 0 ∨ i % 3 = 1 ∨ i % 3 = 2 := by omega
  rcases h3 with h | h | h
  · have hij : i = 3 * j := by omega
    refine h2 (x : ℕ) hgj ?_
    show ((x : ℕ) : ℤ) ∣ n + (i : ℤ) + 2
    rw [hij]; push_cast; push_cast at hA
    rw [show n + 3 * (j : ℤ) + 2 = n + (3 * (j : ℤ) + 2) by ring]
    exact hA
  · have hij : i = 3 * j + 1 := by omega
    refine h1 (x : ℕ) hgj ?_
    show ((x : ℕ) : ℤ) ∣ n + (i : ℤ) + 1
    rw [hij]; push_cast; push_cast at hA
    rw [show n + (3 * (j : ℤ) + 1) + 1 = n + (3 * (j : ℤ) + 2) by ring]
    exact hA
  · have hij : i = 3 * j + 2 := by omega
    refine h0 (x : ℕ) hgj ?_
    show ((x : ℕ) : ℤ) ∣ n + (i : ℤ)
    rw [hij]; push_cast; push_cast at hA
    rw [show n + (3 * (j : ℤ) + 2) = n + (3 * (j : ℤ) + 2) by ring]
    exact hA

/-- **L35, THE TRIPLE RECORD, as an equality.**  For pairwise coprime gears
of size at least `3m + 3`, the longest run of consecutive positions none of
which starts a run of three is EXACTLY `3m` - no parity correction, unlike
the pair machine's `2m - (m mod 2)` (`parity_law`). -/
theorem triple_law {G : Finset ℕ} (hbig : ∀ g ∈ G, 3 * G.card + 3 ≤ g)
    (hcop : ∀ g ∈ G, ∀ h ∈ G, g ≠ h → Nat.Coprime g h) :
    IsGreatest {L : ℕ | ∃ n : ℤ, ∀ i : ℕ, i < L → ¬ IsStart G (n + (i : ℤ))}
      (3 * G.card) := by
  refine ⟨triple_attained hcop, ?_⟩
  rintro L ⟨n, hL⟩
  exact triple_upper (fun g hg => by have := hbig g hg; omega) hL

/-! ## 3. L45 - the holes of the consecutive census

In the pair view the only forbidden gap below the record is `4`
(`open_of_open_add_four` / `no_gap_four`, round 32).  In the triple view the
forbidden gaps are `2` and `3`: two run-of-three starts at distance 2 or 3
are never CONSECUTIVE, because `x + 1` is a start as well. -/

/-- Two starts at distance 2 force a start in between. -/
theorem start_of_start_add_two {G : Finset ℕ} {x : ℤ}
    (h0 : IsStart G x) (h2 : IsStart G (x + 2)) : IsStart G (x + 1) := by
  refine ⟨h0.2.1, ?_, ?_⟩
  · rw [show x + 1 + 1 = x + 2 by ring]; exact h0.2.2
  · rw [show x + 1 + 2 = x + 2 + 1 by ring]; exact h2.2.1

/-- Two starts at distance 3 force a start in between. -/
theorem start_of_start_add_three {G : Finset ℕ} {x : ℤ}
    (h0 : IsStart G x) (h3 : IsStart G (x + 3)) : IsStart G (x + 1) := by
  refine ⟨h0.2.1, ?_, ?_⟩
  · rw [show x + 1 + 1 = x + 2 by ring]; exact h0.2.2
  · rw [show x + 1 + 2 = x + 3 by ring]; exact h3.1

/-- **L45, the triple-view holes.**  If `x` and `x + d` are both run-of-three
starts with `d = 2` or `d = 3`, then a start lies strictly between them: the
gap census of the twin candidates has holes exactly at 2 and 3.  (No
hypothesis at all - not even `0 < g`.) -/
theorem no_start_gap_two_three {G : Finset ℕ} {x : ℤ} {d : ℤ}
    (hd : d = 2 ∨ d = 3) (h0 : IsStart G x) (hdd : IsStart G (x + d)) :
    ∃ y : ℤ, x < y ∧ y < x + d ∧ IsStart G y := by
  refine ⟨x + 1, by omega, by omega, ?_⟩
  rcases hd with rfl | rfl
  · exact start_of_start_add_two h0 hdd
  · exact start_of_start_add_three h0 hdd

/-- The same, in the shape of `no_gap_four`: a gap of exactly 2 or 3 between
consecutive twin candidates never occurs. -/
theorem no_start_gap {G : Finset ℕ} {x : ℤ} {d : ℤ} (hd : d = 2 ∨ d = 3)
    (h0 : IsStart G x) (hdd : IsStart G (x + d))
    (hmid : ∀ z : ℤ, x < z → z < x + d → ¬ IsStart G z) : False := by
  obtain ⟨y, hy1, hy2, hy3⟩ := no_start_gap_two_three hd h0 hdd
  exact hmid y hy1 hy2 hy3

/-- The pair-view counterpart, restated from `open_of_open_add_four` (L4):
two open pairs at distance 4 are never consecutive. -/
theorem no_pair_gap_four {G : Finset ℕ} {n : ℤ}
    (h0 : IsOpen G n) (h4 : IsOpen G (n + 4)) :
    ∃ y : ℤ, n < y ∧ y < n + 4 ∧ IsOpen G y :=
  ⟨n + 2, by omega, by omega, open_of_open_add_four h0 h4⟩

/-! ## 4. L44 - the pair correlation is a true product

`B(d) = #{n mod W : n and n + d both open}`.  "Both open" is a PER-GEAR
condition - unlike "all struck" (L37), which is not - so it counts
multiplicatively over the gears, on the same CRT engine `card_filter_crt`
that gives `wheel_count`.  The per-gear factor is `g` minus the number of
distinct forbidden residues, and the forbidden set is the four offsets
`{0, -2} ∪ {-d, -d-2}`, whose only coincidences are `d ≡ 0` and `d ≡ ±2`. -/

/-- Residue form of "`n` and `n + d` are both open", for one gear. -/
def BothR (g d r : ℕ) : Prop := ¬ StrikesR g r ∧ ¬ StrikesR g (r + d)

/-- Residue form of "`n` and `n + d` are both open", for a gear set. -/
def BothN (G : Finset ℕ) (d n : ℕ) : Prop := ∀ g ∈ G, BothR g d n

instance decBothR (g d r : ℕ) : Decidable (BothR g d r) := by
  unfold BothR; infer_instance

instance decBothN (G : Finset ℕ) (d n : ℕ) : Decidable (BothN G d n) := by
  unfold BothN; infer_instance

theorem bothN_iff (G : Finset ℕ) (d n : ℕ) :
    BothN G d n ↔ (OpenN G n ∧ OpenN G (n + d)) := by
  unfold BothN BothR OpenN
  constructor
  · intro h; exact ⟨fun g hg => (h g hg).1, fun g hg => (h g hg).2⟩
  · rintro ⟨h1, h2⟩ g hg; exact ⟨h1 g hg, h2 g hg⟩

theorem bothR_congr {g d x y : ℕ} (h : x % g = y % g) : BothR g d x ↔ BothR g d y := by
  unfold BothR
  have h2 : (x + d) % g = (y + d) % g := by
    rw [Nat.add_mod x d g, Nat.add_mod y d g, h]
  exact and_congr (not_congr (strikesR_congr h)) (not_congr (strikesR_congr h2))

theorem bothN_congr {G : Finset ℕ} {W d x y : ℕ} (hdvd : ∀ g ∈ G, g ∣ W)
    (h : x % W = y % W) : BothN G d x ↔ BothN G d y := by
  unfold BothN
  refine forall_congr' fun g => forall_congr' fun hg => bothR_congr ?_
  rw [← Nat.mod_mod_of_dvd x (hdvd g hg), ← Nat.mod_mod_of_dvd y (hdvd g hg), h]

theorem bothN_insert {a : ℕ} {s : Finset ℕ} {d n : ℕ} :
    BothN (insert a s) d n ↔ (BothR a d n ∧ BothN s d n) := by
  constructor
  · intro h
    exact ⟨h a (Finset.mem_insert_self a s), fun g hg => h g (Finset.mem_insert_of_mem hg)⟩
  · rintro ⟨h1, h2⟩ g hg
    rcases Finset.mem_insert.mp hg with rfl | hg'
    · exact h1
    · exact h2 g hg'

/-- **L44, the product form.**  The correlation count factors over the gears:
"both open" is a per-gear condition, so `card_filter_crt` applies exactly as
in `wheel_count`. -/
theorem corr_prod : ∀ (G : Finset ℕ), (∀ g ∈ G, 0 < g) →
    (∀ g ∈ G, ∀ h ∈ G, g ≠ h → Nat.Coprime g h) → ∀ d : ℕ,
    ((Finset.range (∏ g ∈ G, g)).filter (fun n => BothN G d n)).card
      = ∏ g ∈ G, ((Finset.range g).filter (fun r => BothR g d r)).card := by
  classical
  intro G
  induction G using Finset.induction_on with
  | empty => intro _ _ d; simp [BothN]
  | insert a s hasnot ih =>
      intro hG0 hcop d
      have hmemA : a ∈ insert a s := Finset.mem_insert_self a s
      have ha0 : 0 < a := hG0 a hmemA
      have hs0 : ∀ g ∈ s, 0 < g := fun g hg => hG0 g (Finset.mem_insert_of_mem hg)
      have hscop : ∀ g ∈ s, ∀ h ∈ s, g ≠ h → Nat.Coprime g h := fun g hg h hh hne =>
        hcop g (Finset.mem_insert_of_mem hg) h (Finset.mem_insert_of_mem hh) hne
      have hP0 : 0 < ∏ g ∈ s, g :=
        Finset.prod_pos fun i hi => hs0 i hi
      have hcopr : Nat.Coprime a (∏ g ∈ s, g) :=
        Nat.Coprime.prod_right fun i hi =>
          hcop a hmemA i (Finset.mem_insert_of_mem hi) (by rintro rfl; exact hasnot hi)
      have hdvd : ∀ g ∈ s, g ∣ (∏ g ∈ s, g) := fun g hg => Finset.dvd_prod_of_mem _ hg
      rw [Finset.prod_insert hasnot, Finset.prod_insert hasnot]
      have hfil : (Finset.range (a * ∏ g ∈ s, g)).filter (fun n => BothN (insert a s) d n)
          = (Finset.range (a * ∏ g ∈ s, g)).filter
              (fun n => BothR a d n ∧ BothN s d n) :=
        Finset.filter_congr fun n _ => bothN_insert
      rw [hfil,
        card_filter_crt ha0 hP0 hcopr (BothR a d) (BothN s d)
          (fun x y h => bothR_congr h)
          (fun x y h => bothN_congr hdvd h),
        ih hs0 hscop d]

/-! ### The per-gear factor -/

theorem dvd_iff_off_eq {g r : ℕ} (hg : 0 < g) (hr : r < g) (y : ℤ) :
    (g : ℤ) ∣ y + (r : ℕ) ↔ off g y = r := by
  constructor
  · exact off_eq_of_dvd hg hr
  · intro h; rw [← h]; exact dvd_add_off hg y

/-- The four forbidden offsets of gear `g` at distance `d`: its own two teeth
`{0, -2}` and the two teeth of the shifted pair `{-d, -d-2}`. -/
def CorrTeeth (g d : ℕ) : Finset ℕ :=
  ({off g 0, off g 2, off g (d : ℤ), off g ((d : ℤ) + 2)} : Finset ℕ)

theorem corrTeeth_subset {g d : ℕ} (hg : 0 < g) : CorrTeeth g d ⊆ Finset.range g := by
  intro r hr
  simp only [CorrTeeth, Finset.mem_insert, Finset.mem_singleton] at hr
  rw [Finset.mem_range]
  rcases hr with rfl | rfl | rfl | rfl <;> exact off_lt hg _

theorem bothR_iff_not_teeth {g d r : ℕ} (hg : 0 < g) (hr : r < g) :
    BothR g d r ↔ r ∉ CorrTeeth g d := by
  have e0 : ((g : ℤ) ∣ (r : ℤ)) ↔ off g 0 = r := by
    have h := dvd_iff_off_eq hg hr (0 : ℤ)
    rwa [show (0 : ℤ) + (r : ℕ) = (r : ℤ) by ring] at h
  have e2 : ((g : ℤ) ∣ (r : ℤ) + 2) ↔ off g 2 = r := by
    have h := dvd_iff_off_eq hg hr (2 : ℤ)
    rwa [show (2 : ℤ) + (r : ℕ) = (r : ℤ) + 2 by ring] at h
  have ed : ((g : ℤ) ∣ ((r + d : ℕ) : ℤ)) ↔ off g (d : ℤ) = r := by
    have h := dvd_iff_off_eq hg hr ((d : ℕ) : ℤ)
    rwa [show ((d : ℕ) : ℤ) + (r : ℕ) = ((r + d : ℕ) : ℤ) by push_cast; ring] at h
  have ed2 : ((g : ℤ) ∣ ((r + d : ℕ) : ℤ) + 2) ↔ off g ((d : ℤ) + 2) = r := by
    have h := dvd_iff_off_eq hg hr (((d : ℕ) : ℤ) + 2)
    rwa [show ((d : ℕ) : ℤ) + 2 + (r : ℕ) = ((r + d : ℕ) : ℤ) + 2 by push_cast; ring] at h
  have s1 : StrikesR g r ↔ (off g 0 = r ∨ off g 2 = r) := by
    rw [← strikes_natCast]; unfold Strikes; rw [e0, e2]
  have s2 : StrikesR g (r + d) ↔ (off g (d : ℤ) = r ∨ off g ((d : ℤ) + 2) = r) := by
    rw [← strikes_natCast]; unfold Strikes; rw [ed, ed2]
  unfold BothR CorrTeeth
  rw [s1, s2]
  simp only [Finset.mem_insert, Finset.mem_singleton, not_or]
  constructor
  · rintro ⟨⟨h1, h2⟩, ⟨h3, h4⟩⟩
    exact ⟨fun e => h1 e.symm, fun e => h2 e.symm, fun e => h3 e.symm, fun e => h4 e.symm⟩
  · rintro ⟨h1, h2, h3, h4⟩
    exact ⟨⟨fun e => h1 e.symm, fun e => h2 e.symm⟩,
      ⟨fun e => h3 e.symm, fun e => h4 e.symm⟩⟩

theorem both_residues {g d : ℕ} (hg : 0 < g) :
    (Finset.range g).filter (fun r => BothR g d r)
      = Finset.range g \ CorrTeeth g d := by
  ext r
  simp only [Finset.mem_filter, Finset.mem_sdiff, Finset.mem_range]
  constructor
  · rintro ⟨hr, hb⟩; exact ⟨hr, (bothR_iff_not_teeth hg hr).mp hb⟩
  · rintro ⟨hr, hb⟩; exact ⟨hr, (bothR_iff_not_teeth hg hr).mpr hb⟩

/-- The per-gear factor is `g` minus the number of DISTINCT forbidden
offsets. -/
theorem card_both_residues {g d : ℕ} (hg : 0 < g) :
    ((Finset.range g).filter (fun r => BothR g d r)).card = g - (CorrTeeth g d).card := by
  rw [both_residues hg, Finset.card_sdiff_of_subset (corrTeeth_subset hg),
    Finset.card_range]

/-! ### The coincidences: which of the four offsets collide -/

theorem not_dvd_two {g : ℕ} (hg : 3 ≤ g) : ¬ (g : ℤ) ∣ (2 : ℤ) := by
  have hgz : (3 : ℤ) ≤ (g : ℤ) := by exact_mod_cast hg
  exact not_dvd_of_abs_lt (by omega) (by norm_num) (by omega) (by omega)

theorem not_dvd_four {g : ℕ} (hg : 5 ≤ g) : ¬ (g : ℤ) ∣ (4 : ℤ) := by
  have hgz : (5 : ℤ) ≤ (g : ℤ) := by exact_mod_cast hg
  exact not_dvd_of_abs_lt (by omega) (by norm_num) (by omega) (by omega)

/-- Two offsets differ as soon as the gear misses the difference. -/
theorem off_ne_of_not_dvd {g : ℕ} (hg : 0 < g) {y z : ℤ} (h : ¬ (g : ℤ) ∣ y - z) :
    off g y ≠ off g z := fun hc => h ((off_eq_iff hg y z).mp hc)

theorem off_zero_ne_off_two {g : ℕ} (hg : 3 ≤ g) : off g 0 ≠ off g 2 := by
  refine off_ne_of_not_dvd (by omega) (fun hc => not_dvd_two hg ?_)
  have h : (g : ℤ) ∣ -(2 : ℤ) := by
    rw [show -(2 : ℤ) = (0 : ℤ) - 2 by ring]; exact hc
  exact dvd_neg.mp h

theorem off_d_ne_off_d_two {g d : ℕ} (hg : 3 ≤ g) :
    off g (d : ℤ) ≠ off g ((d : ℤ) + 2) := by
  refine off_ne_of_not_dvd (by omega) (fun hc => not_dvd_two hg ?_)
  rw [show (d : ℤ) - ((d : ℤ) + 2) = -2 by ring] at hc
  exact dvd_neg.mp hc

/-- `0` and `-d` collide only when the gear divides `d`. -/
theorem off_zero_ne_off_d {g d : ℕ} (hg : 0 < g) (h0 : ¬ (g : ℤ) ∣ (d : ℤ)) :
    off g 0 ≠ off g (d : ℤ) := by
  refine off_ne_of_not_dvd hg (fun hc => h0 ?_)
  have h : (g : ℤ) ∣ -(d : ℤ) := by
    rw [show -(d : ℤ) = (0 : ℤ) - (d : ℤ) by ring]; exact hc
  exact dvd_neg.mp h

/-- `0` and `-d-2` collide only when the gear divides `d + 2`. -/
theorem off_zero_ne_off_d_two {g d : ℕ} (hg : 0 < g) (h2 : ¬ (g : ℤ) ∣ (d : ℤ) + 2) :
    off g 0 ≠ off g ((d : ℤ) + 2) := by
  refine off_ne_of_not_dvd hg (fun hc => h2 ?_)
  have h : (g : ℤ) ∣ -((d : ℤ) + 2) := by
    rw [show -((d : ℤ) + 2) = (0 : ℤ) - ((d : ℤ) + 2) by ring]; exact hc
  exact dvd_neg.mp h

/-- `-2` and `-d-2` collide only when the gear divides `d`. -/
theorem off_two_ne_off_d_two {g d : ℕ} (hg : 0 < g) (h0 : ¬ (g : ℤ) ∣ (d : ℤ)) :
    off g 2 ≠ off g ((d : ℤ) + 2) := by
  refine off_ne_of_not_dvd hg (fun hc => h0 ?_)
  have h : (g : ℤ) ∣ -(d : ℤ) := by
    rw [show -(d : ℤ) = (2 : ℤ) - ((d : ℤ) + 2) by ring]; exact hc
  exact dvd_neg.mp h

/-- `-2` and `-d` collide only when the gear divides `d - 2`. -/
theorem off_two_ne_off_d {g d : ℕ} (hg : 0 < g) (hm2 : ¬ (g : ℤ) ∣ (d : ℤ) - 2) :
    off g 2 ≠ off g (d : ℤ) := by
  refine off_ne_of_not_dvd hg (fun hc => hm2 ?_)
  have h : (g : ℤ) ∣ -((d : ℤ) - 2) := by
    rw [show -((d : ℤ) - 2) = (2 : ℤ) - (d : ℤ) by ring]; exact hc
  exact dvd_neg.mp h

/-- **Case `g | d`**: the two pairs of teeth coincide, so only two residues
are forbidden and the factor is `g - 2`. -/
theorem corrTeeth_card_of_dvd {g d : ℕ} (hg : 3 ≤ g) (h : (g : ℤ) ∣ (d : ℤ)) :
    (CorrTeeth g d).card = 2 := by
  have hg0 : 0 < g := by omega
  have hA : off g (d : ℤ) = off g 0 := by
    rw [off_eq_iff hg0]; simpa using h
  have hB : off g ((d : ℤ) + 2) = off g 2 := by
    rw [off_eq_iff hg0]
    rw [show (d : ℤ) + 2 - 2 = (d : ℤ) by ring]; exact h
  have hset : CorrTeeth g d = ({off g 0, off g 2} : Finset ℕ) := by
    unfold CorrTeeth
    rw [hA, hB]
    ext x
    simp only [Finset.mem_insert, Finset.mem_singleton]
    tauto
  rw [hset, Finset.card_pair (off_zero_ne_off_two hg)]

/-- **Case `g | d + 2` (and `g ∤ d`)**: exactly one coincidence, factor
`g - 3`. -/
theorem corrTeeth_card_of_dvd_add {g d : ℕ} (hg : 5 ≤ g)
    (h0 : ¬ (g : ℤ) ∣ (d : ℤ)) (h : (g : ℤ) ∣ (d : ℤ) + 2) :
    (CorrTeeth g d).card = 3 := by
  have hg0 : 0 < g := by omega
  have hB : off g ((d : ℤ) + 2) = off g 0 := by
    rw [off_eq_iff hg0]; simpa using h
  have hne1 : off g 0 ≠ off g 2 := off_zero_ne_off_two (by omega)
  have hne2 : off g 0 ≠ off g (d : ℤ) := off_zero_ne_off_d hg0 h0
  have hne3 : off g 2 ≠ off g (d : ℤ) := by
    refine off_ne_of_not_dvd hg0 (fun hc => not_dvd_four hg ?_)
    have hs := dvd_add hc h
    rwa [show (2 : ℤ) - (d : ℤ) + ((d : ℤ) + 2) = 4 by ring] at hs
  have hset : CorrTeeth g d = ({off g 0, off g 2, off g (d : ℤ)} : Finset ℕ) := by
    unfold CorrTeeth
    rw [hB]
    ext x
    simp only [Finset.mem_insert, Finset.mem_singleton]
    tauto
  rw [hset, Finset.card_insert_of_notMem (by
      simp only [Finset.mem_insert, Finset.mem_singleton, not_or]
      exact ⟨hne1, hne2⟩), Finset.card_pair hne3]

/-- **Case `g | d - 2` (and `g ∤ d`, `g ∤ d + 2`)**: exactly one coincidence,
factor `g - 3`. -/
theorem corrTeeth_card_of_dvd_sub {g d : ℕ} (hg : 5 ≤ g)
    (h0 : ¬ (g : ℤ) ∣ (d : ℤ)) (h2 : ¬ (g : ℤ) ∣ (d : ℤ) + 2)
    (h : (g : ℤ) ∣ (d : ℤ) - 2) :
    (CorrTeeth g d).card = 3 := by
  have hg0 : 0 < g := by omega
  have hA : off g (d : ℤ) = off g 2 := by rw [off_eq_iff hg0]; exact h
  have hne1 : off g 0 ≠ off g 2 := off_zero_ne_off_two (by omega)
  have hne2 : off g 0 ≠ off g ((d : ℤ) + 2) := off_zero_ne_off_d_two hg0 h2
  have hne3 : off g 2 ≠ off g ((d : ℤ) + 2) := off_two_ne_off_d_two hg0 h0
  have hset : CorrTeeth g d = ({off g 0, off g 2, off g ((d : ℤ) + 2)} : Finset ℕ) := by
    unfold CorrTeeth
    rw [hA]
    ext x
    simp only [Finset.mem_insert, Finset.mem_singleton]
    tauto
  rw [hset, Finset.card_insert_of_notMem (by
      simp only [Finset.mem_insert, Finset.mem_singleton, not_or]
      exact ⟨hne1, hne2⟩), Finset.card_pair hne3]

/-- **The generic case**: all four offsets distinct, factor `g - 4`. -/
theorem corrTeeth_card_generic {g d : ℕ} (hg : 5 ≤ g)
    (h0 : ¬ (g : ℤ) ∣ (d : ℤ)) (h2 : ¬ (g : ℤ) ∣ (d : ℤ) + 2)
    (hm2 : ¬ (g : ℤ) ∣ (d : ℤ) - 2) :
    (CorrTeeth g d).card = 4 := by
  have hg0 : 0 < g := by omega
  have hne1 : off g 0 ≠ off g 2 := off_zero_ne_off_two (by omega)
  have hne2 : off g 0 ≠ off g (d : ℤ) := off_zero_ne_off_d hg0 h0
  have hne3 : off g 0 ≠ off g ((d : ℤ) + 2) := off_zero_ne_off_d_two hg0 h2
  have hne4 : off g 2 ≠ off g (d : ℤ) := off_two_ne_off_d hg0 hm2
  have hne5 : off g 2 ≠ off g ((d : ℤ) + 2) := off_two_ne_off_d_two hg0 h0
  have hne6 : off g (d : ℤ) ≠ off g ((d : ℤ) + 2) := off_d_ne_off_d_two (by omega)
  unfold CorrTeeth
  rw [Finset.card_insert_of_notMem (by
      simp only [Finset.mem_insert, Finset.mem_singleton, not_or]
      exact ⟨hne1, hne2, hne3⟩),
    Finset.card_insert_of_notMem (by
      simp only [Finset.mem_insert, Finset.mem_singleton, not_or]
      exact ⟨hne4, hne5⟩),
    Finset.card_pair hne6]

/-- **L44's coefficient** `c_g(d)`, in residue form: `g - 2` if `g | d`,
`g - 3` if `g | d ± 2`, `g - 4` otherwise. -/
def corrCoeff (g d : ℕ) : ℕ :=
  if d % g = 0 then g - 2
  else if (d + 2) % g = 0 ∨ (d + g - 2) % g = 0 then g - 3
  else g - 4

theorem card_both_residues_eval {g d : ℕ} (hg : 5 ≤ g) :
    ((Finset.range g).filter (fun r => BothR g d r)).card = corrCoeff g d := by
  have hg0 : 0 < g := by omega
  have b0 : d % g = 0 ↔ (g : ℤ) ∣ (d : ℤ) := by
    rw [← Nat.dvd_iff_mod_eq_zero, ← Int.natCast_dvd_natCast]
  have b2 : (d + 2) % g = 0 ↔ (g : ℤ) ∣ (d : ℤ) + 2 := by
    rw [← Nat.dvd_iff_mod_eq_zero, ← Int.natCast_dvd_natCast]
    push_cast
    rfl
  have bm2 : (d + g - 2) % g = 0 ↔ (g : ℤ) ∣ (d : ℤ) - 2 := by
    rw [← Nat.dvd_iff_mod_eq_zero, ← Int.natCast_dvd_natCast]
    have hc : ((d + g - 2 : ℕ) : ℤ) = (d : ℤ) - 2 + (g : ℤ) := by
      have h2 : 2 ≤ d + g := by omega
      push_cast [Nat.cast_sub h2]
      ring
    rw [hc]
    constructor
    · intro h
      have := dvd_sub h (dvd_refl (g : ℤ))
      rwa [show (d : ℤ) - 2 + (g : ℤ) - (g : ℤ) = (d : ℤ) - 2 by ring] at this
    · intro h
      have := dvd_add h (dvd_refl (g : ℤ))
      exact this
  rw [card_both_residues hg0]
  unfold corrCoeff
  split_ifs with hd0 hd2
  · rw [corrTeeth_card_of_dvd (by omega) (b0.mp hd0)]
  · have h0 : ¬ (g : ℤ) ∣ (d : ℤ) := fun h => hd0 (b0.mpr h)
    rcases hd2 with h | h
    · rw [corrTeeth_card_of_dvd_add (by omega) h0 (b2.mp h)]
    · by_cases hcase : (d + 2) % g = 0
      · rw [corrTeeth_card_of_dvd_add (by omega) h0 (b2.mp hcase)]
      · rw [corrTeeth_card_of_dvd_sub (by omega) h0 (fun hx => hcase (b2.mpr hx))
          (bm2.mp h)]
  · have h0 : ¬ (g : ℤ) ∣ (d : ℤ) := fun h => hd0 (b0.mpr h)
    have hA : ¬ (d + 2) % g = 0 := fun hx => hd2 (Or.inl hx)
    have hB : ¬ (d + g - 2) % g = 0 := fun hx => hd2 (Or.inr hx)
    rw [corrTeeth_card_generic (by omega) h0 (fun hx => hA (b2.mpr hx))
      (fun hx => hB (bm2.mpr hx))]

/-- **L44, THE CORRELATION IS A TRUE PRODUCT.**  For gears at least 5 and
pairwise coprime, the number of residues `n` mod the wheel with `n` and
`n + d` both open is exactly `prod_g c_g(d)`.

Contrast L37: "all struck" is not a per-gear condition and its count is an
alternating sum of shifted wheel products, not a product - which is why the
blocked record cannot be read off a product. -/
theorem pair_corr {G : Finset ℕ} (h5 : ∀ g ∈ G, 5 ≤ g)
    (hcop : ∀ g ∈ G, ∀ h ∈ G, g ≠ h → Nat.Coprime g h) (d : ℕ) :
    ((Finset.range (∏ g ∈ G, g)).filter
        (fun n => OpenN G n ∧ OpenN G (n + d))).card
      = ∏ g ∈ G, corrCoeff g d := by
  classical
  have hfil : (Finset.range (∏ g ∈ G, g)).filter
      (fun n => OpenN G n ∧ OpenN G (n + d))
      = (Finset.range (∏ g ∈ G, g)).filter (fun n => BothN G d n) :=
    Finset.filter_congr fun n _ => (bothN_iff G d n).symm
  rw [hfil, corr_prod G (fun g hg => by have := h5 g hg; omega) hcop d]
  exact Finset.prod_congr rfl fun g hg => card_both_residues_eval (h5 g hg)

end TopMachine
