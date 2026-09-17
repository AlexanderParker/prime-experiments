/-
MirrorWalkSettleFree (round 57, 2026-09-16): the settle walk's step lemma in the free regime.

At a step of the settle walk the candidates are the columns `x + s k` for `k = 0, 1, 2, …`
(`s = 12 g` for the mirror `{2, 3, g}`, up; the same for down with `s = -12 g`).  A gear `h`
coprime to `s` strikes the candidate `k` iff `k` lies in one of two residue classes modulo `h`
(one for each member), so on `2n + 1` consecutive candidates it strikes at most two when
`h > 2n`.  Hence, for a set `G` of `n` gears all coprime to `s` and all larger than `2n`, some
`k ≤ 2n` gives a candidate open to every gear of `G`: the keeping move exists, exactly, with
no residue read.  This is the kernel's `mex_form` (TopMachineWalk) carried from the line
`x + j` to the progression `x + s k`.

The regime it covers: the first steps of the descending walk, while the visited gears are few
and large (`n` visited, all above `2n`).  Beyond it the small gears enter and the statement is
no longer available by this route.
-/
import TopMachineWalk

namespace MirrorWalk

/-- Two residues below `g` that agree modulo `g` are equal. -/
theorem eq_of_modEq_lt {g : ℕ} {k o : ℕ} (hk : k < g) (ho : o < g) (h : (k : ℤ) ≡ (o : ℤ) [ZMOD g]) : k = o := by
  unfold Int.ModEq at h
  rw [Int.emod_eq_of_lt (by omega) (by exact_mod_cast hk), Int.emod_eq_of_lt (by omega) (by exact_mod_cast ho)] at h
  exact_mod_cast h

/-- The class of `k` at which gear `g` strikes `y + s k`: with `a` an inverse of `s` modulo `g`
(`s a ≡ 1`), the class is `-y a`. -/
def offA (g : ℕ) (a y : ℤ) : ℕ := ((-y * a) % (g : ℤ)).toNat

theorem offA_lt {g : ℕ} (hg : 0 < g) (a y : ℤ) : offA g a y < g := by
  have hgz : (0 : ℤ) < (g : ℤ) := by exact_mod_cast hg
  have h1 : (-y * a) % (g : ℤ) < (g : ℤ) := Int.emod_lt_of_pos _ hgz
  have h2 : (0 : ℤ) ≤ (-y * a) % (g : ℤ) := Int.emod_nonneg _ (by omega)
  unfold offA
  have : ((((-y * a) % (g : ℤ)).toNat : ℕ) : ℤ) = (-y * a) % (g : ℤ) := Int.toNat_of_nonneg h2
  omega

/-- `g` strikes `y + s k` iff `k ≡ offA g a y (mod g)`, when `s a ≡ 1 (mod g)`. -/
theorem strikes_iff_offA {g : ℕ} (hg : 0 < g) {s a : ℤ} (hsa : s * a ≡ 1 [ZMOD g]) (y : ℤ) (k : ℤ) :
    (g : ℤ) ∣ y + s * k ↔ k ≡ (offA g a y : ℤ) [ZMOD g] := by
  have hgz : (0 : ℤ) < (g : ℤ) := by exact_mod_cast hg
  have hoff : ((offA g a y : ℕ) : ℤ) = (-y * a) % (g : ℤ) := by
    unfold offA; exact Int.toNat_of_nonneg (Int.emod_nonneg _ (by omega))
  rw [hoff]
  constructor
  · intro hd
    have h0 : y + s * k ≡ 0 [ZMOD g] := (Int.modEq_zero_iff_dvd).mpr hd
    have hsk : s * k ≡ -y [ZMOD g] := by
      have := h0.sub_left y
      have e1 : y - (y + s * k) = -(s * k) := by ring
      rw [e1, sub_zero] at this
      have := this.neg
      simpa using this
    have h1 : k * (s * a) ≡ k * 1 [ZMOD g] := hsa.mul_left k
    have h2 : k * (s * a) = a * (s * k) := by ring
    have h3 : a * (s * k) ≡ a * -y [ZMOD g] := hsk.mul_left a
    have h4 : k ≡ a * -y [ZMOD g] := by
      have h5 : k * 1 ≡ a * (s * k) [ZMOD g] := by rw [← h2]; exact h1.symm
      have h6 : k * 1 ≡ a * -y [ZMOD g] := h5.trans h3
      rwa [mul_one] at h6
    have e : a * -y = -y * a := by ring
    rw [e] at h4
    exact h4.trans (Int.mod_modEq _ _).symm
  · intro hk
    have hk' : k ≡ -y * a [ZMOD g] := hk.trans (Int.mod_modEq _ _)
    have h1 : s * k ≡ s * (-y * a) [ZMOD g] := hk'.mul_left s
    have h2 : s * (-y * a) = -y * (s * a) := by ring
    have h3 : -y * (s * a) ≡ -y * 1 [ZMOD g] := hsa.mul_left (-y)
    have h4 : y + s * k ≡ y + -y * 1 [ZMOD g] := (h1.trans (by rw [h2]; exact h3)).add_left y
    have e : y + -y * 1 = 0 := by ring
    rw [e] at h4
    exact (Int.modEq_zero_iff_dvd).mp h4

/-- The classes of `k` at which some gear of `G` strikes a member of the candidate `x + s k`. -/
def ResA (G : Finset ℕ) (a : ℕ → ℤ) (x : ℤ) : Finset ℕ :=
  G.biUnion (fun g => ({offA g (a g) x, offA g (a g) (x + 2)} : Finset ℕ))

theorem resA_card_le (G : Finset ℕ) (a : ℕ → ℤ) (x : ℤ) : (ResA G a x).card ≤ 2 * G.card := by
  classical
  refine le_trans (Finset.card_biUnion_le) ?_
  have hstep : ∀ g ∈ G, ({offA g (a g) x, offA g (a g) (x + 2)} : Finset ℕ).card ≤ 2 := by
    intro g _; exact Finset.card_le_two
  calc (∑ g ∈ G, ({offA g (a g) x, offA g (a g) (x + 2)} : Finset ℕ).card)
      ≤ ∑ g ∈ G, 2 := Finset.sum_le_sum hstep
    _ = 2 * G.card := by rw [Finset.sum_const, smul_eq_mul, mul_comm]

/-- **The keeping move exists in the free regime.**  If every gear of `G` (`n` gears) exceeds
`2n` and `a g` inverts `s` modulo `g`, some `k ≤ 2n` has `x + s k` open to every gear of `G`. -/
theorem keeping_move_free {G : Finset ℕ} {s : ℤ} {a : ℕ → ℤ}
    (hbig : ∀ g ∈ G, 2 * G.card < g) (hinv : ∀ g ∈ G, s * a g ≡ 1 [ZMOD g]) (x : ℤ) :
    ∃ k : ℕ, k ≤ 2 * G.card ∧ ∀ g ∈ G, ¬ (g : ℤ) ∣ x + s * k ∧ ¬ (g : ℤ) ∣ x + s * k + 2 := by
  classical
  -- a k in range (2n + 1) outside ResA exists since ResA has at most 2n members
  have hsub : ¬ (Finset.range (2 * G.card + 1) ⊆ ResA G a x) := by
    intro h
    have h1 := Finset.card_le_card h
    rw [Finset.card_range] at h1
    have h2 := resA_card_le G a x
    omega
  rw [Finset.not_subset] at hsub
  obtain ⟨k, hk, hnot⟩ := hsub
  refine ⟨k, by have := Finset.mem_range.mp hk; omega, ?_⟩
  intro g hg
  have hgpos : 0 < g := by have := hbig g hg; omega
  have hnotmem : k ∉ ({offA g (a g) x, offA g (a g) (x + 2)} : Finset ℕ) := by
    intro hm; exact hnot (Finset.mem_biUnion.mpr ⟨g, hg, hm⟩)
  simp only [Finset.mem_insert, Finset.mem_singleton, not_or] at hnotmem
  obtain ⟨hne1, hne2⟩ := hnotmem
  have hklt : k < g := by have := Finset.mem_range.mp hk; have := hbig g hg; omega
  constructor
  · intro hd
    have hc := (strikes_iff_offA hgpos (hinv g hg) x (k : ℤ)).mp hd
    exact hne1 (eq_of_modEq_lt hklt (offA_lt hgpos (a g) x) hc)
  · intro hd
    have e : x + s * k + 2 = (x + 2) + s * k := by ring
    rw [e] at hd
    have hc := (strikes_iff_offA hgpos (hinv g hg) (x + 2) (k : ℤ)).mp hd
    exact hne2 (eq_of_modEq_lt hklt (offA_lt hgpos (a g) (x + 2)) hc)

/-- **The free regime's cut is sharp.**  The hypothesis that every visited gear exceeds twice
their number cannot be dropped: with the three gears `{5, 7, 11}` - one of which, 5, is below
`2 · 3` - and the start 370, every candidate of the run `k = 0 … 6` is struck, so no keeping move
exists.  (370 and 375 are struck by 5, 371 and 378 by 7, 374 by 11.)

This is what closes the pigeonhole route: the lemma covers exactly the steps where the gears are
large against their number, and one small gear is enough to take the conclusion away. -/
theorem keeping_move_free_sharp :
    ∃ (G : Finset ℕ) (x : ℕ), G.card = 3 ∧ (∃ g ∈ G, g ≤ 2 * G.card) ∧
      ∀ k, k ≤ 2 * G.card → ∃ g ∈ G, g ∣ (x + k) ∨ g ∣ (x + k + 2) := by
  refine ⟨{5, 7, 11}, 370, by decide, ⟨5, by decide, by decide⟩, ?_⟩
  decide

end MirrorWalk
