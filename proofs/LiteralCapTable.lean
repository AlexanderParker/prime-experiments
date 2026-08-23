/-
The word-list enumeration: the EXACT literal cap for each of the 48
invertible classes mod 210 - part (A) of the route, closed.

`LiteralCap.lean` kernel-checked the uniform bound (no class admits a run
of seven, so every literal chain has at most 6 members) and the sharpness
of 6. What remained COMPUTED but not checked was the per-class table -
which cap each class actually carries - and that table is what fixes the
WORD LIST of constructor's R21/R26: for gear `q'` the compatible literal
words are exactly the alternating words over the letter alphabet
`{2u', q' - 2u'}` of length `1 .. capC(q' mod 210) - 1` (two words per
length, one per starting letter; a chain of `m` members carries `m - 1`
letters). The list depends on `q' mod 210` alone because the cap does.

This file kernel-checks the table both ways:

* `cap_table_maximal`  - NO class admits a run of `capC c + 1` (the walk
  leaves the exposed set), so `capC c` is an upper bound;
* `cap_table_realized` - every class admits a run of `capC c` in the
  corridor, so the table is EXACT, not just safe.

`literal_chain_le_capC` then replaces the uniform 6 by the class's own
value in the chain bound, and `word_length_lt_capC` restates it for the
word: a literal word at gear `q'` has fewer than `capC (q' mod 210)`
letters. The spectrum theorems record the full census {2: 24, 3: 4,
4: 14, 6: 6} - in particular NO class has cap 5 (`no_cap_five`).

Verified against research/literal_cap_gap_d.py before formalising: the
per-class caps match the 140-step max-run computation at all 48 classes
(zero mismatches; that script also checked class invariance against every
prime to 2000), and every realized chain length in
research/data/fuel_census.csv respects its class cap (saturating it at
q' = 19 and 31).
-/

import LiteralCap

namespace LiteralCapTable

/-! ## The run test at every length -/

/-- The first `L` walk members are all exposed - `LiteralCap.run6/run7`
generalised to any length. -/
def runL (t s r ph L : ℕ) : Bool :=
  (List.range L).all fun i => decide (LiteralCap.wpos t s r ph i ∈ Corridor.exposedSet)

/-- Some start and parity admit a run of `L` at class `c`. -/
def hasRunL (c L : ℕ) : Bool :=
  (List.range 35).any fun r => (List.range 2).any fun ph =>
    runL (c % 35) (LiteralCap.sOf c) r ph L

/-- A run witnesses every shorter run: its prefix. -/
theorem hasRunL_mono (c : ℕ) {L L' : ℕ} (h : L ≤ L')
    (hL' : hasRunL c L' = true) : hasRunL c L = true := by
  rw [hasRunL, List.any_eq_true] at hL' ⊢
  obtain ⟨r, hr, hL'⟩ := hL'
  refine ⟨r, hr, ?_⟩
  rw [List.any_eq_true] at hL' ⊢
  obtain ⟨ph, hph, hL'⟩ := hL'
  refine ⟨ph, hph, ?_⟩
  rw [runL, List.all_eq_true] at hL' ⊢
  intro i hi
  exact hL' i (List.mem_range.mpr (lt_of_lt_of_le (List.mem_range.mp hi) h))

/-! ## The table -/

/-- **The cap table.** The literal cap of each invertible class mod 210,
read off the census: 6 at six classes, 4 at fourteen, 3 at four, 2 at the
remaining twenty-four. -/
def capC (c : ℕ) : ℕ :=
  if c ∈ [37, 53, 83, 127, 157, 173] then 6
  else if c ∈ [1, 23, 31, 61, 67, 89, 97, 113, 121, 143, 149, 179, 187, 209] then 4
  else if c ∈ [29, 59, 151, 181] then 3
  else 2

theorem capC_le_six (c : ℕ) : capC c ≤ 6 := by
  unfold capC; split_ifs <;> omega

set_option maxRecDepth 40000 in
/-- **The table is an upper bound.** No invertible class mod 210 admits a
run of `capC c + 1` exposed walk members: 48 classes x 35 starts x 2
parities, decided by the kernel. This refines `LiteralCap.no_run_seven`
to each class's own value. -/
theorem cap_table_maximal :
    ∀ c < 210, Nat.gcd c 210 = 1 →
      ∀ r < 35, ∀ ph < 2,
        runL (c % 35) (LiteralCap.sOf c) r ph (capC c + 1) = false := by
  decide

set_option maxRecDepth 40000 in
/-- **The table is attained.** Every invertible class admits a run of
exactly `capC c` in the corridor, so no entry can be lowered. -/
theorem cap_table_realized :
    ∀ c < 210, Nat.gcd c 210 = 1 → hasRunL c (capC c) = true := by
  decide

/-! ## From the table to the chains -/

/-- **The per-class literal cap.** A literal chain at gear `q` has at most
`capC (q mod 210)` members - the exact class value, replacing the uniform
6 of `LiteralCap.literal_chain_le_six`. -/
theorem literal_chain_le_capC {q u r ph L : ℕ}
    (hu : 6 * u + 1 = q ∨ 6 * u = q + 1) (hq : Nat.gcd q 210 = 1)
    (hph : ph < 2) (hr : 1 ≤ r)
    (hE : ∀ i < L, Corridor.Exposed (LiteralCap.member r q u ph i)) :
    L ≤ capC (q % 210) := by
  by_contra hL
  have hc : Nat.gcd (q % 210) 210 = 1 := by
    rw [← Nat.gcd_rec 210 q, Nat.gcd_comm]; exact hq
  have hcheck := cap_table_maximal (q % 210) (Nat.mod_lt _ (by omega)) hc
    (r % 35) (Nat.mod_lt _ (by omega)) ph hph
  have hcap : capC (q % 210) ≤ 6 := capC_le_six _
  have hrun : runL ((q % 210) % 35) (LiteralCap.sOf (q % 210)) (r % 35) ph
      (capC (q % 210) + 1) = true := by
    rw [runL, List.all_eq_true]
    intro i hi
    have hi7 : i < 7 := by have := List.mem_range.mp hi; omega
    have hEi : Corridor.Exposed (LiteralCap.member r q u ph i) := by
      have := List.mem_range.mp hi
      exact hE i (by omega)
    have hmem := (Corridor.exposed_iff_mem
      (show 1 ≤ LiteralCap.member r q u ph i by unfold LiteralCap.member; omega)).mp hEi
    have hres : LiteralCap.wpos ((q % 210) % 35) (LiteralCap.sOf (q % 210)) (r % 35) ph i
        = LiteralCap.member r q u ph i % 35 := by
      rw [← LiteralCap.s_eq hu]
      unfold LiteralCap.wpos LiteralCap.member
      interval_cases ph <;> interval_cases i <;> simp <;> omega
    rw [decide_eq_true_iff, hres]
    exact hmem
  rw [hcheck] at hrun
  exact Bool.false_ne_true hrun

/-- **(A) in word form.** A literal word of `ell` letters is carried by a
chain of `ell + 1` members, so `ell < capC (q mod 210)`: the word list of
R21/R26 - alternating words of length `1 .. capC - 1` - is complete, as a
kernel-checked function of `q' mod 210` alone. -/
theorem word_length_lt_capC {q u r ph ell : ℕ}
    (hu : 6 * u + 1 = q ∨ 6 * u = q + 1) (hq : Nat.gcd q 210 = 1)
    (hph : ph < 2) (hr : 1 ≤ r)
    (hE : ∀ i < ell + 1, Corridor.Exposed (LiteralCap.member r q u ph i)) :
    ell < capC (q % 210) :=
  literal_chain_le_capC hu hq hph hr hE

/-! ## The census -/

/-- Cap 2: the twenty-four generic classes. -/
theorem cap_two_classes :
    ((Finset.range 210).filter fun c => Nat.gcd c 210 = 1 ∧ capC c = 2)
      = {11, 13, 17, 19, 41, 43, 47, 71, 73, 79, 101, 103, 107, 109,
         131, 137, 139, 163, 167, 169, 191, 193, 197, 199} := by
  decide

/-- Cap 3: four classes. -/
theorem cap_three_classes :
    ((Finset.range 210).filter fun c => Nat.gcd c 210 = 1 ∧ capC c = 3)
      = {29, 59, 151, 181} := by
  decide

/-- Cap 4: fourteen classes. -/
theorem cap_four_classes :
    ((Finset.range 210).filter fun c => Nat.gcd c 210 = 1 ∧ capC c = 4)
      = {1, 23, 31, 61, 67, 89, 97, 113, 121, 143, 149, 179, 187, 209} := by
  decide

/-- Cap 6: the six sharp classes - the same set as
`LiteralCap.cap_six_classes_sharp`. -/
theorem cap_six_classes :
    ((Finset.range 210).filter fun c => Nat.gcd c 210 = 1 ∧ capC c = 6)
      = {37, 53, 83, 127, 157, 173} := by
  decide

/-- **No class has cap 5.** The spectrum {2, 3, 4, 6} has a hole. -/
theorem no_cap_five :
    ((Finset.range 210).filter fun c => Nat.gcd c 210 = 1 ∧ capC c = 5)
      = ∅ := by
  decide

/-- The census counts: 24 + 4 + 14 + 6 = 48. -/
theorem cap_spectrum_counts :
    ((Finset.range 210).filter fun c => Nat.gcd c 210 = 1 ∧ capC c = 2).card = 24 ∧
    ((Finset.range 210).filter fun c => Nat.gcd c 210 = 1 ∧ capC c = 3).card = 4 ∧
    ((Finset.range 210).filter fun c => Nat.gcd c 210 = 1 ∧ capC c = 4).card = 14 ∧
    ((Finset.range 210).filter fun c => Nat.gcd c 210 = 1 ∧ capC c = 6).card = 6 := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> decide

/-! ## The T3 law (lateral round 20, frequency frame)

Lateral's DFT factorisation found that at local frequency 3 every gear is
nearly a single point in phase: the TRIPLED teeth `+-3u` are the two
adjacent residues at the antipode `(q -+ 1)/2`. That is pure tooth
arithmetic - `6u = q -+ 1` tripled - and lands as a kernel fact with no
Fourier machinery in the statement. Lateral asserted it numerically to
100,000; here it is for every gear forever. -/

/-- **The T3 law.** Tripled tooth offsets sit at the antipode: if
`6u = q - 1` then `2 * (3u) + 1 = q` (so `3u = (q-1)/2` exactly), and if
`6u = q + 1` then `2 * (3u) - 1 = q` (so `3u = (q+1)/2`). The pair
`{3u, q - 3u}` is `{(q-1)/2, (q+1)/2}` in both cases: two ADJACENT
residues straddling the antipode - the phase form of the tooth law. -/
theorem tripled_teeth_antipode {q u : ℕ} (hu : 6 * u + 1 = q ∨ 6 * u = q + 1) :
    (2 * (3 * u) + 1 = q ∧ q - 3 * u = 3 * u + 1) ∨
      (2 * (3 * u) = q + 1 ∧ 3 * u = (q - 3 * u) + 1) := by
  rcases hu with h | h
  · exact Or.inl (by omega)
  · exact Or.inr (by omega)

end LiteralCapTable
