/-
THE BARE-ALTERNATION LEMMA AT REAL MACHINES (Formalist, round 31).

`BareAlternation.lean` proves the necessary condition over an abstract machine
and computes the inadmissible set `S` (28 of the 48 classes mod 210).  This
file discharges the two blocking hypotheses at the corpus's machines and reads
off the consequences.  Nothing here is a census: every step is either a
projection of the machine's `Exposed` conjunction or a `decide` on a 3- or
4-element list of numerals.

    machine M   q'   a    b    q' in S?   what is proved here
    m23         29   10   19   yes        no realised bare (10,19,10) or
                                          (19,10,19) - L_bare(23) <= 2
    m37         41   14   27   yes        no realised bare PAIR (14,27) or
                                          (27,14) at all - L_bare(37) <= 1
    m41         43   14   29   yes        no slot has k, k+14, k+43 all open,
                                          and none has k, k+29, k+43
    m43         47   16   31   yes        the same at (16, 47) and (31, 47)

The last two are stated on the OPENING PREDICATE only (`MachineUp.Exposed41`,
`Exposed43` have no opening enumeration in the ledger), which is the stronger
statement anyway: the three slots need not be consecutive openings.

CONTEXT - what this does and does not say.  `L_bare(37) <= 1` is NOT
`L(37) <= 1`: the corpus has `L(37) = 2`, and the two are consistent because a
legal word may use a padded letter (a gap of exactly `q' = 41`) or a non-bare
literal (`a + q' = 55`, ...).  So this file's m37/m41/m43 rows are a
PREDICTION about the census: every depth-2 legal word at m37, m41 and m43 has
a non-bare letter.  Constructor's counted census R102 already exhibits such
words at m37 (`occ(14,41;37) = 1,525`, `occ(27,41;37) = 1` - both padded).
-/

import BareAlternation
import MachineUp

namespace BareAltInst

open BareAlt

/-! ## 1. Gears 5 and 7 block every machine of the corpus

Gear 5's teeth are the slot residues `{1, 4}` and gear 7's are `{6, 1}`
(`u = 6⁻¹ mod g`; `Machine11.expT`).  An opening of any machine
`{5, 7, ...}` carries the gear-5 and gear-7 clauses, and
`5 ∣ 6k-1 ↔ k ≡ 1`, `5 ∣ 6k+1 ↔ k ≡ 4`, `7 ∣ 6k-1 ↔ k ≡ 6`,
`7 ∣ 6k+1 ↔ k ≡ 1` (`omega`, exactly as in `Machine19.exposed19_iff`). -/

theorem blocks19_five : Blocks Machine19.Exposed19 5 1 := by
  intro k hk
  have h1 : ¬ (5 ∣ Census.lo k) := hk.1
  have h2 : ¬ (5 ∣ Census.hi k) := hk.2.1
  simp only [Census.lo, Census.hi] at h1 h2
  exact ⟨by omega, by omega⟩

theorem blocks19_seven : Blocks Machine19.Exposed19 7 6 := by
  intro k hk
  have h1 : ¬ (7 ∣ Census.lo k) := hk.2.2.1
  have h2 : ¬ (7 ∣ Census.hi k) := hk.2.2.2.1
  simp only [Census.lo, Census.hi] at h1 h2
  exact ⟨by omega, by omega⟩

/-- Blocking transports down any refinement of the opening predicate. -/
theorem blocks_mono {E E' : ℕ → Prop} {g u : ℕ} (h : Blocks E g u)
    (hsub : ∀ k, E' k → E k) : Blocks E' g u := fun k hk => h k (hsub k hk)

theorem exposed23_exposed19 {k : ℕ} (h : Machine23.Exposed23 k) :
    Machine19.Exposed19 k := h.1

theorem exposed37_exposed19 {k : ℕ} (h : Machine37.Exposed37 k) :
    Machine19.Exposed19 k := h.1.1.1.1

theorem exposed41_exposed19 {k : ℕ} (h : MachineUp.Exposed41 k) :
    Machine19.Exposed19 k := exposed37_exposed19 h.1

theorem exposed43_exposed19 {k : ℕ} (h : MachineUp.Exposed43 k) :
    Machine19.Exposed19 k := exposed41_exposed19 h.1

theorem blocks23_five : Blocks Machine23.Exposed23 5 1 :=
  blocks_mono blocks19_five fun _ => exposed23_exposed19
theorem blocks23_seven : Blocks Machine23.Exposed23 7 6 :=
  blocks_mono blocks19_seven fun _ => exposed23_exposed19
theorem blocks37_five : Blocks Machine37.Exposed37 5 1 :=
  blocks_mono blocks19_five fun _ => exposed37_exposed19
theorem blocks37_seven : Blocks Machine37.Exposed37 7 6 :=
  blocks_mono blocks19_seven fun _ => exposed37_exposed19
theorem blocks41_five : Blocks MachineUp.Exposed41 5 1 :=
  blocks_mono blocks19_five fun _ => exposed41_exposed19
theorem blocks41_seven : Blocks MachineUp.Exposed41 7 6 :=
  blocks_mono blocks19_seven fun _ => exposed41_exposed19
theorem blocks43_five : Blocks MachineUp.Exposed43 5 1 :=
  blocks_mono blocks19_five fun _ => exposed43_exposed19
theorem blocks43_seven : Blocks MachineUp.Exposed43 7 6 :=
  blocks_mono blocks19_seven fun _ => exposed43_exposed19

/-! ## 2. Machine 23 with gear 29: the general lemma, instantiated

`q' = 29`, `29 % 210 = 29 ∈ S`, and the bare letters are `a = 10 = 2u'`
(`u' = 6⁻¹ = 5 mod 29`), `b = 19`.  The class arithmetic is checked by the
kernel, not asserted. -/

theorem letters29 : aOfClass 29 = 10 ∧ bOfClass 29 = 19 := by decide

theorem mem29 : (29 : ℕ) % 210 ∈ S := by decide

/-- **L_bare(23) ≤ 2**, from the class membership `29 ∈ S` through
`no_bare3_of_class_mem`: no index of machine 23's opening enumeration carries
the three consecutive gaps `(10,19,10)` or `(19,10,19)`. -/
theorem m23_no_bare3 (i : ℕ) :
    ¬ GapWordAt Machine23.opSeq23 i (altWord 10 19 3) ∧
      ¬ GapWordAt Machine23.opSeq23 i (altWord 19 10 3) :=
  no_bare3_of_class_mem Machine23.opSeq23_exposed blocks23_five blocks23_seven
    (q := 29) (a := 10) (b := 19) (Or.inr (by norm_num)) (by norm_num)
    mem29 (by norm_num) (by decide) i

/-- ... and hence no bare alternating run of ANY length ≥ 3. -/
theorem m23_no_bare_ge {n : ℕ} (hn : 3 ≤ n) (i : ℕ) :
    ¬ GapWordAt Machine23.opSeq23 i (altWord 10 19 n) ∧
      ¬ GapWordAt Machine23.opSeq23 i (altWord 19 10 n) :=
  no_bare_run_ge Machine23.opSeq23_exposed blocks23_five blocks23_seven
    (by decide : bareAdmAB 10 19 3 = false) hn i

/-! ## 3. Machine 37 with gear 41: the bare alternation dies at TWO letters

`q' = 41`, `a = 14`, `b = 27`.  Already the 3-point set `{0, 14, 41}` fits
nowhere at gears 5 and 7 (and `{0, 27, 41}` likewise), so machine 37 has no
bare literal PAIR at all - `L_bare(37) ≤ 1` against the corpus's `L(37) = 2`. -/

theorem letters41 : aOfClass 41 = 14 ∧ bOfClass 41 = 27 := by decide

theorem mem41 : (41 : ℕ) % 210 ∈ S := by decide

theorem m37_no_bare2 (i : ℕ) :
    ¬ GapWordAt Machine37.opSeq37 i (altWord 14 27 2) ∧
      ¬ GapWordAt Machine37.opSeq37 i (altWord 27 14 2) :=
  no_bare_run Machine37.opSeq37_exposed blocks37_five blocks37_seven
    (by decide : bareAdmAB 14 27 2 = false) i

theorem m37_no_bare_ge {n : ℕ} (hn : 2 ≤ n) (i : ℕ) :
    ¬ GapWordAt Machine37.opSeq37 i (altWord 14 27 n) ∧
      ¬ GapWordAt Machine37.opSeq37 i (altWord 27 14 n) :=
  no_bare_run_ge Machine37.opSeq37_exposed blocks37_five blocks37_seven
    (by decide : bareAdmAB 14 27 2 = false) hn i

/-! ## 4. Machines 41 and 43, on the opening predicate alone

No opening enumeration is needed: the offsets of a bare pair are never all
open, consecutive or not. -/

theorem m41_no_bare_offsets (k : ℕ) :
    ¬ (MachineUp.Exposed41 k ∧ MachineUp.Exposed41 (k + 14) ∧
        MachineUp.Exposed41 (k + 43)) := by
  intro h
  refine not_open_of_not_fits (by norm_num) blocks41_five
    (by decide : fitsB 5 1 [0, 14, 43] = false) k ?_
  intro o ho
  simp only [List.mem_cons, List.not_mem_nil, or_false] at ho
  rcases ho with rfl | rfl | rfl
  · simpa using h.1
  · exact h.2.1
  · exact h.2.2

theorem m41_no_bare_offsets_B (k : ℕ) :
    ¬ (MachineUp.Exposed41 k ∧ MachineUp.Exposed41 (k + 29) ∧
        MachineUp.Exposed41 (k + 43)) := by
  intro h
  refine not_open_of_not_fits (by norm_num) blocks41_five
    (by decide : fitsB 5 1 [0, 29, 43] = false) k ?_
  intro o ho
  simp only [List.mem_cons, List.not_mem_nil, or_false] at ho
  rcases ho with rfl | rfl | rfl
  · simpa using h.1
  · exact h.2.1
  · exact h.2.2

theorem m43_no_bare_offsets (k : ℕ) :
    ¬ (MachineUp.Exposed43 k ∧ MachineUp.Exposed43 (k + 16) ∧
        MachineUp.Exposed43 (k + 47)) := by
  intro h
  refine not_open_of_not_fits (by norm_num) blocks43_five
    (by decide : fitsB 5 1 [0, 16, 47] = false) k ?_
  intro o ho
  simp only [List.mem_cons, List.not_mem_nil, or_false] at ho
  rcases ho with rfl | rfl | rfl
  · simpa using h.1
  · exact h.2.1
  · exact h.2.2

theorem m43_no_bare_offsets_B (k : ℕ) :
    ¬ (MachineUp.Exposed43 k ∧ MachineUp.Exposed43 (k + 31) ∧
        MachineUp.Exposed43 (k + 47)) := by
  intro h
  refine not_open_of_not_fits (by norm_num) blocks43_five
    (by decide : fitsB 5 1 [0, 31, 47] = false) k ?_
  intro o ho
  simp only [List.mem_cons, List.not_mem_nil, or_false] at ho
  rcases ho with rfl | rfl | rfl
  · simpa using h.1
  · exact h.2.1
  · exact h.2.2

end BareAltInst
