/-
MACHINES 41, 43, 47, 53, 59 - THE OPENING PREDICATES, AND THE RESIDUE TEST
(Formalist, round 30).

The corpus defines machines by successive gear additions (`Machine23` ..
`Machine37`: `Exposed_q k := Exposed_prev k ∧ ¬ (q ∣ 6k-1) ∧ ¬ (q ∣ 6k+1)`).
This file carries that chain five gears further, in exactly that shape, so
that a slot of machine 59 is a statement about the REAL machine - the
divisibility of `6k -+ 1` by every prime `5 <= p <= 59`.

    gear   u = 6^{-1} mod q   teeth {u, q - u}
     41         7               {7, 34}      6*7 = 42 = 41 + 1
     43        36               {36, 7}      6*36 = 216 = 5*43 + 1
     47         8               {8, 39}      6*8 = 48 = 47 + 1
     53         9               {9, 44}      6*9 = 54 = 53 + 1
     59        10               {10, 49}     6*10 = 60 = 59 + 1

and the RESIDUE TEST `Open_q k` - the same predicate written on `k mod p`
alone (`Machine19.expT` for the six smallest gears, `Killed_p` for the rest) -
proved equivalent to `Exposed_q k` for `k >= 1`.  Slot facts about numerals
of size 10^20 are then `decide +kernel` on `%`, which the kernel evaluates with
bignum arithmetic; nothing here is `native_decide`.

Only the opening predicates are built (no `opSeq`, no gap function): the
round-30 use is the CRT-slot realisers of `CrtSlots.lean`, which are
statements about consecutive openings, in the shape of round 28's
`Increment.AdjPair`.
-/

import Machine37

namespace MachineUp

/-! ## Gears 41 .. 59 -/

/-- Gear 41 kills slot `k`: teeth `{7, 34}`. -/
def Killed41 (k : ℕ) : Prop := k % 41 = 7 ∨ k % 41 = 34
instance (k : ℕ) : Decidable (Killed41 k) := by unfold Killed41; infer_instance
/-- An opening of machine 41 = gears `{5, .., 41}`. -/
def Exposed41 (k : ℕ) : Prop :=
  Machine37.Exposed37 k ∧ ¬ (41 ∣ Census.lo k) ∧ ¬ (41 ∣ Census.hi k)
instance (k : ℕ) : Decidable (Exposed41 k) := by unfold Exposed41; infer_instance
theorem killed41_iff {k : ℕ} (hk : 1 ≤ k) :
    Killed41 k ↔ (41 ∣ Census.lo k ∨ 41 ∣ Census.hi k) := by
  simp only [Killed41, Census.lo, Census.hi]
  omega

/-- Gear 43 kills slot `k`: teeth `{36, 7}`. -/
def Killed43 (k : ℕ) : Prop := k % 43 = 36 ∨ k % 43 = 7
instance (k : ℕ) : Decidable (Killed43 k) := by unfold Killed43; infer_instance
def Exposed43 (k : ℕ) : Prop :=
  Exposed41 k ∧ ¬ (43 ∣ Census.lo k) ∧ ¬ (43 ∣ Census.hi k)
instance (k : ℕ) : Decidable (Exposed43 k) := by unfold Exposed43; infer_instance
theorem killed43_iff {k : ℕ} (hk : 1 ≤ k) :
    Killed43 k ↔ (43 ∣ Census.lo k ∨ 43 ∣ Census.hi k) := by
  simp only [Killed43, Census.lo, Census.hi]
  omega

/-- Gear 47 kills slot `k`: teeth `{8, 39}`. -/
def Killed47 (k : ℕ) : Prop := k % 47 = 8 ∨ k % 47 = 39
instance (k : ℕ) : Decidable (Killed47 k) := by unfold Killed47; infer_instance
def Exposed47 (k : ℕ) : Prop :=
  Exposed43 k ∧ ¬ (47 ∣ Census.lo k) ∧ ¬ (47 ∣ Census.hi k)
instance (k : ℕ) : Decidable (Exposed47 k) := by unfold Exposed47; infer_instance
theorem killed47_iff {k : ℕ} (hk : 1 ≤ k) :
    Killed47 k ↔ (47 ∣ Census.lo k ∨ 47 ∣ Census.hi k) := by
  simp only [Killed47, Census.lo, Census.hi]
  omega

/-- Gear 53 kills slot `k`: teeth `{9, 44}`. -/
def Killed53 (k : ℕ) : Prop := k % 53 = 9 ∨ k % 53 = 44
instance (k : ℕ) : Decidable (Killed53 k) := by unfold Killed53; infer_instance
def Exposed53 (k : ℕ) : Prop :=
  Exposed47 k ∧ ¬ (53 ∣ Census.lo k) ∧ ¬ (53 ∣ Census.hi k)
instance (k : ℕ) : Decidable (Exposed53 k) := by unfold Exposed53; infer_instance
theorem killed53_iff {k : ℕ} (hk : 1 ≤ k) :
    Killed53 k ↔ (53 ∣ Census.lo k ∨ 53 ∣ Census.hi k) := by
  simp only [Killed53, Census.lo, Census.hi]
  omega

/-- Gear 59 kills slot `k`: teeth `{10, 49}`. -/
def Killed59 (k : ℕ) : Prop := k % 59 = 10 ∨ k % 59 = 49
instance (k : ℕ) : Decidable (Killed59 k) := by unfold Killed59; infer_instance
def Exposed59 (k : ℕ) : Prop :=
  Exposed53 k ∧ ¬ (59 ∣ Census.lo k) ∧ ¬ (59 ∣ Census.hi k)
instance (k : ℕ) : Decidable (Exposed59 k) := by unfold Exposed59; infer_instance
theorem killed59_iff {k : ℕ} (hk : 1 ≤ k) :
    Killed59 k ↔ (59 ∣ Census.lo k ∨ 59 ∣ Census.hi k) := by
  simp only [Killed59, Census.lo, Census.hi]
  omega

/-! ## The residue tests -/

/-- Machine 37 on residues alone. -/
def Open37 (k : ℕ) : Prop :=
  Machine19.expT (k % 5) (k % 7) (k % 11) (k % 13) (k % 17) (k % 19) = true ∧
  ¬ Machine23.Killed23 k ∧ ¬ Machine29.Killed29 k ∧ ¬ Machine31.Killed31 k ∧
  ¬ Machine37.Killed37 k
instance (k : ℕ) : Decidable (Open37 k) := by unfold Open37; infer_instance

/-- Machine 41 on residues alone. -/
def Open41 (k : ℕ) : Prop := Open37 k ∧ ¬ Killed41 k
instance (k : ℕ) : Decidable (Open41 k) := by unfold Open41; infer_instance

/-- Machine 53 on residues alone. -/
def Open53 (k : ℕ) : Prop := Open41 k ∧ ¬ Killed43 k ∧ ¬ Killed47 k ∧ ¬ Killed53 k
instance (k : ℕ) : Decidable (Open53 k) := by unfold Open53; infer_instance

/-- Machine 59 on residues alone. -/
def Open59 (k : ℕ) : Prop := Open53 k ∧ ¬ Killed59 k
instance (k : ℕ) : Decidable (Open59 k) := by unfold Open59; infer_instance

theorem exposed37_iff {k : ℕ} (hk : 1 ≤ k) : Machine37.Exposed37 k ↔ Open37 k := by
  have h23 : (¬ (23 ∣ Census.lo k) ∧ ¬ (23 ∣ Census.hi k)) ↔ ¬ Machine23.Killed23 k := by
    rw [Machine23.killed23_iff hk, not_or]
  have h29 : (¬ (29 ∣ Census.lo k) ∧ ¬ (29 ∣ Census.hi k)) ↔ ¬ Machine29.Killed29 k := by
    rw [Machine29.killed29_iff hk, not_or]
  have h31 : (¬ (31 ∣ Census.lo k) ∧ ¬ (31 ∣ Census.hi k)) ↔ ¬ Machine31.Killed31 k := by
    rw [Machine31.killed31_iff hk, not_or]
  have h37 : (¬ (37 ∣ Census.lo k) ∧ ¬ (37 ∣ Census.hi k)) ↔ ¬ Machine37.Killed37 k := by
    rw [Machine37.killed37_iff hk, not_or]
  unfold Machine37.Exposed37 Machine31.Exposed31 Machine29.Exposed29 Machine23.Exposed23 Open37
  rw [h23, h29, h31, h37, Machine19.exposed19_iff hk]
  simp only [and_assoc]

theorem exposed41_iff {k : ℕ} (hk : 1 ≤ k) : Exposed41 k ↔ Open41 k := by
  have h41 : (¬ (41 ∣ Census.lo k) ∧ ¬ (41 ∣ Census.hi k)) ↔ ¬ Killed41 k := by
    rw [killed41_iff hk, not_or]
  unfold Exposed41 Open41
  rw [h41, exposed37_iff hk]

theorem exposed53_iff {k : ℕ} (hk : 1 ≤ k) : Exposed53 k ↔ Open53 k := by
  have h43 : (¬ (43 ∣ Census.lo k) ∧ ¬ (43 ∣ Census.hi k)) ↔ ¬ Killed43 k := by
    rw [killed43_iff hk, not_or]
  have h47 : (¬ (47 ∣ Census.lo k) ∧ ¬ (47 ∣ Census.hi k)) ↔ ¬ Killed47 k := by
    rw [killed47_iff hk, not_or]
  have h53 : (¬ (53 ∣ Census.lo k) ∧ ¬ (53 ∣ Census.hi k)) ↔ ¬ Killed53 k := by
    rw [killed53_iff hk, not_or]
  unfold Exposed53 Exposed47 Exposed43 Open53
  rw [h43, h47, h53, exposed41_iff hk]
  simp only [and_assoc]

theorem exposed59_iff {k : ℕ} (hk : 1 ≤ k) : Exposed59 k ↔ Open59 k := by
  have h59 : (¬ (59 ∣ Census.lo k) ∧ ¬ (59 ∣ Census.hi k)) ↔ ¬ Killed59 k := by
    rw [killed59_iff hk, not_or]
  unfold Exposed59 Open59
  rw [h59, exposed53_iff hk]

end MachineUp
