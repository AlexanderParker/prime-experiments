/-
THE UNIFORM-ORDER FINITE CHECK: 48 CLASSES MOD 210 (Constructor R74).

R74's theorem is

    A_relax(M) <= 5 for every machine M = {5..y}, and <= 4 unless
    q' = nextprime(y) is 37, 53, 83, 127, 157 or 173 (mod 210),

and its whole content is arithmetic at gears 5 and 7.  For the m-letter
alternation over the two literal letters `a = 2u'`, `b = q' - 2u'` the
prefix-sum offsets are

    X = {0, a, q', q' + a, 2q', ...}          (R74's m-point set)

and `3a = q' -+ 1` with the sign fixed by `q' mod 6`, so `X mod g` is a
function of `q' mod 210` alone for `g = 5, 7`.  The word occurs NOWHERE if for
some gear no translate of `X` fits inside `E_g = Z_g \ {+-6^{-1} mod g}`
(Mechanic's phase saturation).  Enumerating the 48 invertible classes is a
FINITE CHECK, and this file is that check in the kernel.

WHAT IS CHECKED HERE, exactly:

  `fitsAt g u inv3 c m sA`  a translate of the m-point alternation set of class
                            `c`, started with letter `a` (`sA = true`) or `b`,
                            fits inside `E_g`.
  `survMin c m`             it fits at gears 5 AND 7, for BOTH start letters
                            (R74's minimising convention - one broken window
                            kills a cycle).
  `survMax c m`             it fits at gears 5 and 7 for SOME start letter
                            (the maximising convention - chain existence).
  `psMin c` / `psMax c`     the largest such `m` (a count over sizes 1..9;
                            `surv_downward` certifies that the count IS the
                            largest, so the definition is not a fudge).

  THEOREM `ps_min_le_five`     psMin <= 5 at all 48 classes           <- R74
  THEOREM `ps_min_five_iff`    psMin = 5 exactly at {37,53,83,127,157,173}
  THEOREM `ps_min_four_iff`    psMin >= 4 exactly there and at {23,187}
  THEOREM `ps_min_counts`      the distribution 24 / 16 / 2 / 6
  THEOREM `ps_max_eq_capC`     psMax c = LiteralCapTable.capC c, ALL 48

The last one is R74's "two invariants the project found independently, five
rounds apart, are one object", as a kernel identity: the maximising order at
gears 5 and 7 IS the literal cap, class by class, 48 of 48.  The corpus
computes `capC` by a WALK IN THE CORRIDOR mod 35 (`LiteralCapTable.runL`);
this file computes the order by TRANSLATES AT THE TWO GEARS SEPARATELY.  The
two vehicles never share a line of code, and they agree everywhere.

WHAT IS NOT CHECKED HERE, and it is the honest boundary.  `A_relax` itself -
the nilpotency index of Constructor's residue-qualifying successor operator -
is not defined in this file, so `A_relax <= 5` is not asserted here.  The step
from `psMin` to `A_relax` is R74's own reduction (a cycle in the operator
forces both rotations, hence the minimising convention), and it is stated with
a labelled `sorry` in `AlternationARelax.lean`, which is NOT registered in the
default build.  Gate: `research/anchor235/r29_arelax_gate.py`, which
reproduces every number below from an independent Python implementation
(distribution 24/16/2/6, the order-4 classes {23, 187}, the order-5 classes =
the litcap six, and `psMax = capC` at all 48 classes).
-/

import LiteralCapTable

namespace AlternationOrder

/-- `a = 2u'` reduced mod `g`, from the class `c = q' mod 210`:
`3a = c - 1` when `c = 1 mod 6` and `3a = c + 1` otherwise, and `inv3` is
`3⁻¹ mod g`. -/
def aMod (g inv3 c : Nat) : Nat :=
  if c % 6 = 1 then (inv3 * (c - 1)) % g else (inv3 * (c + 1)) % g

/-- **The finite fit test at one gear.**  `u` is `6⁻¹ mod g`, so `E_g` is
`Z_g` minus `{u, g - u}`; `m` is the number of POINTS of the alternation. -/
def fitsAt (g u inv3 c m : Nat) (sA : Bool) : Bool :=
  let a := aMod g inv3 c
  let b := (c + g - a) % g
  let lead := if sA then a else b
  (List.range g).any fun t =>
    (List.range m).all fun i =>
      let x := (t + (i / 2) * c + (if i % 2 = 1 then lead else 0)) % g
      x != u && x != g - u

/-- Fits at gear 5 (`u = 1`, `3⁻¹ = 2`) and gear 7 (`u = 6`, `3⁻¹ = 5`). -/
def fits (c m : Nat) (sA : Bool) : Bool :=
  fitsAt 5 1 2 c m sA && fitsAt 7 6 5 c m sA

/-- R74's minimising convention: both start letters survive. -/
def survMin (c m : Nat) : Bool := fits c m true && fits c m false

/-- The maximising convention: some start letter survives. -/
def survMax (c m : Nat) : Bool := fits c m true || fits c m false

/-- The order: the number of sizes `1..9` that survive.  `surv_downward` makes
this the LARGEST surviving size. -/
def psMin (c : Nat) : Nat := ((List.range 9).filter fun m => survMin c (m + 1)).length

/-- The same in the maximising convention. -/
def psMax (c : Nat) : Nat := ((List.range 9).filter fun m => survMax c (m + 1)).length

/-! ## The finite checks -/

set_option maxRecDepth 40000

/-- Survival is downward closed in the size (the point sets are nested), so
`psMin` and `psMax` really are maxima and not just counts. -/
theorem surv_downward : ∀ c < 210, Nat.gcd c 210 = 1 → ∀ m < 9,
    (survMin c (m + 2) = true → survMin c (m + 1) = true) ∧
    (survMax c (m + 2) = true → survMax c (m + 1) = true) := by decide

/-- **THE UNIFORM ORDER, finite half.**  No class reaches order 6. -/
theorem ps_min_le_five : ∀ c < 210, Nat.gcd c 210 = 1 → psMin c ≤ 5 := by decide

/-- **The six exceptional classes are exactly the litcap-6 classes.** -/
theorem ps_min_five_iff : ∀ c < 210, Nat.gcd c 210 = 1 →
    (psMin c = 5 ↔ (c = 37 ∨ c = 53 ∨ c = 83 ∨ c = 127 ∨ c = 157 ∨ c = 173)) := by
  decide

/-- Order 4 or more happens only at the six plus `{23, 187}`. -/
theorem ps_min_four_iff : ∀ c < 210, Nat.gcd c 210 = 1 →
    (4 ≤ psMin c ↔ (c = 23 ∨ c = 187 ∨ c = 37 ∨ c = 53 ∨ c = 83 ∨ c = 127 ∨
      c = 157 ∨ c = 173)) := by decide

/-- **The distribution: 24 / 16 / 2 / 6.** -/
theorem ps_min_counts :
    ((Finset.range 210).filter fun c => Nat.gcd c 210 = 1 ∧ psMin c = 2).card = 24 ∧
    ((Finset.range 210).filter fun c => Nat.gcd c 210 = 1 ∧ psMin c = 3).card = 16 ∧
    ((Finset.range 210).filter fun c => Nat.gcd c 210 = 1 ∧ psMin c = 4).card = 2 ∧
    ((Finset.range 210).filter fun c => Nat.gcd c 210 = 1 ∧ psMin c = 5).card = 6 := by
  decide

/-- **PHASE SATURATION AT `{5,7}` IS THE LITERAL CAP.**  The maximising order
equals `LiteralCapTable.capC` at every one of the 48 invertible classes -
computed here by translates at the two gears, computed there by a walk in the
corridor mod 35.  Two independently-derived project invariants, one object. -/
theorem ps_max_eq_capC : ∀ c < 210, Nat.gcd c 210 = 1 →
    psMax c = LiteralCapTable.capC c := by decide

/-! ## R74's statement, with its one non-finite step as an explicit hypothesis

Rather than leave `A_relax <= 5` as a `sorry`, it is stated here in the corpus's
hypothesis-explicit style: `A_relax` is any integer-valued invariant of the
incoming gear that R74's reduction bounds by the minimising order (a cycle in
the residue-qualifying successor operator forces BOTH rotations of the
alternation, so both start letters must survive).  The finite half is
kernel-proved above; the reduction is Constructor's and is the hypothesis. -/

/-- **`A_relax <= 5` EVERYWHERE**, given R74's reduction to the minimising
order.  Uniform in the machine: no gear list appears. -/
theorem arelax_le_five (ARelax : ℕ → ℕ)
    (hred : ∀ q, Nat.gcd (q % 210) 210 = 1 → ARelax q ≤ psMin (q % 210))
    (q : ℕ) (hq : Nat.gcd (q % 210) 210 = 1) : ARelax q ≤ 5 :=
  le_trans (hred q hq) (ps_min_le_five _ (Nat.mod_lt q (by norm_num)) hq)

/-- **`A_relax <= 4` off the six exceptional classes.** -/
theorem arelax_le_four (ARelax : ℕ → ℕ)
    (hred : ∀ q, Nat.gcd (q % 210) 210 = 1 → ARelax q ≤ psMin (q % 210))
    (q : ℕ) (hq : Nat.gcd (q % 210) 210 = 1)
    (hex : ¬ (q % 210 = 37 ∨ q % 210 = 53 ∨ q % 210 = 83 ∨ q % 210 = 127 ∨
      q % 210 = 157 ∨ q % 210 = 173)) : ARelax q ≤ 4 := by
  have hlt : q % 210 < 210 := Nat.mod_lt q (by norm_num)
  have h5 := ps_min_le_five _ hlt hq
  have hne : psMin (q % 210) ≠ 5 := fun h =>
    hex ((ps_min_five_iff _ hlt hq).mp h)
  exact le_trans (hred q hq) (by omega)

/-- `psMax` is never more than 6 - the literal cap's ceiling, re-derived. -/
theorem ps_max_le_six : ∀ c < 210, Nat.gcd c 210 = 1 → psMax c ≤ 6 := by
  intro c hc hg
  rw [ps_max_eq_capC c hc hg]
  exact LiteralCapTable.capC_le_six c

end AlternationOrder
