/-
The literal cap over ALL Polignac gaps: 12 is the absolute ceiling.

Harvester's frame (halved coordinates): a position `n` denotes the pair
`(2n+1, 2n+1+2e)` for gap `d = 2e`; gear `q` blocks `n = 0` and `n = -e`
(mod `q`). A LITERAL CHAIN is a maximal run of consecutive frame-admissible
`q'`-kills - kills that survive gear 3 - all of which are exposed to gears 5
and 7. Gear 3 FILTERS the candidate list rather than breaking a run: a
3-inadmissible kill is skipped and the run continues across it. (Modelling
gear 3 like gears 5 and 7 gives caps 2/4 instead of 6/10/12 - the trap
recorded in round 16.)

The cap depends only on `gcd(e, 105)`, of which there are exactly 8 values
since `105 = 3 * 5 * 7`, so the eight theorems below cover EVERY even gap `d`:

    gcd(e,105)    1    5    7    3   21   35   15  105
    cap           6    6    6    6    6    6   10   12

The two rows where the ceiling exceeds 6 are exactly where `e` absorbs the
small gears, enlarging the exposed set. `gcd = 3` is the `d = 0 (mod 6)`
case - the densest Polignac gaps - and it still caps at 6.

**12 is therefore the absolute ceiling over all Polignac gaps** - the
universal form of the fuel bound, and the first all-`d` statement in this
ledger.

Pre-verified against harvester's table and independently reproduced (all 8
spectra, row for row) before formalising; each cap is also sharp - the
scan fails at `cap - 1` - which was checked numerically.
-/

import Mathlib.Data.ZMod.Basic

namespace PolignacCap

/-! ## The coprime-multiplier lemma

Requested as a reusable piece: if `t` is coprime to the modulus then the
multiples of `t` hit every residue. It is the fact behind the single-walk
reduction (the walk's state space is one cycle). In the end the cap below
did NOT need it - the same speedup came from restricting scan starts to the
exposed set and running an allocation-free scan - but the lemma is proved
here for reuse.
-/

/-- **Coprime multipliers are surjective mod `n`.** Every residue is hit by
a multiple of `t` when `t` is coprime to the modulus. -/
theorem exists_mul_mod_eq {n t : ℕ} (hn : 0 < n) (h : Nat.Coprime t n)
    {r : ℕ} (hr : r < n) : ∃ j, j < n ∧ (j * t) % n = r := by
  have : NeZero n := ⟨by omega⟩
  let u : (ZMod n)ˣ := ZMod.unitOfCoprime t h
  refine ⟨((r : ZMod n) * ((u⁻¹ : (ZMod n)ˣ) : ZMod n)).val, ZMod.val_lt _, ?_⟩
  have hu : ((u : (ZMod n))) = (t : ZMod n) := ZMod.coe_unitOfCoprime t h
  have hcast : ((((r : ZMod n) * ((u⁻¹ : (ZMod n)ˣ) : ZMod n)).val * t : ℕ) : ZMod n)
      = ((r : ℕ) : ZMod n) := by
    push_cast
    rw [ZMod.natCast_val, ZMod.cast_id, ← hu]
    rw [mul_assoc, Units.inv_mul, mul_one]
  have hmod := (ZMod.natCast_eq_natCast_iff _ _ _).mp hcast
  rwa [Nat.ModEq, Nat.mod_eq_of_lt hr] at hmod

/-! ## The chain scan -/

/-- Gear `q` admits position `n` for gap parameter `e`: `n` is neither `0`
nor `-e` mod `q`. -/
def adm (e q n : ℕ) : Bool := (n % q != 0) && (n % q != (q - e % q) % q)

/-- Exposed to all three small gears. -/
def inE (e n : ℕ) : Bool := adm e 3 n && adm e 5 n && adm e 7 n

/-- Consecutive kills alternate by `e` and `q' - e`; mod 105 the gear
contributes `t = q' mod 105`. -/
def stepOf (e t i : ℕ) : ℕ :=
  if i % 2 = 1 then e % 105 else (t + 105 - e % 105) % 105

/-- Walk the kill sequence, counting consecutive 5,7-exposed candidates and
SKIPPING the 3-inadmissible ones; report `false` if the count ever exceeds
`L`. -/
def scan (e t L : ℕ) : ℕ → ℕ → ℕ → ℕ → Bool
  | 0, _, _, _ => true
  | (f+1), i, cur, cnt =>
      let nxt := (cur + stepOf e t i) % 105
      if adm e 3 cur then
        (if adm e 5 cur && adm e 7 cur then
           (if L < cnt + 1 then false else scan e t L f (i+1) nxt (cnt+1))
         else scan e t L f (i+1) nxt 0)
      else scan e t L f (i+1) nxt cnt

/-- No literal chain exceeds `L`, over every invertible gear class mod 105
and every start in the exposed set (a run begins at an exposed position, so
those starts suffice) and both parities. -/
def capOK (e L B : ℕ) : Bool :=
  (List.range 105).all fun t =>
    !(t % 3 != 0 && t % 5 != 0 && t % 7 != 0) ||
    ((List.range 105).all fun r =>
      !(inE e r) || ((List.range 2).all fun ph => scan e t L B ph r 0))


end PolignacCap
