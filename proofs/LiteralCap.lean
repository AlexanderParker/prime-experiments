/-
The literal cap: a literal chain has at most 6 members, for every gear,
forever.

Constructor section 23.2. A literal chain for a new gear `q'` has members at

    r, r + 2u', r + q', r + q' + 2u', r + 2q', ...

(the interleaved walk with the alternating literal spacings `{2u', q'-2u'}`,
`u'` the tooth offset with `6u' = q' -+ 1`). Every member is an opening, so
every member's residue lies in the 15-element exposed set `E` mod 35 of the
(5,7) corridor. The walk's residues depend only on `q' mod 35` and
`2u' mod 35`, and both are determined by `q' mod 210` - so the whole question
is a finite check over the 48 invertible classes mod 210.

The check, `no_run_seven`, is stated in the sharpest useful form: NO class
admits seven consecutive exposed walk members. Hence every literal chain has
at most 6 members (`literal_chain_le_six`) - unconditionally, at every gear,
with no bound on `q'`.

`cap_six_classes_sharp` records that the bound is attained, at exactly the
six classes `q' = 37, 53, 83, 127, 157, 173 (mod 210)`, so 6 cannot be
improved.

Verified against research/literal_cap_gap_d.py and the constructor's table
before formalising: 48 classes, cap spectrum {2:24, 3:4, 4:14, 6:6}, max cap
6, cap-6 classes exactly as above; `6u' = q' -+ 1` and the closed form for
`2u' mod 35` checked against every prime to 5000, zero mismatches.
-/

import Corridor

namespace LiteralCap

/-! ## The walk, as a function of the class mod 210 -/

/-- The doubled tooth offset `2u' mod 35`, read off the class `c = q' mod 210`.
For `q' = 1 mod 6` we have `6u' = q' - 1`, hence `2u' = (q'-1)/3`; for
`q' = 5 mod 6`, `6u' = q' + 1` and `2u' = (q'+1)/3`. Both descend to `c`
because the discarded multiple of 210 contributes a multiple of 70. -/
def sOf (c : ℕ) : ℕ := (if c % 6 = 1 then (c - 1) / 3 else (c + 1) / 3) % 35

/-- Residue of the `i`-th walk member: `j` full steps of `q'` plus, on odd
parity, one tooth step. `ph` is the parity the chain starts on. -/
def wpos (t s r ph i : ℕ) : ℕ :=
  (r + ((i + ph) / 2) * t + (if (i + ph) % 2 = 1 then s else 0)) % 35

/-- Seven consecutive walk members, all exposed. -/
def run7 (t s r ph : ℕ) : Bool :=
  (List.range 7).all fun i => decide (wpos t s r ph i ∈ Corridor.exposedSet)

/-- Six consecutive walk members, all exposed. -/
def run6 (t s r ph : ℕ) : Bool :=
  (List.range 6).all fun i => decide (wpos t s r ph i ∈ Corridor.exposedSet)

set_option maxRecDepth 40000 in
/-- **The finite check.** No invertible class mod 210 admits seven
consecutive exposed members: 48 classes x 35 starts x 2 parities, decided by
the kernel. -/
theorem no_run_seven :
    ∀ c < 210, Nat.gcd c 210 = 1 →
      ∀ r < 35, ∀ ph < 2, run7 (c % 35) (sOf c) r ph = false := by
  decide

/-! ## From the check to the chains -/

/-- The tooth step descends to the class: `2u' mod 35` is `sOf (q' mod 210)`. -/
theorem s_eq {q u : ℕ} (hu : 6 * u + 1 = q ∨ 6 * u = q + 1) :
    (2 * u) % 35 = sOf (q % 210) := by
  unfold sOf
  split <;> rcases hu with h | h <;> omega

/-- The `i`-th member of a literal chain based at `r`, starting on parity
`ph`: `j` steps of `q` plus a tooth step on odd parity. -/
def member (r q u ph i : ℕ) : ℕ :=
  r + ((i + ph) / 2) * q + (if (i + ph) % 2 = 1 then 2 * u else 0)

/-- **THE LITERAL CAP.** A literal chain has at most six members - for every
gear `q`, with no bound on `q`, forever. -/
theorem literal_chain_le_six {q u r ph L : ℕ}
    (hu : 6 * u + 1 = q ∨ 6 * u = q + 1) (hq : Nat.gcd q 210 = 1)
    (hph : ph < 2) (hr : 1 ≤ r)
    (hE : ∀ i < L, Corridor.Exposed (member r q u ph i)) : L ≤ 6 := by
  by_contra hL
  -- the class of q, and the check at that class
  have hc : Nat.gcd (q % 210) 210 = 1 := by
    rw [← Nat.gcd_rec 210 q, Nat.gcd_comm]; exact hq
  have hcheck := no_run_seven (q % 210) (Nat.mod_lt _ (by omega)) hc
    (r % 35) (Nat.mod_lt _ (by omega)) ph hph
  -- but all seven of the first members are exposed, so the run is there
  have hrun : run7 ((q % 210) % 35) (sOf (q % 210)) (r % 35) ph = true := by
    rw [run7, List.all_eq_true]
    intro i hi
    have hi7 := List.mem_range.mp hi
    have hEi : Corridor.Exposed (member r q u ph i) := hE i (by omega)
    have hmem := (Corridor.exposed_iff_mem
      (show 1 ≤ member r q u ph i by unfold member; omega)).mp hEi
    -- the member's residue is exactly the walk position
    have hres : wpos ((q % 210) % 35) (sOf (q % 210)) (r % 35) ph i
        = member r q u ph i % 35 := by
      rw [← s_eq hu]
      unfold wpos member
      interval_cases ph <;> interval_cases i <;> simp <;> omega
    rw [decide_eq_true_iff, hres]
    exact hmem
  rw [hcheck] at hrun
  exact Bool.false_ne_true hrun

/-! ## Sharpness -/

/-- Some start and parity gives six consecutive exposed members. -/
def hasRun6 (c : ℕ) : Bool :=
  (List.range 35).any fun r => (List.range 2).any fun ph => run6 (c % 35) (sOf c) r ph

/-- The cap is attained: exactly the six classes `37, 53, 83, 127, 157, 173`
mod 210 admit a run of six, so `6` cannot be lowered. -/
theorem cap_six_classes_sharp :
    ((Finset.range 210).filter fun c =>
      Nat.gcd c 210 = 1 ∧ hasRun6 c = true)
      = {37, 53, 83, 127, 157, 173} := by
  decide

end LiteralCap
