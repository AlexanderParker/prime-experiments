/-
Tier A of the flank-sum certificate: the corridor law for a chain of
openings of any length.

Round 9 proved the three-point case (`Corridor.allowed3`, and
`no_11_11_chain`: two maximal gaps are never adjacent at machine 13). The
flank-sum bound of constructor sec 24.3 needs the same statement with a WORD
in between - an `(l+2)`-point correlation where `allowed3` was a 3-point one.
This file proves it for an arbitrary list of gaps.

`carrier steps` is the set of residues mod 35 at which a chain of openings
with consecutive gaps `steps` could sit: every partial sum lands in the (5,7)
exposed set. `mem_carrier_of_chain` says a real chain pins its base residue
into the carrier, so an EMPTY carrier forbids the configuration outright, at
every machine refining gears `{5,7}`, with no period scan
(`no_chain_of_carrier_empty`). This is the piece that scales: unlike the
period scans of `Machine13`/`Machine17`, its cost does not grow with the
machine.

Specialised to `steps = F :: w ++ [F]` this is exactly constructor 24.3's
question - can a compatible word occurrence carry a MAXIMAL gap on both
flanks? - and the answer at four of the measured steps is no, by corridor
arithmetic alone (`no_maximal_flanks_*`). The honest counterexample is
recorded too: at `19 -> 23` the carrier is nonempty, so tier A does not close
that step and the deeper tiers are genuinely needed.

Verified against research/flank_tierA_fix.py before formalising: the carrier
computed here agrees with the tool's joint-flank counts at m = 35 on every
case checked, including the nonzero ones (`[25,8,25]` has 4 residues,
`[25,15,25]` has 6 - the tool's `both4` and `both6`).
-/

import Corridor

namespace TierA

/-! ## Chains of openings with prescribed gaps -/

/-- The positions of a chain with consecutive gaps `steps`, as offsets from
its base: the partial sums, starting at `0`. -/
def offsets : List ℕ → List ℕ
  | [] => [0]
  | s :: rest => 0 :: (offsets rest).map (· + s)

/-- The residues mod 35 that can carry such a chain: every position lands in
the (5,7) exposed set. -/
def carrier (steps : List ℕ) : Finset ℕ :=
  (Finset.range 35).filter fun r =>
    ((offsets steps).all fun t => decide ((r + t) % 35 ∈ Corridor.exposedSet)) = true

/-- **A real chain pins its residue.** If every position of the chain is an
opening, the base residue lies in the carrier. -/
theorem mem_carrier_of_chain {x : ℕ} {steps : List ℕ} (hx : 1 ≤ x)
    (h : ∀ t ∈ offsets steps, Corridor.Exposed (x + t)) :
    x % 35 ∈ carrier steps := by
  rw [carrier, Finset.mem_filter]
  refine ⟨Finset.mem_range.mpr (Nat.mod_lt _ (by omega)), ?_⟩
  rw [List.all_eq_true]
  intro t ht
  have hm := (Corridor.exposed_iff_mem (show 1 ≤ x + t by omega)).mp (h t ht)
  have e : (x % 35 + t) % 35 = (x + t) % 35 := by omega
  simp only [decide_eq_true_eq, e]
  exact hm

/-- **Tier A.** An empty carrier forbids the configuration everywhere: no
base, no machine, no period scan. -/
theorem no_chain_of_carrier_empty {x : ℕ} {steps : List ℕ}
    (hc : carrier steps = ∅) (hx : 1 ≤ x)
    (h : ∀ t ∈ offsets steps, Corridor.Exposed (x + t)) : False := by
  have := mem_carrier_of_chain hx h
  rw [hc] at this
  exact Finset.notMem_empty _ this

/-! ## The flanked-word form (constructor 24.3)

A compatible word `w` with a gap `F` on each side is the chain
`F :: w ++ [F]`. Asking whether a MAXIMAL gap can flank the occurrence on
both sides is asking whether that chain's carrier is empty.
-/

/-- The chain of a word `w` carrying flanks of size `F` on both sides. -/
def flanked (F : ℕ) (w : List ℕ) : List ℕ := F :: (w ++ [F])

/-- If the flanked carrier is empty, no occurrence of `w` can have maximal
gaps on both flanks - the tier-A half of the flank-sum bound. -/
theorem no_maximal_flanks {x F : ℕ} {w : List ℕ}
    (hc : carrier (flanked F w) = ∅) (hx : 1 ≤ x)
    (h : ∀ t ∈ offsets (flanked F w), Corridor.Exposed (x + t)) : False :=
  no_chain_of_carrier_empty hc hx h

/-! ## Instances

The measured steps of constructor 24.2, binding word and `F` per step. Four
close at tier A; `19 -> 23` does not, and that is recorded rather than hidden.
-/

/-- `11 -> 13`, word `(4)`, `F = 7`. -/
theorem flanks_11_13 : carrier (flanked 7 [4]) = ∅ := by decide

/-- `13 -> 17`, word `(6)`, `F = 11`. -/
theorem flanks_13_17 : carrier (flanked 11 [6]) = ∅ := by decide

/-- `17 -> 19`, word `(13)`, `F = 18`. Each flank alone is feasible mod 35
(the tool's `L1 R1`); BOTH together are not. This is precisely the content
of 24.3 - the two flanks cannot both be near-maximal. -/
theorem flanks_17_19 : carrier (flanked 18 [13]) = ∅ := by decide

/-- `23 -> 29`, word `(19)`, `F = 34`. -/
theorem flanks_23_29 : carrier (flanked 34 [19]) = ∅ := by decide

/-- `29 -> 31`, word `(10)`, `F = 43`. -/
theorem flanks_29_31 : carrier (flanked 43 [10]) = ∅ := by decide

/-- **The honest exception.** At `19 -> 23` with word `(8)` and `F = 25` the
carrier is NOT empty - four residues survive - so tier A does not close this
step and the mod-385 / direct tiers are genuinely needed. Recorded so that
nobody assumes tier A suffices in general. -/
theorem flanks_19_23_nonempty : carrier (flanked 25 [8]) = {0, 5, 7, 12} := by decide

/-- The round-9 result as the `l = 0` case: two maximal gaps are never
adjacent at machine 13. -/
theorem no_adjacent_maximal_13 : carrier [11, 11] = ∅ := by decide

/-! ## Adjacent padded links (lateral's corridor law)

A padded link's interior gap is a multiple of `q'`, so two ADJACENT padded
links of sizes `a q'`, `b q'` put three consecutive openings at
`r`, `r + a q'`, `r + (a+b) q'` - a three-point chain whose carrier is
computed by the machinery above. Lateral's result: at `q' = 41` the equal
shape `(1,1)` has NO carrier at all, so two adjacent equal padded links are
impossible there by the (5,7) corridor alone - no spectrum input, hence
independent of the fact that machine-37 `F_j` values are only prefix lower
bounds.

The general law: feasibility depends only on `q' mod 35`, and exactly 12 of
the 24 invertible classes forbid the equal shape. There is also a perfect
dichotomy - the equal shape is impossible exactly when both unequal shapes
`(1,2)` and `(2,1)` are possible.

Verified against lateral.md before formalising: the forbidden class list, the
12/24 split, the dichotomy, and the "exactly 2 phases each" count.
-/

/-- Two adjacent padded links, of sizes `a` and `b` in units of the gear. -/
theorem no_adjacent_equal_padded {x q : ℕ} (hc : carrier [q, q] = ∅) (hx : 1 ≤ x)
    (h : ∀ t ∈ offsets [q, q], Corridor.Exposed (x + t)) : False :=
  no_chain_of_carrier_empty hc hx h

/-- **The 37 -> 41 case.** Two adjacent equal padded links are impossible at
`q' = 41`, by corridor arithmetic alone. -/
theorem no_adjacent_padded_41 : carrier [41, 41] = ∅ := by decide

/-- **The general law.** Exactly these 12 of the 24 invertible classes mod 35
forbid two adjacent equal padded links. A 50/50 property of `q' mod 35` - not
a trend in scale. -/
theorem equal_padding_forbidden_classes :
    ((Finset.range 35).filter fun g => Nat.gcd g 35 = 1 ∧ carrier [g, g] = ∅)
      = {1, 4, 6, 9, 11, 16, 19, 24, 26, 29, 31, 34} := by decide

/-- Twelve of twenty-four. -/
theorem equal_padding_forbidden_card :
    ((Finset.range 35).filter fun g => Nat.gcd g 35 = 1 ∧ carrier [g, g] = ∅).card = 12 := by
  rw [equal_padding_forbidden_classes]; decide

/-- **The dichotomy.** The equal shape `(1,1)` is impossible exactly when both
unequal shapes `(1,2)` and `(2,1)` are possible. Padding structure switches on
and off with the residue of `q'` - which is why a smooth `supply^2/gaps` model
cannot predict it. -/
theorem padding_shape_dichotomy : ∀ g < 35, Nat.gcd g 35 = 1 →
    (carrier [g, g] = ∅ ↔
      carrier [g, (2*g) % 35] ≠ ∅ ∧ carrier [(2*g) % 35, g] ≠ ∅) := by decide

/-! ## Padding is count-capped

A padded link's interior gap is `0 mod q'`, hence at least `q'`, while the
tolerance budget grants only `(5/6) q'` beyond `F`. So the number of padded
links in a run is bounded - the arithmetic core, with denominators cleared.
-/

/-- If each of `p` padded links consumes at least `q` of a span `S`, and the
span is within the budget `F + (5/6) q`, then `6 p q <= 6 F + 5 q`. In
particular `p` is bounded by roughly `F/q + 5/6`. -/
theorem padding_count_le {p q F S : ℕ} (hspan : p * q ≤ S)
    (hbud : 6 * S ≤ 6 * F + 5 * q) : 6 * (p * q) ≤ 6 * F + 5 * q :=
  le_trans (Nat.mul_le_mul_left 6 hspan) hbud

/-- One padded link already costs more than the budget allows twice over:
with `F < q` (the onset condition of the mechanic's census) at most one
padded link fits. -/
theorem padding_at_most_one {p q F S : ℕ} (_hq : 0 < q) (hF : F < q)
    (hspan : p * q ≤ S) (hbud : 6 * S ≤ 6 * F + 5 * q) : p ≤ 1 := by
  have h := padding_count_le hspan hbud
  by_contra hp
  have h2 : 2 * q ≤ p * q := Nat.mul_le_mul_right q (by omega)
  omega

end TierA
