/-
THE MIRROR LEVER, INSTANTIATED AT A MACHINE (round 28).

Round 27 closed the lever's counting core (`Mirror.even_card_involution`,
`Mirror.window_count_even`, `Mirror.none_of_at_most_one`) and named what was
missing in the same breath: those theorems quantify over an ABSTRACT index
involution, and nothing tied that involution to a real machine.  This file ties
it, at machine 11 - the smallest machine with a complete kernel enumeration
(`Machine11Per.opSeq_shift`: 135 openings per period of 385 slots).

THE CHAIN, and only the first link is a finite computation:

  1. `opSeq_133`      opSeq 133 = 382                      (kernel, one `decide`)
  2. `opSeq_mirror`   opSeq n + opSeq (133 - n) = 385      (INDUCTION from 1 and
                      for every n <= 133                    `Mirror.mirror_exposed11`
                                                            - NOT a table)
  3. `g11_mirror`     g11 (132 - n) = g11 n                (two lines from 2)
  4. `mir2`/`L2`      the depth-2 window involution and its length, and the
                      three hypotheses of `Mirror.window_count_even` discharged
                      FROM THE MACHINE
  5. `window2_even`   every depth-2 window length except 6 occurs an EVEN
                      number of times in machine 11's period
  6. `adjacent_max_none_of_at_most_one`  the lever in the form the live route
                      quotes, bound to the (F, F) configuration at F = 7.

Step 2 is the mathematical content: the opening SET being mirror-closed
(`mirror_exposed11`, round 26) does not by itself say that the ENUMERATION
reverses.  The induction that upgrades one to the other is the composition
round 27 named and did not build; it uses only `nextOp`'s minimality, so it
transfers verbatim to any machine that has a kernel base case.

The exceptional length is 6 = g11 133 + g11 134 = 3 + 3, at the unique
self-mirror index 133 - and 6 is exactly the one depth-2 length whose count in
machine 11's period is odd (11 occurrences; every other length occurs
20, 18, 40, 26, 8, 6, 6 times).  The parity law is not vacuous here.
-/

import Mirror
import Machine11Per

namespace Machine11

/-! ## 1. The base case, in the kernel -/

/-- Machine 11's 134th opening is slot 382 - the mirror of its first (`3`).
The walk `ow` is `opSeq` with `Nat.find` removed (`Machine11Per.opSeq_eq_ow`),
so this is a single kernel evaluation. -/
theorem ow_133 : ow 133 = 379 := by decide +kernel

theorem ow_134 : ow 134 = 382 := by decide +kernel

theorem opSeq_133 : opSeq 133 = 382 := by
  rw [opSeq_eq_ow, ow_133]

/-! ## 2. The enumeration reverses -/

/-- **THE INDEX-REVERSAL LAW AT MACHINE 11.**  The mirror `k -> 385 - k` sends
the `n`-th opening to the `(133 - n)`-th.  Proved by induction, not by a table:
the step shows `385 - opSeq (132 - n)` is exposed (mirror-closure), lies above
`opSeq n`, and has nothing exposed between - which is exactly `nextOp`'s
defining property. -/
theorem opSeq_mirror : ∀ n, n ≤ 133 → opSeq n + opSeq (133 - n) = 385 := by
  intro n
  induction n with
  | zero =>
      intro _
      rw [show (133 - 0) = 133 from rfl, opSeq_133, opSeq_zero]
  | succ n ih =>
      intro hn
      have hih := ih (by omega)
      have hp0 : 1 ≤ opSeq n := opSeq_pos n
      have hA1 : 1 ≤ opSeq (132 - n) := opSeq_pos _
      have hAle : opSeq (132 - n) ≤ 382 := by
        have h := opSeq_le_add (132 - n) (133 - (132 - n))
        rw [show (132 - n) + (133 - (132 - n)) = 133 by omega, opSeq_133] at h
        exact h
      have hB : opSeq (133 - n) = nextOp (opSeq (132 - n)) := by
        rw [show 133 - n = (132 - n) + 1 by omega]
        exact opSeq_succ _
      have hAB : opSeq (132 - n) < opSeq (133 - n) := by
        rw [hB]; exact nextOp_gt _
      have hEA : Exposed11 (385 - opSeq (132 - n)) :=
        (Mirror.mirror_exposed11 hA1 (by omega)).mpr (opSeq_exposed _)
      have hlt : opSeq n < 385 - opSeq (132 - n) := by omega
      have hle : opSeq (n + 1) ≤ 385 - opSeq (132 - n) := by
        rw [opSeq_succ]
        exact Nat.find_min' (exists_exposed_above (opSeq n)) ⟨hlt, hEA⟩
      have hge : 385 - opSeq (132 - n) ≤ opSeq (n + 1) := by
        by_contra hc
        push Not at hc
        have hm1 : opSeq n < opSeq (n + 1) := opSeq_lt_succ n
        have hEm : Exposed11 (opSeq (n + 1)) := opSeq_exposed _
        have hp : 1 ≤ opSeq (n + 1) := opSeq_pos _
        have hEmm : Exposed11 (385 - opSeq (n + 1)) :=
          (Mirror.mirror_exposed11 hp (by omega)).mpr hEm
        refine nextOp_min (k := opSeq (132 - n)) (m := 385 - opSeq (n + 1))
          (by omega) ?_ hEmm
        rw [← hB]; omega
      rw [show 133 - (n + 1) = 132 - n by omega]
      omega

/-- **The gap word reverses.**  `g11` read backwards from index 132 is `g11`
read forwards - the word-level form of the index-reversal law. -/
theorem g11_mirror {n : ℕ} (hn : n ≤ 132) : g11 (132 - n) = g11 n := by
  have h1 := opSeq_mirror n (by omega)
  have h2 := opSeq_mirror (n + 1) (by omega)
  rw [show 133 - (n + 1) = 132 - n by omega] at h2
  have h3 : g11 (132 - n) = opSeq (133 - n) - opSeq (132 - n) := by
    show opSeq ((132 - n) + 1) - opSeq (132 - n) = _
    rw [show (132 - n) + 1 = 133 - n by omega]
  have h4 : g11 n = opSeq (n + 1) - opSeq n := rfl
  omega

/-! ## 3. The depth-2 window family and its involution -/

/-- The depth-2 window length at index `t`: two consecutive gaps. -/
def L2 (t : ℕ) : ℕ := g11 t + g11 (t + 1)

/-- The mirror on depth-2 window INDICES.  The mirror sends the window
`[o t, o (t+2)]` to `[385 - o (t+2), 385 - o t]`, which is the window at index
`131 - t` - taken mod 135 so that it is an involution of the whole period. -/
def mir2 (t : ℕ) : ℕ := (266 - t) % 135

theorem mir2_lt (t : ℕ) : mir2 t < 135 := Nat.mod_lt _ (by norm_num)

theorem mir2_invol : ∀ t, t < 135 → mir2 (mir2 t) = t := by decide

theorem mir2_fix : ∀ t, t < 135 → mir2 t = t → t = 133 := by decide

/-- `g11 133 = g11 134 = 3` - the two letters straddling the period seam, which
the reversal law of section 2 does not reach. -/
theorem g11_seam : g11 133 = 3 ∧ g11 134 = 3 := by
  constructor
  · rw [g11_eq_ow, ow_134, ow_133]
  · rw [g11_eq_ow, ow_135, ow_134]

/-- **THE LENGTH FUNCTION IS MIRROR-INVARIANT** - the hypothesis of
`Mirror.window_count_even` that has to come from the machine.  Interior indices
use the reversal law twice; the three indices whose window crosses the seam are
closed by `g11_seam` and the period law. -/
theorem L2_mirror : ∀ t, t < 135 → L2 (mir2 t) = L2 t := by
  intro t ht
  rcases Nat.lt_or_ge t 132 with hlow | hhigh
  · have hm : mir2 t = 131 - t := by
      show (266 - t) % 135 = 131 - t
      omega
    have e1 : g11 (131 - t) = g11 (t + 1) := by
      have := g11_mirror (n := t + 1) (by omega)
      rw [show 132 - (t + 1) = 131 - t by omega] at this
      exact this
    have e2 : g11 (131 - t + 1) = g11 t := by
      rw [show 131 - t + 1 = 132 - t by omega]
      exact g11_mirror (n := t) (by omega)
    show L2 (mir2 t) = L2 t
    rw [hm]
    show g11 (131 - t) + g11 (131 - t + 1) = g11 t + g11 (t + 1)
    rw [e1, e2]
    omega
  · have h132 : g11 132 = g11 0 := g11_mirror (n := 0) (by omega)
    have h135 : g11 135 = g11 0 := by
      have := g11_shift 0
      simpa using this
    have hs := g11_seam
    interval_cases t
    · show L2 (mir2 132) = L2 132
      show g11 (mir2 132) + g11 (mir2 132 + 1) = g11 132 + g11 (132 + 1)
      norm_num [mir2]
      omega
    · rfl
    · show L2 (mir2 134) = L2 134
      show g11 (mir2 134) + g11 (mir2 134 + 1) = g11 134 + g11 (134 + 1)
      norm_num [mir2]
      omega

/-! ## 4. The lever, loaded -/

/-- The self-mirror depth-2 window of machine 11 sits at index 133 and has
length `6`. -/
theorem L2_133 : L2 133 = 6 := by
  have hs := g11_seam
  have e : L2 133 = g11 133 + g11 134 := by norm_num [L2]
  omega

/-- **THE MIRROR PARITY LAW AT MACHINE 11**: every depth-2 window length other
than the self-mirror window's own length `6` occurs an EVEN number of times in
the period.  All three hypotheses of `Mirror.window_count_even` are discharged
from the machine - `mir2_invol` and `mir2_fix` by arithmetic, `L2_mirror` by
the index-reversal law of section 2. -/
theorem window2_even {g : ℕ} (hg : g ≠ 6) :
    (((Finset.range 135).filter (fun t => L2 t = g)).card) % 2 = 0 :=
  Mirror.window_count_even mir2 L2 (fun t _ => mir2_lt t) mir2_invol L2_mirror
    (fun t ht hfix => by rw [mir2_fix t ht hfix, L2_133]; exact fun h => hg h.symm)

/-- **THE LEVER IN THE FORM THE LIVE ROUTE QUOTES, AT A MACHINE.**  An adjacent
equal pair `(F, F)` at `F = 7 = F(11)` has depth-2 length `14`, which the
self-mirror window does not carry - so its count is even, and a counting
argument that caps it at ONE thereby proves it is ZERO.  That is a strictly
cheaper obligation than proving the cap is zero, and it is now a theorem about
a real machine rather than about an abstract involution. -/
theorem adjacent_max_none_of_at_most_one
    (hone : (((Finset.range 135).filter (fun t => L2 t = 2 * 7)).card) ≤ 1) :
    (((Finset.range 135).filter (fun t => L2 t = 2 * 7)).card) = 0 :=
  Mirror.none_of_at_most_one (F := 7) (t0 := 133) mir2 L2
    (fun t _ => mir2_lt t) mir2_invol L2_mirror (by norm_num)
    (by decide) mir2_fix (by rw [L2_133]; norm_num) hone

/-- The cross-check, by a completely different route: machine 11's kernel
spectrum ladder already gives `F_2(11) <= 11 < 14`, so the count of `(7,7)`
pairs is zero outright.  The two routes agree, which is what makes the parity
statement above a lever rather than a coincidence. -/
theorem adjacent_max_none :
    (((Finset.range 135).filter (fun t => L2 t = 2 * 7)).card) = 0 := by
  rw [Finset.card_eq_zero, Finset.filter_eq_empty_iff]
  intro t _
  have h := spectrum_two t
  have hw : Spectrum.windowSum g11 t 2 = g11 t + g11 (t + 1) := by
    simp [Spectrum.windowSum, Finset.sum_range_succ]
  show ¬ (g11 t + g11 (t + 1) = 2 * 7)
  omega

end Machine11
