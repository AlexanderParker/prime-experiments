/-
THE SURVIVOR GENERATOR, FIRST KERNEL STATEMENT: machine 13's F and F_2 read
off MACHINE 11's gap word alone (round 25).

Constructor's R46 Kleene generator and its round-24 survivor extension
(`docs/novel/kleene-generator.md`, `docs/novel/survivor-generator.md`) say
that the low spectrum of machine `M + q'` is a max-plus language over machine
`M`: `F(M+q') = L (x) K* (x) R` and `F_2(M+q') = L (x) K* (x) SIGMA (x) K*
(x) R`, the extra letter SIGMA being the step that skips THROUGH the unique
surviving opening.  Nothing in this project had ever been checked by a kernel.

At `M = 11`, `q' = 13` the whole algebra is small enough to write out:

  * machine 11 has 135 openings in a period of 385 slots, and `gw11` below is
    its CYCLIC gap word (sum 385, max 7 = F(11));
  * gear 13 kills the slot residues 2 and 11 (`6 * 2 = 12 = 13 - 1`,
    `6 * 11 = 66 = 5 * 13 + 1`), so relative to a base opening whose slot is
    `c` mod 13, the opening at offset `d` is killed iff `(c + d) % 13` is 2
    or 11.  The phase `c` is free: 385 and 13 are coprime, so as machine 11's
    period repeats 13 times inside machine 13's, every base residue occurs;
  * `walk` is the generator: from a base opening it advances through machine
    11's word, passing over killed openings, and stops at the survivor that
    is the `(ns+1)`-st.  `ns = 0` is `K*` alone - a gap of machine 13.
    `ns = 1` inserts exactly one SIGMA - a two-gap window of machine 13.

`gen 0 = 11` and `gen 1 = 16` are then decided by the kernel, and they are
exactly `F(13)` and `F_2(13)`: the same integers `Machine13Q.spectrum_ladder`
proves from machine 13's own 5005-slot period, here obtained from a
135-letter word over a 385-slot period with no mention of machine 13's period
at all.

WHAT IS AND IS NOT CLAIMED.  What is kernel-checked is the GENERATOR'S VALUE:
`gen 0 = 11`, `gen 1 = 16`, over every base, every phase, and every window of
span at most 30 (`no_truncation` shows the fuel never binds inside that cap -
thirteen consecutive machine-11 gaps already span 33).  What was NOT
formalised in round 25 is the SOUNDNESS BRIDGE - that every machine-13 gap
really is one of the windows this search enumerates.

ROUND 26: THE BRIDGE IS BUILT, in `proofs/Gen11Sound.lean`.  `gw11` is
certified as machine 11's own gap word (`Gen11.gAt_succ`), the periodicity
glue `opSeq11 (n + 135) = opSeq11 n + 385` is `Machine11Per.opSeq_shift`, and
`Gen11.walk_sound` shows the walk simulates the machine, giving
`Gen11.generator_sound : F_1..F_4(13) <= 11, 16, 23, 26` with machine 13's own
period nowhere in the derivation (gated by `proofs/DepAudit.lean`).  So this
file states that the generator COMPUTES the right integers, and `Gen11Sound`
proves that it MUST.
-/

import Machine13Q

namespace Gen11

set_option maxRecDepth 20000

/-- **Machine 11's cyclic gap word**: the 135 gaps between consecutive
openings of gears `{5, 7, 11}` over one period of 385 slots, the last gap
wrapping round.  Computed and gate-checked by `research/gen11.py`. -/
def gw11 : List ℕ :=
  [3, 2, 2, 3, 2, 5, 1, 5, 2, 3, 2, 2, 1, 4, 1, 2, 5, 2, 5, 6,
   2, 3, 2, 2, 3, 2, 1, 4, 3, 2, 5, 1, 5, 2, 3, 2, 2, 1, 2, 2,
   3, 5, 2, 5, 6, 5, 2, 2, 1, 2, 2, 1, 4, 3, 7, 1, 7, 3, 2, 2,
   1, 2, 2, 3, 2, 5, 5, 1, 5, 5, 2, 3, 2, 2, 1, 2, 2, 3, 7, 1,
   7, 3, 4, 1, 2, 2, 1, 2, 2, 5, 6, 5, 2, 5, 3, 2, 2, 1, 2, 2,
   3, 2, 5, 1, 5, 2, 3, 4, 1, 2, 3, 2, 2, 3, 2, 6, 5, 2, 5, 2,
   1, 4, 1, 2, 2, 3, 2, 5, 1, 5, 2, 3, 2, 2, 3]

/-- The word is cyclic: index mod 135. -/
def gAt (i : ℕ) : ℕ := gw11.getD (i % 135) 0

/-- Span of `k` consecutive machine-11 gaps from position `i`. -/
def off (i : ℕ) : ℕ → ℕ
  | 0 => 0
  | k + 1 => off i k + gAt (i + k)

/-- Gear 13's teeth on slot residues: `{2, 11}`. -/
def kil13 (r : ℕ) : Bool := (r % 13 == 2) || (r % 13 == 11)

/-- **The generator's walk.**  From the base opening at index `i` and slot
residue `c` mod 13, advance through machine 11's word; killed openings are
passed over (that is `K`), surviving ones are counted.  Stop at the
`(ns+1)`-st survivor and return its offset - the merged span.  Return the
SENTINEL 999 if the span cap 30 is exceeded or the fuel runs out.

(Round 26: the bail value was `0` when this file was written, which is a
sound value for a MAXIMUM but destroys soundness in the other direction - a
walk that gives up would silently lower nothing and be indistinguishable
from a short gap.  With a sentinel above every attainable span, `gen ns` is
small ONLY IF no walk ever bailed, which is exactly the hypothesis
`Gen11Sound` needs.  The computed values are unchanged: 11, 16, 23, 26.) -/
def walk (i c ns : ℕ) : ℕ → ℕ → ℕ → ℕ → ℕ
  | 0, _, _, _ => 999
  | fuel + 1, k, d, surv =>
      let d' := d + gAt (i + k)
      if 30 < d' then 999
      else if kil13 (c + d') then walk i c ns fuel (k + 1) d' surv
      else if surv == ns then d'
      else walk i c ns fuel (k + 1) d' (surv + 1)

/-- **The generator's value**: the largest merged span over every base
opening, every free phase `c` at which the base itself survives, and every
window with exactly `ns` surviving interior openings.  `ns = 0` is R46's
`L (x) K* (x) R`; `ns = 1` inserts Constructor's SIGMA letter once. -/
def gen (ns : ℕ) : ℕ :=
  ((List.range 135).map fun i =>
    ((List.range 13).map fun c =>
      if kil13 c then 0 else walk i c ns 13 0 0 0).foldl max 0).foldl max 0

/-! ## The kernel checks -/

/-- The word really is one period: 135 gaps summing to 385. -/
theorem gw11_len : gw11.length = 135 := by decide +kernel

theorem gw11_sum : gw11.sum = 385 := by decide +kernel

/-- `F(11) = 7` re-read off the word (the corpus value, and
`Machine11.spectrum_ladder`'s first entry). -/
theorem gw11_max : gw11.all (fun g => Nat.ble g 7) = true := by decide +kernel

/-- **The fuel never binds inside the span cap**: thirteen consecutive
machine-11 gaps already span more than 30, so a walk of fuel 13 exits by the
cap, never by exhaustion, and `gen` is a maximum over ALL windows of span at
most 30 - not merely over short ones. -/
theorem no_truncation : ∀ i < 135, 30 < off i 13 := by decide +kernel

/-- **THE PLAIN GENERATOR**: `L (x) K* (x) R = 11 = F(13)`, computed from
machine 11's 135-letter word with no reference to machine 13's period. -/
theorem gen_zero : gen 0 = 11 := by decide +kernel

/-- **THE SURVIVOR GENERATOR**: one SIGMA letter gives
`L (x) K* (x) SIGMA (x) K* (x) R = 16 = F_2(13)` - Constructor's survivor
identity at its first step, in the kernel. -/
theorem gen_one : gen 1 = 16 := by decide +kernel

/-- Two SIGMA letters: `F_3(13) = 23`. -/
theorem gen_two : gen 2 = 23 := by decide +kernel

/-- Three: `F_4(13) = 26`.  The generator reproduces machine 13's whole low
spectrum ladder `11, 16, 23, 26` from machine 11's word. -/
theorem gen_three : gen 3 = 26 := by decide +kernel

/-- **The identity at 11 -> 13, both sides kernel-checked.**  The left-hand
sides are the generator over machine 11's word (this file); the right-hand
sides are machine 13's own spectrum, proved in `Machine13Q` from its 5005-slot
period.  The two routes share only the definition of a machine.

(What is asserted is the AGREEMENT of the two computations.  That the
generator must agree - the soundness bridge - is not formalised here; see the
file header.) -/
theorem generator_matches_machine13 :
    gen 0 = 11 ∧ gen 1 = 16 ∧
      Spectrum.SpectrumBound Machine13.g13 1 11 ∧
      Spectrum.SpectrumBound Machine13.g13 2 16 :=
  ⟨gen_zero, gen_one, (Machine13.spectrum_ladder).1, (Machine13.spectrum_ladder).2.1⟩

end Gen11
